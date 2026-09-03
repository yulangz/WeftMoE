#!/usr/bin/env python3
"""Compare cached-state MegaMOE Backward artifacts with PyTorch autograd.

The oracle consumes the exact canonical X and BF16 A/B values retained by the
tested Forward invocation. It independently rebuilds only the deterministic
weights and reverse routing, then lets PyTorch autograd evaluate the standard
SwiGLU/linear graph.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F


MASK32 = (1 << 32) - 1


def mix32_scalar(value):
    value &= MASK32
    value ^= value >> 16
    value = (value * 0x7FEB352D) & MASK32
    value ^= value >> 15
    value = (value * 0x846CA68B) & MASK32
    value ^= value >> 16
    return value


def mix32_tensor(value):
    value = torch.bitwise_and(value, MASK32)
    value = torch.bitwise_xor(value, value >> 16)
    value = torch.bitwise_and(value * 0x7FEB352D, MASK32)
    value = torch.bitwise_xor(value, value >> 15)
    value = torch.bitwise_and(value * 0x846CA68B, MASK32)
    return torch.bitwise_xor(value, value >> 16)


def deterministic_weight(expert, kind, rows, columns, device):
    output = torch.empty((rows, columns), dtype=torch.bfloat16, device=device)
    column = torch.arange(columns, dtype=torch.int64, device=device)[None, :]
    base = mix32_scalar(expert ^ 0x9E3779B9)
    base ^= mix32_scalar(kind ^ 0x85EBCA6B)
    column_key = mix32_tensor(torch.bitwise_xor(column, 0x27D4EB2F))
    chunk_rows = 64
    for begin in range(0, rows, chunk_rows):
        end = min(rows, begin + chunk_rows)
        row = torch.arange(
            begin, end, dtype=torch.int64, device=device
        )[:, None]
        key = torch.bitwise_xor(
            base,
            torch.bitwise_xor(
                mix32_tensor(torch.bitwise_xor(row, 0xC2B2AE35)),
                column_key,
            ),
        )
        sign = (torch.bitwise_and(mix32_tensor(key), 1) * 2 - 1).float()
        output[begin:end] = (sign / 256.0).to(torch.bfloat16)
    return output


def read_numpy(path, dtype, count=None):
    result = np.fromfile(path, dtype=dtype, count=-1 if count is None else count)
    if count is not None and result.size != count:
        raise RuntimeError(f"short artifact {path}: {result.size} != {count}")
    return result


def bf16_from_raw(values, device):
    copied = np.array(values, dtype=np.uint16, copy=True)
    return torch.from_numpy(copied).view(torch.bfloat16).to(device)


class ErrorSummary:
    def __init__(self):
        self.count = 0
        self.squared_error = 0.0
        self.squared_reference = 0.0
        self.max_abs = 0.0
        self.exact = True

    def add(self, reference, observed):
        reference = reference.detach()
        observed = observed.detach()
        self.exact = self.exact and torch.equal(reference, observed)
        difference = reference.float() - observed.float()
        self.count += difference.numel()
        self.squared_error += difference.square().sum().item()
        self.squared_reference += reference.float().square().sum().item()
        self.max_abs = max(self.max_abs, difference.abs().max().item())

    def reduce(self, device):
        sums = torch.tensor(
            [self.count, self.squared_error, self.squared_reference],
            dtype=torch.float64,
            device=device,
        )
        maximum = torch.tensor(self.max_abs, dtype=torch.float64, device=device)
        exact = torch.tensor(int(self.exact), dtype=torch.int32, device=device)
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        dist.all_reduce(exact, op=dist.ReduceOp.MIN)
        count, squared_error, squared_reference = sums.tolist()
        return {
            "all_ranks_byte_identical": bool(exact.item()),
            "max_abs_error": maximum.item(),
            "rmse": (squared_error / max(count, 1.0)) ** 0.5,
            "relative_l2": (
                (squared_error / squared_reference) ** 0.5
                if squared_reference != 0.0
                else 0.0
            ),
            "elements": int(count),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--tokens", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--intermediate", type=int, default=3072)
    parser.add_argument("--experts", type=int)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.use_deterministic_algorithms(True)
    experts = args.experts or max(16, world * 2)
    if experts % world:
        raise ValueError("experts must be divisible by world")
    local_experts = experts // world
    top_k = 1
    root = Path(args.artifact_dir)

    all_routes = []
    all_gates = []
    all_dy = []
    for source in range(world):
        prefix = root / f"rank-{source}-"
        all_routes.append(
            read_numpy(prefix.with_name(prefix.name + "routes.i32"), np.int32)
        )
        all_gates.append(
            read_numpy(prefix.with_name(prefix.name + "gates.f32"), np.float32)
        )
        all_dy.append(
            bf16_from_raw(
                read_numpy(
                    prefix.with_name(prefix.name + "source-dy.bf16"),
                    np.uint16,
                    args.tokens * args.hidden,
                ),
                device,
            ).reshape(args.tokens, args.hidden)
        )

    owned_experts = list(
        range(rank * local_experts, (rank + 1) * local_experts)
    )
    row_records = []
    expert_counts = []
    for expert in owned_experts:
        before = len(row_records)
        for source in range(world):
            for route, selected in enumerate(all_routes[source]):
                if int(selected) == expert:
                    row_records.append((source, route // top_k, route))
        expert_counts.append(len(row_records) - before)
    incoming_rows = len(row_records)
    prefix = root / f"rank-{rank}-"
    canonical_x = bf16_from_raw(
        read_numpy(
            prefix.with_name(prefix.name + "canonical-x.bf16"),
            np.uint16,
            incoming_rows * args.hidden,
        ),
        device,
    ).reshape(incoming_rows, args.hidden)
    cached_ab = bf16_from_raw(
        read_numpy(
            prefix.with_name(prefix.name + "canonical-ab.bf16"),
            np.uint16,
            2 * incoming_rows * args.intermediate,
        ),
        device,
    ).reshape(2, incoming_rows, args.intermediate)
    canonical_dr = torch.stack(
        [all_dy[source][token] for source, token, _route in row_records]
    )
    canonical_gate = torch.tensor(
        [all_gates[source][route] for source, _token, route in row_records],
        dtype=torch.float32,
        device=device,
    )

    observed_dw1 = np.memmap(
        prefix.with_name(prefix.name + "dw1.bf16"),
        dtype=np.uint16,
        mode="r",
        shape=(local_experts, 2 * args.intermediate, args.hidden),
    )
    observed_dw2 = np.memmap(
        prefix.with_name(prefix.name + "dw2.bf16"),
        dtype=np.uint16,
        mode="r",
        shape=(local_experts, args.hidden, args.intermediate),
    )
    dx_rows = torch.empty(
        (incoming_rows, args.hidden), dtype=torch.bfloat16, device=device
    )
    dgate_rows = torch.empty(incoming_rows, dtype=torch.float32, device=device)
    summaries = {name: ErrorSummary() for name in ("dx", "dgate", "dw1", "dw2")}

    row_begin = 0
    for local_expert, rows in enumerate(expert_counts):
        row_end = row_begin + rows
        observed_w1 = bf16_from_raw(observed_dw1[local_expert], device)
        observed_w2 = bf16_from_raw(observed_dw2[local_expert], device)
        if rows == 0:
            summaries["dw1"].add(torch.zeros_like(observed_w1), observed_w1)
            summaries["dw2"].add(torch.zeros_like(observed_w2), observed_w2)
            row_begin = row_end
            continue

        # The fixture storage initializes experts contiguously per rank. This
        # is the exact weight identity consumed by the tested Forward kernel.
        stored_expert = rank * local_experts + local_expert
        w2 = deterministic_weight(
            stored_expert, 2, args.hidden, args.intermediate, device
        ).requires_grad_(True)
        a = cached_ab[0, row_begin:row_end].detach().requires_grad_(True)
        b = cached_ab[1, row_begin:row_end].detach().requires_grad_(True)
        gate = canonical_gate[row_begin:row_end].detach().requires_grad_(True)
        u = (gate[:, None] * (F.silu(a.float()) * b.float())).to(torch.bfloat16)
        r = u @ w2.T
        torch.autograd.backward(r, canonical_dr[row_begin:row_end])
        dab = torch.cat((a.grad, b.grad), dim=1)
        dgate_rows[row_begin:row_end] = gate.grad
        summaries["dw2"].add(w2.grad, observed_w2)
        del r, w2, observed_w2

        w1 = deterministic_weight(
            stored_expert, 1, 2 * args.intermediate, args.hidden, device
        ).requires_grad_(True)
        x = canonical_x[row_begin:row_end].detach().requires_grad_(True)
        projection = x @ w1.T
        torch.autograd.backward(projection, dab)
        dx_rows[row_begin:row_end] = x.grad
        summaries["dw1"].add(w1.grad, observed_w1)
        del projection, w1, observed_w1, dab, a, b, gate, u, x
        row_begin = row_end

    global_dx = torch.zeros(
        (world, args.tokens, args.hidden), dtype=torch.float32, device=device
    )
    global_dgate = torch.zeros(
        (world, args.tokens), dtype=torch.float32, device=device
    )
    for row, (source, token, _route) in enumerate(row_records):
        global_dx[source, token] += dx_rows[row].float()
        global_dgate[source, token] = dgate_rows[row]
    dist.all_reduce(global_dx, op=dist.ReduceOp.SUM)
    dist.all_reduce(global_dgate, op=dist.ReduceOp.SUM)
    reference_dx = global_dx[rank].to(torch.bfloat16)
    reference_dgate = global_dgate[rank]
    observed_dx = bf16_from_raw(
        read_numpy(
            prefix.with_name(prefix.name + "dx.bf16"),
            np.uint16,
            args.tokens * args.hidden,
        ),
        device,
    ).reshape(args.tokens, args.hidden)
    observed_dgate = torch.from_numpy(
        np.array(
            read_numpy(
                prefix.with_name(prefix.name + "dgate.f32"),
                np.float32,
                args.tokens,
            ),
            copy=True,
        )
    ).to(device)
    summaries["dx"].add(reference_dx, observed_dx)
    summaries["dgate"].add(reference_dgate, observed_dgate)
    result = {name: summary.reduce(device) for name, summary in summaries.items()}
    if rank == 0:
        print(
            "BACKWARD_REFERENCE_JSON "
            + json.dumps(
                {
                    "schema": 1,
                    "implementation": "PyTorch autograd from exact cached BF16 X/A/B",
                    "world": world,
                    "shape": {
                        "tokens_per_rank": args.tokens,
                        "hidden": args.hidden,
                        "intermediate": args.intermediate,
                        "experts": experts,
                        "top_k": top_k,
                    },
                    "accuracy": result,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
