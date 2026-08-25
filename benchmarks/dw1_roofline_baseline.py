#!/usr/bin/env python3
"""Isolated dW1 compute roofline for the frozen Phase-6 main shape.

The DeepGEMM comparison deliberately measures only grouped GEMM compute.  Its
NT interface consumes X in a pre-transposed [expert, hidden, canonical_rows]
layout, whereas MegaMOE production consumes canonical row-major X directly.
Consequently this is a mature-kernel compute roofline, not a drop-in hot-path
replacement and its transpose/setup cost is excluded explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from typing import Callable

import torch


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def measure_ms(operation: Callable[[], None], warmup: int,
               measured: int) -> list[float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(measured):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--experts", type=int, default=48)
    parser.add_argument("--canonical-rows", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--intermediate", type=int, default=3072)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--measured", type=int, default=100)
    parser.add_argument(
        "--production-dw1-ms", type=float,
        help="Optional exact device timestamp span from BACKWARD_BENCH_JSON")
    args = parser.parse_args()
    if args.warmup < 0 or args.measured <= 0:
        raise SystemExit("warmup must be non-negative and measured positive")

    import deep_gemm  # Imported after argument validation for useful errors.

    torch.cuda.set_device(args.device)
    groups = args.experts
    output_rows = 2 * args.intermediate
    output_columns = args.hidden
    reduction = args.canonical_rows

    # DeepGEMM grouped NT: A[M,K] @ B[N,K]^T.  B is therefore the explicitly
    # pre-transposed X compute operand.  Allocation and initialization are
    # outside the measured region.
    a = torch.empty((groups * output_rows, reduction), device="cuda",
                    dtype=torch.bfloat16)
    b = torch.empty((groups, output_columns, reduction), device="cuda",
                    dtype=torch.bfloat16)
    output = torch.empty((groups * output_rows, output_columns),
                         device="cuda", dtype=torch.bfloat16)
    group_index = torch.arange(groups, device="cuda", dtype=torch.int32)
    group_index = group_index.repeat_interleave(output_rows)
    a.fill_(0.125)
    b.fill_(0.0625)

    def grouped_gemm() -> None:
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
            a, b, output, group_index)

    samples = measure_ms(grouped_gemm, args.warmup, args.measured)
    mean_ms = statistics.fmean(samples)
    flops = 2 * groups * output_rows * output_columns * reduction
    # A + B + BF16 output: an optimistic one-read/one-write HBM floor.
    hbm_floor_bytes = 2 * (
        groups * output_rows * reduction
        + groups * output_columns * reduction
        + groups * output_rows * output_columns)
    result = {
        "schema": "megamoe.dw1-roofline.v1",
        "fixture": {
            "experts": groups,
            "output_rows": output_rows,
            "output_columns": output_columns,
            "canonical_rows": reduction,
            "dtype": "BF16",
            "accumulate": "FP32",
            "split_k": 1,
            "fixture_digest": hashlib.sha256(
                f"{groups}:{output_rows}:{output_columns}:{reduction}:bf16"
                .encode("ascii")).hexdigest(),
        },
        "deep_gemm_grouped_nt": {
            "layout_note": (
                "compute roofline; X is pre-transposed to [expert,N,K]; "
                "transpose/setup excluded"),
            "warmup": args.warmup,
            "measured": args.measured,
            "mean_ms": mean_ms,
            "p50_ms": percentile(samples, 0.50),
            "p95_ms": percentile(samples, 0.95),
            "min_ms": min(samples),
            "max_ms": max(samples),
            "tflops": flops / (mean_ms * 1.0e-3) / 1.0e12,
            "hbm_floor_bytes": hbm_floor_bytes,
            "hbm_floor_gbps": hbm_floor_bytes / (mean_ms * 1.0e-3) / 1.0e9,
        },
        "production_core": {
            "cta_tile": "M128xN256",
            "consumer_wg_tile": "M64xN256",
            "stages": 4,
            "consumes_canonical_row_major_x": True,
            "measured_dw1_ms": args.production_dw1_ms,
        },
    }
    if args.production_dw1_ms is not None:
        result["production_core"]["tflops"] = (
            flops / (args.production_dw1_ms * 1.0e-3) / 1.0e12)
        result["production_core"]["roofline_gap"] = (
            args.production_dw1_ms / mean_ms)
    print("DW1_ROOFLINE_JSON " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
