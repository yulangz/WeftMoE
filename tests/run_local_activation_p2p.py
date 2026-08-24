#!/usr/bin/env python3
"""Launch independent local CUDA ranks and check the synthetic P2P result.

This oracle intentionally reimplements routing offsets, canonical rows, BF16
rounding and expert-order reduction. It never imports or invokes MegaMOE's C++
reference/materialization code.
"""

import argparse
import fcntl
import json
import os
import select
import statistics
import struct
import subprocess
import sys
import time

BOOTSTRAP_BYTES = 128


def read_exact_frame(fd, timeout):
    output = bytearray()
    deadline = time.monotonic() + timeout
    while len(output) < BOOTSTRAP_BYTES:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
            raise TimeoutError("local P2P bootstrap timed out")
        chunk = os.read(fd, BOOTSTRAP_BYTES - len(output))
        if not chunk:
            raise RuntimeError("local P2P bootstrap was short")
        output.extend(chunk)
    # A sanitizer launcher can retain the inherited write descriptor in its
    # subreaper even after the rank target has written the fixed frame. Never
    # wait indefinitely for EOF: perform a short, nonblocking trailing-byte
    # check under the launcher's outer timeout. A byte is still an oversize
    # error; EAGAIN means the complete fixed frame is available to peers.
    prior_flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    try:
        fcntl.fcntl(fd, fcntl.F_SETFL, prior_flags | os.O_NONBLOCK)
        readable = select.select([fd], [], [], min(0.05, max(0.0, deadline - time.monotonic())))[0]
        if readable and os.read(fd, 1):
            raise RuntimeError("local P2P bootstrap was oversized")
    finally:
        fcntl.fcntl(fd, fcntl.F_SETFL, prior_flags)
    return bytes(output)


def send_exact(process, frame):
    view = memoryview(frame)
    offset = 0
    try:
        while offset < len(view):
            try:
                written = os.write(process.stdin.fileno(), view[offset:])
            except InterruptedError:
                continue
            if written is None or written <= 0:
                raise RuntimeError("local P2P bootstrap write made no progress")
            offset += written
    finally:
        process.stdin.close()
        process.stdin = None


def bf16(value):
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFFFFFF
    return (bits >> 16) & 0xFFFF


def bf16_float(bits):
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def seeded_routes(world, experts, top_k, seed, max_tokens=10):
    mask = (1 << 64) - 1

    def random():
        nonlocal seed
        seed = (seed * 6364136223846793005 + 1442695040888963407) & mask
        return seed

    result = []
    for _rank in range(world):
        tokens = random() % (max_tokens + 1)
        selected = []
        for _token in range(tokens):
            choices = list(range(experts))
            for slot in range(top_k):
                chosen = slot + random() % (experts - slot)
                choices[slot], choices[chosen] = choices[chosen], choices[slot]
                selected.append(choices[slot])
        result.append((tokens, selected))
    return result


def fixed_random_selected_gates(tokens, top_k, rank,
                                seed=0x4D4547414D4F4547):
    result = []
    for token in range(tokens):
        scores = [1 + ((seed + token * 1315423911 + slot * 2654435761 +
                        rank * 97531) % 65521) for slot in range(top_k)]
        total = sum(scores)
        result.extend(float(score / total) for score in scores)
    # Force the exact FP32 values consumed by the C++/CUDA path.
    return [struct.unpack("<f", struct.pack("<f", value))[0]
            for value in result]


def fp32_hash(values):
    result = 1469598103934665603
    for value in values:
        for byte in struct.pack("<f", value):
            result = ((result ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return result


def expected(world, seed, hidden=17, requested_top_k=6,
             gate_mode="fixed-random"):
    experts = max(16, world * 2)
    top_k = min(requested_top_k, experts)
    routes = seeded_routes(world, experts, top_k, seed)
    counts = [[0] * experts for _ in range(world)]
    offsets = []
    for rank, (_tokens, selected) in enumerate(routes):
        local = [0] * experts
        rank_offsets = []
        for expert in selected:
            rank_offsets.append(local[expert])
            local[expert] += 1
            counts[rank][expert] += 1
        offsets.append(rank_offsets)

    expert_base = [0] * experts
    rank_rows = [0] * world
    for owner in range(world):
        prefix = 0
        for expert in range(experts):
            if expert % world == owner:
                expert_base[expert] = prefix
                prefix += sum(counts[source][expert] for source in range(world))
        rank_rows[owner] = prefix

    capacity_rows = world * 10 * top_k
    canonical = [[0] * (capacity_rows * hidden) for _ in range(world)]
    output = []
    for rank, (tokens, selected) in enumerate(routes):
        gates = ([1.0] * (tokens * top_k) if gate_mode == "unit" else
                 fixed_random_selected_gates(tokens, top_k, rank))
        activations = [
            bf16((rank * 101 + index % 97) / 32.0)
            for index in range(tokens * hidden)
        ]
        contributions = [0] * (tokens * top_k * hidden)
        for route, expert in enumerate(selected):
            token = route // top_k
            source_prefix = sum(counts[source][expert] for source in range(rank))
            row = expert_base[expert] + source_prefix + offsets[rank][route]
            owner = expert % world
            for column in range(hidden):
                value = activations[token * hidden + column]
                canonical[owner][row * hidden + column] = value
                transformed = bf16(
                    bf16_float(value)
                    + ((expert % 251) + 1) / 128.0
                    + (column % 17) / 1024.0
                )
                contributions[route * hidden + column] = transformed
        combined = [0] * (tokens * hidden)
        for token in range(tokens):
            order = sorted(range(top_k), key=lambda slot: selected[token * top_k + slot])
            for column in range(hidden):
                total = 0.0
                for slot in order:
                    route = token * top_k + slot
                    product = struct.unpack(
                        "<f", struct.pack(
                            "<f", gates[route] * bf16_float(
                                contributions[route * hidden + column])))[0]
                    total = struct.unpack(
                        "<f", struct.pack("<f", total + product))[0]
                combined[token * hidden + column] = bf16(total)
        output.append({
            "tokens": tokens,
            "top_k": top_k,
            "gate_hash": str(fp32_hash(gates)),
            "combined": combined,
            "contribution": contributions,
        })
    for rank in range(world):
        output[rank]["canonical"] = canonical[rank]
    return output


def parse_vector(value):
    return [] if value == "" else [int(item) for item in value.split(",")]


def parse_fields(stdout, marker):
    line = next((line for line in stdout.splitlines() if line.startswith(marker)), None)
    if line is None:
        raise RuntimeError(f"rank output has no {marker.strip()}")
    fields = {}
    for item in line[len(marker):].split(" "):
        key, value = item.split("=", 1)
        fields[key] = value
    return fields


def parse_result(stdout):
    fields = parse_fields(stdout, "P2P_RESULT ")
    for key in ("combined", "contribution", "canonical"):
        fields[key] = parse_vector(fields[key])
    return fields


def runner(binary, rank, world, seed, fd, output, scenario, prefix=()):
    return list(prefix) + [
        binary, "--rank-runner", "--rank", str(rank), "--nodes", "1",
        "--gpus-per-node", str(world), "--local-rank", str(rank),
        "--seed", str(seed), "--scenario", scenario,
        "--bootstrap-output-fd" if output else "--bootstrap-input-fd", str(fd),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--world", type=int, choices=(1, 2, 8), required=True)
    parser.add_argument("--seed", type=int, default=0x5EED1234)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--integrated-benchmark", action="store_true")
    parser.add_argument("--forward-smoke", action="store_true")
    parser.add_argument("--forward-benchmark", action="store_true")
    parser.add_argument("--top-k", type=int, choices=(1, 6), default=6)
    parser.add_argument("--gate-mode", choices=("fixed-random", "unit"),
                        default="fixed-random")
    parser.add_argument("--forward-output-dir")
    parser.add_argument("--cancel", action="store_true")
    parser.add_argument("--no-wrap", action="store_true")
    parser.add_argument("--ring-slots", type=int, choices=(1, 2))
    parser.add_argument("--compute-sanitizer-tool",
                        choices=("memcheck", "racecheck", "initcheck"))
    args = parser.parse_args()
    if sum((args.benchmark, args.integrated_benchmark,
            args.forward_smoke, args.forward_benchmark)) > 1:
        parser.error("choose only one benchmark/smoke mode")
    benchmark_mode = (args.benchmark or args.integrated_benchmark or
                      args.forward_benchmark)
    if args.rounds is None:
        args.rounds = 110 if benchmark_mode else 1
    if benchmark_mode and args.rounds <= 10:
        parser.error("--benchmark requires more than 10 total rounds")
    environment = os.environ.copy()
    environment.setdefault("NCCL_DEBUG", "WARN")
    environment["MEGAMOE_P2P_TOP_K"] = str(args.top_k)
    environment["MEGAMOE_GATE_MODE"] = args.gate_mode
    if args.integrated_benchmark:
        environment["MEGAMOE_INTEGRATED_BENCH_ROUNDS"] = str(args.rounds)
        environment["MEGAMOE_INTEGRATED_TOP_K"] = str(args.top_k)
    elif args.forward_benchmark:
        environment["MEGAMOE_FORWARD_BENCH_ROUNDS"] = str(args.rounds)
        if args.forward_output_dir:
            os.makedirs(args.forward_output_dir, mode=0o700, exist_ok=True)
            environment["MEGAMOE_FORWARD_OUTPUT_DIR"] = os.path.abspath(
                args.forward_output_dir)
    else:
        environment["MEGAMOE_P2P_BENCH_ROUNDS" if args.benchmark else
                    "MEGAMOE_P2P_ROUNDS"] = str(args.rounds)
    if args.no_wrap:
        environment["MEGAMOE_P2P_NO_WRAP"] = "1"
    if args.ring_slots is not None:
        environment["MEGAMOE_P2P_RING_SLOTS"] = str(args.ring_slots)
    prefix = (() if args.compute_sanitizer_tool is None else
              ("compute-sanitizer", "--tool", args.compute_sanitizer_tool,
               "--print-limit=10", "--error-exitcode=1"))
    read_fd, write_fd = os.pipe()
    processes = []
    scenario = ("forward_bench" if args.forward_benchmark else
                ("forward_smoke" if args.forward_smoke else
                ("activation_integrated_bench" if args.integrated_benchmark else
                ("activation_p2p_bench" if args.benchmark else
                 ("activation_p2p_cancel" if args.cancel else
                  "activation_p2p")))))
    try:
        rank0 = subprocess.Popen(
            runner(args.binary, 0, args.world, args.seed, write_fd, True,
                   scenario,
                   prefix),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            pass_fds=(write_fd,), env=environment,
        )
        processes.append(rank0)
        os.close(write_fd)
        write_fd = -1
        frame = read_exact_frame(read_fd, min(args.timeout, 60.0))
        os.close(read_fd)
        read_fd = -1
        for rank in range(1, args.world):
            process = subprocess.Popen(
                runner(args.binary, rank, args.world, args.seed, 0, False,
                       scenario,
                       prefix),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=False, env=environment,
            )
            send_exact(process, frame)
            processes.append(process)
        deadline = time.monotonic() + args.timeout
        # Fail fast if any peer exits while another rank is still blocked in a
        # collective or peer signal wait; waiting rank 0 first hides the true
        # origin and turns deterministic setup failures into launcher hangs.
        while time.monotonic() < deadline:
            failed = next((index for index, process in enumerate(processes)
                           if process.poll() not in (None, 0)), None)
            if failed is not None:
                process = processes[failed]
                stdout, stderr = process.communicate()
                if isinstance(stdout, bytes):
                    stdout = stdout.decode("utf-8", "replace")
                    stderr = stderr.decode("utf-8", "replace")
                raise RuntimeError(
                    f"rank={failed} exited early code={process.returncode} "
                    f"stdout_tail={stdout[-1000:]} stderr_tail={stderr[-3000:]}")
            if all(process.poll() is not None for process in processes):
                break
            time.sleep(0.02)
        observed = []
        for rank, process in enumerate(processes):
            timeout = max(0.1, deadline - time.monotonic())
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                if isinstance(stdout, bytes):
                    stdout = stdout.decode("utf-8", "replace")
                    stderr = stderr.decode("utf-8", "replace")
                peer_tails = []
                for peer_rank, peer in enumerate(processes):
                    if peer is process:
                        continue
                    if peer.poll() is None:
                        peer.kill()
                    peer_stdout, peer_stderr = peer.communicate()
                    if isinstance(peer_stdout, bytes):
                        peer_stdout = peer_stdout.decode("utf-8", "replace")
                        peer_stderr = peer_stderr.decode("utf-8", "replace")
                    peer_tails.append(
                        f"peer_rank={peer_rank} stdout_tail={peer_stdout[-500:]} "
                        f"stderr_tail={peer_stderr[-1500:]}")
                raise RuntimeError(
                    f"rank={rank} timed out stdout_tail={stdout[-1000:]} "
                    f"stderr_tail={stderr[-3000:]} " + " ".join(peer_tails))
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", "replace")
                stderr = stderr.decode("utf-8", "replace")
            if process.returncode != 0:
                sanitizer = "\n".join(
                    line for line in stdout.splitlines()
                    if ("=========" in line or "Invalid" in line or
                        "uninitialized" in line.lower() or
                        "hazard" in line.lower()))[-12000:]
                raise RuntimeError(
                    f"rank={rank} exit={process.returncode} "
                    f"sanitizer={sanitizer} "
                    f"stdout_tail={stdout[-1000:]} stderr={stderr[-3000:]}"
                )
            if args.forward_smoke:
                item = parse_fields(stdout, "FORWARD_SMOKE ")
                if (item.get("specialization") !=
                        "sm90-wgmma-tma-7168x3072" or
                        item.get("persistent_launches") != "1" or
                        item.get("cache_equal") != "1" or
                        item.get("repeat_equal") != "1" or
                        item.get("repeats") != "20" or
                        item.get("gate_mode") != args.gate_mode or
                        item.get("top_k") != str(args.top_k)):
                    raise AssertionError(
                        f"rank={rank} invalid forward smoke audit: {item}")
                observed.append(item)
                continue
            if args.cancel:
                cancelled = parse_fields(stdout, "P2P_CANCEL ")
                if cancelled.get("cancelled") != "1" or \
                        cancelled.get("quarantined") != "1":
                    raise AssertionError(f"rank={rank} bad cancellation audit")
            else:
                marker = ("FORWARD_BENCH " if args.forward_benchmark else
                          ("INTEGRATED_BENCH "
                           if args.integrated_benchmark else "P2P_BENCH "))
                item = (parse_fields(stdout, marker)
                        if benchmark_mode else parse_result(stdout))
                if benchmark_mode:
                    runtime = parse_fields(stdout, "NCCL_RUNTIME ")
                    item["_nccl_runtime"] = runtime["runtime"]
                    item["_nccl_library"] = runtime["library"]
                observed.append(item)
        if args.forward_smoke:
            hashes = [item["output_hash"] for item in observed]
            if len(hashes) != args.world:
                raise AssertionError("missing forward ranks")
            print(f"[PASS] LocalCudaP2p Forward MegaKernel world={args.world} "
                  "single-launch WGMMA/TMA cache/no-cache smoke "
                  f"rank0_output_hash={hashes[0]}")
            return
        if args.cancel:
            print(f"[PASS] LocalCudaP2p all-rank cancel/quarantine world={args.world}")
            return
        if benchmark_mode:
            if args.forward_benchmark:
                stage_keys = ("wall", "notify", "forward", "dispatch",
                              "fc1", "fc2", "return", "reduce",
                              "fc1_fc2_overlap", "fc2_return_overlap",
                              "return_reduce_overlap")
                for rank, item in enumerate(observed):
                    if (item["specialization"] !=
                            "sm90-wgmma-tma-7168x3072" or
                            item["persistent_launches"] != "1"):
                        raise AssertionError(
                            f"rank={rank} invalid forward audit: {item}")
                    for key in stage_keys:
                        item[key] = [float(value)
                                     for value in item[key].split(",")]
                        if len(item[key]) != args.rounds - 10:
                            raise AssertionError(
                                f"rank={rank} stage={key} bad sample count")

                def stats(values):
                    ordered = sorted(values)
                    def percentile(fraction):
                        ordinal = max(
                            1, int(len(ordered) * fraction + 0.999999))
                        return ordered[ordinal - 1]
                    return {"mean": statistics.fmean(values),
                            "p50": percentile(0.50),
                            "p95": percentile(0.95),
                            "min": ordered[0], "max": ordered[-1]}

                summary = {}
                for key in stage_keys:
                    slowest = [max(item[key][iteration]
                                   for item in observed)
                               for iteration in range(args.rounds - 10)]
                    summary[key + "_slowest_rank_us"] = stats(slowest)
                result = {
                    "schema": 1,
                    "benchmark": "single-persistent-forward-main-shape",
                    "world": args.world,
                    "warmup": 10,
                    "measured": args.rounds - 10,
                    "shape": {"tokens_per_rank": 8192, "hidden": 7168,
                              "intermediate": 3072, "experts": 384,
                              "top_k": 6},
                    "per_rank_device_pool_bytes": [
                        int(item["device_pool_bytes"]) for item in observed],
                    "per_rank_weight_bytes": [
                        int(item["weight_bytes"]) for item in observed],
                    "per_rank_dispatch_peer_bytes": [
                        int(item["dispatch_peer_bytes"]) for item in observed],
                    "per_rank_return_peer_bytes": [
                        int(item["return_peer_bytes"]) for item in observed],
                    "per_rank_output_hash": [
                        item["output_hash"] for item in observed],
                    "per_rank_wgmma_instructions": [
                        int(item["wgmma"]) for item in observed],
                    "per_rank_tma_loads": [int(item["tma"])
                                           for item in observed],
                    "per_rank_tma_multicast_loads": [
                        int(item["multicast"]) for item in observed],
                    "per_rank_nccl_runtime": [
                        item["_nccl_runtime"] for item in observed],
                    "per_rank_nccl_library": [
                        item["_nccl_library"] for item in observed],
                    **summary,
                }
                print("FORWARD_BENCH_JSON " +
                      json.dumps(result, sort_keys=True))
                return
            if args.integrated_benchmark:
                stage_keys = (
                    "wall", "notify", "mapping", "control",
                    "count_summary", "materialize", "transport",
                    "transport_setup", "dispatch", "relay_return",
                    "reduce", "terminal")
                for rank, item in enumerate(observed):
                    if item["backend"] != "local-cuda-p2p":
                        raise AssertionError(
                            f"rank={rank} backend={item['backend']}")
                    if item["output_equal"] != "1":
                        raise AssertionError(
                            f"rank={rank} integrated output mismatch")
                    for key in stage_keys:
                        item[key] = [float(value)
                                     for value in item[key].split(",")]
                        if len(item[key]) != args.rounds - 10:
                            raise AssertionError(
                                f"rank={rank} stage={key} "
                                f"samples={len(item[key])}")

                def stats(values):
                    ordered = sorted(values)
                    def percentile(fraction):
                        ordinal = max(
                            1, int(len(ordered) * fraction + 0.999999))
                        return ordered[ordinal - 1]
                    return {
                        "mean": statistics.fmean(values),
                        "p50": percentile(0.50),
                        "p95": percentile(0.95),
                        "min": ordered[0],
                        "max": ordered[-1],
                    }

                summary = {}
                for key in stage_keys:
                    slowest = [max(item[key][iteration] for item in observed)
                               for iteration in range(args.rounds - 10)]
                    summary[key + "_slowest_rank_us"] = stats(slowest)
                result = {
                    "schema": 1,
                    "benchmark": "notify-local-p2p-identity-integrated",
                    "backend": "local-cuda-p2p",
                    "world": args.world,
                    "per_rank_nccl_runtime": [
                        item["_nccl_runtime"] for item in observed],
                    "per_rank_nccl_library": [
                        item["_nccl_library"] for item in observed],
                    "warmup": 10,
                    "measured": args.rounds - 10,
                    "one_time_setup_slowest_rank_us": max(
                        float(item["setup_us"]) for item in observed),
                    "expert_ownership": "contiguous-48-per-rank",
                    "expert_body": "identity",
                    "gate_mode": args.gate_mode,
                    "combine": (
                        "source-side per-route FP32 gate-after; expert-id "
                        "FP32 order; final BF16"),
                    "per_rank_gate_hash": [item["gate_hash"]
                                             for item in observed],
                    "per_rank_output_hash": [item["output_hash"]
                                             for item in observed],
                    "per_rank_device_pool_bytes": [int(item["device_bytes"])
                                                    for item in observed],
                    "per_rank_pinned_pool_bytes": [int(item["pinned_bytes"])
                                                    for item in observed],
                    "per_rank_route_input_bytes": [
                        int(item["route_input_bytes"]) for item in observed],
                    "per_rank_local_gate_bytes": [
                        int(item["local_gate_bytes"]) for item in observed],
                    "per_rank_derived_route_bytes": [
                        int(item["derived_route_bytes"]) for item in observed],
                    "per_rank_derived_local_gate_bytes": [
                        int(item["derived_local_gate_bytes"])
                        for item in observed],
                    "per_rank_derived_peer_gate_bytes": [
                        int(item["derived_peer_gate_bytes"])
                        for item in observed],
                    "per_rank_actual_dispatch_peer_bytes": [
                        int(item["actual_dispatch_peer_bytes"])
                        for item in observed],
                    "per_rank_actual_return_peer_bytes": [
                        int(item["actual_return_peer_bytes"])
                        for item in observed],
                    "steady_dynamic_allocations": [
                        int(item["steady_allocs"]) for item in observed],
                    "per_rank_steady_h2d_bytes": [
                        int(item["steady_h2d_bytes"]) for item in observed],
                    "per_rank_steady_d2h_bytes": [
                        int(item["steady_d2h_bytes"]) for item in observed],
                    "per_rank_steady_d2d_bytes": [
                        int(item["steady_d2d_bytes"]) for item in observed],
                    "per_rank_generation_memset_calls": [
                        int(item["generation_memset_calls"])
                        for item in observed],
                    "per_rank_generation_memset_bytes": [
                        int(item["generation_memset_bytes"])
                        for item in observed],
                    "per_rank_kernel_launches": [
                        int(item["kernel_launches"]) for item in observed],
                    "per_rank_collective_enqueues": [{
                        "control": int(item["control_enqueues"]),
                        "config": int(item["config_enqueues"]),
                        "count": int(item["count_enqueues"]),
                        "summary": int(item["summary_enqueues"]),
                    } for item in observed],
                    "per_rank_dispatch_kernel_blocks": [
                        int(item["dispatch_kernel_blocks"])
                        for item in observed],
                    "per_rank_return_kernel_blocks": [
                        int(item["return_kernel_blocks"])
                        for item in observed],
                    "per_rank_reduce_kernel_blocks": [
                        int(item["reduce_kernel_blocks"])
                        for item in observed],
                    **summary,
                }
                peer_bytes = sum(
                    int(item["actual_dispatch_peer_bytes"]) +
                    int(item["actual_return_peer_bytes"])
                    for item in observed)
                result["aggregate_effective_peer_GBps"] = (
                    peer_bytes /
                    (summary["transport_slowest_rank_us"]["mean"] * 1e3))
                print("INTEGRATED_BENCH_JSON " +
                      json.dumps(result, sort_keys=True))
                return
            stage_keys = ("total", "dispatch", "relay", "synthetic", "reduce")
            for rank, item in enumerate(observed):
                if item["backend"] != "local-cuda-p2p":
                    raise AssertionError(f"rank={rank} backend={item['backend']}")
                for key in stage_keys:
                    item[key] = [float(value) for value in item[key].split(",")]
                    expected_samples = args.rounds - 10
                    if len(item[key]) != expected_samples:
                        raise AssertionError(
                            f"rank={rank} stage={key} samples={len(item[key])}")
            def stats(values):
                ordered = sorted(values)
                def percentile(fraction):
                    ordinal = max(1, int(len(ordered) * fraction + 0.999999))
                    return ordered[ordinal - 1]
                return {
                    "mean": statistics.fmean(values),
                    "p50": percentile(0.50),
                    "p95": percentile(0.95),
                    "min": ordered[0],
                    "max": ordered[-1],
                }
            summary = {}
            for key in stage_keys:
                # Each iteration is represented by its slowest rank; values
                # always come from the same per-rank device timing sample.
                slowest = [max(item[key][iteration] for item in observed)
                           for iteration in range(len(observed[0][key]))]
                summary[key + "_slowest_rank_us"] = stats(slowest)
            setup = [float(item["setup_us"]) for item in observed]
            result = {
                "schema": 1,
                "backend": "local-cuda-p2p",
                "world": args.world,
                "warmup": 10,
                "measured": len(observed[0]["total"]),
                "one_time_setup_slowest_rank_us": max(setup),
                "per_rank_device_pool_bytes": [int(item["device_bytes"])
                                                for item in observed],
                "per_rank_pinned_pool_bytes": [int(item["pinned_bytes"])
                                                for item in observed],
                "per_rank_raw_bytes": [int(item["raw_bytes"])
                                       for item in observed],
                "per_rank_dedup_bytes": [int(item["dedup_bytes"])
                                         for item in observed],
                "per_rank_actual_dispatch_peer_bytes": [
                    int(item["actual_dispatch_peer_bytes"])
                    for item in observed],
                "per_rank_actual_return_peer_bytes": [
                    int(item["actual_return_peer_bytes"])
                    for item in observed],
                "per_rank_credit_stalls": [int(item["stalls"])
                                            for item in observed],
                "per_rank_peak_ring": [int(item["peak_ring"])
                                       for item in observed],
                **summary,
            }
            total_peer = [
                int(item["actual_dispatch_peer_bytes"]) +
                int(item["actual_return_peer_bytes"])
                for item in observed]
            result["aggregate_effective_peer_GBps"] = (
                sum(total_peer) /
                (summary["total_slowest_rank_us"]["mean"] * 1e3))
            print("P2P_BENCH_JSON " + json.dumps(result, sort_keys=True))
            return
        oracle = expected(args.world, args.seed, requested_top_k=args.top_k,
                          gate_mode=args.gate_mode)
        for rank, (actual, wanted) in enumerate(zip(observed, oracle)):
            if actual["backend"] != "local-cuda-p2p":
                raise AssertionError(f"rank={rank} backend={actual['backend']}")
            if actual.get("gate_mode") != args.gate_mode:
                raise AssertionError(
                    f"rank={rank} gate mode was not propagated")
            if actual.get("gate_hash") != wanted["gate_hash"]:
                raise AssertionError(
                    f"rank={rank} dynamic gate lease/hash mismatch")
            for field in ("combined", "contribution", "canonical"):
                if actual[field] != wanted[field]:
                    raise AssertionError(
                        f"rank={rank} field={field} differs at length "
                        f"{len(actual[field])}/{len(wanted[field])}"
                    )
        print(
            f"[PASS] independent-process LocalCudaP2p world={args.world} "
            f"rounds={args.rounds} Python synthetic oracle full-field match"
        )
    finally:
        if read_fd >= 0:
            os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        raise
