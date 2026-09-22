#!/usr/bin/env python3
"""Launch independent local CUDA ranks and check the synthetic P2P result.

This oracle intentionally reimplements routing offsets, canonical rows, BF16
rounding and expert-order reduction. It never imports or invokes WeftMoE's C++
reference/materialization code.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import select
import statistics
import struct
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def validate_buffer_mode_results(observed, world):
    required = {"inference_graph_replays": "6", "retained_capacity": "1",
                "skew": "1", "gradient_bits": "all"}
    if (len(observed) != world or
            {item.get("rank") for item in observed} !=
            {str(rank) for rank in range(world)} or
            any(item.get(key) != value for item in observed
                for key, value in required.items())):
        raise AssertionError("incomplete buffer mode checks")


def runner(binary, rank, world, seed, fd, output, scenario, prefix=()):
    return list(prefix) + [
        binary, "--rank-runner", "--rank", str(rank), "--nodes", "1",
        "--gpus-per-node", str(world), "--local-rank", str(rank),
        "--seed", str(seed), "--scenario", scenario,
        "--bootstrap-output-fd" if output else "--bootstrap-input-fd", str(fd),
    ]


def start_process_capture(process):
    """Continuously drain both pipes so a verbose rank cannot block peers."""
    buffers = {"stdout": bytearray(), "stderr": bytearray()}

    def drain(stream, destination):
        while True:
            chunk = stream.read(65536)
            if not chunk:
                return
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8", "replace")
            destination.extend(chunk)

    threads = []
    for name in ("stdout", "stderr"):
        thread = threading.Thread(
            target=drain, args=(getattr(process, name), buffers[name]),
            daemon=True)
        thread.start()
        threads.append(thread)
    return buffers, threads


def captured_text(capture):
    buffers, threads = capture
    for thread in threads:
        thread.join(timeout=2.0)
    return (bytes(buffers["stdout"]).decode("utf-8", "replace"),
            bytes(buffers["stderr"]).decode("utf-8", "replace"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--world", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--seed", type=int, default=0x5EED1234)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--integrated-benchmark", action="store_true")
    parser.add_argument("--buffer-modes", action="store_true")
    parser.add_argument("--communication-benchmark", action="store_true")
    parser.add_argument("--forward-smoke", action="store_true")
    parser.add_argument("--forward-benchmark", action="store_true")
    parser.add_argument("--backward-validation", action="store_true")
    parser.add_argument("--backward-benchmark", action="store_true")
    parser.add_argument("--backward-tokens", type=int)
    parser.add_argument("--backward-warmups", type=int)
    parser.add_argument("--backward-output-dir")
    parser.add_argument("--top-k", type=int, choices=(1, 4, 6, 8), default=6)
    parser.add_argument("--gate-mode", choices=("fixed-random", "unit"),
                        default="fixed-random")
    parser.add_argument("--forward-output-dir")
    parser.add_argument("--cancel", action="store_true")
    parser.add_argument("--no-wrap", action="store_true")
    parser.add_argument("--ring-slots", type=int, choices=(1, 2))
    parser.add_argument("--compute-sanitizer-tool",
                        choices=("memcheck", "racecheck", "initcheck"))
    args = parser.parse_args()
    if sum((args.benchmark, args.integrated_benchmark, args.buffer_modes,
            args.forward_smoke, args.forward_benchmark,
            args.backward_validation, args.backward_benchmark,
            args.communication_benchmark)) > 1:
        parser.error("choose only one benchmark/smoke mode")
    benchmark_mode = (args.benchmark or args.integrated_benchmark or
                      args.forward_benchmark or args.backward_benchmark or
                      args.communication_benchmark)
    if args.rounds is None:
        args.rounds = (30 if args.backward_benchmark or
                       args.communication_benchmark else
                       (2 if args.backward_validation else
                        (110 if benchmark_mode else 1)))
    if benchmark_mode and not args.backward_benchmark and args.rounds <= 10:
        parser.error("--benchmark requires more than 10 total rounds")
    if args.backward_warmups is not None and not (
            args.backward_validation or args.backward_benchmark):
        parser.error("--backward-warmups requires a Backward mode")
    backward_warmups = args.backward_warmups
    if backward_warmups is None:
        backward_warmups = 0 if args.backward_validation else 10
    if backward_warmups < 0 or (
            args.backward_benchmark and backward_warmups >= args.rounds):
        parser.error(
            "Backward warmups must be non-negative and leave a measured round"
        )
    environment = os.environ.copy()
    environment.setdefault("NCCL_DEBUG", "WARN")
    environment["WEFTMOE_P2P_TOP_K"] = str(args.top_k)
    environment["WEFTMOE_GATE_MODE"] = args.gate_mode
    if args.integrated_benchmark:
        environment["WEFTMOE_INTEGRATED_BENCH_ROUNDS"] = str(args.rounds)
        environment["WEFTMOE_INTEGRATED_TOP_K"] = str(args.top_k)
    elif args.communication_benchmark:
        environment["WEFTMOE_COMM_BENCH_ROUNDS"] = str(args.rounds)
    elif args.forward_benchmark:
        environment["WEFTMOE_FORWARD_BENCH_ROUNDS"] = str(args.rounds)
    elif args.backward_validation or args.backward_benchmark:
        environment["WEFTMOE_BACKWARD_ROUNDS"] = str(args.rounds)
        environment["WEFTMOE_BACKWARD_WARMUPS"] = str(backward_warmups)
        if args.backward_tokens is not None:
            if args.backward_tokens <= 0:
                parser.error("--backward-tokens must be positive")
            environment[
                "WEFTMOE_BACKWARD_TOKENS" if args.backward_validation
                else "WEFTMOE_P2P_BENCH_TOKENS"
            ] = str(args.backward_tokens)
    else:
        environment["WEFTMOE_P2P_BENCH_ROUNDS" if args.benchmark else
                    "WEFTMOE_P2P_ROUNDS"] = str(args.rounds)
    if args.forward_output_dir:
        if not (args.forward_benchmark or args.forward_smoke):
            parser.error("--forward-output-dir requires a Forward mode")
        os.makedirs(args.forward_output_dir, mode=0o700, exist_ok=True)
        environment["WEFTMOE_FORWARD_OUTPUT_DIR"] = os.path.abspath(
            args.forward_output_dir)
    if args.backward_output_dir:
        if not args.backward_validation:
            parser.error("--backward-output-dir requires validation mode")
        os.makedirs(args.backward_output_dir, mode=0o700, exist_ok=True)
        environment["WEFTMOE_BACKWARD_OUTPUT_DIR"] = os.path.abspath(
            args.backward_output_dir)
    if args.no_wrap:
        environment["WEFTMOE_P2P_NO_WRAP"] = "1"
    if args.ring_slots is not None:
        environment["WEFTMOE_P2P_RING_SLOTS"] = str(args.ring_slots)
    prefix = (() if args.compute_sanitizer_tool is None else
              ("compute-sanitizer", "--tool", args.compute_sanitizer_tool,
               "--print-limit=10", "--error-exitcode=1"))
    read_fd, write_fd = os.pipe()
    processes = []
    if args.communication_benchmark:
        scenario = "communication_bench"
    elif args.buffer_modes:
        scenario = "buffer_modes"
    elif args.backward_benchmark:
        scenario = "backward_bench"
    elif args.backward_validation:
        scenario = "backward_validate"
    elif args.forward_benchmark:
        scenario = "forward_bench"
    elif args.forward_smoke:
        scenario = "forward_smoke"
    elif args.integrated_benchmark:
        scenario = "activation_integrated_bench"
    elif args.benchmark:
        scenario = "activation_p2p_bench"
    elif args.cancel:
        scenario = "activation_p2p_cancel"
    else:
        scenario = "activation_p2p"
    try:
        rank0 = subprocess.Popen(
            runner(args.binary, 0, args.world, args.seed, write_fd, True,
                   scenario,
                   prefix),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False,
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
        captures = [start_process_capture(process) for process in processes]
        deadline = time.monotonic() + args.timeout
        # Fail fast if any peer exits while another rank is still blocked in a
        # collective or peer signal wait; waiting rank 0 first hides the true
        # origin and turns deterministic setup failures into launcher hangs.
        while time.monotonic() < deadline:
            failed = next((index for index, process in enumerate(processes)
                           if process.poll() not in (None, 0)), None)
            if failed is not None:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                for process in processes:
                    try:
                        process.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                stdout, stderr = captured_text(captures[failed])
                sanitizer = ""
                if args.compute_sanitizer_tool:
                    sanitizer = "\n".join(
                        line for line in stdout.splitlines()
                        if line.startswith("========="))[-12000:]
                raise RuntimeError(
                    f"rank={failed} exited early "
                    f"code={processes[failed].returncode} "
                    f"sanitizer={sanitizer} "
                    f"stdout_tail={stdout[-1000:]} stderr_tail={stderr[-3000:]}")
            if all(process.poll() is not None for process in processes):
                break
            time.sleep(0.02)
        if any(process.poll() is None for process in processes):
            for process in processes:
                if process.poll() is None:
                    process.kill()
            for process in processes:
                process.wait()
            tails = []
            for rank, capture in enumerate(captures):
                stdout, stderr = captured_text(capture)
                tails.append(
                    f"rank={rank} stdout_tail={stdout[-500:]} "
                    f"stderr_tail={stderr[-1500:]}")
            raise RuntimeError("rank launcher timed out " + " ".join(tails))
        observed = []
        for rank, process in enumerate(processes):
            stdout, stderr = captured_text(captures[rank])
            if process.returncode != 0:
                sanitizer = "\n".join(
                    line for line in stdout.splitlines()
                    if (line.startswith("=========") or "Invalid" in line or
                        "Error:" in line or
                        "uninitialized" in line.lower() or
                        "hazard" in line.lower()))[-12000:]
                raise RuntimeError(
                    f"rank={rank} exit={process.returncode} "
                    f"sanitizer={sanitizer} "
                    f"stdout_tail={stdout[-1000:]} stderr={stderr[-3000:]}"
                )
            if args.communication_benchmark:
                observed.append(parse_fields(stdout, "COMM_BENCH "))
                continue
            if args.buffer_modes:
                observed.append(parse_fields(stdout, "BUFFER_MODES "))
                continue
            if args.forward_smoke:
                item = parse_fields(stdout, "FORWARD_SMOKE ")
                if (item.get("specialization") !=
                        "sm90-wgmma-tma-7168x3072" or
                        item.get("persistent_launches") != "1" or
                        item.get("repeat_equal") != "1" or
                        item.get("backward_state_equal") != "1" or
                        item.get("repeats") != "20" or
                        item.get("gate_mode") != args.gate_mode or
                        item.get("top_k") != str(args.top_k)):
                    raise AssertionError(
                        f"rank={rank} invalid forward smoke audit: {item}")
                observed.append(item)
                continue
            if args.backward_validation or args.backward_benchmark:
                item = parse_fields(stdout, "BACKWARD_RESULT ")
                item["_nccl_runtime"] = parse_fields(
                    stdout, "NCCL_RUNTIME ")["runtime"]
                observed.append(item)
                continue
            if args.cancel:
                cancelled = parse_fields(stdout, "P2P_CANCEL ")
                if cancelled.get("cancelled") != "1" or \
                        cancelled.get("quarantined") != "1":
                    raise AssertionError(f"rank={rank} bad cancellation audit")
            else:
                marker = (("FORWARD_GRAPH_BENCH "
                           if (args.forward_benchmark and
                               os.getenv("WEFTMOE_FORWARD_BENCH_GRAPH_REPLAY"))
                           else "FORWARD_BENCH ") if args.forward_benchmark else
                          ("INTEGRATED_BENCH "
                           if args.integrated_benchmark else "P2P_BENCH "))
                item = (parse_fields(stdout, marker)
                        if benchmark_mode else parse_result(stdout))
                if benchmark_mode:
                    runtime = parse_fields(stdout, "NCCL_RUNTIME ")
                    item["_nccl_runtime"] = runtime["runtime"]
                    item["_nccl_library"] = runtime["library"]
                observed.append(item)
        if args.communication_benchmark:
            for item in observed:
                print("COMM_BENCH " + " ".join(
                    f"{key}={value}" for key, value in item.items()), flush=True)
            samples = [[float(value) for value in item["samples"].split(",")]
                       for item in observed]
            if len(samples) != args.world or any(len(row) != args.rounds - 10
                                                for row in samples):
                raise AssertionError("incomplete communication benchmark")
            slowest = [max(row[i] for row in samples)
                       for i in range(args.rounds - 10)]
            print(f"COMM_RESULT world={args.world} "
                  f"p50_us={statistics.median(slowest):.3f} "
                  f"mean_us={statistics.mean(slowest):.3f}", flush=True)
            return
        if args.buffer_modes:
            validate_buffer_mode_results(observed, args.world)
            print("BUFFER_MODES_JSON " + json.dumps(
                {"world": args.world, "ranks": observed}), flush=True)
            print("BUFFER_MODES passed", flush=True)
            return
        if args.forward_smoke:
            hashes = [item["output_hash"] for item in observed]
            if len(hashes) != args.world:
                raise AssertionError("missing forward ranks")
            print(f"[PASS] LocalCudaP2p Forward MegaKernel world={args.world} "
                  "split-role WGMMA/TMA-7168x3072 smoke "
                  f"rank0_output_hash={hashes[0]}")
            return
        if args.backward_validation:
            if any(item["mode"] != "validation" or
                   item["repeat_equal"] != "1" for item in observed):
                raise AssertionError("Backward repeat stability failed")
            # Preserve each rank's actual shape so the outer validator can
            # reject an old binary that ignores runtime dimension overrides.
            for item in observed:
                print("BACKWARD_RESULT " + " ".join(
                    f"{key}={value}" for key, value in item.items()), flush=True)
            result = {
                "schema": 1,
                "benchmark": "cached-x-ab-local-backward-validation",
                "world": args.world,
                "shape": {
                    "tokens_per_rank": int(observed[0]["tokens"]),
                    "hidden": int(observed[0]["hidden"]),
                    "intermediate": int(observed[0]["intermediate"]),
                    "experts": int(observed[0]["experts"]),
                    "top_k": int(observed[0]["top_k"]),
                },
                "per_rank_dx_hash": [int(item["dx_hash"])
                                     for item in observed],
                "per_rank_dgate_hash": [int(item["dgate_hash"])
                                         for item in observed],
                "per_rank_dw1_hash": [int(item["dw1_hash"])
                                       for item in observed],
                "per_rank_dw2_hash": [int(item["dw2_hash"])
                                       for item in observed],
                "repeat_equal": True,
            }
            print("BACKWARD_VALIDATION_JSON " +
                  json.dumps(result, sort_keys=True))
            print(f"[PASS] cached-state Backward stability world={args.world} "
                  f"tokens={args.backward_tokens or 8}")
            return
        if args.backward_benchmark:
            stage_keys = ("wall", "total", "prepare", "compute", "reduce",
                          "transport", "dispatch", "dab", "dx",
                          "dw2", "dw1", "combine",
                          "dw_combine_overlap")
            measured = args.rounds - backward_warmups
            for rank, item in enumerate(observed):
                if item["mode"] != "benchmark":
                    raise AssertionError(
                        f"rank={rank} invalid Backward mode: {item}")
                for key in stage_keys:
                    item[key] = [float(value)
                                 for value in item[key].split(",")]
                    if len(item[key]) != measured:
                        raise AssertionError(
                            f"rank={rank} stage={key} bad sample count")

            def backward_stats(values):
                ordered = sorted(values)
                def percentile(fraction):
                    ordinal = max(1, int(len(ordered) * fraction + 0.999999))
                    return ordered[ordinal - 1]
                return {"mean": statistics.fmean(values),
                        "p50": percentile(0.50),
                        "p95": percentile(0.95),
                        "min": ordered[0], "max": ordered[-1]}

            summary = {}
            for key in stage_keys:
                slowest = [max(item[key][iteration] for item in observed)
                           for iteration in range(measured)]
                summary[key + "_slowest_rank_us"] = backward_stats(slowest)
                summary[key + "_slowest_rank_samples_us"] = slowest
                summary["per_rank_" + key + "_samples_us"] = [
                    item[key] for item in observed]
            summary["per_rank_benchmark_pause_us"] = [
                int(item.get("benchmark_pause_us", 0)) for item in observed]
            if any(("sample_begin_unix_ns" in item) !=
                   ("sample_end_unix_ns" in item) for item in observed):
                raise RuntimeError("Backward timestamp window is incomplete")
            for key in ("sample_begin_unix_ns", "sample_end_unix_ns"):
                if any(key in item for item in observed):
                    values = [[int(value) for value in item.get(key, "").split(",")]
                              for item in observed]
                    if any(len(row) != measured for row in values):
                        raise RuntimeError(f"Backward timestamp count mismatch: {key}")
                    summary["per_rank_" + key] = values
            if "per_rank_sample_begin_unix_ns" in summary:
                ends = summary["per_rank_sample_end_unix_ns"]
                if any(end < begin for starts, stops in zip(
                        summary["per_rank_sample_begin_unix_ns"], ends)
                       for begin, end in zip(starts, stops)):
                    raise RuntimeError("Backward wall-clock timestamps moved backwards")
            result = {
                "schema": 1,
                "benchmark": "cached-x-ab-local-backward-pipeline",
                "allocate_tensors": all(item.get("allocate_tensors") == "1" for item in observed),
                "synchronize_ranks": all(item.get("synchronize_ranks") == "1" for item in observed),
                "per_rank_tensor_allocation_requests": [int(item.get("tensor_allocation_requests", 0)) for item in observed],
                "per_rank_checked_repetitions": [int(item.get("checked_repetitions", 0)) for item in observed],
                "world": args.world,
                "warmup": backward_warmups,
                "measured": measured,
                "shape": {"tokens_per_rank": args.backward_tokens or 8192,
                          "hidden": int(observed[0]["hidden"]),
                          "intermediate": int(observed[0]["intermediate"]),
                          "experts": int(observed[0]["experts"]), "top_k": args.top_k},
                "per_rank_workspace_bytes": [
                    int(item["workspace_bytes"]) for item in observed],
                "per_rank_gemm_launches": [
                    int(item["gemm_launches"]) for item in observed],
                "per_rank_nccl_runtime": [
                    item["_nccl_runtime"] for item in observed],
                **summary,
            }
            print("BACKWARD_BENCH_JSON " +
                  json.dumps(result, sort_keys=True))
            return
        if args.cancel:
            print(f"[PASS] LocalCudaP2p all-rank cancel/quarantine world={args.world}")
            return
        if benchmark_mode:
            if args.forward_benchmark:
                if os.getenv("WEFTMOE_FORWARD_BENCH_GRAPH_REPLAY"):
                    from tools.forward_graph_benchmark import aggregate_forward_graph

                    result = aggregate_forward_graph(observed)
                    if result["world"] != args.world or result["measured"] != args.rounds - 10:
                        raise AssertionError("incomplete Forward graph benchmark")
                    print("FORWARD_GRAPH_BENCH_JSON " +
                          json.dumps(result, sort_keys=True))
                    return
                stage_keys = ("wall", "notify", "forward", "device_kernel",
                              "route_exchange",
                              "dispatch", "fc1", "fc2", "before_combine",
                              "return", "reduce", "after_combine",
                              "fc1_fc2_overlap", "fc2_return_overlap",
                              "return_reduce_overlap")
                if all("buffer_prepare" in item for item in observed):
                    stage_keys += ("buffer_prepare",)
                for rank, item in enumerate(observed):
                    if (item["specialization"] !=
                            f"sm90-wgmma-tma-{item['hidden']}x{item['intermediate']}" or
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
                    summary[key + "_slowest_rank_samples_us"] = slowest
                    summary["per_rank_" + key + "_samples_us"] = [
                        item[key] for item in observed]
                result = {
                    "schema": 1,
                    "benchmark": "single-persistent-forward-main-shape",
                    "world": args.world,
                    "warmup": 10,
                    "measured": args.rounds - 10,
                    "shape": {
                        "tokens_per_rank": int(observed[0]["tokens"]),
                        "hidden": int(observed[0]["hidden"]),
                        "intermediate": int(observed[0]["intermediate"]),
                        "experts": int(observed[0]["experts"]),
                        "top_k": int(observed[0]["top_k"]),
                    },
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
                    "per_rank_first_token_output_unique_at_least": [
                        int(item["first_token_output_unique_at_least"])
                        for item in observed],
                    "per_rank_activation_hash": [
                        int(item["activation_hash"]) for item in observed],
                    "per_rank_route_hash": [
                        int(item["route_hash"]) for item in observed],
                    "per_rank_gate_hash": [
                        int(item["gate_hash"]) for item in observed],
                    "per_rank_weight_sample_hash": [
                        int(item["weight_hash"]) for item in observed],
                    "weight_generator_version": int(
                        observed[0]["weight_generator_version"]),
                    "per_rank_actual_w1_device_hash": [
                        [int(item["actual_w1_hash_low"]),
                         int(item["actual_w1_hash_high"])]
                        for item in observed],
                    "per_rank_actual_w2_device_hash": [
                        [int(item["actual_w2_hash_low"]),
                         int(item["actual_w2_hash_high"])]
                        for item in observed],
                    "per_rank_cluster_supported": [
                        int(item["cluster_supported"]) for item in observed],
                    "per_rank_cluster_requested": [
                        int(item["cluster_requested"]) for item in observed],
                    "per_rank_cluster_actual": [
                        int(item["cluster_actual"]) for item in observed],
                    "per_rank_cluster_rank_mask": [
                        int(item["cluster_rank_mask"]) for item in observed],
                    "per_rank_max_active_clusters": [
                        int(item["max_active_clusters"]) for item in observed],
                    "per_rank_steady_forbidden_cuda_api_calls": [
                        int(item["steady_forbidden_cuda_api_calls"])
                        for item in observed],
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
                if all("first_use_wall_us" in item for item in observed):
                    scopes = {item.get("timing_scope", "notify-buffer-forward") for item in observed}
                    if len(scopes) != 1:
                        raise RuntimeError("Forward ranks disagree on timing scope")
                    result["timing_scope"] = scopes.pop()
                    result["allocate_tensors"] = all(item.get("allocate_tensors") == "1" for item in observed)
                    result["synchronize_ranks"] = all(item.get("synchronize_ranks") == "1" for item in observed)
                    result["per_rank_tensor_allocation_requests"] = [int(item.get("tensor_allocation_requests", 0)) for item in observed]
                    result["per_rank_checked_repetitions"] = [int(item.get("checked_repetitions", 0)) for item in observed]
                    result["per_rank_retained_backward_state"] = [
                        int(item.get("retained_backward_state", -1))
                        for item in observed]
                    for key in ("fc1_compare_mode", "fc1_compare_samples", "fc1_compare_wgmma"):
                        if all(key in item for item in observed):
                            result["per_rank_" + key] = [item[key] for item in observed]
                    result["first_use"] = {
                        key: {
                            "slowest_rank_us": max(float(item[key])
                                                   for item in observed),
                            "per_rank_us": [float(item[key])
                                            for item in observed],
                        }
                        for key in ("first_use_wall_us",
                                    "first_use_notify_pool_us",
                                    "first_use_notify_us",
                                    "first_use_buffer_prepare_us",
                                    "first_use_forward_us")
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
