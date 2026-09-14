"""Opt-in, untimed-setup diagnostics for the training Forward baseline.

Keep these measurements separate from the allocation-inclusive end-to-end
benchmark. CUDA stage events follow the current stream's real dependencies;
they include host launch gaps and are not sums of kernel execution time.
"""
from pathlib import Path
import json
import time

import torch
import torch.distributed as dist

from megatron_dispatch_baseline import measure, stats


def diagnose_forward(args, dispatcher, experts, hidden, selected, gates,
                     routing, build_graph):
    directory = Path(args.forward_diagnostics_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rank = dist.get_rank()
    names = ["input_allocation_and_routing", "dispatch_preprocess",
             "deepep_dispatch", "dispatch_postprocess", "experts",
             "combine_preprocess", "deepep_combine", "combine_postprocess"]

    def forward(events=None, communication_only=False, fresh=True):
        index = 0

        def mark():
            nonlocal index
            if events is not None:
                events[index].record()
            index += 1

        mark()
        x = hidden.detach().clone().requires_grad_(True) if fresh else hidden
        ids = selected.clone() if fresh else selected
        gs = gates.detach().clone().requires_grad_(True) if fresh else gates
        mask = torch.zeros_like(routing).scatter_(1, ids, True) if fresh else routing
        probs = torch.zeros_like(routing, dtype=torch.float32).scatter(1, ids, gs)
        mark()
        x, p = dispatcher.dispatch_preprocess(x, mask, probs)
        mark()
        x, p = dispatcher.token_dispatch(x, p)
        mark()
        x, counts, p = dispatcher.dispatch_postprocess(x, p)
        mark()
        if not communication_only:
            x, _ = experts(x, counts, p)
        mark()
        x = dispatcher.combine_preprocess(x)
        mark()
        x = dispatcher.token_combine(x)
        mark()
        x = dispatcher.combine_postprocess(x)
        mark()
        return x

    reference = build_graph(True)
    candidate = forward()
    torch.cuda.synchronize()
    exact = torch.tensor(int(torch.equal(reference, candidate)), device="cuda")
    dist.all_reduce(exact, op=dist.ReduceOp.MIN)
    if not exact.item():
        raise RuntimeError("diagnostic Forward differs from training Forward")
    del reference, candidate

    def stages(communication_only):
        events = [torch.cuda.Event(enable_timing=True) for _ in range(len(names) + 1)]
        for event in events:
            event.record()
        events[-1].synchronize()
        samples = []
        for iteration in range(5 + args.iterations):
            dist.barrier()
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            output = forward(events, communication_only)
            events[-1].synchronize()
            wall = (time.perf_counter_ns() - start) / 1e6
            row = [events[i].elapsed_time(events[i + 1]) for i in range(len(names))]
            row += [events[0].elapsed_time(events[-1]), wall]
            del output
            local = torch.tensor(row, dtype=torch.float64, device="cuda")
            gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, local)
            if iteration >= 5:
                samples.append([item.cpu().tolist() for item in gathered])
        critical = [max(rows, key=lambda row: row[-2]) for rows in samples]
        max_cuda = [max(row[-2] for row in rows) for rows in samples]
        max_wall = [max(row[-1] for row in rows) for rows in samples]
        return {"names": names + ["cuda_total", "wall"], "all_rank_samples_ms": samples,
                "cuda_event": stats(max_cuda), "host_wall": stats(max_wall),
                "max_rank_cuda_samples_ms": max_cuda, "max_rank_wall_samples_ms": max_wall,
                "critical_rank_stages": {name: stats([row[i] for row in critical])
                                         for i, name in enumerate(names + ["cuda_total", "wall"])}}

    result = {"world": dist.get_world_size(), "tokens_per_rank": args.tokens,
              "output_byte_identical": True,
              "forward_stages": stages(False), "identity_stages": stages(True)}

    # Prepare the exact expert input once; counts are the actual routing counts.
    probs = torch.zeros_like(routing, dtype=torch.float32).scatter(1, selected, gates)
    x, p = dispatcher.dispatch_preprocess(hidden, routing, probs)
    x, p = dispatcher.token_dispatch(x, p)
    x, counts, p = dispatcher.dispatch_postprocess(x, p)
    count_list = counts.tolist()
    permuted_probs = p.unsqueeze(-1)
    # Complete the dispatch/combine protocol before the isolated measurements.
    returned = dispatcher.token_combine(dispatcher.combine_preprocess(x))
    dispatcher.combine_postprocess(returned)
    torch.cuda.synchronize()
    result["local_expert_rows"] = count_list
    def graph_measure(function):
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        error = None
        captured = None
        try:
            with torch.cuda.graph(graph):
                captured = function()
        except RuntimeError as exception:
            error = str(exception)
        failed = torch.tensor(int(error is not None), device="cuda")
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if failed.item():
            return {"unavailable": error or "capture failed on a peer"}
        def replay():
            graph.replay()
            return captured
        return measure(replay, 5, args.iterations)

    result["isolated_experts"] = measure(lambda: experts(x, counts, p), 5, args.iterations)
    result["isolated_fc1"] = measure(lambda: experts.linear_fc1(x, count_list), 5, args.iterations)
    result["graph_fc1"] = graph_measure(lambda: experts.linear_fc1(x, count_list))
    ab, bias = experts.linear_fc1(x, count_list)
    result["isolated_activation"] = measure(
        lambda: experts.bias_act_func(ab, bias, permuted_probs), 5, args.iterations)
    u = experts.bias_act_func(ab, bias, permuted_probs)
    result["isolated_fc2"] = measure(lambda: experts.linear_fc2(u, count_list), 5, args.iterations)
    result["graph_fc2"] = graph_measure(lambda: experts.linear_fc2(u, count_list))
    del ab, u, x, p, probs, returned

    if dist.get_world_size() > 8:
        # Aggregate physical HCA counters per node. Rank barriers add a small
        # amount of control traffic, so these are wire bytes, not payload bytes.
        def counters():
            if rank % 8:
                return None
            return {p.name: {name: int((p / "ports/1/counters" / name).read_text()) * 4
                            for name in ("port_xmit_data", "port_rcv_data")}
                    for p in Path("/sys/class/infiniband").iterdir()}
        wire = []
        for _ in range(3):
            probs = torch.zeros_like(routing, dtype=torch.float32).scatter(1, selected, gates)
            dist.barrier()
            torch.cuda.synchronize()
            before = counters()
            dist.barrier()
            x, p = dispatcher.dispatch_preprocess(hidden, routing, probs)
            x, p = dispatcher.token_dispatch(x, p)
            x, counts, p = dispatcher.dispatch_postprocess(x, p)
            torch.cuda.synchronize()
            dist.barrier()
            torch.cuda.synchronize()
            middle = counters()
            dist.barrier()
            returned = dispatcher.token_combine(dispatcher.combine_preprocess(x))
            output = dispatcher.combine_postprocess(returned)
            torch.cuda.synchronize()
            dist.barrier()
            torch.cuda.synchronize()
            after = counters()
            if before is not None:
                wire.append({"dispatch": {n: {k: middle[n][k] - before[n][k] for k in before[n]} for n in before},
                             "combine": {n: {k: after[n][k] - middle[n][k] for k in before[n]} for n in before}})
            del x, p, counts, probs, returned, output
        result["hca_wire_bytes"] = wire

    # Only rank zero writes a Chrome trace; every rank executes the same calls.
    dist.barrier()
    torch.cuda.synchronize()
    if rank == 0:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profile:
            output = forward()
            torch.cuda.synchronize()
        profile.export_chrome_trace(str(directory / "flex_rank0_trace.json"))
    else:
        output = forward()
        torch.cuda.synchronize()
    del output
    dist.barrier()
    (directory / f"flex_rank{rank}_diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
    return {"directory": str(directory), "output_byte_identical": True}
