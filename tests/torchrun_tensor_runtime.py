"""Distributed application-tensor forward/backward parity; run with torchrun."""

import os
import gc
import weakref

import torch
import torch.distributed as dist

from weftmoe.runtime import TensorRuntime, moe


def reference_moe(x, route, gate, w1, w2):
    output = torch.zeros_like(x)
    for expert in range(w1.shape[0]):
        rows, slots = torch.where(route == expert)
        ab = x[rows] @ w1[expert].T
        a, b = ab.chunk(2, dim=-1)
        u = (
            torch.nn.functional.silu(a.float()) * b.float() * gate[rows, slots, None]
        ).to(torch.bfloat16)
        output.index_add_(0, rows, u @ w2[expert].T)
    return output


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    experts = int(os.environ.get("TEST_EXPERTS", max(16, world * 2)))
    tokens = int(os.environ.get("TEST_TOKENS", 8))
    top_k = int(os.environ.get("TEST_TOP_K", 4))
    max_rows = int(os.environ.get(
        "TEST_MAX_ROWS", max(2048, 2 * tokens * top_k + (experts // world) * 128)
    ))
    hidden = int(os.environ.get("TEST_HIDDEN", 2048))
    intermediate = int(os.environ.get("TEST_INTERMEDIATE", 512))
    generator = torch.Generator(device="cuda").manual_seed(123)

    def rand(*shape):
        return (
            torch.randn(
                *shape, device="cuda", dtype=torch.bfloat16, generator=generator
            )
            * 0.03
        ).requires_grad_(True)

    all_x = rand(world * tokens, hidden)
    all_w1 = rand(experts, 2 * intermediate, hidden)
    all_w2 = rand(experts, hidden, intermediate)
    route = (
        torch.arange(world * tokens * top_k, device="cuda").reshape(-1, top_k) % experts
    ).int()
    if os.environ.get("TEST_ROUTE_SKEW") == "1":
        # Concentrate every route on the first rank while retaining distinct
        # top-k experts.  This covers zero-incoming-row ranks, multi-M-block
        # experts, and source-major Return intervals in the static inference
        # scheduler.
        route = torch.arange(top_k, device="cuda", dtype=torch.int32).expand(
            world * tokens, top_k
        ).contiguous()
    gate = torch.full(
        (world * tokens, top_k), 1 / top_k, device="cuda", requires_grad=True
    )
    dy = rand(world * tokens, hidden).detach()
    x = all_x.detach()[rank * tokens : (rank + 1) * tokens].clone().requires_grad_()
    w1 = all_w1.detach().chunk(world)[rank].clone().requires_grad_()
    w2 = all_w2.detach().chunk(world)[rank].clone().requires_grad_()
    g = gate.detach().chunk(world)[rank].clone().requires_grad_()
    runtime = TensorRuntime(
        group=dist.group.WORLD,
        experts=experts,
        top_k=top_k,
        hidden=hidden,
        intermediate=intermediate,
        max_tokens=int(os.environ.get("TEST_TOKEN_CAPACITY", tokens)),
        max_rows=max_rows,
        training=True,
        max_saved=4,
    )
    retry_checks = {}
    if os.environ.get("TEST_ROUTE_VALIDATION") == "1":
        local_routes = route.chunk(world)[rank].contiguous()
        for invalid_expert in (experts, int(local_routes[0, 0])):
            invalid_routes = local_routes.clone()
            if rank == world - 1:
                invalid_routes[0, 1] = invalid_expert
            try:
                runtime.forward(x, invalid_routes, g, w1, w2)
            except RuntimeError:
                pass
            else:
                raise AssertionError("invalid routes did not fail collectively")
            assert runtime.native.usage()["notify_slots_in_use"] == 0
        print(rank, "TRAINING_ROUTE_REJECTION_PASS", flush=True)
    if os.environ.get("TEST_TENSOR_PREPARATION_FAILURE") == "1":
        import ctypes

        cuda = ctypes.CDLL("libcudart.so")
        cuda.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        # Fail after partial allocation on one rank, without exhausting the
        # machine. Peers must throw before main; the retained Backward tape
        # must remain usable after its workspace allocation fails.
        def fail_allocation(call, ordinal):
            original = torch.empty
            calls = 0

            def allocate(*args, **kwargs):
                nonlocal calls
                calls += 1
                if rank == world - 1 and calls == ordinal:
                    if ordinal == 5:
                        # An impossible allocation returns a real recoverable
                        # CUDA OOM, including its thread-local last-error state.
                        address = ctypes.c_void_p()
                        assert cuda.cudaMalloc(ctypes.byref(address), 1 << 63) == 2
                    raise torch.cuda.OutOfMemoryError("injected tensor allocation failure")
                return original(*args, **kwargs)

            torch.empty = allocate
            try:
                try:
                    call()
                except RuntimeError as error:
                    assert f"tensor preparation failed on rank {world - 1}" in str(error)
                else:
                    raise AssertionError("preparation failure did not reach every rank")
            finally:
                torch.empty = original
            print(rank, "TENSOR_PREPARATION_FAILURE_PASS", ordinal, flush=True)

        local_routes = route.chunk(world)[rank].contiguous()
        local_dy = dy.chunk(world)[rank].contiguous()
        fail_allocation(lambda: runtime.forward(x, local_routes, g, w1, w2), 3)
        assert runtime.native.usage()["notify_slots_in_use"] == 0
        tape = runtime.forward(x, local_routes, g, w1, w2)
        retry_checks["output"] = tape.output
        fail_allocation(lambda: runtime.backward(tape, local_dy), 5)
        assert runtime.native.usage()["notify_slots_in_use"] == 1
        retry_checks.update(zip(("dx", "dgate", "dw1", "dw2"),
                                runtime.backward(tape, local_dy)))
        assert runtime.native.usage()["notify_slots_in_use"] == 0
        del tape
    output = moe(runtime, x, route.chunk(world)[rank].contiguous(), g, w1, w2)
    (output.float() * dy.chunk(world)[rank].float()).sum().backward()
    reference = reference_moe(all_x, route, gate, all_w1, all_w2)
    (reference.float() * dy.float()).sum().backward()
    checks = dict(
        output=(output, reference.chunk(world)[rank]),
        dx=(x.grad, all_x.grad.chunk(world)[rank]),
        dgate=(g.grad, gate.grad.chunk(world)[rank]),
        dw1=(w1.grad, all_w1.grad.chunk(world)[rank]),
        dw2=(w2.grad, all_w2.grad.chunk(world)[rank]),
    )
    for name, actual in retry_checks.items():
        torch.testing.assert_close(actual.float(), checks[name][1].float(),
                                   rtol=0.02, atol=0.00002)
    if retry_checks:
        print(rank, "TENSOR_PREPARATION_RETRY_PASS", flush=True)
    for name, (actual, expected) in checks.items():
        delta = (actual.float() - expected.float()).abs()
        print(
            rank,
            name,
            "max_abs",
            delta.max().item(),
            "rms",
            delta.square().mean().sqrt().item(),
            flush=True,
        )
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=0.02, atol=0.00002
        )
    print(rank, "TENSOR_PARITY_PASS", runtime.native.usage(), flush=True)
    if os.environ.get("TEST_PEER_PAYLOAD_REUSE") == "1":
        # Retain three independent tapes with moving hotspots and shorter source
        # batches. Checkpoints remain independent; once both destination
        # hotspots are warmed, the shorter call must reuse communication capacity.
        tapes = []
        cases = ((tokens, 0), (tokens, experts - top_k), (max(1, tokens // 4), 0))
        for count, first_expert in cases:
            case_x = all_x[:world * count]
            case_g = gate[:world * count]
            ids = (torch.arange(top_k, device="cuda", dtype=torch.int32)
                   + first_expert).expand(world * count, top_k).contiguous()
            tapes.append(runtime.forward(
                case_x.chunk(world)[rank].contiguous(),
                ids.chunk(world)[rank].contiguous(),
                case_g.chunk(world)[rank].contiguous(), w1, w2))
        assert runtime.native.usage()["notify_slots_in_use"] == len(cases)
        for index, ((count, first_expert), tape) in enumerate(zip(cases, tapes)):
            case_x = all_x[:world * count]
            case_g = gate[:world * count]
            case_dy = dy[:world * count]
            ids = (torch.arange(top_k, device="cuda", dtype=torch.int32)
                   + first_expert).expand(world * count, top_k).contiguous()
            expected = reference_moe(case_x, ids, case_g, all_w1, all_w2)
            gradients = torch.autograd.grad(
                expected, (case_x, case_g, all_w1, all_w2), case_dy)
            actual = (tape.output, *runtime.backward(
                tape, case_dy.chunk(world)[rank].contiguous()))
            for value, target in zip(actual, (expected, *gradients)):
                torch.testing.assert_close(value.float(), target.chunk(world)[rank].float(),
                                           rtol=0.02, atol=0.00002)
            usage = runtime.native.usage()
            assert usage["notify_slots_in_use"] == len(cases) - index - 1
            if index == 1:
                warmed_compute_bytes = usage["compute_bytes"]
            elif index == 2:
                assert usage["compute_bytes"] == warmed_compute_bytes
            print(rank, "PEER_PAYLOAD_REUSE_PASS", index, usage, flush=True)
        del tapes, tape
        # Reusing packing storage must not cache values across optimizer steps.
        # Keep the weight addresses fixed and change both gate/up projections.
        with torch.no_grad():
            w1.neg_()
            all_w1.neg_()
        expected = reference_moe(all_x, route, gate, all_w1, all_w2)
        gradients = torch.autograd.grad(
            expected, (all_x, gate, all_w1, all_w2), dy)
        tape = runtime.forward(x, route.chunk(world)[rank].contiguous(), g, w1, w2)
        actual = (tape.output, *runtime.backward(
            tape, dy.chunk(world)[rank].contiguous()))
        for value, target in zip(actual, (expected, *gradients)):
            torch.testing.assert_close(value.float(), target.chunk(world)[rank].float(),
                                       rtol=0.02, atol=0.00002)
        del tape
        print(rank, "TRAINING_WEIGHT_REFRESH_PASS", flush=True)
    usage = runtime.native.usage()
    resident_bytes = sum(usage[key] for key in ("compute_bytes", "gin_bytes", "notify_bytes"))
    torch.cuda.empty_cache()
    free_before = torch.cuda.mem_get_info()[0]
    runtime.close()
    torch.cuda.empty_cache()
    freed = torch.cuda.mem_get_info()[0] - free_before
    assert freed >= resident_bytes, (freed, resident_bytes)
    print(rank, "TRAINING_WORKSPACE_RELEASE_PASS", freed, resident_bytes, flush=True)
    del output, runtime
    # AReaL trains across EP16 but serves with independent node-local EP4
    # engines. Allow the test to exercise those distinct process groups.
    inference_world = int(os.environ.get("TEST_INFERENCE_EP_SIZE", world))
    assert world % inference_world == 0
    inference_group = dist.group.WORLD
    if inference_world != world:
        for start in range(0, world, inference_world):
            group = dist.new_group(list(range(start, start + inference_world)))
            if start <= rank < start + inference_world:
                inference_group = group
        inference_rank = dist.get_rank(inference_group)
        w1 = all_w1.detach().chunk(inference_world)[inference_rank].contiguous()
        w2 = all_w2.detach().chunk(inference_world)[inference_rank].contiguous()
    inference = TensorRuntime(
        group=inference_group,
        experts=experts,
        top_k=top_k,
        hidden=hidden,
        intermediate=intermediate,
        max_tokens=tokens,
        max_rows=max(max_rows, 2 * tokens * top_k + (experts // inference_world) * 128),
        training=False,
    )
    ids = route.chunk(world)[rank].contiguous()
    with torch.no_grad():
        # External producer provenance must not create a token/backend cycle
        # that retains application tensors and the NCCL session after completion.
        probe = x.detach().clone()
        probe_ref = weakref.ref(probe)
        inference.inference_eager(probe, ids, g, w1, w2)
        prepared = inference.inference_eager(probe, ids, g, w1, w2, layer=2)
        fresh = inference.inference_eager(probe, ids, g, w1, w2)
        torch.testing.assert_close(prepared, fresh, rtol=0, atol=0)
        inference.invalidate_inference_layer(2)
        prepared = inference.inference_eager(probe[:3], ids[:3], g[:3], w1, w2, layer=2)
        fresh = inference.inference_eager(probe[:3], ids[:3], g[:3], w1, w2)
        torch.testing.assert_close(prepared, fresh, rtol=0, atol=0)
        del prepared, fresh
        del probe
        gc.collect()
        assert probe_ref() is None, "External CUDA producer retained its input"
        # Long prompts use eager prefill. The application's largest default
        # decode bucket is 512 TP-replicated rows, or 128 per EP4 rank.
        graph_tokens = min(tokens, 128)
        x = x[:graph_tokens]
        ids = ids[:graph_tokens]
        g = g[:graph_tokens]
        # Two layers share the communication workspace and bucket staging.
        first = inference.inference(0, x, ids, g, w1, w2)
        inference.inference(1, first, ids, g, w1, w2)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        capture = torch.cuda.graph(graph)
        with capture:
            graph_first = inference.inference(0, x, ids, g, w1, w2)
            graph_second = inference.inference(1, graph_first, ids, g, w1, w2)
        for i in range(4):
            x.mul_(0.9)
            ids.add_(1).remainder_(experts)
            # Other buckets and eager prefill must not invalidate captured
            # metadata or retain an exclusive lease on the shared workspace.
            inference.inference(0, x[:3], ids[:3], g[:3], w1, w2)
            inference.inference_eager(x[:5], ids[:5], g[:5], w1, w2)
            expected_first = inference.inference(0, x, ids, g, w1, w2)
            expected_second = inference.inference(
                1, expected_first, ids, g, w1, w2
            ).clone()
            expected_first = expected_first.clone()
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(graph_first, expected_first, rtol=0, atol=0)
            torch.testing.assert_close(graph_second, expected_second, rtol=0, atol=0)
        benchmark_replays = int(os.environ.get("TEST_GRAPH_BENCH_REPLAYS", "0"))
        if benchmark_replays:
            for _ in range(10):
                graph.replay()
            torch.cuda.synchronize()
            dist.barrier(group=inference_group)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(benchmark_replays):
                graph.replay()
            end.record()
            end.synchronize()
            per_layer_us = start.elapsed_time(end) * 1000 / benchmark_replays / 2
            timings = [None] * inference_world
            dist.all_gather_object(timings, per_layer_us, group=inference_group)
            if dist.get_rank(inference_group) == 0:
                print(
                    "TWO_LAYER_GRAPH_PER_LAYER_US",
                    max(timings),
                    "replays",
                    benchmark_replays,
                    flush=True,
                )
        print(rank, "MULTILAYER_GRAPH_PASS", inference.native.usage(), flush=True)
        del graph
    usage = inference.native.usage()
    resident_bytes = sum(usage[key] for key in ("compute_bytes", "gin_bytes", "notify_bytes"))
    torch.cuda.empty_cache()
    free_before = torch.cuda.mem_get_info()[0]
    inference.close()
    del inference, graph_first, graph_second, capture
    torch.cuda.empty_cache()
    freed = torch.cuda.mem_get_info()[0] - free_before
    assert freed >= resident_bytes, (freed, resident_bytes)
    print(rank, "GRAPH_DESTROY_AND_OFFLOAD_PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
