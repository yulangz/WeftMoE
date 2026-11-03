"""Padding must never introduce duplicate experts into a valid top-k row."""

import pytest
import torch
import triton

from weftmoe._sglang_kernels import sanitize_routes, stage_inputs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("top_k", [4, 6, 8])
def test_partial_invalid_routes_eager_and_graph(top_k):
    ids = torch.arange(top_k, device="cuda", dtype=torch.int32).repeat(4, 1)
    ids[0] = -1
    ids[1, 0], ids[1, 1] = -1, 0
    ids[2, 0], ids[2, 1] = 1, -1
    ids[2, -1] = -1
    weights = torch.rand(ids.shape, device="cuda")
    routes, gates = torch.empty_like(ids), torch.empty_like(weights)

    def check():
        assert (routes >= 0).all() and (routes < top_k).all()
        assert (routes.sort(1).values[:, 1:] != routes.sort(1).values[:, :-1]).all()
        assert torch.equal(routes[ids >= 0], ids[ids >= 0])
        assert torch.equal(gates, torch.where(ids >= 0, weights, 0))

    sanitize_routes[(4,)](
        ids, weights, routes, gates, top_k, triton.next_power_of_2(top_k)
    )
    check()
    x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    staged = torch.empty_like(x)

    def stage():
        stage_inputs[(4,)](
            x,
            ids,
            weights,
            staged,
            routes,
            gates,
            *x.stride(),
            *ids.stride(),
            *weights.stride(),
            0,
            4,
            128,
            top_k,
            128,
            triton.next_power_of_2(top_k),
        )

    stage()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        stage()
    ids[3] = ids[1]
    graph.replay()
    check()
    assert torch.equal(staged, x)
