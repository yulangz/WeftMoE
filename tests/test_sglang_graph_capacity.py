from types import SimpleNamespace

import pytest

from weftmoe._sglang_api import _legacy_workspace_tokens, _split_workspace_tokens


@pytest.mark.parametrize("api", ["legacy", "split", "split_server_args"])
@pytest.mark.parametrize(
    ("graph", "max_bs", "buckets", "requests", "spec", "expected"),
    [
        # Graph padding and eager target verify both use the target width.
        (True, None, [1, 4, 63], 8, {"speculative_num_draft_tokens": 9}, (567, 567)),
        (True, 64, None, 1024, {"speculative_num_draft_tokens": 9}, (9216, 1024)),
        (False, 64, [64], 1024, {"speculative_num_draft_tokens": 9}, (9216, 1024)),
        (None, None, None, 1024, {"speculative_num_draft_tokens": 9}, (9216, 1024)),
        # Draft decode can be wider than verify, including outside capture.
        (True, 32, [32], 80, {"speculative_num_draft_tokens": 4,
                               "speculative_eagle_topk": 6}, (480, 480)),
        # Draft extend can exceed both target verify and draft decode.
        (True, 64, [64], 80, {"speculative_num_draft_tokens": 4,
                               "speculative_eagle_topk": 2,
                               "speculative_num_steps": 8}, (720, 576)),
        (True, 64, [64], 16, {"speculative_num_draft_tokens": 3,
                               "max_speculative_num_draft_tokens": 8}, (512, 512)),
        # Optional speculative fields may be absent in a non-spec worker.
        (True, 16, [1, 32], 8, {}, (128, 32)),
    ],
)
def test_workspace_tokens(api, graph, max_bs, buckets, requests, spec, expected):
    schedule = dict(chunked_prefill_size=64, max_prefill_tokens=128,
                    max_running_requests=requests)
    if api == "legacy":
        args = SimpleNamespace(**schedule, **spec, disable_cuda_graph=not graph,
                               cuda_graph_bs=buckets, cuda_graph_max_bs=max_bs)
        capacity = _legacy_workspace_tokens
    else:
        config = None if graph is None else SimpleNamespace(
            decode=SimpleNamespace(backend="full" if graph else "disabled",
                                   bs=buckets, max_bs=max_bs))
        args = SimpleNamespace(
            get_exec=lambda: SimpleNamespace(graph=SimpleNamespace(cuda_graph_config=config)),
            get_schedule=lambda: SimpleNamespace(**schedule),
        )
        accessor = "get_spec" if api == "split" else "get_server_args"
        setattr(args, accessor, lambda: SimpleNamespace(**spec))
        capacity = _split_workspace_tokens

    assert (capacity(args, decode=False), capacity(args, decode=True)) == expected
