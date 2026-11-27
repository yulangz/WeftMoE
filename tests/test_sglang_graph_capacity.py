import pytest

from weftmoe._sglang_api import _graph_tokens


@pytest.mark.parametrize(("max_bs", "buckets", "verify_width", "tokens"), [
    (64, [1, 32, 64], 9, 576),
    (None, [1, 4, 63], 9, 567),
    (64, None, 9, 576),
    (64, [1, 32, 64], None, 64),
    (None, None, None, 1),
])
def test_decode_graph_token_capacity(max_bs, buckets, verify_width, tokens):
    assert _graph_tokens(max_bs, buckets, verify_width) == tokens
