"""Scale range diagnostics for representative NVFP4 training blocks."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
from weftmoe.nvfp4 import nvfp4_block_stats, prepare_nvfp4_weights  # noqa: E402


def test_default_dy_scale_boundaries():
    scale = 1 / 32
    # Zero input, underflow, half-ulp tie, min subnormal, normal, max, overflow.
    maxima = torch.tensor([0, 2**-16, 6 * scale * 2**-10, 6 * scale * 2**-9, 0.1, 84, 100])
    blocks = maxima[:, None].expand(-1, 16)
    assert nvfp4_block_stats(blocks, scale) == {
        "blocks": 7, "nonzero_to_zero": 2, "subnormal_scale": 1, "above_range": 1,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Weight preparation requires CUDA")
def test_tiled_pack_rejects_partial_k_tile():
    # I=256 selects a 128-byte K tile. FC1 only has 192 packed bytes per row.
    w1 = torch.empty((1, 512, 384), dtype=torch.bfloat16, device="cuda")
    w2 = torch.empty((1, 384, 256), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="whole 128x128 weight tiles"):
        prepare_nvfp4_weights(w1, w2, tiled=True)
