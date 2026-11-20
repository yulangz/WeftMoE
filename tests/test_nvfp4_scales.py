"""Scale range diagnostics for representative NVFP4 training blocks."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
from weftmoe.nvfp4 import nvfp4_block_stats


def test_default_dy_scale_boundaries():
    scale = 1 / 32
    # Zero input, underflow, half-ulp tie, min subnormal, normal, max, overflow.
    maxima = torch.tensor([0, 2**-16, 6 * scale * 2**-10, 6 * scale * 2**-9, 0.1, 84, 100])
    blocks = maxima[:, None].expand(-1, 16)
    assert nvfp4_block_stats(blocks, scale) == {
        "blocks": 7, "nonzero_to_zero": 2, "subnormal_scale": 1, "above_range": 1,
    }
