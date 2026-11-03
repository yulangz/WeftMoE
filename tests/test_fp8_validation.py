import math

import pytest
import torch

from benchmarks.fp8_validation import DIFF_LIMIT, assert_diff, global_diff, limit_for, passes


def test_global_diff_zero_and_empty():
    for x in (torch.zeros(3), torch.empty(0), torch.tensor([1., -2.])):
        assert global_diff(x, x) == 0
    assert global_diff(torch.ones(3), torch.zeros(3)) == 1
    assert global_diff(torch.ones(3), -torch.ones(3)) == 2


def test_near_zero_element_does_not_impose_elementwise_relative_tolerance():
    assert_diff(torch.tensor([1., 1e-4]), torch.tensor([1., 1e-8]))
    x, y = torch.tensor([1., 2., -3.]), torch.tensor([2., 2., -1.])
    expected = (x.double() - y.double()).square().sum() / (x.double().square().sum() + y.double().square().sum())
    assert global_diff(x, y) == expected.item()


@pytest.mark.parametrize('name', ['output', 'gemm', 'dx', 'dw1', 'async_dw2'])
def test_strict_shared_boundary(name):
    assert limit_for(name) == DIFF_LIMIT == 5e-3
    assert passes(math.nextafter(DIFF_LIMIT, 0), name)
    assert not passes(DIFF_LIMIT, name)
    assert not passes(float('nan'), name)


@pytest.mark.parametrize('name', ['dg', 'dgate', 'dprob', 'async_dg', 'checkpoint_dg'])
def test_dprob_is_report_only_but_must_be_finite(name):
    assert limit_for(name) is None
    assert passes(.5, name)
    assert not passes(float('nan'), name)
    assert not passes(float('inf'), name)


def test_nonfinite_and_wrong_shape_are_not_hidden():
    assert math.isnan(global_diff(torch.tensor([float('inf')]), torch.ones(1)))
    with pytest.raises(AssertionError):
        assert_diff(torch.tensor([float('nan')]), torch.ones(1), 'dprob')
    with pytest.raises(ValueError):
        global_diff(torch.zeros(1), torch.zeros(2))


def test_quantized_reference_catches_missing_small_k_block():
    reference = torch.ones(128)
    # Dropping a small K-block contribution must not use the BF16 error budget.
    actual = reference - 0.01
    assert_diff(actual, reference, 'gemm')
    for name in ('quantized_gemm', 'quantized_gemm_accumulate'):
        assert limit_for(name) == 1e-6
        with pytest.raises(AssertionError):
            assert_diff(actual, reference, name)
