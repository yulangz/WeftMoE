"""Shared FP8 numerical acceptance policy (no elementwise tolerances)."""
import math

DIFF_LIMIT = 5e-3
QUANTIZED_DIFF_LIMIT = 1e-6
POLICY = dict(version='bf16-global-diff-v1', reference='original BF16 operands',
              metric='sum((actual-reference)^2)/(sum(actual^2)+sum(reference^2))',
              diff_lt=DIFF_LIMIT, reduction='FP64; check every rank/expert separately',
              dprob='report only; finite values required',
              quantized_reference_diff_lt=QUANTIZED_DIFF_LIMIT,
              source_and_graph_equivalence='bit-exact')


def limit_for(name):
    if name.startswith('quantized_gemm'):
        return QUANTIZED_DIFF_LIMIT
    return None if name.rsplit('_', 1)[-1] in ('dg', 'dgate', 'dprob') else DIFF_LIMIT


def passes(value, name='output'):
    limit = limit_for(name)
    return math.isfinite(value) and value >= 0 and (limit is None or value < limit)


def global_diff(actual, expected):
    """Finite-input global diff; NaN/Inf is invalid even for report-only dprob."""
    import torch
    if actual.shape != expected.shape:
        raise ValueError(f'shape mismatch: {actual.shape} != {expected.shape}')
    a, b = actual.detach().double(), expected.detach().double()
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        return float('nan')
    numerator = (a - b).square().sum()
    denominator = (a.square() + b.square()).sum()
    return (numerator / denominator).item() if denominator.item() else 0.0


def assert_diff(actual, expected, name='output'):
    value = global_diff(actual, expected)
    assert passes(value, name), (name, value, 'diff limit', limit_for(name))
    return value
