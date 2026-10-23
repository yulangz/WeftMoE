"""Framework-independent preparation settings (no process-group initialization)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    hidden: int
    intermediate: int
    experts: int
    top_k: int

    def __post_init__(self):
        for name in ("hidden", "intermediate", "experts", "top_k"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.top_k > self.experts:
            raise ValueError("top_k exceeds experts")


@dataclass(frozen=True)
class TokenBucket:
    lower: int
    upper: int

    @property
    def tune_tokens(self) -> int:
        return self.upper


TOKEN_BUCKETS = (TokenBucket(0, 64), TokenBucket(65, 8192),
                 TokenBucket(8193, 65536))


def token_bucket(tokens: int) -> TokenBucket:
    """Select a tuning representative, without padding the actual input.

    Empty local batches participate in EP and share the smallest bucket.
    Larger inputs extend the buckets geometrically; there is no token ceiling.
    """
    if type(tokens) is not int or tokens < 0:
        raise ValueError("tokens must be a nonnegative integer")
    for bucket in TOKEN_BUCKETS:
        if tokens <= bucket.upper:
            return bucket
    upper = 1 << (tokens - 1).bit_length()
    return TokenBucket(upper // 2 + 1, upper)
