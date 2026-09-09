"""Framework-independent preparation settings (no process-group initialization)."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping


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
    Inputs beyond the prepared range require an explicit larger workload.
    """
    if type(tokens) is not int or tokens < 0:
        raise ValueError("tokens must be a nonnegative integer")
    for bucket in TOKEN_BUCKETS:
        if tokens <= bucket.upper:
            return bucket
    raise ValueError("tokens exceed the largest prepared bucket (65536)")


@dataclass(frozen=True)
class RuntimeSettings:
    root: Path
    autotune_config: Path | None = None

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None):
        env = os.environ if environment is None else environment
        root = Path(env.get("WEFTMOE_ROOT") or "~/.cache/weftmoe").expanduser().resolve()
        config = env.get("WEFTMOE_AUTOTUNE_CONFIG")
        return cls(root, Path(config).expanduser().resolve() if config else None)

    def compile_directory(self, phase: str) -> Path:
        self._check_phase(phase)
        return self.root / "compile" / phase

    def tune_directory(self, phase: str) -> Path:
        self._check_phase(phase)
        return self.root / "tune" / phase

    @staticmethod
    def _check_phase(phase: str):
        if phase not in ("forward", "backward"):
            raise ValueError(f"unknown phase: {phase}")
