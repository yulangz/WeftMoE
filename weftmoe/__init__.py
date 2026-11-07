"""WeftMoE configuration and lazy, shape-specific CUDA compilation."""

from .config import ModelSpec, TokenBucket, TOKEN_BUCKETS, token_bucket

__all__ = ["ModelSpec", "TokenBucket", "TOKEN_BUCKETS", "token_bucket"]

from .library import get_include, get_library_path

__all__ += ["get_include", "get_library_path"]
