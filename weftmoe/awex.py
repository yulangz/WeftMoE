"""Weight metadata for WeftMoE's DP-local SGLang layout."""

from functools import cache, wraps


@cache
def replicate_qwen35_shared_experts():
    """Match the shared MLP instantiated with TP=1 by the SGLang adapter."""
    from awex.models.qwen3_5 import Qwen3_5ShardingStrategy
    from awex.sharding.param_sharding import ShardingType

    original = Qwen3_5ShardingStrategy.get_shared_expert_sharding_strategy

    @wraps(original)
    def shared_expert_sharding(self, parameter_name, **kwargs):
        kind, dimension, shards = original(self, parameter_name, **kwargs)
        if self.engine_name == "sglang" and self.enable_dp_attention:
            return ShardingType.NO_SHARDING, dimension, 1
        return kind, dimension, shards

    Qwen3_5ShardingStrategy.get_shared_expert_sharding_strategy = shared_expert_sharding
