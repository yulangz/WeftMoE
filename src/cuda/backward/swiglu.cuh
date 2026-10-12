#pragma once

struct BackwardSwigluPair {
  unsigned da, db, u;
};

// Keep the BF16 dU boundary and the per-element FP32 arithmetic identical for
// scalar-pair and vectorized activation passes.
__device__ __forceinline__ BackwardSwigluPair backward_swiglu_pair(
    unsigned a, unsigned b, unsigned du, float gate, float &partial) {
  BackwardSwigluPair result{};
#pragma unroll
  for (unsigned element = 0U; element < 2U; ++element) {
    const auto shift = element * 16U;
    const auto av = __uint_as_float((a >> shift) << 16U);
    const auto bv = __uint_as_float((b >> shift) << 16U);
    const auto duv = __uint_as_float((du >> shift) << 16U);
    const float denominator = 1.0F + __expf(-av);
    float sigmoid;
    if (!(denominator < 0x1p126f)) {
      sigmoid = 1.0F / denominator;
    } else {
      asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(sigmoid) : "f"(denominator));
      sigmoid =
          __fmaf_rn(sigmoid, __fmaf_rn(-denominator, sigmoid, 1.0F), sigmoid);
    }
    const auto silu = av * sigmoid;
    const auto s = silu * bv;
    const auto ds = gate * duv;
    const auto derivative = sigmoid * (1.0F + av * (1.0F - sigmoid));
    result.da |= unsigned(float_to_bf16_bits(ds * bv * derivative)) << shift;
    result.db |= unsigned(float_to_bf16_bits(ds * silu)) << shift;
    result.u |= unsigned(float_to_bf16_bits(gate * s)) << shift;
    partial = fmaf(duv, s, partial);
  }
  return result;
}
