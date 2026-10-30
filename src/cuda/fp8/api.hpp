#pragma once

#include <cstdint>
#include "weftmoe/version.hpp"

#ifndef WEFTMOE_FP8_PACKED_RING
#define WEFTMOE_FP8_PACKED_RING 1
#endif
#ifndef WEFTMOE_FP8_SOURCE_X
#define WEFTMOE_FP8_SOURCE_X 1
#endif
#ifndef WEFTMOE_FP8_SOURCE_DY
#define WEFTMOE_FP8_SOURCE_DY 1
#endif

struct CUstream_st;

// Internal launch ABI shared by the tensor bindings and native PF kernels.
// All matrices are K-contiguous. Scales are dequantization multipliers:
// value = e4m3_value * scale. The caller owns storage and stream dependencies.
namespace weftmoe::cuda::fp8 {

enum class InputType : unsigned { BFloat16, Float32 };

struct Weights {
  const std::uint8_t *w1 = nullptr, *w2 = nullptr;
  const float *s1 = nullptr, *s2 = nullptr;
  const std::uint8_t *w1_t = nullptr, *w2_t = nullptr;
  const float *s1_t = nullptr, *s2_t = nullptr;
};

struct Matrix {
  const std::uint8_t *data;
  const float *scales;
  unsigned rows, columns;
  unsigned stride;       // Physical row stride, at least ceil(columns/128)*128.
  unsigned scale_rows;   // 1 (activation) or 128 (weight).
};

WEFTMOE_API void quantize(const void *input, InputType type,
                         std::uint8_t *output, float *scales,
                         unsigned batches, unsigned rows, unsigned columns,
                         unsigned block_rows, bool transpose,
                         CUstream_st *stream);

// Quantize high-precision weights once and emit both physical orientations.
// Output data pointers must be 4-byte aligned; arbitrary input alignment is supported.
WEFTMOE_API void quantize_weight(const void *input, InputType type,
                                 std::uint8_t *output, float *scales,
                                 std::uint8_t *output_t, float *scales_t,
                                 unsigned batches, unsigned rows, unsigned columns,
                                 CUstream_st *stream);

// Transpose an already quantized 128x128-block matrix without requantizing.
WEFTMOE_API void transpose_weight(Matrix input, std::uint8_t *output,
                                 float *scales, unsigned batches,
                                 CUstream_st *stream);

// D = A @ B.T. Optional accumulation reads FP32 D before writing it.
// Both operands have per-128 K scaling; each may use 1 or 128 rows/scale.
WEFTMOE_API void gemm(Matrix a, Matrix b, void *output, bool output_float,
                     bool accumulate, CUstream_st *stream);

} // namespace weftmoe::cuda::fp8
