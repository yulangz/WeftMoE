#include "cuda/fp8/api.hpp"
#include "cuda/fp8/quantization.cuh"
#include "cuda/fp8/gemm.cuh"

#include <stdexcept>
#include <string>

namespace weftmoe::cuda::fp8 {
namespace {
void check_launch() {
  const auto error = cudaGetLastError();
  if (error != cudaSuccess)
    throw std::runtime_error(std::string("FP8 launch: ") + cudaGetErrorString(error));
}

template <class T>
void launch_quantize(const void *input, std::uint8_t *output, float *scales,
                     unsigned batches, unsigned rows, unsigned columns,
                     unsigned block_rows, bool transpose, cudaStream_t stream) {
  const auto m = transpose ? columns : rows, k = transpose ? rows : columns;
  const dim3 grid(((m + block_rows - 1U) / block_rows) * ((k + 127U) / 128U), batches);
  if (!m || !k || !batches)
    return;
  if (block_rows == 1U) {
    if (transpose)
      quantize_kernel<T, 1, true><<<grid, 256, 0, stream>>>(static_cast<const T *>(input), output, scales, rows, columns);
    else
      quantize_kernel<T, 1, false><<<grid, 256, 0, stream>>>(static_cast<const T *>(input), output, scales, rows, columns);
  } else {
    if (transpose)
      quantize_kernel<T, 128, true><<<grid, 256, 0, stream>>>(static_cast<const T *>(input), output, scales, rows, columns);
    else
      quantize_kernel<T, 128, false><<<grid, 256, 0, stream>>>(static_cast<const T *>(input), output, scales, rows, columns);
  }
  check_launch();
}
} // namespace

template <class T>
void launch_quantize_weight(const void *input, std::uint8_t *output, float *scales,
    std::uint8_t *output_t, float *scales_t, unsigned batches, unsigned rows,
    unsigned columns, cudaStream_t stream) {
  if (!rows || !columns || !batches) return;
  const dim3 grid(((rows + 127U) / 128U) * ((columns + 127U) / 128U), batches);
  if ((rows % 128U == 0U) && (columns % 128U == 0U) &&
      (reinterpret_cast<std::uintptr_t>(input) % (4U * sizeof(T)) == 0U))
    quantize_weight_kernel<T, true><<<grid, 256, 0, stream>>>(
        static_cast<const T *>(input), output, scales, output_t, scales_t, rows, columns);
  else
    quantize_weight_kernel<T, false><<<grid, 256, 0, stream>>>(
        static_cast<const T *>(input), output, scales, output_t, scales_t, rows, columns);
  check_launch();
}

void quantize_weight(const void *input, InputType type, std::uint8_t *output,
    float *scales, std::uint8_t *output_t, float *scales_t, unsigned batches,
    unsigned rows, unsigned columns, CUstream_st *stream) {
  if (type == InputType::BFloat16)
    launch_quantize_weight<__nv_bfloat16>(input, output, scales, output_t, scales_t,
                                       batches, rows, columns, stream);
  else if (type == InputType::Float32)
    launch_quantize_weight<float>(input, output, scales, output_t, scales_t,
                                 batches, rows, columns, stream);
  else
    throw std::invalid_argument("FP8 quantization input must be BF16 or FP32");
}

void quantize(const void *input, InputType type, std::uint8_t *output,
              float *scales, unsigned batches, unsigned rows, unsigned columns,
              unsigned block_rows, bool transpose, CUstream_st *stream) {
  if (block_rows != 1U && block_rows != 128U)
    throw std::invalid_argument("FP8 block_rows must be 1 or 128");
  if (type == InputType::BFloat16)
    launch_quantize<__nv_bfloat16>(input, output, scales, batches, rows, columns, block_rows, transpose, stream);
  else if (type == InputType::Float32)
    launch_quantize<float>(input, output, scales, batches, rows, columns, block_rows, transpose, stream);
  else
    throw std::invalid_argument("FP8 quantization input must be BF16 or FP32");
}

void transpose_weight(Matrix input, std::uint8_t *output, float *scales,
                      unsigned batches, CUstream_st *stream) {
  if (input.scale_rows != 128U)
    throw std::invalid_argument("Only 128x128 quantization is transpose invariant");
  if (!input.rows || !input.columns || !batches)
    return;
  transpose_weight_kernel<<<dim3((input.columns + 31U) / 32U,
                                (input.rows + 127U) / 128U * 4U, batches),
                            dim3(32, 8), 0, stream>>>(input, output, scales);
  check_launch();
}

void gemm(Matrix a, Matrix b, void *output, bool output_float,
          bool accumulate, CUstream_st *stream) {
  if (a.columns != b.columns ||
      (a.scale_rows != 1U && a.scale_rows != 128U) ||
      (b.scale_rows != 1U && b.scale_rows != 128U) ||
      a.stride < (a.columns + 127U) / 128U * 128U ||
      b.stride < (b.columns + 127U) / 128U * 128U ||
      (accumulate && !output_float))
    throw std::invalid_argument("Invalid FP8 GEMM layout or output dtype");
  if (!a.rows || !b.rows)
    return;
  const dim3 grid((b.rows + 63U) / 64U, (a.rows + 63U) / 64U);
  if (accumulate)
    gemm_kernel<true, true><<<grid, 128, 0, stream>>>(a, b, output);
  else if (output_float)
    gemm_kernel<true, false><<<grid, 128, 0, stream>>>(a, b, output);
  else
    gemm_kernel<false, false><<<grid, 128, 0, stream>>>(a, b, output);
  check_launch();
}
} // namespace weftmoe::cuda::fp8
