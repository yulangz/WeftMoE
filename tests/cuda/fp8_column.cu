// nvcc -std=c++17 -O2 -arch=sm_90a -Iinclude -Isrc -I/opt/cutlass/include \
//   tests/cuda/fp8_column.cu -o /tmp/fp8_column
#include "cuda/fp8/column.cuh"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

__global__ void convert(const unsigned char *input, const float *scales,
                        unsigned rows, unsigned char *output, float *out_scales) {
  __shared__ unsigned char tile[128 * 128];
  __shared__ float sf[128];
  for (unsigned i = threadIdx.x; i < 128 * 128; i += blockDim.x)
    tile[i / 128 * 128 + ((i % 128) ^ ((i / 128 % 8) * 16))] = input[i];
  if (threadIdx.x < 128)
    sf[threadIdx.x] = scales[threadIdx.x];
  __syncthreads();
  weftmoe::cuda::fp8::row_to_column(tile, sf, rows, output, out_scales);
}

void check(cudaError_t result) {
  if (result != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(result));
}
float scale(double maximum) {
  return maximum == 0 ? 1.0f : float(std::max(std::ldexp(1.0, -126),
      std::exp2(std::ceil(std::log2(maximum / 448.0)))));
}
int main() {
  unsigned char *input, *output;
  float *sf, *out_sf;
  check(cudaMalloc(&input, 128 * 128));
  check(cudaMalloc(&output, 128 * 128));
  check(cudaMalloc(&sf, 128 * sizeof(float)));
  check(cudaMalloc(&out_sf, 128 * sizeof(float)));
  unsigned cases = 0;
  for (unsigned pattern = 0; pattern < 6; ++pattern)
    for (unsigned rows : {0, 1, 31, 65, 127, 128}) {
      std::vector<unsigned char> q(128 * 128, 0x7f), actual(q.size());
      std::vector<float> scales(128, NAN), actual_sf(128), decoded(q.size(), 0.0f);
      for (unsigned r = 0; r < rows; ++r) {
        float values[128];
        double maximum = 0;
        for (unsigned c = 0; c < 128; ++c) {
          const int exponent = pattern == 0 ? int(r % 17) - 8
              : pattern == 1 ? int(r * 7 % 247) - 126 : -126;
          values[c] = pattern == 2 ? 0.0f : float(__nv_bfloat16(
              std::ldexp(float(int((r * 37 + c * 13) % 257) - 128) / 128.0f, exponent)));
          maximum = std::max(maximum, std::abs(double(values[c])));
        }
        scales[r] = scale(maximum);
        for (unsigned c = 0; c < 128; ++c) {
          const __nv_fp8_e4m3 value(values[c] / scales[r]);
          q[r * 128 + c] = value.__x;
          decoded[r * 128 + c] = float(value) * scales[r];
        }
      }
      if (rows && pattern == 4) {
        q[3] = 0x7f;
        decoded[3] = NAN;
      }
      if (rows && pattern == 5) {
        scales[0] = NAN;
        for (unsigned c = 0; c < 128; ++c)
          decoded[c] = NAN;
      }
      check(cudaMemcpy(input, q.data(), q.size(), cudaMemcpyHostToDevice));
      check(cudaMemcpy(sf, scales.data(), 128 * sizeof(float), cudaMemcpyHostToDevice));
      convert<<<1, 256>>>(input, sf, rows, output, out_sf);
      check(cudaGetLastError());
      check(cudaMemcpy(actual.data(), output, actual.size(), cudaMemcpyDeviceToHost));
      check(cudaMemcpy(actual_sf.data(), out_sf, 128 * sizeof(float), cudaMemcpyDeviceToHost));
      for (unsigned c = 0; c < 128; ++c) {
        double maximum = 0;
        bool invalid = false;
        for (unsigned r = 0; r < rows; ++r) {
          invalid |= !std::isfinite(decoded[r * 128 + c]);
          maximum = std::max(maximum, std::abs(double(decoded[r * 128 + c])));
        }
        const auto expected_sf = invalid ? NAN : scale(maximum);
        if (invalid ? !std::isnan(actual_sf[c]) : actual_sf[c] != expected_sf)
          throw std::runtime_error("column scale mismatch");
        for (unsigned r = 0; r < 128; ++r) {
          const __nv_fp8_e4m3 expected(decoded[r * 128 + c] / expected_sf);
          if (actual[c * 128 + (r ^ ((c % 8) * 16))] != expected.__x)
            throw std::runtime_error("column value or swizzle mismatch");
        }
      }
      ++cases;
    }
  check(cudaFree(input)); check(cudaFree(output));
  check(cudaFree(sf)); check(cudaFree(out_sf));
  std::cout << "PASS " << cases << " row-to-column cases, including poisoned tails, BF16 subnormals and NaNs\n";
}
