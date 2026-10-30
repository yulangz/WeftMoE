// Standalone source-packet quantization parity, including partial warps and tails.
#include "cuda/fp8/source.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

static void check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

template <unsigned H> void run(unsigned rows, bool special) {
  using namespace weftmoe::cuda::fp8;
  constexpr unsigned Pitch = source_pitch(H);
  static_assert(Pitch <= H * 2U);
  std::vector<__nv_bfloat16> input(std::max(1U, rows * H));
  for (unsigned i = 0; i < rows * H; ++i) {
    const float value = std::sin(float(i % 997U)) * std::ldexp(1.0F, int(i / H % 13U) - 6);
    input[i] = __float2bfloat16(value);
    if (special && i % 128U == 3U) input[i] = __float2bfloat16(1.0e-38F);
    if (special && i % 4096U == 257U) input[i] = __float2bfloat16(NAN);
    if (special && i % 4096U == 513U) input[i] = __float2bfloat16(INFINITY);
    if (i / 128U % 11U == 0U) input[i] = __float2bfloat16(0.0F);
  }
  __nv_bfloat16 *x;
  unsigned char *packet, *reference;
  float *scales;
  check(cudaMalloc(&x, input.size() * 2U));
  check(cudaMalloc(&packet, std::size_t(rows) * Pitch + 256U));
  check(cudaMalloc(&reference, std::max(1U, rows * H)));
  check(cudaMalloc(&scales, std::max(1U, rows * H / 128U) * 4U));
  check(cudaMemcpy(x, input.data(), input.size() * 2U, cudaMemcpyHostToDevice));
  check(cudaMemset(packet, 0xa5, std::size_t(rows) * Pitch + 256U));
  prepare_source_x<H><<<std::max(1U, std::min(528U, (rows + 3U) / 4U)), 128>>>(x, packet, rows);
  check(cudaGetLastError());
  if (rows) quantize_kernel<__nv_bfloat16, 1U, false><<<rows * (H / 128U), 256>>>(x, reference, scales, rows, H);
  check(cudaGetLastError());
  check(cudaDeviceSynchronize());
  std::vector<unsigned char> actual(std::size_t(rows) * Pitch + 256U), expected(rows * H);
  std::vector<float> sf(rows * H / 128U);
  check(cudaMemcpy(actual.data(), packet, actual.size(), cudaMemcpyDeviceToHost));
  if (rows) {
    check(cudaMemcpy(expected.data(), reference, expected.size(), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(sf.data(), scales, sf.size() * 4U, cudaMemcpyDeviceToHost));
  }
  for (unsigned r = 0; r < rows; ++r) {
    if (std::memcmp(actual.data() + r * Pitch, expected.data() + r * H, H) ||
        std::memcmp(actual.data() + r * Pitch + H, sf.data() + r * (H / 128U), H / 128U * 4U))
      throw std::runtime_error("source packet differs from independent row quantization");
  }
  for (std::size_t i = std::size_t(rows) * Pitch; i < actual.size(); ++i)
    if (actual[i] != 0xa5) throw std::runtime_error("source quantizer overwrote tail guard");
  check(cudaFree(x)); check(cudaFree(packet)); check(cudaFree(reference)); check(cudaFree(scales));
}
int main() {
  unsigned cases = 0;
  for (bool special : {false, true}) for (unsigned rows : {0U, 1U, 3U, 65U, 257U}) {
    run<128>(rows, special); run<384>(rows, special); run<2048>(rows, special);
    run<3072>(rows, special); run<7168>(rows, special); cases += 5;
  }
  run<2048>(8193, true); ++cases;
  std::printf("FP8_SOURCE_PASS %u cases\n", cases);
}
