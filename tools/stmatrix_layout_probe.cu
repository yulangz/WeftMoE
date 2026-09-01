#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <vector>

__device__ __forceinline__ void stmatrix_x2_n(std::uint32_t src0,
                                               std::uint32_t src1,
                                               void *destination) {
  asm volatile("stmatrix.sync.aligned.x2.m8n8.shared.b16.trans [%0], {%1, %2};\n"
               :
               : "l"(__cvta_generic_to_shared(destination)), "r"(src0),
                 "r"(src1));
}

__host__ __device__ constexpr std::uint32_t swizzled_index(
    std::uint32_t outer, std::uint32_t inner) {
  const auto atom = inner / 64U;
  const auto in_atom = inner % 64U;
  const auto bank_group = (in_atom / 8U) ^ (outer % 8U);
  return atom * 128U * 64U + outer * 64U + bank_group * 8U + in_atom % 8U;
}

__global__ void probe(std::uint16_t *output) {
  __shared__ alignas(128) std::uint16_t tile[128U * 128U];
  for (std::uint32_t index = threadIdx.x; index < 128U * 128U;
       index += blockDim.x)
    tile[index] = 0xffffU;
  __syncthreads();

  const auto warpgroup = threadIdx.x / 128U;
  const auto warpgroup_thread = threadIdx.x % 128U;
  const auto warp = warpgroup_thread / 32U;
  const auto lane = warpgroup_thread % 32U;
  const auto row0 = warpgroup * 64U + warp * 16U + lane / 4U;
  const auto row1 = row0 + 8U;
#pragma unroll
  for (std::uint32_t index = 0; index < 16U; ++index) {
    const auto column = (lane % 4U) * 2U + index * 8U;
    const auto pack = [](std::uint32_t low, std::uint32_t high) {
      return low | (high << 16U);
    };
    const auto src0 = pack(row0 * 128U + column,
                           row0 * 128U + column + 1U);
    const auto src1 = pack(row1 * 128U + column,
                           row1 * 128U + column + 1U);
    const auto destination_column = index * 8U + lane % 8U;
    const auto destination_row = warpgroup * 64U + warp * 16U +
                                 ((lane / 8U) & 1U) * 8U;
    auto *destination =
        tile + swizzled_index(destination_column, destination_row);
    stmatrix_x2_n(src0, src1, destination);
  }
  __syncthreads();
  for (std::uint32_t index = threadIdx.x; index < 128U * 128U;
       index += blockDim.x)
    output[index] = tile[index];
}

int main() {
  std::uint16_t *device = nullptr;
  if (cudaMalloc(&device, 128U * 128U * sizeof(std::uint16_t)) != cudaSuccess)
    return 1;
  probe<<<1, 256>>>(device);
  if (cudaDeviceSynchronize() != cudaSuccess)
    return 2;
  std::vector<std::uint16_t> actual(128U * 128U);
  cudaMemcpy(actual.data(), device, actual.size() * sizeof(std::uint16_t),
             cudaMemcpyDeviceToHost);
  cudaFree(device);
  std::uint32_t mismatches = 0U;
  for (std::uint32_t row = 0; row < 16U; ++row) {
    for (std::uint32_t column = 0; column < 16U; ++column) {
      const auto value = actual[swizzled_index(column, row)];
      std::printf("%3u,%3u%s", value / 128U, value % 128U,
                  column == 15U ? "\n" : "  ");
    }
  }
  for (std::uint32_t row = 0; row < 128U; ++row) {
    for (std::uint32_t column = 0; column < 128U; ++column) {
      const auto physical = swizzled_index(column, row);
      const auto expected = static_cast<std::uint16_t>(row * 128U + column);
      if (actual[physical] != expected && mismatches++ < 16U)
        std::printf("row=%u column=%u physical=%u expected=%u actual=%u\n",
                    row, column, physical, expected, actual[physical]);
    }
  }
  std::printf("mismatches=%u\n", mismatches);
  return mismatches == 0U ? 0 : 3;
}
