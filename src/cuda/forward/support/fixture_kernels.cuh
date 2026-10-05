__device__ std::uint32_t forward_fixture_mix32(std::uint32_t value) {
  value ^= value >> 16U;
  value *= 0x7feb352dU;
  value ^= value >> 15U;
  value *= 0x846ca68bU;
  value ^= value >> 16U;
  return value;
}

__device__ std::uint16_t forward_fixture_weight_bits(
    std::uint32_t expert, std::uint32_t tensor_kind, std::uint64_t row,
    std::uint64_t column) {
  auto key = forward_fixture_mix32(expert ^ 0x9e3779b9U);
  key ^= forward_fixture_mix32(tensor_kind ^ 0x85ebca6bU);
  key ^= forward_fixture_mix32(static_cast<std::uint32_t>(row) ^
                               0xc2b2ae35U);
  key ^= forward_fixture_mix32(static_cast<std::uint32_t>(column) ^
                               0x27d4eb2fU);
  const auto signed_value =
      static_cast<int>(forward_fixture_mix32(key) & 1U) * 2 - 1;
  return float_to_bf16_bits(static_cast<float>(signed_value) *
                            (1.0F / 256.0F));
}

constexpr std::uint64_t kWeightHashChunkElements = 1ULL << 22U;
constexpr std::uint64_t kWeightHashSeedLow = 0x243f6a8885a308d3ULL;
constexpr std::uint64_t kWeightHashSeedHigh = 0x13198a2e03707344ULL;
constexpr std::uint64_t kWeightHashMultiplierLow = 0x9e3779b185ebca87ULL;
constexpr std::uint64_t kWeightHashMultiplierHigh = 0xc2b2ae3d27d4eb4fULL;

struct WeightHashPair {
  std::uint64_t low;
  std::uint64_t high;
};

__device__ std::uint32_t weight_hash_mix(std::uint32_t value) {
  return forward_fixture_mix32(value);
}

__global__ void hash_forward_weight_chunks_kernel(
    const transport::BFloat16 *values, std::uint64_t elements,
    WeightHashPair *chunks) {
  __shared__ std::uint64_t low[256];
  __shared__ std::uint64_t high[256];
  const auto begin = static_cast<std::uint64_t>(blockIdx.x) *
                     kWeightHashChunkElements;
  const auto end = min(elements, begin + kWeightHashChunkElements);
  std::uint64_t local_low = 0;
  std::uint64_t local_high = 0;
  for (auto index = begin + threadIdx.x; index < end; index += blockDim.x) {
    const auto bits = static_cast<std::uint32_t>(values[index].bits);
    const auto logical_index = static_cast<std::uint32_t>(index);
    const auto mixed_low = weight_hash_mix(
        bits ^ (logical_index * 0x9e3779b1U) ^ 0xa5a5f00dU);
    const auto mixed_high = weight_hash_mix(
        (bits << 16U) ^ (logical_index * 0x85ebca77U) ^ 0x5a5a0ff1U);
    local_low += static_cast<std::uint64_t>(mixed_low) *
                 kWeightHashMultiplierLow;
    local_high += static_cast<std::uint64_t>(mixed_high) *
                  kWeightHashMultiplierHigh;
  }
  low[threadIdx.x] = local_low;
  high[threadIdx.x] = local_high;
  __syncthreads();
  for (std::uint32_t offset = blockDim.x / 2U; offset != 0U; offset /= 2U) {
    if (threadIdx.x < offset) {
      low[threadIdx.x] += low[threadIdx.x + offset];
      high[threadIdx.x] += high[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
    chunks[blockIdx.x] = {low[0], high[0]};
}

__global__ void finalize_forward_weight_hash_kernel(
    const WeightHashPair *chunks, std::uint64_t chunk_count,
    std::uint64_t elements, WeightHashPair *result) {
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;
  WeightHashPair value{
      kWeightHashSeedLow ^ (elements * 0x100000001b3ULL),
      kWeightHashSeedHigh ^ (elements * 0x9e3779b97f4a7c15ULL)};
  for (std::uint64_t chunk = 0; chunk < chunk_count; ++chunk) {
    value.low += chunks[chunk].low;
    value.high += chunks[chunk].high;
  }
  *result = value;
}

__global__ void initialize_forward_weights_kernel(
    transport::BFloat16 *w1, transport::BFloat16 *w1_interleaved,
    std::uint64_t w1_elements,
    transport::BFloat16 *w2, std::uint64_t w2_elements,
    std::uint32_t global_expert_begin, std::uint64_t hidden,
    std::uint64_t intermediate) {
  const auto stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < w1_elements; index += stride) {
    const auto expert_stride = 2ULL * intermediate * hidden;
    const auto expert = global_expert_begin +
                        static_cast<std::uint32_t>(index / expert_stride);
    const auto within = index % expert_stride;
    const auto row = within / hidden;
    const auto column = within % hidden;
    const auto bits = forward_fixture_weight_bits(expert, 1U, row, column);
    w1[index].bits = bits;
    const auto projection = row / intermediate;
    const auto channel = row % intermediate;
    const auto packed_row =
        (channel / 8ULL) * 16ULL + projection * 8ULL + channel % 8ULL;
    w1_interleaved[(index / expert_stride) * expert_stride +
                   packed_row * hidden + column]
        .bits = bits;
  }
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < w2_elements; index += stride) {
    const auto expert_stride = hidden * intermediate;
    const auto expert = global_expert_begin +
                        static_cast<std::uint32_t>(index / expert_stride);
    const auto within = index % expert_stride;
    w2[index].bits = forward_fixture_weight_bits(
        expert, 2U, within / intermediate, within % intermediate);
  }
}

__global__ void interleave_forward_w1_kernel(
    const transport::BFloat16 *source, transport::BFloat16 *destination,
    std::uint64_t elements, std::uint64_t hidden,
    std::uint64_t intermediate) {
  // TMA-compatible weight rows are 16-byte aligned. A CTA owns complete rows,
  // so the gate/up permutation is computed once per row, not per BF16 value.
  const auto vectors = hidden / 8U;
  const auto rows = elements / hidden;
  const auto *input = reinterpret_cast<const uint4 *>(source);
  auto *output = reinterpret_cast<uint4 *>(destination);
  for (std::uint64_t index = blockIdx.x; index < rows; index += gridDim.x) {
    const auto expert = index / (2ULL * intermediate);
    const auto row = index % (2ULL * intermediate);
    const auto projection = row / intermediate;
    const auto channel = row % intermediate;
    const auto packed_row =
        (channel / 8ULL) * 16ULL + projection * 8ULL + channel % 8ULL;
    const auto target = (expert * 2ULL * intermediate + packed_row) * vectors;
    for (auto column = std::uint64_t(threadIdx.x); column < vectors;
         column += blockDim.x)
      output[target + column] = input[index * vectors + column];
  }
}

#include "cuda/forward/kernel_contract.cuh"
