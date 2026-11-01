#pragma once

// Descriptor/address and array adapters derived from DeepGEMM (MIT).
// Instructions are provided by CuTe/CUTLASS.
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm90_desc.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cutlass/arch/barrier.h>

// This primitive header can precede either phase's tuning header in a TU.
// Read the selected layout before instantiating its instruction adapters.
#if defined(WEFTMOE_FORWARD_USE_GENERATED_TUNING)
#include <weftmoe_forward_tuning_generated.hpp>
#elif defined(WEFTMOE_BACKWARD_USE_GENERATED_TUNING)
#include <weftmoe_backward_tuning_generated.hpp>
#endif

#ifndef WEFTMOE_WGMMA_N256_ATOM_N
#define WEFTMOE_WGMMA_N256_ATOM_N 256
#endif

namespace weftmoe::cuda::hopper {
using Major = cute::SM90::GMMA::Major;

static_assert(WEFTMOE_WGMMA_N256_ATOM_N == 128 ||
              WEFTMOE_WGMMA_N256_ATOM_N == 256);

template <int Columns, Major MajorA = Major::K, Major MajorB = Major::K,
          uint32_t StageK = 64, uint32_t SwizzleBytes = 128,
          int AtomN = WEFTMOE_WGMMA_N256_ATOM_N>
struct BF16Mma {
  static_assert(AtomN == 128 || AtomN == 256);
  static constexpr int M = 64, N = Columns, K = 16;
  static constexpr int kNumAccum = Columns / 2;
  static constexpr auto operation() {
    using namespace cute::SM90::GMMA;
    static_assert(Columns == 64 || Columns == 128 || Columns == 256);
    if constexpr (Columns == 64)
      return MMA_64x64x16_F32BF16BF16_SS<MajorA, MajorB>{};
    else if constexpr (Columns == 128)
      return MMA_64x128x16_F32BF16BF16_SS<MajorA, MajorB>{};
    else
      return MMA_64x256x16_F32BF16BF16_SS<MajorA, MajorB>{};
  }
  template <size_t... I>
  CUTLASS_DEVICE static void apply(uint64_t const& a, uint64_t const& b,
                                  float* d, bool accumulate,
                                  cute::index_sequence<I...>) {
    decltype(operation())::fma(a, b, d[I]...,
        accumulate ? cute::SM90::GMMA::ScaleOut::One
                   : cute::SM90::GMMA::ScaleOut::Zero);
  }
  CUTLASS_DEVICE static void wgmma(uint64_t const& a, uint64_t const& b,
                                  float* d, bool accumulate) {
    if constexpr (Columns == 256 && AtomN == 128) {
      // CUDA 12.9 ptxas can reject a native N256 instruction at the static
      // register limit of a 512/640-thread CTA despite setmaxnreg. Keep the
      // N256 tile, all accumulators, and each output's K order; only split
      // the instruction. Both supported B-major layouts place N128 at
      // 128 * K64 BF16 elements with the existing 128-byte swizzle.
      static_assert(StageK == 64 && SwizzleBytes == 128,
                    "split N256 WGMMA requires K64 / 128-byte swizzle");
      using Half = BF16Mma<128, MajorA, MajorB, StageK, SwizzleBytes>;
      Half::wgmma(a, b, d, accumulate);
      constexpr uint32_t kSecondBDescriptorOffset =
          128U * StageK * sizeof(uint16_t) / 16U;
      const uint64_t second_b = b + kSecondBDescriptorOffset;
      Half::wgmma(a, second_b, d + Half::kNumAccum, accumulate);
    } else {
      apply(a, b, d, accumulate, cute::make_index_sequence<kNumAccum>{});
    }
  }
};

template <int N, Major A = Major::K, Major B = Major::K,
          uint32_t StageK = 64, uint32_t SwizzleBytes = 128>
struct BF16MMASelector {
  using type = BF16Mma<N, A, B, StageK, SwizzleBytes>;
};

// Hopper FP8 uses K-major shared operands and K32 instruction atoms.
// Scaling is applied by the caller after each K128 accumulation interval.
template <int Columns> struct FP8Mma {
  static constexpr int M = 64, N = Columns, K = 32;
  static constexpr int kNumAccum = Columns / 2;
  static constexpr auto operation() {
    using namespace cute::SM90::GMMA;
    static_assert(Columns == 64 || Columns == 128 || Columns == 256);
    if constexpr (Columns == 64)
      return MMA_64x64x32_F32E4M3E4M3_SS_TN<>{};
    else if constexpr (Columns == 128)
      return MMA_64x128x32_F32E4M3E4M3_SS_TN<>{};
    else
      return MMA_64x256x32_F32E4M3E4M3_SS_TN<>{};
  }
  template <size_t... I>
  CUTLASS_DEVICE static void apply(uint64_t const &a, uint64_t const &b,
                                  float *d, bool accumulate,
                                  cute::index_sequence<I...>) {
    decltype(operation())::fma(a, b, d[I]...,
        accumulate ? cute::SM90::GMMA::ScaleOut::One
                   : cute::SM90::GMMA::ScaleOut::Zero);
  }
  CUTLASS_DEVICE static void wgmma(uint64_t const &a, uint64_t const &b,
                                  float *d, bool accumulate) {
    apply(a, b, d, accumulate, cute::make_index_sequence<kNumAccum>{});
  }
};

template <class Pointer>
CUTLASS_DEVICE cute::GmmaDescriptor make_smem_desc(
    Pointer pointer, const int& layout, const uint32_t& leading = 0,
    const uint32_t& stride = 1024) {
  cute::GmmaDescriptor desc;
  desc.bitfield.start_address_ = cute::cast_smem_ptr_to_uint(pointer) >> 4;
  desc.bitfield.layout_type_ = layout;
  desc.bitfield.leading_byte_offset_ = leading >> 4;
  desc.bitfield.stride_byte_offset_ = stride >> 4;
  desc.bitfield.base_offset_ = 0;
  return desc;
}

// Existing kernels use 128-byte swizzles with 64 BF16 elements per atom.
// Keep descriptor-low arithmetic explicit to preserve the pipelined addressing.
template <Major Mode, uint32_t MN, uint32_t K, uint32_t Swizzle, class T>
CUTLASS_DEVICE uint32_t advance_gmma_desc_lo(
    const uint32_t& base, const uint32_t& mn, const uint32_t& k,
    const uint32_t& offset = 0) {
  static_assert(Swizzle == 128);
  constexpr auto atom = Swizzle / sizeof(T);
  // A wider K-major stage consists of contiguous [MN, atom] TMA slabs.
  const auto elements = Mode == Major::K
      ? (k / atom) * MN * atom + mn * atom + k % atom
      : mn * K + k * atom;
  return base + (((offset + elements) * static_cast<uint32_t>(sizeof(T))) >> 4U);
}

template <Major Mode, uint32_t MN, uint32_t K, uint32_t Swizzle, class T>
CUTLASS_DEVICE cute::GmmaDescriptor make_gmma_desc(
    T* base, uint32_t mn, uint32_t k) {
  static_assert(Swizzle == 128);
  constexpr auto atom = Swizzle / sizeof(T);
  constexpr auto layout = static_cast<int>(cute::SM90::GMMA::LayoutType::B128);
  if constexpr (Mode == Major::K) {
    static_assert(K % atom == 0);
    return make_smem_desc(base + (k / atom) * MN * atom + mn * atom + k % atom,
                          layout, 0, 8 * Swizzle);
  } else {
    return make_smem_desc(base + mn * K + k * atom, layout,
                          K * atom * sizeof(T), 8 * atom * sizeof(T));
  }
}

template <uint32_t Inner, uint32_t Outer, uint32_t Swizzle, class T>
CUTLASS_DEVICE void tma_load(
    void const* map, cutlass::arch::ClusterTransactionBarrier* barrier,
    T* shared, const uint32_t& inner, const uint32_t& outer,
    const uint32_t& multicast = 1,
    uint64_t hint = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL)) {
  static_assert(Swizzle == 128);
  constexpr auto atom = Swizzle / sizeof(T);
  if (multicast == 1) {
#pragma unroll
    for (uint32_t i = 0; i < Inner / atom; ++i)
      cute::SM90_TMA_LOAD_2D::copy(map, reinterpret_cast<uint64_t*>(barrier),
          hint, shared + i * Outer * atom, inner + i * atom, outer);
  } else if (cute::block_rank_in_cluster() == 0) {
#pragma unroll
    for (uint32_t i = 0; i < Inner / atom; ++i)
      cute::SM90_TMA_LOAD_MULTICAST_2D::copy(map,
          reinterpret_cast<uint64_t*>(barrier), (1 << multicast) - 1, hint,
          shared + i * Outer * atom, inner + i * atom, outer);
  }
}

template <class T>
struct StoreMatrixX2 {
  CUTLASS_DEVICE static void copy(T a, T b, void* shared) {
    static_assert(sizeof(T) == sizeof(uint32_t));
    cute::SM90_U32x2_STSM_N::copy(
        *reinterpret_cast<uint32_t*>(&a), *reinterpret_cast<uint32_t*>(&b),
        *reinterpret_cast<cute::uint128_t*>(shared));
  }
};
} // namespace weftmoe::cuda::hopper
