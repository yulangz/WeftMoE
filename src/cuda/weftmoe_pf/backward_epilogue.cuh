#pragma once

namespace weftmoe_pf::backward_stream {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED) && __has_include(<nccl_device.h>)
template <unsigned N, bool Network>
__device__ __forceinline__ void store_swiglu(
    const ForwardKernelArgs &f, const BackwardKernelArgs &a,
    const Task &t, const unsigned *packed) {
  const auto lane = threadIdx.x % 32U;
  const auto row0 = threadIdx.x / 32U * 16U + lane / 4U;
  const auto *mapping = peer_at<const unsigned>(a.local_base, a.compact_to_padded_offset);
#pragma unroll
  for (unsigned item = 0U; item < 2U; ++item) {
    const auto row = row0 + item * 8U;
    const bool valid = row < t.rows;
    const auto saved = valid ? (Network ? directory(a)[t.block].compact_row + row
                                       : kCanonicalLocal ? compact_local_row(f, a, t) + row
                                                         : f.checkpoint.rows[t.block * kM + row]) : 0U;
    const auto padded = Network || kCanonicalLocal || !valid ? t.block * kM + row : mapping[saved];
    const auto gate = valid ? a.saved_gates[saved] : 0.0F;
    float partial = 0.0F;
#pragma unroll
    for (unsigned index = 0U; index < N / 8U; index += 4U) {
      const auto col = t.column * N + index * 8U + lane % 4U * 8U;
      uint4 da{}, db{}, u{};
      if (valid) {
        unsigned bits[4];
#pragma unroll
        for (unsigned j = 0U; j < 4U; ++j) {
          bits[j] = packed[(index + j) * 2U + item];
        }
        const auto du = forward_compute_detail::static_v4_transpose_vector(
            make_uint4(bits[0], bits[1], bits[2], bits[3]));
        const auto offset = std::uint64_t(saved) * a.intermediate + col;
        const auto g = *reinterpret_cast<const uint4 *>(a.saved_ab + offset);
        const auto v = *reinterpret_cast<const uint4 *>(
            a.saved_ab + a.saved_rows * a.intermediate + offset);
        const unsigned gs[4] = {g.x, g.y, g.z, g.w};
        const unsigned vs[4] = {v.x, v.y, v.z, v.w};
        const unsigned ds[4] = {du.x, du.y, du.z, du.w};
        unsigned as[4], bs[4], us[4];
#pragma unroll
        for (unsigned j = 0U; j < 4U; ++j) {
          const auto result = backward_swiglu_pair(gs[j], vs[j], ds[j], gate, partial);
          as[j] = result.da;
          bs[j] = result.db;
          us[j] = result.u;
        }
        da = make_uint4(as[0], as[1], as[2], as[3]);
        db = make_uint4(bs[0], bs[1], bs[2], bs[3]);
        u = make_uint4(us[0], us[1], us[2], us[3]);
      }
      auto *dab = a.dab + std::uint64_t(padded) * 2U * a.intermediate + col;
      *reinterpret_cast<uint4 *>(dab) = da;
      *reinterpret_cast<uint4 *>(dab + a.intermediate) = db;
      *reinterpret_cast<uint4 *>(a.recomputed_u +
                                 std::uint64_t(padded) * a.intermediate + col) = u;
    }
    partial += __shfl_down_sync(0xffffffffU, partial, 2, 4);
    partial += __shfl_down_sync(0xffffffffU, partial, 1, 4);
    if (!(lane % 4U))
      a.dgate_rows[std::uint64_t(padded) * kDuTiles + t.column] = partial;
  }
}
#endif
} // namespace weftmoe_pf::backward_stream
