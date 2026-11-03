#pragma once

// Each warp owns 16 complete token rows and four channels per lane. This
// keeps BF16 input loads and FP8 row stores contiguous, including short tails.
__device__ __forceinline__ void fp8_swiglu_column_store(
    const unsigned (&values)[32], const float *scales,
    std::uint8_t *output, float *output_scales) {
  const unsigned lane = threadIdx.x % 32U, warp = threadIdx.x / 32U;
  const unsigned column = lane * 4U;
  const auto s = *reinterpret_cast<const float4 *>(scales + column);
  const auto inverse = [](float v) {
    return isfinite(v) ? __uint_as_float(0x7f000000U - __float_as_uint(v)) : v;
  };
  const float ix = inverse(s.x), iy = inverse(s.y), iz = inverse(s.z), iw = inverse(s.w);
  if (!warp)
    *reinterpret_cast<float4 *>(output_scales + column) = s;
  unsigned a[4], b[4], c[4], d[4];
#pragma unroll
  for (unsigned group = 0; group < 4U; ++group) {
    unsigned q0[4], q1[4];
#pragma unroll
    for (unsigned j = 0; j < 4U; ++j) {
      const auto v0 = bf16x2_bits_to_float2(values[group * 8U + j * 2U]);
      const auto v1 = bf16x2_bits_to_float2(values[group * 8U + j * 2U + 1U]);
      q0[j] = __nv_fp8x2_e4m3(make_float2(v0.x * ix, v0.y * iy)).__x;
      q1[j] = __nv_fp8x2_e4m3(make_float2(v1.x * iz, v1.y * iw)).__x;
    }
    // Four interleaved row pairs become two contiguous column words.
    const auto ab01 = q0[0] | (q0[1] << 16U), ab23 = q0[2] | (q0[3] << 16U);
    const auto cd01 = q1[0] | (q1[1] << 16U), cd23 = q1[2] | (q1[3] << 16U);
    a[group] = __byte_perm(ab01, ab23, 0x6420U);
    b[group] = __byte_perm(ab01, ab23, 0x7531U);
    c[group] = __byte_perm(cd01, cd23, 0x6420U);
    d[group] = __byte_perm(cd01, cd23, 0x7531U);
  }
  *reinterpret_cast<uint4 *>(output + fp8::tile_offset(column, warp * 16U)) = make_uint4(a[0], a[1], a[2], a[3]);
  *reinterpret_cast<uint4 *>(output + fp8::tile_offset(column + 1U, warp * 16U)) = make_uint4(b[0], b[1], b[2], b[3]);
  *reinterpret_cast<uint4 *>(output + fp8::tile_offset(column + 2U, warp * 16U)) = make_uint4(c[0], c[1], c[2], c[3]);
  *reinterpret_cast<uint4 *>(output + fp8::tile_offset(column + 3U, warp * 16U)) = make_uint4(d[0], d[1], d[2], d[3]);
}

// Included inside backward_stream. One CTA owns M128 x I128. Feature tiles
// publish independently; the last arrival combines dGate and releases dAB.
template <bool Network>
__device__ __forceinline__ void fp8_swiglu_columns(
    const ForwardKernelArgs &f, const BackwardKernelArgs &a,
    const Task &t, float *scratch) {
  const unsigned tid = threadIdx.x, lane = tid % 32U, warp = tid / 32U;
  const unsigned channel = lane * 4U, column = t.column * 128U;
  const auto *mapping = peer_at<const unsigned>(a.local_base, a.compact_to_padded_offset);
  const unsigned tiles = unsigned(a.intermediate / 128U);
  // BF16 dAB is not materialized on this path. Its allocation holds only
  // feature-tile route-gradient partials until the last arrival publishes them.
  auto *partials = reinterpret_cast<float *>(a.dab) + std::uint64_t(a.fp8_du_ring ? t.row : t.block * kM) * tiles;
  unsigned da[32], db[32], z[32];
  unsigned ma[2] = {}, mb[2] = {}, mz[2] = {};
#pragma unroll
  for (unsigned i = 0; i < 16U; ++i) {
    const auto row = warp * 16U + i;
    BackwardSwigluPair v0{}, v1{};
    float partial = 0.0F;
    if (row < t.rows) {
      const auto saved = Network ? directory(a)[t.block].compact_row + row
                                 : f.checkpoint.rows[t.block * kM + row];
      const auto offset = saved * a.intermediate + column + channel;
      uint2 av, bv;
      if (a.fp8_ab_data) {
        const auto sf_offset = saved * (a.intermediate / 128U) + column / 128U;
        const auto as = a.fp8_ab_scales[sf_offset];
        const auto bs = a.fp8_ab_scales[sf_offset + a.saved_rows * (a.intermediate / 128U)];
        const auto decode = [](unsigned q, float scale) {
          __nv_fp8x2_e4m3 lo, hi;
          lo.__x = q & 65535U; hi.__x = q >> 16U;
          const auto x = static_cast<float2>(lo), y = static_cast<float2>(hi);
          return make_uint2(float2_to_bf16x2_bits(x.x * scale, x.y * scale),
                            float2_to_bf16x2_bits(y.x * scale, y.y * scale));
        };
        av = decode(*reinterpret_cast<const unsigned *>(a.fp8_ab_data + offset), as);
        bv = decode(*reinterpret_cast<const unsigned *>(a.fp8_ab_data + offset + a.saved_rows * a.intermediate), bs);
      } else {
        av = *reinterpret_cast<const uint2 *>(a.saved_ab + offset);
        bv = *reinterpret_cast<const uint2 *>(a.saved_ab + offset + a.saved_rows * a.intermediate);
      }
      const auto uv = *reinterpret_cast<const uint2 *>(a.recomputed_u +
          std::uint64_t(a.fp8_du_ring ? t.row + row : mapping[saved]) * a.intermediate + column + channel);
      const auto gate = a.saved_gates[saved];
      v0 = backward_swiglu_pair(av.x, bv.x, uv.x, gate, partial);
      v1 = backward_swiglu_pair(av.y, bv.y, uv.y, gate, partial);
    }
    da[2U * i] = v0.da; da[2U * i + 1U] = v1.da;
    db[2U * i] = v0.db; db[2U * i + 1U] = v1.db;
    z[2U * i] = v0.u; z[2U * i + 1U] = v1.u;
    ma[0] = __vmaxu2(ma[0], v0.da & 0x7fff7fffU);
    ma[1] = __vmaxu2(ma[1], v1.da & 0x7fff7fffU);
    mb[0] = __vmaxu2(mb[0], v0.db & 0x7fff7fffU);
    mb[1] = __vmaxu2(mb[1], v1.db & 0x7fff7fffU);
    mz[0] = __vmaxu2(mz[0], v0.u & 0x7fff7fffU);
    mz[1] = __vmaxu2(mz[1], v1.u & 0x7fff7fffU);
    const auto am = __vmaxu2(v0.da & 0x7fff7fffU, v1.da & 0x7fff7fffU);
    const auto bm = __vmaxu2(v0.db & 0x7fff7fffU, v1.db & 0x7fff7fffU);
    // Each warp owns a complete row. Hardware integer reductions replace
    // five shuffle/max rounds for each of the two BF16 magnitude planes.
    const unsigned maximum =
        __reduce_max_sync(0xffffffffU, max(am & 0xffffU, am >> 16U)) |
        (__reduce_max_sync(0xffffffffU, max(bm & 0xffffU, bm >> 16U)) << 16U);
#pragma unroll
    for (unsigned delta = 16U; delta; delta >>= 1U) {
      partial += __shfl_down_sync(0xffffffffU, partial, delta);
    }
    const float sa = fp8::block_scale((maximum & 0xffffU) >= 0x7f80U
        ? __uint_as_float(0x7f800000U) : __uint_as_float(maximum << 16U));
    const float sb = fp8::block_scale((maximum >> 16U) >= 0x7f80U
        ? __uint_as_float(0x7f800000U) : __uint_as_float(maximum & 0xffff0000U));
    const float ia = isfinite(sa) ? __uint_as_float(0x7f000000U - __float_as_uint(sa)) : sa;
    const float ib = isfinite(sb) ? __uint_as_float(0x7f000000U - __float_as_uint(sb)) : sb;
    const auto cast_row = [](unsigned x, float inv) {
      const auto v = bf16x2_bits_to_float2(x);
      return unsigned(__nv_fp8x2_e4m3(make_float2(v.x * inv, v.y * inv)).__x);
    };
    constexpr bool Packed = WEFTMOE_FP8_PACKED_RING && local::kN == 256U;
    auto *buffer = reinterpret_cast<std::uint8_t *>(a.dx_rows);
    auto *out = Packed ? fp8::ring_row(buffer, t.row + row,
        unsigned(2U * a.intermediate), unsigned(a.fp8_activation_pitch))
        : buffer + std::uint64_t(t.row + row) * a.activation_stride * 2U;
    *reinterpret_cast<unsigned *>(out + column + channel) = cast_row(v0.da, ia) | (cast_row(v1.da, ia) << 16U);
    *reinterpret_cast<unsigned *>(out + a.intermediate + column + channel) = cast_row(v0.db, ib) | (cast_row(v1.db, ib) << 16U);
    if (!lane) {
      auto *row_scales = Packed ? fp8::ring_scales(buffer, t.row + row,
          unsigned(2U * a.intermediate), unsigned(a.fp8_activation_pitch))
          : reinterpret_cast<float *>(out + 2U * a.intermediate);
      row_scales[column / 128U * (Packed ? 128U : 1U)] = sa;
      row_scales[(a.intermediate + column) / 128U * (Packed ? 128U : 1U)] = sb;
      partials[t.column * kM + row] = partial;
    }
  }
  auto *maxima = reinterpret_cast<unsigned *>(scratch);
#pragma unroll
  for (unsigned pair = 0; pair < 2U; ++pair) {
    const unsigned index = warp * 64U + lane * 2U + pair;
    maxima[index] = ma[pair];
    maxima[512U + index] = mb[pair];
    maxima[1024U + index] = mz[pair];
  }
  auto *sf = scratch + 1536U;
  fp8::MathSync::sync();
  if (tid < 64U) {
#pragma unroll
    for (unsigned plane = 0; plane < 3U; ++plane) {
      unsigned maximum = 0U;
#pragma unroll
      for (unsigned w = 0; w < 8U; ++w)
        maximum = __vmaxu2(maximum, maxima[plane * 512U + w * 64U + tid]);
      const auto x = (maximum & 0xffffU) >= 0x7f80U ? nanf("") : fp8::block_scale(__uint_as_float(maximum << 16U));
      const auto y = (maximum >> 16U) >= 0x7f80U ? nanf("") : fp8::block_scale(__uint_as_float(maximum & 0xffff0000U));
      *reinterpret_cast<float2 *>(sf + plane * 128U + tid * 2U) = make_float2(x, y);
    }
  }
  fp8::MathSync::sync();
  const auto cast = [&](const unsigned (&values)[32], const float *scales,
                        std::uint64_t plane, unsigned channels, unsigned col) {
    const auto offset = plane * a.fp8_wgrad_rows + std::uint64_t(t.block) * kM * channels + col * kM;
    fp8_swiglu_column_store(values, scales, a.fp8_wgrad_data + offset, a.fp8_wgrad_scales + offset / 128U);
  };
  cast(z, sf + 256U, a.hidden, unsigned(a.intermediate), column);
  cast(da, sf, a.hidden + a.intermediate, unsigned(2U * a.intermediate), column);
  cast(db, sf + 128U, a.hidden + a.intermediate, unsigned(2U * a.intermediate), unsigned(a.intermediate) + column);
  __threadfence();
  fp8::MathSync::sync();
  auto &slot = f.ring.slots[t.row / kM];
  auto *last = reinterpret_cast<unsigned *>(scratch);
  if (!tid) {
    ::cuda::atomic_ref<std::uint64_t, ::cuda::thread_scope_device> ready(slot.fp8_swiglu_ready);
    *last = ready.fetch_add(1U, ::cuda::memory_order_acq_rel) + 1U ==
            std::uint64_t(t.generation) * tiles;
  }
  fp8::MathSync::sync();
  const bool publish = *last;
  fp8::MathSync::sync();
  if (!publish)
    return;
  if (tid < t.rows) {
    float total = 0.0F;
    for (unsigned tile = 0; tile < tiles; ++tile)
      total += partials[tile * kM + tid];
    if constexpr (Network) {
      a.dgate_rows[t.block * kM + tid] = total;
      *gradient_output(f, a, t, tid) = total;
    } else {
      const auto saved = f.checkpoint.rows[t.block * kM + tid];
      a.dgate_rows[mapping[saved]] = total;
      const auto identity = f.ring.rows[t.block * kM + tid];
      a.peer_payloads[identity.source].dgate[identity.route] = total;
    }
  }
  if constexpr (Network) __threadfence_system();
  else __threadfence();
  fp8::MathSync::sync();
  if (!tid)
    ring::arrive(slot.activation_ready, 2U);
}
