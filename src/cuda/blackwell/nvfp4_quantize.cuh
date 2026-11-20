// Encode sixteen consecutive BF16 values, with one E4M3 block scale.
#pragma once
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

namespace weft_nvfp4 {
struct alignas(8) Encoded {
    uint64_t pairs;
    uint8_t scale;
};

template <bool ExactScale>
__device__ __forceinline__ Encoded encode_block(const __nv_bfloat162 (&v)[8], float layer, float inv_layer) {
    __nv_bfloat162 largest = __habs2(v[0]);
#pragma unroll
    for (int j = 1; j < 8; ++j)
        largest = __hmax2(largest, __habs2(v[j]));
    float2 mx = __bfloat1622float2(largest);
    const float maximum = fmaxf(mx.x, mx.y);
    // A rounded reciprocal can put an exact E4M3 midpoint on the wrong side.
    __nv_fp8_e4m3 sf(ExactScale ? __fdiv_rn(maximum, 6.f * layer)
                                  : maximum * (inv_layer * (1.f / 6.f)));
    float divisor = static_cast<float>(sf);
    float inv;
    asm("rcp.approx.ftz.f32 %0,%1;" : "=f"(inv) : "f"(fmaxf(divisor, 0x1p-9f)));
    const float2 in = {inv, inv}, neg = {-divisor, -divisor};
    uint64_t packed = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        float2 a = __bfloat1622float2(v[j]);
        float2 q;
        if constexpr (ExactScale) {
            const float denominator = divisor * layer;
            q = denominator > 0.f ? make_float2(__fdiv_rn(a.x, denominator), __fdiv_rn(a.y, denominator))
                                 : make_float2(0.f, 0.f);
        } else {
            a = __fmul2_rn(a, {inv_layer, inv_layer});
            q = __fmul2_rn(a, in);
            q = __ffma2_rn(__ffma2_rn(neg, q, a), in, q);
        }
        packed |= uint64_t(__nv_fp4x2_e2m1(q).__x) << (8 * j);
    }
    return {packed, sf.__x};
}

__device__ __forceinline__ int64_t sf_offset(int row, int group, int k) {
    return (((int64_t(row / 128) * (k / 64) + group / 4) * 128 + row % 32 * 4 + row % 128 / 32) * 4 +
            group % 4);
}

template <bool Row, bool Col, bool ExactScale, int R = 64, int C = 128>
__global__ void quantize_blocks(const __nv_bfloat16 *__restrict__ x, const int *ends, uint8_t *__restrict__ d,
                                uint8_t *__restrict__ sf, uint8_t *__restrict__ dt, uint8_t *__restrict__ st,
                                int experts, int features, int regular, float layer, float inv_layer) {
    constexpr int G = R / 16, blocks = R * C / 16, threads = 256;
    __shared__ __align__(16) __nv_bfloat16 tile[R][C + 8];
    __shared__ uint64_t packed[C][G + 1];
    __shared__ __align__(4) uint8_t scales[C][G];
    __shared__ int start, extent;
    const int r0 = blockIdx.y * R, c0 = blockIdx.x * C;
    if (!threadIdx.x) {
        if (regular) {
            extent = regular;
            start = r0 / regular * regular;
        } else {
            int lo = 0, hi = experts;
            while (lo < hi) {
                int mid = (lo + hi) / 2;
                if (ends[mid] <= r0)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            start = lo ? ends[lo - 1] : 0;
            extent = ends[lo] - start;
        }
    }
#pragma unroll
    for (int j = threadIdx.x; j < blocks; j += threads) {
        int r = j / (C / 16), g = j % (C / 16);
        auto src = reinterpret_cast<const uint4 *>(x + int64_t(r0 + r) * features + c0 + g * 16);
        alignas(16) __nv_bfloat162 v[8];
        reinterpret_cast<uint4 *>(v)[0] = src[0];
        reinterpret_cast<uint4 *>(v)[1] = src[1];
        if constexpr (Col) {
            auto dst = reinterpret_cast<uint4 *>(&tile[r][g * 16]);
            dst[0] = reinterpret_cast<uint4 *>(v)[0];
            dst[1] = reinterpret_cast<uint4 *>(v)[1];
        }
        if constexpr (Row) {
            auto out = encode_block<ExactScale>(v, layer, inv_layer);
            reinterpret_cast<uint64_t *>(d)[int64_t(r0 + r) * (features / 16) + c0 / 16 + g] = out.pairs;
            sf[sf_offset(r0 + r, c0 / 16 + g, features)] = out.scale;
        }
    }
    if constexpr (Col) {
        __syncthreads();
#pragma unroll
        for (int j = threadIdx.x; j < blocks; j += threads) {
            int g = j / C, c = j % C;
            __nv_bfloat162 v[8];
#pragma unroll
            for (int p = 0; p < 8; ++p)
                v[p] = __halves2bfloat162(tile[g * 16 + 2 * p][c], tile[g * 16 + 2 * p + 1][c]);
            auto out = encode_block<ExactScale>(v, layer, inv_layer);
            packed[c][g] = out.pairs;
            scales[c][g] = out.scale;
        }
        __syncthreads();
        auto base = int64_t(start) * features;
#pragma unroll
        for (int j = threadIdx.x; j < blocks; j += threads) {
            int c = j / G, g = j % G;
            reinterpret_cast<uint64_t *>(
                dt + base / 2)[int64_t(c0 + c) * (extent / 16) + (r0 - start) / 16 + g] = packed[c][g];
        }
        for (int j = threadIdx.x; j < C * G / 4; j += threads) {
            int c = j / (G / 4), g = j % (G / 4) * 4;
            auto ptr = st + base / 16 + sf_offset(c0 + c, (r0 - start) / 16 + g, extent);
            *reinterpret_cast<uint32_t *>(ptr) = *reinterpret_cast<uint32_t *>(&scales[c][g]);
        }
    }
}

inline void quantize(uintptr_t x, uintptr_t ends, uintptr_t d, uintptr_t sf, uintptr_t dt, uintptr_t st,
                     int rows, int cols, int experts, int regular, float layer, bool row, uintptr_t stream) {
    int exponent;
    const bool power_of_two = std::frexp(layer, &exponent) == 0.5f;
    auto launch = row ? (power_of_two ? quantize_blocks<true, true, false> : quantize_blocks<true, true, true>)
                      : (power_of_two ? quantize_blocks<false, true, false> : quantize_blocks<false, true, true>);
    launch<<<dim3(cols / 128, rows / 64), 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<__nv_bfloat16 const *>(x), reinterpret_cast<int const *>(ends),
        reinterpret_cast<uint8_t *>(d), reinterpret_cast<uint8_t *>(sf), reinterpret_cast<uint8_t *>(dt),
        reinterpret_cast<uint8_t *>(st), experts, cols, regular, layer, 1.f / layer);
    if (auto error = cudaGetLastError(); error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}
} // namespace weft_nvfp4
