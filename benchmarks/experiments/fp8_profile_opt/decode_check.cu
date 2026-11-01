#include "cuda/fp8/column.cuh"
#include <iostream>
#include <stdexcept>
__global__ void check_decode(unsigned long long *first, unsigned *errors) {
  constexpr unsigned scales = 260U;
  const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= scales * 65536U) return;
  const unsigned q = index & 65535U, s = index >> 16U;
  const float scale = s < 254U ? __uint_as_float((s + 1U) << 23U)
    : s == 254U ? 0.0F : s == 255U ? -1.0F : s == 256U ? 1.25F
    : s == 257U ? __uint_as_float(1U) : s == 258U ? __uint_as_float(0x7f800000U) : __uint_as_float(0x7fc00000U);
  __nv_fp8x2_e4m3 x; x.__x = q;
  const auto f = static_cast<float2>(x);
  const auto pair = __float22bfloat162_rn(make_float2(f.x * scale, f.y * scale));
  const unsigned expected = reinterpret_cast<const unsigned &>(pair);
  const unsigned actual = weftmoe::cuda::fp8::decode_scaled_e4m3_pair(q, scale);
  if (expected != actual) { atomicAdd(errors, 1U); atomicMin(first, static_cast<unsigned long long>(index)); }
}
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
int main() {
  unsigned *errors, count; unsigned long long *first, f = ~0ULL;
  ck(cudaMalloc(&errors, sizeof(count))); ck(cudaMalloc(&first, sizeof(f)));
  ck(cudaMemset(errors, 0, sizeof(count))); ck(cudaMemcpy(first, &f, sizeof(f), cudaMemcpyHostToDevice));
  check_decode<<<260U*256U,256>>>(first,errors); ck(cudaGetLastError());
  ck(cudaMemcpy(&count,errors,sizeof(count),cudaMemcpyDeviceToHost)); ck(cudaMemcpy(&f,first,sizeof(f),cudaMemcpyDeviceToHost));
  std::cout << "decode pairs=" << 260U*65536U << " errors=" << count << " first=" << f << '\n';
  ck(cudaFree(errors)); ck(cudaFree(first)); return count ? 1 : 0;
}
