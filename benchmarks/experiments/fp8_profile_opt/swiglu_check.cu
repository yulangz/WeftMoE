#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include <stdexcept>
__device__ std::uint16_t float_to_bf16_bits(float value) {
  auto bits = __float_as_uint(value); bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}
__device__ unsigned float2_to_bf16x2_bits(float lo,float hi) {
  const auto p=static_cast<__nv_bfloat162_raw>(__float22bfloat162_rn(make_float2(lo,hi)));
  return unsigned(p.x)|(unsigned(p.y)<<16U);
}
#include "cuda/backward/swiglu.cuh"
__global__ void compare(unsigned *errors) {
  const unsigned index=blockIdx.x*blockDim.x+threadIdx.x;
  unsigned a=index*1664525U+1013904223U,b=a*1664525U+1013904223U,d=b*1664525U+1013904223U;
  if (index<65536U) a=index|((65535U-index)<<16U);
  const float gate=float(int(index%257U)-128)*0.125F;
  float p0=0,p1=0;
  const auto x=backward_swiglu_pair<false>(a,b,d,gate,p0);
  const auto y=backward_swiglu_pair<true>(a,b,d,gate,p1);
  if(x.da!=y.da||x.db!=y.db||x.u!=y.u||__float_as_uint(p0)!=__float_as_uint(p1)) atomicAdd(errors,1U);
}
void ck(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
int main(){unsigned *p,n;ck(cudaMalloc(&p,4));ck(cudaMemset(p,0,4));compare<<<8192,256>>>(p);ck(cudaGetLastError());ck(cudaMemcpy(&n,p,4,cudaMemcpyDeviceToHost));ck(cudaFree(p));std::cout<<"swiglu pairs=2097152 exact_mismatches="<<n<<'\n';return n?1:0;}
