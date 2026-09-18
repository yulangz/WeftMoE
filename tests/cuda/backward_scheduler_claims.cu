#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <cuda_runtime.h>
#include "cuda/persistent_v3/backward/scheduler.cuh"
namespace bwd = weftmoe::cuda::persistent_v3::backward;
constexpr unsigned Owners=4226, Words=(Owners+63)/64, MaxTiles=257;
struct State {
 unsigned long long ready[Words], summary[(Words+63)/64];
 unsigned long long claims[Owners*((MaxTiles+63)/64)];
 unsigned hits[6*MaxTiles], failures, tasks;
};
__host__ __device__ unsigned owner_at(unsigned i) {
 return i==0?0:i==1?63:i==2?64:i==3?4095:i==4?4096:4225;
}
__device__ bwd::TileDomain domain(State* s,unsigned tiles,unsigned granularity) {
 return bwd::TileDomain{s->ready,s->claims,Owners,tiles,bwd::TaskKind::Dx,s->summary,granularity};
}
__global__ void publish(State* s,unsigned tiles,unsigned granularity) {
 if(threadIdx.x<6)bwd::publish_owner(domain(s,tiles,granularity),owner_at(threadIdx.x));
}
__global__ void consume(State* s,unsigned tiles,unsigned granularity) {
 auto d=domain(s,tiles,granularity);bwd::TileTask task;
 while(bwd::try_claim_domain_once(d,task,blockIdx.x*blockDim.x+threadIdx.x)) {
  unsigned slot=0;while(slot<6&&owner_at(slot)!=task.owner)++slot;
  if(slot==6||task.tile_count==0||task.tile_count>granularity||task.tile+task.tile_count>tiles) {atomicAdd(&s->failures,1);continue;}
  for(unsigned i=0;i<task.tile_count;++i)atomicAdd(s->hits+slot*MaxTiles+task.tile+i,1);
  atomicAdd(&s->tasks,1);
 }
}
void ck(cudaError_t e) {if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}
int main(){
 State *s,*h=new State;ck(cudaMalloc(&s,sizeof(State)));
 const unsigned sizes[]{1,2,3,63,64,65,127,128,129,257};
 unsigned cases=0;
 for(unsigned granularity: {1U,2U,4U,64U})for(unsigned tiles:sizes)for(unsigned repeat=0;repeat<3;++repeat){
  ck(cudaMemset(s,0,sizeof(State)));publish<<<1,32>>>(s,tiles,granularity);consume<<<66,128>>>(s,tiles,granularity);ck(cudaGetLastError());ck(cudaDeviceSynchronize());ck(cudaMemcpy(h,s,sizeof(State),cudaMemcpyDeviceToHost));
  if(h->failures)return 3;
  for(unsigned i=0;i<6;++i)for(unsigned j=0;j<MaxTiles;++j)if(h->hits[i*MaxTiles+j]!=(j<tiles?1U:0U)){fprintf(stderr,"coverage failure gran=%u tiles=%u owner=%u tile=%u hits=%u\n",granularity,tiles,owner_at(i),j,h->hits[i*MaxTiles+j]);return 4;}
  for(auto v:h->ready)if(v)return 5;for(auto v:h->summary)if(v)return 6;++cases;
 }
 ck(cudaFree(s));delete h;printf("PASS concurrent contiguous claims: %u cases, 66 CTAs, word/summary boundaries and tails\n",cases);
}
