// Verify the mapped stream-completion primitive on the target CUDA platform.
// nvcc -std=c++17 -arch=sm_90a tools/stream_completion_probe.cu -lcuda -o /tmp/stream_completion_probe
#include <cuda.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {
void check(cudaError_t error) {
  if (error != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(error));
    std::exit(1);
  }
}
__global__ void fill(unsigned *data, unsigned generation) {
  data[threadIdx.x] = generation + threadIdx.x;
}
} // namespace

int main() {
  int devices = 0;
  check(cudaGetDeviceCount(&devices));
  check(cudaSetDevice(0));
  unsigned *host = nullptr, *mapped = nullptr, *data = nullptr;
  check(cudaHostAlloc(&host, 257 * sizeof(unsigned),
                      cudaHostAllocMapped | cudaHostAllocPortable));
  check(cudaHostGetDevicePointer(&mapped, host, 0));
  check(cudaMalloc(&data, 256 * sizeof(unsigned)));
  cudaStream_t stream = nullptr;
  cudaEvent_t event = nullptr;
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  for (unsigned generation = 1; generation <= 1000; ++generation) {
    check(cudaSetDevice(0));
    __atomic_store_n(host + 256, 0U, __ATOMIC_RELEASE);
    fill<<<1, 256, 0, stream>>>(data, generation);
    check(cudaGetLastError());
    check(cudaMemcpyAsync(host, data, 256 * sizeof(unsigned),
                          cudaMemcpyDeviceToHost, stream));
    check(cudaEventRecord(event, stream));
    // A handle may be waited from a thread whose current device differs.
    if (devices > 1)
      check(cudaSetDevice(1));
    const auto status = cuStreamWriteValue32(
        stream, reinterpret_cast<CUdeviceptr>(mapped + 256), generation,
        CU_STREAM_WRITE_VALUE_DEFAULT);
    if (status != CUDA_SUCCESS) {
      std::fprintf(stderr, "cuStreamWriteValue32 error=%d\n", int(status));
      return 1;
    }
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(5);
    while (__atomic_load_n(host + 256, __ATOMIC_ACQUIRE) != generation) {
      if (std::chrono::steady_clock::now() >= deadline) {
        std::fprintf(stderr, "completion timeout at generation=%u\n", generation);
        return 1;
      }
    }
    check(cudaEventQuery(event));
    for (unsigned index = 0; index < 256; ++index) {
      if (host[index] != generation + index) {
        std::fprintf(stderr, "preceding D2H data not visible at generation=%u\n",
                     generation);
        return 1;
      }
    }
  }
  check(cudaSetDevice(0));
  check(cudaStreamSynchronize(stream));
  check(cudaStreamDestroy(stream));
  check(cudaEventDestroy(event));
  check(cudaFree(data));
  check(cudaFreeHost(host));
  std::printf("PASS: 1000 completion generations; preceding data/event visible; "
              "alternate current device=%d\n", devices > 1);
}
