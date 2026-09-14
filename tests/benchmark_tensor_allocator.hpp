#pragma once

#include <algorithm>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "weftmoe/cuda/tensor_storage.hpp"

__global__ void benchmark_compare_bytes(const std::uint32_t *left,
                                        const std::uint32_t *right,
                                        std::uint64_t words,
                                        unsigned int *mismatch) {
  for (std::uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < words; index += gridDim.x * blockDim.x)
    if (left[index] != right[index])
      atomicExch(mismatch, 1U);
}

// A synchronous, caller-owned caching allocator for the native benchmark.
// Each invocation requests fresh tensor ownership. A block becomes reusable
// only after all handles/consumers release it, after the benchmark's wait().
// Like torch.empty, allocation requests are timed but warm cache misses are
// excluded. This is not a production allocator or a cross-stream free list.
class BenchmarkTensorAllocator {
public:
  weftmoe::cuda::CudaTensorBuffer allocate(std::uint64_t bytes) {
    ++requests;
    requested_bytes += bytes;
    for (const auto &block : blocks_)
      if (block.bytes == bytes && block.owner.use_count() == 1)
        return block;
    void *data = nullptr;
    const auto error = cudaMalloc(&data, std::max<std::uint64_t>(128, bytes));
    if (error != cudaSuccess)
      throw std::runtime_error(std::string("benchmark tensor allocation: ") +
                               cudaGetErrorString(error));
    weftmoe::cuda::CudaTensorBuffer block{
        data, bytes, std::shared_ptr<const void>(data, [](const void *pointer) {
          (void)cudaFree(const_cast<void *>(pointer));
        })};
    blocks_.push_back(block);
    ++misses;
    return block;
  }

  std::uint64_t requests = 0;
  std::uint64_t requested_bytes = 0;
  std::uint64_t misses = 0;

private:
  std::vector<weftmoe::cuda::CudaTensorBuffer> blocks_;
};

// Correctness passes compare every output bit after completion, outside timing.
class BenchmarkRepeatCheck {
public:
  void check(const std::vector<weftmoe::cuda::CudaTensorBuffer> &outputs,
             cudaStream_t stream) {
    const auto check_cuda = [](cudaError_t error) {
      if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
    };
    if (reference_.empty()) {
      mismatch_ = allocator_.allocate(sizeof(unsigned int));
      for (const auto &output : outputs) {
        reference_.push_back(allocator_.allocate(output.bytes));
        check_cuda(cudaMemcpyAsync(reference_.back().data, output.data,
                                   output.bytes, cudaMemcpyDeviceToDevice,
                                   stream));
      }
    } else {
      if (outputs.size() != reference_.size())
        throw std::runtime_error("repeat output count changed");
      check_cuda(
          cudaMemsetAsync(mismatch_.data, 0, sizeof(unsigned int), stream));
      for (std::size_t i = 0; i < outputs.size(); ++i) {
        if (outputs[i].bytes != reference_[i].bytes || outputs[i].bytes % 4U)
          throw std::runtime_error(
              "repeat output size changed or is not word aligned");
        benchmark_compare_bytes<<<256, 256, 0, stream>>>(
            static_cast<const std::uint32_t *>(outputs[i].data),
            static_cast<const std::uint32_t *>(reference_[i].data),
            outputs[i].bytes / 4U, static_cast<unsigned int *>(mismatch_.data));
        check_cuda(cudaGetLastError());
      }
      unsigned int mismatch = 0;
      check_cuda(cudaMemcpyAsync(&mismatch, mismatch_.data, sizeof(mismatch),
                                 cudaMemcpyDeviceToHost, stream));
      check_cuda(cudaStreamSynchronize(stream));
      if (mismatch)
        throw std::runtime_error("output bits changed across repetitions");
    }
    check_cuda(cudaStreamSynchronize(stream));
    ++repetitions;
  }
  unsigned int repetitions = 0;

private:
  BenchmarkTensorAllocator allocator_;
  std::vector<weftmoe::cuda::CudaTensorBuffer> reference_;
  weftmoe::cuda::CudaTensorBuffer mismatch_;
};
