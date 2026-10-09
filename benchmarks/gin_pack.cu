// Isolated two-node GIN pack ablation. No WeftMoE kernels are linked.
// Build: nvcc -O3 -std=c++17 --expt-relaxed-constexpr -arch=sm_90a
//        benchmarks/gin_pack.cu -lnccl -o build/gin_pack
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

#define CUDA(call)                                                             \
  do {                                                                         \
    const auto e = (call);                                                     \
    if (e != cudaSuccess)                                                      \
      throw std::runtime_error(std::string(#call) + ": " +                     \
                               cudaGetErrorString(e));                         \
  } while (0)
#define NCCL(call)                                                             \
  do {                                                                         \
    const auto e = (call);                                                     \
    if (e != ncclSuccess)                                                      \
      throw std::runtime_error(std::string(#call) + ": " +                     \
                               ncclGetErrorString(e));                         \
  } while (0)

namespace {

int setting(const char *name, int fallback) {
  const auto *value = std::getenv(name);
  return value ? std::stoi(value) : fallback;
}

std::vector<int> token_shapes() {
  const auto *value = std::getenv("WEFTMOE_P2P_BENCH_TOKENS");
  std::stringstream stream(value ? value : "8192,20480");
  std::vector<int> result;
  std::string item;
  while (std::getline(stream, item, ','))
    result.push_back(std::stoi(item));
  return result;
}

void bootstrap(int fd, ncclUniqueId &id, bool writer) {
  auto *bytes = reinterpret_cast<char *>(&id);
  size_t offset = 0;
  while (offset < sizeof(id)) {
    const auto count = writer ? write(fd, bytes + offset, sizeof(id) - offset)
                              : read(fd, bytes + offset, sizeof(id) - offset);
    if (count <= 0)
      throw std::runtime_error("bootstrap pipe failed");
    offset += size_t(count);
  }
  close(fd);
}

__host__ __device__ unsigned pattern(unsigned rank, unsigned row,
                                     unsigned word) {
  unsigned value = (rank + 1U) * 0x9e3779b9U ^ row * 0x85ebca6bU ^ word;
  value ^= value >> 16;
  value *= 0x7feb352dU;
  return value ^ (value >> 15);
}

__device__ unsigned long long timer() {
  unsigned long long result;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(result));
  return result;
}

struct Shape {
  int capacity, row_bytes, iterations, contexts, batch_tokens, packet_tokens;
};

struct Sample {
  unsigned long long begin, copied, network, end;
};

__global__ void initialize(unsigned *input, int capacity, int row_words,
                           int rank) {
  const auto size = size_t(capacity) * row_words;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < size;
       i += size_t(gridDim.x) * blockDim.x)
    input[i] = pattern(rank, i / row_words, i % row_words);
}

__global__ void reset(ncclDevComm_t dev, int contexts) {
  const ncclGin gin(dev, blockIdx.x, NCCL_GIN_RESOURCE_SHARING_THREAD);
  gin.resetSignal(blockIdx.x);
  gin.resetSignal(contexts + blockIdx.x);
}

template <bool Pull>
__global__ void
transfer(ncclDevComm_t dev, ncclWindow_t window, char *buffer, Shape shape,
         int peer, const int *tokens, const unsigned *local_order,
         const unsigned *peer_order, Sample *samples, unsigned *errors) {
  const auto grid = cooperative_groups::this_grid();
  const auto tid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto stride = size_t(gridDim.x) * blockDim.x;
  const auto plane = size_t(shape.capacity) * shape.row_bytes;
  const auto *input = reinterpret_cast<const uint4 *>(buffer);
  auto *packed = reinterpret_cast<uint4 *>(buffer + plane);
  auto *received = reinterpret_cast<uint4 *>(buffer + 2 * plane);
  const int row_vectors = shape.row_bytes / sizeof(uint4);
  const auto context = blockIdx.x;
  const bool owner = context < shape.contexts && threadIdx.x == 0;
  const ncclGin gin(dev, context % shape.contexts,
                    NCCL_GIN_RESOURCE_SHARING_THREAD);
  const auto team = ncclTeamWorld(dev);
  uint64_t epoch = 0, ready = 0;

  for (int iteration = 0; iteration < shape.iterations; ++iteration) {
    const int rows = tokens[iteration];
    const auto vectors = size_t(rows) * row_vectors;
    const auto *send_order = local_order + size_t(iteration) * shape.capacity;
    const auto *recv_order = peer_order + size_t(iteration) * shape.capacity;
    // Poison every destination before its peer is allowed to issue this sample.
    // Validation is outside timing and covers every byte, including tail
    // packets.
    for (auto i = tid; i < vectors; i += stride)
      received[i] = make_uint4(0, 0, 0, 0);
    __threadfence_system();
    grid.sync();
    if (owner) {
      gin.signal(team, peer, ncclGin_WeakSignalInc{context}, ncclCoopThread{},
                 ncclGin_None{}, cuda::thread_scope_system,
                 cuda::thread_scope_system);
      gin.waitSignal(ncclCoopThread{}, context, ++epoch);
    }
    grid.sync();
    if (tid == 0)
      samples[iteration].begin = timer();
    grid.sync();

    if (shape.packet_tokens > 1) {
      for (auto i = tid; i < vectors; i += stride) {
        const auto row = i / row_vectors;
        packed[i] =
            input[size_t(send_order[row]) * row_vectors + i % row_vectors];
      }
      // The pack writers, not just the issuing lane, publish the NIC source.
      __threadfence_system();
    }
    grid.sync();
    if (tid == 0)
      samples[iteration].copied = timer();
    // Both variants pay the same source-ready rendezvous. Pull must not read
    // the peer's packed plane until all its pack writers have finished.
    if (owner) {
      gin.signal(team, peer, ncclGin_WeakSignalInc{context}, ncclCoopThread{},
                 ncclGin_None{}, cuda::thread_scope_system,
                 cuda::thread_scope_system);
      gin.waitSignal(ncclCoopThread{}, context, ++epoch);
    }
    grid.sync();
    if (tid == 0)
      samples[iteration].network = timer();
    grid.sync();

    if (owner) {
      // Keep bytes in flight constant, not requests in flight. Each context
      // retires a batch_tokens-row prefix before issuing its next disjoint
      // batch.
      for (int first = context * shape.batch_tokens; first < rows;
           first += shape.contexts * shape.batch_tokens) {
        const int end = min(rows, first + shape.batch_tokens);
        for (int row = first; row < end; row += shape.packet_tokens) {
          const auto count = min(shape.packet_tokens, end - row);
          const auto source =
              shape.packet_tokens == 1
                  ? size_t((Pull ? recv_order : send_order)[row]) *
                        shape.row_bytes
                  : plane + size_t(row) * shape.row_bytes;
          const auto destination = 2 * plane + size_t(row) * shape.row_bytes;
          if constexpr (Pull) {
            gin.get(team, peer, window, source, window, destination,
                    size_t(count) * shape.row_bytes);
          } else {
            gin.put(team, peer, window, destination, window, source,
                    size_t(count) * shape.row_bytes,
                    ncclGin_WeakSignalInc{unsigned(shape.contexts) + context},
                    ncclGin_None{}, ncclCoopThread{}, ncclGin_None{},
                    cuda::thread_scope_system, cuda::thread_scope_system);
            ++ready;
          }
        }
        ncclGinRequest_t completion;
        gin.flushAsync(team, peer, &completion);
        gin.wait(completion);
        if constexpr (!Pull)
          gin.waitSignal(ncclCoopThread{}, shape.contexts + context, ready);
      }
    }
    grid.sync();
    if (tid == 0)
      samples[iteration].end = timer();
    // Republish the NIC acquire from each owner to all validation threads.
    grid.sync();
    bool mismatch = false;
    const auto *words = reinterpret_cast<const unsigned *>(received);
    const auto row_words = shape.row_bytes / sizeof(unsigned);
    for (auto i = tid; i < size_t(rows) * row_words; i += stride)
      mismatch |=
          words[i] != pattern(peer, recv_order[i / row_words], i % row_words);
    if (mismatch)
      atomicOr(errors + iteration, 1U);
    grid.sync();
  }
}

template <class T> T *upload(const std::vector<T> &values) {
  T *result;
  CUDA(cudaMalloc(&result, values.size() * sizeof(T)));
  CUDA(cudaMemcpy(result, values.data(), values.size() * sizeof(T),
                  cudaMemcpyHostToDevice));
  return result;
}

std::vector<unsigned> orders(int capacity, int iterations, unsigned seed) {
  std::mt19937 random(seed);
  std::vector<unsigned> result(size_t(capacity) * iterations),
      permutation(capacity);
  std::iota(permutation.begin(), permutation.end(), 0U);
  for (int i = 0; i < iterations; ++i) {
    std::shuffle(permutation.begin(), permutation.end(), random);
    std::copy(permutation.begin(), permutation.end(),
              result.begin() + size_t(i) * capacity);
  }
  return result;
}

void run(const std::map<std::string, std::string> &args) {
  const int rank = std::stoi(args.at("--rank"));
  const int local = std::stoi(args.at("--local-rank"));
  const int local_world = std::stoi(args.at("--gpus-per-node"));
  const int world = 2 * local_world;
  const int peer = (rank + local_world) % world;
  const auto seed = unsigned(std::stoull(args.at("--seed")));
  const int contexts = setting("WEFTMOE_GIN_BENCH_CONTEXTS", 16);
  const int warmup = setting("WEFTMOE_GIN_BENCH_WARMUP", 5);
  const int measured = setting("WEFTMOE_GIN_BENCH_MEASURED", 20);
  const int batch = setting("WEFTMOE_GIN_BENCH_BATCH", 128);
  const int hidden = setting("WEFTMOE_BENCH_HIDDEN", 4096);
  if (contexts < 1 || contexts > 128 || batch < 64 || batch % 64 ||
      hidden < 8 || hidden % 8 || warmup < 0 || measured < 1)
    throw std::runtime_error("invalid contexts/batch/hidden/sample settings");
  CUDA(cudaSetDevice(local));
  ncclUniqueId id;
  if (rank == 0) {
    NCCL(ncclGetUniqueId(&id));
    bootstrap(std::stoi(args.at("--bootstrap-output-fd")), id, true);
  } else {
    bootstrap(std::stoi(args.at("--bootstrap-input-fd")), id, false);
  }
  ncclComm_t comm;
  NCCL(ncclCommInitRank(&comm, world, id, rank));
  ncclCommProperties_t properties{};
  properties.size = sizeof(properties);
  properties.magic = NCCL_API_MAGIC;
  properties.version = NCCL_VERSION_CODE;
  NCCL(ncclCommQueryProperties(comm, &properties));
  if (properties.railedGinType == NCCL_GIN_TYPE_NONE)
    throw std::runtime_error("GIN backend unavailable");
  ncclDevCommRequirements_t requirements =
      NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  requirements.ginContextCount = contexts;
  requirements.ginExclusiveContexts = true;
  requirements.ginQueueDepth = 1024;
  requirements.ginSignalCount = 2 * contexts;
  requirements.ginConnectionType = NCCL_GIN_CONNECTION_RAIL;
  ncclDevComm_t dev;
  NCCL(ncclDevCommCreate(comm, &requirements, &dev));
  int version;
  NCCL(ncclGetVersion(&version));
  std::cout << "GIN_PACK_RUNTIME rank=" << rank << " version=" << version
            << " railed_gin_type=" << properties.railedGinType
            << " contexts=" << contexts << std::endl;

  for (const int nominal : token_shapes()) {
    const int iterations = warmup + measured;
    std::mt19937 random(seed + nominal);
    std::normal_distribution<double> distribution(nominal, nominal * 0.1);
    std::vector<int> tokens(iterations);
    for (auto &count : tokens)
      count = std::max(1, int(std::lround(distribution(random))));
    const int capacity =
        (*std::max_element(tokens.begin(), tokens.end()) + 63) / 64 * 64;
    const auto plane = size_t(capacity) * hidden * 2;
    const auto bytes = 3 * plane + 4096;
    void *allocation;
    NCCL(ncclMemAlloc(&allocation, bytes));
    auto *buffer = static_cast<char *>(allocation);
    CUDA(cudaMemset(buffer, 0, bytes));
    initialize<<<128, 256>>>(reinterpret_cast<unsigned *>(buffer), capacity,
                             hidden / 2, rank);
    CUDA(cudaDeviceSynchronize());
    ncclWindow_t window;
    NCCL(ncclCommWindowRegister(comm, buffer, bytes, &window,
                                NCCL_WIN_STRICT_ORDERING));
    auto *device_tokens = upload(tokens);
    auto *local_order =
        upload(orders(capacity, iterations, seed + nominal + rank * 7919U));
    auto *peer_order =
        upload(orders(capacity, iterations, seed + nominal + peer * 7919U));
    Sample *device_samples;
    unsigned *device_errors;
    CUDA(cudaMalloc(&device_samples, iterations * sizeof(Sample)));
    CUDA(cudaMalloc(&device_errors, iterations * sizeof(unsigned)));
    std::vector<Sample> samples(iterations);
    std::vector<unsigned> errors(iterations);
    for (int repeat = 0; repeat < 2; ++repeat) {
      std::vector<int> packets{1, 4, 16, 64};
      if (repeat)
        std::reverse(packets.begin(), packets.end());
      for (const int packet : packets) {
        for (int variant = 0; variant < 2; ++variant) {
          const bool pull = bool(variant ^ repeat);
          reset<<<contexts, 1>>>(dev, contexts);
          CUDA(cudaMemset(device_errors, 0, iterations * sizeof(unsigned)));
          auto *sync = buffer + 3 * plane;
          NCCL(
              ncclAllReduce(sync, sync, 1, ncclUint64, ncclSum, comm, nullptr));
          cudaLaunchAttribute attribute{};
          attribute.id = cudaLaunchAttributeCooperative;
          attribute.val.cooperative = 1;
          cudaLaunchConfig_t launch{};
          launch.gridDim = dim3(128);
          launch.blockDim = dim3(128);
          launch.attrs = &attribute;
          launch.numAttrs = 1;
          const Shape shape{capacity, hidden * 2, iterations,
                            contexts, batch,      packet};
          CUDA(cudaLaunchKernelEx(
              &launch, pull ? transfer<true> : transfer<false>, dev, window,
              buffer, shape, peer, device_tokens, local_order, peer_order,
              device_samples, device_errors));
          CUDA(cudaDeviceSynchronize());
          CUDA(cudaMemcpy(samples.data(), device_samples,
                          iterations * sizeof(Sample), cudaMemcpyDeviceToHost));
          CUDA(cudaMemcpy(errors.data(), device_errors,
                          iterations * sizeof(unsigned),
                          cudaMemcpyDeviceToHost));
          if (std::any_of(errors.begin(), errors.end(),
                          [](unsigned e) { return e != 0; }))
            throw std::runtime_error("received payload mismatch");
          std::cout << "GIN_PACK_CASE {\"rank\":" << rank
                    << ",\"nominal_tokens\":" << nominal << ",\"operation\":\""
                    << (pull ? "pull" : "push")
                    << "\",\"packet_tokens\":" << packet
                    << ",\"repeat\":" << repeat << ",\"contexts\":" << contexts
                    << ",\"batch_tokens\":" << batch
                    << ",\"row_bytes\":" << hidden * 2
                    << ",\"registered_bytes\":" << bytes
                    << ",\"warmup\":" << warmup
                    << ",\"validated_samples\":" << iterations
                    << ",\"samples\":[";
          for (int i = warmup; i < iterations; ++i) {
            if (i != warmup)
              std::cout << ',';
            const auto &s = samples[i];
            std::cout << "{\"tokens\":" << tokens[i]
                      << ",\"network_us\":" << (s.end - s.network) / 1000.0
                      << ",\"pack_us\":" << (s.copied - s.begin) / 1000.0
                      << ",\"ready_us\":" << (s.network - s.copied) / 1000.0
                      << ",\"total_us\":" << (s.end - s.begin) / 1000.0 << '}';
          }
          std::cout << "]}" << std::endl;
        }
      }
    }
    CUDA(cudaFree(device_errors));
    CUDA(cudaFree(device_samples));
    CUDA(cudaFree(peer_order));
    CUDA(cudaFree(local_order));
    CUDA(cudaFree(device_tokens));
    NCCL(ncclCommWindowDeregister(comm, window));
    NCCL(ncclMemFree(allocation));
  }
  NCCL(ncclDevCommDestroy(comm, &dev));
  NCCL(ncclCommDestroy(comm));
}

} // namespace

int main(int argc, char **argv) {
  try {
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; ++i) {
      const std::string key = argv[i];
      if (key != "--rank-runner" && i + 1 < argc)
        args[key] = argv[++i];
    }
    std::cout << std::fixed << std::setprecision(3);
    run(args);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "[FAIL] " << error.what() << std::endl;
    return 1;
  }
}
