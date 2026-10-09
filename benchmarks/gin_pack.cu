// Isolated two-node GIN pack ablation. No WeftMoE kernels are linked.
// Build: nvcc -O3 -std=c++17 --expt-relaxed-constexpr -arch=sm_90a
//        benchmarks/gin_pack.cu -lnccl -o build/gin_pack
#include <cooperative_groups.h>
#include <cuda/atomic>
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
  int depth;
};

struct Sample {
  unsigned long long begin, copied, network, end;
  unsigned long long first_issue, producer_end;
};

enum class Mode { Burst, Window, Staged, Stream };

const char *mode_name(Mode mode) {
  switch (mode) {
  case Mode::Burst:
    return "burst";
  case Mode::Window:
    return "window";
  case Mode::Staged:
    return "staged";
  case Mode::Stream:
    return "stream";
  }
  return "";
}

using Ready = cuda::atomic_ref<unsigned, cuda::thread_scope_device>;

__device__ int first_row(int group, int context, Shape shape) {
  // Subdivide the original owner's batch, without redistributing its tokens.
  // Otherwise a deeper window also improves tail load balance, confounding
  // the asynchronous-issue ablation.
  return (group / shape.depth * shape.contexts + context) * shape.batch_tokens +
         (group & (shape.depth - 1)) * (shape.batch_tokens / shape.depth);
}

// One producer CTA owns an ordered stripe. The full TX plane is immutable
// until the sample ends; this experiment pipelines requests, not RX/TX reuse.
__device__ void produce(const uint4 *input, uint4 *packed,
                        const unsigned *order, unsigned *pack_ready,
                        Shape shape, int rows, int context, unsigned base,
                        Sample *sample) {
  const int chunk = shape.batch_tokens / shape.depth;
  const int row_vectors = shape.row_bytes / sizeof(uint4);
  for (int group = 0;; ++group) {
    const int first = first_row(group, context, shape);
    if (first >= rows)
      break;
    const int end = min(rows, first + chunk);
    // Keep row-address calculation outside the vector loop. A flattened
    // index otherwise performs runtime 64-bit division for every uint4.
    for (int row = first; row < end; ++row) {
      const auto *source = input + size_t(order[row]) * row_vectors;
      auto *destination = packed + size_t(row) * row_vectors;
      for (int v = threadIdx.x; v < row_vectors; v += blockDim.x)
        destination[v] = source[v];
    }
    // Every writer releases its NIC-readable stores before the CTA leader
    // publishes the contiguous ready prefix to its communication owner.
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0)
      Ready(pack_ready[context]).store(++base, cuda::memory_order_release);
    __syncthreads();
  }
  if (context == 0 && threadIdx.x == 0)
    sample->producer_end = timer();
}

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
  gin.resetSignal(2 * contexts + blockIdx.x);
}

template <bool Pull, Mode Pipeline>
__global__ void transfer(ncclDevComm_t dev, ncclWindow_t window, char *buffer,
                         Shape shape, int peer, const int *tokens,
                         const unsigned *local_order,
                         const unsigned *peer_order, Sample *samples,
                         unsigned *errors, unsigned *pack_ready) {
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
  unsigned pack_base = 0, published = 0;
  // Only the owner lane uses these handles. Waiting the oldest prefix leaves
  // the other sub-batches in flight, then the vacated slot is refilled.
  __shared__ ncclGinRequest_t requests[4];

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

    if constexpr (Pipeline == Mode::Burst || Pipeline == Mode::Window) {
      if (shape.packet_tokens > 1) {
        for (auto i = tid; i < vectors; i += stride) {
          const auto row = i / row_vectors;
          packed[i] =
              input[size_t(send_order[row]) * row_vectors + i % row_vectors];
        }
        __threadfence_system();
      }
    } else if constexpr (Pipeline == Mode::Staged) {
      if (context >= shape.contexts && context < 2 * shape.contexts)
        produce(input, packed, send_order, pack_ready, shape, rows,
                context - shape.contexts, pack_base, samples + iteration);
    }
    grid.sync();
    if (tid == 0) {
      samples[iteration].copied = timer();
      if constexpr (Pipeline != Mode::Stream)
        samples[iteration].producer_end = samples[iteration].copied;
    }
    // A common rendezvous follows the serial pack phase (empty for Stream).
    // Stream publishes packed readiness per owner below, not through this join.
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

    if constexpr (Pipeline == Mode::Stream) {
      if (context >= shape.contexts && context < 2 * shape.contexts)
        produce(input, packed, send_order, pack_ready, shape, rows,
                context - shape.contexts, pack_base, samples + iteration);
    }
    if (owner) {
      if constexpr (Pipeline == Mode::Burst) {
        // Keep bytes in flight constant, not requests in flight. Each context
        // retires a batch_tokens-row prefix before issuing its next disjoint
        // batch.
        for (int first = context * shape.batch_tokens; first < rows;
             first += shape.contexts * shape.batch_tokens) {
          const int end = min(rows, first + shape.batch_tokens);
          if (context == 0 && first == 0)
            samples[iteration].first_issue = timer();
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
      } else {
        const int chunk = shape.batch_tokens / shape.depth;
        int issued = 0;
        for (;; ++issued) {
          const int first = first_row(issued, context, shape);
          if (first >= rows)
            break;
          const int slot = issued & (shape.depth - 1);
          if (issued >= shape.depth)
            gin.wait(requests[slot]);

          if constexpr (Pipeline == Mode::Stream) {
            unsigned head;
            do {
              head =
                  Ready(pack_ready[context]).load(cuda::memory_order_acquire);
            } while (head < pack_base + issued + 1);
            if constexpr (Pull) {
              // Publish before waiting for the peer, so symmetric pull cannot
              // deadlock. One ordered producer makes this a true ready prefix.
              if (head > published) {
                gin.signal(team, peer,
                           ncclGin_WeakSignalAdd{2 * unsigned(shape.contexts) +
                                                     context,
                                                 head - published},
                           ncclCoopThread{}, ncclGin_None{},
                           cuda::thread_scope_system,
                           cuda::thread_scope_system);
                published = head;
              }
              gin.waitSignal(ncclCoopThread{}, 2 * shape.contexts + context,
                             pack_base + issued + 1);
            }
          }
          if (context == 0 && issued == 0)
            samples[iteration].first_issue = timer();
          const int end = min(rows, first + chunk);
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
          gin.flushAsync(team, peer, &requests[slot]);
        }
        for (int pending = max(0, issued - shape.depth); pending < issued;
             ++pending)
          gin.wait(requests[pending & (shape.depth - 1)]);
        // Full RX does not wrap within a sample. Only the final aggregate
        // signal proves every inbound Put arrived; no prefix inference from
        // out-of-order weak completions and no premature RX reuse.
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
    const int chunk = shape.batch_tokens / shape.depth;
    const int wave = shape.contexts * shape.batch_tokens;
    const int tail = max(
        0, min(shape.batch_tokens, rows % wave - int(context % shape.contexts) *
                                                     shape.batch_tokens));
    pack_base += rows / wave * shape.depth + (tail + chunk - 1) / chunk;
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
  const auto *suite_env = std::getenv("WEFTMOE_GIN_BENCH_MODE");
  const std::string suite = suite_env ? suite_env : "burst";
  if (suite != "burst" && suite != "pipeline")
    throw std::runtime_error(
        "WEFTMOE_GIN_BENCH_MODE must be burst or pipeline");
  if (contexts < 1 || contexts > 128 || batch < 64 || batch % 64 ||
      hidden < 8 || hidden % 8 || warmup < 0 || measured < 1 ||
      (suite == "pipeline" && contexts > 64))
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
  requirements.ginSignalCount = 3 * contexts;
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
    unsigned *device_errors, *pack_ready;
    CUDA(cudaMalloc(&device_samples, iterations * sizeof(Sample)));
    CUDA(cudaMalloc(&device_errors, iterations * sizeof(unsigned)));
    CUDA(cudaMalloc(&pack_ready, contexts * sizeof(unsigned)));
    std::vector<Sample> samples(iterations);
    std::vector<unsigned> errors(iterations);
    for (int repeat = 0; repeat < 2; ++repeat) {
      struct Case {
        Mode mode;
        int depth, packet;
      };
      std::vector<Case> cases;
      if (suite == "burst") {
        for (const int packet : {1, 4, 16, 64})
          cases.push_back({Mode::Burst, 1, packet});
      } else {
        for (const int packet : {1, 16}) {
          cases.push_back({Mode::Burst, 1, packet});
          for (const int depth : {1, 2, 4})
            cases.push_back({Mode::Window, depth, packet});
          if (packet > 1) {
            cases.push_back({Mode::Staged, 4, packet});
            cases.push_back({Mode::Stream, 4, packet});
          }
        }
      }
      if (repeat)
        std::reverse(cases.begin(), cases.end());
      for (const auto test : cases) {
        const auto packet = test.packet;
        for (int variant = 0; variant < 2; ++variant) {
          const bool pull = bool(variant ^ repeat);
          reset<<<contexts, 1>>>(dev, contexts);
          CUDA(cudaMemset(device_errors, 0, iterations * sizeof(unsigned)));
          CUDA(cudaMemset(pack_ready, 0, contexts * sizeof(unsigned)));
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
          const Shape shape{capacity, hidden * 2, iterations, contexts,
                            batch,    packet,     test.depth};
          auto kernel =
              pull ? transfer<true, Mode::Burst> : transfer<false, Mode::Burst>;
          switch (test.mode) {
          case Mode::Burst:
            break;
          case Mode::Window:
            kernel = pull ? transfer<true, Mode::Window>
                          : transfer<false, Mode::Window>;
            break;
          case Mode::Staged:
            kernel = pull ? transfer<true, Mode::Staged>
                          : transfer<false, Mode::Staged>;
            break;
          case Mode::Stream:
            kernel = pull ? transfer<true, Mode::Stream>
                          : transfer<false, Mode::Stream>;
            break;
          }
          CUDA(cudaLaunchKernelEx(&launch, kernel, dev, window, buffer, shape,
                                  peer, device_tokens, local_order, peer_order,
                                  device_samples, device_errors, pack_ready));
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
                    << "\",\"packet_tokens\":" << packet << ",\"mode\":\""
                    << mode_name(test.mode) << "\""
                    << ",\"depth\":" << test.depth << ",\"repeat\":" << repeat
                    << ",\"contexts\":" << contexts
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
                      << ",\"pack_tail_after_issue_us\":"
                      << (s.producer_end > s.first_issue
                              ? (s.producer_end - s.first_issue) / 1000.0
                              : 0.0)
                      << ",\"total_us\":" << (s.end - s.begin) / 1000.0 << '}';
          }
          std::cout << "]}" << std::endl;
        }
      }
    }
    CUDA(cudaFree(pack_ready));
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
