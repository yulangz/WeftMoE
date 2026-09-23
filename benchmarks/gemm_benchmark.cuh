#pragma once
#include <chrono>
#include <nvml.h>
#include <thread>

// Shared by the FWD/BWD static-task diagnostics. Both sides use BF16 operands
// and output, with FP32 accumulation and cuBLAS reduced-precision reduction
// off.
#define CK(call)                                                               \
  do {                                                                         \
    auto error = (call);                                                       \
    if (error != cudaSuccess)                                                  \
      throw std::runtime_error(std::string(#call) + ": " +                     \
                               cudaGetErrorString(error));                     \
  } while (0)
#define DR(call)                                                               \
  do {                                                                         \
    auto error = (call);                                                       \
    if (error != CUDA_SUCCESS)                                                 \
      throw std::runtime_error(#call);                                         \
  } while (0)
#define BL(call)                                                               \
  do {                                                                         \
    auto error = (call);                                                       \
    if (error != CUBLAS_STATUS_SUCCESS)                                        \
      throw std::runtime_error(std::string(#call) + ": " +                     \
                               std::to_string(error));                         \
  } while (0)
CUtensorMap tensor_map(void *pointer, int width, int height, int box_rows) {
  CUtensorMap result;
  cuuint64_t dims[] = {cuuint64_t(width), cuuint64_t(height)};
  cuuint64_t strides[] = {cuuint64_t(width) * 2};
  cuuint32_t box[] = {64, unsigned(box_rows)}, elements[] = {1, 1};
  DR(cuTensorMapEncodeTiled(
      &result, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, pointer, dims, strides, box,
      elements, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return result;
}
__global__ void fill(transport::BFloat16 *p, std::size_t n) {
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    unsigned value = unsigned(i) ^ 0x9e3779b9U;
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    value ^= value >> 16;
    p[i] = transport::BFloat16((int(value & 255) - 128) / 1024.f);
  }
}
__global__ void sample_output(const transport::BFloat16 *p, std::size_t n,
                              transport::BFloat16 *sample) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 512)
    sample[i] = p[std::uint64_t(i) * (n - 1) / 511];
}
__global__ void check_output(const transport::BFloat16 *expected,
                             const transport::BFloat16 *actual, std::size_t n,
                             unsigned *error, bool exact) {
  bool mismatch = false;
  float maximum = 0.0F;
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    const float reference = float(expected[i]), observed = float(actual[i]);
    const float difference = fabsf(reference - observed);
    maximum = fmaxf(maximum, difference);
    mismatch |= exact ? reinterpret_cast<const unsigned short *>(expected)[i] !=
                            reinterpret_cast<const unsigned short *>(actual)[i]
                      : !isfinite(reference) || !isfinite(observed) ||
                            difference > 0.005F + 0.01F * fabsf(reference);
  }
  if (mismatch)
    atomicOr(error, 1U);
  atomicMax(error + 1, __float_as_uint(maximum));
}
// Alternate implementations and order after warming both. Cooldown preserves
// the warm cache while avoiding sustained power-cap throttling in diagnostics.
// Tensor and handle allocation is excluded from this pure-GEMM diagnostic.
template <class Native, class Reference>
void compare_gemms(Native native, Reference cublas, transport::BFloat16 *output,
                   transport::BFloat16 *sample, std::size_t no,
                   const std::string &op, int M, int N, int K, int groups,
                   int ctas, int device, bool reverse, cudaStream_t stream,
                   double work_flops = 0.0) {
  CK(cudaDeviceSynchronize());
  native();
  cublas();
  CK(cudaStreamSynchronize(stream));
  cudaGraph_t graphs[2];
  cudaGraphExec_t executables[2];
  for (int implementation = 0; implementation < 2; ++implementation) {
    CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    if (implementation)
      cublas();
    else
      native();
    CK(cudaStreamEndCapture(stream, &graphs[implementation]));
    CK(cudaGraphInstantiate(&executables[implementation],
                            graphs[implementation], 0));
  }
  const auto launch = [&] { CK(cudaGraphLaunch(executables[0], stream)); };
  const auto reference = [&] { CK(cudaGraphLaunch(executables[1], stream)); };
  std::vector<float> times[2];
  std::vector<std::pair<unsigned, unsigned>> clocks[2];
  std::vector<std::pair<long long, long long>> windows[2];
  if (nvmlInit_v2() != NVML_SUCCESS)
    throw std::runtime_error("NVML initialization failed");
  char pci[32];
  CK(cudaDeviceGetPCIBusId(pci, sizeof(pci), device));
  nvmlDevice_t nvml_device;
  if (nvmlDeviceGetHandleByPciBusId_v2(pci, &nvml_device) != NVML_SUCCESS)
    throw std::runtime_error("NVML GPU lookup failed");
  const auto sm_clock = [&] {
    unsigned mhz;
    if (nvmlDeviceGetClockInfo(nvml_device, NVML_CLOCK_SM, &mhz) !=
        NVML_SUCCESS)
      throw std::runtime_error("NVML clock query failed");
    return mhz;
  };
  const char *cooldown_env = std::getenv("WEFTMOE_GEMM_COOLDOWN_MS");
  const int cooldown_ms = cooldown_env ? std::atoi(cooldown_env) : 50;
  if (cooldown_ms < 0 || cooldown_ms > 2000)
    throw std::runtime_error("WEFTMOE_GEMM_COOLDOWN_MS must be in [0, 2000]");
  const auto cooldown = [=] {
    std::this_thread::sleep_for(std::chrono::milliseconds(cooldown_ms));
  };
  const auto wall_ns = [] {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
  };
  std::vector<transport::BFloat16> snapshots[2];
  const auto warm_start = std::chrono::steady_clock::now();
  do {
    launch();
    CK(cudaDeviceSynchronize());
    cooldown();
    reference();
    CK(cudaDeviceSynchronize());
    cooldown();
  } while (std::chrono::steady_clock::now() - warm_start <
           std::chrono::milliseconds(2000));
  cudaEvent_t begin, end;
  CK(cudaEventCreate(&begin));
  CK(cudaEventCreate(&end));
  for (int round = 0; round < 35; ++round) {
    for (int pass = 0; pass < 2; ++pass) {
      const int implementation = (round + pass + reverse) % 2;
      cooldown();
      if (implementation)
        reference();
      else
        launch();
      CK(cudaDeviceSynchronize());
      cooldown();
      const auto before_clock = sm_clock();
      const auto host_begin = wall_ns();
      CK(cudaEventRecord(begin, stream));
      if (implementation)
        reference();
      else
        launch();
      CK(cudaEventRecord(end, stream));
      CK(cudaEventSynchronize(end));
      const auto host_end = wall_ns();
      const auto after_clock = sm_clock();
      float ms;
      CK(cudaEventElapsedTime(&ms, begin, end));
      if (round >= 5) {
        times[implementation].push_back(ms);
        clocks[implementation].emplace_back(before_clock, after_clock);
        windows[implementation].emplace_back(host_begin, host_end);
      }
    }
  }
  CK(cudaEventDestroy(begin));
  CK(cudaEventDestroy(end));
  // Synchronize graph replay before validation copies on the legacy stream.
  for (int implementation = 0; implementation < 2; ++implementation) {
    if (implementation)
      reference();
    else
      launch();
    CK(cudaStreamSynchronize(stream));
    sample_output<<<2, 256>>>(output, no, sample);
    snapshots[implementation].resize(512);
    CK(cudaMemcpy(snapshots[implementation].data(), sample, 1024,
                  cudaMemcpyDeviceToHost));
  }
  double max_error = 0;
  for (int i = 0; i < 512; ++i) {
    const double error =
        std::abs(float(snapshots[0][i]) - float(snapshots[1][i]));
    max_error = std::max(max_error, error);
    if (error > 0.005 + 0.01 * std::abs(float(snapshots[1][i])))
      throw std::runtime_error("GEMM sample mismatch");
  }
  // Validate the entire output and repeated-run determinism outside timing.
  transport::BFloat16 *saved_output;
  unsigned *device_error, host_error[2];
  CK(cudaMalloc(&saved_output, no * 2));
  CK(cudaMalloc(&device_error, sizeof(host_error)));
  launch();
  CK(cudaStreamSynchronize(stream));
  CK(cudaMemcpy(saved_output, output, no * 2, cudaMemcpyDeviceToDevice));
  launch();
  CK(cudaStreamSynchronize(stream));
  CK(cudaMemset(device_error, 0, sizeof(host_error)));
  check_output<<<132, 256>>>(saved_output, output, no, device_error, true);
  CK(cudaMemcpy(host_error, device_error, sizeof(host_error),
                cudaMemcpyDeviceToHost));
  if (host_error[0])
    throw std::runtime_error("GEMM full-output repeat mismatch");
  reference();
  CK(cudaStreamSynchronize(stream));
  CK(cudaMemset(device_error, 0, sizeof(host_error)));
  check_output<<<132, 256>>>(output, saved_output, no, device_error, false);
  CK(cudaMemcpy(host_error, device_error, sizeof(host_error),
                cudaMemcpyDeviceToHost));
  if (host_error[0])
    throw std::runtime_error("GEMM full-output cuBLAS mismatch");
  float full_max_error;
  std::memcpy(&full_max_error, host_error + 1, sizeof(float));
  CK(cudaFree(saved_output));
  CK(cudaFree(device_error));
  std::printf("VALIDATION full_output=1 repeat_byte_identical=1 max_abs=%.9f\n",
              full_max_error);
  const double ms =
      std::accumulate(times[0].begin(), times[0].end(), 0.) / times[0].size();
  const double cu_ms =
      std::accumulate(times[1].begin(), times[1].end(), 0.) / times[1].size();
  const double flops = work_flops > 0.0 ? work_flops : 2. * groups * M * N * K;
  std::printf(
      "{\"op\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,\"groups\":%d,\"ctas\":%d,"
      "\"device\":%d,\"cuda_graph\":true,\"cooldown_ms\":%d,\"flops\":%.0f,"
      "\"ms\":%.6f,\"tflops\":%.3f,\"cublas_ms\":%"
      ".6f,\"cublas_tflops\":%.3f,\"sample_max_abs\":%.6f}\n",
      op.c_str(), M, N, K, groups, ctas, device, cooldown_ms, flops, ms,
      flops / ms / 1e9, cu_ms, flops / cu_ms / 1e9, max_error);
  for (int impl = 0; impl < 2; ++impl) {
    std::printf("SAMPLES %d", impl);
    for (auto ms : times[impl])
      std::printf(" %.9f", ms);
    std::printf("\n");
    std::printf("CLOCKS %d", impl);
    for (const auto &mhz : clocks[impl])
      std::printf(" %u:%u", mhz.first, mhz.second);
    std::printf("\n");
    std::printf("WINDOWS %d", impl);
    for (const auto &window : windows[impl])
      std::printf(" %lld:%lld", window.first, window.second);
    std::printf("\n");
  }
  nvmlShutdown();
  for (int implementation = 0; implementation < 2; ++implementation) {
    CK(cudaGraphExecDestroy(executables[implementation]));
    CK(cudaGraphDestroy(graphs[implementation]));
  }
}
