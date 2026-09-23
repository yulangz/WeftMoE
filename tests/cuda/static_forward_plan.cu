#include "cuda/static_v4/source_plan.cuh"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace plan = weftmoe::cuda::static_forward;

namespace {

void check(cudaError_t result) {
  if (result != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(result));
}

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <class T> class DeviceArray {
public:
  explicit DeviceArray(std::size_t size) : size_(size) {
    check(cudaMalloc(reinterpret_cast<void **>(&data_),
                     std::max(size, std::size_t{1}) * sizeof(T)));
    check(cudaMemset(data_, 0, std::max(size, std::size_t{1}) * sizeof(T)));
  }
  explicit DeviceArray(const std::vector<T> &values) : DeviceArray(values.size()) {
    if (size_)
      check(cudaMemcpy(data_, values.data(), size_ * sizeof(T), cudaMemcpyHostToDevice));
  }
  ~DeviceArray() { cudaFree(data_); }
  DeviceArray(const DeviceArray &) = delete;
  DeviceArray &operator=(const DeviceArray &) = delete;
  T *data() const { return data_; }
  std::vector<T> read() const {
    std::vector<T> values(size_);
    if (size_)
      check(cudaMemcpy(values.data(), data_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
    return values;
  }

private:
  T *data_ = nullptr;
  std::size_t size_;
};

__global__ void build_windows(plan::SourcePlanView view) {
  extern __shared__ unsigned scratch[];
  for (auto w = std::uint64_t(blockIdx.x); w < view.local_windows(); w += gridDim.x)
    plan::build_source_window<128U>(view, w, scratch);
}

__global__ void build_packets(plan::SourcePlanView view) {
  plan::build_source_packet_prefix(view, blockIdx.x * blockDim.x + threadIdx.x,
                                  gridDim.x * blockDim.x);
}

void run(unsigned nodes, unsigned tokens, bool hotspot, bool invalid = false) {
  constexpr unsigned local_world = 8U, top_k = 4U;
  const auto world = nodes * local_world;
  const auto experts = world * 4U;
  const auto source = world - 1U;
  const auto experts_per_node = experts / nodes;
  const plan::WindowShape shape{world, 16U, 128U, top_k};
  const plan::PacketPartition partition{32U, 2U};
  const auto windows = plan::ceil_div(tokens, shape.tokens);
  const auto capacity = windows * shape.tokens * top_k;
  std::vector<std::int32_t> selected(std::size_t(tokens) * top_k);
  std::vector<std::uint8_t> valid(selected.size(), 1U);
  for (unsigned token = 0; token < tokens; ++token) {
    for (unsigned k = 0; k < top_k; ++k) {
      selected[std::size_t(token) * top_k + k] = static_cast<std::int32_t>(
          hotspot ? 0U : (token * 17U + source * 13U + k / 2U) % experts);
      valid[std::size_t(token) * top_k + k] = !(k == 3U && token % 7U == 0U);
    }
  }
  if (invalid)
    selected[0] = static_cast<std::int32_t>(experts);
  DeviceArray<std::int32_t> d_selected(selected);
  DeviceArray<std::uint8_t> d_valid(valid);
  DeviceArray<std::uint64_t> expert_prefix(windows * (experts + 1U));
  DeviceArray<std::uint64_t> node_prefix(windows * (nodes + 1U));
  DeviceArray<std::uint64_t> route_order(capacity);
  DeviceArray<std::uint32_t> source_slot(selected.size());
  DeviceArray<plan::NodeToken> node_tokens(capacity);
  DeviceArray<std::uint64_t> packet_prefix(nodes * partition.lanes * (windows + 1U));
  DeviceArray<unsigned> error(1U);
  const plan::SourcePlanView view{
      shape, partition, source, experts, nodes, tokens, d_selected.data(),
      d_valid.data(), expert_prefix.data(), node_prefix.data(), route_order.data(),
      source_slot.data(), node_tokens.data(), packet_prefix.data(), error.data()};

  build_windows<<<7U, 128U, view.scratch_words() * sizeof(unsigned)>>>(view);
  check(cudaGetLastError());
  build_packets<<<1U, 128U>>>(view);
  check(cudaGetLastError());
  check(cudaDeviceSynchronize());
  require(error.read()[0] == unsigned(invalid), "invalid route status");
  if (invalid)
    return;

  const auto ep = expert_prefix.read();
  const auto np = node_prefix.read();
  const auto order = route_order.read();
  const auto ordinal = source_slot.read();
  const auto nt = node_tokens.read();
  const auto pp = packet_prefix.read();
  std::vector<std::uint64_t> sequences(nodes * partition.lanes, 0U);
  for (std::uint64_t w = 0; w < windows; ++w) {
    const auto begin = w * shape.tokens;
    const auto end = std::min<std::uint64_t>(begin + shape.tokens, tokens);
    std::vector<std::set<std::uint64_t>> expected_routes(experts);
    std::vector<std::set<std::uint64_t>> expected_tokens(nodes);
    std::vector<unsigned> fanouts((end - begin) * nodes, 0U);
    for (auto token = begin; token < end; ++token) {
      for (unsigned k = 0; k < top_k; ++k) {
        const auto route = token * top_k + k;
        if (!valid[route]) {
          require(ordinal[route] == UINT32_MAX, "masked route ordinal");
          continue;
        }
        const auto expert = unsigned(selected[route]);
        const auto node = expert / experts_per_node;
        expected_routes[expert].insert(route);
        expected_tokens[node].insert(token);
        ++fanouts[(token - begin) * nodes + node];
      }
    }
    for (unsigned e = 0; e < experts; ++e) {
      const auto first = ep[w * (experts + 1U) + e];
      const auto last = ep[w * (experts + 1U) + e + 1U];
      require(last - first == expected_routes[e].size(), "expert count");
      for (auto index = first; index < last; ++index) {
        const auto route = order[w * view.window_capacity() + index];
        require(expected_routes[e].erase(route) == 1U, "route coverage/uniqueness");
        require(ordinal[route] == index - first, "source ordinal inverse");
      }
    }
    for (unsigned node = 0; node < nodes; ++node) {
      const auto first = np[w * (nodes + 1U) + node];
      const auto last = np[w * (nodes + 1U) + node + 1U];
      require(last - first == expected_tokens[node].size(), "node dedup count");
      for (auto index = first; index < last; ++index) {
        const auto record = nt[w * view.window_capacity() + index];
        require(expected_tokens[node].erase(record.token) == 1U, "token coverage/uniqueness");
        require(record.fanouts == fanouts[(record.token - begin) * nodes + node],
                "packet expected fanout count");
      }
      for (unsigned lane = 0; lane < partition.lanes; ++lane) {
        const auto channel = node * partition.lanes + lane;
        const auto base = channel * (windows + 1U);
        require(pp[base + w] == sequences[channel], "packet start sequence");
        if (node != source / local_world)
          sequences[channel] += partition.lane_packets(last - first, lane);
        require(pp[base + w + 1U] == sequences[channel], "packet terminal sequence");
      }
    }
  }
  if (tokens == 0U)
    for (auto value : pp)
      require(value == 0U, "empty source channel");
}

} // namespace

int main() {
  try {
    for (unsigned nodes : {1U, 2U, 3U}) {
      for (unsigned tokens : {0U, 1U, 64U, 8192U, 20480U}) {
        run(nodes, tokens, false);
        run(nodes, tokens, true);
      }
    }
    run(3U, 64U, false, true);
    std::cout << "static V4 GPU source plan: 31 cases passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
