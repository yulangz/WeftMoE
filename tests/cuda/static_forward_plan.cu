#include "cuda/static_v4/packet_plan.cuh"
#include "cuda/static_v4/window_geometry.cuh"

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
  explicit DeviceArray(const std::vector<T> &values)
      : DeviceArray(values.size()) {
    if (size_)
      check(cudaMemcpy(data_, values.data(), size_ * sizeof(T),
                       cudaMemcpyHostToDevice));
  }
  ~DeviceArray() { cudaFree(data_); }
  DeviceArray(const DeviceArray &) = delete;
  DeviceArray &operator=(const DeviceArray &) = delete;
  T *data() const { return data_; }
  std::vector<T> read() const {
    std::vector<T> values(size_);
    if (size_)
      check(cudaMemcpy(values.data(), data_, size_ * sizeof(T),
                       cudaMemcpyDeviceToHost));
    return values;
  }

private:
  T *data_ = nullptr;
  std::size_t size_;
};

__global__ void build_windows(plan::SourcePlanView view) {
  extern __shared__ unsigned scratch[];
  for (auto w = std::uint64_t(blockIdx.x); w < view.local_windows();
       w += gridDim.x)
    plan::build_source_window<128U>(view, w, scratch);
}

__global__ void build_packets(plan::SourcePlanView view,
                              plan::ReturnPlanView returning) {
  const auto worker = blockIdx.x * blockDim.x + threadIdx.x;
  const auto workers = gridDim.x * blockDim.x;
  plan::build_source_packet_prefix(view, worker, workers);
  plan::build_return_packet_prefix(view, returning, worker, workers);
}

__global__ void build_directory(plan::SourcePlanView view,
                                plan::DispatchPlanView output,
                                plan::ReturnPlanView returning) {
  if (threadIdx.x == 0U) {
    plan::build_channel_offsets(view.packet_prefix, view.nodes * view.packets.lanes,
                                view.local_windows(), output.channel_offsets);
    plan::build_channel_offsets(returning.packet_prefix,
                                view.nodes * view.packets.lanes,
                                view.local_windows(), returning.channel_offsets);
  }
  __syncthreads();
  plan::build_dispatch_packets(view, output, threadIdx.x, blockDim.x);
  plan::build_return_packets(view, returning, threadIdx.x, blockDim.x);
}

void run(unsigned nodes, unsigned tokens, bool hotspot, bool invalid = false) {
  constexpr unsigned local_world = 8U, top_k = 4U;
  const auto world = nodes * local_world;
  const auto experts = world * 4U;
  const auto source = world - 1U;
  const auto experts_per_node = experts / nodes;
  // Mirror native pools, which clip B to capacity for nonempty short shapes.
  const auto window_tokens = tokens ? std::min(tokens, 128U) : 128U;
  const plan::WindowShape shape{world, local_world, window_tokens, top_k};
  const auto partition = plan::kNetworkPackets;
  const auto return_partition = plan::return_partition(shape, tokens);
  if (nodes == 2U && !hotspot && !invalid && (tokens == 1U || tokens == 64U))
    std::cout << "Return partition: T=" << tokens << " B=" << shape.tokens
              << " Q=" << return_partition.rows << '\n';
  const auto windows = plan::ceil_div(tokens, shape.tokens);
  const auto capacity = windows * shape.tokens * top_k;
  std::vector<std::int32_t> selected(std::size_t(tokens) * top_k);
  std::vector<std::uint8_t> valid(selected.size(), 1U);
  for (unsigned token = 0; token < tokens; ++token) {
    for (unsigned k = 0; k < top_k; ++k) {
      selected[std::size_t(token) * top_k + k] = static_cast<std::int32_t>(
          hotspot ? 0U : (token * 17U + source * 13U + k / 2U) % experts);
      valid[std::size_t(token) * top_k + k] =
          hotspot || !(k == 3U && token % 7U == 0U);
    }
  }
  if (invalid)
    selected[0] = static_cast<std::int32_t>(experts);
  DeviceArray<std::int32_t> d_selected(selected);
  DeviceArray<std::uint8_t> d_valid(valid);
  DeviceArray<std::uint64_t> expert_prefix(windows * (experts + 1U));
  DeviceArray<std::uint64_t> node_prefix(windows * (nodes + 1U));
  DeviceArray<std::uint64_t> route_order(capacity);
  DeviceArray<plan::NodeToken> node_tokens(capacity);
  DeviceArray<std::uint64_t> packet_prefix(nodes * partition.lanes *
                                           (windows + 1U));
  DeviceArray<std::uint64_t> channel_offsets(nodes * partition.lanes + 1U);
  // Every nonempty packet has at least one node-deduplicated token and hence
  // one valid route. This bounds compact directory capacity without per-peer
  // worst-case replication.
  DeviceArray<plan::DispatchPacketPlan> packets(selected.size());
  DeviceArray<std::uint32_t> consumers(selected.size() * local_world);
  const plan::DispatchPlanView directory{channel_offsets.data(), packets.data(),
                                         consumers.data()};
  DeviceArray<std::uint64_t> return_prefix(nodes * partition.lanes *
                                           (windows + 1U));
  DeviceArray<std::uint64_t> return_offsets(nodes * partition.lanes + 1U);
  DeviceArray<plan::ReturnPacketPlan> return_packets(selected.size());
  const plan::ReturnPlanView returning{return_prefix.data(), return_offsets.data(),
                                       return_packets.data(), return_partition};
  DeviceArray<unsigned> error(1U);
  const plan::SourcePlanView view{shape,
                                  partition,
                                  source,
                                  experts,
                                  nodes,
                                  tokens,
                                  d_selected.data(),
                                  hotspot ? nullptr : d_valid.data(),
                                  expert_prefix.data(),
                                  node_prefix.data(),
                                  route_order.data(),
                                  node_tokens.data(),
                                  packet_prefix.data(),
                                  error.data()};

  build_windows<<<7U, 128U, view.scratch_words() * sizeof(unsigned)>>>(view);
  check(cudaGetLastError());
  build_packets<<<1U, 128U>>>(view, returning);
  check(cudaGetLastError());
  build_directory<<<1U, 128U>>>(view, directory, returning);
  check(cudaGetLastError());
  check(cudaDeviceSynchronize());
  require(error.read()[0] == unsigned(invalid), "invalid route status");
  if (invalid)
    return;

  const auto ep = expert_prefix.read();
  const auto np = node_prefix.read();
  const auto order = route_order.read();
  const auto nt = node_tokens.read();
  const auto pp = packet_prefix.read();
  const auto channels = channel_offsets.read();
  const auto packet_plan = packets.read();
  const auto consumer_rows = consumers.read();
  const auto rp = return_prefix.read();
  const auto return_channels = return_offsets.read();
  const auto return_plan = return_packets.read();
  std::vector<std::uint64_t> sequences(nodes * partition.lanes, 0U);
  std::vector<std::uint64_t> return_sequences(nodes * partition.lanes, 0U);
  for (std::uint64_t w = 0; w < windows; ++w) {
    const auto begin = w * shape.tokens;
    const auto end = std::min<std::uint64_t>(begin + shape.tokens, tokens);
    std::vector<std::set<std::uint64_t>> expected_routes(experts);
    std::vector<std::set<std::uint64_t>> expected_tokens(nodes);
    std::vector<unsigned> fanouts((end - begin) * nodes, 0U);
    for (auto token = begin; token < end; ++token) {
      for (unsigned k = 0; k < top_k; ++k) {
        const auto route = token * top_k + k;
        if (!valid[route])
          continue;
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
        require(expected_routes[e].erase(route) == 1U,
                "route coverage/uniqueness");
      }
    }
    for (unsigned node = 0; node < nodes; ++node) {
      const auto first = np[w * (nodes + 1U) + node];
      const auto last = np[w * (nodes + 1U) + node + 1U];
      require(last - first == expected_tokens[node].size(), "node dedup count");
      for (auto index = first; index < last; ++index) {
        const auto record = nt[w * view.window_capacity() + index];
        require(expected_tokens[node].erase(record.token) == 1U,
                "token coverage/uniqueness");
        require(record.fanouts ==
                    fanouts[(record.token - begin) * nodes + node],
                "packet expected fanout count");
      }
      for (unsigned lane = 0; lane < partition.lanes; ++lane) {
        const auto channel = node * partition.lanes + lane;
        const auto base = channel * (windows + 1U);
        require(pp[base + w] == sequences[channel], "packet start sequence");
        if (node != source / local_world)
          sequences[channel] += partition.lane_packets(last - first, lane);
        require(pp[base + w + 1U] == sequences[channel],
                "packet terminal sequence");
        require(rp[base + w] == return_sequences[channel],
                "Return packet start sequence");
        if (node != source / local_world) {
          unsigned contributions = 0U;
          for (auto token = begin; token < end; ++token)
            contributions += fanouts[(token - begin) * nodes + node];
          const auto count = (contributions + return_partition.rows - 1U) /
                             return_partition.rows;
          return_sequences[channel] += count > lane
              ? (count - lane + partition.lanes - 1U) / partition.lanes : 0U;
        }
        require(rp[base + w + 1U] == return_sequences[channel],
                "Return packet terminal sequence");
      }
    }
  }
  if (tokens == 0U)
    for (auto value : pp)
      require(value == 0U, "empty source channel");

  std::vector<std::set<std::uint64_t>> delivered(nodes);
  for (unsigned channel = 0; channel < nodes * partition.lanes; ++channel) {
    const auto node = channel / partition.lanes;
    require(channels[channel + 1U] - channels[channel] == sequences[channel],
            "compact channel directory extent");
    for (auto packet = channels[channel]; packet < channels[channel + 1U];
         ++packet) {
      const auto &desc = packet_plan[packet];
      require(desc.tokens > 0U && desc.tokens <= partition.rows,
              "packet tail size");
      std::array<unsigned, local_world> expected{};
      unsigned fanouts = 0U;
      for (unsigned row = 0; row < desc.tokens; ++row) {
        const auto token = nt[desc.token_begin + row].token;
        require(delivered[node].insert(token).second,
                "packet token uniqueness");
        require(desc.window == shape.window(source, token),
                "packet global window");
        for (unsigned k = 0; k < top_k; ++k) {
          const auto route = token * top_k + k;
          const auto expert = unsigned(selected[route]);
          if (valid[route] && expert / experts_per_node == node) {
            ++expected[(expert / (experts / world)) % local_world];
            ++fanouts;
          }
        }
      }
      require(fanouts == desc.fanouts, "packet retirement units");
      for (unsigned local = 0; local < local_world; ++local)
        require(consumer_rows[packet * local_world + local] == expected[local],
                "immutable packet consumer directory");
    }
  }
  for (unsigned token = 0; token < tokens; ++token)
    for (unsigned k = 0; k < top_k; ++k) {
      const auto route = std::size_t(token) * top_k + k;
      const auto node = unsigned(selected[route]) / experts_per_node;
      if (valid[route] && node != source / local_world)
        require(delivered[node].count(token) == 1U, "packet no-drop coverage");
    }

  std::vector<unsigned> returned(selected.size(), 0U);
  for (unsigned channel = 0; channel < nodes * partition.lanes; ++channel) {
    const auto node = channel / partition.lanes;
    require(return_channels[channel + 1U] - return_channels[channel] ==
                return_sequences[channel], "Return channel extent");
    for (auto packet = return_channels[channel];
         packet < return_channels[channel + 1U]; ++packet) {
      const auto &desc = return_plan[packet];
      require(desc.rows > 0U && desc.rows <= return_partition.rows, "Return packet tail");
      for (unsigned row = 0U; row < desc.rows; ++row) {
        const auto route = order[desc.route_begin + row];
        require(valid[route] && unsigned(selected[route]) / experts_per_node == node,
                "Return packet node/validity");
        require(desc.window == shape.window(source, route / top_k),
                "Return packet window");
        require(++returned[route] == 1U, "Return route uniqueness");
      }
    }
  }
  for (std::size_t route = 0U; route < selected.size(); ++route)
    require(returned[route] == unsigned(valid[route] &&
                unsigned(selected[route]) / experts_per_node != source / local_world),
            "Return route no-drop coverage");
}

struct SourceOrdinal {
  unsigned expert, source, ordinal;
};

__global__ void build_geometry(plan::WindowGeometryView view,
                               const SourceOrdinal *routes, unsigned count,
                               std::uint64_t *rows) {
  plan::build_window_geometry<128U>(view);
  for (unsigned r = threadIdx.x; r < count; r += blockDim.x)
    rows[r] =
        view.expert_row(routes[r].expert, routes[r].source, routes[r].ordinal);
}

void run_geometry(unsigned nodes, bool hotspot) {
  constexpr unsigned owned = 4U;
  const plan::WindowShape shape{nodes * 8U, nodes == 3U ? 16U : 8U, 128U, 4U};
  // Two nodes use node-aligned groups; three retain a partial G-sized group.
  const auto window = shape.groups() - 1U;
  const auto first_source = shape.source_begin(window);
  const auto sources = shape.source_end(window) - first_source;
  std::vector<std::uint32_t> counts(owned * shape.sources, 0U);
  for (unsigned source = 0; source < sources; ++source) {
    const auto tokens = source == 0U ? 0U : shape.tokens - source;
    for (unsigned token = 0; token < tokens; ++token)
      for (unsigned k = 0; k < shape.top_k; ++k)
        ++counts[(hotspot ? 0U : (source + token + k / 2U) % owned) *
                     shape.sources +
                 source];
  }
  // Independent round-robin oracle, including unequal/empty source columns.
  std::vector<SourceOrdinal> routes;
  std::vector<std::uint64_t> expected;
  std::array<unsigned, owned> totals{};
  for (unsigned expert = 0; expert < owned; ++expert) {
    for (unsigned ordinal = 0; ordinal < shape.tokens * shape.top_k; ++ordinal)
      for (unsigned source = 0; source < sources; ++source)
        if (ordinal < counts[expert * shape.sources + source]) {
          routes.push_back({expert, first_source + source, ordinal});
          expected.push_back(totals[expert]++);
        }
  }
  DeviceArray<std::uint32_t> d_counts(counts);
  DeviceArray<std::uint64_t> expert_rows(owned), prefix(owned + 1U);
  DeviceArray<plan::WindowMblock> blocks(shape.max_mblocks(owned));
  DeviceArray<SourceOrdinal> d_routes(routes);
  DeviceArray<std::uint64_t> assigned(routes.size());
  const plan::WindowGeometryView view{
      shape,         window,       owned, d_counts.data(), expert_rows.data(),
      prefix.data(), blocks.data()};
  build_geometry<<<1U, 128U>>>(view, d_routes.data(), unsigned(routes.size()),
                               assigned.data());
  check(cudaGetLastError());
  check(cudaDeviceSynchronize());
  require(assigned.read() == expected, "round-robin source ordinal inverse");
  const auto actual_rows = expert_rows.read();
  const auto actual_prefix = prefix.read();
  const auto actual_blocks = blocks.read();
  std::uint64_t next_block = 0U;
  for (unsigned expert = 0; expert < owned; ++expert) {
    require(actual_rows[expert] == totals[expert], "destination window counts");
    require(actual_prefix[expert] == next_block, "window mblock prefix");
    unsigned retired = 0U;
    while (retired < totals[expert]) {
      const auto &block = actual_blocks[next_block++];
      const auto rows = std::min(shape.tile_m, totals[expert] - retired);
      require(block.expert == expert && block.expert_row == retired &&
                  block.rows == rows,
              "window block and C retirement rows");
      retired += rows;
    }
  }
  require(actual_prefix[owned] == next_block &&
              next_block <= actual_blocks.size(),
          "window no-drop capacity bound");
}

} // namespace

int main() {
  try {
    for (unsigned nodes : {1U, 2U, 3U}) {
      for (unsigned tokens : {0U, 1U, 64U, 4097U, 8192U, 20480U}) {
        run(nodes, tokens, false);
        run(nodes, tokens, true);
      }
    }
    run(3U, 64U, false, true);
    for (unsigned nodes : {1U, 2U, 3U}) {
      run_geometry(nodes, false);
      run_geometry(nodes, true);
    }
    std::cout << "static V4 GPU plan: 37 source/packet + 6 destination cases "
                 "passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
