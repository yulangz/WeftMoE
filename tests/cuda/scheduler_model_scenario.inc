void print_nccl_runtime_info() {
  const auto info = megamoe::cuda::nccl_runtime_info();
  std::cout << "NCCL_RUNTIME compiled=" << info.compiled_version
            << " runtime=" << info.runtime_version
            << " library=" << info.loaded_library_path
            << " NCCL_SOCKET_IFNAME=" << info.socket_ifname
            << " NCCL_IB_DISABLE=" << info.ib_disable << '\n';
}

std::uint64_t parse_u64(const char *value) {
  return static_cast<std::uint64_t>(std::stoull(value));
}

void test_randomized_generation_scheduler() {
  using megamoe::cuda::testing::GenerationSchedulerModel;
  using megamoe::cuda::transport_state::TaskClass;
  constexpr auto kinds = megamoe::cuda::transport_state::kTaskClassCount;
  constexpr std::uint64_t seed_count = 256;
  for (std::uint64_t seed = 1; seed <= seed_count; ++seed) {
    auto random = seed * 0x9e3779b97f4a7c15ULL;
    auto next = [&random]() {
      random ^= random >> 12U;
      random ^= random << 25U;
      random ^= random >> 27U;
      return random * 0x2545f4914f6cdd1dULL;
    };
    std::array<std::uint32_t, kinds> expected{};
    for (auto &count : expected)
      count = static_cast<std::uint32_t>(next() % 17U);
    const auto contexts = 1U + static_cast<std::uint32_t>(next() % 8U);
    GenerationSchedulerModel model(expected, contexts, 3U);
    for (std::uint32_t context = 0; context < contexts; ++context)
      model.publish_directory(context);

    std::array<std::uint32_t, kinds> next_task{};
    std::vector<std::array<std::uint32_t, 3>> inflight;
    std::uint64_t steps = 0;
    while (!model.terminal_ready()) {
      require(++steps < 100000U,
              "randomized generation scheduler did not terminate");
      if (!inflight.empty() && (next() & 1U) != 0U) {
        const auto choice = static_cast<std::size_t>(next() % inflight.size());
        const auto item = inflight[choice];
        require(model.complete(static_cast<TaskClass>(item[0]), item[1],
                               item[2]),
                "claimed task failed completion");
        inflight.erase(inflight.begin() + static_cast<std::ptrdiff_t>(choice));
        continue;
      }
      const auto kind = static_cast<std::uint32_t>(next() % kinds);
      if (next_task[kind] >= expected[kind])
        continue;
      const auto context = static_cast<std::uint32_t>(next() % contexts);
      if (model.claim(static_cast<TaskClass>(kind), next_task[kind], context)) {
        inflight.push_back(
            std::array<std::uint32_t, 3>{kind, next_task[kind], context});
        ++next_task[kind];
      }
    }
    require(inflight.empty(), "terminal published with inflight work");
    require(model.publish_terminal_once(),
            "elected terminal checker failed first publication");
    require(!model.publish_terminal_once() &&
                model.terminal_publish_count() == 1U,
            "terminal was published more than once");
    for (std::uint32_t context = 0; context < contexts; ++context)
      require(model.context_credit(context) == 3U,
              "context credit was not restored");
  }
  std::cout << "[PASS] randomized static generation scheduler seeds="
            << seed_count << '\n';
}
