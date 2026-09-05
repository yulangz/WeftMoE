void test_bootstrap_exact_frame_io() {
  std::vector<std::uint8_t> frame(
      megamoe::cuda::NcclBootstrapInputChannel::kWireBytes);
  for (std::size_t index = 0; index < frame.size(); ++index)
    frame[index] = static_cast<std::uint8_t>((index * 37U + 11U) & 0xffU);

  const auto read_valid = [&](std::size_t chunk, bool interrupted) {
    BootstrapIoScript script;
    script.input = frame;
    script.read_chunk = chunk;
    script.read_eintr = interrupted;
    BootstrapIoHookGuard guard(script);
    megamoe::cuda::NcclBootstrapInputChannel channel(7);
    auto result = channel.read();
    REQUIRE(static_cast<bool>(result));
    REQUIRE(script.input_offset == frame.size());
    REQUIRE(script.close_count == 1);
    REQUIRE(channel.state() ==
            megamoe::cuda::BootstrapChannelState::Consumed);
    expect_error(NotifyErrorCode::NcclFailure,
                 [&] { (void)channel.read(); },
                 {"field=read-state", "expected=state=Open",
                  "actual=state=Consumed"});
    REQUIRE(script.close_count == 1);
    return result;
  };

  auto bootstrap = read_valid(frame.size(), false);
  (void)read_valid(1, false);
  (void)read_valid(17, true);
  (void)read_valid(127, false);

  for (const auto short_size : {std::size_t{0}, std::size_t{1},
                                std::size_t{127}}) {
    BootstrapIoScript script;
    script.input.assign(frame.begin(), frame.begin() + short_size);
    script.read_chunk = 17;
    BootstrapIoHookGuard guard(script);
    megamoe::cuda::NcclBootstrapInputChannel channel(7);
    expect_error(NotifyErrorCode::NcclFailure,
                 [&] { (void)channel.read(); },
                 {"phase=bootstrap-channel-read", "field=bytes",
                  "expected=128", "actual=" + std::to_string(short_size)});
    REQUIRE(channel.state() == megamoe::cuda::BootstrapChannelState::Failed);
    REQUIRE(script.close_count == 1);
  }

  for (const auto oversized : {std::size_t{129}, std::size_t{257}}) {
    BootstrapIoScript script;
    script.input.resize(oversized, 0x5aU);
    script.read_chunk = 17;
    BootstrapIoHookGuard guard(script);
    megamoe::cuda::NcclBootstrapInputChannel channel(7);
    expect_error(NotifyErrorCode::NcclFailure,
                 [&] { (void)channel.read(); },
                 {"phase=bootstrap-channel-read", "field=frame-length",
                  "expected=128", "actual=>=129"});
    REQUIRE(script.input_offset == 129);
    REQUIRE(channel.state() == megamoe::cuda::BootstrapChannelState::Failed);
    expect_error(NotifyErrorCode::NcclFailure,
                 [&] { (void)channel.read(); },
                 {"field=read-state", "actual=state=Failed"});
    REQUIRE(script.input_offset == 129);
    REQUIRE(script.close_count == 1);
  }

  for (const auto chunk : {std::size_t{1}, std::size_t{17},
                           std::size_t{127}}) {
    BootstrapIoScript script;
    script.write_chunk = chunk;
    BootstrapIoHookGuard guard(script);
    megamoe::cuda::NcclBootstrapOutputChannel channel(8);
    channel.write(bootstrap);
    REQUIRE(script.output == frame);
    REQUIRE(script.close_count == 1);
    REQUIRE(channel.state() ==
            megamoe::cuda::BootstrapChannelState::Consumed);
  }

  {
    BootstrapIoScript script;
    script.write_chunk = 17;
    script.write_eintr = true;
    BootstrapIoHookGuard guard(script);
    megamoe::cuda::NcclBootstrapOutputChannel channel(8);
    channel.write(bootstrap);
    REQUIRE(script.output == frame);
    REQUIRE(script.close_count == 1);
  }
  {
    BootstrapIoScript script;
    script.write_zero = true;
    BootstrapIoHookGuard guard(script);
    megamoe::cuda::NcclBootstrapOutputChannel channel(8);
    expect_error(NotifyErrorCode::NcclFailure,
                 [&] { channel.write(bootstrap); },
                 {"phase=bootstrap-channel-write", "field=write-progress",
                  "expected=>0", "actual=0"});
    REQUIRE(channel.state() == megamoe::cuda::BootstrapChannelState::Failed);
    REQUIRE(script.close_count == 1);
  }
  {
    BootstrapIoScript script;
    script.write_broken_pipe = true;
    BootstrapIoHookGuard guard(script);
    megamoe::cuda::NcclBootstrapOutputChannel channel(8);
    expect_error(NotifyErrorCode::NcclFailure,
                 [&] { channel.write(bootstrap); },
                 {"phase=bootstrap-channel-write", "field=write",
                  "actual=I/O failure"});
    REQUIRE(channel.state() == megamoe::cuda::BootstrapChannelState::Failed);
    REQUIRE(script.close_count == 1);
  }
  {
    BootstrapIoScript script;
    script.input = frame;
    BootstrapIoHookGuard guard(script);
    megamoe::cuda::NcclBootstrapInputChannel source(7);
    megamoe::cuda::NcclBootstrapInputChannel moved(std::move(source));
    REQUIRE(source.state() == megamoe::cuda::BootstrapChannelState::Closed);
    REQUIRE(source.native_fd_for_testing() == -1);
    expect_error(NotifyErrorCode::NcclFailure,
                 [&] { (void)source.read(); }, {"actual=state=Closed"});
    auto *self_alias = &moved;
    moved = std::move(*self_alias);
    (void)moved.read();
    REQUIRE(script.close_count == 1);
  }
  {
    BootstrapIoScript script;
    BootstrapIoHookGuard guard(script);
    {
      megamoe::cuda::NcclBootstrapOutputChannel channel(8);
      channel.cancel();
      channel.cancel();
      REQUIRE(channel.state() == megamoe::cuda::BootstrapChannelState::Closed);
    }
    REQUIRE(script.close_count == 1);
  }
  {
    BootstrapIoScript script;
    BootstrapIoHookGuard guard(script);
    {
      megamoe::cuda::NcclBootstrapOutputChannel source(8);
      megamoe::cuda::NcclBootstrapOutputChannel moved(std::move(source));
      REQUIRE(source.state() == megamoe::cuda::BootstrapChannelState::Closed);
      REQUIRE(source.native_fd_for_testing() == -1);
      moved.write(bootstrap);
    }
    REQUIRE(script.output == frame);
    REQUIRE(script.close_count == 1);
  }
  {
    BootstrapIoScript script;
    BootstrapIoHookGuard guard(script);
    megamoe::cuda::NcclBootstrapInputChannel source(7);
    megamoe::cuda::NcclBootstrapInputChannel destination(9);
    destination = std::move(source);
    REQUIRE(source.state() == megamoe::cuda::BootstrapChannelState::Closed);
    REQUIRE(destination.native_fd_for_testing() == 7);
    REQUIRE(script.close_count == 1);
    REQUIRE(script.closed_fds == std::vector<int>{9});
    destination.cancel();
    REQUIRE(script.close_count == 2);
    REQUIRE(script.closed_fds == (std::vector<int>{9, 7}));
  }
  {
    BootstrapIoScript script;
    BootstrapIoHookGuard guard(script);
    { megamoe::cuda::NcclBootstrapInputChannel abandoned(7); }
    REQUIRE(script.close_count == 1);
    REQUIRE(script.closed_fds == std::vector<int>{7});
  }
}

void write_pipe_frame(int fd, const std::vector<std::uint8_t> &bytes) {
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    const auto result = ::write(fd, bytes.data() + offset,
                                bytes.size() - offset);
    if (result < 0 && errno == EINTR)
      continue;
    if (result <= 0)
      throw std::runtime_error("pipe fixture write failed");
    offset += static_cast<std::size_t>(result);
  }
  if (::close(fd) != 0)
    throw std::runtime_error("pipe fixture close failed");
}

void test_bootstrap_real_pipe_one_shot() {
  std::vector<std::uint8_t> frame(
      megamoe::cuda::NcclBootstrapInputChannel::kWireBytes);
  for (std::size_t index = 0; index < frame.size(); ++index)
    frame[index] = static_cast<std::uint8_t>((index * 19U + 5U) & 0xffU);

  for (const auto size : {std::size_t{0}, std::size_t{1}, std::size_t{127},
                          std::size_t{128}, std::size_t{129},
                          std::size_t{257}}) {
    int descriptors[2] = {-1, -1};
    REQUIRE(::pipe(descriptors) == 0);
    std::vector<std::uint8_t> bytes(size, 0x6dU);
    if (size == 128)
      bytes = frame;
    write_pipe_frame(descriptors[1], bytes);
    const int original_read_fd = descriptors[0];
    megamoe::cuda::NcclBootstrapInputChannel channel(original_read_fd);
    bool first_rejected = false;
    if (size == 128) {
      auto id = channel.read();
      REQUIRE(static_cast<bool>(id));
      REQUIRE(channel.state() ==
              megamoe::cuda::BootstrapChannelState::Consumed);
    } else {
      try {
        (void)channel.read();
      } catch (const NotifyDispatchError &) {
        first_rejected = true;
      }
      REQUIRE(first_rejected);
      REQUIRE(channel.state() ==
              megamoe::cuda::BootstrapChannelState::Failed);
    }
    errno = 0;
    std::uint8_t probe = 0;
    REQUIRE(::read(original_read_fd, &probe, 1) == -1);
    REQUIRE(errno == EBADF);
    expect_error(NotifyErrorCode::NcclFailure,
                 [&] { (void)channel.read(); },
                 {"field=read-state", "expected=state=Open"});
    if (size == 257) {
      std::cout << "first_oversize_rejected = " << (first_rejected ? 1 : 0)
                << "\nremaining_128_accepted_as_next_frame = 0\n"
                << "channel_terminal = "
                << (channel.state() ==
                            megamoe::cuda::BootstrapChannelState::Failed
                        ? 1
                        : 0)
                << '\n';
    }
  }

  // A reader rejects at byte 129 and closes immediately; it never drains an
  // attacker that keeps the write end open.
  struct sigaction ignore_action {};
  struct sigaction old_action {};
  ignore_action.sa_handler = SIG_IGN;
  REQUIRE(::sigemptyset(&ignore_action.sa_mask) == 0);
  REQUIRE(::sigaction(SIGPIPE, &ignore_action, &old_action) == 0);
  int streaming[2] = {-1, -1};
  REQUIRE(::pipe(streaming) == 0);
  std::atomic<bool> saw_epipe{false};
  std::thread writer([&] {
    std::array<std::uint8_t, 256> bytes{};
    while (true) {
      const auto result = ::write(streaming[1], bytes.data(), bytes.size());
      if (result < 0 && errno == EINTR)
        continue;
      if (result < 0 && errno == EPIPE) {
        saw_epipe = true;
        break;
      }
      if (result <= 0)
        break;
    }
    (void)::close(streaming[1]);
  });
  megamoe::cuda::NcclBootstrapInputChannel streaming_channel(streaming[0]);
  expect_error(NotifyErrorCode::NcclFailure,
               [&] { (void)streaming_channel.read(); },
               {"field=frame-length", "actual=>=129"});
  writer.join();
  REQUIRE(saw_epipe.load());
  REQUIRE(::sigaction(SIGPIPE, &old_action, nullptr) == 0);

  // Obtain an opaque ID through the input channel, then prove that output is
  // exact, closes on completion, and cannot be reused.
  int source_pipe[2] = {-1, -1};
  REQUIRE(::pipe(source_pipe) == 0);
  write_pipe_frame(source_pipe[1], frame);
  megamoe::cuda::NcclBootstrapInputChannel source(source_pipe[0]);
  auto id = source.read();
  int output_pipe[2] = {-1, -1};
  REQUIRE(::pipe(output_pipe) == 0);
  const int original_write_fd = output_pipe[1];
  megamoe::cuda::NcclBootstrapOutputChannel output(original_write_fd);
  output.write(id);
  REQUIRE(output.state() == megamoe::cuda::BootstrapChannelState::Consumed);
  errno = 0;
  REQUIRE(::write(original_write_fd, frame.data(), 1) == -1);
  REQUIRE(errno == EBADF);
  std::vector<std::uint8_t> received(frame.size());
  std::size_t offset = 0;
  while (offset < received.size()) {
    const auto count =
        ::read(output_pipe[0], received.data() + offset,
               received.size() - offset);
    REQUIRE(count > 0);
    offset += static_cast<std::size_t>(count);
  }
  std::uint8_t trailing = 0;
  REQUIRE(::read(output_pipe[0], &trailing, 1) == 0);
  REQUIRE(::close(output_pipe[0]) == 0);
  REQUIRE(received == frame);
  expect_error(NotifyErrorCode::NcclFailure,
               [&] { output.write(id); },
               {"field=write-state", "actual=state=Consumed"});

  // Once a terminal channel has relinquished its number, it cannot touch a
  // later object that the OS assigns that same descriptor number.
  int stale_pipe[2] = {-1, -1};
  REQUIRE(::pipe(stale_pipe) == 0);
  std::vector<std::uint8_t> oversized(129, 0xaaU);
  write_pipe_frame(stale_pipe[1], oversized);
  const int stale_number = stale_pipe[0];
  megamoe::cuda::NcclBootstrapInputChannel stale(stale_number);
  expect_error(NotifyErrorCode::NcclFailure,
               [&] { (void)stale.read(); }, {"actual=>=129"});
  int replacement[2] = {-1, -1};
  REQUIRE(::pipe(replacement) == 0);
  if (replacement[0] != stale_number) {
    REQUIRE(::dup2(replacement[0], stale_number) == stale_number);
    REQUIRE(::close(replacement[0]) == 0);
    replacement[0] = stale_number;
  }
  const std::uint8_t marker = 0x4cU;
  REQUIRE(::write(replacement[1], &marker, 1) == 1);
  REQUIRE(::close(replacement[1]) == 0);
  expect_error(NotifyErrorCode::NcclFailure,
               [&] { (void)stale.read(); }, {"actual=state=Failed"});
  std::uint8_t observed = 0;
  REQUIRE(::read(replacement[0], &observed, 1) == 1);
  REQUIRE(observed == marker);
  REQUIRE(::close(replacement[0]) == 0);
}
