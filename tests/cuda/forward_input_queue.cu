#include <cassert>
#include <cstdio>
#include <deque>
#include <initializer_list>
#include <vector>

#include "cuda/persistent_v3/forward/pipeline_state.cuh"

namespace v3 = weftmoe::cuda::persistent_v3;

int main() {
  unsigned partitions = 0;
  for (unsigned blocks = 0; blocks <= 65; ++blocks)
    for (unsigned startup = 0; startup <= 6; ++startup)
      for (unsigned locality : {1U, 2U, 4U})
        for (unsigned begin : {0U, 3U, 4U, 7U})
          for (unsigned end : {0U, 5U, 12U, 15U}) {
            unsigned next = 0;
            while (next < blocks) {
              const auto span = v3::input_group_span(
                  next, blocks, startup, locality, begin, end);
              assert(span.begin == next && span.count > 0);
              assert(span.begin + span.count <= blocks);
              for (unsigned member = next; member < next + span.count;
                   ++member) {
                const auto same = v3::input_group_span(
                    member, blocks, startup, locality, begin, end);
                assert(same.begin == span.begin && same.count == span.count);
              }
              next += span.count;
            }
            assert(v3::input_group_span(blocks, blocks, startup, locality,
                                       begin, end)
                       .count == 0);
            ++partitions;
          }

  // Compare the production circular list with an independent FIFO. Sparse
  // group IDs, arbitrary readiness order and repeated resets exercise removal
  // of the first, middle and final entry without leaving a published group in
  // the pending queue.
  unsigned executions = 0;
  for (unsigned capacity : {0U, 1U, 2U, 31U, 64U, 257U})
    for (unsigned execution = 0; execution < 100; ++execution) {
      std::vector<v3::ForwardInputGroup> groups(capacity * 3U);
      v3::ForwardInputPendingQueue queue;
      std::deque<unsigned long long> expected;
      for (unsigned i = 0; i < capacity; ++i) {
        const auto id = 3ULL * i + execution % 3U;
        queue.push(groups.data(), id);
        expected.push_back(id);
      }
      unsigned random = execution + 1U;
      while (!expected.empty()) {
        assert(!queue.empty());
        assert(queue.head == expected.front() && queue.tail == expected.back());
        random = random * 1664525U + 1013904223U;
        const bool ready = (random & 7U) < 3U;
        const auto id = expected.front();
        expected.pop_front();
        if (!ready)
          expected.push_back(id);
        queue.advance(groups.data(), ready);
      }
      assert(queue.empty() && queue.tail == v3::kNoInputGroup);

      ++executions;
    }
  std::printf("PASS local input queue: %u partitions, %u reset/rotation cases\n",
              partitions, executions);
}
