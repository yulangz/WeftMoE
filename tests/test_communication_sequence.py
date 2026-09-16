"""Completion counters: wraparound and independently completing TX slots."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CompletionSequenceTest(unittest.TestCase):
    def test_wrap_and_out_of_order_slots(self):
        compiler = shutil.which('c++')
        if compiler is None:
            self.skipTest('C++ compiler unavailable')
        program = r'''
#include "cuda/communication/sequence.hpp"
#include "cuda/communication/row_ready.hpp"
#include <cassert>
#include <cstdint>
using weftmoe::cuda::communication::reached;
using weftmoe::cuda::communication::row_ready_mask;
using weftmoe::cuda::communication::row_ready_prefix;
int main() {
  for (unsigned first = 0; first < 32; ++first)
    for (unsigned count = 0; count <= 32 - first; ++count) {
      std::uint32_t expected = 0;
      for (unsigned i = first; i < first + count; ++i) expected |= 1U << i;
      assert(row_ready_mask(first, count) == expected);
    }
  for (std::uint32_t bits = 0; bits < 65536; ++bits)
    for (unsigned upper = 0; upper < 2; ++upper) {
      auto word = bits | (upper ? 0xffff0000U : 0U);
      for (unsigned first = 0; first < 32; ++first) {
        unsigned expected = 0;
        while (first + expected < 32 && (word & (1U << (first + expected))))
          ++expected;
        for (unsigned limit = 0; limit <= 32 - first; ++limit)
          assert(row_ready_prefix(word, first, limit) ==
                 (expected < limit ? expected : limit));
      }
    }
  auto ready = row_ready_mask(12, 20); // C finishes first; A/B still missing.
  assert(row_ready_prefix(ready, 0, 32) == 0);
  ready |= row_ready_mask(0, 5);
  assert(row_ready_prefix(ready, 0, 32) == 5);
  ready |= row_ready_mask(5, 7);
  assert(row_ready_prefix(ready, 0, 32) == 32);
  // Exhaust all representable values in a small ring, including the boundary.
  for (unsigned base = 0; base < 256; ++base) {
    for (unsigned n = 1; n < 128; ++n) {
      auto goal = base + n;
      assert(!reached(base, goal, 8));
      for (unsigned done = 0; done <= n; ++done)
        assert(reached((base + done) & 255U, goal, 8) == (done == n));
    }
  }
  constexpr std::uint64_t mask = (std::uint64_t{1} << 56) - 1;
  std::uint64_t completed[3] = {mask, mask, mask};
  std::uint64_t target[3] = {mask + 1, mask + 1, mask + 1};
  // B completes before A: B's increment must never authorize reusing A.
  completed[1] = 0;
  assert(!reached(completed[0], target[0]));
  assert(reached(completed[1], target[1]));
  assert(!reached(completed[2], target[2]));
  completed[0] = 0;
  assert(reached(completed[0], target[0]));
  // Two PUTs reading the same slot: first completion alone is insufficient.
  target[0] += 2;
  completed[0] = 1;
  assert(!reached(completed[0], target[0]));
  completed[0] = 2;
  assert(reached(completed[0], target[0]));
  assert(reached(0, UINT64_MAX, 64));
  assert(!reached(UINT64_MAX, 0, 64));
}
'''
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp, 'sequence.cpp')
            binary = Path(temp, 'sequence')
            source.write_text(program)
            subprocess.run([compiler, '-std=c++17', '-O2', '-I', str(ROOT / 'src'),
                            str(source), '-o', str(binary)], check=True, timeout=20)
            subprocess.run([str(binary)], check=True, timeout=20)


if __name__ == '__main__':
    unittest.main()
