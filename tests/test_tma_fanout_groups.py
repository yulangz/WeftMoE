"""Check shared-buffer lifetime with per-thread bulk-group completion."""

import itertools
import unittest


class FanoutGroupLifetimeTest(unittest.TestCase):
    def exercise(self, fanouts, lanes, depth, segments, commit_empty=True):
        # A group records the SMEM slots this member can still be reading.
        # Even an empty group occupies a position in PTX's group-age order.
        groups = [[] for _ in range(lanes)]
        queued = []

        def wait(member, newest):
            groups[member] = groups[member][-newest:] if newest else []

        def retire(slot, count):
            for member in range(lanes):
                reads = {slot} if member < count else set()
                if reads or commit_empty:
                    groups[member].append(reads)

        for count in fanouts:
            if not count:
                continue
            for _ in range(segments):
                slot = len(queued)
                for member in range(lanes):
                    wait(member, depth - 1 - slot)
                # The group barrier imports every member's read completion
                # before the elected loader overwrites this staging slot.
                if any(slot in reads for pending in groups for reads in pending):
                    return False
                queued.append(count)
                if len(queued) == depth:
                    for index, n in enumerate(queued):
                        retire(index, n)
                    queued.clear()
        for slot, count in enumerate(queued):
            retire(slot, count)
        # Read completion permits reuse; the caller still waits for global
        # completion before publishing readiness or returning a source credit.
        for member in range(lanes):
            wait(member, 0)
        return not any(groups)

    def test_empty_members_partial_batches_and_segmented_rows(self):
        for lanes in (1, 2, 4, 8):
            for depth in (1, 2):
                for segments in (1, 2, 5):
                    for fanouts in itertools.product((0, 1, lanes, lanes + 3), repeat=4):
                        self.assertTrue(self.exercise(fanouts, lanes, depth, segments))

    def test_omitting_empty_commits_can_release_the_wrong_slot(self):
        self.assertFalse(self.exercise((2, 1, 2), 2, 2, 1, commit_empty=False))
        self.assertTrue(self.exercise((2, 1, 2), 2, 2, 1))


if __name__ == "__main__":
    unittest.main()
