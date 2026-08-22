import unittest

from reference.notify_dispatch_reference import (
    ReferenceNotifyError,
    ReferenceTopology,
    build_reference,
)


class NotifyReferenceTest(unittest.TestCase):
    def setUp(self):
        self.topology = ReferenceTopology([0, 0], [0, 1], [0, 0, 1, 1], 1)
        self.routes = [
            [[0, 1], [0, 2], [0, 2]],
            [[0, 1], [0, 2]],
        ]

    def test_exact_prefix_and_partial_tile(self):
        result = build_reference(self.routes, self.topology, 4, 2, 4)
        self.assertEqual(result.expert_row_count, [[5, 2], [3, 0]])
        self.assertEqual(result.expert_row_prefix, [[0, 5, 7], [0, 3, 3]])
        self.assertEqual(result.mtile_expected_rows[0][0], [4, 1])
        self.assertEqual(result.mtile_expected_rows[1][1], [])
        self.assertEqual([result.route_dst_row[0][i] for i in (0, 2, 4)], [0, 1, 2])
        self.assertEqual([result.route_dst_row[1][i] for i in (0, 2)], [3, 4])

    def test_arrival_order_is_irrelevant(self):
        ordered = build_reference(self.routes, self.topology, 4, 2, 4, arrival_order=[0, 1])
        reversed_order = build_reference(self.routes, self.topology, 4, 2, 4, arrival_order=[1, 0])
        self.assertEqual(ordered, reversed_order)

    def test_duplicate_and_capacity_fail(self):
        with self.assertRaises(ReferenceNotifyError):
            build_reference([[[1, 1]], []], self.topology, 4, 2, 4)
        with self.assertRaises(ReferenceNotifyError):
            build_reference(self.routes, self.topology, 4, 2, 4,
                            expert_capacity=[4, 10, 10, 10])


if __name__ == "__main__":
    unittest.main()
