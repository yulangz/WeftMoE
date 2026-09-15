"""Reject resource configurations that can deadlock setmaxnreg or overrun SMEM."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import unified_tuning as tune
from autotune_common import PlanError


class UnifiedTuningTest(unittest.TestCase):
    def test_rdma_role_count_is_shape_independent(self):
        for count in (0, 1, 8, 16, 32, 256):
            tune.validate(dict(tune.DEFAULTS, unified_rdma_ctas=count))
        with self.assertRaisesRegex(PlanError, "RDMA CTA count"):
            tune.validate(dict(tune.DEFAULTS, unified_rdma_ctas=-1))

    def test_register_pool_is_rounded_per_warp(self):
        # 65536 registers fit on the SM, but a 384-thread CTA initially gets
        # only 64512. Both math warpgroups requesting 224 would stall here.
        with self.assertRaisesRegex(PlanError, "initial CTA pool 64512"):
            tune.validate(dict(tune.DEFAULTS, unified_math_registers=224,
                               unified_control_registers=64))
        tune.validate(dict(tune.DEFAULTS, unified_math_registers=216,
                           unified_control_registers=72))

    def test_large_warpgroup_layout_requires_compilable_atom(self):
        point = dict(tune.DEFAULTS, unified_workers=1, unified_threads=512,
                     unified_comm_base=384, unified_comm_threads=128,
                     unified_math_registers=208, unified_control_registers=48)
        with self.assertRaisesRegex(PlanError, "N128 WGMMA"):
            tune.validate(point)
        tune.validate(dict(point, wgmma_n256_atom_n=128))

    def test_setmaxnreg_direction_is_required_even_when_total_fits(self):
        point = dict(tune.DEFAULTS, unified_workers=1, unified_threads=640,
                     unified_comm_base=384, unified_comm_threads=256,
                     unified_math_registers=80, unified_control_registers=104,
                     wgmma_n256_atom_n=128)
        # 60416 < the initial 61440 pool, but .inc80 cannot start at 96.
        with self.assertRaisesRegex(PlanError, r"setmaxnreg.inc.*initial 96"):
            tune.validate(point)
        # A low producer quota makes the total fit, but .dec104 is still
        # invalid for the dedicated communication warpgroups starting at 96.
        with self.assertRaisesRegex(PlanError, r"setmaxnreg.dec.*initial 96"):
            tune.validate(dict(point, unified_math_registers=96,
                               unified_producer_registers=24))
        tune.validate(dict(point, unified_math_registers=144,
                           unified_control_registers=64))

    def test_communication_domain_cannot_cut_a_warp(self):
        with self.assertRaisesRegex(PlanError, "complete warps"):
            tune.validate(dict(tune.DEFAULTS, unified_comm_base=289,
                               unified_comm_threads=64))

    def test_direction_split_needs_two_copy_warps(self):
        with self.assertRaisesRegex(PlanError, "two copy warps"):
            tune.validate(dict(tune.DEFAULTS, unified_split_nvlink=1,
                               unified_comm_threads=64))

    def test_prefetch_controls_keep_transport_domain_bounded(self):
        for key in ("unified_row_prefetch", "unified_optimize_gin_local_transport"):
            with self.subTest(key=key), self.assertRaises(PlanError):
                tune.validate(dict(tune.DEFAULTS, **{key: 2}))
        for rows in (0, 31, 33, 1056):
            with self.subTest(rows=rows), self.assertRaisesRegex(PlanError, "Return drain rows"):
                tune.validate(dict(tune.DEFAULTS, unified_return_drain_rows=rows))
        for rows in (32, 128, 1024):
            tune.validate(dict(tune.DEFAULTS, unified_return_drain_rows=rows))

    def test_dedicated_communication_warpgroup_has_its_own_quota(self):
        tune.validate(dict(tune.DEFAULTS, unified_workers=1, unified_threads=512,
                           unified_comm_base=384, unified_comm_threads=128,
                           unified_math_registers=208, unified_producer_registers=32,
                           unified_control_registers=64, wgmma_n256_atom_n=128))
        with self.assertRaisesRegex(PlanError, "one warpgroup"):
            tune.validate(dict(tune.DEFAULTS, unified_producer_registers=32))

    def test_staging_reserves_compute_and_control_memory(self):
        point = dict(tune.DEFAULTS, unified_segment_bytes=4096,
                     unified_copy_buffers=2)
        self.assertLessEqual(tune.shared_memory_upper_bound(point, 213200),
                             227 * 1024)
        point["unified_comm_threads"] = 128
        self.assertGreater(tune.shared_memory_upper_bound(point, 213200),
                           227 * 1024)

    def test_short_segments_reserve_storage_alignment_padding(self):
        point = dict(tune.DEFAULTS, unified_threads=640,
                     unified_comm_threads=352, unified_segment_bytes=16,
                     unified_copy_buffers=2)
        # Ten copy warps have 320 payload bytes, which need 384 bytes when
        # bounding the separately aligned base and Backward control storage.
        self.assertEqual(tune.shared_memory_upper_bound(point, 128),
                         128 + 2048 + 512 + 384)


if __name__ == "__main__":
    unittest.main()
