#!/usr/bin/env python3
"""Host-only fail-closed tests for the Megatron RDMA launcher."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import run_multinode_megatron_rdma as rdma


class RdmaLauncherTests(unittest.TestCase):
    def test_verified_rdma_log(self):
        private_address = "192" + ".168.4.7"
        evidence = rdma.validate_rdma_logs(
            "host:1:1 [0] NCCL INFO NCCL version 2.29.7+cuda12.9\n"
            "host:1:1 [0] NCCL INFO NET/IB : Using "
            "[0]mlx5_bond_0:1/RoCE; OOB eth0:" + private_address + "\n"
            "host:1:1 [0] NCCL INFO Using network IB\n",
            "host:2:2 [0] NCCL INFO Channel 00/0 : 0 -> 1 "
            "via NET/IB/0/GDRDMA\n")
        self.assertTrue(evidence["net_ib_observed"])
        self.assertTrue(evidence["gdrdma_observed"])
        self.assertFalse(evidence["socket_payload_fallback_observed"])
        self.assertNotIn(private_address, "\n".join(
            evidence["key_log_lines"]))

    def test_socket_fallback_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Socket payload"):
            rdma.validate_rdma_logs(
                "NCCL INFO NET/IB : Using [0]mlx5_bond_0:1/RoCE\n"
                "NCCL INFO Using network Socket\n",
                "NCCL INFO via NET/IB/0/GDRDMA\n")

    def test_missing_gdr_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "GDRDMA"):
            rdma.validate_rdma_logs(
                "NCCL INFO NET/IB : Using [0]mlx5_bond_0:1/RoCE\n", "")

    def test_hca_parser_and_counter_delta(self):
        self.assertEqual(
            rdma.parse_hcas("mlx5_bond_0:1,mlx5_bond_1:1,mlx5_bond_0:1"),
            ["mlx5_bond_0", "mlx5_bond_1"])
        before = {"mlx5_bond_0": {
            "xmit_data_bytes": 10, "rcv_data_bytes": 20,
            "rate": "200 Gb/sec", "gid_type": "RoCE v2", "netdev": "bond0"}}
        after = {"mlx5_bond_0": {
            "xmit_data_bytes": 110, "rcv_data_bytes": 220,
            "rate": "200 Gb/sec", "gid_type": "RoCE v2", "netdev": "bond0"}}
        self.assertEqual(rdma.counter_delta(before, after)["mlx5_bond_0"], {
            "tx_bytes": 100, "rx_bytes": 200, "rate": "200 Gb/sec",
            "gid_type": "RoCE v2", "netdev": "bond0"})


if __name__ == "__main__":
    unittest.main()
