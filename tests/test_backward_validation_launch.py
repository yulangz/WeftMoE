"""Native shape selection must agree with the independent reference request."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "tests"))
import run_backward_validation as validation
import run_multinode_nccl as multinode


def rank_evidence(world, hidden=2560, intermediate=1280):
    return "".join(
        f"BACKWARD_RESULT rank={rank} world={world} tokens=129 "
        f"hidden={hidden} intermediate={intermediate} experts=32 top_k=8 "
        "repeat_equal=1\n" for rank in range(world))


class BackwardValidationLaunchTest(unittest.TestCase):
    def test_backward_windows_preserve_nanoseconds_and_reject_invalid_evidence(self):
        start = 1789689600123456789
        fields = dict(backend="gin-rdma", tokens=8192, hidden=2048,
                      intermediate=512, experts=256, top_k=8,
                      dispatch_peer_bytes=10, return_peer_bytes=10,
                      benchmark_pause_us=100000)
        for key in ("wall", "total", "prepare", "compute", "reduce",
                    "transport", "dispatch", "dab", "dx", "dw2", "dw1",
                    "combine", "dw_combine_overlap"):
            fields[key] = "10,12"
        fields["sample_begin_unix_ns"] = f"{start},{start + 100000000}"
        fields["sample_end_unix_ns"] = f"{start + 10000},{start + 100012000}"

        def aggregate(record):
            line = "BACKWARD_RESULT " + " ".join(
                f"{key}={value}" for key, value in record.items())
            return multinode.aggregate_gin_backward([
                multinode.RankOutput(0, 0, line, "")])

        result = aggregate(fields)
        self.assertEqual(result["per_rank_sample_begin_unix_ns"],
                         [[start, start + 100000000]])
        self.assertEqual(result["per_rank_benchmark_pause_us"], [100000])
        for change in ({"sample_end_unix_ns": f"{start - 1},{start}"},
                       {"sample_end_unix_ns": str(start)}):
            with self.assertRaises(RuntimeError):
                aggregate({**fields, **change})
        incomplete = dict(fields)
        del incomplete["sample_begin_unix_ns"]
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            aggregate(incomplete)

    def test_requested_dimensions_reach_native_for_both_topologies(self):
        for world in (8, 16):
            with self.subTest(world=world), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)
                argv = ["validation", "--binary", "/tmp/native", "--output", tmp,
                        "--world", str(world), "--hidden", "2560",
                        "--intermediate", "1280", "--experts", "32",
                        "--top-k", "8", "--tokens", "129", "--native-only"]

                def native(command, env, log, *, timeout):
                    self.assertEqual(env["WEFTMOE_BENCH_HIDDEN"], "2560")
                    self.assertEqual(env["WEFTMOE_BENCH_INTERMEDIATE"], "1280")
                    self.assertIn("--timeout", command)
                    self.assertEqual(timeout, int(command[command.index("--timeout") + 1]) + 30)
                    launcher = "run_local_activation_p2p.py" if world == 8 else "run_multinode_nccl.py"
                    self.assertTrue(any(item.endswith(launcher) for item in command))
                    log.write_text(rank_evidence(world))

                with mock.patch.object(sys, "argv", argv), \
                        mock.patch.object(validation, "environment", return_value={
                            "WEFTMOE_BENCH_HIDDEN": "2048",
                            "WEFTMOE_BENCH_INTERMEDIATE": "512"}), \
                        mock.patch.object(validation, "run", side_effect=native) as run, \
                        contextlib.redirect_stdout(io.StringIO()):
                    validation.main()
                run.assert_called_once()
                self.assertTrue((output / "native.log").exists())

    def test_stale_binary_and_missing_rank_cannot_pass_native_only(self):
        for evidence in (rank_evidence(8, hidden=2048), rank_evidence(7)):
            with tempfile.TemporaryDirectory() as tmp:
                log = Path(tmp) / "native.log"
                log.write_text(evidence)
                with self.assertRaisesRegex(RuntimeError, "native.*(mismatch|evidence)"):
                    validation.validate_native_shape(
                        log, world=8, tokens=129, hidden=2560,
                        intermediate=1280, experts=32, top_k=8)

    def test_nonpositive_dimensions_fail_before_launch(self):
        for hidden, intermediate in ((0, 1280), (2560, -1)):
            with mock.patch.object(sys, "argv", [
                    "validation", "--binary", "/tmp/native", "--output", "/tmp/unused",
                    "--world", "8", "--hidden", str(hidden),
                    "--intermediate", str(intermediate), "--top-k", "8"]), \
                    mock.patch.object(validation, "run") as run, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                validation.main()
            run.assert_not_called()

    def test_dimensions_are_forwarded_to_remote_without_default_override(self):
        args = mock.Mock(nccl_debug="WARN", socket_ifname="bond0",
                         disable_ib=False, enable_ib=True)
        with mock.patch.dict(multinode.os.environ, {
                "WEFTMOE_BENCH_HIDDEN": "2560",
                "WEFTMOE_BENCH_INTERMEDIATE": "1280",
                "WEFTMOE_BACKWARD_BENCH_PAUSE_US": "500000",
                "WEFTMOE_BACKWARD_BENCH_TIMESTAMPS": "1"}, clear=True):
            env = multinode.environment_overrides(args)
            self.assertEqual(env["WEFTMOE_BENCH_HIDDEN"], "2560")
            self.assertEqual(env["WEFTMOE_BENCH_INTERMEDIATE"], "1280")
            self.assertEqual(env["WEFTMOE_BACKWARD_BENCH_PAUSE_US"], "500000")
            self.assertEqual(env["WEFTMOE_BACKWARD_BENCH_TIMESTAMPS"], "1")
        with mock.patch.dict(multinode.os.environ, {}, clear=True):
            env = multinode.environment_overrides(args)
            self.assertNotIn("WEFTMOE_BENCH_HIDDEN", env)
            self.assertNotIn("WEFTMOE_BENCH_INTERMEDIATE", env)


if __name__ == "__main__":
    unittest.main()
