"""CPU checks for unattended timeout cleanup, resume and result eligibility."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("dt_autotune", Path(__file__).resolve().parents[1] /
    "tools/autotune_dt_fp8.py")
tune = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tune)


class AutotuneTest(unittest.TestCase):
    def test_resume_rejects_obsolete_kernel_policy_before_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tune.atomic_json(root / "manifest.json", dict(validation=tune.POLICY))
            with patch.object(sys, "argv", ["autotune.py", "--output", str(root), "--detach"]), \
                 patch.object(tune.subprocess, "Popen", side_effect=AssertionError("launched stale worker")):
                with self.assertRaisesRegex(RuntimeError, "kernel policy changed"):
                    tune.main()

    def test_known_case_waiver_requires_explicit_manifest_setting(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                native = root / "jobs/qwen35b/default/native"
                native.mkdir(parents=True)
                (native / "complete.json").write_text("{}")
                (native / "_runtime.so").touch()
                manifest = dict(environment={}, python=sys.executable, allow_known_one_token=enabled)
                with patch.object(tune, "run_command", return_value=0) as run, \
                     patch.object(tune, "idle_gpus"), \
                     patch.object(tune, "parse_measurements", return_value=[]):
                    result = tune.execute_job(root, manifest, "qwen35b", tune.candidates()[0], "search", False)
                self.assertEqual(result["status"], "passed")
                check_env = run.call_args_list[0].kwargs["env"]
                self.assertEqual(check_env.get("TEST_DT_ALLOW_KNOWN_ONE_TOKEN"), "1" if enabled else None)

    def test_reuse_builds_checks_native_hashes_and_does_not_reuse_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            previous, output = root / "old", root / "new"
            native = previous / "jobs/qwen35b/default/native"
            native.mkdir(parents=True)
            candidate = tune.candidates()[0]
            tune.atomic_json(native / "complete.json", dict(candidate=candidate))
            tune.atomic_json(native.parent / "correctness.json", dict(passed=True))
            tune.atomic_json(previous / "manifest.json", dict(kernel_policy=tune.KERNEL_POLICY, source_hashes={
                "src/kernel.cu": "original", "tests/check.py": "old test"}))
            info = tune.reuse_builds(output, previous, {
                "src/kernel.cu": "original", "tests/check.py": "BF16 test"})
            self.assertEqual(info["reused_builds"], 1)
            self.assertTrue((output / "jobs/qwen35b/default/native").is_symlink())
            self.assertFalse((output / "jobs/qwen35b/default/correctness.json").exists())
            with self.assertRaisesRegex(RuntimeError, "hashes differ"):
                tune.reuse_builds(root / "bad", previous, {"src/kernel.cu": "changed"})

    def test_timeout_terminates_descendant_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            code = ("import subprocess,sys,time,pathlib; "
                    "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(90)']); "
                    "pathlib.Path('child.pid').write_text(str(p.pid)); time.sleep(90)")
            with self.assertRaisesRegex(RuntimeError, "timeout"):
                tune.run_command([sys.executable, "-c", code], cwd=root, env=dict(os.environ),
                                 log=root / "run.log", timeout=1)
            pid = int((root / "child.pid").read_text())
            # A killed orphan can briefly remain as a zombie before init reaps it.
            stat = Path(f"/proc/{pid}/stat")
            if stat.exists():
                self.assertEqual(stat.read_text().split()[2], "Z")

    def test_resume_never_reexecutes_passed_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = tune.candidates()[0]
            path = root / "jobs/qwen35b/default/search"
            path.mkdir(parents=True)
            tune.atomic_json(path / "result.json", dict(status="passed", rows=[]))
            with patch.object(tune, "run_command", side_effect=AssertionError("reexecuted")):
                result = tune.execute_job(root, {}, "qwen35b", candidate, "search", False)
            self.assertEqual(result["status"], "passed")

    def test_confirmation_requires_both_rounds_and_discards_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def save(cid, phase, value, status="passed"):
                path = root / "jobs/qwen35b" / cid / phase
                path.mkdir(parents=True)
                tune.atomic_json(path / "result.json", dict(model="qwen35b", candidate=dict(id=cid),
                    phase=phase, status=status, error="mock failure", rows=[dict(tokens=n, bf16_us=100,
                    fp8_online_us=value, fp8_prequant_us=value) for n in tune.SIZES]))
            for phase in ("search", "confirm-1300", "confirm-1700"):
                save("default", phase, 100)
                save("stable", phase, 80)
            save("one-round-only", "confirm-1300", 1)
            save("failed", "search", 1, "failed")
            report = tune.summarize(root)
            confirmed = report["models"]["qwen35b"]["confirmed"]["best_per_case"]
            self.assertTrue(all(r["candidate"]["id"] == "stable" for r in confirmed))
            self.assertTrue(all(r["speedup_vs_default_fp8"] == 1.25 for r in confirmed))
            self.assertEqual(report["job_counts"]["failed"], 1)

    def test_incomplete_or_nonfinite_measurement_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "benchmark.log"
            log.write_text('RESULT {"tokens":1}\n')
            with self.assertRaisesRegex(RuntimeError, "missing"):
                tune.parse_measurements(log)
            rows = [dict(tokens=n, bf16_us=10, fp8_online_us=9, fp8_prequant_us=8,
                         fp8_vs_bf16_max_rank_diff=.003)
                    for n in tune.SIZES]
            rows[0]["fp8_online_us"] = float("nan")
            log.write_text("".join("RESULT " + json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(RuntimeError, "invalid timing"):
                tune.parse_measurements(log)
            rows[0]["fp8_online_us"] = 9
            for bad_diff in (.005, .01, float("nan")):
                rows[0]["fp8_vs_bf16_max_rank_diff"] = bad_diff
                log.write_text("".join("RESULT " + json.dumps(row) + "\n" for row in rows))
                with self.assertRaisesRegex(RuntimeError, "numerical sanity"):
                    tune.parse_measurements(log)

    def test_invalid_default_does_not_hide_eligible_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for phase in ("search", "confirm-1300", "confirm-1700"):
                for cid, status, us in (("default", "failed", 1), ("valid", "passed", 80)):
                    path = root / "jobs/qwen35b" / cid / phase
                    path.mkdir(parents=True)
                    tune.atomic_json(path / "result.json", dict(model="qwen35b", candidate=dict(id=cid),
                        phase=phase, status=status, error="strict numerical failure", rows=[dict(tokens=n,
                            bf16_us=100, fp8_online_us=us, fp8_prequant_us=us) for n in tune.SIZES]))
            report = tune.summarize(root)
            for phase in ("search", "confirmed"):
                model = report["models"]["qwen35b"][phase]
                self.assertFalse(model["default_eligible"])
                self.assertTrue(all(row["candidate"]["id"] == "valid" for row in model["best_per_case"]))
                self.assertTrue(all(row["speedup_vs_default_fp8"] is None for row in model["best_per_case"]))
                self.assertAlmostEqual(model["fixed_config_ranking"][0]["geometric_speedup_vs_bf16"], 1.25)

    def test_no_eligible_candidates_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "jobs/qwen35b/default/search"
            path.mkdir(parents=True)
            tune.atomic_json(path / "result.json", dict(model="qwen35b", candidate=dict(id="default"),
                phase="search", status="failed", error="strict numerical failure"))
            report = tune.summarize(root)
            self.assertEqual(report["models"]["qwen35b"], {})
            self.assertEqual(report["job_counts"]["failed"], 1)


if __name__ == "__main__":
    unittest.main()
