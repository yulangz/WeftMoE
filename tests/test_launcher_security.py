#!/usr/bin/env python3
"""Security regressions for the multinode bootstrap secret channel."""

import base64
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_multinode_nccl", ROOT / "tests" / "run_multinode_nccl.py"
)
LAUNCHER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LAUNCHER
SPEC.loader.exec_module(LAUNCHER)

SECRET = bytes((index * 37 + 11) & 0xFF for index in range(128))
SECRET_HEX = SECRET.hex()
SECRET_B64 = base64.b64encode(SECRET).decode("ascii")


RUNNER_SOURCE = r'''#!/usr/bin/env python3
import base64
import os
import signal
import sys
import time

SECRET = base64.b64decode("__SECRET_B64__")
MODE = os.environ.get("FAKE_RUNNER_MODE", "success")
PID_LOG = os.environ["FAKE_PID_LOG"]

with open(PID_LOG, "a", encoding="ascii") as output:
    output.write(str(os.getpid()) + "\n")

arguments = sys.argv[1:]
if arguments and arguments[0] == "--rank-runner":
    arguments = arguments[1:]
values = {arguments[index]: arguments[index + 1]
          for index in range(0, len(arguments), 2)
          if index + 1 < len(arguments)}
rank = int(values.get("--rank", "0"))

representations = (
    SECRET,
    SECRET.hex().encode("ascii"),
    base64.b64encode(SECRET),
)
cmdline = open("/proc/self/cmdline", "rb").read()
environment = b"\0".join(
    key.encode() + b"=" + value.encode()
    for key, value in os.environ.items()
)
argv_secret = any(value in cmdline for value in representations)
env_secret = any(value in environment for value in representations)
print(f"CHANNEL_CHECK rank={rank} argv_secret={str(argv_secret).lower()} "
      f"env_secret={str(env_secret).lower()}", flush=True)
if argv_secret or env_secret:
    raise SystemExit(86)

def leak_and_fail(stage, code):
    os.write(2, b"BOOTSTRAP " + SECRET.hex().encode() + b"\n")
    os.write(2, base64.b64encode(SECRET) + b"\n")
    os.write(2, SECRET + b"\n")
    print(f"failure_context rank={rank} stage={stage}", file=sys.stderr,
          flush=True)
    raise SystemExit(code)

if rank == 0:
    descriptor = int(values["--bootstrap-output-fd"])
    if MODE == "short":
        payload = SECRET[:17]
    elif MODE == "oversize":
        payload = SECRET + b"X"
    else:
        payload = SECRET
    os.write(descriptor, payload)
    os.close(descriptor)
else:
    descriptor = int(values["--bootstrap-input-fd"])
    received = bytearray()
    while len(received) < len(SECRET):
        chunk = os.read(descriptor, len(SECRET) - len(received))
        if not chunk:
            break
        received.extend(chunk)
    if bytes(received) != SECRET:
        print(f"failure_context rank={rank} stage=input-short "
              f"actual={len(received)}", file=sys.stderr, flush=True)
        raise SystemExit(12)

print("NCCL_RUNTIME fake-secure-channel", flush=True)

if MODE == "local_fail" and rank == 0:
    leak_and_fail("local-rank", 7)
if MODE == "remote_fail" and rank == 1:
    leak_and_fail("remote-rank", 9)
if MODE == "runner_exception" and rank == 1:
    os.write(2, b"BOOTSTRAP " + SECRET.hex().encode() + b"\n")
    raise RuntimeError("failure_context rank=1 stage=runner-exception")
if MODE == "timeout":
    def terminate(_signal, _frame):
        os.write(2, base64.b64encode(SECRET) + b"\n")
        raise SystemExit(143)
    signal.signal(signal.SIGTERM, terminate)
    time.sleep(30)
'''.replace("__SECRET_B64__", "__B64__")


SSH_SOURCE = r'''#!/usr/bin/env python3
import base64
import os
import sys

SECRET = base64.b64decode("__SECRET_B64__")
command = sys.argv[-1]
if command.startswith("pkill "):
    raise SystemExit(0)
representations = (SECRET, SECRET.hex().encode(), base64.b64encode(SECRET))
cmdline = open("/proc/self/cmdline", "rb").read()
environment = b"\0".join(
    key.encode() + b"=" + value.encode()
    for key, value in os.environ.items()
)
if any(value in cmdline or value in environment for value in representations):
    raise SystemExit(86)
if os.environ.get("FAKE_SSH_FAIL") == "1":
    received = sys.stdin.buffer.read(len(SECRET))
    os.write(2, b"BOOTSTRAP " + SECRET.hex().encode() + b"\n")
    os.write(2, base64.b64encode(SECRET) + b"\n")
    print("failure_context rank=1 stage=ssh-start", file=sys.stderr,
          flush=True)
    raise SystemExit(255)
os.execv("/bin/sh", ["sh", "-c", command])
'''.replace("__SECRET_B64__", "__B64__")


class LauncherSecretChannelTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.runner = self.directory / "fake_runner"
        self.ssh = self.directory / "ssh"
        self.pid_log = self.directory / "pids"
        self.runner.write_text(
            RUNNER_SOURCE.replace("__B64__", SECRET_B64), encoding="utf-8"
        )
        self.ssh.write_text(
            SSH_SOURCE.replace("__B64__", SECRET_B64), encoding="utf-8"
        )
        self.runner.chmod(0o700)
        self.ssh.chmod(0o700)
        self.old_environment = os.environ.copy()
        os.environ["PATH"] = str(self.directory) + os.pathsep + os.environ["PATH"]
        os.environ["FAKE_PID_LOG"] = str(self.pid_log)
        os.environ.pop("FAKE_SSH_FAIL", None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.old_environment)
        self.temporary.cleanup()

    def args(self, timeout=3.0):
        return LAUNCHER.parse_args([
            "--binary", str(self.runner),
            "--remote", "fake@example.invalid",
            "--remote-binary", str(self.runner),
            "--gpus-per-node", "1",
            "--timeout", str(timeout),
        ])

    def assert_scrubbed(self, value, partial=None):
        text = str(value)
        encoded = text.encode("utf-8", errors="replace")
        for secret in (SECRET, partial):
            if not secret:
                continue
            self.assertNotIn(secret, encoded)
            self.assertNotIn(secret.hex(), text)
            self.assertNotIn(secret.hex().upper(), text)
            self.assertNotIn(base64.b64encode(secret).decode(), text)
        self.assertNotIn("BOOTSTRAP ", text)

    def assert_no_residual_processes(self):
        if not self.pid_log.exists():
            return
        for value in self.pid_log.read_text(encoding="ascii").splitlines():
            proc = Path("/proc") / value / "cmdline"
            if proc.exists():
                self.assertNotIn(
                    str(self.runner).encode(), proc.read_bytes(),
                    f"fake runner pid={value} survived cleanup",
                )
        self.assertEqual([], list(self.directory.glob("*bootstrap*")))

    def launch(self, mode, *, timeout=3.0, ssh_fail=False):
        os.environ["FAKE_RUNNER_MODE"] = mode
        if ssh_fail:
            os.environ["FAKE_SSH_FAIL"] = "1"
        try:
            outputs = LAUNCHER.launch_rank_processes(self.args(timeout))
            combined = "\n".join(
                result.stdout + "\n" + result.stderr for result in outputs
            )
            self.assert_scrubbed(combined)
            for result in outputs:
                self.assertNotIn("argv_secret=true", result.stdout)
                self.assertNotIn("env_secret=true", result.stdout)
            return outputs, combined
        finally:
            self.assert_no_residual_processes()

    def test_success_path_uses_no_text_secret(self):
        outputs, combined = self.launch("success")
        self.assertEqual([0, 0], [result.returncode for result in outputs])
        self.assertEqual(0, combined.count("BOOTSTRAP "))

    def test_local_rank_failure_is_scrubbed(self):
        outputs, combined = self.launch("local_fail")
        self.assertEqual(7, outputs[0].returncode)
        self.assertIn("failure_context rank=0 stage=local-rank", combined)

    def test_remote_rank_failure_is_scrubbed(self):
        outputs, combined = self.launch("remote_fail")
        self.assertEqual(9, outputs[1].returncode)
        self.assertIn("failure_context rank=1 stage=remote-rank", combined)

    def test_timeout_cleans_all_processes(self):
        with self.assertRaises(Exception) as captured:
            self.launch("timeout", timeout=0.25)
        self.assert_scrubbed(captured.exception)

    def test_ssh_start_failure_is_scrubbed(self):
        outputs, combined = self.launch("success", ssh_fail=True)
        self.assertEqual(255, outputs[1].returncode)
        self.assertIn("failure_context rank=1 stage=ssh-start", combined)

    def test_short_bootstrap_is_controlled(self):
        with self.assertRaisesRegex(RuntimeError, "short read") as captured:
            self.launch("short")
        self.assert_scrubbed(captured.exception, SECRET[:17])

    def test_oversize_bootstrap_is_controlled(self):
        with self.assertRaisesRegex(RuntimeError, "malformed") as captured:
            self.launch("oversize")
        self.assert_scrubbed(captured.exception)

    def test_runner_exception_keeps_context_but_not_secret(self):
        outputs, combined = self.launch("runner_exception")
        self.assertNotEqual(0, outputs[1].returncode)
        self.assertIn("failure_context rank=1 stage=runner-exception", combined)
        self.assertEqual(0, combined.count("BOOTSTRAP "))
        print("bootstrap_marker_count = 0")
        print("bootstrap_hex_exposed  = False")
        print("failure_context_present = True")


if __name__ == "__main__":
    unittest.main(verbosity=2)
