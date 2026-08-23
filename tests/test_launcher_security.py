#!/usr/bin/env python3
"""Security regressions for the multinode bootstrap secret channel."""

import base64
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
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


def mixed_hex(secret, salt=0):
    return "".join(
        character.upper()
        if ((index % 2 == 1) if salt == 0
            else ((index * 17 + salt * 13) % 11 < 5))
        else character.lower()
        for index, character in enumerate(secret.hex())
    )


def encoded_forms(secret):
    standard = base64.b64encode(secret).decode("ascii")
    urlsafe = base64.urlsafe_b64encode(secret).decode("ascii")
    return {
        secret.hex(),
        secret.hex().upper(),
        mixed_hex(secret),
        mixed_hex(secret, 1),
        standard,
        urlsafe,
        standard.rstrip("="),
        urlsafe.rstrip("="),
    }


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

def mixed_hex(secret, salt=0):
    return "".join(
        character.upper()
        if ((index % 2 == 1) if salt == 0
            else ((index * 17 + salt * 13) % 11 < 5))
        else character.lower()
        for index, character in enumerate(secret.hex())
    )

def inject_logs(secret, stage):
    standard = base64.b64encode(secret)
    urlsafe = base64.urlsafe_b64encode(secret)
    lines = [
        b"BOOTSTRAP " + secret.hex().encode(),
        secret.hex().encode(),
        secret.hex().upper().encode(),
        mixed_hex(secret).encode(),
        mixed_hex(secret, 1).encode(),
        standard,
        urlsafe,
        standard.rstrip(b"="),
        urlsafe.rstrip(b"="),
        secret,
    ]
    payload = b"\n".join(lines) + b"\n"
    os.write(1, payload)
    os.write(2, payload)
    print(f"failure_context rank={rank} stage={stage}", flush=True)
    print(f"failure_context rank={rank} stage={stage}", file=sys.stderr,
          flush=True)

def write_all(descriptor, payload):
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise RuntimeError("fake bootstrap writer made no progress")
        offset += written

standard = base64.b64encode(SECRET)
urlsafe = base64.urlsafe_b64encode(SECRET)
mixed = "".join(
    character.upper() if index % 2 else character.lower()
    for index, character in enumerate(SECRET.hex())
).encode()
representations = (
    SECRET,
    SECRET.hex().encode("ascii"),
    SECRET.hex().upper().encode("ascii"),
    mixed_hex(SECRET).encode("ascii"),
    mixed_hex(SECRET, 1).encode("ascii"),
    standard,
    urlsafe,
    standard.rstrip(b"="),
    urlsafe.rstrip(b"="),
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

if rank == 0:
    descriptor = int(values["--bootstrap-output-fd"])
    if MODE == "short":
        payload = SECRET[:int(os.environ.get("FAKE_PREFIX_LENGTH", "17"))]
    elif MODE == "oversize":
        payload = SECRET + b"X"
    else:
        payload = SECRET
    inject_logs(payload if MODE == "short" else SECRET,
                "rank0-bootstrap-log")
    write_all(descriptor, payload)
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
    inject_logs(bytes(received), "rank-bootstrap-log")

print("NCCL_RUNTIME fake-secure-channel", flush=True)

if MODE == "local_fail" and rank == 0:
    print("failure_context rank=0 stage=local-rank", file=sys.stderr,
          flush=True)
    raise SystemExit(7)
if MODE == "remote_fail" and rank == 1:
    print("failure_context rank=1 stage=remote-rank", file=sys.stderr,
          flush=True)
    raise SystemExit(9)
if MODE == "runner_exception" and rank == 1:
    raise RuntimeError("failure_context rank=1 stage=runner-exception secret="
                       + mixed_hex(SECRET))
if MODE == "timeout":
    def terminate(_signal, _frame):
        inject_logs(SECRET, "timeout-term")
        raise SystemExit(143)
    signal.signal(signal.SIGTERM, terminate)
    time.sleep(30)
if MODE == "timeout_kill":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
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
standard = base64.b64encode(SECRET)
urlsafe = base64.urlsafe_b64encode(SECRET)
mixed = "".join(
    character.upper() if index % 2 else character.lower()
    for index, character in enumerate(SECRET.hex())
).encode()
representations = (
    SECRET,
    SECRET.hex().encode(),
    SECRET.hex().upper().encode(),
    mixed,
    standard,
    urlsafe,
    standard.rstrip(b"="),
    urlsafe.rstrip(b"="),
)
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
    os.write(2, SECRET.hex().swapcase().encode() + b"\n")
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
            for representation in encoded_forms(secret):
                self.assertNotIn(representation, text)
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
        self.assertIn("failure_context rank=0 stage=rank0-bootstrap-log", combined)

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
        self.assertIn("failure_context", str(captured.exception.output))

    def test_timeout_term_then_kill_cleans_all_processes(self):
        with self.assertRaises(Exception) as captured:
            self.launch("timeout_kill", timeout=0.25)
        self.assert_scrubbed(captured.exception)
        self.assertIn("failure_context", str(captured.exception.output))

    def test_ssh_start_failure_is_scrubbed(self):
        outputs, combined = self.launch("success", ssh_fail=True)
        self.assertEqual(255, outputs[1].returncode)
        self.assertIn("failure_context rank=1 stage=ssh-start", combined)

    def test_all_nonempty_prefix_lengths_are_scrubbed(self):
        for length in (1, 2, 7, 8, 17, 127):
            with self.subTest(length=length):
                os.environ["FAKE_PREFIX_LENGTH"] = str(length)
                with self.assertRaisesRegex(RuntimeError, "short read") as captured:
                    self.launch("short")
                self.assert_scrubbed(captured.exception, SECRET[:length])
                self.assertIn("failure_context", str(captured.exception))
        os.environ.pop("FAKE_PREFIX_LENGTH", None)

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
        print("bootstrap_mixed_hex_exposed = False")
        print("failure_context_present = True")
        print("raw_exposed = False")
        print("lower_hex_exposed = False")
        print("upper_hex_exposed = False")
        print("mixed_hex_exposed = False")
        print("standard_base64_exposed = False")
        print("urlsafe_base64_exposed = False")
        print("unpadded_variants_exposed = False")
        print("old_marker_exposed = False")

    def test_scrubber_injected_matrix_for_every_prefix(self):
        channels = ("stdout", "stderr", "exception", "tail", "cli")
        for length in (1, 2, 7, 8, 17, 127, 128):
            secret = SECRET[:length]
            forms = encoded_forms(secret)
            text_payload = "\n".join(forms)
            for channel in channels:
                with self.subTest(length=length, channel=channel):
                    if channel in ("stdout", "stderr"):
                        value = (f"{channel}_context\n".encode() + secret +
                                 b"\n" + text_payload.encode())
                    else:
                        value = (f"failure_context channel={channel}\n" +
                                 text_payload)
                    scrubbed = LAUNCHER.scrub_output(value, secret)
                    self.assert_scrubbed(scrubbed, secret)
                    self.assertIn(
                        "stdout_context" if channel == "stdout" else
                        "stderr_context" if channel == "stderr" else
                        "failure_context",
                        scrubbed,
                    )
        self.assertEqual("empty-safe", LAUNCHER.scrub_output("empty-safe", b""))
        print("mixed_case_hex_exposed = False")
        print("short_prefix_exposed = False")

    def test_top_level_cli_scrubs_injected_prefixes(self):
        command = [
            sys.executable,
            str(ROOT / "tests" / "run_multinode_nccl.py"),
            "--binary", str(self.runner),
            "--remote", "fake@example.invalid",
            "--remote-binary", str(self.runner),
            "--gpus-per-node", "1",
            "--timeout", "2",
        ]
        for length in (1, 2, 7, 8, 17, 127, 128):
            with self.subTest(length=length):
                environment = os.environ.copy()
                environment["FAKE_RUNNER_MODE"] = "short"
                environment["FAKE_PREFIX_LENGTH"] = str(length)
                result = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=environment,
                    timeout=10,
                    check=False,
                )
                combined = result.stdout + b"\n" + result.stderr
                self.assertNotEqual(0, result.returncode)
                self.assert_scrubbed(
                    combined.decode("utf-8", errors="replace"), SECRET[:length]
                )
                self.assertIn(b"failure_context", combined)
                self.assert_no_residual_processes()

    def test_twenty_failure_recovery_cycles_do_not_leak_fds(self):
        before = len(list(Path("/proc/self/fd").iterdir()))
        for _round in range(20):
            failed, failed_log = self.launch("remote_fail")
            self.assertEqual(9, failed[1].returncode)
            self.assertIn("failure_context rank=1 stage=remote-rank", failed_log)
            recovered, _recovered_log = self.launch("success")
            self.assertEqual([0, 0], [item.returncode for item in recovered])
        after = len(list(Path("/proc/self/fd").iterdir()))
        self.assertEqual(before, after)

    def test_python_exact_frame_writer(self):
        class PoisonStream:
            def __init__(self):
                self.closed_count = 0

            def fileno(self):
                return 91

            def close(self):
                self.closed_count += 1

        class PoisonProcess:
            def __init__(self):
                self.stdin = PoisonStream()

        for chunk in (1, 17, 127):
            process_stream = PoisonStream()
            process = type("Process", (), {"stdin": process_stream})()
            received = bytearray()

            def short_write(_descriptor, payload, limit=chunk):
                count = min(limit, len(payload))
                received.extend(payload[:count])
                return count

            with mock.patch.object(LAUNCHER.os, "write", short_write):
                LAUNCHER.send_bootstrap(process, SECRET)
            self.assertEqual(SECRET, bytes(received))
            self.assertEqual(1, process_stream.closed_count)
            self.assertIsNone(process.stdin)

        process_stream = PoisonStream()
        process = type("Process", (), {"stdin": process_stream})()
        received = bytearray()
        calls = 0

        def interrupted_then_write(_descriptor, payload):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise InterruptedError()
            received.extend(payload)
            return len(payload)

        with mock.patch.object(LAUNCHER.os, "write", interrupted_then_write):
            LAUNCHER.send_bootstrap(process, SECRET)
        self.assertEqual(SECRET, bytes(received))
        self.assertEqual(1, process_stream.closed_count)

        for action, expected in ((0, "zero progress"),
                                 (None, "no byte count"),
                                 ("broken", "closed before"),
                                 ("oserror", "write failed")):
            process = PoisonProcess()
            stream = process.stdin

            def fail_write(_descriptor, _payload, result=action):
                if result == "broken":
                    raise BrokenPipeError()
                if result == "oserror":
                    raise OSError("poison")
                return result

            with mock.patch.object(LAUNCHER.os, "write", fail_write):
                with self.assertRaisesRegex(RuntimeError, expected):
                    LAUNCHER.send_bootstrap(process, SECRET)
            self.assertIsNone(process.stdin)
            self.assertEqual(1, stream.closed_count)

        for invalid in (None, b"", SECRET[:127]):
            process = PoisonProcess()
            stream = process.stdin
            with self.assertRaisesRegex(RuntimeError, "exactly 128 bytes"):
                LAUNCHER.send_bootstrap(process, invalid)
            self.assertIsNone(process.stdin)
            self.assertEqual(1, stream.closed_count)


if __name__ == "__main__":
    unittest.main(verbosity=2)
