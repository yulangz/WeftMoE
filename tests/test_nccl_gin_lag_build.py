#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check dependency patch reuse and protection of existing source changes."""

import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_nccl_gin_lag", ROOT / "tools/build_nccl_gin_lag.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class NcclBuildTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="megamoe-nccl-build-")
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.source = self.work / "source"
        self.source.mkdir()
        self.git("init", "-q")
        self.git("config", "user.name", "Test")
        self.git("config", "user.email", "test@example.invalid")
        self.qp = self.source / builder.PATCHED_SOURCE
        self.qp.parent.mkdir(parents=True)
        self.original = "before\n" + "context\n" * 20 + "tail\n"
        self.qp.write_text(self.original)
        (self.source / "other.cc").write_text("original\n")
        self.git("add", ".")
        self.git("commit", "-qm", "fixture")
        commit = self.git("rev-parse", "HEAD").strip()
        self.qp.write_text(self.original.replace("before", "after", 1))
        self.patched = self.qp.read_bytes()
        tool_dir = self.work / "tools"
        (tool_dir / "nccl").mkdir(parents=True)
        (tool_dir / "nccl/nccl-2.29.7-gin-lag.patch").write_text(
            "Test patch provenance\n\n" + self.git("diff", "--no-ext-diff", "--no-color"))
        self.qp.write_text(self.original)
        for name, value in (("NCCL_COMMIT", commit), ("TOOL_DIR", tool_dir)):
            mock = patch.object(builder, name, value)
            mock.start()
            self.addCleanup(mock.stop)

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.source, text=True)

    def test_initial_patch_and_repeat_are_identical(self):
        self.assertEqual(builder.prepare_source(self.work), self.source)
        self.assertEqual(self.qp.read_bytes(), self.patched)
        builder.prepare_source(self.work)
        self.assertEqual(self.qp.read_bytes(), self.patched)

    def test_wrong_version_is_rejected_without_editing(self):
        with patch.object(builder, "NCCL_COMMIT", "wrong-version"):
            with self.assertRaisesRegex(RuntimeError, "Expected NCCL commit"):
                builder.prepare_source(self.work)
        self.assertEqual(self.qp.read_text(), self.original)

    def test_unrelated_tracked_edit_is_preserved(self):
        other = self.source / "other.cc"
        other.write_text("user edit\n")
        with self.assertRaisesRegex(RuntimeError, "unrelated tracked"):
            builder.prepare_source(self.work)
        self.assertEqual(other.read_text(), "user edit\n")
        self.assertEqual(self.qp.read_text(), self.original)

    def test_extra_edit_outside_patch_hunk_is_rejected_on_repeat(self):
        builder.prepare_source(self.work)
        extra = self.qp.read_text().replace("tail", "user edit")
        self.qp.write_text(extra)
        with self.assertRaisesRegex(RuntimeError, "differs from the supported patch"):
            builder.prepare_source(self.work)
        self.assertEqual(self.qp.read_text(), extra)


if __name__ == "__main__":
    unittest.main()
