"""Compile cache must honor external headers, flags and cache hits."""
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_native import compile_source, PUBLIC_HEADERS
import build_native


class NativeBuildTest(unittest.TestCase):
    def test_header_changed_during_compile_is_not_cached(self):
        compiler = shutil.which('c++')
        if compiler is None:
            self.skipTest('C++ compiler unavailable')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            header = root / 'value.hpp'
            source = root / 'value.cpp'
            header.write_text('#define VALUE 1\n')
            source.write_text('#include "value.hpp"\nint value() { return VALUE; }\n')
            run = build_native.run

            def edit_after_compile(command):
                run(command)
                if '-c' in command:
                    header.write_text('#define VALUE 2\n')

            with mock.patch.object(build_native, 'run', side_effect=edit_after_compile):
                with self.assertRaisesRegex(RuntimeError, 'inputs changed during compilation'):
                    compile_source(str(source), [compiler, '-O2'], root / 'cache')
            self.assertFalse(list((root / 'cache').glob('*.o')))
            self.assertFalse(list((root / 'cache').glob('*.json')))
            self.assertTrue(compile_source(str(source), [compiler, '-O2'], root / 'cache').is_file())

    def test_dependency_and_flag_invalidation(self):
        compiler = shutil.which('c++')
        if compiler is None:
            self.skipTest('C++ compiler unavailable')
        with tempfile.TemporaryDirectory(prefix='weft build ') as temporary:
            root = Path(temporary)
            header = root / 'value.hpp'
            source = root / 'value.cpp'
            header.write_text('#define VALUE 1\n')
            source.write_text('#include "value.hpp"\nint value() { return VALUE; }\n')
            command = [compiler, '-O2']
            obj = compile_source(str(source), command, root / 'cache')
            before = obj.stat().st_mtime_ns
            self.assertEqual(compile_source(str(source), command, root / 'cache').stat().st_mtime_ns, before)
            header.write_text('#define VALUE 2\n')
            os.utime(header, ns=(header.stat().st_atime_ns, header.stat().st_mtime_ns + 1))
            compile_source(str(source), command, root / 'cache')
            self.assertNotEqual(obj.stat().st_mtime_ns, before)
            other = compile_source(str(source), [compiler, '-O0'], root / 'cache')
            self.assertNotEqual(obj, other)

    def test_installed_headers_exclude_repository_only_interfaces(self):
        self.assertEqual(len(PUBLIC_HEADERS), 13)
        self.assertNotIn('include/weftmoe/metadata/serialize.hpp', PUBLIC_HEADERS)
        self.assertNotIn('include/weftmoe/kernels/persistent_contract.hpp', PUBLIC_HEADERS)


if __name__ == '__main__':
    unittest.main()
