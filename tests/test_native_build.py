"""Compile cache must honor external headers, flags and cache hits."""
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_native import compile_source, PUBLIC_HEADERS


class NativeBuildTest(unittest.TestCase):
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
