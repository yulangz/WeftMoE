"""Setuptools hooks; use pip/build rather than invoking setup.py directly."""
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.bdist_wheel import bdist_wheel

from build_native import PUBLIC_HEADERS, ROOT, build


class NativeBuild(build_ext):
    def get_ext_filename(self, name):
        return str(Path(*name.split('.')[:-1]) / 'lib' / 'libweftmoe.so')

    def build_extension(self, ext):
        output = Path(self.get_ext_fullpath(ext.name)).resolve()
        build('lib', output)
        package = output.parent.parent
        for header in PUBLIC_HEADERS:
            destination = package / header
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / header, destination)

    def _get_inplace_equivalent(self, build_py, ext):
        relative = self.get_ext_filename(ext.name)
        return str(ROOT / relative), str(Path(self.build_lib) / relative)

    def copy_extensions_to_source(self):
        (ROOT / 'weftmoe/lib').mkdir(parents=True, exist_ok=True)
        super().copy_extensions_to_source()
        source = Path(self.build_lib) / 'weftmoe/include'
        if source.is_dir():
            shutil.copytree(source, ROOT / 'weftmoe/include', dirs_exist_ok=True)


class NativeWheel(bdist_wheel):
    def get_tag(self):
        # Native C++ library, without a CPython ABI dependency.
        _, _, platform = super().get_tag()
        return 'py3', 'none', platform


setup(ext_modules=[Extension('weftmoe._native', sources=[])],
      cmdclass={'build_ext': NativeBuild, 'bdist_wheel': NativeWheel})
