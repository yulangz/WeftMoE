"""Ship JIT sources; CUDA compilation happens on first use, not pip install."""
from pathlib import Path
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent


class SourceBuild(build_py):
    def run(self):
        super().run()
        destination = Path(self.build_lib) / 'weftmoe/_native_source'
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / 'build_native.py', destination / 'build_native.py')
        for name in ('include', 'src', 'configs/profiles'):
            shutil.copytree(ROOT / name, destination / name, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        (destination / 'tools').mkdir(exist_ok=True)
        shutil.copy2(ROOT / 'tools/build_tensor_runtime.py', destination / 'tools/build_tensor_runtime.py')


setup(cmdclass={'build_py': SourceBuild})
