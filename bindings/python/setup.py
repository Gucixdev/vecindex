"""Builds the vecindex C library into the package as a plain shared object, which vecindex/__init__.py loads with
ctypes: no Python C API, so one build serves every Python 3 version on the platform.

The C sources are vecindex's own src/ and include/. From the repository they are two levels up; an sdist carries a
copy in csrc/ (made here). VECINDEX_PORTABLE=1 builds without -march=native, for wheels meant for other machines.
"""
import os
import shutil

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.sdist import sdist

try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:                                   # setuptools before 70.1: the wheel package's
    from wheel.bdist_wheel import bdist_wheel

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
VENDOR = os.path.join(HERE, "csrc")


def csrc():
    """The directory holding src/ and include/: the repository's, else the copy an sdist carries."""
    if os.path.isfile(os.path.join(REPO, "src", "vecindex.c")):
        return REPO
    if os.path.isfile(os.path.join(VENDOR, "src", "vecindex.c")):
        return VENDOR
    raise SystemExit("vecindex: no C sources (src/vecindex.c) beside this package")


def vendor():
    """A copy of src/ and include/ in csrc/, for an sdist that has to build away from the repository."""
    if os.path.isdir(VENDOR):
        shutil.rmtree(VENDOR)
    for d in ("src", "include"):
        shutil.copytree(os.path.join(REPO, d), os.path.join(VENDOR, d))
    shutil.copy(os.path.join(REPO, "LICENSE"), os.path.join(VENDOR, "LICENSE"))


class BuildLib(build_ext):
    """A shared object without a module init: nothing to export but the library's own vi_* symbols."""

    def get_export_symbols(self, ext):
        return []


class Wheel(bdist_wheel):
    """py3-none-PLATFORM: the library is plain C loaded by ctypes, so one wheel serves every Python 3."""

    def get_tag(self):
        _, _, plat = super().get_tag()
        return "py3", "none", plat


class Sdist(sdist):
    def run(self):
        if os.path.isfile(os.path.join(REPO, "src", "vecindex.c")):
            vendor()
        super().run()


ROOT = csrc()
FLAGS = ["-O2", "-std=gnu99", "-fvisibility=hidden", "-Wno-unused-function"]
if os.environ.get("VECINDEX_PORTABLE") != "1":
    FLAGS.append("-march=native")

setup(
    ext_modules=[
        Extension(
            "vecindex._libvecindex",
            sources=[os.path.relpath(os.path.join(ROOT, "src", "vecindex.c"), HERE)],
            include_dirs=[os.path.join(ROOT, "include"), os.path.join(ROOT, "src")],
            extra_compile_args=FLAGS,
            libraries=["m", "pthread"],
        )
    ],
    cmdclass={"build_ext": BuildLib, "sdist": Sdist, "bdist_wheel": Wheel},
)
