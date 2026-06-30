import sys
import platform
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

if platform.system() != "Darwin":
    sys.exit(
        "pyoomph_mac_accelerate only builds on macOS: it links directly against the "
        "Accelerate framework's Sparse Solvers API, which does not exist "
        "on other platforms."
    )

ext_modules = [
    Pybind11Extension(
        "pyoomph_mac_accelerate._pyoomph_mac_accelerate",
        ["src/bindings.cpp"],
        cxx_std=17,
        extra_compile_args=["-O3", "-fvisibility=hidden"],
        # Link against the Accelerate umbrella framework, which contains
        # the Sparse Solvers C API (SparseFactor / SparseSolve / ...).
        extra_link_args=["-framework", "Accelerate"],
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
