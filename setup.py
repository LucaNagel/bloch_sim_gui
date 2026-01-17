# setup.py - Build configuration for Bloch simulator
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import platform
import os
import sys

# Detect platform for compiler flags
# Check if sys.platform is emscripten OR if CC env var points to emcc (cross-compilation)
# Also check CFLAGS for wasm/emscripten targets (common in pyodide build)
cflags = os.environ.get("CFLAGS", "").lower()
sysconfig_name = os.environ.get("_PYTHON_SYSCONFIGDATA_NAME", "").lower()
is_emscripten = (
    (sys.platform == "emscripten")
    or ("emcc" in os.environ.get("CC", ""))
    or ("wasm" in cflags)
    or ("emscripten" in cflags)
    or ("EMSCRIPTEN" in os.environ)
    or ("emscripten" in sysconfig_name)
)
is_windows = platform.system() == "Windows"
is_mac = platform.system() == "Darwin"
is_linux = platform.system() == "Linux"

# Compiler and linker flags
extra_compile_args = []
extra_link_args = []
define_macros = []

# Architecture optimization flags
arch_flags = []
# Avoid -mcpu=native or -march=native for wheel builds to ensure portability

if is_emscripten:
    # WebAssembly build (Emscripten)
    # Disable OpenMP for initial compatibility (single-threaded)
    extra_compile_args = ["-O3", "-ffast-math"]
    extra_link_args = ["-lm"]
elif is_windows:
    # Windows with MSVC
    extra_compile_args = ["/openmp", "/O2"]
    extra_link_args = []
elif is_mac:
    # macOS with clang; try OpenMP if libomp is available
    libomp_paths = [
        "/opt/homebrew/opt/libomp",  # Apple Silicon Homebrew
    ]

    libomp_root = next((p for p in libomp_paths if os.path.exists(p)), None)
    if libomp_root:
        libomp_lib = os.path.join(libomp_root, "lib")
        extra_compile_args = [
            "-Xpreprocessor",
            "-fopenmp",
            "-O3",
            "-ffast-math",
            f'-I{os.path.join(libomp_root, "include")}',
        ] + arch_flags
        extra_link_args = ["-lomp", f"-L{libomp_lib}"]
    else:
        # Build without OpenMP; prange falls back to serial execution
        extra_compile_args = ["-O3", "-ffast-math"] + arch_flags
        extra_link_args = []
else:
    # Linux with gcc
    extra_compile_args = ["-fopenmp", "-O3", "-ffast-math"] + arch_flags
    extra_link_args = ["-fopenmp", "-lm"]

# Define extension without numpy include first (added in CustomBuildExt)
extensions = [
    Extension(
        "blochsimulator.blochsimulator_cy",
        sources=[
            "src/blochsimulator/bloch_wrapper.pyx",
            "src/blochsimulator/bloch_core_modified.c",
        ],
        include_dirs=["src/blochsimulator"],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c",
    )
]


# Custom build command to lazy-import numpy
class CustomBuildExt(build_ext):
    def build_extensions(self):
        # Prevent numpy import at top level to allow pip to install it first
        import numpy

        numpy_include = numpy.get_include()

        for ext in self.extensions:
            if numpy_include not in ext.include_dirs:
                ext.include_dirs.append(numpy_include)

        super().build_extensions()


setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    ),
    include_package_data=True,
    zip_safe=False,
    cmdclass={"build_ext": CustomBuildExt},
    # Dependencies are handled by pyproject.toml [project] table
)
