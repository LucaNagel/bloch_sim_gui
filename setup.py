# setup.py - Build configuration for Bloch simulator
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import platform
import os

# Detect platform for compiler flags
is_windows = platform.system() == 'Windows'
is_mac = platform.system() == 'Darwin'
is_linux = platform.system() == 'Linux'

# Compiler and linker flags
extra_compile_args = []
extra_link_args = []
define_macros = []

# Architecture optimization flags
arch_flags = []
machine = platform.machine().lower()
if is_mac and 'arm' in machine:
    # Apple Silicon
    arch_flags = ['-mcpu=native']
elif not is_windows:
    # Intel/AMD on Linux/Mac
    arch_flags = ['-march=native']

if is_windows:
    # Windows with MSVC
    extra_compile_args = ['/openmp', '/O2']
    extra_link_args = []
elif is_mac:
    # macOS with clang; try OpenMP if libomp is available
    libomp_paths = [
        '/usr/local/opt/libomp',      # Intel Homebrew
        '/opt/homebrew/opt/libomp',   # Apple Silicon Homebrew
    ]

    libomp_root = next((p for p in libomp_paths if os.path.exists(p)), None)
    if libomp_root:
        extra_compile_args = ['-Xpreprocessor', '-fopenmp', '-O3', '-ffast-math',
                              f'-I{os.path.join(libomp_root, "include")}'] + arch_flags
        extra_link_args = ['-lomp', f'-L{os.path.join(libomp_root, "lib")}']
    else:
        # Build without OpenMP; prange falls back to serial execution
        extra_compile_args = ['-O3', '-ffast-math'] + arch_flags
        extra_link_args = []
else:
    # Linux with gcc
    extra_compile_args = ['-fopenmp', '-O3', '-ffast-math'] + arch_flags
    extra_link_args = ['-fopenmp', '-lm']

# Define the extension
extensions = [
    Extension(
        "bloch_simulator_cy",
        sources=["bloch_wrapper.pyx", "bloch_core_modified.c"],
        include_dirs=[np.get_include(), "."],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c"
    )
]

setup(
    name="bloch_simulator",
    version="1.0.0",
    description="High-performance Bloch equation simulator for MRI",
    author="Your Name",
    ext_modules=cythonize(extensions, 
                          compiler_directives={
                              'language_level': "3",
                              'boundscheck': False,
                              'wraparound': False,
                              'cdivision': True,
                          }),
    zip_safe=False,
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
        "cython>=0.29.0",
    ],
    python_requires=">=3.7",
)

# Build instructions:
# python setup.py build_ext --inplace