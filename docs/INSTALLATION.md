# Installation

Run commands from the repository root. Python 3.11 or later is required by the experiment launchers. Plotting dependencies are listed in `Figures/requirements-figures.txt`.

The native programs require CMake 3.20 or later, a C++17 compiler, LibTorch and Intel oneMKL. Bratu training also requires OpenMP. Use a LibTorch distribution matching the compiler, platform and C++ ABI. The code has been built with CPU LibTorch 2.10 and oneMKL 2025 on Windows and Linux.

## Python

```text
python -m venv .venv
```

Activate `.venv`: `.venv\Scripts\Activate.ps1` in PowerShell, or `source .venv/bin/activate` on Linux. Then install the plotting dependencies:

```text
python -m pip install -r Figures/requirements-figures.txt
```

Scripts that evaluate TorchScript models additionally require the Python `torch` package.

## Windows

Use the Visual Studio C++ build tools. Substitute the locations of LibTorch and oneMKL in the following commands. The MKL directory must contain `MKLConfig.cmake`.

```text
cmake -S src/TwoPhaseTransport -B build/transport -DCRIS_TORCH_ROOT=C:/libraries/libtorch -DMKL_DIR=C:/libraries/mkl/lib/cmake/mkl
cmake --build build/transport --config Release --target tp_transport --parallel 2

cmake -S training/twophasetransport -B build/transport_training -DTWOPHASETRANSPORT_TORCH_ROOT=C:/libraries/libtorch -DMKL_DIR=C:/libraries/mkl/lib/cmake/mkl
cmake --build build/transport_training --config Release --target twophasetransport_lm_trainer twophasetransport_checkpoint_exporter --parallel 2

cmake -S training/bratu -B build/bratu_training -DCMAKE_PREFIX_PATH=C:/libraries/libtorch -DMKL_DIR=C:/libraries/mkl/lib/cmake/mkl
cmake --build build/bratu_training --config Release --target bratu_lm_trainer --parallel 2
```

The executables are in each build directory's `Release` subdirectory. Ensure LibTorch, MKL and OpenMP runtime libraries are available to the process. Training accepts repeatable `--runtime-dir` arguments; the benchmark launcher accepts one `--dll-dir` in addition to directories already on `PATH`.

## Linux

For a GNU toolchain, configure the MKL interface and threading library consistently:

```text
cmake -S src/TwoPhaseTransport -B build/transport -DCMAKE_BUILD_TYPE=Release -DCRIS_TORCH_ROOT=/opt/libtorch -DMKL_DIR=/opt/intel/oneapi/mkl/latest/lib/cmake/mkl -DMKL_INTERFACE=lp64 -DMKL_THREADING=gnu_thread
cmake --build build/transport --target tp_transport --parallel 2

cmake -S training/twophasetransport -B build/transport_training -DCMAKE_BUILD_TYPE=Release -DTWOPHASETRANSPORT_TORCH_ROOT=/opt/libtorch -DMKL_DIR=/opt/intel/oneapi/mkl/latest/lib/cmake/mkl -DMKL_INTERFACE=lp64 -DMKL_THREADING=gnu_thread
cmake --build build/transport_training --target twophasetransport_lm_trainer twophasetransport_checkpoint_exporter --parallel 2

cmake -S training/bratu -B build/bratu_training -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/libtorch -DMKL_DIR=/opt/intel/oneapi/mkl/latest/lib/cmake/mkl -DMKL_INTERFACE=lp64 -DMKL_THREADING=gnu_thread
cmake --build build/bratu_training --target bratu_lm_trainer --parallel 2
```

These executables are directly in the build directories, without a `Release` subdirectory or `.exe` suffix. Make the shared libraries discoverable through the system loader or `LD_LIBRARY_PATH`. The training launcher can also set the library path through `--runtime-dir`.
