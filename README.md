# GPU Projects

CUDA benchmarks and numerical linear-algebra solvers with reproducible CMake builds.

[![CUDA](https://img.shields.io/badge/CUDA-C%2B%2B-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white)](https://isocpp.org/)
[![CMake](https://img.shields.io/badge/build-CMake-064F8C?logo=cmake&logoColor=white)](https://cmake.org/)
[![GPU](https://img.shields.io/badge/benchmarked-NVIDIA%20V100-76B900)](#performance-highlights)

This repository collects GPU implementations of sparse matrix-vector multiplication, stationary and Krylov iterative methods, FP64 GEMM, and adapters for four sparse direct solvers. The default build contains only CUDA Toolkit dependencies; third-party sparse solvers are opt-in.

## Quick start

Prerequisites: CMake 3.24+, a C++17 compiler, the CUDA Toolkit, and an NVIDIA GPU.

```bash
git clone https://github.com/2002WYT/GPU_projects.git
cd GPU_projects
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The default architecture is `sm_70` for the NVIDIA V100 used in the published benchmarks. Override it for another GPU, for example:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80
```

CTest runs small GPU smoke tests for SpMV, Jacobi, multicolor Gauss–Seidel, and CG. A CUDA-capable GPU must be visible when the tests run.

## Projects

| Project | What it demonstrates | Default target |
|---|---|---|
| [SpMV](./spmv) | Five CSR SpMV kernels, correctness checks, timing, GFLOP/s, and effective bandwidth | `spmv_compare` |
| [Jacobi](./jacobi) | Cooperative Groups, device-side convergence, and double buffering | `jacobi_solver` |
| [Gauss–Seidel](./gauss-seidel) | Graph coloring, SOR, CUB reduction, and CUDA Graph replay | `gauss_seidel_solver` |
| [Conjugate Gradient](./CG) | CSR SpMV, dot products, AXPY/AXPBY, and true-residual verification | `cg_solver` |
| [GEMM](./GEMM) | FP64 kernel autotuning, selected kernels, Matrix Market input, and cuBLAS comparison | `gemm_autotune`, `gemm_selected`, `gemm_mtx` |
| [Sparse solver benchmarks](./Four%20sparse%20solvers) | SuperLU_DIST, PanguLU, cuDSS, and STRUMPACK adapters | opt-in |

Detailed implementation notes and benchmark methodology live inside each project directory:

- [SpMV implementation notes](./spmv/spmv_compare.md) and [V100 results](./spmv/spmv_result.md)
- [Jacobi solver notes](./jacobi/README.md)
- [Gauss–Seidel/SOR notes](./gauss-seidel/README.md)
- [Conjugate Gradient notes](./CG/README.md)
- [Four-solver benchmark report](./Four%20sparse%20solvers/four%20sparse%20solvers.md)

## Build options

Every first-party project has a top-level switch:

| Option | Default |
|---|---:|
| `GPU_PROJECTS_BUILD_SPMV` | `ON` |
| `GPU_PROJECTS_BUILD_JACOBI` | `ON` |
| `GPU_PROJECTS_BUILD_GAUSS_SEIDEL` | `ON` |
| `GPU_PROJECTS_BUILD_CG` | `ON` |
| `GPU_PROJECTS_BUILD_GEMM` | `ON` |
| `GPU_PROJECTS_BUILD_SOLVERS` | `OFF` |

For example, build only SpMV and CG:

```bash
cmake -S . -B build \
  -DGPU_PROJECTS_BUILD_SPMV=ON \
  -DGPU_PROJECTS_BUILD_JACOBI=OFF \
  -DGPU_PROJECTS_BUILD_GAUSS_SEIDEL=OFF \
  -DGPU_PROJECTS_BUILD_CG=ON \
  -DGPU_PROJECTS_BUILD_GEMM=OFF
cmake --build build -j
```

## Run examples

```bash
./build/spmv_compare 200000 20 3 1 256
./build/jacobi_solver 10000
./build/gauss_seidel_solver 100000
./build/cg_solver 100000
./build/GEMM/gemm_autotune
```

Run `./build/spmv_compare --help` for the SpMV matrix modes and benchmark arguments. The GEMM executables document their accepted dimensions and Matrix Market inputs in their `--help` output.

## Optional sparse direct solvers

The four external adapters are disabled by default because they require separate installations. Enable the group and pass installation prefixes as CMake cache variables instead of editing repository files:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPU_PROJECTS_BUILD_SOLVERS=ON \
  -DSUPERLU_DIST_ROOT=/path/to/superlu_dist \
  -DPANGULU_ROOT=/path/to/pangulu \
  -DCUDSS_ROOT=/path/to/cudss \
  -DSTRUMPACK_ROOT=/path/to/strumpack \
  -DSCALAPACK_ROOT=/path/to/scalapack \
  -DOPENBLAS_ROOT=/path/to/openblas
cmake --build build -j
```

Individual adapters can be disabled when only part of the dependency stack is installed:

```bash
cmake -S . -B build \
  -DGPU_PROJECTS_BUILD_SOLVERS=ON \
  -DGPU_SOLVERS_BUILD_SUPERLU_DIST=OFF \
  -DGPU_SOLVERS_BUILD_PANGULU=OFF \
  -DGPU_SOLVERS_BUILD_CUDSS=ON \
  -DGPU_SOLVERS_BUILD_STRUMPACK=OFF \
  -DCUDSS_ROOT=/path/to/cudss
```

The resulting executables are placed in `build/bin`. A small Matrix Market input is included at `Four sparse solvers/files/AA1.mtx`.

```bash
mpirun -n 1 ./build/bin/test_slu "Four sparse solvers/files/AA1.mtx"
mpirun -n 1 ./build/bin/test_pangulu "Four sparse solvers/files/AA1.mtx"
mpirun -n 1 ./build/bin/test_cudss "Four sparse solvers/files/AA1.mtx"
mpirun -n 1 ./build/bin/test_strumpack "Four sparse solvers/files/AA1.mtx"
```

The adapters share the repository's `mtx_reader.h`, which accepts real, integer, and pattern coordinate matrices and expands Matrix Market symmetric storage into zero-based full CSR.

## Performance highlights

The following results were measured on an NVIDIA Tesla V100-SXM2-16GB with `n = 2,000,000`. See the [full SpMV report](./spmv/spmv_result.md) for the test procedure and raw comparisons.

### Regular row-length distribution

| Kernel | Best block size | Average time | GFLOP/s | Speedup over CSR Scalar |
|---|---:|---:|---:|---:|
| CSR Vector | 512 | 5.13 ms | 32.61 | 3.04× |
| CSR Adaptive | 512 | 5.13 ms | 32.61 | 3.04× |
| LightSpMV | 256 | 5.20 ms | 32.18 | 3.00× |
| PCSR | 128 | 7.71 ms | 21.72 | 2.02× |
| CSR Scalar | 128 | 15.61 ms | 10.72 | 1.00× |

### Heavy-tailed row-length distribution

| Kernel | Best block size | Average time | GFLOP/s | Speedup over CSR Scalar |
|---|---:|---:|---:|---:|
| CSR Adaptive | 512 | 7.60 ms | 28.86 | 2.00× |
| CSR Vector | 512 | 7.86 ms | 27.92 | 1.94× |
| LightSpMV | 256 | 7.94 ms | 27.65 | 1.92× |
| PCSR | 512 | 11.85 ms | 18.52 | 1.28× |
| CSR Scalar | 128 | 15.22 ms | 14.42 | 1.00× |

These measurements show why the repository reports both matrix structure and kernel throughput: CSR Adaptive becomes the best option when a small number of rows contain much more work.

## Benchmarking conventions

- Kernels run warm-up iterations before timing.
- CUDA Events measure GPU work.
- Repeated runs report average execution time.
- Numerical programs report correctness alongside performance.
- Iterative solvers use the relative residual `||b-Ax||₂ / ||b||₂`.
- Direct-solver reports distinguish reorder, factorization, solve, and external wall-clock time where the APIs expose those phases.

Example profiling commands:

```bash
nsys profile --trace=cuda,nvtx,osrt --stats=true \
  -o spmv_nsys ./build/spmv_compare 200000 10 2 1 256

ncu --section full -o spmv_ncu \
  ./build/spmv_compare 200000 1 0 1 256
```

## Repository layout

```text
GPU_projects/
├── CMakeLists.txt
├── spmv/
├── jacobi/
├── gauss-seidel/
├── CG/
├── GEMM/
└── Four sparse solvers/
```

## Scope and limitations

- The default examples generate synthetic matrices; only selected programs accept Matrix Market input.
- The iterative methods are educational and benchmarking implementations, not a replacement for a production solver package.
- The published performance numbers describe the recorded V100 software/hardware setup; re-run the benchmarks before comparing another GPU or CUDA version.
- Third-party sparse solvers retain their own build requirements and licenses.

## License

No repository-wide license has been selected yet. Until a license is added, the source is publicly visible but reuse rights are not granted automatically.
