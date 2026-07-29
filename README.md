# GPU Projects

<p align="center">
  <strong>A growing collection of CUDA, GPU computing, and high-performance numerical computing projects</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/CUDA-C%2B%2B-76B900?logo=nvidia&logoColor=white" alt="CUDA">
  <img src="https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white" alt="C++17">
  <img src="https://img.shields.io/badge/Platform-Linux-lightgrey" alt="Linux">
  <img src="https://img.shields.io/badge/Focus-GPU%20Computing-blue" alt="GPU Computing">
</p>

---

## Overview

This repository is a growing collection of GPU computing projects focused on CUDA programming, numerical linear algebra, sparse matrix algorithms, iterative methods, direct solvers, and performance analysis.

The repository is intentionally organized as a set of independent subprojects. New projects can be added at any time without redesigning the top-level structure.

Current topics include:

- Sparse Matrix-Vector Multiplication (SpMV)
- Jacobi iteration
- Multicolor Gauss-Seidel and SOR
- Conjugate Gradient
- General Matrix Multiplication (GEMM)
- Sparse direct solver benchmarking
- CUDA kernel optimization
- Nsight Systems and Nsight Compute profiling

Each subproject may contain its own source code, build files, documentation, examples, scripts, input data, and benchmark results.

---

## Project Index

| Project | Description | Main Topics | Documentation |
|---|---|---|---|
| [SpMV](./spmv) | Comparison of five GPU CSR SpMV implementations | Sparse matrices, warp-level reduction, adaptive scheduling, memory bandwidth | [Documentation](./spmv/spmv_compare.md) |
| [Jacobi](./jacobi) | GPU Jacobi iterative solver | Cooperative Groups, residual evaluation, double buffering | [Documentation](./jacobi/README.md) |
| [Gauss-Seidel](./gauss-seidel) | Multicolor Gauss-Seidel and SOR solver | Graph coloring, CUDA Graphs, CUB reduction | [Documentation](./gauss-seidel/README.md) |
| [Conjugate Gradient](./CG) | CUDA implementation of the Conjugate Gradient method | Krylov methods, SpMV, dot product, AXPY | [Documentation](./CG/README.md) |
| [GEMM](./GEMM) | Custom CUDA GEMM and cuBLAS comparison | Shared-memory tiling, DGEMM, GFLOP/s | [Source Directory](./GEMM) |
| [Four Sparse Solvers](./Four%20sparse%20solvers) | Unified tests for four sparse direct solvers | SuperLU_DIST, PanguLU, cuDSS, STRUMPACK | [Benchmark Report](./Four%20sparse%20solvers/four%20sparse%20solvers.md) |

> Add each new project to this table. The rest of this README can remain unchanged unless the new project needs a dedicated overview section.

---

## Repository Structure

```text
GPU_projects/
├── README.md
│
├── spmv/
│   ├── spmv_compare_complete.cu
│   ├── spmv_compare.md
│   └── spmv_result.md
│
├── jacobi/
│   ├── ex_jacobi.cu
│   ├── src/
│   └── README.md
│
├── gauss-seidel/
│   ├── include/
│   ├── src/
│   ├── ex_gs.cu
│   └── README.md
│
├── CG/
│   ├── include/
│   ├── src/
│   ├── ex_CG.cu
│   └── README.md
│
├── GEMM/
│   ├── gemm_compare.cu
│   └── gemm_cuda_simple.cu
│
├── Four sparse solvers/
│   ├── CMakeLists.txt
│   ├── test_superlu_dist.cpp
│   ├── test_pangulu.cpp
│   ├── test_cudss.cpp
│   ├── test_strumpack.cpp
│   ├── run_all.sh
│   ├── files/
│   └── four sparse solvers.md
│
└── future-project/
    ├── README.md
    ├── CMakeLists.txt
    ├── include/
    ├── src/
    ├── examples/
    ├── scripts/
    ├── data/
    └── results/
```

The repository does not currently use one global build system. Each subproject can be compiled independently.

---

## Current Projects

### 1. Sparse Matrix-Vector Multiplication

Directory: [`spmv/`](./spmv)

This project implements and compares five GPU CSR SpMV algorithms:

1. CSR Scalar
2. CSR Vector
3. CSR Adaptive
4. PCSR
5. LightSpMV-style dynamic scheduling

The benchmark reports:

- Average execution time
- GFLOP/s
- Approximate effective memory bandwidth
- Maximum absolute error
- Maximum relative error
- Fastest kernel
- Correctness status

#### Build

```bash
mkdir -p build

nvcc -O3 \
    -std=c++17 \
    -arch=sm_70 \
    spmv/spmv_compare_complete.cu \
    -o build/spmv_compare
```

`sm_70` targets NVIDIA V100. Change the architecture flag when using a different GPU.

#### Run

```bash
./build/spmv_compare
```

Full command-line format:

```bash
./build/spmv_compare \
    [n] \
    [repeats] \
    [warmup] \
    [matrix_mode] \
    [block_size]
```

Example:

```bash
./build/spmv_compare 200000 20 3 1 256
```

For a smaller Nsight Compute run:

```bash
./build/spmv_compare 200000 1 0 1 256
```

More information:

- [Implementation Notes](./spmv/spmv_compare.md)
- [Performance Report](./spmv/spmv_result.md)

---

### 2. GPU Jacobi Solver

Directory: [`jacobi/`](./jacobi)

The Jacobi method solves

```math
Ax=b
```

using the update

```math
x_i^{(k+1)}
=
\frac{
b_i-\sum_{j\ne i}a_{ij}x_j^{(k)}
}{
a_{ii}
}.
```

Main implementation features:

- CSR sparse matrix storage
- Double buffering
- Shared-memory reduction
- Cooperative Groups synchronization
- GPU-side residual computation
- GPU-side convergence control

The current source references `head.cuh` and `jacobi.cuh`. Make sure both headers exist in the local branch before compiling.

#### Example Build

```bash
nvcc -O3 \
    -std=c++17 \
    -arch=sm_70 \
    -rdc=true \
    -I./jacobi/include \
    jacobi/ex_jacobi.cu \
    jacobi/src/jacobi.cu \
    -o build/jacobi_solver
```

#### Run

```bash
./build/jacobi_solver 10000
```

More information:

- [Jacobi Documentation](./jacobi/README.md)

---

### 3. Multicolor Gauss-Seidel and SOR

Directory: [`gauss-seidel/`](./gauss-seidel)

Standard Gauss-Seidel contains sequential data dependencies. This project uses graph coloring to divide matrix rows into independent color groups.

Rows with the same color can be updated in parallel on the GPU.

The relaxed update is

```math
x_i
\leftarrow
(1-\omega)x_i
+
\frac{\omega}{a_{ii}}
\left(
b_i-\sum_{j\ne i}a_{ij}x_j
\right).
```

- `omega = 1` gives multicolor Gauss-Seidel
- `0 < omega < 2` gives an SOR-style iteration

Main features:

- General CSR input
- CPU greedy graph coloring
- Diagonal extraction
- Thread-per-row kernel
- Warp-per-row kernel
- CUB residual reduction
- CUDA Graph replay
- Configurable relaxation parameter

#### Build

```bash
nvcc -O3 \
    -std=c++17 \
    -arch=sm_70 \
    -I./gauss-seidel/include \
    gauss-seidel/ex_gs.cu \
    gauss-seidel/src/gs.cu \
    -o build/gs_solver
```

#### Run

```bash
./build/gs_solver 1000000
```

More information:

- [Gauss-Seidel Documentation](./gauss-seidel/README.md)

---

### 4. Conjugate Gradient Solver

Directory: [`CG/`](./CG)

This project implements the Conjugate Gradient method for symmetric positive-definite systems.

The main GPU operations are:

- CSR SpMV
- Dot product
- AXPY
- AXPBY
- Vector scaling
- Relative residual evaluation

The core iteration is based on

```math
\alpha_k
=
\frac{r_k^Tr_k}{p_k^TAp_k},
```

```math
x_{k+1}
=
x_k+\alpha_kp_k,
```

```math
r_{k+1}
=
r_k-\alpha_kAp_k,
```

```math
\beta_k
=
\frac{r_{k+1}^Tr_{k+1}}{r_k^Tr_k},
```

```math
p_{k+1}
=
r_{k+1}+\beta_kp_k.
```

#### Build

The current common header includes cuDSS headers, so the cuDSS include path must be available.

```bash
export CUDSS_ROOT=$HOME/cudss_install/nvidia/cu12

nvcc -O3 \
    -std=c++17 \
    -arch=sm_70 \
    -I./CG/include \
    -I"$CUDSS_ROOT/include" \
    CG/ex_CG.cu \
    CG/src/CG.cu \
    CG/src/math.cu \
    -o build/cg_solver
```

#### Run

```bash
./build/cg_solver
```

Or specify the matrix size:

```bash
./build/cg_solver 100000
```

More information:

- [Conjugate Gradient Documentation](./CG/README.md)

---

### 5. GEMM

Directory: [`GEMM/`](./GEMM)

This project studies dense matrix multiplication:

```math
C=AB.
```

The directory contains two programs.

#### `gemm_compare.cu`

Compares a custom shared-memory tiled CUDA kernel with `cublasDgemm`.

Build:

```bash
nvcc -O3 \
    -std=c++17 \
    -arch=sm_70 \
    GEMM/gemm_compare.cu \
    -lcublas \
    -o build/gemm_compare
```

Run:

```bash
./build/gemm_compare 1024 1024 1024 100
```

The program reports:

- Custom kernel time
- Custom kernel GFLOP/s
- cuBLAS time
- cuBLAS GFLOP/s
- Relative error

#### `gemm_cuda_simple.cu`

Runs `cublasDgemm` with random matrices or Matrix Market input.

Build:

```bash
nvcc -O3 \
    -std=c++17 \
    -arch=sm_70 \
    GEMM/gemm_cuda_simple.cu \
    -lcublas \
    -o build/gemm_cuda_simple
```

Run with default random matrices:

```bash
./build/gemm_cuda_simple
```

Run with custom dimensions:

```bash
./build/gemm_cuda_simple 2048 2048 2048 50
```

Run with Matrix Market input:

```bash
./build/gemm_cuda_simple A.mtx 32 50
```

or

```bash
./build/gemm_cuda_simple A.mtx B.mtx 50
```

---

### 6. Sparse Direct Solver Benchmark

Directory: [`Four sparse solvers/`](./Four%20sparse%20solvers)

This project provides a unified testing framework for:

- SuperLU_DIST
- PanguLU
- NVIDIA cuDSS
- STRUMPACK

The test programs perform the following steps:

1. Read a Matrix Market file
2. Expand symmetric matrices when required
3. Construct a known exact solution
4. Generate the right-hand side
5. Run the solver
6. Record timing information
7. Compute the relative residual
8. Compute the relative solution error
9. Report success or failure

#### Dependencies

The exact dependencies depend on the solver configuration, but may include:

- CUDA Toolkit
- MPI
- SuperLU_DIST
- PanguLU
- NVIDIA cuDSS
- STRUMPACK
- ScaLAPACK
- OpenBLAS
- METIS
- ParMETIS
- GKlib

#### Configure Paths

The current `CMakeLists.txt` contains machine-specific installation paths.

Update the following variables before building:

```cmake
set(CUDA_ROOT         /path/to/cuda)
set(CUDSS_ROOT        /path/to/cudss)
set(MPI_HOME          /path/to/mpi)
set(SUPERLU_DIST_ROOT /path/to/superlu_dist)
set(PANGULU_ROOT      /path/to/pangulu)
set(STRUMPACK_ROOT    /path/to/strumpack)
set(SCALAPACK_ROOT    /path/to/scalapack)
set(OPENBLAS_ROOT     /path/to/openblas)
```

Also update the CUDA architecture when necessary:

```cmake
set(CMAKE_CUDA_ARCHITECTURES 70)
```

#### Build

```bash
cd "Four sparse solvers"

cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build -j
```

#### Run

The repository includes a small Matrix Market test matrix:

```text
files/AA1.mtx
```

Run each solver independently:

```bash
mpirun -n 1 ./build/test_slu       ./files/AA1.mtx
mpirun -n 1 ./build/test_pangulu   ./files/AA1.mtx
mpirun -n 1 ./build/test_cudss     ./files/AA1.mtx
mpirun -n 1 ./build/test_strumpack ./files/AA1.mtx
```

The provided `run_all.sh` script contains a machine-specific path and should be updated before use.

More information:

- [Sparse Direct Solver Benchmark Report](./Four%20sparse%20solvers/four%20sparse%20solvers.md)

---

## Performance Highlights

### SpMV Results on NVIDIA V100

Test configuration:

- Matrix size: `n = 2,000,000`
- CUDA block sizes: 128, 256, and 512

#### Regular Row-Length Pattern

| Algorithm | Best Block Size | Average Time | GFLOP/s | Speedup over CSR Scalar |
|---|---:|---:|---:|---:|
| CSR Vector | 512 | 5.13 ms | 32.61 | 3.04x |
| CSR Adaptive | 512 | 5.13 ms | 32.61 | 3.04x |
| LightSpMV | 256 | 5.20 ms | 32.18 | 3.00x |
| PCSR | 128 | 7.71 ms | 21.72 | 2.02x |
| CSR Scalar | 128 | 15.61 ms | 10.72 | 1.00x |

#### Heavy-Tailed Row-Length Pattern

| Algorithm | Best Block Size | Average Time | GFLOP/s | Speedup over CSR Scalar |
|---|---:|---:|---:|---:|
| CSR Adaptive | 512 | 7.60 ms | 28.86 | 2.00x |
| CSR Vector | 512 | 7.86 ms | 27.92 | 1.94x |
| LightSpMV | 256 | 7.94 ms | 27.65 | 1.92x |
| PCSR | 512 | 11.85 ms | 18.52 | 1.28x |
| CSR Scalar | 128 | 15.22 ms | 14.42 | 1.00x |

Main observations:

- CSR Vector is a stable general-purpose implementation
- CSR Adaptive performs well on matrices containing very long rows
- Dynamic scheduling can reduce load imbalance
- PCSR introduces additional intermediate memory traffic
- High occupancy does not always imply high performance
- SpMV is primarily memory-bandwidth limited

---

## Benchmarking Principles

This repository attempts to report both correctness and performance.

### Relative Residual

```math
\frac{\|b-Ax\|_2}{\|b\|_2}
```

### Relative Solution Error

```math
\frac{\|x-x^\ast\|_2}{\|x^\ast\|_2}
```

A small residual does not always guarantee a small forward error, especially for ill-conditioned matrices.

### GPU Timing

CUDA kernels are typically measured with CUDA Events:

```cpp
cudaEventRecord(start);

// kernel launches

cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&milliseconds, start, stop);
```

### Warmup

Each benchmark should execute one or more warmup runs before timed runs.

### Repetition

Kernel execution should be repeated multiple times, and average or median runtime should be reported.

### Timing Boundaries

When comparing programs, clearly state whether timing includes:

- File input
- Matrix construction
- Memory allocation
- Host-to-device transfer
- Reordering
- Symbolic analysis
- Numerical factorization
- Solve
- Device-to-host transfer
- Process startup
- Dynamic library loading

Results with different timing boundaries should not be compared directly.

---

## Profiling

### Nsight Systems

Use Nsight Systems to inspect:

- CUDA API calls
- Kernel timelines
- Memory transfers
- CPU and GPU overlap
- Synchronization points

Example:

```bash
nsys profile \
    --trace=cuda,nvtx,osrt \
    --stats=true \
    -o spmv_nsys \
    ./build/spmv_compare 200000 10 2 1 256
```

### Nsight Compute

Use Nsight Compute to inspect:

- DRAM throughput
- SM throughput
- Achieved occupancy
- Warp stalls
- Branch efficiency
- Memory access efficiency
- Shared-memory usage

Example:

```bash
ncu \
    --section full \
    -o spmv_ncu \
    ./build/spmv_compare 200000 1 0 1 256
```

---

## Suggested Learning Path

A useful reading order is:

```text
CSR sparse matrix storage
    ↓
CSR Scalar SpMV
    ↓
CSR Vector and adaptive SpMV
    ↓
Jacobi iteration
    ↓
Multicolor Gauss-Seidel and SOR
    ↓
Conjugate Gradient
    ↓
Dense GEMM and cuBLAS
    ↓
Sparse direct solvers
    ↓
Nsight profiling and bottleneck analysis
```

Recommended documentation order:

1. [SpMV Implementation](./spmv/spmv_compare.md)
2. [Jacobi Solver](./jacobi/README.md)
3. [Gauss-Seidel Solver](./gauss-seidel/README.md)
4. [Conjugate Gradient Solver](./CG/README.md)
5. [GEMM Directory](./GEMM)
6. [Sparse Direct Solver Report](./Four%20sparse%20solvers/four%20sparse%20solvers.md)

---

## Adding a New Project

Add every new project as an independent top-level directory.

Recommended structure:

```text
new-project/
├── README.md
├── CMakeLists.txt
├── include/
├── src/
├── examples/
├── scripts/
├── data/
└── results/
```

A project README should include:

- Project purpose
- Algorithm or method
- Directory structure
- Dependencies
- Build instructions
- Run instructions
- Input format
- Output format
- Correctness checks
- Benchmark methodology
- Known limitations
- Future work

After adding a new project:

1. Add one row to the **Project Index**
2. Add the directory to **Repository Structure**
3. Add a dedicated section only when the project needs top-level visibility
4. Add performance results only when they are reproducible
5. Avoid machine-specific absolute paths
6. Keep detailed implementation notes inside the project directory

This structure allows the repository to grow without requiring a complete rewrite of the top-level README.

---

## Reusable Project Entry Template

Copy and edit the following block when adding a project to the **Project Index**:

```markdown
| [Project Name](./project-directory) | One-sentence description | Main technologies or algorithms | [Documentation](./project-directory/README.md) |
```

Optional top-level project section:

```markdown
### Project Name

Directory: [`project-directory/`](./project-directory)

Briefly explain:

- What the project implements
- Why it is useful
- Which GPU or HPC techniques it demonstrates
- How to build it
- How to run it

More information:

- [Project Documentation](./project-directory/README.md)
```

---

## Development Guidelines

Recommended conventions for future projects:

- Use C++17 or later
- Keep CUDA error checking enabled
- Separate public headers from implementation files
- Use relative paths or CMake cache variables
- Avoid hard-coded user directories
- Add a CPU reference implementation when possible
- Report both residual and solution error
- Separate warmup time from measured time
- Record GPU model, CUDA version, compiler version, and build flags
- Keep raw benchmark output in a dedicated results directory
- Document unsupported matrix types and numerical assumptions
- Add small reproducible examples

---

## Current Status

Implemented:

- Five GPU SpMV kernels
- Jacobi iteration
- Multicolor Gauss-Seidel and SOR
- Conjugate Gradient
- Custom GEMM and cuBLAS comparison
- Four sparse direct solver adapters
- NVIDIA V100 benchmark reports
- Correctness and residual checks
- Nsight-oriented profiling commands

Planned improvements:

- Add a top-level CMake build system
- Remove machine-specific absolute paths
- Complete missing Jacobi headers
- Unify CSR and Matrix Market utilities
- Standardize timing boundaries
- Standardize solver tolerances
- Add more SuiteSparse matrices
- Add repeated statistical benchmarks
- Add GMRES and preconditioned Krylov methods
- Add multi-GPU and multi-node experiments
- Continue adding CUDA and HPC projects

---

## Scope and Limitations

The implementations in this repository are intended for learning, experimentation, benchmarking, and research prototyping.

They are not intended to replace production-grade numerical libraries.

Real applications may require additional work related to:

- Numerical stability
- Matrix scaling
- Reordering
- Preconditioning
- Mixed precision
- Memory capacity
- Multi-GPU communication
- Distributed-memory scalability
- Fault handling
- Reproducibility
- Cross-validation against established libraries

---

## License

No license has been added yet.

Before reusing this repository in another project, add an appropriate open-source license such as MIT, BSD-3-Clause, Apache-2.0, or GPL-3.0.
