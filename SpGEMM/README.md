# SpGEMM Benchmark: Custom CUDA vs AMGX

A dual-backend benchmarking framework for symmetric sparse matrix-matrix multiplication (A × Aᵀ), comparing a hand-written CUDA implementation (`custom`) against NVIDIA's AMGX library (`amgx`) for both performance and correctness.

## Overview

The project computes C = A · B, where B is derived from A according to one of three modes:

- `transpose` (default): B = Aᵀ, i.e. compute A · Aᵀ
- `self`: B = A (A must be square)
- `random`: generate a random sparse B with the same nnz as A, seeded by `--seed`

Each backend runs independently on the same A: a warmup phase followed by timed repeats, and verifies its own product against a CPU reference implementation **in process** — no intermediate `.csr` dumps to disk.

## Architecture (custom backend)

Rows are classified into 5 bins by their *estimated output nnz per row* (the "bound"), each dispatched to a different accumulation path:

| Bin | Bound upper | Path | Description |
|-----|-------------|------|-------------|
| 0 | ≤ 32 | `small_rows<64,8>` | 8 threads per row, shared-memory hash table |
| 1 | ≤ 128 | `small_rows<256,32>` | one warp per row, shared-memory hash table |
| 2 | ≤ 512 | `medium_rows<1024>` | one block per row, shared-memory hash table |
| 3 | ≤ 1024 | `medium_rows<2048>` | one block per row, shared-memory hash table |
| 4 | > 1024 | sort path | batched expand → segmented radix sort → merge |

- **Hash path** (bins 0–3): shared-memory open-addressing hash table with load factor ≤ 0.5. Column order is not guaranteed; explicit zeros are preserved.
- **Sort path** (bin 4): expands all scalar products `(j, aᵣⱼ·bⱼₖ)`, runs a segmented radix sort by column (CUB `DeviceSegmentedRadixSort`), then scans and merges equal columns. Rows are processed in self-adaptive batches whose total expansion fits within ¼ of free VRAM (queried at runtime via `cudaMemGetInfo`), so no single batch overflows device memory.

### Design principle: portable, no hardware-specific tuning

- Bin assignment is driven entirely by per-row matrix work — **no device-name or architecture queries**.
- The VRAM budget is queried at runtime via `cudaMemGetInfo` and allocated as a fraction of free memory, independent of GPU model.
- `CUDA_ARCHITECTURES` is for compilation coverage only — **kernels never dispatch on architecture names**.

## Directory layout

```
├── CMakeLists.txt
├── README.md
├── run_bench.sh                 # one-shot suite runner (calls run_suite.py)
├── include/
│   ├── backend.hpp              # Backend interface, HostCSR, validate_product
│   ├── cpu_verify.hpp           # in-process CPU reference verification
│   └── matrix_io.hpp            # Matrix Market reader, transpose, fingerprint, CPU product
├── src/
│   ├── custom.cu                # hand-written SpGEMM backend
│   ├── amgx_backend.cu          # AMGX reference backend
│   └── bench_main.cpp           # CLI entry: timing + verification + JSON output
├── scripts/
│   ├── run_suite.py             # serial paired benchmark driver (stdlib only)
│   └── build.sh                 # builds AMGX + this project
├── tests/
│   ├── test_input.cpp           # matrix_io.hpp parser tests
│   ├── test_suite.py            # run_suite.py logic unit tests
│   └── test_gpu.py              # generates fixtures and runs both backends
├── test-artifacts/gpu/*.mtx     # fixed test matrices used by test_gpu.py
├── test_input_fixture.mtx       # small input for test_input.cpp
└── docs/superpowers/specs/      # design docs
```

## Dependencies

- **CUDA Toolkit** (bundled CUB; CUB ships with the toolkit from CUDA 11 on)
- **CMake ≥ 3.18**
- **NVIDIA AMGX 2.5.0** (only the `amgx` backend needs it) — requires a usable source tree (with `include/`, `external/rapidjson/`) and a built `libamgxsh.so`
- A GPU supporting FP64 `atomicAdd` (sm_60+)

## Build

```bash
# AMGX_ROOT        points to the AMGX source tree (with internal headers)
# AMGX_BUILD       points to the dir holding libamgxsh.so (default ${AMGX_ROOT}/build)
# CUDA_ARCHITECTURES  compile target arch (default 70; compilation coverage only,
#                     never affects kernel dispatch)
AMGX_ROOT=/path/to/AMGX-2.5.0 \
AMGX_BUILD=/path/to/AMGX-2.5.0/build \
CUDA_ARCHITECTURES=70 \
bash scripts/build.sh
```

`scripts/build.sh` first builds the AMGX `amgxsh` target, then builds this project, producing `build/bench_custom`, `build/bench_amgx`, and `build/test_input`.

You can also invoke cmake directly:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=70 -DAMGX_ROOT=/path/to/AMGX-2.5.0
cmake --build build --parallel
```

> Note: `scripts/build.sh` and `run_bench.sh` ship with absolute paths from the author's local environment. To reproduce, override the AMGX paths via the environment variables above, and edit `--matrix-dir` in `run_bench.sh` to point at your own matrix directory.

## Run

### Single-matrix quick check

```bash
./build/bench_custom --a path/to/A.mtx --b-mode transpose --warmup 2 --repeat 5
./build/bench_amgx   --a path/to/A.mtx --b-mode transpose --warmup 2 --repeat 5
```

CLI flags:

- `--a MATRIX` (required) input A
- `--b MATRIX` input B; when omitted, B is derived via `--b-mode`
- `--b-mode transpose|self|random` (default `transpose`)
- `--seed` seed for the random B (default 20260910)
- `--warmup` warmup iterations (default 5)
- `--repeat` timed iterations (default 20)

Each backend prints two JSON lines:

- `RESULT {...}` timing and shape info (rows/cols/nnz_a/nnz_b/nnz_c, first/median/mean/min_ms, per-iteration times_ms)
- `VERIFY {"passed":...}` CPU verification result (max_error, sample_rows, identity_trials=3)

### Full suite

```bash
python3 scripts/run_suite.py \
    --matrix-dir /path/to/mtxs/ \
    --bin-dir build --results-dir results \
    --b-mode transpose --warmup 2 --repeat 5 --timeout 900
```

`run_suite.py` runs a paired benchmark for every `.mtx` under the directory (recursive; symlink-resolved and de-duplicated so each physical matrix is benchmarked exactly once): it alternates the amgx/custom run order, parses each backend's `RESULT`/`VERIFY`, and computes the speedup `amgx_median / custom_median`.

Output goes to `results/<UTC-timestamp>/`:

- `results.csv` — per-matrix dual-backend median times + speedup
- `summary.json` — aggregate: arithmetic/geometric mean speedup, coverage, failures
- `case_NNNNNN/` — per-case stdout/stderr/iteration/verification logs
- `metadata.json` — platform, Python, SLURM env, source SHA256, nvidia-smi

## Correctness verification

Each backend verifies its own product **in process** (no `.csr` dump) against a CPU reference:

1. **Sampled row exact comparison**: for a set of sampled rows, a heap-merge CPU reference row is compared entry-by-entry (columns and values) against the backend's output.
2. **Freivalds identity** (3 trials): a random ±1 vector x is used to check ‖C·x − A·(B·x)‖ is within tolerance.
3. **Relative tolerance**: `|a−b| ≤ 1e-9 + 1e-8·max(|a|,|b|,scale)`.

`run_suite.py` marks any case that fails `VERIFY` as `verify_failed`.

## Benchmark results

Representative results from a full run over the SuiteSparse matrix set (A × Aᵀ, warmup=2, repeat=5) on a single V100 (16 GB). Speedup is `amgx_median / custom_median`: values > 1 mean `custom` is faster; values < 1 mean `amgx` is faster. All 39 cases where both backends passed verification are listed below.

| Matrix | nnz(C) | amgx median (ms) | custom median (ms) | speedup (amgx/custom) |
|--------|-------:|-----------------:|-------------------:|----------------------:|
| cage14 | 236,999,813 | 1673.6 | 55.5 | **30.16×** custom faster |
| poisson3Db | 21,855,903 | 82.1 | 16.0 | 5.12× custom faster |
| memplus | 5,121,784 | 14.0 | 3.2 | 4.34× custom faster |
| appu | 133,137,482 | 337.0 | 105.4 | 3.20× custom faster |
| Lin | 6,221,600 | 4.8 | 1.7 | 2.77× custom faster |
| add20 | 213,657 | 0.95 | 0.41 | 2.33× custom faster |
| bcsstm25 | 15,439 | 0.148 | 0.086 | 1.73× custom faster |
| torso3 | 21,113,330 | 8.83 | 5.55 | 1.59× custom faster |
| SiH4 | 1,971,009 | 6.75 | 4.26 | 1.59× custom faster |
| thermal | 220,320 | 0.226 | 0.145 | 1.56× custom faster |
| circuit_1 | 6,735,308 | 8.98 | 5.92 | 1.52× custom faster |
| bcsstm08 | 1,074 | 0.086 | 0.058 | 1.48× custom faster |
| shallow_water2 | 819,200 | 0.401 | 0.299 | 1.34× custom faster |
| shallow_water1 | 819,200 | 0.397 | 0.307 | 1.29× custom faster |
| sherman4 | 10,346 | 0.096 | 0.076 | 1.26× custom faster |
| G2_circuit | 2,002,996 | 0.727 | 0.578 | 1.26× custom faster |
| bodyy4 | 332,086 | 0.219 | 0.176 | 1.24× custom faster |
| Dubcova1 | 998,001 | 0.506 | 0.409 | 1.24× custom faster |
| G3_circuit | 21,267,134 | 6.60 | 5.45 | 1.21× custom faster |
| bratu3d | 611,808 | 0.263 | 0.236 | 1.11× custom faster |
| majorbasis | 8,224,288 | 2.62 | 2.45 | 1.07× custom faster |
| Dubcova2 | 4,092,529 | 1.53 | 1.44 | 1.06× custom faster |
| wang4 | 615,132 | 0.280 | 0.264 | 1.06× custom faster |
| epb2 | 641,858 | 0.800 | 0.758 | 1.06× custom faster |
| circuit5M_dc | 41,454,781 | 16.66 | 17.91 | 0.93× amgx faster |
| thermomech_TC | 1,968,538 | 0.728 | 0.797 | 0.91× amgx faster |
| parabolic_fem | 9,959,935 | 3.40 | 3.76 | 0.91× amgx faster |
| wathen100 | 1,657,621 | 0.609 | 0.677 | 0.90× amgx faster |
| thermal2 | 24,018,057 | 6.72 | 8.10 | 0.83× amgx faster |
| atmosmodd | 31,215,208 | 7.19 | 8.68 | 0.83× amgx faster |
| FEM_3D_thermal2 | 14,335,500 | 4.71 | 5.90 | 0.80× amgx faster |
| apache2 | 16,563,324 | 3.75 | 4.81 | 0.78× amgx faster |
| Dubcova3 | 17,480,761 | 9.85 | 13.58 | 0.73× amgx faster |
| GaAsH6 | 37,611,445 | 413.1 | 594.8 | 0.69× amgx faster |
| bundle1 | 24,072,923 | 87.7 | 183.2 | 0.48× amgx faster |
| Kuu | 1,562,636 | 2.33 | 5.94 | 0.39× amgx faster |
| bcsstk16 | 1,038,782 | 1.99 | 5.88 | 0.34× amgx faster |
| nemeth26 | 3,486,190 | 24.1 | 87.1 | 0.28× amgx faster |
| SiO2 | 104,839,083 | 3550.7 | 28835.8 | 0.12× amgx faster |

**Aggregate** (39 verified pairs; 9 failures where one or both backends errored on the input):

| Metric | Value |
|--------|-------|
| Verified pairs | 39 / 48 |
| Speedup = amgx_median / custom_median | >1 ⇒ `custom` faster, <1 ⇒ `amgx` faster |
| **Arithmetic mean speedup** | **2.09×** |
| **Geometric mean speedup** | **1.17×** |
| Matrices where `custom` is faster | 24 / 39 |
| Matrices where `amgx` is faster | 15 / 39 |
| Coverage (speedup within 0.7–1.5) | 22 / 39 (56.4%) |

In words: averaged arithmetically across the 39 verified matrices, `amgx` takes 2.09× the median time of `custom`; averaged geometrically (the outlier-resistant measure), 1.17× — so `custom` is faster overall. `custom`'s largest wins are on matrices with large per-row output work (cage14 at 30.16×, poisson3Db 5.12×, memplus 4.34×); `amgx` still leads on several heavy bin4 expand/sort cases (SiO2, nemeth26, bcsstk16, bundle1). Full per-matrix results are regenerated under `results/` on each run.
