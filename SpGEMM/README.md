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

# Benchmark Results

## Overview

This section summarizes a full paired benchmark of the hand-written CUDA SpGEMM backend (`custom`) against NVIDIA's AMGX library (`amgx`) on **C = A · Bᵀ** (transpose mode), run over a 48-matrix SuiteSparse subset on a single **Tesla V100-SXM2 16 GB** (CUDA 12.4, driver 550.144.03).

Each matrix is benchmarked by both backends independently. Each backend runs a warmup phase followed by 5 timed iterations, verifies its own product against an in-process CPU reference (sampled-row exact comparison + 3 Freivalds identity trials, relative tolerance `1e-9 + 1e-8·scale`), and reports its median time. Speedup is reported as `amgx_median / custom_median`: **values > 1 mean `custom` is faster; values < 1 mean `amgx` is faster.** Run order alternates per case to mitigate caching effects.

**Run parameters:** `--b-mode transpose --warmup 2 --repeat 5 --timeout 900`, seed `20260910`.

## Aggregate Results

| Metric | Value |
|--------|-------|
| Total cases | 48 |
| Verified pairs (both backends passed verification) | 43 |
| Failed cases | 5 |
| Verified fraction | 89.6% |
| **Geometric mean speedup** (outlier-resistant) | **1.21×** (custom faster overall) |
| Arithmetic mean speedup | 2.09× |
| Median speedup | 1.22× |
| Min / Max speedup | 0.12× / 30.18× |
| `custom` faster | 28 / 43 |
| `amgx` faster | 15 / 43 |
| Coverage (speedup within 0.7–1.5) | 20 / 43 (46.5%) |

### By value type

| Value type | Verified pairs | Geometric mean | Arithmetic mean |
|------------|---------------:|---------------:|----------------:|
| `float64` (real) | 39 | 1.18× | 2.10× |
| `complex128` (complex) | 4 | 1.53× | 1.96× |

The `custom` backend is faster overall on both value types. Its geometric-mean advantage is modest but consistent (1.18× real, 1.53× complex); the much larger arithmetic means reflect a handful of extreme wins that AMGX does not have a symmetric counterpart to.

## Full Per-Matrix Results

Speedup = `amgx_median / custom_median` (>1 ⇒ `custom` faster). Sorted by speedup, best `custom` wins first.

| Matrix | Type | nnz(C) | amgx median (ms) | custom median (ms) | speedup |
|--------|------|------:|-----------------:|-------------------:|--------:|
| cage14 | float64 | 236,999,813 | 1673.75 | 55.47 | **30.18×** custom |
| poisson3Db | float64 | 21,855,903 | 82.04 | 16.01 | 5.12× custom |
| memplus | float64 | 5,121,784 | 14.01 | 3.16 | 4.44× custom |
| appu | float64 | 133,137,482 | 337.17 | 105.34 | 3.20× custom |
| Chevron2 | complex128 | 2,221,297 | 4.18 | 1.39 | 3.00× custom |
| Lin | float64 | 6,221,600 | 4.65 | 1.72 | 2.70× custom |
| iChem_Jacobian | complex128 | 19,000,695 | 27.13 | 10.31 | 2.63× custom |
| add20 | float64 | 213,657 | 0.95 | 0.40 | 2.40× custom |
| kim2 | complex128 | 36,578,656 | 79.71 | 43.27 | 1.84× custom |
| torso3 | float64 | 21,113,330 | 8.84 | 5.01 | 1.77× custom |
| bcsstm25 | float64 | 15,439 | 0.148 | 0.086 | 1.72× custom |
| thermal | float64 | 220,320 | 0.230 | 0.146 | 1.58× custom |
| circuit_1 | float64 | 6,735,308 | 9.20 | 5.92 | 1.56× custom |
| bcsstm08 | float64 | 1,074 | 0.085 | 0.056 | 1.52× custom |
| SiH4 | float64 | 1,971,009 | 6.20 | 4.10 | 1.51× custom |
| shallow_water2 | float64 | 819,200 | 0.396 | 0.303 | 1.31× custom |
| shallow_water1 | float64 | 819,200 | 0.396 | 0.305 | 1.30× custom |
| sherman4 | float64 | 10,346 | 0.097 | 0.075 | 1.29× custom |
| G2_circuit | float64 | 2,002,996 | 0.726 | 0.580 | 1.25× custom |
| bodyy4 | float64 | 332,086 | 0.216 | 0.175 | 1.24× custom |
| Dubcova1 | float64 | 998,001 | 0.501 | 0.410 | 1.22× custom |
| G3_circuit | float64 | 21,267,134 | 6.61 | 5.44 | 1.21× custom |
| bratu3d | float64 | 611,808 | 0.265 | 0.239 | 1.11× custom |
| Dubcova2 | float64 | 4,092,529 | 1.558 | 1.441 | 1.08× custom |
| wang4 | float64 | 615,132 | 0.261 | 0.242 | 1.08× custom |
| majorbasis | float64 | 8,224,288 | 2.62 | 2.46 | 1.07× custom |
| epb2 | float64 | 641,858 | 0.799 | 0.758 | 1.05× custom |
| parabolic_fem | float64 | 9,959,935 | 3.807 | 3.763 | 1.01× custom |
| circuit5M_dc | float64 | 41,454,781 | 16.98 | 17.92 | 0.95× amgx |
| thermal2 | float64 | 24,018,057 | 7.66 | 8.11 | 0.94× amgx |
| thermomech_TC | float64 | 1,968,538 | 0.731 | 0.789 | 0.93× amgx |
| wathen100 | float64 | 1,657,621 | 0.538 | 0.609 | 0.88× amgx |
| FEM_3D_thermal2 | float64 | 14,335,500 | 4.72 | 5.89 | 0.80× amgx |
| apache2 | float64 | 16,563,324 | 3.74 | 4.81 | 0.78× amgx |
| atmosmodd | float64 | 31,215,208 | 7.19 | 9.55 | 0.75× amgx |
| GaAsH6 | float64 | 37,611,445 | 413.04 | 590.47 | 0.70× amgx |
| Dubcova3 | float64 | 17,480,761 | 9.83 | 14.25 | 0.69× amgx |
| bundle1 | float64 | 24,072,923 | 87.72 | 183.42 | 0.48× amgx |
| Kuu | float64 | 1,562,636 | 2.30 | 5.91 | 0.39× amgx |
| windscreen | complex128 | 5,613,426 | 24.26 | 65.20 | 0.37× amgx |
| bcsstk16 | float64 | 1,038,782 | 1.98 | 5.87 | 0.34× amgx |
| nemeth26 | float64 | 3,486,190 | 24.06 | 87.20 | 0.28× amgx |
| SiO2 | float64 | 104,839,083 | 3558.59 | 28734.35 | 0.12× amgx |

## Failed Cases (5)

All five failures are out-of-memory (OOM) errors, not correctness failures — no verified pair failed the CPU reference check.

| Matrix | Failure mode | Root cause |
|--------|--------------|------------|
| boyd2 | both OOM | Symmetric matrix with pathological per-row expansion (~142 GB). No 16 GB GPU can hold the product; both backends fail identically. |
| cage15 | both OOM | Symmetric matrix with even larger expansion than boyd2. Both backends OOM; inherent matrix pathology. |
| sls | both OOM | Extreme per-row expansion (~23.5 TB). Both backends OOM. |
| language | amgx OOM | Real matrix with small output (0.1 GB); `custom` succeeds (7.8 ms). AMGX's internal workspace over-allocates on one dense row (nnz ≈ 11.5k), OOM inside `csr_multiply`. |
| fem_hifreq_circuit | custom OOM | Complex matrix, ~7.7 GB real / ~15.4 GB complex expansion. AMGX succeeds (allocates by actual symbolic structure); `custom`'s sort path reserves scratch by a loose per-row upper bound whose total (9.63 B entries × 16 B = 29.4 GB) exceeds 16 GB. |

The `language` case is an AMGX-side workspace issue, not an algorithmic limitation of the `custom` path. The `fem_hifreq_circuit` case is a known `custom` sort-path memory-estimation weakness: the upper-bound estimate can be up to ~7× the actual output for matrices with heavily duplicated intermediate columns.

## Analysis

### Where `custom` wins

`custom`'s largest wins come from **large per-row output work** — matrices where a small number of rows dominate the computation and the hash-based accumulation path (bins 0–3, shared-memory open-addressing hash) absorbs the work efficiently:

- **cage14 (30.18×):** 237 M output nonzeros; `custom`'s bin classification routes the heavy rows to the hash path and finishes in 55 ms vs AMGX's 1674 ms.
- **poisson3Db (5.12×), memplus (4.44×), appu (3.20×):** similar structure — concentrated large-output rows where the custom hash path's direct accumulation beats AMGX's symbolic+numeric two-phase approach.

`custom` is also consistently faster on the **small/fast end** (sub-millisecond cases: bcsstm08, sherman4, add20, bodyy4), where AMGX's fixed setup and workspace-construction overhead dominates actual kernel time.

### Where AMGX wins

AMGX leads on **sort-heavy bin-4 matrices** — large, broadly-distributed expansions where the sort path (expand → segmented radix sort → merge) does the bulk of the work:

- **SiO2 (0.12×):** 105 M output nonzeros; `custom`'s two-pass sort path (count then write) takes 28.7 s vs AMGX's 3.6 s. The expand/sort/merge round-trip over ~113 M intermediate products is far more expensive than AMGX's internal structure-reuse and tuned sort.
- **nemeth26 (0.28×), bcsstk16 (0.34×), windscreen (0.37×), Kuu (0.39×):** the same pattern — moderate outputs (1–6 M) dominated by sort-path rows, where AMGX's approach is leaner.
- **bundle1 (0.48×), GaAsH6 (0.70×):** large bin-4 outputs where the expand/sort overhead is the bottleneck.

The common thread: when the *majority* of rows are bin-4 (large-output) rather than a *minority* of pathological rows, AMGX's symbolic/numerical separation and workspace reuse win.

### Complex vs real

The complex path (4 verified matrices) shows a larger `custom` advantage (1.53× geometric mean vs 1.18× real), but the sample is small and skewed by Chevron2 (3.0×) and iChem_Jacobian (2.6×). The one amgx-favored complex case (windscreen, 0.37×) is a bin-4-dominant matrix — the same sort-path overhead seen in the real cases.

### Overall takeaway

Across 43 verified matrices, **`custom` is faster overall** (geometric mean 1.21×, median 1.22×, wins 28 of 43). The geometric mean is the outlier-resistant measure and gives a fair picture: `custom`'s typical advantage is ~20%, concentrated in hash-path-amenable matrices with concentrated large-output rows, while AMGX retains a clear lead on sort-dominated workloads where the expand→sort→merge round-trip is the bottleneck.
