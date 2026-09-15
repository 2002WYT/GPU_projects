#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
AMGX_ROOT=${AMGX_ROOT:-/home/opsadmin/wangyitong/soft/AMGX-2.5.0}
# Compilation coverage only. Kernels never dispatch on these architecture names.
CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES:-70}
AMGX_BUILD=${AMGX_BUILD:-/home/opsadmin/wangyitong/soft/AMGX-2.5.0/build}
JOBS=${JOBS:-4}
cmake -S "$AMGX_ROOT" -B "$AMGX_BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" -DCMAKE_NO_MPI=ON
cmake --build "$AMGX_BUILD" --target amgxsh --parallel "$JOBS"
cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" \
  -DAMGX_ROOT="$AMGX_ROOT" -DAMGX_BUILD="$AMGX_BUILD"
cmake --build "$ROOT/build" --parallel "$JOBS"
