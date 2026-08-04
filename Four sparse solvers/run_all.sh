#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${1:-${script_dir}/../build}"
matrix="${2:-${script_dir}/files/AA1.mtx}"
mpi_exec="${MPIEXEC:-mpirun}"
mpi_ranks="${NP:-1}"

for executable in test_slu test_pangulu test_cudss test_strumpack; do
    path="${build_dir}/bin/${executable}"
    if [[ -x "${path}" ]]; then
        "${mpi_exec}" -n "${mpi_ranks}" "${path}" "${matrix}"
    fi
done
