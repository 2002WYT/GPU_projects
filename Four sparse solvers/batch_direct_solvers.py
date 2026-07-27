#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量测试 SuperLU_DIST、PanguLU、cuDSS 和 STRUMPACK。

默认：
  矩阵目录：./files/
  汇总文件：direct_solver_results.csv
  日志目录：direct_solver_logs

不使用 mpirun。MPI 程序以单进程 singleton 模式直接运行。

可通过环境变量指定可执行文件：
  SUPERLU_EXE, PANGULU_EXE, CUDSS_EXE, STRUMPACK_EXE
也可指定各项目环境脚本：
  SUPERLU_ENV, PANGULU_ENV, CUDSS_ENV, STRUMPACK_ENV
"""
from __future__ import print_function

import argparse
import csv
import glob
import os
import re
import shlex
import subprocess
import sys
import time


SOLVERS = [
    {
        "key": "superlu", "name": "SuperLU_DIST",
        "exe_env": "SUPERLU_EXE", "env_env": "SUPERLU_ENV",
        "default_env": "~/code/demo02/env.sh",
        "roots": ["~/code/demo02"],
        "names": ["test_slu", "test_superlu_dist"],
        "stages": {
            "reorder_ms": ["Reorder time", "Ordering time", "Analysis time"],
            "factorization_ms": ["Factorization time", "Factor time"],
            "solve_ms": ["Solve time"],
            "total_ms": ["Total wall time", "Total time", "Total timed time"],
        },
    },
    {
        "key": "pangulu", "name": "PanguLU",
        "exe_env": "PANGULU_EXE", "env_env": "PANGULU_ENV",
        "default_env": "~/code/demo02/env.sh",
        "roots": ["~/code/demo02"],
        "names": ["test_pangu", "test_pangulu"],
        "stages": {
            "initialization_ms": ["Initialization time", "Init time", "Analysis time"],
            "factorization_ms": ["Factorization time", "Factor time"],
            "solve_ms": ["Solve time"],
            "total_ms": ["Total timed time", "Total time"],
        },
    },
    {
        "key": "cudss", "name": "cuDSS",
        "exe_env": "CUDSS_EXE", "env_env": "CUDSS_ENV",
        "default_env": "~/code/demo01/env.sh",
        "roots": ["~/code/demo01"],
        "names": ["test_cudss"],
        "stages": {
            "analysis_ms": ["Analysis time", "Initialization time"],
            "factorization_ms": ["Factorization time", "Factor time"],
            "solve_ms": ["Solve time"],
            "total_ms": ["Total time", "Total timed time"],
        },
    },
    {
        "key": "strumpack", "name": "STRUMPACK",
        "exe_env": "STRUMPACK_EXE", "env_env": "STRUMPACK_ENV",
        "default_env": "~/code/demo02/env.sh",
        "roots": ["~/code/demo02"],
        "names": ["test_strumpack"],
        "stages": {
            "reorder_ms": ["Reorder time", "Ordering time", "Analysis time"],
            "factorization_ms": ["Factorization time", "Factor time"],
            "solve_ms": ["Solve time"],
            "total_ms": ["Total time", "Total timed time"],
        },
    },
]

STAGE_COLUMNS = {
    "superlu": ["reorder_ms", "factorization_ms", "solve_ms", "total_ms"],
    "pangulu": ["initialization_ms", "factorization_ms", "solve_ms", "total_ms"],
    "cudss": ["analysis_ms", "factorization_ms", "solve_ms", "total_ms"],
    "strumpack": ["reorder_ms", "factorization_ms", "solve_ms", "total_ms"],
}


def expand(path):
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def executable(path):
    return bool(path and os.path.isfile(path) and os.access(path, os.X_OK))


def find_exe(spec):
    override = os.environ.get(spec["exe_env"], "").strip()
    if override:
        path = expand(override)
        return path if executable(path) else ""

    candidates = []
    roots = [os.getcwd()] + [expand(x) for x in spec["roots"]]
    for root in roots:
        for name in spec["names"]:
            candidates += [
                os.path.join(root, name),
                os.path.join(root, "bin", name),
                os.path.join(root, "build", name),
                os.path.join(root, "build", "bin", name),
            ]
    for path in candidates:
        if executable(path):
            return os.path.abspath(path)

    matches = []
    for root in [expand(x) for x in spec["roots"]]:
        if not os.path.isdir(root):
            continue
        for name in spec["names"]:
            for path in glob.glob(os.path.join(root, "**", name), recursive=True):
                if executable(path) and "CMakeFiles" not in path:
                    matches.append(os.path.abspath(path))
    matches.sort(key=lambda x: (x.count(os.sep), len(x), x))
    return matches[0] if matches else ""


def find_env(spec):
    override = os.environ.get(spec["env_env"], "").strip()
    path = expand(override if override else spec["default_env"])
    return path if os.path.isfile(path) else ""


def read_mtx_metadata(path):
    """按 mtx_reader.h 的规则统计对称展开后的 nnz，不合并重复项。"""
    with open(path, "r") as f:
        header = f.readline().strip().split()
        if len(header) < 5 or header[0].lower() != "%%matrixmarket":
            raise ValueError("invalid Matrix Market header")
        if header[1].lower() != "matrix" or header[2].lower() != "coordinate":
            raise ValueError("only matrix coordinate format is supported")

        dtype = header[3].lower()
        symmetry = header[4].lower()
        size_line = None
        for line in f:
            if line.strip() and not line.lstrip().startswith("%"):
                size_line = line
                break
        if size_line is None:
            raise ValueError("missing size line")

        parts = size_line.split()
        if len(parts) < 3:
            raise ValueError("invalid size line")
        nrows, ncols, stored = map(int, parts[:3])
        expanded = stored

        if symmetry in ("symmetric", "skew-symmetric"):
            expanded = 0
            count = 0
            for line in f:
                s = line.strip()
                if not s or s.startswith("%"):
                    continue
                fields = s.split()
                if len(fields) < 2:
                    raise ValueError("invalid matrix entry")
                i, j = int(fields[0]), int(fields[1])
                expanded += 1 if i == j else 2
                count += 1
            if count != stored:
                raise ValueError("declared nnz={} but read {}".format(stored, count))

        return {
            "matrix_market_type": "{} coordinate {}".format(dtype, symmetry),
            "nrows": nrows,
            "ncols": ncols,
            "nnz_stored": stored,
            "nnz_expanded": expanded,
        }


def norm_label(text):
    return re.sub(r"\s+", " ", text.strip()).lower()


def labeled_values(output):
    result = {}
    for line in output.splitlines():
        m = re.match(r"^\s*([^:]+?)\s*:\s*(.*?)\s*$", line)
        if m:
            result[norm_label(m.group(1))] = m.group(2).strip()
    return result


def first_number(text):
    m = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text or "")
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def lookup(labels, aliases):
    for alias in aliases:
        key = norm_label(alias)
        if key in labels:
            return labels[key]
    return ""


def lookup_number(labels, aliases):
    return first_number(lookup(labels, aliases))


def ms(seconds):
    return "" if seconds is None else "{:.6f}".format(seconds * 1000.0)


def sci(value):
    return "" if value is None else "{:.12e}".format(value)


def error_summary(output, limit=600):
    keys = ("error", "failed", "fatal", "segmentation", "out of memory",
            "cannot", "invalid", "singular", "cuda", "cudss_status")
    lines = [x.strip() for x in output.splitlines()
             if any(k in x.lower() for k in keys)]
    if not lines:
        lines = [x.strip() for x in output.splitlines() if x.strip()][-4:]
    text = " | ".join(lines)
    return text if len(text) <= limit else text[:limit - 3] + "..."


def parse_output(spec, output, returncode, wall_ms):
    labels = labeled_values(output)
    test = lookup(labels, ["TEST"]).upper()
    if returncode == 0 and test != "FAILED":
        status = "SUCCESS"
    elif test == "FAILED":
        status = "FAILED_ACCURACY"
    else:
        status = "FAILED"

    result = {
        "status": status,
        "exit_code": returncode,
        "relative_residual": sci(lookup_number(labels, ["Relative residual"])),
        "relative_solution_error": sci(lookup_number(
            labels, ["Relative x error", "Relative solution error", "Solution error"])),
        "wall_elapsed_ms": "{:.6f}".format(wall_ms),
        "error": "" if status == "SUCCESS" else error_summary(output),
    }
    for column, aliases in spec["stages"].items():
        result[column] = ms(lookup_number(labels, aliases))

    if not result.get("total_ms"):
        values = []
        for column in spec["stages"]:
            if column != "total_ms" and result.get(column) not in (None, ""):
                values.append(float(result[column]))
        if values:
            result["total_ms"] = "{:.6f}".format(sum(values))
    return result


def empty_result(spec, status, error):
    result = {
        "status": status, "exit_code": "", "relative_residual": "",
        "relative_solution_error": "", "wall_elapsed_ms": "",
        "error": error, "log_file": "",
    }
    for column in STAGE_COLUMNS[spec["key"]]:
        result[column] = ""
    return result


def run_solver(spec, exe, env_script, matrix, log_file, timeout, child_env):
    shell = ('set -o pipefail; '
             'if [ -n "$1" ] && [ -f "$1" ]; then source "$1"; fi; '
             'exec "$2" "$3"')
    command = ["bash", "-lc", shell, "batch", env_script, exe, matrix]
    begin = time.time()
    output = ""
    try:
        done = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True,
            timeout=None if timeout <= 0 else timeout,
            env=child_env)
        output = done.stdout or ""
        result = parse_output(spec, output, done.returncode,
                              (time.time() - begin) * 1000.0)
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        result = empty_result(spec, "TIMEOUT", "timeout after {} seconds".format(timeout))
        result["wall_elapsed_ms"] = "{:.6f}".format((time.time() - begin) * 1000.0)
    except Exception as exc:
        result = empty_result(spec, "LAUNCH_ERROR", str(exc))
        result["wall_elapsed_ms"] = "{:.6f}".format((time.time() - begin) * 1000.0)

    with open(log_file, "w") as f:
        f.write("# solver: {}\n".format(spec["name"]))
        f.write("# executable: {}\n".format(exe))
        f.write("# environment: {}\n".format(env_script or "(none)"))
        f.write("# matrix: {}\n".format(matrix))
        f.write("# command: {}\n\n".format(" ".join(shlex.quote(x) for x in command)))
        f.write(output)
    result["log_file"] = os.path.abspath(log_file)
    return result


def safe_name(path):
    name = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name) or "matrix"


def columns():
    result = ["matrix", "matrix_path", "matrix_market_type", "nrows", "ncols",
              "nnz_stored", "nnz_expanded"]
    common = ["status", "exit_code", "relative_residual",
              "relative_solution_error", "wall_elapsed_ms", "error", "log_file"]
    for spec in SOLVERS:
        result += [spec["key"] + "_" + x for x in common]
        result += [spec["key"] + "_" + x for x in STAGE_COLUMNS[spec["key"]]]
    return result


def put_solver(row, key, result):
    for field, value in result.items():
        row[key + "_" + field] = value


def default_mtx_dir():
    value = os.environ.get("MTX_DIR", "").strip()
    if value:
        return expand(value)
    dotted = expand("~/code/demo02/files")
    return dotted

def arguments():
    p = argparse.ArgumentParser(description="批量运行四个稀疏直接求解器，不使用 mpirun")
    p.add_argument("--matrix-dir", default=default_mtx_dir())
    p.add_argument("--output", default=os.environ.get("RESULT_CSV", "direct_solver_results.csv"))
    p.add_argument("--log-dir", default=os.environ.get("LOG_DIR", "direct_solver_logs"))
    p.add_argument("--timeout", type=int, default=int(os.environ.get("SOLVER_TIMEOUT", "0")),
                   help="每个求解器的超时秒数，0 表示不限制")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = arguments()
    mtx_dir = expand(args.matrix_dir)
    output = expand(args.output)
    log_dir = expand(args.log_dir)

    if not os.path.isdir(mtx_dir):
        print("矩阵目录不存在：{}".format(mtx_dir), file=sys.stderr)
        return 2
    matrices = sorted(glob.glob(os.path.join(mtx_dir, "*.mtx")))
    if not matrices:
        print("目录中没有 .mtx 文件：{}".format(mtx_dir), file=sys.stderr)
        return 2
    if os.path.exists(output) and not args.overwrite:
        print("结果文件已存在：{}\n请加 --overwrite 覆盖。".format(output), file=sys.stderr)
        return 2

    parent = os.path.dirname(output)
    if parent:
        os.makedirs(parent, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    resolved = {}
    print("求解器路径：")
    for spec in SOLVERS:
        exe, env_script = find_exe(spec), find_env(spec)
        resolved[spec["key"]] = (exe, env_script)
        print("  {:12s}: {}".format(spec["name"], exe or "NOT FOUND"))
        if env_script:
            print("                env: {}".format(env_script))

    child_env = os.environ.copy()
    child_env.setdefault("CUDA_VISIBLE_DEVICES", "0")

    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns(), extrasaction="ignore")
        writer.writeheader()
        f.flush()

        for index, matrix in enumerate(matrices, 1):
            print("\n[{}/{}] {}".format(index, len(matrices), os.path.basename(matrix)), flush=True)
            row = {
                "matrix": os.path.splitext(os.path.basename(matrix))[0],
                "matrix_path": os.path.abspath(matrix),
                "matrix_market_type": "", "nrows": "", "ncols": "",
                "nnz_stored": "", "nnz_expanded": "",
            }
            metadata_error = ""
            try:
                metadata = read_mtx_metadata(matrix)
                row.update(metadata)
                print("  size={}x{}, nnz(expanded)={}".format(
                    metadata["nrows"], metadata["ncols"], metadata["nnz_expanded"]))
            except Exception as exc:
                metadata_error = str(exc)
                print("  Matrix Market 解析失败：{}".format(exc), file=sys.stderr)

            for spec in SOLVERS:
                exe, env_script = resolved[spec["key"]]
                log_file = os.path.join(log_dir, "{}__{}.log".format(safe_name(matrix), spec["key"]))
                if metadata_error:
                    result = empty_result(spec, "SKIPPED_BAD_MTX", metadata_error)
                elif row["nrows"] != row["ncols"]:
                    result = empty_result(spec, "SKIPPED_NOT_SQUARE",
                                          "matrix is {}x{}".format(row["nrows"], row["ncols"]))
                elif not exe:
                    result = empty_result(spec, "NOT_FOUND",
                                          "{} is not set and executable was not found".format(spec["exe_env"]))
                else:
                    print("  {:12s} ...".format(spec["name"]), end="", flush=True)
                    result = run_solver(spec, exe, env_script, matrix, log_file,
                                        args.timeout, child_env)
                    wall = float(result["wall_elapsed_ms"] or 0.0) / 1000.0
                    print(" {} ({:.3f} s)".format(result["status"], wall), flush=True)
                put_solver(row, spec["key"], result)

            writer.writerow(row)
            f.flush()

    print("\n批量测试完成")
    print("汇总 CSV：{}".format(output))
    print("完整日志：{}".format(log_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
