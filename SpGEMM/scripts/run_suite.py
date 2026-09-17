#!/usr/bin/env python3
"""Serial, auditable paired SpGEMM benchmark driver (standard library only).

Usage:
    # Full run over every .mtx under a directory (one matrix dir, e.g. where
    # the real files live — symlinks in subdirs are resolved and de-duplicated,
    # so each physical matrix is benchmarked exactly once):
    python3 scripts/run_suite.py \
        --matrix-dir /home/opsadmin/wangyitong/mtxs/ \
        --bin-dir build --results-dir results \
        --b-mode transpose --warmup 2 --repeat 5 --timeout 900

    # Single matrix, quick check (amgx vs custom on one A; B derived from A):
    python3 scripts/run_suite.py --a path/to/m.mtx --b-mode transpose \
        --bin-dir build --results-dir results --warmup 2 --repeat 5

    # Two explicit matrices A x B:
    python3 scripts/run_suite.py --a A.mtx --b B.mtx \
        --bin-dir build --results-dir results

    # Or use run_bench.sh (it just calls this script with defaults).
    #
    # --b-mode: how B is derived from A when --b is omitted.
    #           transpose = A x Aᵀ for real input or A x Aᴴ for complex input
    #                       (default), self = A x A, random = random
    #                      pattern with the given --seed.
    # Output: results/<UTC timestamp>/ with
    #           results.csv  — per-matrix amgx vs custom median timings + speedup
    #           summary.json — aggregate: mean speedup, coverage, failures
    #           case_NNNNNN/ — per-case stdout/stderr/iterations/verification logs

    # Exit code is 1 if any case failed, else 0.
"""
import argparse
import collections
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

DEFAULT_MATRIX_DIR = '/home/bingxing2/home/scx8ale/soft/works/mtxs'
PAIR_FIELDS = ('value_type', 'a_hash', 'b_hash', 'rows', 'cols', 'nnz_a', 'nnz_b', 'nnz_c', 'seed', 'warmup', 'repeat')


def benchmark_command(bin_dir, backend, a, b, b_mode, seed, warmup, repeat):
    command = [str(Path(bin_dir) / ('bench_' + backend)), '--a', str(a)]
    if b is not None:
        command += ['--b', str(b)]
    else:
        command += ['--b-mode', str(b_mode)]
    return command + ['--seed', str(seed), '--warmup', str(warmup), '--repeat', str(repeat)]


def parse_record(stdout, prefix):
    matches = [line[len(prefix) + 1:] for line in stdout.splitlines() if line.startswith(prefix + ' ')]
    if len(matches) != 1:
        raise ValueError('expected exactly one ' + prefix + ' JSON line')
    record = json.loads(matches[0])
    if not isinstance(record, dict):
        raise ValueError(prefix + ' must be a JSON object')
    return record


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def validate_record(record, backend, seed, warmup, repeat):
    for key, expected in [('backend', backend), ('seed', seed), ('warmup', warmup), ('repeat', repeat)]:
        if record.get(key) != expected:
            raise ValueError('incorrect ' + key)
    for key in ('a_hash', 'b_hash'):
        if not isinstance(record.get(key), str) or not record[key]:
            raise ValueError('invalid ' + key)
    if record.get('value_type') not in ('float64', 'complex128'):
        raise ValueError('invalid value_type')
    for key in ('rows', 'cols', 'nnz_a', 'nnz_b', 'nnz_c'):
        if type(record.get(key)) is not int or record[key] < 0:
            raise ValueError('invalid ' + key)
    for key in ('first_ms', 'median_ms', 'mean_ms', 'min_ms'):
        if not positive(record.get(key)):
            raise ValueError('nonpositive or nonfinite ' + key)
    times = record.get('times_ms')
    if not isinstance(times, list) or len(times) != repeat or not all(positive(t) for t in times):
        raise ValueError('invalid times_ms')


def validate_pair(records):
    if set(records) != {'amgx', 'custom'}:
        raise ValueError('missing backend record')
    for key in PAIR_FIELDS:
        if records['amgx'].get(key) != records['custom'].get(key):
            raise ValueError('backend mismatch for ' + key)


def aggregate(rows):
    ratios, failures = [], collections.Counter()
    for row in rows:
        if row['status'] != 'ok':
            failures[row['status']] += 1
        elif not positive(row.get('amgx_median_ms')) or not positive(row.get('custom_median_ms')):
            failures['invalid_timing'] += 1
        else:
            ratio = row['amgx_median_ms'] / row['custom_median_ms']
            if positive(ratio):
                ratios.append(ratio)
            else:
                failures['invalid_ratio'] += 1
    count = sum(0.7 <= ratio <= 1.5 for ratio in ratios)
    return dict(total_cases=len(rows), verified_pairs=len(ratios), failed_cases=sum(failures.values()),
                failure_counts=dict(failures), geometric_mean_speedup=math.exp(statistics.mean(map(math.log, ratios))) if ratios else None,
                arithmetic_mean_speedup=statistics.mean(ratios) if ratios else None,
                coverage_count_0_7_to_1_5=count,
                **{'coverage_count_0.7_to_1.5': count, 'coverage_ratio_0.7_to_1.5': count / len(ratios) if ratios else None},
                verified_fraction=len(ratios) / len(rows) if rows else None)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def execute(command, directory, name, timeout):
    """Preserve raw streams even on timeout; never pipe large logs into memory."""
    started = time.monotonic()
    result = {'command': command, 'status': 'ok'}
    with (directory / (name + '.stdout.log')).open('wb') as stdout, (directory / (name + '.stderr.log')).open('wb') as stderr:
        try:
            process = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=timeout, check=False)
            result['returncode'] = process.returncode
            if process.returncode:
                result['status'] = 'exit_error'
        except subprocess.TimeoutExpired:
            result['status'] = 'timeout'
        except OSError as error:
            result.update(status='launch_error', error=str(error))
    result['wall_seconds'] = time.monotonic() - started
    write_json(directory / (name + '.process.json'), result)
    return result


def capture_metadata(args, root):
    source_root = Path(__file__).resolve().parents[1]
    sources = [source_root / 'CMakeLists.txt']
    for folder in ('include', 'src', 'scripts'):
        sources.extend(p for p in (source_root / folder).rglob('*') if p.is_file() and p.suffix in ('.h', '.hpp', '.c', '.cc', '.cpp', '.cu', '.cuh', '.py'))
    metadata = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    platform=platform.platform(), python=sys.version, arguments=vars(args),
                    slurm={key: value for key, value in os.environ.items() if key.startswith('SLURM_')},
                    source_sha256={str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources if p.is_file()})
    metadata['nvidia_smi'] = execute(['nvidia-smi'], root, 'nvidia-smi', 15)
    write_json(root / 'metadata.json', metadata)


def run_case(index, a, b, b_mode, args, root):
    directory = root / ('case_%06d' % index)
    directory.mkdir()
    order = ['amgx', 'custom'] if index % 2 == 0 else ['custom', 'amgx']
    row = dict(case=index, a=str(a), b=str(b) if b else '', b_mode=b_mode, order=','.join(order), status='ok', error='')
    records = {}
    problems = []
    for backend in order:
        command = benchmark_command(args.bin_dir, backend, a, b, b_mode, args.seed, args.warmup, args.repeat)
        result = execute(command, directory, backend, args.timeout)
        row[backend + '_status'] = result['status']
        if result['status'] != 'ok':
            problems.append(backend + ':' + result['status'])
            continue
        try:
            stdout = (directory / (backend + '.stdout.log')).read_text(encoding='utf-8', errors='replace')
            record = parse_record(stdout, 'RESULT')
            validate_record(record, backend, args.seed, args.warmup, args.repeat)
            # Each backend verifies its own product against the CPU reference in
            # process (no .csr dump, no cross-backend comparison).
            verification = parse_record(stdout, 'VERIFY')
            if verification.get('passed') is not True:
                raise ValueError('backend did not pass CPU verification')
            write_json(directory / (backend + '.iterations.json'), record)
            write_json(directory / (backend + '.verification.json'), verification)
            records[backend] = record
            for key in ('first_ms', 'median_ms', 'mean_ms', 'min_ms'):
                row[backend + '_' + key] = record[key]
        except (ValueError, TypeError, OSError) as error:
            row[backend + '_status'] = 'invalid_result'
            problems.append(backend + ':invalid_result:' + str(error))
    if problems:
        if any(':timeout' in problem for problem in problems):
            row['status'] = 'timeout'
        elif any(':verify_failed' in problem for problem in problems):
            row['status'] = 'verify_failed'
        else:
            row['status'] = 'benchmark_failed'
        row['error'] = '; '.join(problems)
    elif 'amgx' in records and 'custom' in records:
        try:
            validate_pair(records)
        except ValueError as error:
            row['status'] = 'benchmark_failed'
            row['error'] = str(error)
            write_json(directory / 'case.json', row)
            return row
        row.update({key: records['amgx'][key] for key in PAIR_FIELDS})
        row['verify_status'] = 'ok'
        row['speedup_amgx_over_custom'] = records['amgx']['median_ms'] / records['custom']['median_ms']
    else:
        row['status'] = 'benchmark_failed'
        row['error'] = 'missing backend record'
    write_json(directory / 'case.json', row)
    return row


def save_results(root, rows):
    fields = ['case', 'a', 'b', 'status', 'error'] + sorted(set().union(*(set(row) for row in rows)) - {'case', 'a', 'b', 'status', 'error'})
    with (root / 'results.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = aggregate(rows)
    write_json(root / 'summary.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix-dir', default=DEFAULT_MATRIX_DIR)
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--bin-dir', default='build')
    parser.add_argument('--a')
    parser.add_argument('--b')
    parser.add_argument('--b-mode', choices=['transpose', 'self', 'random'], default='transpose',
                        help='derive B when omitted (transpose uses A^T for real input and A^H for complex input)')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeat', type=int, default=20)
    parser.add_argument('--seed', type=int, default=20260910)
    parser.add_argument('--timeout', type=float, default=900)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeat < 1 or not positive(args.timeout) or args.seed < 0:
        parser.error('require warmup >= 0, repeat >= 1, seed >= 0 and finite timeout > 0')
    if args.b_mode == 'self' and args.b:
        parser.error('--b-mode self is ignored when --b is given')
    if args.b and not args.a:
        parser.error('--b requires --a')
    args.bin_dir = str(Path(args.bin_dir).resolve())
    if args.a:
        matrices = [Path(args.a).resolve()]
    else:
        # rglob does not recurse into symlinked *directories*, but a directory
        # can hold .mtx *symlinks* (or copies) pointing back at the real files.
        # Resolve each hit to its real path, then de-duplicate while keeping
        # sorted order, so every physical matrix is benchmarked exactly once.
        resolved = [p.resolve() for p in Path(args.matrix_dir).rglob('*')
                    if p.is_file() and p.suffix.lower() == '.mtx']
        seen, matrices = set(), []
        for path in sorted(resolved):
            if path not in seen:
                seen.add(path)
                matrices.append(path)
    b = Path(args.b).resolve() if args.b else None
    root = Path(args.results_dir).resolve() / datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    root.mkdir(parents=True, exist_ok=False)
    print('Results: ' + str(root), flush=True)
    capture_metadata(args, root)
    write_json(root / 'manifest.json', [{'a': str(a), 'b': str(b) if b else None, 'b_mode': (None if b else args.b_mode)} for a in matrices])
    rows = []
    for index, a in enumerate(matrices):
        print('[%d/%d] %s' % (index + 1, len(matrices), a), flush=True)
        try:
            row = run_case(index, a, b, args.b_mode, args, root)
        except Exception as error:
            row = dict(case=index, a=str(a), b=str(b) if b else '', b_mode=(None if b else args.b_mode), status='runner_error', error=str(error))
        rows.append(row)
        save_results(root, rows)
        print('  ' + row['status'] + (': ' + row['error'] if row['error'] else ''), flush=True)
    summary = save_results(root, rows)
    print(json.dumps(summary, allow_nan=False), flush=True)
    return 1 if not rows or summary['failed_cases'] else 0


if __name__ == '__main__':
    sys.exit(main())
