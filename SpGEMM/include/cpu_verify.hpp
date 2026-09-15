#pragma once
// In-process CPU verification of a SpGEMM product C = A*B. The reference
// algorithm (reference_row, identities, close) is identical to the former
// standalone verify_results.cpp; only the cross-backend structural comparison
// (which required a .csr dump from each backend) has been removed. Each
// backend is now verified independently against the CPU reference.
#include "matrix_io.hpp"
#include <algorithm>
#include <cmath>
#include <queue>
#include <random>
#include <utility>
#include <vector>

namespace spgemm {
namespace cpu_verify {
using Row = std::vector<std::pair<int, long double>>;

inline Row row(const HostCSR& c, int r) {
    Row v; v.reserve(c.row[r + 1] - c.row[r]);
    for (int p = c.row[r]; p < c.row[r + 1]; ++p) v.emplace_back(c.col[p], c.val[p]);
    std::sort(v.begin(), v.end());
    for (size_t i = 1; i < v.size(); ++i) if (v[i - 1].first == v[i].first) throw std::runtime_error("duplicate output column");
    return v;
}
inline bool close(long double a, long double b, long double scale, double& maxerr) {
    auto err = std::abs(a - b); maxerr = std::max(maxerr, double(err));
    return std::isfinite(a) && std::isfinite(b) && err <= 1e-9L + 1e-8L * std::max({std::abs(a), std::abs(b), scale});
}
inline std::pair<Row, long double> reference_row(const HostCSR& a, const HostCSR& b, int r) {
    // Merge sorted input rows without materializing every scalar product.
    struct Cursor { int col, p, q, end; };
    auto later = [](const Cursor& x, const Cursor& y) { return x.col > y.col; };
    std::priority_queue<Cursor, std::vector<Cursor>, decltype(later)> heap(later);
    for (int p = a.row[r]; p < a.row[r + 1]; ++p) {
        int q = b.row[a.col[p]], end = b.row[a.col[p] + 1];
        if (q < end) heap.push({b.col[q], p, q, end});
    }
    Row terms;
    long double scale = 0;
    while (!heap.empty()) {
        auto cur = heap.top(); heap.pop();
        long double v = static_cast<long double>(a.val[cur.p]) * b.val[cur.q];
        scale += std::abs(v);
        if (terms.empty() || terms.back().first != cur.col) terms.emplace_back(cur.col, v);
        else terms.back().second += v;
        if (++cur.q < cur.end) { cur.col = b.col[cur.q]; heap.push(cur); }
    }
    return {std::move(terms), scale};
}
inline bool compare(const Row& a, const Row& b, long double scale, double& err) {
    if (a.size() != b.size()) return false;
    bool ok = true;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].first != b[i].first) ok = false;
        if (!close(a[i].second, b[i].second, scale, err)) ok = false;
    }
    return ok;
}
inline bool identities(const HostCSR& a, const HostCSR& b, const HostCSR& c, double& err) {
    std::mt19937_64 rng(0x6672656976616c64ULL);
    std::vector<long double> x(b.cols), bx(b.rows), bound(b.rows);
    for (int trial = 0; trial < 3; ++trial) {
        for (auto& v : x) v = (rng() & 1) ? 1 : -1;
        for (int r = 0; r < b.rows; ++r) {
            bx[r] = bound[r] = 0;
            for (int p = b.row[r]; p < b.row[r + 1]; ++p) { bx[r] += b.val[p] * x[b.col[p]]; bound[r] += std::abs(b.val[p]); }
        }
        for (int r = 0; r < a.rows; ++r) {
            long double expected = 0, actual = 0, scale = 0;
            for (int p = a.row[r]; p < a.row[r + 1]; ++p) { expected += a.val[p] * bx[a.col[p]]; scale += std::abs(a.val[p]) * bound[a.col[p]]; }
            for (int p = c.row[r]; p < c.row[r + 1]; ++p) actual += c.val[p] * x[c.col[p]];
            if (!close(expected, actual, scale, err)) return false;
        }
    }
    return true;
}

struct VerifyResult { bool passed; double max_error; int sample_rows; };

// Verify a single backend's product C against the CPU reference: sampled rows
// compared exactly to reference_row, plus 3 Freivalds identity trials over the
// whole matrix. Returns {passed, max_error, sample_rows}.
inline VerifyResult verify(const HostCSR& a, const HostCSR& b, const HostCSR& c) {
    bool passed = true; double maxerr = 0;
    std::vector<int> samples;
    int count = (a.rows <= 128 && a.val.size() + b.val.size() <= 10000) ? a.rows : std::min(a.rows, 32);
    for (int i = 0; i < count; ++i) samples.push_back(count == 1 ? 0 : int(int64_t(i) * (a.rows - 1) / (count - 1)));
    for (int i : samples) {
        auto ref = reference_row(a, b, i);
        if (!compare(ref.first, row(c, i), ref.second, maxerr)) passed = false;
    }
    if (!identities(a, b, c, maxerr)) passed = false;
    return {passed, maxerr, int(samples.size())};
}
} // namespace cpu_verify
} // namespace spgemm
