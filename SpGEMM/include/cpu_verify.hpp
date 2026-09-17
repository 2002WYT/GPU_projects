#pragma once
#include "matrix_io.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <queue>
#include <random>
#include <utility>
#include <vector>

namespace spgemm {
namespace cpu_verify {
using Complex = std::complex<long double>;
using Row = std::vector<std::pair<int, Complex>>;

inline Complex at(const HostCSR& a, int p) {
    return {static_cast<long double>(a.val[p]), a.complex ? static_cast<long double>(a.imag[p]) : 0.0L};
}
inline Row row(const HostCSR& c, int r) {
    Row values; values.reserve(c.row[r + 1] - c.row[r]);
    for (int p = c.row[r]; p < c.row[r + 1]; ++p) values.emplace_back(c.col[p], at(c, p));
    std::sort(values.begin(), values.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
    for (size_t i = 1; i < values.size(); ++i)
        if (values[i - 1].first == values[i].first) throw std::runtime_error("duplicate output column");
    return values;
}
inline bool finite(Complex value) { return std::isfinite(value.real()) && std::isfinite(value.imag()); }
inline bool close(Complex actual, Complex expected, long double scale, double& maxerr) {
    const long double error = std::abs(actual - expected);
    maxerr = std::max(maxerr, static_cast<double>(error));
    return finite(actual) && finite(expected) &&
           error <= 1e-9L + 1e-8L * std::max({std::abs(actual), std::abs(expected), scale});
}
inline std::pair<Row, long double> reference_row(const HostCSR& a, const HostCSR& b, int r) {
    struct Cursor { int col, p, q, end; };
    auto later = [](const Cursor& x, const Cursor& y) { return x.col > y.col; };
    std::priority_queue<Cursor, std::vector<Cursor>, decltype(later)> heap(later);
    for (int p = a.row[r]; p < a.row[r + 1]; ++p) {
        const int q = b.row[a.col[p]], end = b.row[a.col[p] + 1];
        if (q < end) heap.push({b.col[q], p, q, end});
    }
    Row terms;
    long double scale = 0;
    while (!heap.empty()) {
        auto cur = heap.top(); heap.pop();
        const Complex value = at(a, cur.p) * at(b, cur.q);
        scale += std::abs(value);
        if (terms.empty() || terms.back().first != cur.col) terms.emplace_back(cur.col, value);
        else terms.back().second += value;
        if (++cur.q < cur.end) { cur.col = b.col[cur.q]; heap.push(cur); }
    }
    return {std::move(terms), scale};
}
inline bool compare(const Row& expected, const Row& actual, long double scale, double& error) {
    if (expected.size() != actual.size()) return false;
    bool ok = true;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (expected[i].first != actual[i].first) ok = false;
        if (!close(actual[i].second, expected[i].second, scale, error)) ok = false;
    }
    return ok;
}
inline bool identities(const HostCSR& a, const HostCSR& b, const HostCSR& c, double& error) {
    std::mt19937_64 rng(0x6672656976616c64ULL);
    std::vector<Complex> x(b.cols), bx(b.rows);
    std::vector<long double> bound(b.rows);
    for (int trial = 0; trial < 3; ++trial) {
        for (auto& value : x) value = Complex((rng() & 1) ? 1 : -1, 0);
        for (int r = 0; r < b.rows; ++r) {
            bx[r] = {}; bound[r] = 0;
            for (int p = b.row[r]; p < b.row[r + 1]; ++p) {
                bx[r] += at(b, p) * x[b.col[p]];
                bound[r] += std::abs(at(b, p));
            }
        }
        for (int r = 0; r < a.rows; ++r) {
            Complex expected{}, actual{};
            long double scale = 0;
            for (int p = a.row[r]; p < a.row[r + 1]; ++p) {
                expected += at(a, p) * bx[a.col[p]];
                scale += std::abs(at(a, p)) * bound[a.col[p]];
            }
            for (int p = c.row[r]; p < c.row[r + 1]; ++p) actual += at(c, p) * x[c.col[p]];
            if (!close(actual, expected, scale, error)) return false;
        }
    }
    return true;
}

struct VerifyResult { bool passed; double max_error; int sample_rows; };

inline VerifyResult verify(const HostCSR& a, const HostCSR& b, const HostCSR& c) {
    detail::validate(a); detail::validate(b);
    bool passed = c.rows == a.rows && c.cols == b.cols;
    double maxerr = 0;
    std::vector<int> samples;
    const int count = (a.rows <= 128 && a.val.size() + b.val.size() <= 10000) ? a.rows : std::min(a.rows, 32);
    for (int i = 0; i < count; ++i) samples.push_back(count == 1 ? 0 : int(int64_t(i) * (a.rows - 1) / (count - 1)));
    if (passed) {
        for (int r : samples) {
            const auto reference = reference_row(a, b, r);
            if (!compare(reference.first, row(c, r), reference.second, maxerr)) passed = false;
        }
        if (!identities(a, b, c, maxerr)) passed = false;
    }
    return {passed, maxerr, static_cast<int>(samples.size())};
}
} // namespace cpu_verify
} // namespace spgemm
