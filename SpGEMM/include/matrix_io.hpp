#pragma once
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace spgemm {
struct HostCSR {
    int rows = 0, cols = 0;
    std::vector<int> row, col;
    std::vector<double> val, imag;
    bool complex = false;
    bool is_complex() const { return complex; }
};

namespace detail {
inline void require(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
inline bool data_line(std::istream& in, std::string& line) {
    while (std::getline(in, line)) {
        const auto p = line.find_first_not_of(" \t\r\n");
        if (p != std::string::npos && line[p] != '%') return true;
    }
    return false;
}
inline void end_line(std::istringstream& line) { line >> std::ws; require(line.eof(), "unexpected extra Matrix Market field"); }
inline void lowercase(std::string& s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
}
inline void validate(const HostCSR& a) {
    require(a.rows >= 0 && a.cols >= 0, "negative CSR dimensions");
    require(a.row.size() == static_cast<size_t>(a.rows) + 1 && !a.row.empty() && a.row[0] == 0, "invalid CSR row offsets");
    require(a.col.size() == a.val.size() && a.val.size() <= static_cast<size_t>(INT32_MAX), "CSR nnz exceeds int32 or value count mismatch");
    require((a.complex && a.imag.size() == a.val.size()) || (!a.complex && a.imag.empty()), "invalid CSR complex storage");
    require(a.row.back() == static_cast<int>(a.val.size()), "CSR terminal offset mismatch");
    for (int r = 0; r < a.rows; ++r) {
        require(a.row[r] >= 0 && a.row[r] <= a.row[r + 1] && a.row[r + 1] <= a.row.back(), "invalid CSR row interval");
        for (int p = a.row[r]; p < a.row[r + 1]; ++p) {
            require(a.col[p] >= 0 && a.col[p] < a.cols && (p == a.row[r] || a.col[p - 1] < a.col[p]), "CSR columns must be sorted and unique");
            require(std::isfinite(a.val[p]) && (!a.complex || std::isfinite(a.imag[p])), "nonfinite CSR value");
        }
    }
}

struct Entry { int r, c; double re, im; };
inline HostCSR canonical(int rows, int cols, std::vector<Entry>& entries, bool complex) {
    std::stable_sort(entries.begin(), entries.end(), [](const Entry& x, const Entry& y) {
        return x.r < y.r || (x.r == y.r && x.c < y.c);
    });
    HostCSR a; a.rows = rows; a.cols = cols; a.complex = complex;
    a.row.assign(static_cast<size_t>(rows) + 1, 0);
    a.col.reserve(entries.size()); a.val.reserve(entries.size());
    if (complex) a.imag.reserve(entries.size());
    int previous_row = -1, previous_col = -1;
    for (const auto& e : entries) {
        if (e.r == previous_row && e.c == previous_col) {
            a.val.back() += e.re;
            if (complex) a.imag.back() += e.im;
            require(std::isfinite(a.val.back()) && (!complex || std::isfinite(a.imag.back())), "duplicate sum is nonfinite");
        } else {
            require(a.val.size() < static_cast<size_t>(INT32_MAX), "canonical nnz exceeds int32");
            a.col.push_back(e.c); a.val.push_back(e.re);
            if (complex) a.imag.push_back(e.im);
            ++a.row[static_cast<size_t>(e.r) + 1];
            previous_row = e.r; previous_col = e.c;
        }
    }
    for (size_t r = 1; r < a.row.size(); ++r) a.row[r] += a.row[r - 1];
    return a;
}

struct SplitMix64 {
    uint64_t state;
    uint64_t next() {
        uint64_t z = (state += UINT64_C(0x9e3779b97f4a7c15));
        z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
        return z ^ (z >> 31);
    }
    uint64_t bounded(uint64_t n) {
        require(n != 0, "random bound must be positive");
        const uint64_t threshold = (uint64_t(0) - n) % n;
        for (;;) { const uint64_t x = next(); if (x >= threshold) return x % n; }
    }
};
inline double random_value(SplitMix64& rng) {
    const int raw = static_cast<int>(rng.bounded(2048));
    return double(raw < 1024 ? raw - 1024 : raw - 1023) / 1024.0;
}
} // namespace detail

inline HostCSR read_mtx(std::string path) {
    std::ifstream in(path); in.imbue(std::locale::classic());
    detail::require(bool(in), "cannot open matrix: " + path);
    std::string line, magic, object, format, field, symmetry;
    detail::require(bool(std::getline(in, line)), "empty Matrix Market file");
    std::istringstream header(line); header >> magic >> object >> format >> field >> symmetry;
    detail::lowercase(object); detail::lowercase(format); detail::lowercase(field); detail::lowercase(symmetry);
    detail::require(magic == "%%MatrixMarket" && object == "matrix" && format == "coordinate", "only Matrix Market coordinate matrices are supported");
    detail::require(field == "real" || field == "integer" || field == "pattern" || field == "complex", "unsupported Matrix Market field");
    detail::require(symmetry == "general" || symmetry == "symmetric" || symmetry == "skew-symmetric" || symmetry == "hermitian", "unsupported Matrix Market symmetry");
    detail::end_line(header);
    detail::require(detail::data_line(in, line), "missing matrix dimensions");
    std::istringstream dimensions(line); long long rows = -1, cols = -1, count = -1;
    detail::require(bool(dimensions >> rows >> cols >> count), "invalid matrix dimensions"); detail::end_line(dimensions);
    detail::require(rows >= 0 && cols >= 0 && count >= 0 && rows <= INT32_MAX && cols <= INT32_MAX && count <= INT32_MAX, "matrix dimensions or nnz outside int32 range");
    detail::require(symmetry == "general" || rows == cols, "structured matrix must be square");
    detail::require((rows > 0 && cols > 0) || count == 0, "zero dimension matrix has nonzero entries");
    const bool complex = field == "complex";
    std::vector<detail::Entry> entries; entries.reserve(static_cast<size_t>(count));
    for (long long i = 0; i < count; ++i) {
        detail::require(detail::data_line(in, line), "fewer entries than declared");
        std::istringstream record(line); record.imbue(std::locale::classic());
        long long r = 0, c = 0; double re = 1, im = 0;
        detail::require(bool(record >> r >> c), "invalid matrix coordinate");
        if (field == "integer") { long long integer = 0; detail::require(bool(record >> integer), "invalid integer value"); re = static_cast<double>(integer); }
        else if (field == "real") detail::require(bool(record >> re), "invalid real value");
        else if (field == "complex") detail::require(bool(record >> re >> im), "invalid complex value");
        detail::end_line(record);
        detail::require(r >= 1 && r <= rows && c >= 1 && c <= cols, "matrix coordinate out of bounds");
        detail::require(std::isfinite(re) && std::isfinite(im), "matrix value is nonfinite");
        detail::require(symmetry != "skew-symmetric" || r != c || (re == 0 && im == 0), "skew-symmetric diagonal must be zero");
        detail::require(symmetry != "hermitian" || r != c || im == 0, "Hermitian diagonal must be real");
        const size_t added = (symmetry != "general" && r != c) ? 2 : 1;
        detail::require(entries.size() <= static_cast<size_t>(INT32_MAX) - added, "expanded nnz exceeds int32");
        entries.push_back({static_cast<int>(r - 1), static_cast<int>(c - 1), re, im});
        if (added == 2) {
            double mirror_re = re, mirror_im = im;
            if (symmetry == "skew-symmetric") { mirror_re = -mirror_re; mirror_im = -mirror_im; }
            else if (symmetry == "hermitian") mirror_im = -mirror_im;
            entries.push_back({static_cast<int>(c - 1), static_cast<int>(r - 1), mirror_re, mirror_im});
        }
    }
    detail::require(!detail::data_line(in, line), "more entries than declared");
    return detail::canonical(static_cast<int>(rows), static_cast<int>(cols), entries, complex);
}

inline HostCSR generate_b(const HostCSR& a, uint64_t seed = 20260910) {
    detail::validate(a);
    HostCSR b; b.rows = a.cols; b.cols = a.rows; b.complex = a.complex;
    b.row.assign(static_cast<size_t>(b.rows) + 1, 0);
    const uint64_t population = uint64_t(b.rows) * uint64_t(b.cols), count = a.val.size();
    detail::require(count <= population, "requested B nnz exceeds its coordinate capacity");
    detail::SplitMix64 rng{seed};
    std::vector<uint64_t> positions; positions.reserve(static_cast<size_t>(count));
    std::unordered_set<uint64_t> selected; selected.reserve(static_cast<size_t>(count));
    for (uint64_t j = population - count; j < population; ++j) {
        const uint64_t candidate = rng.bounded(j + 1);
        const uint64_t coordinate = selected.insert(candidate).second ? candidate : j;
        if (coordinate != candidate) selected.insert(coordinate);
        positions.push_back(coordinate);
    }
    std::sort(positions.begin(), positions.end());
    b.col.reserve(static_cast<size_t>(count)); b.val.reserve(static_cast<size_t>(count));
    if (b.complex) b.imag.reserve(static_cast<size_t>(count));
    for (uint64_t p : positions) {
        ++b.row[static_cast<size_t>(p / uint64_t(b.cols)) + 1];
        b.col.push_back(static_cast<int>(p % uint64_t(b.cols)));
        b.val.push_back(detail::random_value(rng));
        if (b.complex) b.imag.push_back(detail::random_value(rng));
    }
    for (size_t r = 1; r < b.row.size(); ++r) b.row[r] += b.row[r - 1];
    return b;
}

inline HostCSR transpose_impl(const HostCSR& a, bool conjugate) {
    detail::validate(a);
    HostCSR t; t.rows = a.cols; t.cols = a.rows; t.complex = a.complex;
    t.row.assign(static_cast<size_t>(t.rows) + 1, 0);
    for (int x : a.col) ++t.row[static_cast<size_t>(x) + 1];
    for (size_t r = 1; r < t.row.size(); ++r) t.row[r] += t.row[r - 1];
    t.col.resize(a.val.size()); t.val.resize(a.val.size());
    if (t.complex) t.imag.resize(a.val.size());
    std::vector<int> cursor(t.row.begin(), t.row.end());
    for (int r = 0; r < a.rows; ++r)
        for (int p = a.row[r]; p < a.row[r + 1]; ++p) {
            const int pos = cursor[static_cast<size_t>(a.col[p])]++;
            t.col[pos] = r; t.val[pos] = a.val[p];
            if (t.complex) t.imag[pos] = conjugate ? -a.imag[p] : a.imag[p];
        }
    return t;
}
inline HostCSR transpose(const HostCSR& a) { return transpose_impl(a, false); }
inline HostCSR conjugate_transpose(const HostCSR& a) { return transpose_impl(a, a.complex); }

inline std::string fingerprint(const HostCSR& a) {
    detail::validate(a); static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559, "IEEE754 binary64 required");
    uint64_t hash = UINT64_C(14695981039346656037);
    const auto add = [&hash](uint64_t x, int bytes) { for (int i = 0; i < bytes; ++i) { hash ^= x & 255; hash *= UINT64_C(1099511628211); x >>= 8; } };
    add(static_cast<uint32_t>(a.rows), 4); add(static_cast<uint32_t>(a.cols), 4); add(a.val.size(), 8); add(a.complex ? 1 : 0, 1);
    for (int x : a.row) add(static_cast<uint32_t>(x), 4);
    for (int x : a.col) add(static_cast<uint32_t>(x), 4);
    for (double x : a.val) { uint64_t bits = 0; std::memcpy(&bits, &x, 8); add(bits, 8); }
    for (double x : a.imag) { uint64_t bits = 0; std::memcpy(&bits, &x, 8); add(bits, 8); }
    std::ostringstream out; out.imbue(std::locale::classic()); out << std::hex << std::setw(16) << std::setfill('0') << hash; return out.str();
}

inline HostCSR cpu_product(const HostCSR& a, const HostCSR& b) {
    detail::validate(a); detail::validate(b); detail::require(a.cols == b.rows, "incompatible product dimensions");
    HostCSR c; c.rows = a.rows; c.cols = b.cols; c.complex = a.complex || b.complex; c.row.push_back(0);
    std::unordered_map<int, std::pair<double, double>> sums; std::vector<int> columns;
    for (int r = 0; r < a.rows; ++r) {
        sums.clear(); columns.clear();
        for (int p = a.row[r]; p < a.row[r + 1]; ++p) {
            const double ar = a.val[p], ai = a.complex ? a.imag[p] : 0;
            for (int q = b.row[a.col[p]]; q < b.row[a.col[p] + 1]; ++q) {
                const double br = b.val[q], bi = b.complex ? b.imag[q] : 0;
                auto& sum = sums[b.col[q]];
                sum.first += ar * br - ai * bi;
                sum.second += ar * bi + ai * br;
            }
        }
        detail::require(sums.size() <= static_cast<size_t>(INT32_MAX) - c.val.size(), "product nnz exceeds int32");
        for (const auto& pair : sums) columns.push_back(pair.first);
        std::sort(columns.begin(), columns.end());
        for (int col : columns) {
            const auto v = sums.at(col);
            detail::require(std::isfinite(v.first) && std::isfinite(v.second), "CPU product contains nonfinite value");
            c.col.push_back(col); c.val.push_back(v.first); if (c.complex) c.imag.push_back(v.second);
        }
        c.row.push_back(static_cast<int>(c.val.size()));
    }
    return c;
}
} // namespace spgemm
