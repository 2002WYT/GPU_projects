#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <climits>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

struct MtxMatrix {
    long nrows;
    long ncols;
    long nnz;
    int symmetric;
    int* rowptr;
    int* colind;
    double* values;
};

inline void mtx_init(MtxMatrix* matrix)
{
    if (!matrix) return;
    matrix->nrows = 0;
    matrix->ncols = 0;
    matrix->nnz = 0;
    matrix->symmetric = 0;
    matrix->rowptr = nullptr;
    matrix->colind = nullptr;
    matrix->values = nullptr;
}

inline void mtx_free(MtxMatrix* matrix)
{
    if (!matrix) return;
    std::free(matrix->rowptr);
    std::free(matrix->colind);
    std::free(matrix->values);
    mtx_init(matrix);
}

namespace mtx_reader_detail {

struct Entry {
    int row;
    int column;
    double value;
};

inline std::string lowercase(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
        [](unsigned char character) {
            return static_cast<char>(std::tolower(character));
        });
    return value;
}

inline bool next_data_line(std::ifstream& input, std::string& line)
{
    while (std::getline(input, line)) {
        const auto first = line.find_first_not_of(" \t\r");
        if (first != std::string::npos && line[first] != '%') return true;
    }
    return false;
}

}  // namespace mtx_reader_detail

inline int mtx_read(const char* path, MtxMatrix* matrix)
{
    if (!path || !matrix) return 0;
    mtx_free(matrix);

    std::ifstream input(path);
    if (!input) return 0;

    std::string header;
    if (!std::getline(input, header)) return 0;

    std::istringstream header_stream(header);
    std::string banner;
    std::string object;
    std::string format;
    std::string field;
    std::string symmetry;
    header_stream >> banner >> object >> format >> field >> symmetry;
    banner = mtx_reader_detail::lowercase(banner);
    object = mtx_reader_detail::lowercase(object);
    format = mtx_reader_detail::lowercase(format);
    field = mtx_reader_detail::lowercase(field);
    symmetry = mtx_reader_detail::lowercase(symmetry);

    if (banner != "%%matrixmarket" || object != "matrix" ||
        format != "coordinate") return 0;
    if (field != "real" && field != "integer" && field != "pattern") return 0;

    const bool symmetric = symmetry == "symmetric" || symmetry == "hermitian";
    const bool skew_symmetric = symmetry == "skew-symmetric";
    if (!symmetric && !skew_symmetric && symmetry != "general") return 0;

    std::string line;
    if (!mtx_reader_detail::next_data_line(input, line)) return 0;

    long long rows = 0;
    long long columns = 0;
    long long input_nnz = 0;
    std::istringstream dimensions(line);
    if (!(dimensions >> rows >> columns >> input_nnz) ||
        rows <= 0 || columns <= 0 || input_nnz < 0 ||
        rows > INT_MAX || columns > INT_MAX) return 0;

    std::vector<mtx_reader_detail::Entry> entries;
    entries.reserve(static_cast<size_t>(input_nnz) *
        ((symmetric || skew_symmetric) ? 2u : 1u));

    for (long long index = 0; index < input_nnz; ++index) {
        if (!mtx_reader_detail::next_data_line(input, line)) return 0;
        std::istringstream entry_stream(line);
        long long row = 0;
        long long column = 0;
        double value = 1.0;
        if (!(entry_stream >> row >> column)) return 0;
        if (field != "pattern" && !(entry_stream >> value)) return 0;
        --row;
        --column;
        if (row < 0 || row >= rows || column < 0 || column >= columns ||
            !std::isfinite(value)) return 0;

        entries.push_back({static_cast<int>(row), static_cast<int>(column), value});
        if ((symmetric || skew_symmetric) && row != column) {
            entries.push_back({static_cast<int>(column), static_cast<int>(row),
                skew_symmetric ? -value : value});
        }
    }

    std::sort(entries.begin(), entries.end(),
        [](const auto& left, const auto& right) {
            return std::tie(left.row, left.column) <
                   std::tie(right.row, right.column);
        });

    std::vector<mtx_reader_detail::Entry> combined;
    combined.reserve(entries.size());
    for (const auto& entry : entries) {
        if (!combined.empty() && combined.back().row == entry.row &&
            combined.back().column == entry.column) {
            combined.back().value += entry.value;
        } else {
            combined.push_back(entry);
        }
    }
    combined.erase(std::remove_if(combined.begin(), combined.end(),
        [](const auto& entry) { return entry.value == 0.0; }), combined.end());

    if (combined.size() > static_cast<size_t>(INT_MAX)) return 0;

    matrix->rowptr = static_cast<int*>(
        std::calloc(static_cast<size_t>(rows) + 1, sizeof(int)));
    matrix->colind = static_cast<int*>(
        std::malloc(combined.size() * sizeof(int)));
    matrix->values = static_cast<double*>(
        std::malloc(combined.size() * sizeof(double)));
    if (!matrix->rowptr || (!combined.empty() &&
        (!matrix->colind || !matrix->values))) {
        mtx_free(matrix);
        return 0;
    }

    for (const auto& entry : combined) ++matrix->rowptr[entry.row + 1];
    for (int row = 0; row < rows; ++row)
        matrix->rowptr[row + 1] += matrix->rowptr[row];
    for (size_t index = 0; index < combined.size(); ++index) {
        matrix->colind[index] = combined[index].column;
        matrix->values[index] = combined[index].value;
    }

    matrix->nrows = static_cast<long>(rows);
    matrix->ncols = static_cast<long>(columns);
    matrix->nnz = static_cast<long>(combined.size());
    matrix->symmetric = symmetric ? 1 : 0;
    return 1;
}

inline double* mtx_compute_rhs_one(const MtxMatrix* matrix)
{
    if (!matrix || !matrix->rowptr || !matrix->colind || !matrix->values)
        return nullptr;
    auto* rhs = static_cast<double*>(
        std::malloc(static_cast<size_t>(matrix->nrows) * sizeof(double)));
    if (!rhs) return nullptr;
    for (long row = 0; row < matrix->nrows; ++row) {
        double sum = 0.0;
        for (int position = matrix->rowptr[row];
             position < matrix->rowptr[row + 1]; ++position)
            sum += matrix->values[position];
        rhs[row] = sum;
    }
    return rhs;
}

inline double mtx_relative_residual(
    const MtxMatrix* matrix, const double* x, const double* rhs)
{
    if (!matrix || !x || !rhs) return std::numeric_limits<double>::infinity();
    double residual_squared = 0.0;
    double rhs_squared = 0.0;
    for (long row = 0; row < matrix->nrows; ++row) {
        double product = 0.0;
        for (int position = matrix->rowptr[row];
             position < matrix->rowptr[row + 1]; ++position)
            product += matrix->values[position] * x[matrix->colind[position]];
        const double residual = rhs[row] - product;
        residual_squared += residual * residual;
        rhs_squared += rhs[row] * rhs[row];
    }
    return std::sqrt(residual_squared) /
        (std::sqrt(rhs_squared) > 0.0 ? std::sqrt(rhs_squared) : 1.0);
}

inline double mtx_relative_error(const MtxMatrix* matrix, const double* x)
{
    if (!matrix || !x || matrix->ncols <= 0)
        return std::numeric_limits<double>::infinity();
    double error_squared = 0.0;
    for (long index = 0; index < matrix->ncols; ++index) {
        const double error = x[index] - 1.0;
        error_squared += error * error;
    }
    return std::sqrt(error_squared / static_cast<double>(matrix->ncols));
}
