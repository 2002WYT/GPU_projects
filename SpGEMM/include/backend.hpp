#pragma once
#include "matrix_io.hpp"
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace spgemm {
class Backend {
public:
    virtual ~Backend() = default;
    virtual void multiply() = 0;
    virtual HostCSR download() = 0;
};
std::unique_ptr<Backend> make_backend(const HostCSR&, const HostCSR&);
const char* backend_name();

inline void validate_product(const HostCSR& c) {
    if (c.rows < 0 || c.cols < 0 || c.val.size() > size_t(INT32_MAX) ||
        c.row.size() != size_t(c.rows) + 1 || c.col.size() != c.val.size() ||
        (c.complex ? c.imag.size() != c.val.size() : !c.imag.empty()) ||
        c.row.front() != 0 || c.row.back() != int(c.val.size()))
        throw std::runtime_error("invalid CSR dimensions or endpoints");
    for (int r = 0; r < c.rows; ++r) {
        if (c.row[r] < 0 || c.row[r] > c.row[r+1] || c.row[r+1] > int(c.val.size()))
            throw std::runtime_error("invalid CSR row offsets");
        for (int p = c.row[r]; p < c.row[r+1]; ++p)
            if (c.col[p] < 0 || c.col[p] >= c.cols || !std::isfinite(c.val[p]) ||
                (c.complex && !std::isfinite(c.imag[p])))
                throw std::runtime_error("invalid CSR column or non-finite value");
    }
}
} // namespace spgemm
