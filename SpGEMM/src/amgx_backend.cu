#include "backend.hpp"
#include <amgx_c.h>
#include <csr_multiply.h>
#include <matrix.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <type_traits>
#include <vector>

namespace spgemm {
namespace {
void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

using RealConfig = amgx::TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>;
using ComplexConfig = amgx::TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matDoubleComplex, AMGX_indInt>;

template<class DeviceValue> struct ValueCodec;
template<> struct ValueCodec<double> {
    static std::vector<double> pack(const HostCSR& h) { return h.val; }
    static void unpack(const std::vector<double>& values, HostCSR& h) { h.val = values; }
};
template<> struct ValueCodec<cuDoubleComplex> {
    static std::vector<cuDoubleComplex> pack(const HostCSR& h) {
        std::vector<cuDoubleComplex> values(h.val.size());
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = make_cuDoubleComplex(h.val[i], h.complex ? h.imag[i] : 0.0);
        return values;
    }
    static void unpack(const std::vector<cuDoubleComplex>& values, HostCSR& h) {
        h.complex = true; h.val.resize(values.size()); h.imag.resize(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            h.val[i] = cuCreal(values[i]); h.imag[i] = cuCimag(values[i]);
        }
    }
};

template<class Config, class DeviceValue>
class Amgx final : public Backend {
    using Matrix = amgx::Matrix<Config>;
    using Multiply = amgx::CSR_Multiply<Config>;
    std::unique_ptr<Matrix> a, b, c;
    void* workspace = nullptr;
    bool initialized = false;

    static void upload(Matrix& device, const HostCSR& host) {
        device.set_initialized(0); device.addProps(amgx::CSR);
        device.resize(host.rows, host.cols, int(host.val.size()), 1, 1, 1);
        check(cudaMemcpy(device.row_offsets.raw(), host.row.data(), host.row.size() * sizeof(int), cudaMemcpyHostToDevice));
        if (!host.val.empty()) {
            const auto values = ValueCodec<DeviceValue>::pack(host);
            check(cudaMemcpy(device.col_indices.raw(), host.col.data(), host.col.size() * sizeof(int), cudaMemcpyHostToDevice));
            check(cudaMemcpy(device.values.raw(), values.data(), values.size() * sizeof(DeviceValue), cudaMemcpyHostToDevice));
        }
        device.set_initialized(1);
    }

public:
    Amgx(const HostCSR& aa, const HostCSR& bb) {
        if (aa.cols != bb.rows) throw std::runtime_error("A.cols must equal B.rows");
        if (AMGX_initialize() != AMGX_RC_OK) throw std::runtime_error("AMGX_initialize failed");
        initialized = true;
        a = std::make_unique<Matrix>(); b = std::make_unique<Matrix>(); c = std::make_unique<Matrix>();
        upload(*a, aa); upload(*b, bb); c->addProps(amgx::CSR);
    }
    ~Amgx() override {
        if (workspace) Multiply::csr_workspace_delete(workspace);
        c.reset(); b.reset(); a.reset();
        if (initialized) AMGX_finalize();
    }
    void multiply() override {
        if (!workspace) {
            workspace = Multiply::csr_workspace_create();
            if constexpr (std::is_same<DeviceValue, cuDoubleComplex>::value) {
                // AMGX 2.5.0's precompiled complex hash kernel has incorrect
                // cuDoubleComplex shuffle argument ordering. Zero attempts is
                // the library's supported route to its cuSPARSE SpGEMM
                // fallback, which is correct for CUDA_C_64F.
                using Impl = amgx::CSR_Multiply_Impl<Config>;
                static_cast<Impl*>(workspace)->set_max_attempts(0);
            }
        }
        Multiply::csr_multiply(*a, *b, *c, workspace);
    }
    HostCSR download() override {
        HostCSR host; host.rows = c->get_num_rows(); host.cols = c->get_num_cols();
        const int nz = c->get_num_nz();
        host.row.resize(size_t(host.rows) + 1); host.col.resize(nz);
        check(cudaMemcpy(host.row.data(), c->row_offsets.raw(), host.row.size() * sizeof(int), cudaMemcpyDeviceToHost));
        if (nz) {
            std::vector<DeviceValue> values(nz);
            check(cudaMemcpy(host.col.data(), c->col_indices.raw(), size_t(nz) * sizeof(int), cudaMemcpyDeviceToHost));
            check(cudaMemcpy(values.data(), c->values.raw(), size_t(nz) * sizeof(DeviceValue), cudaMemcpyDeviceToHost));
            ValueCodec<DeviceValue>::unpack(values, host);
        } else if constexpr (std::is_same<DeviceValue, cuDoubleComplex>::value) {
            host.complex = true;
        }
        return host;
    }
};
} // namespace

std::unique_ptr<Backend> make_backend(const HostCSR& a, const HostCSR& b) {
    if (a.complex || b.complex) return std::make_unique<Amgx<ComplexConfig, cuDoubleComplex>>(a, b);
    return std::make_unique<Amgx<RealConfig, double>>(a, b);
}
const char* backend_name() { return "amgx"; }
} // namespace spgemm
