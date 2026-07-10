// Complete benchmark for five GPU SpMV algorithms:
//   1. CSR Scalar
//   2. CSR Vector
//   3. CSR Adaptive (CSR Stream + long-row block reduction)
//   4. PCSR (product kernel + segmented row summation kernel)
//   5. LightSpMV-style dynamic row scheduling
//
// Main corrections compared with the original version:
//   - GPU allocation, H2D copies, and Adaptive preprocessing are outside timing.
//   - CUDA events measure only the operations required for one SpMV call.
//   - CSR Vector no longer uses __syncthreads() inside a conditional branch.
//   - LightSpMV keeps the whole warp active for the final partial row batch.
//   - Correctness is checked against a CPU reference using a non-constant x.
//   - PCSR uses a separate estimated-byte count because it writes and reads d_v.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

constexpr int WARP_SIZE = 32;
constexpr int ADAPTIVE_NNZ_PER_BLOCK = 1024;
constexpr int PCSR_THREADS_PER_BLOCK = 256;
constexpr int PCSR_SHARED_ELEMENTS = 4096; // number of double elements, not bytes
constexpr unsigned FULL_WARP_MASK = 0xffffffffu;

inline void cuda_check(
    cudaError_t status,
    const char* expression,
    const char* file,
    int line)
{
    if (status != cudaSuccess)
    {
        std::ostringstream oss;
        oss << "CUDA error at " << file << ':' << line
            << " for " << expression << ": "
            << cudaGetErrorString(status);
        throw std::runtime_error(oss.str());
    }
}

#define CUDA_CHECK(call) cuda_check((call), #call, __FILE__, __LINE__)

inline int ceil_div(int a, int b)
{
    return (a + b - 1) / b;
}

inline bool is_power_of_two(int x)
{
    return x > 0 && (x & (x - 1)) == 0;
}

const char* kernel_name(int kernel_type)
{
    switch (kernel_type)
    {
        case 1: return "CSR Scalar";
        case 2: return "CSR Vector";
        case 3: return "CSR Adaptive";
        case 4: return "PCSR";
        case 5: return "LightSpMV";
        default: return "Unknown";
    }
}

// matrix_mode = 0: original pattern 3, 8, 16, 32, 64, 128
// matrix_mode = 1: the same basic pattern plus sparse 1024/4096-nnz rows,
//                  so CSR Adaptive's long-row path is actually exercised.
int choose_row_length(int row, int n, int matrix_mode)
{
    static constexpr int pattern[] = {3, 8, 16, 32, 64, 128};

    int len = pattern[row % 6];

    if (matrix_mode == 1)
    {
        if (row % 1000 == 0)
        {
            len = 4096;
        }
        else if (row % 100 == 0)
        {
            len = 1024;
        }
    }

    return std::min(len, n);
}

template <typename T>
void build_irregular_csr_matrix(
    int n,
    int matrix_mode,
    std::vector<int>& row_ptr,
    std::vector<int>& col_idx,
    std::vector<T>& values)
{
    if (n <= 0)
    {
        throw std::invalid_argument("n must be positive.");
    }
    if (matrix_mode != 0 && matrix_mode != 1)
    {
        throw std::invalid_argument("matrix_mode must be 0 or 1.");
    }

    row_ptr.assign(static_cast<std::size_t>(n) + 1, 0);

    long long nnz_ll = 0;
    for (int row = 0; row < n; ++row)
    {
        const int len = choose_row_length(row, n, matrix_mode);
        nnz_ll += len;

        if (nnz_ll > static_cast<long long>(INT_MAX))
        {
            throw std::overflow_error(
                "nnz exceeds INT_MAX; this program uses 32-bit CSR indices.");
        }

        row_ptr[row + 1] = static_cast<int>(nnz_ll);
    }

    const int nnz = static_cast<int>(nnz_ll);
    col_idx.resize(nnz);
    values.resize(nnz);

    for (int row = 0; row < n; ++row)
    {
        const int begin = row_ptr[row];
        const int end = row_ptr[row + 1];
        const int len = end - begin;

        for (int local = 0; local < len; ++local)
        {
            // Deterministic but non-contiguous column access pattern.
            // Duplicate column indices are allowed by this synthetic CSR test.
            const long long raw_col =
                static_cast<long long>(row) * 17LL
                + static_cast<long long>(local) * 9973LL
                + static_cast<long long>(local) * static_cast<long long>(local) * 13LL;

            col_idx[begin + local] = static_cast<int>(raw_col % n);

            // The values are well-scaled, but x is non-constant, so a wrong
            // column access can no longer pass the correctness check by accident.
            values[begin + local] = static_cast<T>(1.0) / static_cast<T>(len);
        }
    }
}

template <typename T>
void spmv_cpu_reference(
    int n,
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_idx,
    const std::vector<T>& values,
    const std::vector<T>& x,
    std::vector<T>& y)
{
    y.assign(n, static_cast<T>(0));

    for (int row = 0; row < n; ++row)
    {
        T sum = 0;
        for (int jj = row_ptr[row]; jj < row_ptr[row + 1]; ++jj)
        {
            sum += values[jj] * x[col_idx[jj]];
        }
        y[row] = sum;
    }
}

template <typename T>
T max_abs_error(const std::vector<T>& actual, const std::vector<T>& reference)
{
    if (actual.size() != reference.size())
    {
        throw std::invalid_argument("Vector sizes differ in max_abs_error.");
    }

    T result = 0;
    for (std::size_t i = 0; i < actual.size(); ++i)
    {
        result = std::max(result, std::abs(actual[i] - reference[i]));
    }
    return result;
}

template <typename T>
T max_relative_error(const std::vector<T>& actual, const std::vector<T>& reference)
{
    if (actual.size() != reference.size())
    {
        throw std::invalid_argument("Vector sizes differ in max_relative_error.");
    }

    const T tiny = static_cast<T>(100) * std::numeric_limits<T>::epsilon();
    T result = 0;

    for (std::size_t i = 0; i < actual.size(); ++i)
    {
        const T denominator = std::max(std::abs(reference[i]), tiny);
        result = std::max(result, std::abs(actual[i] - reference[i]) / denominator);
    }
    return result;
}

void build_csr_adaptive_row_blocks(
    int n,
    const int* row_ptr,
    std::vector<int>& row_blocks)
{
    row_blocks.clear();
    row_blocks.push_back(0);

    int start_row = 0;

    while (start_row < n)
    {
        const int row_nnz = row_ptr[start_row + 1] - row_ptr[start_row];

        // A long row is assigned to one block and handled by all block threads.
        if (row_nnz >= ADAPTIVE_NNZ_PER_BLOCK)
        {
            row_blocks.push_back(start_row + 1);
            ++start_row;
            continue;
        }

        // Pack several short rows into one row block with at most
        // ADAPTIVE_NNZ_PER_BLOCK nonzeros in total.
        int end_row = start_row + 1;
        int block_nnz = row_nnz;

        while (end_row < n)
        {
            const int next_row_nnz = row_ptr[end_row + 1] - row_ptr[end_row];

            if (next_row_nnz >= ADAPTIVE_NNZ_PER_BLOCK)
            {
                break;
            }
            if (block_nnz + next_row_nnz > ADAPTIVE_NNZ_PER_BLOCK)
            {
                break;
            }

            block_nnz += next_row_nnz;
            ++end_row;
        }

        row_blocks.push_back(end_row);
        start_row = end_row;
    }
}

// -----------------------------------------------------------------------------
// GPU kernels
// -----------------------------------------------------------------------------

template <typename T>
__global__ void spmv_csr_scalar_kernel(
    int n,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const T* __restrict__ values,
    const T* __restrict__ x,
    T* __restrict__ y)
{
    const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int row = global_thread; row < n; row += stride)
    {
        T sum = 0;
        for (int jj = row_ptr[row]; jj < row_ptr[row + 1]; ++jj)
        {
            sum += values[jj] * x[col_idx[jj]];
        }
        y[row] = sum;
    }
}

template <typename T>
__global__ void spmv_csr_vector_kernel(
    int n,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const T* __restrict__ values,
    const T* __restrict__ x,
    T* __restrict__ y)
{
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp_in_block = threadIdx.x / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int row = blockIdx.x * warps_per_block + warp_in_block;

    T sum = 0;

    // All lanes of a warp have the same row condition, and every lane reaches
    // the shuffle reduction. No block-wide barrier is needed.
    if (row < n)
    {
        const int row_begin = row_ptr[row];
        const int row_end = row_ptr[row + 1];

        for (int jj = row_begin + lane; jj < row_end; jj += WARP_SIZE)
        {
            sum += values[jj] * x[col_idx[jj]];
        }
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    {
        sum += __shfl_down_sync(FULL_WARP_MASK, sum, offset);
    }

    if (lane == 0 && row < n)
    {
        y[row] = sum;
    }
}

template <typename T>
__global__ void spmv_csr_adaptive_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const T* __restrict__ values,
    const T* __restrict__ x,
    const int* __restrict__ row_blocks,
    T* __restrict__ y)
{
    const int start_row = row_blocks[blockIdx.x];
    const int next_start_row = row_blocks[blockIdx.x + 1];
    const int num_rows = next_start_row - start_row;
    const int tid = threadIdx.x;

    __shared__ T shared_values[ADAPTIVE_NNZ_PER_BLOCK];

    if (num_rows > 1)
    {
        // CSR Stream path: stage all products belonging to this row block,
        // then let each thread sum one or more complete rows.
        const int first_nnz = row_ptr[start_row];
        const int last_nnz = row_ptr[next_start_row];
        const int block_nnz = last_nnz - first_nnz;

        for (int local = tid; local < block_nnz; local += blockDim.x)
        {
            const int global_j = first_nnz + local;
            shared_values[local] = values[global_j] * x[col_idx[global_j]];
        }

        __syncthreads();

        for (int row = start_row + tid; row < next_start_row; row += blockDim.x)
        {
            const int local_begin = row_ptr[row] - first_nnz;
            const int local_end = row_ptr[row + 1] - first_nnz;

            T sum = 0;
            for (int local = local_begin; local < local_end; ++local)
            {
                sum += shared_values[local];
            }
            y[row] = sum;
        }
    }
    else if (num_rows == 1)
    {
        // Long-row path: one entire CUDA block reduces one row.
        const int row = start_row;
        const int row_begin = row_ptr[row];
        const int row_end = row_ptr[row + 1];

        T sum = 0;
        for (int jj = row_begin + tid; jj < row_end; jj += blockDim.x)
        {
            sum += values[jj] * x[col_idx[jj]];
        }

        shared_values[tid] = sum;
        __syncthreads();

        // blockDim.x is validated to be a power of two and no larger than
        // ADAPTIVE_NNZ_PER_BLOCK.
        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
        {
            if (tid < offset)
            {
                shared_values[tid] += shared_values[tid + offset];
            }
            __syncthreads();
        }

        if (tid == 0)
        {
            y[row] = shared_values[0];
        }
    }
}

template <typename T>
__global__ void spmv_pcsr_product_kernel(
    int nnz,
    const T* __restrict__ values,
    const T* __restrict__ x,
    const int* __restrict__ col_idx,
    T* __restrict__ products)
{
    const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int jj = global_thread; jj < nnz; jj += stride)
    {
        products[jj] = values[jj] * x[col_idx[jj]];
    }
}

template <typename T>
__global__ void spmv_pcsr_sum_kernel(
    int n,
    const T* __restrict__ products,
    const int* __restrict__ row_ptr,
    T* __restrict__ y)
{
    const int tid = threadIdx.x;
    const int block_start_row = blockIdx.x * blockDim.x;
    const int row = block_start_row + tid;

    __shared__ int shared_ptr[PCSR_THREADS_PER_BLOCK + 1];
    __shared__ T shared_products[PCSR_SHARED_ELEMENTS];

    shared_ptr[tid] = (row < n) ? row_ptr[row] : row_ptr[n];

    if (tid == 0)
    {
        const int block_end_row = min(block_start_row + blockDim.x, n);
        shared_ptr[blockDim.x] = row_ptr[block_end_row];
    }

    __syncthreads();

    const int block_first_nnz = shared_ptr[0];
    const int block_last_nnz = shared_ptr[blockDim.x];

    T sum = 0;

    for (int chunk_start = block_first_nnz;
         chunk_start < block_last_nnz;
         chunk_start += PCSR_SHARED_ELEMENTS)
    {
        const int chunk_length =
            min(PCSR_SHARED_ELEMENTS, block_last_nnz - chunk_start);

        for (int local = tid; local < chunk_length; local += blockDim.x)
        {
            shared_products[local] = products[chunk_start + local];
        }

        __syncthreads();

        if (row < n)
        {
            const int row_begin = shared_ptr[tid];
            const int row_end = shared_ptr[tid + 1];
            const int sum_begin = max(row_begin, chunk_start);
            const int sum_end = min(row_end, chunk_start + chunk_length);

            for (int jj = sum_begin; jj < sum_end; ++jj)
            {
                sum += shared_products[jj - chunk_start];
            }
        }

        // Protect shared_products before the next chunk overwrites it.
        __syncthreads();
    }

    if (row < n)
    {
        y[row] = sum;
    }
}

template <typename T, int THREADS_PER_VECTOR>
__global__ void spmv_light_kernel(
    int* row_counter,
    int n,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const T* __restrict__ values,
    const T* __restrict__ x,
    T* __restrict__ y)
{
    static_assert(THREADS_PER_VECTOR >= 2, "THREADS_PER_VECTOR must be >= 2.");
    static_assert(THREADS_PER_VECTOR <= WARP_SIZE,
                  "THREADS_PER_VECTOR must be <= warp size.");
    static_assert((THREADS_PER_VECTOR & (THREADS_PER_VECTOR - 1)) == 0,
                  "THREADS_PER_VECTOR must be a power of two.");

    const int tid = threadIdx.x;
    const int warp_lane = tid & (WARP_SIZE - 1);
    const int lane_in_vector = tid & (THREADS_PER_VECTOR - 1);
    const int vector_in_warp = warp_lane / THREADS_PER_VECTOR;
    constexpr int vectors_per_warp = WARP_SIZE / THREADS_PER_VECTOR;

    while (true)
    {
        int base_row = 0;
        if (warp_lane == 0)
        {
            base_row = atomicAdd(row_counter, vectors_per_warp);
        }
        base_row = __shfl_sync(FULL_WARP_MASK, base_row, 0);

        // This condition is identical for all lanes in the warp, so the whole
        // warp exits together. For a partial final batch, invalid vectors stay
        // active and participate in all shuffle operations with zero values.
        if (base_row >= n)
        {
            break;
        }

        const int row = base_row + vector_in_warp;
        const bool valid_row = row < n;

        int row_begin = 0;
        int row_end = 0;

        if (valid_row && lane_in_vector == 0)
        {
            row_begin = row_ptr[row];
            row_end = row_ptr[row + 1];
        }

        const int vector_base_lane = warp_lane - lane_in_vector;
        row_begin = __shfl_sync(FULL_WARP_MASK, row_begin, vector_base_lane);
        row_end = __shfl_sync(FULL_WARP_MASK, row_end, vector_base_lane);

        T sum = 0;
        if (valid_row)
        {
            for (int jj = row_begin + lane_in_vector;
                 jj < row_end;
                 jj += THREADS_PER_VECTOR)
            {
                sum += values[jj] * x[col_idx[jj]];
            }
        }

        for (int offset = THREADS_PER_VECTOR / 2; offset > 0; offset >>= 1)
        {
            sum += __shfl_down_sync(
                FULL_WARP_MASK,
                sum,
                offset,
                THREADS_PER_VECTOR);
        }

        if (valid_row && lane_in_vector == 0)
        {
            y[row] = sum;
        }
    }
}

// -----------------------------------------------------------------------------
// RAII GPU context: allocation and preprocessing happen only once.
// -----------------------------------------------------------------------------

class SpmvGpuContext
{
public:
    SpmvGpuContext(
        int n,
        int nnz,
        const std::vector<int>& row_ptr,
        const std::vector<int>& col_idx,
        const std::vector<double>& values,
        const std::vector<double>& x,
        int block_size)
        : n_(n), nnz_(nnz), block_size_(block_size)
    {
        if (n_ <= 0 || nnz_ <= 0)
        {
            throw std::invalid_argument("n and nnz must be positive.");
        }
        if (block_size_ <= 0 || block_size_ > 1024)
        {
            throw std::invalid_argument("block_size must be in [1, 1024].");
        }
        if (block_size_ % WARP_SIZE != 0)
        {
            throw std::invalid_argument("block_size must be a multiple of 32.");
        }
        if (!is_power_of_two(block_size_))
        {
            throw std::invalid_argument(
                "block_size must be a power of two for Adaptive reduction.");
        }
        if (block_size_ > ADAPTIVE_NNZ_PER_BLOCK)
        {
            throw std::invalid_argument(
                "block_size cannot exceed ADAPTIVE_NNZ_PER_BLOCK.");
        }

        build_csr_adaptive_row_blocks(n_, row_ptr.data(), host_row_blocks_);
        num_adaptive_blocks_ = static_cast<int>(host_row_blocks_.size()) - 1;

        const double average_nnz = static_cast<double>(nnz_) / n_;
        if (average_nnz <= 4.0)
        {
            light_threads_per_vector_ = 2;
        }
        else if (average_nnz <= 8.0)
        {
            light_threads_per_vector_ = 4;
        }
        else if (average_nnz <= 16.0)
        {
            light_threads_per_vector_ = 8;
        }
        else if (average_nnz <= 32.0)
        {
            light_threads_per_vector_ = 16;
        }
        else
        {
            light_threads_per_vector_ = 32;
        }

        cudaDeviceProp property{};
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&property, device));
        multiprocessor_count_ = property.multiProcessorCount;

        scalar_grid_size_ = ceil_div(n_, block_size_);
        vector_grid_size_ = ceil_div(n_, block_size_ / WARP_SIZE);
        pcsr_product_grid_size_ = ceil_div(nnz_, block_size_);
        pcsr_sum_grid_size_ = ceil_div(n_, PCSR_THREADS_PER_BLOCK);

        // LightSpMV uses persistent warps and dynamic row allocation. Launching
        // n/blockSize blocks, as in the original code, leaves little work for
        // dynamic scheduling. Four blocks per SM is a practical starting point.
        const int vectors_per_block = block_size_ / light_threads_per_vector_;
        const int blocks_needed_for_one_batch = ceil_div(n_, vectors_per_block);
        light_grid_size_ = std::max(
            1,
            std::min(blocks_needed_for_one_batch, 4 * multiprocessor_count_));

        try
        {
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_row_ptr_),
                static_cast<std::size_t>(n_ + 1) * sizeof(int)));
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_col_idx_),
                static_cast<std::size_t>(nnz_) * sizeof(int)));
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_values_),
                static_cast<std::size_t>(nnz_) * sizeof(double)));
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_x_),
                static_cast<std::size_t>(n_) * sizeof(double)));
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_y_),
                static_cast<std::size_t>(n_) * sizeof(double)));
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_products_),
                static_cast<std::size_t>(nnz_) * sizeof(double)));
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_row_blocks_),
                host_row_blocks_.size() * sizeof(int)));
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_row_counter_),
                sizeof(int)));

            CUDA_CHECK(cudaMemcpy(
                d_row_ptr_,
                row_ptr.data(),
                static_cast<std::size_t>(n_ + 1) * sizeof(int),
                cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(
                d_col_idx_,
                col_idx.data(),
                static_cast<std::size_t>(nnz_) * sizeof(int),
                cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(
                d_values_,
                values.data(),
                static_cast<std::size_t>(nnz_) * sizeof(double),
                cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(
                d_x_,
                x.data(),
                static_cast<std::size_t>(n_) * sizeof(double),
                cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(
                d_row_blocks_,
                host_row_blocks_.data(),
                host_row_blocks_.size() * sizeof(int),
                cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(
                d_y_,
                0,
                static_cast<std::size_t>(n_) * sizeof(double)));
        }
        catch (...)
        {
            release_noexcept();
            throw;
        }
    }

    ~SpmvGpuContext()
    {
        release_noexcept();
    }

    SpmvGpuContext(const SpmvGpuContext&) = delete;
    SpmvGpuContext& operator=(const SpmvGpuContext&) = delete;

    void run(int kernel_type, cudaStream_t stream = nullptr)
    {
        switch (kernel_type)
        {
            case 1:
                spmv_csr_scalar_kernel<double>
                    <<<scalar_grid_size_, block_size_, 0, stream>>>(
                        n_, d_row_ptr_, d_col_idx_, d_values_, d_x_, d_y_);
                break;

            case 2:
                spmv_csr_vector_kernel<double>
                    <<<vector_grid_size_, block_size_, 0, stream>>>(
                        n_, d_row_ptr_, d_col_idx_, d_values_, d_x_, d_y_);
                break;

            case 3:
                spmv_csr_adaptive_kernel<double>
                    <<<num_adaptive_blocks_, block_size_, 0, stream>>>(
                        d_row_ptr_,
                        d_col_idx_,
                        d_values_,
                        d_x_,
                        d_row_blocks_,
                        d_y_);
                break;

            case 4:
                spmv_pcsr_product_kernel<double>
                    <<<pcsr_product_grid_size_, block_size_, 0, stream>>>(
                        nnz_, d_values_, d_x_, d_col_idx_, d_products_);
                CUDA_CHECK(cudaGetLastError());

                spmv_pcsr_sum_kernel<double>
                    <<<pcsr_sum_grid_size_, PCSR_THREADS_PER_BLOCK, 0, stream>>>(
                        n_, d_products_, d_row_ptr_, d_y_);
                break;

            case 5:
                // Resetting the dynamic row counter is required for every SpMV,
                // so it is intentionally part of the timed operation.
                CUDA_CHECK(cudaMemsetAsync(d_row_counter_, 0, sizeof(int), stream));
                launch_light_kernel(stream);
                break;

            default:
                throw std::invalid_argument("kernel_type must be in [1, 5].");
        }

        CUDA_CHECK(cudaGetLastError());
    }

    void copy_result(std::vector<double>& result) const
    {
        result.resize(n_);
        CUDA_CHECK(cudaMemcpy(
            result.data(),
            d_y_,
            static_cast<std::size_t>(n_) * sizeof(double),
            cudaMemcpyDeviceToHost));
    }

    int num_adaptive_blocks() const
    {
        return num_adaptive_blocks_;
    }

    int light_threads_per_vector() const
    {
        return light_threads_per_vector_;
    }

    int light_grid_size() const
    {
        return light_grid_size_;
    }

    int multiprocessor_count() const
    {
        return multiprocessor_count_;
    }

private:
    void launch_light_kernel(cudaStream_t stream)
    {
        switch (light_threads_per_vector_)
        {
            case 2:
                spmv_light_kernel<double, 2>
                    <<<light_grid_size_, block_size_, 0, stream>>>(
                        d_row_counter_, n_, d_row_ptr_, d_col_idx_, d_values_, d_x_, d_y_);
                break;
            case 4:
                spmv_light_kernel<double, 4>
                    <<<light_grid_size_, block_size_, 0, stream>>>(
                        d_row_counter_, n_, d_row_ptr_, d_col_idx_, d_values_, d_x_, d_y_);
                break;
            case 8:
                spmv_light_kernel<double, 8>
                    <<<light_grid_size_, block_size_, 0, stream>>>(
                        d_row_counter_, n_, d_row_ptr_, d_col_idx_, d_values_, d_x_, d_y_);
                break;
            case 16:
                spmv_light_kernel<double, 16>
                    <<<light_grid_size_, block_size_, 0, stream>>>(
                        d_row_counter_, n_, d_row_ptr_, d_col_idx_, d_values_, d_x_, d_y_);
                break;
            case 32:
                spmv_light_kernel<double, 32>
                    <<<light_grid_size_, block_size_, 0, stream>>>(
                        d_row_counter_, n_, d_row_ptr_, d_col_idx_, d_values_, d_x_, d_y_);
                break;
            default:
                throw std::logic_error("Invalid LightSpMV vector width.");
        }
    }

    void release_noexcept() noexcept
    {
        // Ignore cleanup errors in a destructor. Earlier CUDA errors are reported
        // at the point where they occur.
        if (d_row_counter_ != nullptr) cudaFree(d_row_counter_);
        if (d_row_blocks_ != nullptr) cudaFree(d_row_blocks_);
        if (d_products_ != nullptr) cudaFree(d_products_);
        if (d_y_ != nullptr) cudaFree(d_y_);
        if (d_x_ != nullptr) cudaFree(d_x_);
        if (d_values_ != nullptr) cudaFree(d_values_);
        if (d_col_idx_ != nullptr) cudaFree(d_col_idx_);
        if (d_row_ptr_ != nullptr) cudaFree(d_row_ptr_);

        d_row_counter_ = nullptr;
        d_row_blocks_ = nullptr;
        d_products_ = nullptr;
        d_y_ = nullptr;
        d_x_ = nullptr;
        d_values_ = nullptr;
        d_col_idx_ = nullptr;
        d_row_ptr_ = nullptr;
    }

    int n_ = 0;
    int nnz_ = 0;
    int block_size_ = 0;

    int scalar_grid_size_ = 0;
    int vector_grid_size_ = 0;
    int num_adaptive_blocks_ = 0;
    int pcsr_product_grid_size_ = 0;
    int pcsr_sum_grid_size_ = 0;
    int light_grid_size_ = 0;
    int light_threads_per_vector_ = 0;
    int multiprocessor_count_ = 0;

    std::vector<int> host_row_blocks_;

    int* d_row_ptr_ = nullptr;
    int* d_col_idx_ = nullptr;
    double* d_values_ = nullptr;
    double* d_x_ = nullptr;
    double* d_y_ = nullptr;
    double* d_products_ = nullptr;
    int* d_row_blocks_ = nullptr;
    int* d_row_counter_ = nullptr;
};

class CudaEvent
{
public:
    CudaEvent()
    {
        CUDA_CHECK(cudaEventCreate(&event_));
    }

    ~CudaEvent()
    {
        if (event_ != nullptr)
        {
            cudaEventDestroy(event_);
        }
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    operator cudaEvent_t() const
    {
        return event_;
    }

private:
    cudaEvent_t event_ = nullptr;
};

struct BenchmarkResult
{
    int kernel_type = 0;
    double average_ms = 0;
    double gflops = 0;
    double approximate_gbs = 0;
    double max_abs_error = 0;
    double max_relative_error = 0;
};

BenchmarkResult benchmark_kernel(
    SpmvGpuContext& gpu,
    int kernel_type,
    int n,
    int nnz,
    int warmup,
    int repeats,
    const std::vector<double>& reference)
{
    if (warmup < 0)
    {
        throw std::invalid_argument("warmup must be non-negative.");
    }
    if (repeats <= 0)
    {
        throw std::invalid_argument("repeats must be positive.");
    }

    for (int iteration = 0; iteration < warmup; ++iteration)
    {
        gpu.run(kernel_type);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaEvent start;
    CudaEvent stop;

    CUDA_CHECK(cudaEventRecord(start));
    for (int iteration = 0; iteration < repeats; ++iteration)
    {
        gpu.run(kernel_type);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms_float = 0;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms_float, start, stop));

    std::vector<double> gpu_result;
    gpu.copy_result(gpu_result);

    const double average_ms = static_cast<double>(total_ms_float) / repeats;
    const double average_seconds = average_ms * 1.0e-3;

    // One multiply and one add per nonzero.
    const double gflops =
        (2.0 * static_cast<double>(nnz)) / average_seconds / 1.0e9;

    // Approximate minimum traffic. Cache reuse is deliberately ignored.
    double bytes =
        static_cast<double>(nnz)
            * (sizeof(double) + sizeof(int) + sizeof(double))
        + static_cast<double>(n + 1) * sizeof(int)
        + static_cast<double>(n) * sizeof(double);

    // PCSR additionally writes and then reads one product per nonzero.
    if (kernel_type == 4)
    {
        bytes += 2.0 * static_cast<double>(nnz) * sizeof(double);
    }

    BenchmarkResult result;
    result.kernel_type = kernel_type;
    result.average_ms = average_ms;
    result.gflops = gflops;
    result.approximate_gbs = bytes / average_seconds / 1.0e9;
    result.max_abs_error = max_abs_error(gpu_result, reference);
    result.max_relative_error = max_relative_error(gpu_result, reference);
    return result;
}

void print_usage(const char* program)
{
    std::cout
        << "Usage:\n  " << program
        << " [n] [repeats] [warmup] [matrix_mode] [block_size]\n\n"
        << "Arguments:\n"
        << "  n            Number of rows and columns. Default: 2000000\n"
        << "  repeats      Timed repetitions per algorithm. Default: 10\n"
        << "  warmup       Warmup repetitions per algorithm. Default: 2\n"
        << "  matrix_mode  0 = original 3..128 pattern;\n"
        << "               1 = heavy-tail pattern with 1024/4096 rows. Default: 1\n"
        << "  block_size   CUDA threads per block; power of two and multiple of 32.\n"
        << "               Default: 256\n\n"
        << "Examples:\n"
        << "  " << program << " 200000 20 3 1 256\n"
        << "  " << program << " 200000 1 0 1 256   # suitable for NCU\n";
}

int parse_int_argument(const char* text, const char* name)
{
    try
    {
        std::size_t parsed = 0;
        const long long value = std::stoll(text, &parsed);
        if (parsed != std::string(text).size())
        {
            throw std::invalid_argument("trailing characters");
        }
        if (value < std::numeric_limits<int>::min()
            || value > std::numeric_limits<int>::max())
        {
            throw std::out_of_range("outside int range");
        }
        return static_cast<int>(value);
    }
    catch (const std::exception& e)
    {
        std::ostringstream oss;
        oss << "Invalid " << name << " argument '" << text << "': " << e.what();
        throw std::invalid_argument(oss.str());
    }
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc >= 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help"))
        {
            print_usage(argv[0]);
            return 0;
        }

        int n = 2'000'000;
        int repeats = 10;
        int warmup = 2;
        int matrix_mode = 1;
        int block_size = 256;

        if (argc >= 2) n = parse_int_argument(argv[1], "n");
        if (argc >= 3) repeats = parse_int_argument(argv[2], "repeats");
        if (argc >= 4) warmup = parse_int_argument(argv[3], "warmup");
        if (argc >= 5) matrix_mode = parse_int_argument(argv[4], "matrix_mode");
        if (argc >= 6) block_size = parse_int_argument(argv[5], "block_size");
        if (argc > 6)
        {
            print_usage(argv[0]);
            throw std::invalid_argument("Too many command-line arguments.");
        }

        if (n <= 0) throw std::invalid_argument("n must be positive.");
        if (repeats <= 0) throw std::invalid_argument("repeats must be positive.");
        if (warmup < 0) throw std::invalid_argument("warmup must be non-negative.");
        if (matrix_mode != 0 && matrix_mode != 1)
        {
            throw std::invalid_argument("matrix_mode must be 0 or 1.");
        }

        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count == 0)
        {
            std::cerr << "No CUDA device found.\n";
            return 1;
        }

        CUDA_CHECK(cudaSetDevice(0));

        cudaDeviceProp property{};
        CUDA_CHECK(cudaGetDeviceProperties(&property, 0));

        std::cout << "Building irregular CSR matrix...\n";
        const auto build_start = std::chrono::steady_clock::now();

        std::vector<int> row_ptr;
        std::vector<int> col_idx;
        std::vector<double> values;
        build_irregular_csr_matrix<double>(
            n, matrix_mode, row_ptr, col_idx, values);

        const int nnz = static_cast<int>(values.size());

        std::vector<double> x(n);
        for (int i = 0; i < n; ++i)
        {
            x[i] =
                std::sin(0.001 * static_cast<double>(i))
                + 0.5 * std::cos(0.003 * static_cast<double>(i))
                + 0.25;
        }

        const auto build_stop = std::chrono::steady_clock::now();

        std::cout << "Computing CPU reference...\n";
        const auto cpu_start = std::chrono::steady_clock::now();
        std::vector<double> reference;
        spmv_cpu_reference(n, row_ptr, col_idx, values, x, reference);
        const auto cpu_stop = std::chrono::steady_clock::now();

        SpmvGpuContext gpu(
            n, nnz, row_ptr, col_idx, values, x, block_size);

        const double matrix_build_seconds =
            std::chrono::duration<double>(build_stop - build_start).count();
        const double cpu_reference_seconds =
            std::chrono::duration<double>(cpu_stop - cpu_start).count();

        std::cout << "\n------------------------------------------------------------\n";
        std::cout << "Device and matrix information\n";
        std::cout << "GPU                    = " << property.name << '\n';
        std::cout << "Compute capability     = " << property.major << '.' << property.minor << '\n';
        std::cout << "SM count               = " << gpu.multiprocessor_count() << '\n';
        std::cout << "n                      = " << n << '\n';
        std::cout << "nnz                    = " << nnz << '\n';
        std::cout << "average nnz per row    = "
                  << static_cast<double>(nnz) / n << '\n';
        std::cout << "matrix mode            = " << matrix_mode
                  << (matrix_mode == 0 ? " (original)" : " (heavy-tail)") << '\n';
        std::cout << "block size             = " << block_size << '\n';
        std::cout << "Adaptive row blocks    = " << gpu.num_adaptive_blocks() << '\n';
        std::cout << "Light threads/vector   = " << gpu.light_threads_per_vector() << '\n';
        std::cout << "Light grid size        = " << gpu.light_grid_size() << '\n';
        std::cout << "repeats                = " << repeats << '\n';
        std::cout << "warmup                 = " << warmup << '\n';
        std::cout << "matrix build time (s)  = " << matrix_build_seconds << '\n';
        std::cout << "CPU reference time (s) = " << cpu_reference_seconds << '\n';
        std::cout << "------------------------------------------------------------\n\n";

        std::cout << std::left
                  << std::setw(16) << "kernel"
                  << std::right
                  << std::setw(14) << "avg_ms"
                  << std::setw(14) << "GFLOP/s"
                  << std::setw(16) << "approx_GB/s"
                  << std::setw(16) << "max_abs_err"
                  << std::setw(16) << "max_rel_err"
                  << '\n';
        std::cout << std::string(92, '-') << '\n';

        std::vector<BenchmarkResult> results;
        results.reserve(5);

        for (int kernel_type = 1; kernel_type <= 5; ++kernel_type)
        {
            const BenchmarkResult result = benchmark_kernel(
                gpu,
                kernel_type,
                n,
                nnz,
                warmup,
                repeats,
                reference);

            results.push_back(result);

            std::cout << std::left
                      << std::setw(16) << kernel_name(kernel_type)
                      << std::right << std::scientific << std::setprecision(6)
                      << std::setw(14) << result.average_ms
                      << std::setw(14) << result.gflops
                      << std::setw(16) << result.approximate_gbs
                      << std::setw(16) << result.max_abs_error
                      << std::setw(16) << result.max_relative_error
                      << '\n';
        }

        std::cout << std::string(92, '-') << '\n';

        const auto best = std::min_element(
            results.begin(),
            results.end(),
            [](const BenchmarkResult& lhs, const BenchmarkResult& rhs)
            {
                return lhs.average_ms < rhs.average_ms;
            });

        if (best != results.end())
        {
            std::cout << "Fastest kernel: " << kernel_name(best->kernel_type)
                      << ", average time = " << best->average_ms << " ms\n";
        }

        constexpr double tolerance = 1.0e-10;
        bool all_correct = true;
        for (const BenchmarkResult& result : results)
        {
            if (result.max_abs_error > tolerance
                && result.max_relative_error > tolerance)
            {
                all_correct = false;
                std::cerr << "Warning: " << kernel_name(result.kernel_type)
                          << " exceeds correctness tolerance.\n";
            }
        }

        std::cout << "Correctness check: "
                  << (all_correct ? "PASS" : "FAIL")
                  << " (tolerance = " << tolerance << ")\n";

        CUDA_CHECK(cudaDeviceSynchronize());
        return all_correct ? 0 : 2;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
