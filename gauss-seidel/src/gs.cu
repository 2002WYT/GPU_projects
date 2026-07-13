#include "gs.cuh"

#include <cub/device/device_reduce.cuh>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>
#include <vector>

static void validate_csr(const CSRMatrix& A)
{
    if (A.n <= 0) {
        throw std::invalid_argument("A.n must be positive.");
    }
    if (A.row_ptr.size() != static_cast<std::size_t>(A.n + 1)) {
        throw std::invalid_argument("row_ptr.size() must equal n + 1.");
    }
    if (A.row_ptr.front() != 0) {
        throw std::invalid_argument("row_ptr[0] must be zero.");
    }
    if (A.col_idx.size() != A.values.size()) {
        throw std::invalid_argument("col_idx and values must have equal sizes.");
    }
    if (A.row_ptr.back() != static_cast<int>(A.values.size())) {
        throw std::invalid_argument("row_ptr[n] must equal nnz.");
    }

    for (int i = 0; i < A.n; ++i) {
        if (A.row_ptr[i] > A.row_ptr[i + 1]) {
            throw std::invalid_argument("row_ptr must be nondecreasing.");
        }
        for (int jj = A.row_ptr[i]; jj < A.row_ptr[i + 1]; ++jj) {
            const int col = A.col_idx[jj];
            if (col < 0 || col >= A.n) {
                throw std::invalid_argument("CSR column index is out of range.");
            }
            if (!std::isfinite(A.values[jj])) {
                throw std::invalid_argument("Matrix contains a non-finite value.");
            }
        }
    }
}

// Merge duplicate diagonal entries and remove the diagonal from the SpMV-like
// inner loop. Duplicate off-diagonal entries are allowed and remain valid.
static PreparedMatrix prepare_matrix(const CSRMatrix& A)
{
    validate_csr(A);

    PreparedMatrix P;
    P.n = A.n;
    P.off_row_ptr.resize(A.n + 1, 0);
    P.diagonal_inverse.resize(A.n);
    P.off_col_idx.reserve(A.col_idx.size());
    P.off_values.reserve(A.values.size());

    for (int i = 0; i < A.n; ++i) {
        double diagonal = 0.0;

        for (int jj = A.row_ptr[i]; jj < A.row_ptr[i + 1]; ++jj) {
            const int col = A.col_idx[jj];
            const double value = A.values[jj];

            if (col == i) {
                diagonal += value;
            } else {
                P.off_col_idx.push_back(col);
                P.off_values.push_back(value);
            }
        }

        if (!std::isfinite(diagonal) || std::abs(diagonal) <= 1.0e-30) {
            throw std::invalid_argument(
                "Every row must contain a finite, nonzero diagonal entry. "
                "Bad row: " + std::to_string(i));
        }

        P.diagonal_inverse[i] = 1.0 / diagonal;
        P.off_row_ptr[i + 1] = static_cast<int>(P.off_col_idx.size());
    }
    return P;
}

struct Coloring {
    int num_colors = 0;
    std::vector<int> rows_by_color;
    std::vector<int> color_offsets;
};

// Build an undirected conflict graph. Rows i and j conflict whenever A(i,j)
// or A(j,i) is nonzero. Therefore rows in one color can be updated concurrently.
static Coloring greedy_symmetric_coloring(const PreparedMatrix& A)
{
    const int n = A.n;
    std::vector<std::vector<int>> adjacency(n);

    for (int row = 0; row < n; ++row) {
        for (int jj = A.off_row_ptr[row]; jj < A.off_row_ptr[row + 1]; ++jj) {
            const int col = A.off_col_idx[jj];
            adjacency[row].push_back(col);
            adjacency[col].push_back(row);
        }
    }

    for (auto& neighbors : adjacency) {
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()),
                        neighbors.end());
    }

    // Largest-degree-first ordering usually reduces the number of colors.
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int lhs, int rhs) {
        return adjacency[lhs].size() > adjacency[rhs].size();
    });

    std::vector<int> row_color(n, -1);
    std::vector<int> forbidden(n, -1);
    int num_colors = 0;

    for (int position = 0; position < n; ++position) {
        const int row = order[position];

        for (const int neighbor : adjacency[row]) {
            const int color = row_color[neighbor];
            if (color >= 0) {
                forbidden[color] = row;
            }
        }

        int color = 0;
        while (color < num_colors && forbidden[color] == row) {
            ++color;
        }
        if (color == num_colors) {
            ++num_colors;
        }
        row_color[row] = color;
    }

    Coloring result;
    result.num_colors = num_colors;
    result.color_offsets.assign(num_colors + 1, 0);

    for (const int color : row_color) {
        ++result.color_offsets[color + 1];
    }
    std::partial_sum(result.color_offsets.begin(), result.color_offsets.end(),
                     result.color_offsets.begin());

    result.rows_by_color.resize(n);
    std::vector<int> next = result.color_offsets;
    for (int row = 0; row < n; ++row) {
        const int color = row_color[row];
        result.rows_by_color[next[color]++] = row;
    }

    return result;
}

template <class T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) { allocate(count); }

    ~DeviceBuffer() { cudaFree(pointer_); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : pointer_(other.pointer_), count_(other.count_)
    {
        other.pointer_ = nullptr;
        other.count_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        if (this != &other) {
            cudaFree(pointer_);
            pointer_ = other.pointer_;
            count_ = other.count_;
            other.pointer_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    void allocate(std::size_t count)
    {
        cudaFree(pointer_);
        pointer_ = nullptr;
        count_ = count;
        if (count > 0) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer_),
                                  count * sizeof(T)));
        }
    }

    T* get() { return pointer_; }
    const T* get() const { return pointer_; }
    std::size_t size() const { return count_; }

private:
    T* pointer_ = nullptr;
    std::size_t count_ = 0;
};

__global__ void colored_gauss_seidel_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const double* __restrict__ values,
    const double* __restrict__ diagonal_inverse,
    const double* __restrict__ b,
    double* __restrict__ x,
    const int* __restrict__ rows_by_color,
    int color_begin,
    int color_end,
    double omega)
{
    for (int k = color_begin + blockIdx.x * blockDim.x + threadIdx.x;
         k < color_end;
         k += blockDim.x * gridDim.x) {
        const int row = rows_by_color[k];
        double rhs = b[row];

        for (int jj = row_ptr[row]; jj < row_ptr[row + 1]; ++jj) {
            rhs -= values[jj] * x[col_idx[jj]];
        }

        const double gs_value = rhs * diagonal_inverse[row];
        x[row] += omega * (gs_value - x[row]);
    }
}

__global__ void colored_gauss_seidel_warp_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const double* __restrict__ values,
    const double* __restrict__ diagonal_inverse,
    const double* __restrict__ b,
    double* __restrict__ x,
    const int* __restrict__ rows_by_color,
    int color_begin,
    int color_end,
    double omega)
{
    constexpr unsigned full_mask = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_warp = global_thread >> 5;
    const int total_warps = (gridDim.x * blockDim.x) >> 5;

    for (int k = color_begin + global_warp;
         k < color_end;
         k += total_warps) {
        const int row = rows_by_color[k];
        double off_diagonal_sum = 0.0;

        for (int jj = row_ptr[row] + lane;
             jj < row_ptr[row + 1];
             jj += 32) {
            off_diagonal_sum += values[jj] * x[col_idx[jj]];
        }

        for (int offset = 16; offset > 0; offset >>= 1) {
            off_diagonal_sum +=
                __shfl_down_sync(full_mask, off_diagonal_sum, offset);
        }

        if (lane == 0) {
            const double gs_value =
                (b[row] - off_diagonal_sum) * diagonal_inverse[row];
            x[row] += omega * (gs_value - x[row]);
        }
    }
}

__global__ void square_vector_kernel(const double* __restrict__ x,
                                     double* __restrict__ terms,
                                     int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        const double value = x[i];
        terms[i] = value * value;
    }
}

__global__ void residual_square_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const double* __restrict__ off_values,
    const double* __restrict__ diagonal_inverse,
    const double* __restrict__ b,
    const double* __restrict__ x,
    double* __restrict__ terms,
    int n)
{
    for (int row = blockIdx.x * blockDim.x + threadIdx.x;
         row < n;
         row += blockDim.x * gridDim.x) {
        double Ax = x[row] / diagonal_inverse[row];
        for (int jj = row_ptr[row]; jj < row_ptr[row + 1]; ++jj) {
            Ax += off_values[jj] * x[col_idx[jj]];
        }
        const double residual = b[row] - Ax;
        terms[row] = residual * residual;
    }
}

static int launch_grid_size(int work_items, int block_size)
{
    constexpr int max_blocks = 65535;
    const int blocks = (work_items + block_size - 1) / block_size;
    return std::max(1, std::min(blocks, max_blocks));
}

SolverResult solve_multicolor_gauss_seidel_gpu(
    const CSRMatrix& A,
    const std::vector<double>& b,
    std::vector<double>& x,
    const SolverOptions& options = {})
{
    if (static_cast<int>(b.size()) != A.n) {
        throw std::invalid_argument("b.size() must equal A.n.");
    }
    if (x.empty()) {
        x.assign(A.n, 0.0);
    }
    if (static_cast<int>(x.size()) != A.n) {
        throw std::invalid_argument("x.size() must equal A.n.");
    }
    if (options.max_iterations <= 0 || options.residual_check_interval <= 0) {
        throw std::invalid_argument("Iteration counts must be positive.");
    }
    if (!(options.relative_tolerance > 0.0)) {
        throw std::invalid_argument("relative_tolerance must be positive.");
    }
    if (!(options.omega > 0.0 && options.omega < 2.0)) {
        throw std::invalid_argument("omega must satisfy 0 < omega < 2.");
    }
    if (options.block_size <= 0 || options.block_size > 1024) {
        throw std::invalid_argument("block_size must be in [1, 1024].");
    }
    if (options.warp_row_threshold <= 0) {
        throw std::invalid_argument("warp_row_threshold must be positive.");
    }

    const PreparedMatrix P = prepare_matrix(A);
    const Coloring coloring = greedy_symmetric_coloring(P);
    const int n = P.n;
    const int nnz_off = static_cast<int>(P.off_values.size());
    const double average_off_diagonal_nnz =
        static_cast<double>(nnz_off) / static_cast<double>(n);
    const bool use_warp_per_row =
        average_off_diagonal_nnz >= options.warp_row_threshold;
    if (use_warp_per_row && options.block_size % 32 != 0) {
        throw std::invalid_argument(
            "block_size must be a multiple of 32 for the warp-per-row kernel.");
    }

    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;

    try {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));

        DeviceBuffer<int> d_row_ptr(n + 1);
        DeviceBuffer<int> d_col_idx(nnz_off);
        DeviceBuffer<double> d_values(nnz_off);
        DeviceBuffer<double> d_diagonal_inverse(n);
        DeviceBuffer<double> d_b(n);
        DeviceBuffer<double> d_x(n);
        DeviceBuffer<int> d_rows_by_color(n);
        DeviceBuffer<double> d_terms(n);
        DeviceBuffer<double> d_sum(1);

        CUDA_CHECK(cudaMemcpyAsync(d_row_ptr.get(), P.off_row_ptr.data(),
                                   (n + 1) * sizeof(int),
                                   cudaMemcpyHostToDevice, stream));
        if (nnz_off > 0) {
            CUDA_CHECK(cudaMemcpyAsync(d_col_idx.get(), P.off_col_idx.data(),
                                       nnz_off * sizeof(int),
                                       cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d_values.get(), P.off_values.data(),
                                       nnz_off * sizeof(double),
                                       cudaMemcpyHostToDevice, stream));
        }
        CUDA_CHECK(cudaMemcpyAsync(d_diagonal_inverse.get(),
                                   P.diagonal_inverse.data(),
                                   n * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_b.get(), b.data(), n * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_x.get(), x.data(), n * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_rows_by_color.get(),
                                   coloring.rows_by_color.data(),
                                   n * sizeof(int),
                                   cudaMemcpyHostToDevice, stream));

        const int vector_grid = launch_grid_size(n, options.block_size);

        std::size_t reduce_temp_bytes = 0;
        CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, reduce_temp_bytes,
                                          d_terms.get(), d_sum.get(), n, stream));
        DeviceBuffer<unsigned char> d_reduce_temp(reduce_temp_bytes);

        square_vector_kernel<<<vector_grid, options.block_size, 0, stream>>>(
            d_b.get(), d_terms.get(), n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cub::DeviceReduce::Sum(d_reduce_temp.get(), reduce_temp_bytes,
                                          d_terms.get(), d_sum.get(), n, stream));

        double b_norm_squared = 0.0;
        CUDA_CHECK(cudaMemcpyAsync(&b_norm_squared, d_sum.get(), sizeof(double),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const double residual_denominator =
            std::sqrt(std::max(b_norm_squared, 0.0));

        // Capture a complete multicolor sweep once. Replaying the CUDA Graph
        // avoids paying one normal CPU launch overhead per color per iteration.
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        for (int color = 0; color < coloring.num_colors; ++color) {
            const int begin = coloring.color_offsets[color];
            const int end = coloring.color_offsets[color + 1];
            const int count = end - begin;
            if (count == 0) {
                continue;
            }
            if (use_warp_per_row) {
                const int warps_per_block = options.block_size / 32;
                const int grid = launch_grid_size(
                    count, warps_per_block);
                colored_gauss_seidel_warp_kernel
                    <<<grid, options.block_size, 0, stream>>>(
                        d_row_ptr.get(), d_col_idx.get(), d_values.get(),
                        d_diagonal_inverse.get(), d_b.get(), d_x.get(),
                        d_rows_by_color.get(), begin, end, options.omega);
            } else {
                const int grid = launch_grid_size(count, options.block_size);
                colored_gauss_seidel_kernel
                    <<<grid, options.block_size, 0, stream>>>(
                        d_row_ptr.get(), d_col_idx.get(), d_values.get(),
                        d_diagonal_inverse.get(), d_b.get(), d_x.get(),
                        d_rows_by_color.get(), begin, end, options.omega);
            }
        }
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

        SolverResult result;
        result.num_colors = coloring.num_colors;

        CUDA_CHECK(cudaEventRecord(start_event, stream));

        for (int iteration = 1; iteration <= options.max_iterations; ++iteration) {
            CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));

            const bool should_check =
                (iteration % options.residual_check_interval == 0) ||
                (iteration == options.max_iterations);

            if (!should_check) {
                continue;
            }

            residual_square_kernel<<<vector_grid, options.block_size, 0, stream>>>(
                d_row_ptr.get(), d_col_idx.get(), d_values.get(),
                d_diagonal_inverse.get(), d_b.get(), d_x.get(),
                d_terms.get(), n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cub::DeviceReduce::Sum(d_reduce_temp.get(),
                                              reduce_temp_bytes,
                                              d_terms.get(), d_sum.get(), n,
                                              stream));

            double residual_squared = 0.0;
            CUDA_CHECK(cudaMemcpyAsync(&residual_squared, d_sum.get(),
                                       sizeof(double), cudaMemcpyDeviceToHost,
                                       stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            const double residual_norm =
                std::sqrt(std::max(residual_squared, 0.0));
            result.relative_residual =
                (residual_denominator > 0.0)
                    ? residual_norm / residual_denominator
                    : residual_norm;
            result.iterations = iteration;

            if (result.relative_residual <= options.relative_tolerance) {
                break;
            }
        }

        CUDA_CHECK(cudaEventRecord(stop_event, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        CUDA_CHECK(cudaEventElapsedTime(&result.elapsed_ms,
                                        start_event, stop_event));

        CUDA_CHECK(cudaMemcpyAsync(x.data(), d_x.get(), n * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        cudaGraphExecDestroy(graph_exec);
        cudaGraphDestroy(graph);
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        cudaStreamDestroy(stream);
        return result;
    } catch (...) {
        if (graph_exec) cudaGraphExecDestroy(graph_exec);
        if (graph) cudaGraphDestroy(graph);
        if (start_event) cudaEventDestroy(start_event);
        if (stop_event) cudaEventDestroy(stop_event);
        if (stream) cudaStreamDestroy(stream);
        throw;
    }
}
