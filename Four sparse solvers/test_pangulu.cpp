/*
 * test_pangulu.cpp
 *
 * PanguLU 5.0.0 MPI/GPU direct-solver test.
 *
 * Input:
 *   mpirun -np 1 ./test_pangu
 *   mpirun -np 4 ./test_pangu matrix.mtx
 *   mpirun -np 4 ./test_pangu --poisson 256
 *
 * Optional final argument:
 *   block size nb
 *
 * Examples:
 *   mpirun -np 4 ./test_pangu matrix.mtx 64
 *   mpirun -np 4 ./test_pangu --poisson 256 64
 *
 * The manufactured exact solution is x*=1 and b=A*x*.
 * mtx_reader.h supplies a full zero-based CSR matrix. PanguLU's public
 * interface expects complete zero-based CSC arrays on rank 0, so this
 * program converts CSR to CSC before pangulu_init().
 */

#include <mpi.h>
#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

typedef unsigned long long int sparse_pointer_t;
typedef unsigned int sparse_index_t;
typedef double sparse_value_t;

#include "pangulu.h"
#include "mtx_reader.h"

namespace {

constexpr int DEFAULT_POISSON_SIDE = 64;
constexpr int DEFAULT_BLOCK_SIZE = 64;
constexpr double TEST_TOL = 1.0e-6;

#define MPI_SPARSE_INDEX_T MPI_UNSIGNED
#define MPI_SPARSE_POINTER_T MPI_UNSIGNED_LONG_LONG

struct InputSpec {
    bool use_file = false;
    std::string path;
    int poisson_side = DEFAULT_POISSON_SIDE;
    int next_arg = 1;
};

[[noreturn]] void fail(int rank, const char* message)
{
    if (rank == 0) {
        std::fprintf(stderr, "Error: %s\n", message);
        std::fflush(stderr);
    }
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

int parse_positive_int(const char* text, const char* name, int rank)
{
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (text == end || *end != '\0' || value <= 0 ||
        value > std::numeric_limits<int>::max()) {
        if (rank == 0)
            std::fprintf(stderr, "Error: invalid %s: %s\n", name, text);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

bool is_positive_integer(const char* text)
{
    if (!text || *text == '\0') return false;
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    return text != end && *end == '\0' && value > 0 &&
           value <= std::numeric_limits<int>::max();
}

InputSpec parse_input(int argc, char** argv, int rank)
{
    InputSpec spec;
    if (argc == 1) return spec;

    if (std::strcmp(argv[1], "--poisson") == 0) {
        if (argc < 3) fail(rank, "--poisson requires a side length");
        spec.poisson_side =
            parse_positive_int(argv[2], "Poisson side", rank);
        spec.next_arg = 3;
        return spec;
    }

    if (std::strcmp(argv[1], "--matrix") == 0) {
        if (argc < 3) fail(rank, "--matrix requires a file path");
        spec.use_file = true;
        spec.path = argv[2];
        spec.next_arg = 3;
        return spec;
    }

    if (is_positive_integer(argv[1])) {
        spec.poisson_side =
            parse_positive_int(argv[1], "Poisson side", rank);
        spec.next_arg = 2;
    } else {
        spec.use_file = true;
        spec.path = argv[1];
        spec.next_arg = 2;
    }
    return spec;
}

void* checked_malloc(size_t bytes, int rank, const char* name)
{
    void* pointer = std::malloc(bytes);
    if (!pointer && bytes != 0) {
        if (rank == 0)
            std::fprintf(stderr, "Allocation failed for %s\n", name);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    return pointer;
}

bool build_poisson_2d(int side, MtxMatrix* mat)
{
    if (!mat || side <= 0) return false;
    const long long n64 = static_cast<long long>(side) * side;
    const long long nnz64 = 5LL * n64 - 4LL * side;
    if (n64 > INT_MAX || nnz64 > INT_MAX) return false;

    mtx_init(mat);
    mat->nrows = static_cast<long>(n64);
    mat->ncols = static_cast<long>(n64);
    mat->nnz = static_cast<long>(nnz64);
    mat->symmetric = 1;
    mat->rowptr = static_cast<int*>(
        std::malloc((static_cast<size_t>(n64) + 1) * sizeof(int)));
    mat->colind = static_cast<int*>(
        std::malloc(static_cast<size_t>(nnz64) * sizeof(int)));
    mat->values = static_cast<double*>(
        std::malloc(static_cast<size_t>(nnz64) * sizeof(double)));
    if (!mat->rowptr || !mat->colind || !mat->values) {
        mtx_free(mat);
        return false;
    }

    int position = 0;
    mat->rowptr[0] = 0;
    for (int row = 0; row < static_cast<int>(n64); ++row) {
        const int iy = row / side;
        const int ix = row % side;
        if (iy > 0) {
            mat->colind[position] = row - side;
            mat->values[position++] = -1.0;
        }
        if (ix > 0) {
            mat->colind[position] = row - 1;
            mat->values[position++] = -1.0;
        }
        mat->colind[position] = row;
        mat->values[position++] = 4.0;
        if (ix + 1 < side) {
            mat->colind[position] = row + 1;
            mat->values[position++] = -1.0;
        }
        if (iy + 1 < side) {
            mat->colind[position] = row + side;
            mat->values[position++] = -1.0;
        }
        mat->rowptr[row + 1] = position;
    }
    if (position != nnz64) {
        mtx_free(mat);
        return false;
    }
    return true;
}

void csr_to_csc(long nrows, long ncols, long nnz,
                const int* csr_rowptr, const int* csr_colind,
                const double* csr_values,
                sparse_pointer_t** colptr_out,
                sparse_index_t** rowidx_out,
                sparse_value_t** values_out,
                int rank)
{
    auto* colptr = static_cast<sparse_pointer_t*>(
        checked_malloc((static_cast<size_t>(ncols) + 1) *
                           sizeof(sparse_pointer_t),
                       rank, "CSC colptr"));
    auto* rowidx = static_cast<sparse_index_t*>(
        checked_malloc(static_cast<size_t>(nnz) *
                           sizeof(sparse_index_t),
                       rank, "CSC rowidx"));
    auto* values = static_cast<sparse_value_t*>(
        checked_malloc(static_cast<size_t>(nnz) *
                           sizeof(sparse_value_t),
                       rank, "CSC values"));

    std::memset(colptr, 0,
                (static_cast<size_t>(ncols) + 1) *
                    sizeof(sparse_pointer_t));

    for (long p = 0; p < nnz; ++p) {
        const int column = csr_colind[p];
        if (column < 0 || column >= ncols)
            fail(rank, "CSR column index is out of range");
        ++colptr[column + 1];
    }
    for (long column = 0; column < ncols; ++column)
        colptr[column + 1] += colptr[column];

    auto* offset = static_cast<sparse_pointer_t*>(
        checked_malloc(static_cast<size_t>(ncols) *
                           sizeof(sparse_pointer_t),
                       rank, "CSC offsets"));
    std::memcpy(offset, colptr,
                static_cast<size_t>(ncols) *
                    sizeof(sparse_pointer_t));

    /*
     * Iterating CSR rows in ascending order makes row indices sorted
     * inside every resulting CSC column.
     */
    for (long row = 0; row < nrows; ++row) {
        for (int p = csr_rowptr[row]; p < csr_rowptr[row + 1]; ++p) {
            const int column = csr_colind[p];
            const sparse_pointer_t position = offset[column]++;
            rowidx[position] = static_cast<sparse_index_t>(row);
            values[position] =
                static_cast<sparse_value_t>(csr_values[p]);
        }
    }

    std::free(offset);
    *colptr_out = colptr;
    *rowidx_out = rowidx;
    *values_out = values;
}

void configure_local_gpu(int rank, int* selected_gpu, int* gpu_count)
{
    *selected_gpu = -1;
    *gpu_count = 0;

    MPI_Comm local_comm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                        rank, MPI_INFO_NULL, &local_comm);
    int local_rank = 0;
    MPI_Comm_rank(local_comm, &local_rank);

    const cudaError_t count_status = cudaGetDeviceCount(gpu_count);
    if (count_status == cudaSuccess && *gpu_count > 0) {
        *selected_gpu = local_rank % *gpu_count;
        const cudaError_t set_status = cudaSetDevice(*selected_gpu);
        if (set_status != cudaSuccess)
            fail(rank, cudaGetErrorString(set_status));
        cudaFree(nullptr);  // initialize the CUDA context
    } else {
        cudaGetLastError();  // clear a possible runtime error
    }
    MPI_Comm_free(&local_comm);
}

}  // namespace

int main(int argc, char** argv)
{
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided) !=
        MPI_SUCCESS) {
        std::fprintf(stderr, "MPI_Init_thread failed\n");
        return EXIT_FAILURE;
    }

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const InputSpec input = parse_input(argc, argv, rank);
    const int remaining = argc - input.next_arg;
    if (remaining != 0 && remaining != 1) {
        if (rank == 0) {
            std::fprintf(stderr,
                "Usage:\n"
                "  mpirun -np P ./test_pangu\n"
                "  mpirun -np P ./test_pangu matrix.mtx [nb]\n"
                "  mpirun -np P ./test_pangu --matrix matrix.mtx [nb]\n"
                "  mpirun -np P ./test_pangu --poisson side [nb]\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    const bool block_size_explicit = remaining == 1;
    int block_size = block_size_explicit
        ? parse_positive_int(argv[input.next_arg], "nb", rank)
        : DEFAULT_BLOCK_SIZE;

    int selected_gpu = -1;
    int gpu_count = 0;
    configure_local_gpu(rank, &selected_gpu, &gpu_count);

    MtxMatrix matrix;
    mtx_init(&matrix);
    std::string matrix_name;

    sparse_index_t n = 0;
    sparse_pointer_t nnz = 0;
    sparse_pointer_t* colptr = nullptr;
    sparse_index_t* rowidx = nullptr;
    sparse_value_t* values = nullptr;
    sparse_value_t* rhs_and_solution = nullptr;
    double* exact_rhs = nullptr;

    if (rank == 0) {
        bool loaded = false;
        if (input.use_file) {
            loaded = mtx_read(input.path.c_str(), &matrix) != 0;
            matrix_name = input.path;
        } else {
            loaded = build_poisson_2d(input.poisson_side, &matrix);
            matrix_name = "Poisson-2D(" +
                          std::to_string(input.poisson_side) + "x" +
                          std::to_string(input.poisson_side) + ")";
        }

        if (!loaded) fail(rank, "failed to create/read the matrix");
        if (matrix.nrows != matrix.ncols)
            fail(rank, "PanguLU requires a square matrix");
        if (matrix.nrows <= 0 || matrix.nnz <= 0)
            fail(rank, "matrix is empty");
        if (static_cast<unsigned long long>(matrix.nrows) >
            std::numeric_limits<sparse_index_t>::max())
            fail(rank, "matrix order exceeds sparse_index_t");
        if (static_cast<unsigned long long>(matrix.nnz) >
            std::numeric_limits<sparse_pointer_t>::max())
            fail(rank, "matrix nnz exceeds sparse_pointer_t");

        n = static_cast<sparse_index_t>(matrix.nrows);
        nnz = static_cast<sparse_pointer_t>(matrix.nnz);

        if (!block_size_explicit && block_size > static_cast<int>(n))
            block_size = static_cast<int>(n);
        if (block_size <= 0 ||
            static_cast<sparse_index_t>(block_size) > n)
            fail(rank, "nb must be in [1,n]");

        csr_to_csc(matrix.nrows, matrix.ncols, matrix.nnz,
                   matrix.rowptr, matrix.colind, matrix.values,
                   &colptr, &rowidx, &values, rank);

        exact_rhs = mtx_compute_rhs_one(&matrix);
        if (!exact_rhs) fail(rank, "failed to construct b=A*1");

        rhs_and_solution = static_cast<sparse_value_t*>(
            checked_malloc(static_cast<size_t>(n) *
                               sizeof(sparse_value_t),
                           rank, "PanguLU rhs/solution"));
        for (sparse_index_t i = 0; i < n; ++i)
            rhs_and_solution[i] =
                static_cast<sparse_value_t>(exact_rhs[i]);
    }

    MPI_Bcast(&n, 1, MPI_SPARSE_INDEX_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nnz, 1, MPI_SPARSE_POINTER_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&block_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    pangulu_init_options init_options{};
    pangulu_gstrf_options factor_options{};
    pangulu_gstrs_options solve_options{};

    init_options.nthread = 1;
    init_options.nb = block_size;
    init_options.gpu_kernel_warp_per_block = 4;
    init_options.gpu_data_move_warp_per_block = 4;
    init_options.sizeof_value =
        static_cast<int>(sizeof(sparse_value_t));
    init_options.is_complex_matrix = 0;
    init_options.mpi_recv_buffer_level = 0.5f;

    if (rank == 0) {
        std::printf("========================================\n");
        std::printf("Solver          : PanguLU\n");
        std::printf("Input           : %s\n", matrix_name.c_str());
        std::printf("Matrix size     : %u x %u\n", n, n);
        std::printf("Global nnz      : %llu\n",
                    static_cast<unsigned long long>(nnz));
        std::printf("Block size nb   : %d\n", block_size);
        std::printf("MPI ranks       : %d\n", size);
        std::printf("MPI thread      : requested MULTIPLE, provided %d\n",
                    provided);
        std::printf("Visible GPUs    : %d per current node\n", gpu_count);
        std::printf("GPU use         : determined by the PanguLU build "
                    "(GPU_OPEN)\n");
        std::printf("========================================\n");
        std::fflush(stdout);
    }
    if (selected_gpu >= 0) {
        std::printf("MPI rank %d selects GPU %d\n", rank, selected_gpu);
        std::fflush(stdout);
    }

    void* pangulu_handle = nullptr;

    MPI_Barrier(MPI_COMM_WORLD);
    const double init_begin = MPI_Wtime();
    pangulu_init(n, nnz, colptr, rowidx, values,
                 &init_options, &pangulu_handle);
    MPI_Barrier(MPI_COMM_WORLD);
    const double init_end = MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);
    const double factor_begin = MPI_Wtime();
    pangulu_gstrf(&factor_options, &pangulu_handle);
    MPI_Barrier(MPI_COMM_WORLD);
    const double factor_end = MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);
    const double solve_begin = MPI_Wtime();
    pangulu_gstrs(rhs_and_solution, &solve_options, &pangulu_handle);
    MPI_Barrier(MPI_COMM_WORLD);
    const double solve_end = MPI_Wtime();

    const double local_init = init_end - init_begin;
    const double local_factor = factor_end - factor_begin;
    const double local_solve = solve_end - solve_begin;

    double init_time = 0.0;
    double factor_time = 0.0;
    double solve_time = 0.0;
    MPI_Reduce(&local_init, &init_time, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_factor, &factor_time, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_solve, &solve_time, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);

    int failed = 0;
    if (rank == 0) {
        auto* x = static_cast<double*>(
            checked_malloc(static_cast<size_t>(n) * sizeof(double),
                           rank, "verification solution"));
        for (sparse_index_t i = 0; i < n; ++i)
            x[i] = static_cast<double>(rhs_and_solution[i]);

        const double relative_residual =
            mtx_relative_residual(&matrix, x, exact_rhs);
        const double relative_error =
            mtx_relative_error(&matrix, x);

        failed =
            !std::isfinite(relative_residual) ||
            !std::isfinite(relative_error) ||
            relative_residual > TEST_TOL ||
            relative_error > TEST_TOL;

        std::printf("\n--- Results ---\n");
        std::printf("Initialization time : %.6f s\n", init_time);
        std::printf("Factorization time  : %.6f s\n", factor_time);
        std::printf("Solve time          : %.6f s\n", solve_time);
        std::printf("Total timed time    : %.6f s\n",
                    init_time + factor_time + solve_time);
        std::printf("Relative residual   : %.12e\n",
                    relative_residual);
        std::printf("Relative x error    : %.12e\n",
                    relative_error);
        std::printf("TEST                : %s\n",
                    failed ? "FAILED" : "PASSED");
        std::fflush(stdout);
        std::free(x);
    }
    MPI_Bcast(&failed, 1, MPI_INT, 0, MPI_COMM_WORLD);

    pangulu_finalize(&pangulu_handle);

    if (rank == 0) {
        std::free(colptr);
        std::free(rowidx);
        std::free(values);
        std::free(rhs_and_solution);
        std::free(exact_rhs);
        mtx_free(&matrix);
    }

    MPI_Finalize();
    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
