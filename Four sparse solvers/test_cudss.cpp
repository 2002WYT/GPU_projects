/*
 * test_cudss.cpp
 *
 * NVIDIA cuDSS 0.8+ GPU direct-solver test with MPI-aware GPU selection.
 *
 * Input:
 *   mpirun -np 1 ./test_cudss
 *   mpirun -np 1 ./test_cudss matrix.mtx
 *   mpirun -np 1 ./test_cudss --poisson 256
 *
 * Optional final argument:
 *   CUDA device id. Without it, each node-local MPI rank selects
 *   local_rank % visible_gpu_count.
 *
 * Examples:
 *   ./test_cudss matrix.mtx 0
 *   mpirun -np 2 ./test_cudss --poisson 256
 *
 * Important:
 *   cuDSS in this program performs one complete GPU solve per MPI rank.
 *   It is MPI-aware, but the sparse matrix is replicated rather than
 *   row-distributed. For a single V100, use -np 1 for normal benchmarking.
 *
 * The manufactured exact solution is x*=1 and b=A*x*.
 */

#include <mpi.h>
#include <cuda_runtime.h>
#include <cudss.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

#include "mtx_reader.h"

namespace {

constexpr int DEFAULT_POISSON_SIDE = 64;
constexpr double TEST_TOL = 1.0e-6;

struct InputSpec {
    bool use_file = false;
    std::string path;
    int poisson_side = DEFAULT_POISSON_SIDE;
    int next_arg = 1;
};

[[noreturn]] void mpi_fail(int rank, const char* message)
{
    if (rank == 0) {
        std::fprintf(stderr, "Error: %s\n", message);
        std::fflush(stderr);
    }
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

const char* cudss_status_name(cudssStatus_t status)
{
    switch (status) {
        case CUDSS_STATUS_SUCCESS:
            return "CUDSS_STATUS_SUCCESS";
        case CUDSS_STATUS_NOT_INITIALIZED:
            return "CUDSS_STATUS_NOT_INITIALIZED";
        case CUDSS_STATUS_ALLOC_FAILED:
            return "CUDSS_STATUS_ALLOC_FAILED";
        case CUDSS_STATUS_INVALID_VALUE:
            return "CUDSS_STATUS_INVALID_VALUE";
        case CUDSS_STATUS_NOT_SUPPORTED:
            return "CUDSS_STATUS_NOT_SUPPORTED";
        case CUDSS_STATUS_EXECUTION_FAILED:
            return "CUDSS_STATUS_EXECUTION_FAILED";
        case CUDSS_STATUS_INTERNAL_ERROR:
            return "CUDSS_STATUS_INTERNAL_ERROR";
        case CUDSS_STATUS_IR_FAILED:
            return "CUDSS_STATUS_IR_FAILED";
        default:
            return "CUDSS_STATUS_UNKNOWN";
    }
}

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        const cudaError_t error__ = (call);                                \
        if (error__ != cudaSuccess) {                                      \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",              \
                         __FILE__, __LINE__, cudaGetErrorString(error__));  \
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                       \
        }                                                                  \
    } while (0)

#define CUDSS_CHECK(call)                                                  \
    do {                                                                   \
        const cudssStatus_t status__ = (call);                             \
        if (status__ != CUDSS_STATUS_SUCCESS) {                            \
            std::fprintf(stderr,                                           \
                         "cuDSS error at %s:%d: %s (%d)\n",                 \
                         __FILE__, __LINE__,                                \
                         cudss_status_name(status__),                       \
                         static_cast<int>(status__));                       \
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                       \
        }                                                                  \
    } while (0)

int parse_nonnegative_int(const char* text, const char* name, int rank)
{
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (text == end || *end != '\0' || value < 0 ||
        value > std::numeric_limits<int>::max()) {
        if (rank == 0)
            std::fprintf(stderr, "Invalid %s: %s\n", name, text);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

int parse_positive_int(const char* text, const char* name, int rank)
{
    const int value = parse_nonnegative_int(text, name, rank);
    if (value == 0) {
        if (rank == 0)
            std::fprintf(stderr, "%s must be positive\n", name);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    return value;
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
        if (argc < 3) mpi_fail(rank, "--poisson requires a side length");
        spec.poisson_side =
            parse_positive_int(argv[2], "Poisson side", rank);
        spec.next_arg = 3;
        return spec;
    }
    if (std::strcmp(argv[1], "--matrix") == 0) {
        if (argc < 3) mpi_fail(rank, "--matrix requires a file path");
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

int select_gpu(int requested_device, bool explicit_device,
               int rank, int* device_count, int* local_rank_out)
{
    CUDA_CHECK(cudaGetDeviceCount(device_count));
    if (*device_count <= 0) mpi_fail(rank, "no CUDA device is visible");

    MPI_Comm local_comm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                        rank, MPI_INFO_NULL, &local_comm);
    int local_rank = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);
    *local_rank_out = local_rank;

    const int device = explicit_device
        ? requested_device
        : local_rank % *device_count;
    if (device < 0 || device >= *device_count)
        mpi_fail(rank, "requested CUDA device is out of range");

    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaFree(nullptr));
    return device;
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
                "  ./test_cudss\n"
                "  ./test_cudss matrix.mtx [device]\n"
                "  ./test_cudss --matrix matrix.mtx [device]\n"
                "  ./test_cudss --poisson side [device]\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    const bool explicit_device = remaining == 1;
    const int requested_device = explicit_device
        ? parse_nonnegative_int(argv[input.next_arg], "device", rank)
        : 0;

    int device_count = 0;
    int local_rank = 0;
    const int device = select_gpu(requested_device, explicit_device,
                                  rank, &device_count, &local_rank);

    MtxMatrix matrix;
    mtx_init(&matrix);
    std::string matrix_name;
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

    if (!loaded) mpi_fail(rank, "failed to create/read the matrix");
    if (matrix.nrows != matrix.ncols)
        mpi_fail(rank, "cuDSS requires a square matrix");
    if (matrix.nrows <= 0 || matrix.nnz <= 0)
        mpi_fail(rank, "matrix is empty");
    if (matrix.nrows > INT_MAX || matrix.nnz > INT_MAX)
        mpi_fail(rank,
                 "mtx_reader.h uses int CSR offsets; n and nnz "
                 "must fit in int");

    double* host_rhs = mtx_compute_rhs_one(&matrix);
    if (!host_rhs) mpi_fail(rank, "failed to construct b=A*1");
    auto* host_x = static_cast<double*>(
        checked_malloc(static_cast<size_t>(matrix.nrows) *
                           sizeof(double),
                       rank, "host solution"));
    std::memset(host_x, 0,
                static_cast<size_t>(matrix.nrows) * sizeof(double));

    const int64_t n = static_cast<int64_t>(matrix.nrows);
    const int64_t nnz = static_cast<int64_t>(matrix.nnz);

    int* device_rowptr = nullptr;
    int* device_colind = nullptr;
    double* device_values = nullptr;
    double* device_x = nullptr;
    double* device_rhs = nullptr;

    CUDA_CHECK(cudaMalloc(&device_rowptr,
              (static_cast<size_t>(n) + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&device_colind,
              static_cast<size_t>(nnz) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&device_values,
              static_cast<size_t>(nnz) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&device_x,
              static_cast<size_t>(n) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&device_rhs,
              static_cast<size_t>(n) * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(device_rowptr, matrix.rowptr,
              (static_cast<size_t>(n) + 1) * sizeof(int),
              cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_colind, matrix.colind,
              static_cast<size_t>(nnz) * sizeof(int),
              cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_values, matrix.values,
              static_cast<size_t>(nnz) * sizeof(double),
              cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_rhs, host_rhs,
              static_cast<size_t>(n) * sizeof(double),
              cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(device_x, 0,
              static_cast<size_t>(n) * sizeof(double)));

    cudssHandle_t handle = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t data = nullptr;
    cudssMatrix_t matA = nullptr;
    cudssMatrix_t matX = nullptr;
    cudssMatrix_t matB = nullptr;

    CUDSS_CHECK(cudssCreate(&handle));
    CUDSS_CHECK(cudssConfigCreate(&config));
    CUDSS_CHECK(cudssDataCreate(handle, &data));

    /*
     * cuDSS 0.8 standard CSR:
     * rowStart=rowptr, rowEnd=nullptr, separate offset/index datatypes.
     */
    CUDSS_CHECK(cudssMatrixCreateCsr(
        &matA, n, n, nnz,
        device_rowptr,
        nullptr,
        device_colind,
        device_values,
        CUDSS_R_32I,
        CUDSS_R_32I,
        CUDSS_R_64F,
        CUDSS_MTYPE_GENERAL,
        CUDSS_MVIEW_FULL,
        CUDSS_BASE_ZERO));

    CUDSS_CHECK(cudssMatrixCreateDn(
        &matX, n, 1, n, device_x,
        CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));
    CUDSS_CHECK(cudssMatrixCreateDn(
        &matB, n, 1, n, device_rhs,
        CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));

    int major = 0, minor = 0, patch = 0;
    CUDSS_CHECK(cudssGetProperty(MAJOR_VERSION, &major));
    CUDSS_CHECK(cudssGetProperty(MINOR_VERSION, &minor));
    CUDSS_CHECK(cudssGetProperty(PATCH_LEVEL, &patch));

    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

    if (rank == 0) {
        std::printf("========================================\n");
        std::printf("Solver          : NVIDIA cuDSS\n");
        std::printf("Version         : %d.%d.%d\n", major, minor, patch);
        std::printf("Input           : %s\n", matrix_name.c_str());
        std::printf("Matrix size     : %lld x %lld\n",
                    static_cast<long long>(n),
                    static_cast<long long>(n));
        std::printf("Global nnz      : %lld\n",
                    static_cast<long long>(nnz));
        std::printf("MPI ranks       : %d\n", size);
        std::printf("MPI mode        : one replicated solve per rank\n");
        std::printf("MPI thread      : requested MULTIPLE, provided %d\n",
                    provided);
        std::printf("Rank-0 GPU      : %s (device %d)\n",
                    properties.name, device);
        std::printf("========================================\n");
        std::fflush(stdout);
    }
    std::printf("MPI rank %d (local rank %d) uses GPU %d\n",
                rank, local_rank, device);
    std::fflush(stdout);

    cudaEvent_t event_start = nullptr;
    cudaEvent_t event_end = nullptr;
    CUDA_CHECK(cudaEventCreate(&event_start));
    CUDA_CHECK(cudaEventCreate(&event_end));

    auto execute_phase = [&](cudssPhase_t phase) -> double {
        CUDA_CHECK(cudaEventRecord(event_start));
        const cudssStatus_t status =
            cudssExecute(handle, phase, config, data, matA, matX, matB);
        CUDA_CHECK(cudaEventRecord(event_end));
        CUDA_CHECK(cudaEventSynchronize(event_end));

        if (status != CUDSS_STATUS_SUCCESS) {
            std::fprintf(stderr,
                         "Rank %d: cuDSS phase %d failed: %s (%d)\n",
                         rank, static_cast<int>(phase),
                         cudss_status_name(status),
                         static_cast<int>(status));
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(
            &milliseconds, event_start, event_end));
        return static_cast<double>(milliseconds) / 1000.0;
    };

    MPI_Barrier(MPI_COMM_WORLD);
    const double analysis_time =
        execute_phase(CUDSS_PHASE_ANALYSIS);
    const double factor_time =
        execute_phase(CUDSS_PHASE_FACTORIZATION);
    const double solve_time =
        execute_phase(CUDSS_PHASE_SOLVE);
    MPI_Barrier(MPI_COMM_WORLD);

    CUDA_CHECK(cudaMemcpy(host_x, device_x,
              static_cast<size_t>(n) * sizeof(double),
              cudaMemcpyDeviceToHost));

    const double local_residual =
        mtx_relative_residual(&matrix, host_x, host_rhs);
    const double local_error =
        mtx_relative_error(&matrix, host_x);

    double max_analysis = 0.0;
    double max_factor = 0.0;
    double max_solve = 0.0;
    double max_residual = 0.0;
    double max_error = 0.0;
    MPI_Reduce(&analysis_time, &max_analysis, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&factor_time, &max_factor, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&solve_time, &max_solve, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_residual, &max_residual, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_error, &max_error, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);

    int local_failed =
        !std::isfinite(local_residual) ||
        !std::isfinite(local_error) ||
        local_residual > TEST_TOL ||
        local_error > TEST_TOL;
    int failed = 0;
    MPI_Allreduce(&local_failed, &failed, 1,
                  MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0) {
        std::printf("\n--- Results (maximum over MPI ranks) ---\n");
        std::printf("Analysis time       : %.6f s\n", max_analysis);
        std::printf("Factorization time  : %.6f s\n", max_factor);
        std::printf("Solve time          : %.6f s\n", max_solve);
        std::printf("Total time          : %.6f s\n",
                    max_analysis + max_factor + max_solve);
        std::printf("Relative residual   : %.12e\n", max_residual);
        std::printf("Relative x error    : %.12e\n", max_error);
        std::printf("TEST                : %s\n",
                    failed ? "FAILED" : "PASSED");
        std::fflush(stdout);
    }

    CUDA_CHECK(cudaEventDestroy(event_start));
    CUDA_CHECK(cudaEventDestroy(event_end));

    CUDSS_CHECK(cudssMatrixDestroy(matA));
    CUDSS_CHECK(cudssMatrixDestroy(matX));
    CUDSS_CHECK(cudssMatrixDestroy(matB));
    CUDSS_CHECK(cudssDataDestroy(handle, data));
    CUDSS_CHECK(cudssConfigDestroy(config));
    CUDSS_CHECK(cudssDestroy(handle));

    CUDA_CHECK(cudaFree(device_rowptr));
    CUDA_CHECK(cudaFree(device_colind));
    CUDA_CHECK(cudaFree(device_values));
    CUDA_CHECK(cudaFree(device_x));
    CUDA_CHECK(cudaFree(device_rhs));

    std::free(host_rhs);
    std::free(host_x);
    mtx_free(&matrix);

    MPI_Finalize();
    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
