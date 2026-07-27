/*
 * test_strumpack.cpp
 *
 * STRUMPACK distributed MPI/GPU direct-solver test.
 *
 * Input:
 *   mpirun -np 1 ./test_strumpack
 *   mpirun -np 4 ./test_strumpack matrix.mtx
 *   mpirun -np 4 ./test_strumpack --poisson 256
 *
 * Optional final argument:
 *   number of GPU streams (default 4)
 *
 * Examples:
 *   mpirun -np 2 ./test_strumpack matrix.mtx 4
 *   mpirun -np 2 ./test_strumpack --poisson 256 4
 *
 * The manufactured exact solution is x*=1 and b=A*x*.
 */

#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "StrumpackSparseSolverMPIDist.hpp"
#include "mtx_reader.h"

using strumpack::CompressionType;
using strumpack::KrylovSolver;
using strumpack::MatchingJob;
using strumpack::ReorderingStrategy;
using strumpack::ReturnCode;
using strumpack::StrumpackSparseSolverMPIDist;

namespace {

constexpr int DEFAULT_POISSON_SIDE = 64;
constexpr int DEFAULT_GPU_STREAMS = 4;
constexpr double TEST_TOL = 1.0e-8;

struct InputSpec {
    bool use_file = false;
    std::string path;
    int poisson_side = DEFAULT_POISSON_SIDE;
    int next_arg = 1;
};

[[noreturn]] void fail(int rank, const char* message)
{
    if (rank == 0) {
        std::cerr << "Error: " << message << std::endl;
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
            std::cerr << "Invalid " << name << ": " << text << '\n';
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

std::vector<int> make_distribution(int global_rows, int mpi_size)
{
    std::vector<int> distribution(mpi_size + 1, 0);
    const int base = global_rows / mpi_size;
    const int remainder = global_rows % mpi_size;
    for (int rank = 0; rank < mpi_size; ++rank) {
        distribution[rank + 1] =
            distribution[rank] + base +
            (rank < remainder ? 1 : 0);
    }
    return distribution;
}

bool all_ranks_success(bool local_success)
{
    const int local = local_success ? 1 : 0;
    int global = 0;
    MPI_Allreduce(&local, &global, 1,
                  MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    return global == 1;
}

bool configure_local_gpu(int rank, int* selected_gpu, int* gpu_count)
{
    *selected_gpu = -1;
    *gpu_count = 0;

    MPI_Comm local_comm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                        rank, MPI_INFO_NULL, &local_comm);
    int local_rank = 0;
    MPI_Comm_rank(local_comm, &local_rank);

    const cudaError_t count_status = cudaGetDeviceCount(gpu_count);
    bool enabled = count_status == cudaSuccess && *gpu_count > 0;
    if (enabled) {
        *selected_gpu = local_rank % *gpu_count;
        const cudaError_t set_status = cudaSetDevice(*selected_gpu);
        if (set_status != cudaSuccess)
            fail(rank, cudaGetErrorString(set_status));
        cudaFree(nullptr);
    } else {
        cudaGetLastError();
    }

    MPI_Comm_free(&local_comm);
    return enabled;
}

}  // namespace

int main(int argc, char** argv)
{
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided) !=
        MPI_SUCCESS) {
        std::cerr << "MPI_Init_thread failed\n";
        return EXIT_FAILURE;
    }

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const InputSpec input = parse_input(argc, argv, rank);
    const int remaining = argc - input.next_arg;
    if (remaining != 0 && remaining != 1) {
        if (rank == 0) {
            std::cerr
                << "Usage:\n"
                << "  mpirun -np P ./test_strumpack\n"
                << "  mpirun -np P ./test_strumpack matrix.mtx "
                   "[gpu_streams]\n"
                << "  mpirun -np P ./test_strumpack "
                   "--matrix matrix.mtx [gpu_streams]\n"
                << "  mpirun -np P ./test_strumpack "
                   "--poisson side [gpu_streams]\n";
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    const int gpu_streams = remaining == 1
        ? parse_positive_int(argv[input.next_arg],
                             "GPU stream count", rank)
        : DEFAULT_GPU_STREAMS;

    if (provided < MPI_THREAD_MULTIPLE && rank == 0) {
        std::cerr
            << "Warning: MPI did not provide MPI_THREAD_MULTIPLE. "
            << "A SLATE-enabled GPU build can require it.\n";
    }

    int selected_gpu = -1;
    int gpu_count = 0;
    const bool gpu_enabled =
        configure_local_gpu(rank, &selected_gpu, &gpu_count);

    MtxMatrix global_matrix;
    mtx_init(&global_matrix);
    std::string matrix_name;

    long metadata[3] = {0, 0, 0};
    if (rank == 0) {
        bool loaded = false;
        if (input.use_file) {
            loaded = mtx_read(input.path.c_str(), &global_matrix) != 0;
            matrix_name = input.path;
        } else {
            loaded = build_poisson_2d(input.poisson_side, &global_matrix);
            matrix_name = "Poisson-2D(" +
                          std::to_string(input.poisson_side) + "x" +
                          std::to_string(input.poisson_side) + ")";
        }

        if (!loaded) fail(rank, "failed to create/read the matrix");
        if (global_matrix.nrows != global_matrix.ncols)
            fail(rank, "matrix must be square");
        if (global_matrix.nrows <= 0 || global_matrix.nnz <= 0)
            fail(rank, "matrix is empty");
        if (global_matrix.nrows > INT_MAX ||
            global_matrix.nnz > INT_MAX)
            fail(rank, "STRUMPACK test uses 32-bit integer indices");

        metadata[0] = global_matrix.nrows;
        metadata[1] = global_matrix.ncols;
        metadata[2] = global_matrix.nnz;
    }
    MPI_Bcast(metadata, 3, MPI_LONG, 0, MPI_COMM_WORLD);

    const int global_rows = static_cast<int>(metadata[0]);
    const long long global_nnz =
        static_cast<long long>(metadata[2]);
    if (global_rows < size)
        fail(rank,
             "matrix order must be at least the number of MPI ranks");

    const std::vector<int> distribution =
        make_distribution(global_rows, size);
    const int first_global_row = distribution[rank];
    const int local_rows =
        distribution[rank + 1] - distribution[rank];

    std::vector<int> rowptr(local_rows + 1, 0);
    std::vector<int> colind;
    std::vector<double> values;
    std::vector<double> rhs(local_rows, 0.0);
    std::vector<double> x(local_rows, 0.0);

    if (rank == 0) {
        double* global_rhs = mtx_compute_rhs_one(&global_matrix);
        if (!global_rhs) fail(rank, "failed to construct b=A*1");

        for (int destination = 0; destination < size; ++destination) {
            const int destination_first = distribution[destination];
            const int destination_rows =
                distribution[destination + 1] -
                distribution[destination];

            std::vector<int> d_rowptr(destination_rows + 1, 0);
            std::vector<int> d_colind;
            std::vector<double> d_values;
            std::vector<double> d_rhs(destination_rows, 0.0);

            for (int local_row = 0;
                 local_row < destination_rows; ++local_row) {
                const int global_row =
                    destination_first + local_row;
                for (int p = global_matrix.rowptr[global_row];
                     p < global_matrix.rowptr[global_row + 1]; ++p) {
                    d_colind.push_back(global_matrix.colind[p]);
                    d_values.push_back(global_matrix.values[p]);
                }
                d_rhs[local_row] = global_rhs[global_row];
                d_rowptr[local_row + 1] =
                    static_cast<int>(d_colind.size());
            }

            if (destination == 0) {
                rowptr = std::move(d_rowptr);
                colind = std::move(d_colind);
                values = std::move(d_values);
                rhs = std::move(d_rhs);
            } else {
                const int destination_nnz =
                    static_cast<int>(d_colind.size());
                MPI_Send(&destination_nnz, 1, MPI_INT,
                         destination, 100, MPI_COMM_WORLD);
                MPI_Send(d_rowptr.data(), destination_rows + 1,
                         MPI_INT, destination, 101, MPI_COMM_WORLD);
                MPI_Send(d_colind.data(), destination_nnz,
                         MPI_INT, destination, 102, MPI_COMM_WORLD);
                MPI_Send(d_values.data(), destination_nnz,
                         MPI_DOUBLE, destination, 103, MPI_COMM_WORLD);
                MPI_Send(d_rhs.data(), destination_rows,
                         MPI_DOUBLE, destination, 104, MPI_COMM_WORLD);
            }
        }
        std::free(global_rhs);
    } else {
        int local_nnz = 0;
        MPI_Recv(&local_nnz, 1, MPI_INT, 0, 100,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        rowptr.resize(local_rows + 1);
        colind.resize(local_nnz);
        values.resize(local_nnz);
        rhs.resize(local_rows);

        MPI_Recv(rowptr.data(), local_rows + 1, MPI_INT,
                 0, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(colind.data(), local_nnz, MPI_INT,
                 0, 102, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(values.data(), local_nnz, MPI_DOUBLE,
                 0, 103, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(rhs.data(), local_rows, MPI_DOUBLE,
                 0, 104, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (rank == 0) {
        std::cout << "========================================\n"
                  << "Solver          : STRUMPACK\n"
                  << "Input           : " << matrix_name << '\n'
                  << "Matrix size     : " << global_rows
                  << " x " << global_rows << '\n'
                  << "Global nnz      : " << global_nnz << '\n'
                  << "MPI ranks       : " << size << '\n'
                  << "MPI thread      : requested MULTIPLE, provided "
                  << provided << '\n'
                  << "GPU requested   : "
                  << (gpu_enabled ? "yes" : "no visible CUDA GPU")
                  << '\n'
                  << "GPU streams     : " << gpu_streams << '\n'
                  << "========================================\n";
    }
    if (gpu_enabled) {
        std::cout << "MPI rank " << rank
                  << " selects GPU " << selected_gpu << std::endl;
    }

    int exit_code = EXIT_SUCCESS;
    {
        StrumpackSparseSolverMPIDist<double, int>
            solver(MPI_COMM_WORLD, true);

        solver.options().set_matching(MatchingJob::NONE);
        solver.options().set_compression(CompressionType::NONE);
        solver.options().set_Krylov_solver(KrylovSolver::DIRECT);
        solver.options().set_reordering_method(
            ReorderingStrategy::METIS);

        if (gpu_enabled) {
            solver.options().enable_gpu();
            solver.options().set_gpu_streams(gpu_streams);
        } else {
            solver.options().disable_gpu();
        }

        /*
         * The final false requests general (not symmetric-only) handling,
         * so the same code accepts both general Matrix Market input and
         * the symmetric Poisson fallback.
         */
        solver.set_distributed_csr_matrix(
            local_rows,
            rowptr.data(),
            colind.data(),
            values.data(),
            distribution.data(),
            false);

        MPI_Barrier(MPI_COMM_WORLD);
        const double reorder_begin = MPI_Wtime();
        const ReturnCode reorder_status = solver.reorder();
        MPI_Barrier(MPI_COMM_WORLD);
        const double reorder_end = MPI_Wtime();

        bool ok = all_ranks_success(
            reorder_status == ReturnCode::SUCCESS);
        if (!ok) {
            if (rank == 0)
                std::cerr << "STRUMPACK reorder failed\n";
            exit_code = EXIT_FAILURE;
        }

        double factor_begin = reorder_end;
        double factor_end = reorder_end;
        bool factor_completed = false;
        ReturnCode factor_status = reorder_status;
        if (ok) {
            MPI_Barrier(MPI_COMM_WORLD);
            factor_begin = MPI_Wtime();
            factor_status = solver.factor();
            MPI_Barrier(MPI_COMM_WORLD);
            factor_end = MPI_Wtime();
            factor_completed = true;

            ok = all_ranks_success(
                factor_status == ReturnCode::SUCCESS);
            if (!ok) {
                if (rank == 0)
                    std::cerr << "STRUMPACK factorization failed\n";
                exit_code = EXIT_FAILURE;
            }
        }

        double solve_begin = factor_end;
        double solve_end = factor_end;
        bool solve_completed = false;
        ReturnCode solve_status = factor_status;
        if (ok) {
            MPI_Barrier(MPI_COMM_WORLD);
            solve_begin = MPI_Wtime();
            solve_status = solver.solve(rhs.data(), x.data());
            MPI_Barrier(MPI_COMM_WORLD);
            solve_end = MPI_Wtime();
            solve_completed = true;

            ok = all_ranks_success(
                solve_status == ReturnCode::SUCCESS);
            if (!ok) {
                if (rank == 0)
                    std::cerr << "STRUMPACK solve failed\n";
                exit_code = EXIT_FAILURE;
            }
        }

        const double local_reorder =
            reorder_end - reorder_begin;
        const double local_factor =
            factor_completed ? factor_end - factor_begin : 0.0;
        const double local_solve =
            solve_completed ? solve_end - solve_begin : 0.0;

        double max_reorder = 0.0;
        double max_factor = 0.0;
        double max_solve = 0.0;
        MPI_Reduce(&local_reorder, &max_reorder, 1,
                   MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_factor, &max_factor, 1,
                   MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_solve, &max_solve, 1,
                   MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (ok) {
            std::vector<int> recv_counts(size);
            for (int process = 0; process < size; ++process)
                recv_counts[process] =
                    distribution[process + 1] -
                    distribution[process];

            std::vector<double> x_global(global_rows, 0.0);
            MPI_Allgatherv(
                x.data(), local_rows, MPI_DOUBLE,
                x_global.data(), recv_counts.data(),
                distribution.data(), MPI_DOUBLE,
                MPI_COMM_WORLD);

            double local_residual_sq = 0.0;
            double local_rhs_sq = 0.0;
            double local_error_sq = 0.0;
            double local_exact_sq = 0.0;
            double local_max_error = 0.0;

            for (int local_row = 0;
                 local_row < local_rows; ++local_row) {
                double ax = 0.0;
                for (int p = rowptr[local_row];
                     p < rowptr[local_row + 1]; ++p) {
                    ax += values[p] * x_global[colind[p]];
                }

                const double residual = rhs[local_row] - ax;
                const double error = x[local_row] - 1.0;
                local_residual_sq += residual * residual;
                local_rhs_sq += rhs[local_row] * rhs[local_row];
                local_error_sq += error * error;
                local_exact_sq += 1.0;
                local_max_error =
                    std::max(local_max_error, std::abs(error));
            }

            double global_residual_sq = 0.0;
            double global_rhs_sq = 0.0;
            double global_error_sq = 0.0;
            double global_exact_sq = 0.0;
            double global_max_error = 0.0;

            MPI_Reduce(&local_residual_sq, &global_residual_sq,
                       1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&local_rhs_sq, &global_rhs_sq,
                       1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&local_error_sq, &global_error_sq,
                       1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&local_exact_sq, &global_exact_sq,
                       1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&local_max_error, &global_max_error,
                       1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

            if (rank == 0) {
                const double relative_residual =
                    global_rhs_sq > 0.0
                    ? std::sqrt(global_residual_sq / global_rhs_sq)
                    : (global_residual_sq == 0.0
                       ? 0.0
                       : std::numeric_limits<double>::infinity());
                const double relative_error =
                    std::sqrt(global_error_sq / global_exact_sq);

                const bool passed =
                    std::isfinite(relative_residual) &&
                    std::isfinite(relative_error) &&
                    relative_residual <= TEST_TOL &&
                    relative_error <= TEST_TOL;

                std::cout << std::scientific
                          << std::setprecision(12)
                          << "\n--- Results ---\n"
                          << "Reorder time       : "
                          << max_reorder << " s\n"
                          << "Factorization time : "
                          << max_factor << " s\n"
                          << "Solve time         : "
                          << max_solve << " s\n"
                          << "Total time         : "
                          << max_reorder + max_factor + max_solve
                          << " s\n"
                          << "Relative residual  : "
                          << relative_residual << '\n'
                          << "Relative x error   : "
                          << relative_error << '\n'
                          << "Maximum abs error  : "
                          << global_max_error << '\n'
                          << "TEST               : "
                          << (passed ? "PASSED" : "FAILED")
                          << '\n';

                if (!passed) exit_code = EXIT_FAILURE;
            }
        }

        MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) mtx_free(&global_matrix);

    MPI_Finalize();
    return exit_code;
}
