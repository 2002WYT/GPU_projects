/*
 * test_superlu_dist.cpp
 *
 * SuperLU_DIST distributed direct-solver test.
 *
 * Input:
 *   1) No matrix argument:
 *        mpirun -np 1 ./test_slu
 *      Generates a 64 x 64 five-point Poisson grid (n = 4096).
 *
 *   2) Read Matrix Market:
 *        mpirun -np 4 ./test_slu matrix.mtx
 *
 *   3) Generate another Poisson problem:
 *        mpirun -np 4 ./test_slu --poisson 256
 *
 *   4) Optional SuperLU_DIST process grid:
 *        mpirun -np 4 ./test_slu matrix.mtx 2 2
 *        mpirun -np 4 ./test_slu --poisson 256 2 2
 *
 * The manufactured exact solution is x*=1 and b=A*x*.
 * Matrix input is the zero-based full CSR produced by mtx_reader.h.
 */

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "superlu_ddefs.h"
#include "mtx_reader.h"

namespace {

constexpr int DEFAULT_POISSON_SIDE = 64;
constexpr double TEST_TOL = 1.0e-8;

struct InputSpec {
    bool use_file = false;
    std::string path;
    int poisson_side = DEFAULT_POISSON_SIDE;
    int next_arg = 1;
};

[[noreturn]] void fail(MPI_Comm comm, int rank, const char* message)
{
    if (rank == 0) {
        std::fprintf(stderr, "Error: %s\n", message);
        std::fflush(stderr);
    }
    MPI_Abort(comm, EXIT_FAILURE);
    std::abort();
}

int parse_positive_int(const char* text, const char* name,
                       MPI_Comm comm, int rank)
{
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (text == end || *end != '\0' || value <= 0 ||
        value > std::numeric_limits<int>::max()) {
        if (rank == 0)
            std::fprintf(stderr, "Error: invalid %s: %s\n", name, text);
        MPI_Abort(comm, EXIT_FAILURE);
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

InputSpec parse_input(int argc, char** argv, MPI_Comm comm, int rank)
{
    InputSpec spec;
    if (argc == 1) return spec;

    if (std::strcmp(argv[1], "--poisson") == 0) {
        if (argc < 3)
            fail(comm, rank, "--poisson requires a positive side length");
        spec.poisson_side =
            parse_positive_int(argv[2], "Poisson side", comm, rank);
        spec.next_arg = 3;
        return spec;
    }

    if (std::strcmp(argv[1], "--matrix") == 0) {
        if (argc < 3)
            fail(comm, rank, "--matrix requires a Matrix Market path");
        spec.use_file = true;
        spec.path = argv[2];
        spec.next_arg = 3;
        return spec;
    }

    if (is_positive_integer(argv[1])) {
        spec.poisson_side =
            parse_positive_int(argv[1], "Poisson side", comm, rank);
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
    if (n64 > INT_MAX || nnz64 > INT_MAX) {
        std::fprintf(stderr,
                     "Poisson matrix exceeds the int CSR range: "
                     "n=%lld, nnz=%lld\n", n64, nnz64);
        return false;
    }

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
        std::fprintf(stderr,
                     "Internal Poisson generation error: %d != %lld\n",
                     position, nnz64);
        mtx_free(mat);
        return false;
    }
    return true;
}

MPI_Datatype mpi_int_t_type()
{
    if (sizeof(int_t) == sizeof(int)) return MPI_INT;
    if (sizeof(int_t) == sizeof(long)) return MPI_LONG;
    if (sizeof(int_t) == sizeof(long long)) return MPI_LONG_LONG;

    std::fprintf(stderr, "Unsupported sizeof(int_t)=%zu\n", sizeof(int_t));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    return MPI_DATATYPE_NULL;
}

int checked_mpi_count(int_t value, MPI_Comm comm, int rank,
                      const char* what)
{
    if (value < 0 ||
        value > static_cast<int_t>(std::numeric_limits<int>::max())) {
        if (rank == 0)
            std::fprintf(stderr, "%s exceeds the MPI count range\n", what);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

int_t local_row_count(int_t n, int rank, int size)
{
    return n / size + (rank < n % size ? 1 : 0);
}

int_t first_local_row(int_t n, int rank, int size)
{
    const int_t base = n / size;
    const int_t remainder = n % size;
    return static_cast<int_t>(rank) * base +
           (rank < remainder ? rank : remainder);
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

    const InputSpec input = parse_input(argc, argv, MPI_COMM_WORLD, rank);

    const int remaining = argc - input.next_arg;
    if (remaining != 0 && remaining != 2) {
        if (rank == 0) {
            std::fprintf(stderr,
                "Usage:\n"
                "  mpirun -np P ./test_slu\n"
                "  mpirun -np P ./test_slu matrix.mtx [nprow npcol]\n"
                "  mpirun -np P ./test_slu --matrix matrix.mtx "
                "[nprow npcol]\n"
                "  mpirun -np P ./test_slu --poisson side "
                "[nprow npcol]\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int dims[2] = {0, 0};
    if (remaining == 2) {
        dims[0] = parse_positive_int(argv[input.next_arg], "nprow",
                                     MPI_COMM_WORLD, rank);
        dims[1] = parse_positive_int(argv[input.next_arg + 1], "npcol",
                                     MPI_COMM_WORLD, rank);
    } else {
        MPI_Dims_create(size, 2, dims);
    }
    if (dims[0] * dims[1] != size)
        fail(MPI_COMM_WORLD, rank,
             "nprow*npcol must equal the number of MPI ranks");

    MtxMatrix global_matrix;
    mtx_init(&global_matrix);
    std::string matrix_name;

    int64_t metadata[3] = {0, 0, 0};
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

        if (!loaded)
            fail(MPI_COMM_WORLD, rank, "failed to create/read the matrix");
        if (global_matrix.nrows != global_matrix.ncols)
            fail(MPI_COMM_WORLD, rank, "matrix must be square");
        if (global_matrix.nrows <= 0 || global_matrix.nnz <= 0)
            fail(MPI_COMM_WORLD, rank, "matrix is empty");

        metadata[0] = static_cast<int64_t>(global_matrix.nrows);
        metadata[1] = static_cast<int64_t>(global_matrix.ncols);
        metadata[2] = static_cast<int64_t>(global_matrix.nnz);
    }
    MPI_Bcast(metadata, 3, MPI_INT64_T, 0, MPI_COMM_WORLD);

    if (metadata[0] >
            static_cast<int64_t>(std::numeric_limits<int_t>::max()) ||
        metadata[2] >
            static_cast<int64_t>(std::numeric_limits<int_t>::max())) {
        fail(MPI_COMM_WORLD, rank,
             "matrix exceeds this SuperLU_DIST int_t build");
    }

    const int_t n = static_cast<int_t>(metadata[0]);
    const int_t global_nnz = static_cast<int_t>(metadata[2]);
    if (n < size)
        fail(MPI_COMM_WORLD, rank,
             "matrix order must be at least the number of MPI ranks");

    const MPI_Datatype mpi_index_type = mpi_int_t_type();
    int_t m_loc = local_row_count(n, rank, size);
    int_t fst_row = first_local_row(n, rank, size);
    int_t nnz_loc = 0;

    int_t* rowptr = nullptr;
    int_t* colind = nullptr;
    double* values = nullptr;
    double* rhs = nullptr;

    if (rank == 0) {
        double* global_rhs = mtx_compute_rhs_one(&global_matrix);
        if (!global_rhs)
            fail(MPI_COMM_WORLD, rank, "failed to construct b=A*1");

        for (int destination = 0; destination < size; ++destination) {
            const int_t destination_rows =
                local_row_count(n, destination, size);
            const int_t destination_first =
                first_local_row(n, destination, size);

            int_t destination_nnz = 0;
            for (int_t i = destination_first;
                 i < destination_first + destination_rows; ++i) {
                destination_nnz += static_cast<int_t>(
                    global_matrix.rowptr[i + 1] -
                    global_matrix.rowptr[i]);
            }

            int_t* d_rowptr = intMalloc_dist(destination_rows + 1);
            int_t* d_colind = intMalloc_dist(destination_nnz);
            double* d_values = doubleMalloc_dist(destination_nnz);
            double* d_rhs = doubleMalloc_dist(destination_rows);
            if (!d_rowptr || !d_colind || !d_values || !d_rhs)
                fail(MPI_COMM_WORLD, rank,
                     "distributed matrix allocation failed");

            d_rowptr[0] = 0;
            int_t position = 0;
            for (int_t local_row = 0;
                 local_row < destination_rows; ++local_row) {
                const int_t global_row = destination_first + local_row;
                for (int p = global_matrix.rowptr[global_row];
                     p < global_matrix.rowptr[global_row + 1]; ++p) {
                    d_colind[position] =
                        static_cast<int_t>(global_matrix.colind[p]);
                    d_values[position] = global_matrix.values[p];
                    ++position;
                }
                d_rhs[local_row] = global_rhs[global_row];
                d_rowptr[local_row + 1] = position;
            }

            if (destination == 0) {
                m_loc = destination_rows;
                fst_row = destination_first;
                nnz_loc = destination_nnz;
                rowptr = d_rowptr;
                colind = d_colind;
                values = d_values;
                rhs = d_rhs;
            } else {
                const int_t local_meta[3] = {
                    destination_rows, destination_first, destination_nnz
                };
                MPI_Send(local_meta, 3, mpi_index_type,
                         destination, 100, MPI_COMM_WORLD);
                MPI_Send(d_rowptr,
                         checked_mpi_count(destination_rows + 1,
                                           MPI_COMM_WORLD, rank,
                                           "rowptr length"),
                         mpi_index_type, destination, 101, MPI_COMM_WORLD);
                MPI_Send(d_colind,
                         checked_mpi_count(destination_nnz,
                                           MPI_COMM_WORLD, rank, "local nnz"),
                         mpi_index_type, destination, 102, MPI_COMM_WORLD);
                MPI_Send(d_values,
                         checked_mpi_count(destination_nnz,
                                           MPI_COMM_WORLD, rank, "local nnz"),
                         MPI_DOUBLE, destination, 103, MPI_COMM_WORLD);
                MPI_Send(d_rhs,
                         checked_mpi_count(destination_rows,
                                           MPI_COMM_WORLD, rank, "local rows"),
                         MPI_DOUBLE, destination, 104, MPI_COMM_WORLD);

                SUPERLU_FREE(d_rowptr);
                SUPERLU_FREE(d_colind);
                SUPERLU_FREE(d_values);
                SUPERLU_FREE(d_rhs);
            }
        }
        std::free(global_rhs);
    } else {
        int_t local_meta[3] = {0, 0, 0};
        MPI_Recv(local_meta, 3, mpi_index_type, 0, 100,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        m_loc = local_meta[0];
        fst_row = local_meta[1];
        nnz_loc = local_meta[2];

        rowptr = intMalloc_dist(m_loc + 1);
        colind = intMalloc_dist(nnz_loc);
        values = doubleMalloc_dist(nnz_loc);
        rhs = doubleMalloc_dist(m_loc);
        if (!rowptr || !colind || !values || !rhs)
            fail(MPI_COMM_WORLD, rank,
                 "distributed matrix allocation failed");

        MPI_Recv(rowptr,
                 checked_mpi_count(m_loc + 1, MPI_COMM_WORLD,
                                   rank, "rowptr length"),
                 mpi_index_type, 0, 101,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(colind,
                 checked_mpi_count(nnz_loc, MPI_COMM_WORLD,
                                   rank, "local nnz"),
                 mpi_index_type, 0, 102,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(values,
                 checked_mpi_count(nnz_loc, MPI_COMM_WORLD,
                                   rank, "local nnz"),
                 MPI_DOUBLE, 0, 103,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(rhs,
                 checked_mpi_count(m_loc, MPI_COMM_WORLD,
                                   rank, "local rows"),
                 MPI_DOUBLE, 0, 104,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    superlu_dist_options_t options;
    set_default_options_dist(&options);
    options.IterRefine = SLU_DOUBLE;
    options.PrintStat = NO;

    int gpu_count = 0;
    int selected_gpu = -1;
    int local_rank = 0;
    int local_size = 1;
    int gpu_offload = 0;

#ifdef GPU_ACC
    MPI_Comm local_comm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                        rank, MPI_INFO_NULL, &local_comm);
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_size);

    gpuGetDeviceCount(&gpu_count);
    const char* sharing_env = std::getenv("SLU_ALLOW_GPU_SHARING");
    const bool allow_sharing =
        sharing_env && std::atoi(sharing_env) != 0;

    gpu_offload =
        gpu_count > 0 && (local_size <= gpu_count || allow_sharing);
    options.superlu_acc_offload = gpu_offload ? 1 : 0;

    if (gpu_offload) {
        selected_gpu = local_rank % gpu_count;
        gpuSetDevice(selected_gpu);
        gpuFree(0);
    }
    MPI_Comm_free(&local_comm);
#else
    options.superlu_acc_offload = 0;
#endif

    gridinfo_t grid;
    superlu_gridinit(MPI_COMM_WORLD, dims[0], dims[1], &grid);

    SuperMatrix A;
    dCreate_CompRowLoc_Matrix_dist(
        &A, n, n, nnz_loc, m_loc, fst_row,
        values, colind, rowptr,
        SLU_NR_loc, SLU_D, SLU_GE);

    dScalePermstruct_t scale_perm;
    dLUstruct_t lu;
    dSOLVEstruct_t solve;
    SuperLUStat_t stat;
    std::memset(&solve, 0, sizeof(solve));

    dScalePermstructInit(n, n, &scale_perm);
    dLUstructInit(n, &lu);
    PStatInit(&stat);

    double* backward_error = doubleMalloc_dist(1);
    double* x = doubleMalloc_dist(m_loc);
    if (!backward_error || !x)
        fail(MPI_COMM_WORLD, rank, "solver allocation failed");
    for (int_t i = 0; i < m_loc; ++i) x[i] = rhs[i];

    int major = 0, minor = 0, patch = 0;
    superlu_dist_GetVersionNumber(&major, &minor, &patch);

    if (grid.iam == 0) {
        std::printf("========================================\n");
        std::printf("Solver          : SuperLU_DIST\n");
        std::printf("Version         : %d.%d.%d\n", major, minor, patch);
        std::printf("Input           : %s\n", matrix_name.c_str());
        std::printf("Matrix size     : %lld x %lld\n",
                    static_cast<long long>(n),
                    static_cast<long long>(n));
        std::printf("Global nnz      : %lld\n",
                    static_cast<long long>(global_nnz));
        std::printf("MPI ranks       : %d (%d x %d)\n",
                    size, dims[0], dims[1]);
        std::printf("MPI thread      : requested MULTIPLE, provided %d\n",
                    provided);
#ifdef GPU_ACC
        std::printf("GPU offload     : %s\n",
                    gpu_offload ? "enabled" : "disabled");
        if (!gpu_offload && gpu_count > 0 && local_size > gpu_count)
            std::printf("GPU note        : set SLU_ALLOW_GPU_SHARING=1 "
                        "to share GPUs between local ranks\n");
#else
        std::printf("GPU offload     : unavailable in this build\n");
#endif
        std::printf("========================================\n");
        std::fflush(stdout);
    }

#ifdef GPU_ACC
    if (gpu_offload) {
        std::printf("MPI rank %d uses GPU %d\n", rank, selected_gpu);
        std::fflush(stdout);
    }
#endif

    MPI_Barrier(grid.comm);
    const double wall_begin = MPI_Wtime();

    int info = 0;
    pdgssvx(&options, &A, &scale_perm, x, m_loc, 1,
            &grid, &lu, &solve, backward_error, &stat, &info);

    MPI_Barrier(grid.comm);
    const double wall_end = MPI_Wtime();

    const double local_total = wall_end - wall_begin;

    /*
     * SuperLU_DIST records many phases, not only ETREE/FACT/SOLVE.
     * Some fields are nested diagnostic timers (for example COMM is part of
     * FACT, and the solve sub-timers are part of SOLVE), so all NPHASES
     * entries must not be summed directly.
     *
     * For a meaningful critical-path breakdown, first find the MPI rank with
     * the largest pdgssvx wall time, then report that same rank's phase times.
     * Separately retain the maximum value of every phase across all ranks to
     * expose load imbalance without mixing maxima from different ranks.
     */
    std::vector<double> phase_max(static_cast<size_t>(NPHASES), 0.0);
    MPI_Reduce(stat.utime, phase_max.data(), NPHASES, MPI_DOUBLE,
               MPI_MAX, 0, grid.comm);

    std::vector<double> all_wall_times;
    if (grid.iam == 0)
        all_wall_times.resize(static_cast<size_t>(size), 0.0);
    MPI_Gather(&local_total, 1, MPI_DOUBLE,
               grid.iam == 0 ? all_wall_times.data() : nullptr,
               1, MPI_DOUBLE, 0, grid.comm);

    int critical_rank = 0;
    double total_time = 0.0;
    if (grid.iam == 0) {
        for (int r = 0; r < size; ++r) {
            if (all_wall_times[static_cast<size_t>(r)] > total_time) {
                total_time = all_wall_times[static_cast<size_t>(r)];
                critical_rank = r;
            }
        }
    }
    MPI_Bcast(&critical_rank, 1, MPI_INT, 0, grid.comm);
    MPI_Bcast(&total_time, 1, MPI_DOUBLE, 0, grid.comm);

    std::vector<double> critical_phase(static_cast<size_t>(NPHASES), 0.0);
    if (grid.iam == critical_rank) {
        for (int p = 0; p < NPHASES; ++p)
            critical_phase[static_cast<size_t>(p)] = stat.utime[p];
    }
    MPI_Bcast(critical_phase.data(), NPHASES, MPI_DOUBLE,
              critical_rank, grid.comm);

    if (n > std::numeric_limits<int>::max())
        fail(MPI_COMM_WORLD, rank,
             "solution verification requires n <= INT_MAX");

    std::vector<int> recv_counts(size);
    std::vector<int> displacements(size);
    for (int r = 0; r < size; ++r) {
        recv_counts[r] =
            checked_mpi_count(local_row_count(n, r, size),
                              MPI_COMM_WORLD, rank, "local rows");
        displacements[r] =
            checked_mpi_count(first_local_row(n, r, size),
                              MPI_COMM_WORLD, rank, "row displacement");
    }

    std::vector<double> x_global;
    if (rank == 0) x_global.resize(static_cast<size_t>(n));

    MPI_Gatherv(x,
                checked_mpi_count(m_loc, MPI_COMM_WORLD,
                                  rank, "local rows"),
                MPI_DOUBLE,
                rank == 0 ? x_global.data() : nullptr,
                recv_counts.data(), displacements.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double relative_residual =
        std::numeric_limits<double>::infinity();
    double relative_error =
        std::numeric_limits<double>::infinity();

    if (rank == 0 && info == 0) {
        double* global_rhs = mtx_compute_rhs_one(&global_matrix);
        if (!global_rhs)
            fail(MPI_COMM_WORLD, rank, "verification rhs allocation failed");
        relative_residual =
            mtx_relative_residual(&global_matrix, x_global.data(), global_rhs);
        relative_error =
            mtx_relative_error(&global_matrix, x_global.data());
        std::free(global_rhs);
    }

    int local_bad_info = info != 0 ? 1 : 0;
    int any_bad_info = 0;
    MPI_Allreduce(&local_bad_info, &any_bad_info, 1,
                  MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    int failed = any_bad_info;
    if (rank == 0) {
        failed = failed ||
                 !std::isfinite(relative_residual) ||
                 !std::isfinite(relative_error) ||
                 relative_residual > TEST_TOL ||
                 relative_error > TEST_TOL;
    }
    MPI_Bcast(&failed, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (grid.iam == 0) {
        const auto phase = [&](int p) -> double {
            return critical_phase[static_cast<size_t>(p)];
        };
        const auto phase_maximum = [&](int p) -> double {
            return phase_max[static_cast<size_t>(p)];
        };
        const auto percent = [&](double seconds) -> double {
            return total_time > 0.0 ? 100.0 * seconds / total_time : 0.0;
        };
        const auto print_phase = [&](const char* name, int p) {
            std::printf("%-27s %12.6f %9.2f%% %14.6f\n",
                        name, phase(p), percent(phase(p)), phase_maximum(p));
        };

        /*
         * Primary phases used for a conservative accounted-time estimate.
         * SOLVE and REFINE can overlap because refinement invokes triangular
         * solves, so use their maximum rather than adding both. Diagnostic
         * sub-phases such as COMM/SOL_COMM/SOL_GEMM/SOL_TRSM are excluded.
         */
        const double ordering_time =
            phase(ROWPERM) + phase(COLPERM) +
            phase(RELAX) + phase(ETREE);
        const double analysis_time =
            phase(EQUIL) + ordering_time +
            phase(SYMBFAC) + phase(DIST);
        const double solve_refine_envelope =
            std::max(phase(SOLVE), phase(REFINE));
        const double accounted_time =
            analysis_time + phase(FACT) + solve_refine_envelope +
            phase(RCOND);
        const double other_time =
            std::max(0.0, total_time - accounted_time);

        std::printf("\n--- Results ---\n");
        std::printf("pdgssvx info       : %d\n", info);
        std::printf("Critical MPI rank  : %d\n", critical_rank);
        std::printf("Total wall time    : %.6f s\n", total_time);
        std::printf("Analysis/setup     : %.6f s\n", analysis_time);
        std::printf("  Ordering total   : %.6f s\n", ordering_time);
        std::printf("Factorization      : %.6f s\n", phase(FACT));
        std::printf("Solve (raw)        : %.6f s\n", phase(SOLVE));
        std::printf("Refinement (raw)   : %.6f s\n", phase(REFINE));
        std::printf("Driver/other est.  : %.6f s (%.2f%%)\n",
                    other_time, percent(other_time));

        std::printf("\n--- SuperLU_DIST top-level timing ---\n");
        std::printf("%-27s %12s %10s %14s\n",
                    "Phase", "critical(s)", "% wall", "rank max(s)");
        print_phase("Equilibration", EQUIL);
        print_phase("Row permutation", ROWPERM);
        print_phase("Column permutation", COLPERM);
        print_phase("Relaxed supernodes", RELAX);
        print_phase("Elimination tree", ETREE);
        print_phase("Symbolic factorization", SYMBFAC);
        print_phase("Matrix distribution", DIST);
        print_phase("Numerical factorization", FACT);
        print_phase("Triangular solve", SOLVE);
        print_phase("Iterative refinement", REFINE);
        print_phase("Condition estimate", RCOND);
        print_phase("Forward-error estimate", FERR);

        std::printf("\n--- Overlapping diagnostic sub-phases (not added) ---\n");
        std::printf("%-27s %12s %10s %14s\n",
                    "Phase", "critical(s)", "% wall", "rank max(s)");
        print_phase("Factor communication", COMM);
        print_phase("  Diagonal broadcast", COMM_DIAG);
        print_phase("  L-panel communication", COMM_RIGHT);
        print_phase("  U-panel communication", COMM_DOWN);
        print_phase("Solve communication", SOL_COMM);
        print_phase("Solve GEMM", SOL_GEMM);
        print_phase("Solve TRSM", SOL_TRSM);
        print_phase("Solve internal total", SOL_TOT);
        print_phase("Factor TRSV subset", TRSV);
        print_phase("Factor GEMV subset", GEMV);

        std::printf("\nTiming note        : raw timers are not all disjoint; ");
        std::printf("Driver/other is a conservative estimate.\n");
        std::printf("Backward error     : %.12e\n", backward_error[0]);
        std::printf("Relative residual  : %.12e\n", relative_residual);
        std::printf("Relative x error   : %.12e\n", relative_error);
        std::printf("TEST               : %s\n",
                    failed ? "FAILED" : "PASSED");
        std::fflush(stdout);
    }

    SUPERLU_FREE(x);
    SUPERLU_FREE(backward_error);
    SUPERLU_FREE(rhs);

    Destroy_CompRowLoc_Matrix_dist(&A);
    dScalePermstructFree(&scale_perm);
    dDestroy_LU(n, &grid, &lu);
    dLUstructFree(&lu);
    if (options.SolveInitialized)
        dSolveFinalize(&options, &solve);
    PStatFree(&stat);

    if (rank == 0) mtx_free(&global_matrix);

    superlu_gridexit(&grid);
    MPI_Finalize();
    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
