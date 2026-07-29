#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <tuple>
#include <vector>

#define CUDA_CHECK(x) do {                                                     \
    cudaError_t e = (x);                                                       \
    if (e != cudaSuccess) {                                                    \
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e));        \
        std::exit(EXIT_FAILURE);                                               \
    }                                                                          \
} while (0)

#define CUBLAS_CHECK(x) do {                                                   \
    cublasStatus_t s = (x);                                                    \
    if (s != CUBLAS_STATUS_SUCCESS) {                                          \
        std::fprintf(stderr, "cuBLAS error: %d\n", static_cast<int>(s));        \
        std::exit(EXIT_FAILURE);                                               \
    }                                                                          \
} while (0)

using LaunchFunction = void (*)(
    int, int, int,
    const double*, const double*, double*);

struct KernelConfig {
    int TM;
    int TN;
    int BK;
    int threads;
    int BM;
    int BN;
    LaunchFunction launch;
};

/*
 * Column-major FP64 GEMM:
 *
 *     C(m x n) = A(m x k) * B(k x n)
 *
 * Only TM, TN, BK, and THREADS are independent parameters.
 *
 * The following values are derived automatically:
 *
 *     BM = 32 * TM
 *     BN = (THREADS / 32) * TN
 *     number of A/B loads per thread
 *     shared-memory dimensions
 *     grid dimensions
 */
template<int TM, int TN, int BK, int THREADS>
__global__ void gemm_kernel(
    int m,
    int n,
    int k,
    const double* __restrict__ A,
    const double* __restrict__ B,
    double* __restrict__ C)
{
    static_assert(THREADS % 32 == 0,
                  "THREADS must be a multiple of 32");

    constexpr int WARPS = THREADS / 32;
    constexpr int BM = 32 * TM;
    constexpr int BN = WARPS * TN;

    constexpr int A_TILE_ELEMENTS = BM * BK;
    constexpr int B_TILE_ELEMENTS = BN * BK;

    constexpr int A_LOADS =
        (A_TILE_ELEMENTS + THREADS - 1) / THREADS;

    constexpr int B_LOADS =
        (B_TILE_ELEMENTS + THREADS - 1) / THREADS;

    __shared__ double As[2][BK][BM + 1];
    __shared__ double Bs[2][BN][BK + 1];

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    const int local_col = warp * TN;

    double accumulator[TM][TN] = {};
    double next_A[A_LOADS];
    double next_B[B_LOADS];

    /*
     * Load the first K tile.
     *
     * The loop bounds are derived automatically, so changing TM, TN,
     * BK, or THREADS does not require changing this code.
     */
#pragma unroll
    for (int q = 0; q < A_LOADS; ++q) {
        const int index = tid + q * THREADS;

        if (index < A_TILE_ELEMENTS) {
            const int p = index / BM;
            const int row = index % BM;

            const int global_row = block_row + row;
            const int global_k = p;

            As[0][p][row] =
                (global_row < m && global_k < k)
                    ? A[global_row + global_k * m]
                    : 0.0;
        }
    }

#pragma unroll
    for (int q = 0; q < B_LOADS; ++q) {
        const int index = tid + q * THREADS;

        if (index < B_TILE_ELEMENTS) {
            const int col = index / BK;
            const int p = index % BK;

            const int global_col = block_col + col;
            const int global_k = p;

            Bs[0][col][p] =
                (global_col < n && global_k < k)
                    ? B[global_k + global_col * k]
                    : 0.0;
        }
    }

    __syncthreads();

    int buffer = 0;

    for (int base = 0; base < k; base += BK) {
        const int next_base = base + BK;
        const bool has_next = next_base < k;

        /*
         * Prefetch the next tile into registers.
         */
        if (has_next) {
#pragma unroll
            for (int q = 0; q < A_LOADS; ++q) {
                const int index = tid + q * THREADS;

                if (index < A_TILE_ELEMENTS) {
                    const int p = index / BM;
                    const int row = index % BM;

                    const int global_row = block_row + row;
                    const int global_k = next_base + p;

                    next_A[q] =
                        (global_row < m && global_k < k)
                            ? A[global_row + global_k * m]
                            : 0.0;
                }
            }

#pragma unroll
            for (int q = 0; q < B_LOADS; ++q) {
                const int index = tid + q * THREADS;

                if (index < B_TILE_ELEMENTS) {
                    const int col = index / BK;
                    const int p = index % BK;

                    const int global_col = block_col + col;
                    const int global_k = next_base + p;

                    next_B[q] =
                        (global_col < n && global_k < k)
                            ? B[global_k + global_col * k]
                            : 0.0;
                }
            }
        }

        /*
         * Register-blocked outer-product computation.
         */
#pragma unroll
        for (int p = 0; p < BK; ++p) {
            double a_fragment[TM];
            double b_fragment[TN];

#pragma unroll
            for (int i = 0; i < TM; ++i) {
                a_fragment[i] =
                    As[buffer][p][lane + i * 32];
            }

#pragma unroll
            for (int j = 0; j < TN; ++j) {
                b_fragment[j] =
                    Bs[buffer][local_col + j][p];
            }

#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    accumulator[i][j] =
                        fma(a_fragment[i],
                            b_fragment[j],
                            accumulator[i][j]);
                }
            }
        }

        /*
         * Store the prefetched tile into the alternate shared buffer.
         */
        if (has_next) {
            const int next_buffer = buffer ^ 1;

#pragma unroll
            for (int q = 0; q < A_LOADS; ++q) {
                const int index = tid + q * THREADS;

                if (index < A_TILE_ELEMENTS) {
                    const int p = index / BM;
                    const int row = index % BM;

                    As[next_buffer][p][row] = next_A[q];
                }
            }

#pragma unroll
            for (int q = 0; q < B_LOADS; ++q) {
                const int index = tid + q * THREADS;

                if (index < B_TILE_ELEMENTS) {
                    const int col = index / BK;
                    const int p = index % BK;

                    Bs[next_buffer][col][p] = next_B[q];
                }
            }

            __syncthreads();
            buffer = next_buffer;
        }
    }

    /*
     * Coalesced column-major output.
     */
#pragma unroll
    for (int j = 0; j < TN; ++j) {
        const int col = block_col + local_col + j;

        if (col < n) {
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                const int row = block_row + lane + i * 32;

                if (row < m) {
                    C[row + col * m] = accumulator[i][j];
                }
            }
        }
    }
}

template<int TM, int TN, int BK, int THREADS>
void launch_config(
    int m,
    int n,
    int k,
    const double* A,
    const double* B,
    double* C)
{
    constexpr int BM = 32 * TM;
    constexpr int BN = (THREADS / 32) * TN;

    const dim3 block(THREADS);
    const dim3 grid(
        (n + BN - 1) / BN,
        (m + BM - 1) / BM);

    gemm_kernel<TM, TN, BK, THREADS>
        <<<grid, block>>>(m, n, k, A, B, C);

    CUDA_CHECK(cudaGetLastError());
}

template<int TM, int TN, int BK, int THREADS>
KernelConfig make_config()
{
    return {
        TM,
        TN,
        BK,
        THREADS,
        32 * TM,
        (THREADS / 32) * TN,
        &launch_config<TM, TN, BK, THREADS>
    };
}

/*
 * ================================================================
 * USER-EDITABLE AUTOTUNING CANDIDATES
 * ================================================================
 *
 * Add or remove candidates only in this list.
 *
 * Format:
 *
 *     make_config<TM, TN, BK, THREADS>()
 *
 * BM, BN, load counts, shared-memory sizes, and launch dimensions
 * are all derived automatically.
 */
static const KernelConfig CONFIGS[] = {
    make_config<4, 4, 8, 256>(),
    make_config<4, 8, 8, 256>(),
    make_config<2, 8, 8, 256>(),
    make_config<2, 4, 8, 256>(),

    make_config<4, 4, 4, 256>(),
    make_config<4, 4, 16, 256>(),
    make_config<2, 8, 16, 256>(),

    // Candidates for narrow matrices.
    make_config<4, 4, 8, 128>(),
    make_config<4, 2, 8, 128>(),
    make_config<2, 4, 8, 128>()
};

constexpr int CONFIG_COUNT =
    static_cast<int>(sizeof(CONFIGS) / sizeof(CONFIGS[0]));

template<class Function>
float benchmark(Function launch, int warmup, int repeat)
{
    for (int i = 0; i < warmup; ++i) {
        launch();
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < repeat; ++i) {
        launch();
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &total_ms,
        start,
        stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / repeat;
}

/*
 * Autotune once for each exact (m, n, k) shape and cache the result.
 *
 * This cache lives for the duration of the process. A solver that
 * repeatedly encounters the same dense-block shape will tune it only once.
 */
int select_best_config(
    int m,
    int n,
    int k,
    const double* A,
    const double* B,
    double* C,
    int tune_repeat)
{
    static std::map<std::tuple<int, int, int>, int> cache;

    const auto key = std::make_tuple(m, n, k);
    const auto found = cache.find(key);

    if (found != cache.end()) {
        return found->second;
    }

    int best_index = 0;
    float best_ms = std::numeric_limits<float>::max();

    std::printf("Autotuning custom GEMM candidates:\n");

    for (int index = 0; index < CONFIG_COUNT; ++index) {
        const KernelConfig& config = CONFIGS[index];

        const float ms = benchmark(
            [&] {
                config.launch(m, n, k, A, B, C);
            },
            2,
            tune_repeat);

        std::printf(
            "  [%2d] TM=%d TN=%d BK=%d threads=%d "
            "tile=%dx%d : %8.3f ms\n",
            index,
            config.TM,
            config.TN,
            config.BK,
            config.threads,
            config.BM,
            config.BN,
            ms);

        if (ms < best_ms) {
            best_ms = ms;
            best_index = index;
        }
    }

    cache[key] = best_index;

    const KernelConfig& best = CONFIGS[best_index];

    std::printf(
        "Selected: TM=%d TN=%d BK=%d threads=%d "
        "tile=%dx%d\n\n",
        best.TM,
        best.TN,
        best.BK,
        best.threads,
        best.BM,
        best.BN);

    return best_index;
}

int main(int argc, char** argv)
{
    const int m =
        argc > 1 ? std::atoi(argv[1]) : 1024;

    const int n =
        argc > 2 ? std::atoi(argv[2]) : 1024;

    const int k =
        argc > 3 ? std::atoi(argv[3]) : 1024;

    const int repeat =
        argc > 4 ? std::atoi(argv[4]) : 100;

    const int tune_repeat =
        argc > 5 ? std::atoi(argv[5]) : 5;

    if (m <= 0 || n <= 0 || k <= 0 ||
        repeat <= 0 || tune_repeat <= 0) {
        std::fprintf(
            stderr,
            "Usage: %s [m n k repeat tune_repeat]\n",
            argv[0]);

        return EXIT_FAILURE;
    }

    std::vector<double> A(
        static_cast<size_t>(m) * k);

    std::vector<double> B(
        static_cast<size_t>(k) * n);

    std::vector<double> C_custom(
        static_cast<size_t>(m) * n);

    std::vector<double> C_cublas(
        static_cast<size_t>(m) * n);

    for (size_t i = 0; i < A.size(); ++i) {
        A[i] =
            (static_cast<int>(i % 100) - 50) * 0.01;
    }

    for (size_t i = 0; i < B.size(); ++i) {
        B[i] =
            (static_cast<int>(i % 80) - 40) * 0.01;
    }

    double* d_A = nullptr;
    double* d_B = nullptr;
    double* d_C_custom = nullptr;
    double* d_C_cublas = nullptr;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_A),
        A.size() * sizeof(double)));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_B),
        B.size() * sizeof(double)));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_C_custom),
        C_custom.size() * sizeof(double)));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_C_cublas),
        C_cublas.size() * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(
        d_A,
        A.data(),
        A.size() * sizeof(double),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_B,
        B.data(),
        B.size() * sizeof(double),
        cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const double alpha = 1.0;
    const double beta = 0.0;

    /*
     * Automatically choose the fastest precompiled custom kernel.
     */
    const int best_index = select_best_config(
        m,
        n,
        k,
        d_A,
        d_B,
        d_C_custom,
        tune_repeat);

    const KernelConfig& best = CONFIGS[best_index];

    /*
     * Correctness comparison against cuBLAS.
     */
    best.launch(
        m,
        n,
        k,
        d_A,
        d_B,
        d_C_custom);

    CUBLAS_CHECK(cublasDgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m,
        n,
        k,
        &alpha,
        d_A,
        m,
        d_B,
        k,
        &beta,
        d_C_cublas,
        m));

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        C_custom.data(),
        d_C_custom,
        C_custom.size() * sizeof(double),
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
        C_cublas.data(),
        d_C_cublas,
        C_cublas.size() * sizeof(double),
        cudaMemcpyDeviceToHost));

    long double difference_squared = 0.0;
    long double reference_squared = 0.0;

    for (size_t i = 0; i < C_custom.size(); ++i) {
        const long double difference =
            static_cast<long double>(C_custom[i]) -
            static_cast<long double>(C_cublas[i]);

        difference_squared += difference * difference;

        reference_squared +=
            static_cast<long double>(C_cublas[i]) *
            static_cast<long double>(C_cublas[i]);
    }

    const double relative_error =
        static_cast<double>(
            std::sqrt(difference_squared) /
            (std::sqrt(reference_squared) + 1.0e-30L));

    /*
     * Final performance measurement.
     */
    const float custom_ms = benchmark(
        [&] {
            best.launch(
                m,
                n,
                k,
                d_A,
                d_B,
                d_C_custom);
        },
        5,
        repeat);

    const float cublas_ms = benchmark(
        [&] {
            CUBLAS_CHECK(cublasDgemm(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                m,
                n,
                k,
                &alpha,
                d_A,
                m,
                d_B,
                k,
                &beta,
                d_C_cublas,
                m));
        },
        5,
        repeat);

    const double operations =
        2.0 *
        static_cast<double>(m) *
        static_cast<double>(n) *
        static_cast<double>(k);

    std::printf(
        "m=%d n=%d k=%d repeat=%d\n",
        m,
        n,
        k,
        repeat);

    std::printf(
        "best custom configuration: "
        "TM=%d TN=%d BK=%d threads=%d tile=%dx%d\n",
        best.TM,
        best.TN,
        best.BK,
        best.threads,
        best.BM,
        best.BN);

    std::printf(
        "custom : %8.3f ms  %10.2f GFLOP/s\n",
        custom_ms,
        operations / (custom_ms * 1.0e6));

    std::printf(
        "cuBLAS : %8.3f ms  %10.2f GFLOP/s\n",
        cublas_ms,
        operations / (cublas_ms * 1.0e6));

    std::printf(
        "custom/cuBLAS performance ratio: %.3f\n",
        cublas_ms / custom_ms);

    std::printf(
        "relative error: %.3e\n",
        relative_error);

    CUBLAS_CHECK(cublasDestroy(handle));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_custom));
    CUDA_CHECK(cudaFree(d_C_cublas));

    return EXIT_SUCCESS;
}
