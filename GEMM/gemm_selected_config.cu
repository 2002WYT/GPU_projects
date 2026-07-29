/*
 * gemm_selected_config.cu
 *
 * Purpose
 * -------
 * Run one GEMM configuration selected by a previous autotuning run.
 *
 * The program computes column-major FP64 GEMM:
 *
 *     C(m x n) = A(m x k) * B(k x n)
 *
 * The tuning parameters are passed on the command line:
 *
 *     TM       output rows computed by each thread
 *     TN       output columns computed by each thread
 *     BK       K-direction shared-memory tile size
 *     THREADS  threads per CUDA block
 *
 * BM, BN, shared-memory sizes, load counts, and grid dimensions are
 * derived automatically. They must not be entered manually:
 *
 *     BM = 32 * TM
 *     BN = (THREADS / 32) * TN
 *
 * Modes
 * -----
 * 1. Compare the selected custom GEMM with cuBLAS:
 *
 *    ./gemm_selected compare m n k repeat TM TN BK THREADS
 *
 *    Example:
 *
 *    ./gemm_selected compare 4096 4096 4096 20 2 8 8 256
 *
 *    This mode reports custom performance, cuBLAS performance, and
 *    the relative error of the custom result.
 *
 * 2. Run only the selected custom GEMM:
 *
 *    ./gemm_selected custom m n k repeat TM TN BK THREADS
 *
 *    Example:
 *
 *    ./gemm_selected custom 4096 4096 4096 20 2 8 8 256
 *
 * 3. Show all precompiled parameter combinations:
 *
 *    ./gemm_selected list
 *
 * Important
 * ---------
 * Template parameters must be compiled in advance. A command-line
 * combination is accepted only when it appears in GEMM_CONFIGS below.
 * To support another combination, add one X(TM,TN,BK,THREADS) line
 * and rebuild the program. No other code needs to be changed.
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#define CUDA_CHECK(call) do {                                                  \
    cudaError_t status = (call);                                               \
    if (status != cudaSuccess) {                                               \
        std::fprintf(stderr, "CUDA error: %s\n",                               \
                     cudaGetErrorString(status));                              \
        std::exit(EXIT_FAILURE);                                               \
    }                                                                          \
} while (0)

#define CUBLAS_CHECK(call) do {                                                \
    cublasStatus_t status = (call);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
        std::fprintf(stderr, "cuBLAS error: %d\n",                             \
                     static_cast<int>(status));                                \
        std::exit(EXIT_FAILURE);                                               \
    }                                                                          \
} while (0)

/*
 * Add or remove supported configurations only here.
 *
 * Format:
 *
 *     X(TM, TN, BK, THREADS)
 */
#define GEMM_CONFIGS(X)                                                        \
    X(2, 4,  8, 256)                                                          \
    X(2, 8,  8, 256)                                                          \
    X(4, 4,  8, 256)                                                          \
    X(4, 8,  8, 256)                                                          \
    X(4, 4,  4, 256)                                                          \
    X(4, 4, 16, 256)                                                          \
    X(2, 8, 16, 256)                                                          \
    X(4, 4,  8, 128)                                                          \
    X(4, 2,  8, 128)                                                          \
    X(2, 4,  8, 128)

using LaunchFunction = void (*)(
    int,
    int,
    int,
    const double*,
    const double*,
    double*);

struct KernelConfig {
    int tm;
    int tn;
    int bk;
    int threads;
    int bm;
    int bn;
    LaunchFunction launch;
};

template<int TM, int TN, int BK, int THREADS>
__global__ __launch_bounds__(THREADS)
void gemm_kernel(
    int m,
    int n,
    int k,
    const double* __restrict__ A,
    const double* __restrict__ B,
    double* __restrict__ C)
{
    static_assert(TM > 0 && TN > 0 && BK > 0,
                  "TM, TN, and BK must be positive");
    static_assert(THREADS >= 32 && THREADS <= 1024,
                  "THREADS must be between 32 and 1024");
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

    // Load the first A tile.
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
                    ? __ldg(A + global_row + global_k * m)
                    : 0.0;
        }
    }

    // Load the first B tile.
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
                    ? __ldg(B + global_k + global_col * k)
                    : 0.0;
        }
    }

    __syncthreads();

    int buffer = 0;

    for (int base = 0; base < k; base += BK) {
        const int next_base = base + BK;
        const bool has_next = next_base < k;

        // Prefetch the next A/B tile into registers.
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
                            ? __ldg(A + global_row + global_k * m)
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
                            ? __ldg(B + global_k + global_col * k)
                            : 0.0;
                }
            }
        }

        // Compute the current tile.
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

        // Move the prefetched data into the alternate shared buffer.
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

    // Store C in column-major order.
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
void launch_gemm(
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
    return KernelConfig{
        TM,
        TN,
        BK,
        THREADS,
        32 * TM,
        (THREADS / 32) * TN,
        &launch_gemm<TM, TN, BK, THREADS>
    };
}

#define MAKE_CONFIG(TM, TN, BK, THREADS) \
    make_config<TM, TN, BK, THREADS>(),

static const KernelConfig CONFIGS[] = {
    GEMM_CONFIGS(MAKE_CONFIG)
};

#undef MAKE_CONFIG

constexpr int CONFIG_COUNT =
    static_cast<int>(sizeof(CONFIGS) / sizeof(CONFIGS[0]));

const KernelConfig* find_config(
    int tm,
    int tn,
    int bk,
    int threads)
{
    for (int i = 0; i < CONFIG_COUNT; ++i) {
        const KernelConfig& config = CONFIGS[i];

        if (config.tm == tm &&
            config.tn == tn &&
            config.bk == bk &&
            config.threads == threads) {
            return &config;
        }
    }

    return nullptr;
}

void print_configs()
{
    std::printf("Supported custom GEMM configurations:\n");

    for (int i = 0; i < CONFIG_COUNT; ++i) {
        const KernelConfig& config = CONFIGS[i];

        std::printf(
            "  TM=%d TN=%d BK=%d threads=%d tile=%dx%d\n",
            config.tm,
            config.tn,
            config.bk,
            config.threads,
            config.bm,
            config.bn);
    }
}

void print_usage(const char* program)
{
    std::printf(
        "Usage:\n"
        "  %s list\n"
        "  %s compare m n k repeat TM TN BK THREADS\n"
        "  %s custom  m n k repeat TM TN BK THREADS\n\n"
        "Examples:\n"
        "  %s compare 4096 4096 4096 20 2 8 8 256\n"
        "  %s custom  4096 4096 4096 20 2 8 8 256\n",
        program,
        program,
        program,
        program,
        program);
}

template<class Function>
float benchmark(
    Function launch,
    int warmup,
    int repeat)
{
    for (int i = 0; i < warmup; ++i) {
        launch();
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

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

double calculate_relative_error(
    const std::vector<double>& custom_result,
    const std::vector<double>& reference_result)
{
    long double difference_squared = 0.0L;
    long double reference_squared = 0.0L;

    for (size_t i = 0; i < custom_result.size(); ++i) {
        const long double difference =
            static_cast<long double>(custom_result[i]) -
            static_cast<long double>(reference_result[i]);

        difference_squared += difference * difference;

        reference_squared +=
            static_cast<long double>(reference_result[i]) *
            static_cast<long double>(reference_result[i]);
    }

    return static_cast<double>(
        std::sqrt(difference_squared) /
        (std::sqrt(reference_squared) + 1.0e-30L));
}

int main(int argc, char** argv)
{
    if (argc == 2 && std::strcmp(argv[1], "list") == 0) {
        print_configs();
        return EXIT_SUCCESS;
    }

    if (argc != 10) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    const bool compare_mode =
        std::strcmp(argv[1], "compare") == 0;

    const bool custom_mode =
        std::strcmp(argv[1], "custom") == 0;

    if (!compare_mode && !custom_mode) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    const int m = std::atoi(argv[2]);
    const int n = std::atoi(argv[3]);
    const int k = std::atoi(argv[4]);
    const int repeat = std::atoi(argv[5]);

    const int tm = std::atoi(argv[6]);
    const int tn = std::atoi(argv[7]);
    const int bk = std::atoi(argv[8]);
    const int threads = std::atoi(argv[9]);

    if (m <= 0 || n <= 0 || k <= 0 || repeat <= 0) {
        std::fprintf(
            stderr,
            "m, n, k, and repeat must be positive.\n");

        return EXIT_FAILURE;
    }

    const KernelConfig* config =
        find_config(tm, tn, bk, threads);

    if (config == nullptr) {
        std::fprintf(
            stderr,
            "Unsupported configuration: "
            "TM=%d TN=%d BK=%d threads=%d\n\n",
            tm,
            tn,
            bk,
            threads);

        print_configs();

        std::fprintf(
            stderr,
            "\nAdd the configuration to GEMM_CONFIGS and rebuild.\n");

        return EXIT_FAILURE;
    }

    const size_t a_elements =
        static_cast<size_t>(m) * static_cast<size_t>(k);

    const size_t b_elements =
        static_cast<size_t>(k) * static_cast<size_t>(n);

    const size_t c_elements =
        static_cast<size_t>(m) * static_cast<size_t>(n);

    std::vector<double> A(a_elements);
    std::vector<double> B(b_elements);

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
        a_elements * sizeof(double)));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_B),
        b_elements * sizeof(double)));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_C_custom),
        c_elements * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(
        d_A,
        A.data(),
        a_elements * sizeof(double),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_B,
        B.data(),
        b_elements * sizeof(double),
        cudaMemcpyHostToDevice));

    if (compare_mode) {
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_C_cublas),
            c_elements * sizeof(double)));
    }

    const double operations =
        2.0 *
        static_cast<double>(m) *
        static_cast<double>(n) *
        static_cast<double>(k);

    const float custom_ms = benchmark(
        [&] {
            config->launch(
                m,
                n,
                k,
                d_A,
                d_B,
                d_C_custom);
        },
        5,
        repeat);

    std::printf(
        "mode=%s\n"
        "m=%d n=%d k=%d repeat=%d\n"
        "TM=%d TN=%d BK=%d threads=%d tile=%dx%d\n",
        compare_mode ? "compare" : "custom",
        m,
        n,
        k,
        repeat,
        config->tm,
        config->tn,
        config->bk,
        config->threads,
        config->bm,
        config->bn);

    std::printf(
        "custom : %8.3f ms  %10.2f GFLOP/s\n",
        custom_ms,
        operations / (custom_ms * 1.0e6));

    if (compare_mode) {
        cublasHandle_t handle = nullptr;
        CUBLAS_CHECK(cublasCreate(&handle));

        const double alpha = 1.0;
        const double beta = 0.0;

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

        std::vector<double> C_custom(c_elements);
        std::vector<double> C_cublas(c_elements);

        CUDA_CHECK(cudaMemcpy(
            C_custom.data(),
            d_C_custom,
            c_elements * sizeof(double),
            cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaMemcpy(
            C_cublas.data(),
            d_C_cublas,
            c_elements * sizeof(double),
            cudaMemcpyDeviceToHost));

        const double relative_error =
            calculate_relative_error(
                C_custom,
                C_cublas);

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
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_custom));

    if (d_C_cublas != nullptr) {
        CUDA_CHECK(cudaFree(d_C_cublas));
    }

    return EXIT_SUCCESS;
}
