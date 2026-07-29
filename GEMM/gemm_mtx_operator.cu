/*
 * gemm_mtx_operator.cu
 *
 * Purpose
 * -------
 * Run the optimized custom FP64 GEMM kernel selected after autotuning.
 * Matrices use column-major storage and the operation is:
 *
 *     C(m x n) = A(m x k) * B(k x n)
 *
 * Input modes
 * -----------
 * 1. No Matrix Market files:
 *
 *    ./gemm_mtx --tm 2 --tn 8 --bk 8 --threads 256
 *
 *    The program generates deterministic A and B internally.
 *    Default dimensions: m=n=k=1024.
 *
 * 2. Custom generated dimensions:
 *
 *    ./gemm_mtx --m 4096 --n 4096 --k 4096 --repeat 20 \
 *               --tm 2 --tn 8 --bk 8 --threads 256
 *
 * 3. Read A.mtx and generate a compatible B:
 *
 *    ./gemm_mtx --a ../files/cfd1.mtx --n 64 --tm 2 --tn 8 --bk 8 --threads 256
 *               
 *
 *    If --n is omitted, B has one column.
 *
 * 4. Read both A.mtx and B.mtx:
 *
 *    ./gemm_mtx --a A.mtx --b B.mtx --tm 2 --tn 8 --bk 8 --threads 256
 *               
 *
 *    A.cols must equal B.rows.
 *
 * Optional features
 * -----------------
 * --compare       Also run cuBLAS and report performance and relative error.
 * --output C.mtx  Write the custom GEMM result as Matrix Market array format.
 * --list          Show all precompiled kernel configurations.
 *
 * Important
 * ---------
 * TM, TN, BK, and THREADS determine template sizes and must be compiled
 * in advance. Add a new X(TM,TN,BK,THREADS) entry to GEMM_CONFIGS and
 * rebuild if the requested combination is not already present.
 *
 * BM, BN, load counts, shared-memory sizes, and grid dimensions are
 * derived automatically:
 *
 *     BM = 32 * TM
 *     BN = (THREADS / 32) * TN
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
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

/* Add supported autotuning results here. */
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

struct Matrix {
    int rows = 0;
    int cols = 0;
    std::vector<double> data;  // Column-major: data[row + col * rows].

    Matrix() = default;

    Matrix(int r, int c)
        : rows(r),
          cols(c),
          data(static_cast<size_t>(r) * static_cast<size_t>(c), 0.0) {}
};

std::string lower_copy(std::string text)
{
    std::transform(
        text.begin(),
        text.end(),
        text.begin(),
        [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });

    return text;
}

bool next_data_line(std::ifstream& input, std::string& line)
{
    while (std::getline(input, line)) {
        const size_t first =
            line.find_first_not_of(" \t\r\n");

        if (first != std::string::npos &&
            line[first] != '%') {
            return true;
        }
    }

    return false;
}

/*
 * Supported Matrix Market variants:
 *
 *   coordinate / array
 *   real / integer / pattern
 *   general / symmetric / skew-symmetric
 *
 * Coordinate input is expanded into a dense column-major matrix because
 * GEMM operates on dense blocks.
 */
Matrix read_mtx(const std::string& path)
{
    std::ifstream input(path);

    if (!input) {
        throw std::runtime_error(
            "Cannot open Matrix Market file: " + path);
    }

    std::string line;
    std::string banner;
    std::string object;
    std::string format;
    std::string field;
    std::string symmetry;

    if (!std::getline(input, line)) {
        throw std::runtime_error(
            "Empty Matrix Market file: " + path);
    }

    std::istringstream(line)
        >> banner
        >> object
        >> format
        >> field
        >> symmetry;

    banner = lower_copy(banner);
    object = lower_copy(object);
    format = lower_copy(format);
    field = lower_copy(field);
    symmetry = lower_copy(symmetry);

    if (banner != "%%matrixmarket" ||
        object != "matrix") {
        throw std::runtime_error(
            "Invalid Matrix Market banner: " + path);
    }

    if (!next_data_line(input, line)) {
        throw std::runtime_error(
            "Missing Matrix Market size line: " + path);
    }

    int rows = 0;
    int cols = 0;
    int entries = 0;

    std::istringstream size_line(line);

    if (format == "coordinate") {
        size_line >> rows >> cols >> entries;
    } else if (format == "array") {
        size_line >> rows >> cols;
    } else {
        throw std::runtime_error(
            "Only coordinate and array formats are supported");
    }

    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error(
            "Matrix dimensions must be positive");
    }

    if (symmetry != "general" && rows != cols) {
        throw std::runtime_error(
            "Symmetric input must be square");
    }

    Matrix matrix(rows, cols);

    if (format == "coordinate") {
        for (int entry = 0; entry < entries; ++entry) {
            if (!next_data_line(input, line)) {
                throw std::runtime_error(
                    "Unexpected end of coordinate data: " + path);
            }

            std::istringstream item(line);

            int row = 0;
            int col = 0;
            double value = 1.0;

            item >> row >> col;

            if (field != "pattern") {
                item >> value;
            }

            --row;
            --col;

            if (row < 0 || row >= rows ||
                col < 0 || col >= cols) {
                throw std::runtime_error(
                    "Matrix Market index is out of range: " + path);
            }

            matrix.data[row + col * rows] += value;

            if (row != col && symmetry == "symmetric") {
                matrix.data[col + row * rows] += value;
            } else if (row != col &&
                       symmetry == "skew-symmetric") {
                matrix.data[col + row * rows] -= value;
            }
        }
    } else if (symmetry == "general") {
        for (double& value : matrix.data) {
            if (!next_data_line(input, line)) {
                throw std::runtime_error(
                    "Unexpected end of array data: " + path);
            }

            std::istringstream(line) >> value;
        }
    } else {
        /*
         * Matrix Market array symmetric storage contains the lower
         * triangle column by column. Skew-symmetric storage omits
         * diagonal entries.
         */
        for (int col = 0; col < cols; ++col) {
            const int first_row =
                symmetry == "skew-symmetric"
                    ? col + 1
                    : col;

            for (int row = first_row;
                 row < rows;
                 ++row) {
                if (!next_data_line(input, line)) {
                    throw std::runtime_error(
                        "Unexpected end of packed array data: " + path);
                }

                double value = 0.0;
                std::istringstream(line) >> value;

                matrix.data[row + col * rows] = value;

                if (row != col) {
                    matrix.data[col + row * rows] =
                        symmetry == "symmetric"
                            ? value
                            : -value;
                }
            }
        }
    }

    return matrix;
}

Matrix generate_matrix(int rows, int cols, unsigned int seed)
{
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    Matrix matrix(rows, cols);

    for (double& value : matrix.data) {
        value = distribution(generator);
    }

    return matrix;
}

void write_mtx(
    const std::string& path,
    const Matrix& matrix)
{
    std::ofstream output(path);

    if (!output) {
        throw std::runtime_error(
            "Cannot create output file: " + path);
    }

    output
        << "%%MatrixMarket matrix array real general\n"
        << matrix.rows << ' ' << matrix.cols << '\n';

    output.precision(17);

    for (double value : matrix.data) {
        output << value << '\n';
    }
}

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

void list_configs()
{
    std::printf("Supported configurations:\n");

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

struct Options {
    int m = 1024;
    int n = 1024;
    int k = 1024;
    int repeat = 100;

    int tm = 2;
    int tn = 8;
    int bk = 8;
    int threads = 256;

    bool compare = false;
    bool list = false;

    std::string a_path;
    std::string b_path;
    std::string output_path;
};

int parse_positive(
    const std::string& value,
    const std::string& option)
{
    const int parsed = std::stoi(value);

    if (parsed <= 0) {
        throw std::runtime_error(
            option + " must be positive");
    }

    return parsed;
}

Options parse_options(int argc, char** argv)
{
    Options options;

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];

        auto value_after =
            [&](const std::string& option) -> std::string {
                if (index + 1 >= argc) {
                    throw std::runtime_error(
                        "Missing value after " + option);
                }

                return argv[++index];
            };

        if (argument == "--help" ||
            argument == "-h") {
            std::printf(
                "Usage:\n"
                "  %s [options]\n\n"
                "Input:\n"
                "  --a FILE.mtx       Read A\n"
                "  --b FILE.mtx       Read B; requires --a\n"
                "  --m N --n N --k N  Generated dimensions\n\n"
                "Kernel:\n"
                "  --tm N --tn N --bk N --threads N\n\n"
                "Other:\n"
                "  --repeat N         Default 100\n"
                "  --compare          Compare with cuBLAS\n"
                "  --output FILE.mtx  Write custom result\n"
                "  --list             List configurations\n",
                argv[0]);

            std::exit(EXIT_SUCCESS);
        } else if (argument == "--a") {
            options.a_path = value_after(argument);
        } else if (argument == "--b") {
            options.b_path = value_after(argument);
        } else if (argument == "--m") {
            options.m =
                parse_positive(value_after(argument), argument);
        } else if (argument == "--n") {
            options.n =
                parse_positive(value_after(argument), argument);
        } else if (argument == "--k") {
            options.k =
                parse_positive(value_after(argument), argument);
        } else if (argument == "--repeat") {
            options.repeat =
                parse_positive(value_after(argument), argument);
        } else if (argument == "--tm") {
            options.tm =
                parse_positive(value_after(argument), argument);
        } else if (argument == "--tn") {
            options.tn =
                parse_positive(value_after(argument), argument);
        } else if (argument == "--bk") {
            options.bk =
                parse_positive(value_after(argument), argument);
        } else if (argument == "--threads") {
            options.threads =
                parse_positive(value_after(argument), argument);
        } else if (argument == "--compare") {
            options.compare = true;
        } else if (argument == "--output") {
            options.output_path = value_after(argument);
        } else if (argument == "--list") {
            options.list = true;
        } else {
            throw std::runtime_error(
                "Unknown option: " + argument);
        }
    }

    if (!options.b_path.empty() &&
        options.a_path.empty()) {
        throw std::runtime_error(
            "--b requires --a");
    }

    return options;
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

double relative_error(
    const std::vector<double>& result,
    const std::vector<double>& reference)
{
    long double difference_squared = 0.0L;
    long double reference_squared = 0.0L;

    for (size_t index = 0;
         index < result.size();
         ++index) {
        const long double difference =
            static_cast<long double>(result[index]) -
            static_cast<long double>(reference[index]);

        difference_squared += difference * difference;

        reference_squared +=
            static_cast<long double>(reference[index]) *
            static_cast<long double>(reference[index]);
    }

    return static_cast<double>(
        std::sqrt(difference_squared) /
        (std::sqrt(reference_squared) + 1.0e-30L));
}

int main(int argc, char** argv)
{
    try {
        const Options options =
            parse_options(argc, argv);

        if (options.list) {
            list_configs();
            return EXIT_SUCCESS;
        }

        const KernelConfig* config =
            find_config(
                options.tm,
                options.tn,
                options.bk,
                options.threads);

        if (config == nullptr) {
            std::fprintf(
                stderr,
                "Unsupported configuration: "
                "TM=%d TN=%d BK=%d threads=%d\n\n",
                options.tm,
                options.tn,
                options.bk,
                options.threads);

            list_configs();

            return EXIT_FAILURE;
        }

        Matrix A;
        Matrix B;
        std::string a_source;
        std::string b_source;

        if (options.a_path.empty()) {
            A = generate_matrix(
                options.m,
                options.k,
                1);

            B = generate_matrix(
                options.k,
                options.n,
                2);

            a_source = "generated";
            b_source = "generated";
        } else {
            A = read_mtx(options.a_path);
            a_source = options.a_path;

            if (options.b_path.empty()) {
                B = generate_matrix(
                    A.cols,
                    options.n,
                    2);

                b_source = "generated";
            } else {
                B = read_mtx(options.b_path);
                b_source = options.b_path;
            }
        }

        if (A.cols != B.rows) {
            throw std::runtime_error(
                "Dimension mismatch: A.cols must equal B.rows");
        }

        const int m = A.rows;
        const int n = B.cols;
        const int k = A.cols;

        const size_t a_elements = A.data.size();
        const size_t b_elements = B.data.size();
        const size_t c_elements =
            static_cast<size_t>(m) *
            static_cast<size_t>(n);

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
            A.data.data(),
            a_elements * sizeof(double),
            cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(
            d_B,
            B.data.data(),
            b_elements * sizeof(double),
            cudaMemcpyHostToDevice));

        if (options.compare) {
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_C_cublas),
                c_elements * sizeof(double)));
        }

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
            options.repeat);

        const double operations =
            2.0 *
            static_cast<double>(m) *
            static_cast<double>(n) *
            static_cast<double>(k);

        std::printf(
            "A source: %s\n"
            "B source: %s\n"
            "A=%dx%d B=%dx%d C=%dx%d repeat=%d\n"
            "TM=%d TN=%d BK=%d threads=%d tile=%dx%d\n",
            a_source.c_str(),
            b_source.c_str(),
            m,
            k,
            k,
            n,
            m,
            n,
            options.repeat,
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

        if (options.compare) {
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
                options.repeat);

            std::vector<double> custom_result(c_elements);
            std::vector<double> cublas_result(c_elements);

            CUDA_CHECK(cudaMemcpy(
                custom_result.data(),
                d_C_custom,
                c_elements * sizeof(double),
                cudaMemcpyDeviceToHost));

            CUDA_CHECK(cudaMemcpy(
                cublas_result.data(),
                d_C_cublas,
                c_elements * sizeof(double),
                cudaMemcpyDeviceToHost));

            const double error =
                relative_error(
                    custom_result,
                    cublas_result);

            std::printf(
                "cuBLAS : %8.3f ms  %10.2f GFLOP/s\n",
                cublas_ms,
                operations / (cublas_ms * 1.0e6));

            std::printf(
                "custom/cuBLAS performance ratio: %.3f\n",
                cublas_ms / custom_ms);

            std::printf(
                "relative error: %.3e\n",
                error);

            if (error > 1.0e-10) {
                std::fprintf(
                    stderr,
                    "WARNING: custom GEMM failed the accuracy check.\n");
            }

            CUBLAS_CHECK(cublasDestroy(handle));
        }

        if (!options.output_path.empty()) {
            Matrix result(m, n);

            CUDA_CHECK(cudaMemcpy(
                result.data.data(),
                d_C_custom,
                c_elements * sizeof(double),
                cudaMemcpyDeviceToHost));

            write_mtx(
                options.output_path,
                result);

            std::printf(
                "Output written to: %s\n",
                options.output_path.c_str());
        }

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C_custom));

        if (d_C_cublas != nullptr) {
            CUDA_CHECK(cudaFree(d_C_cublas));
        }

        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::fprintf(
            stderr,
            "ERROR: %s\n",
            error.what());

        return EXIT_FAILURE;
    }
}
