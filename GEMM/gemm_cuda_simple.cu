#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#define CUDA_CHECK(x) do {                                                     \
    cudaError_t e = (x);                                                       \
    if (e != cudaSuccess) {                                                    \
        std::cerr << "CUDA error: " << cudaGetErrorString(e) << '\n';          \
        std::exit(1);                                                          \
    }                                                                          \
} while (0)

#define CUBLAS_CHECK(x) do {                                                   \
    if ((x) != CUBLAS_STATUS_SUCCESS) {                                        \
        std::cerr << "cuBLAS error\n";                                         \
        std::exit(1);                                                          \
    }                                                                          \
} while (0)

struct Matrix {
    int rows = 0;
    int cols = 0;

    // Column-major storage.
    std::vector<double> data;

    Matrix() = default;

    Matrix(int r, int c)
        : rows(r),
          cols(c),
          data(static_cast<size_t>(r) * c) {}
};

bool next_data_line(std::ifstream& in, std::string& line) {
    while (std::getline(in, line)) {
        size_t p = line.find_first_not_of(" \t\r\n");

        if (p != std::string::npos && line[p] != '%') {
            return true;
        }
    }

    return false;
}

// Supports common Matrix Market formats:
// coordinate/array and general/symmetric.
Matrix read_mtx(const std::string& path) {
    std::ifstream in(path);

    if (!in) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    std::string line;
    std::string banner;
    std::string object;
    std::string format;
    std::string field;
    std::string symmetry;

    std::getline(in, line);

    std::istringstream(line)
        >> banner
        >> object
        >> format
        >> field
        >> symmetry;

    if (banner != "%%MatrixMarket" || object != "matrix") {
        throw std::runtime_error(
            "Invalid Matrix Market file: " + path
        );
    }

    if (!next_data_line(in, line)) {
        throw std::runtime_error(
            "Missing matrix size information: " + path
        );
    }

    std::istringstream size_line(line);

    int rows = 0;
    int cols = 0;
    int nnz = 0;

    if (format == "coordinate") {
        size_line >> rows >> cols >> nnz;
    } else if (format == "array") {
        size_line >> rows >> cols;
    } else {
        throw std::runtime_error(
            "Only coordinate and array formats are supported"
        );
    }

    Matrix matrix(rows, cols);

    if (format == "coordinate") {
        for (int p = 0; p < nnz; ++p) {
            if (!next_data_line(in, line)) {
                throw std::runtime_error(
                    "Unexpected end of matrix data: " + path
                );
            }

            std::istringstream item(line);

            int row;
            int col;
            double value = 1.0;

            item >> row >> col;

            if (field != "pattern") {
                item >> value;
            }

            // Matrix Market indices start from 1.
            --row;
            --col;

            matrix.data[row + col * rows] += value;

            if (symmetry == "symmetric" && row != col) {
                matrix.data[col + row * rows] += value;
            }
        }
    } else {
        if (symmetry != "general") {
            throw std::runtime_error(
                "The array format currently supports only general matrices"
            );
        }

        for (double& value : matrix.data) {
            if (!next_data_line(in, line)) {
                throw std::runtime_error(
                    "Unexpected end of matrix data: " + path
                );
            }

            std::istringstream(line) >> value;
        }
    }

    return matrix;
}

Matrix random_matrix(int rows, int cols) {
    static std::mt19937 generator(1);

    static std::uniform_real_distribution<double> distribution(
        -1.0,
        1.0
    );

    Matrix matrix(rows, cols);

    for (double& value : matrix.data) {
        value = distribution(generator);
    }

    return matrix;
}

bool is_integer(const std::string& text) {
    if (text.empty()) {
        return false;
    }

    char* end = nullptr;

    std::strtol(
        text.c_str(),
        &end,
        10
    );

    return *end == '\0';
}

template <class Function>
float benchmark(Function function, int repeat) {
    // Warm-up runs.
    for (int i = 0; i < 5; ++i) {
        function();
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < repeat; ++i) {
        function();
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;

    CUDA_CHECK(
        cudaEventElapsedTime(
            &total_ms,
            start,
            stop
        )
    );

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / repeat;
}

void print_usage(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program << "\n"
        << "      Use random A and B with m = n = k = 1024\n\n"

        << "  " << program << " m n k [repeat]\n"
        << "      Use random matrices with custom dimensions\n\n"

        << "  " << program << " A.mtx [n] [repeat]\n"
        << "      Read A and randomly generate B with size k x n\n"
        << "      The default value of n is 1\n\n"

        << "  " << program << " A.mtx B.mtx [repeat]\n"
        << "      Read both A and B from Matrix Market files\n";
}

int main(int argc, char** argv) {
    try {
        int m = 1024;
        int n = 1024;
        int k = 1024;
        int repeat = 50;

        Matrix A;
        Matrix B;

        if (argc == 1) {
            A = random_matrix(m, k);
            B = random_matrix(k, n);
        } else if (is_integer(argv[1])) {
            if (argc < 4 || argc > 5) {
                print_usage(argv[0]);
                return 1;
            }

            m = std::stoi(argv[1]);
            n = std::stoi(argv[2]);
            k = std::stoi(argv[3]);

            if (argc == 5) {
                repeat = std::stoi(argv[4]);
            }

            A = random_matrix(m, k);
            B = random_matrix(k, n);
        } else {
            A = read_mtx(argv[1]);

            m = A.rows;
            k = A.cols;

            if (argc == 2) {
                n = 1;
                B = random_matrix(k, n);
            } else if (is_integer(argv[2])) {
                n = std::stoi(argv[2]);

                if (argc >= 4) {
                    repeat = std::stoi(argv[3]);
                }

                B = random_matrix(k, n);
            } else {
                B = read_mtx(argv[2]);
                n = B.cols;

                if (argc >= 4) {
                    repeat = std::stoi(argv[3]);
                }
            }
        }

        if (m <= 0 || n <= 0 || k <= 0 || repeat <= 0) {
            throw std::runtime_error(
                "m, n, k, and repeat must be greater than zero"
            );
        }

        if (A.cols != B.rows) {
            throw std::runtime_error(
                "Matrix dimension mismatch: A.cols must equal B.rows"
            );
        }

        const size_t bytes_A =
            A.data.size() * sizeof(double);

        const size_t bytes_B =
            B.data.size() * sizeof(double);

        const size_t bytes_C =
            static_cast<size_t>(m) * n * sizeof(double);

        double* d_A = nullptr;
        double* d_B = nullptr;
        double* d_C = nullptr;

        CUDA_CHECK(
            cudaMalloc(
                reinterpret_cast<void**>(&d_A),
                bytes_A
            )
        );

        CUDA_CHECK(
            cudaMalloc(
                reinterpret_cast<void**>(&d_B),
                bytes_B
            )
        );

        CUDA_CHECK(
            cudaMalloc(
                reinterpret_cast<void**>(&d_C),
                bytes_C
            )
        );

        CUDA_CHECK(
            cudaMemcpy(
                d_A,
                A.data.data(),
                bytes_A,
                cudaMemcpyHostToDevice
            )
        );

        CUDA_CHECK(
            cudaMemcpy(
                d_B,
                B.data.data(),
                bytes_B,
                cudaMemcpyHostToDevice
            )
        );

        cublasHandle_t handle;

        CUBLAS_CHECK(
            cublasCreate(&handle)
        );

        const double alpha = 1.0;
        const double beta = 0.0;

        float average_ms = benchmark(
            [&] {
                CUBLAS_CHECK(
                    cublasDgemm(
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
                        d_C,
                        m
                    )
                );
            },
            repeat
        );

        double gflops =
            2.0 *
            static_cast<double>(m) *
            static_cast<double>(n) *
            static_cast<double>(k) /
            (average_ms * 1.0e6);

        std::vector<double> C(
            static_cast<size_t>(m) * n
        );

        CUDA_CHECK(
            cudaMemcpy(
                C.data(),
                d_C,
                bytes_C,
                cudaMemcpyDeviceToHost
            )
        );

        long double checksum = 0.0;

        for (double value : C) {
            checksum += std::abs(value);
        }

        std::cout
            << "A: " << m << " x " << k << '\n';

        std::cout
            << "B: " << k << " x " << n << '\n';

        std::cout
            << "C: " << m << " x " << n << '\n';

        std::cout
            << "Repeat: " << repeat << '\n';

        std::cout
            << "Average time: "
            << average_ms
            << " ms\n";

        std::cout
            << "Performance: "
            << gflops
            << " GFLOP/s\n";

        std::cout
            << "Checksum: "
            << static_cast<double>(checksum)
            << '\n';

        CUBLAS_CHECK(
            cublasDestroy(handle)
        );

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    } catch (const std::exception& error) {
        std::cerr
            << "ERROR: "
            << error.what()
            << '\n';

        return 1;
    }

    return 0;
}
