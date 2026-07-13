#include "gs.cuh"
#include <iostream>

// Example only: A = tridiag(-1, 4, -1), exact solution x_i = i + 1.
static void build_example_system(CSRMatrix& A,
                                 std::vector<double>& b,
                                 std::vector<double>& exact,
                                 int n)
{
    A.n = n;
    A.row_ptr.resize(n + 1, 0);
    A.col_idx.clear();
    A.values.clear();
    A.col_idx.reserve(3 * n);
    A.values.reserve(3 * n);

    exact.resize(n);
    for (int i = 0; i < n; ++i) {
        exact[i] = static_cast<double>(i + 1);
    }

    for (int i = 0; i < n; ++i) {
        A.row_ptr[i] = static_cast<int>(A.col_idx.size());
        if (i > 0) {
            A.col_idx.push_back(i - 1);
            A.values.push_back(-1.0);
        }
        A.col_idx.push_back(i);
        A.values.push_back(4.0);
        if (i + 1 < n) {
            A.col_idx.push_back(i + 1);
            A.values.push_back(-1.0);
        }
    }
    A.row_ptr[n] = static_cast<int>(A.col_idx.size());

    b.assign(n, 0.0);
    for (int row = 0; row < n; ++row) {
        for (int jj = A.row_ptr[row]; jj < A.row_ptr[row + 1]; ++jj) {
            b[row] += A.values[jj] * exact[A.col_idx[jj]];
        }
    }
}

int main(int argc, char** argv)
{
    try {
        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count == 0) {
            std::cerr << "No CUDA device found.\n";
            return EXIT_FAILURE;
        }
        CUDA_CHECK(cudaSetDevice(0));

        const int n = (argc >= 2) ? std::stoi(argv[1]) : 1000000;
        if (n <= 0) {
            throw std::invalid_argument("n must be positive.");
        }

        CSRMatrix A;
        std::vector<double> b;
        std::vector<double> exact;
        build_example_system(A, b, exact, n);

        std::vector<double> x(n, 0.0);
        SolverOptions options;
        options.max_iterations = 10000;
        options.residual_check_interval = 1;
        options.relative_tolerance = 1.0e-10;
        options.omega = 1.1;
        options.block_size = 256;
        options.warp_row_threshold = 16;

        const SolverResult result =
            solve_multicolor_gauss_seidel_gpu(A, b, x, options);

        double max_abs_error = 0.0;
        for (int i = 0; i < n; ++i) {
            max_abs_error = std::max(max_abs_error, std::abs(x[i] - exact[i]));
        }

        std::cout << "n                  = " << n << '\n'
                  << "colors             = " << result.num_colors << '\n'
                  << "iterations          = " << result.iterations << '\n'
                  << "relative residual   = " << result.relative_residual << '\n'
                  << "max absolute error  = " << max_abs_error << '\n'
                  << "GPU iteration time  = " << result.elapsed_ms << " ms\n";

        return (result.relative_residual <= options.relative_tolerance)
                   ? EXIT_SUCCESS
                   : EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
