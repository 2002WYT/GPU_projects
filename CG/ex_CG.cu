#include "head.cuh"
#include "CG.cuh"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

int main(int argc, char** argv)
{
    try
    {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            std::cerr << "No CUDA device found.\n";
            return EXIT_FAILURE;
        }
        cudaSetDevice(0);
        const int n = (argc >= 2) ? std::stoi(argv[1]) : 1000000;
        if (n <= 0) {
            throw std::invalid_argument("n must be positive.");
        }

        CSRMatrix A;
        A.n = n;
        A.row_ptr.resize(n + 1);
        A.col_idx.reserve(3 * n);
        A.values.reserve(3 * n);
        for (int row = 0; row < n; ++row)
        {
            A.row_ptr[row] =
                static_cast<int>(A.col_idx.size());
            if (row > 0)
            {
                A.col_idx.push_back(row - 1);
                A.values.push_back(-1.0);
            }
            A.col_idx.push_back(row);
            A.values.push_back(4.0);
            if (row < n - 1)
            {
                A.col_idx.push_back(row + 1);
                A.values.push_back(-1.0);
            }
        }
        A.row_ptr[n] = static_cast<int>(A.col_idx.size());

        std::vector<double> x_exact(n);

        for (int i = 0; i < n; ++i)
        {
            x_exact[i] =
                static_cast<double>(i + 1);
        }
        std::vector<double> b(n, 0.0);

        for (int row = 0; row < n; ++row)
        {
            double sum = 0.0;
            for (int jj = A.row_ptr[row];
                 jj < A.row_ptr[row + 1];
                 ++jj)
            {
                const int col = A.col_idx[jj];
                sum +=
                    A.values[jj] * x_exact[col];
            }
            b[row] = sum;
        }

        std::vector<double> x(n, 0.0);

        SolverOptions options;
        options.max_iterations = 1000;
        options.residual_check_interval = 10;
        options.relative_tolerance = 1e-10;
        options.block_size = 256;

        const int warps_per_block = options.block_size / 32;
        options.grid_size = (n + warps_per_block - 1) /
            warps_per_block;

        std::cout
            << "========================================\n"
            << "GPU Conjugate Gradient Solver\n"
            << "n   = " << n << '\n'
            << "nnz = " << A.values.size() << '\n'
            << "========================================\n";

        const int status = CG_gpu(A, b, x, options);

        if (status != 0)
        {
            std::cerr
                << "CG_gpu failed with status = "
                << status
                << std::endl;
            return 1;
        }

        double max_abs_error = 0.0;
        double error_sq = 0.0;
        double exact_sq = 0.0;
        for (int i = 0; i < n; ++i)
        {
            const double error =
                x[i] - x_exact[i];
            max_abs_error =
                std::max(
                    max_abs_error,
                    std::abs(error));
            error_sq += error * error;
            exact_sq += x_exact[i] * x_exact[i];
        }
        const double relative_solution_error =
            std::sqrt(error_sq / exact_sq);

        std::cout
            << std::scientific
            << std::setprecision(12);
        std::cout
            << "\n========================================\n"
            << "Solution check\n"
            << "Maximum absolute error = "
            << max_abs_error
            << '\n'
            << "Relative solution error = "
            << relative_solution_error
            << '\n'
            << "========================================\n";


        /*
         * 输出解的前 10 个分量。
         */
        const int print_count =
            std::min(n, 10);

        std::cout
            << "\nFirst "
            << print_count
            << " entries:\n";

        for (int i = 0; i < print_count; ++i)
        {
            std::cout
                << "x[" << i << "] = "
                << x[i]
                << ", exact = "
                << x_exact[i]
                << ", error = "
                << std::abs(x[i] - x_exact[i])
                << '\n';
        }
    }
    catch (const std::exception& error)
    {
        std::cerr
            << "Error: "
            << error.what()
            << std::endl;
        return 1;
    }
    return 0;
}
