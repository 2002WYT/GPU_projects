// A program of Jacobi iteration method on GPU
// Jacobi:
// x_new[i] = (b[i] - sum_{j != i} Aij * x_old[j]) / Aii

#include "head.cuh"
#include "jacobi.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

int main(int argc, char* argv[])
{
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));

    if (device_count == 0)
    {
        std::cerr << "No CUDA device found." << std::endl;
        return 1;
    }

    CHECK_CUDA(cudaSetDevice(0));

    int n = 10000;

    if (argc >= 2)
    {
        n = std::atoi(argv[1]);
    }

    if (n <= 0)
    {
        std::cerr << "Error: n must be positive." << std::endl;
        return 1;
    }

    /*
     * ========================================================
     * 1. 在 CPU 上构造 CSR 矩阵
     * ========================================================
     */

    std::vector<int> row_ptr(n + 1, 0);
    std::vector<int> col_idx;
    std::vector<double> values;

    col_idx.reserve(3 * n);
    values.reserve(3 * n);

    // 构造 A = tridiag(-1, 4, -1)
    for (int i = 0; i < n; i++)
    {
        row_ptr[i] = static_cast<int>(col_idx.size());

        if (i > 0)
        {
            col_idx.push_back(i - 1);
            values.push_back(-1.0);
        }

        col_idx.push_back(i);
        values.push_back(4.0);

        if (i < n - 1)
        {
            col_idx.push_back(i + 1);
            values.push_back(-1.0);
        }
    }

    row_ptr[n] = static_cast<int>(col_idx.size());

    int nnz = static_cast<int>(values.size());

    /*
     * ========================================================
     * 2. 设置真实解并构造 b = A*x_true
     * ========================================================
     */

    // 设置真实解 x_true = [1, 2, ..., n]
    std::vector<double> x_true(n);

    for (int i = 0; i < n; i++)
    {
        x_true[i] = static_cast<double>(i + 1);
    }

    // 构造 b = A*x_true
    std::vector<double> b(n, 0.0);

    for (int i = 0; i < n; i++)
    {
        for (int jj = row_ptr[i];
             jj < row_ptr[i + 1];
             jj++)
        {
            int col = col_idx[jj];
            b[i] += values[jj] * x_true[col];
        }
    }

    /*
     * ========================================================
     * 3. 设置 Jacobi 参数
     * ========================================================
     */

    int max_iter = 100;
    double tol = 1e-6;

    // 用于接收最终结果
    std::vector<double> x_jacobi(n, 0.0);

    std::cout
        << "--------------------------------------------------"
        << std::endl;
    std::cout << "GPU Jacobi Example" << std::endl;
    std::cout
        << "Solving A x = b, A = tridiag(-1, 4, -1)"
        << std::endl;
    std::cout
        << "n = " << n
        << ", nnz = " << nnz
        << std::endl;
    std::cout
        << "--------------------------------------------------"
        << std::endl;

    /*
     * ========================================================
     * 4. 申请 GPU 内存
     * ========================================================
     */

    int* d_row_ptr = nullptr;
    int* d_col_idx = nullptr;

    double* d_values = nullptr;
    double* d_b = nullptr;

    double* d_x0 = nullptr;
    double* d_x1 = nullptr;
    double* d_x_result = nullptr;

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&d_row_ptr),
        (n + 1) * sizeof(int)));

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&d_col_idx),
        nnz * sizeof(int)));

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&d_values),
        nnz * sizeof(double)));

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&d_b),
        n * sizeof(double)));

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&d_x0),
        n * sizeof(double)));

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&d_x1),
        n * sizeof(double)));

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&d_x_result),
        n * sizeof(double)));

    /*
     * ========================================================
     * 5. 把 CSR 矩阵和 b 复制到 GPU
     * ========================================================
     */

    CHECK_CUDA(cudaMemcpy(
        d_row_ptr,
        row_ptr.data(),
        (n + 1) * sizeof(int),
        cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(
        d_col_idx,
        col_idx.data(),
        nnz * sizeof(int),
        cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(
        d_values,
        values.data(),
        nnz * sizeof(double),
        cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(
        d_b,
        b.data(),
        n * sizeof(double),
        cudaMemcpyHostToDevice));

    /*
     * 初始解 x0 = 0。
     *
     * x1 和 x_result 也先清零，虽然 kernel 会覆盖它们，
     * 这样做更安全，也便于调试。
     */
    CHECK_CUDA(cudaMemset(
        d_x0,
        0,
        n * sizeof(double)));

    CHECK_CUDA(cudaMemset(
        d_x1,
        0,
        n * sizeof(double)));

    CHECK_CUDA(cudaMemset(
        d_x_result,
        0,
        n * sizeof(double)));

    /*
     * ========================================================
     * 6. 调用 GPU Jacobi 求解器
     * ========================================================
     */

    JacobiState state{};

    CHECK_CUDA(jacobi_gpu_solve(
        n,
        d_row_ptr,
        d_col_idx,
        d_values,
        d_b,
        d_x0,
        d_x1,
        d_x_result,
        max_iter,
        tol,
        &state));

    /*
     * ========================================================
     * 7. 将最终结果复制回 CPU
     * ========================================================
     */

    CHECK_CUDA(cudaMemcpy(
        x_jacobi.data(),
        d_x_result,
        n * sizeof(double),
        cudaMemcpyDeviceToHost));

    /*
     * ========================================================
     * 8. 输出迭代信息
     * ========================================================
     */

    std::cout << "Jacobi iterations       = "
              << state.iterations << std::endl;

    std::cout << "Relative residual       = "
              << state.relative_residual << std::endl;

    std::cout << "Converged               = "
              << (state.converged ? "Yes" : "No")
              << std::endl;

    std::cout << "Zero diagonal detected  = "
              << (state.zero_diagonal ? "Yes" : "No")
              << std::endl;

    /*
     * ========================================================
     * 9. 和真实解比较
     * ========================================================
     */

    double max_abs_error = 0.0;

    for (int i = 0; i < n; i++)
    {
        double error =
            std::abs(x_jacobi[i] - x_true[i]);

        max_abs_error =
            std::max(max_abs_error, error);
    }

    std::cout << "Maximum absolute error  = "
              << max_abs_error << std::endl;

    std::cout
        << "--------------------------------------------------"
        << std::endl;

    std::cout
        << "First 10 entries of Jacobi solution:"
        << std::endl;

    int print_count = std::min(n, 10);

    for (int i = 0; i < print_count; i++)
    {
        std::cout
            << "x_jacobi[" << i << "] = "
            << x_jacobi[i]
            << ", exact = "
            << x_true[i]
            << std::endl;
    }

    /*
     * ========================================================
     * 10. 释放 GPU 内存
     * ========================================================
     */

    CHECK_CUDA(cudaFree(d_row_ptr));
    CHECK_CUDA(cudaFree(d_col_idx));
    CHECK_CUDA(cudaFree(d_values));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_x0));
    CHECK_CUDA(cudaFree(d_x1));
    CHECK_CUDA(cudaFree(d_x_result));

    CHECK_CUDA(cudaDeviceReset());

    return 0;
}
