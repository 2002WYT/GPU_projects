#pragma once

#include <limits>
#include <vector>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t error__ = (call);                                            \
        if (error__ != cudaSuccess) {                                            \
            throw std::runtime_error(                                            \
                std::string("CUDA error at ") + __FILE__ + ":" +               \
                std::to_string(__LINE__) + ": " + cudaGetErrorString(error__)); \
        }                                                                       \
    } while (0)

struct CSRMatrix
{
    int n = 0;

    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<double> values;
};

struct SolverOptions
{
    // 最大迭代次数
    int max_iterations = 10000;

    // 每隔多少次迭代检查一次残差
    int residual_check_interval = 20;

    // 相对残差收敛容差
    double relative_tolerance = 1.0e-8;

    // omega = 1：Gauss-Seidel
    // 0 < omega < 2：SOR
    double omega = 1.0;

    // CUDA block 大小
    int block_size = 256;

    // 平均每行非对角元达到该值时，
    // 使用一个 warp 处理一行
    int warp_row_threshold = 16;
};

struct SolverResult
{
    // 实际迭代次数
    int iterations = 0;

    // 图着色使用的颜色数
    int num_colors = 0;

    // 最终相对残差
    double relative_residual =
        std::numeric_limits<double>::infinity();

    // GPU 迭代时间
    float elapsed_ms = 0.0f;
};

SolverResult solve_multicolor_gauss_seidel_gpu(
    const CSRMatrix& A,
    const std::vector<double>& b,
    std::vector<double>& x,
    const SolverOptions& options);


struct PreparedMatrix {
    int n = 0;
    std::vector<int> off_row_ptr;
    std::vector<int> off_col_idx;
    std::vector<double> off_values;
    std::vector<double> diagonal_inverse;
};
