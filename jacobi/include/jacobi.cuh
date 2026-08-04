#pragma once
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

/*
 * GPU 计算结束后返回的信息
 */
struct JacobiState
{
    double b_norm;              // ||b||_2
    double residual_sq;         // ||b-Ax||_2^2，内部工作变量
    double relative_residual;   // ||b-Ax||_2 / ||b||_2

    int iterations;             // 实际完成的 Jacobi 迭代次数
    int converged;              // 1：收敛，0：未收敛
    int stop;                   // GPU 内部停止标志
    int zero_diagonal;          // 是否遇到零对角元
};
cudaError_t jacobi_gpu_solve(
    int n,
    const int* d_row_ptr,
    const int* d_col_idx,
    const double* d_values,
    const double* d_b,
    double* d_x0,
    double* d_x1,
    double* d_x_result,
    int max_iter,
    double tol,
    JacobiState* host_state);

