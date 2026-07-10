#include <cuda_runtime.h>
#include <math_constants.h>
#include <iostream>
#include <cmath>
#include "jacobi.cuh"

/*
 * block 内求和归约。
 *
 * 要求 blockDim.x 是 2 的幂，例如 128、256、512。
 */
__device__ double block_reduce_sum(double value)
{
    extern __shared__ double shared_data[];

    int tid = threadIdx.x;

    shared_data[tid] = value;
    __syncthreads();

    for (int offset = blockDim.x / 2;
         offset > 0;
         offset >>= 1)
    {
        if (tid < offset)
        {
            shared_data[tid] += shared_data[tid + offset];
        }

        __syncthreads();
    }

    return shared_data[0];
}


/*
 * 完全在 GPU 上运行的 Jacobi 迭代。
 *
 * x0:
 *     初始解向量。
 *
 * x1:
 *     Jacobi 工作向量。
 *
 * x_result:
 *     最终解向量，可以与 x0 指向同一块显存。
 *
 * state:
 *     GPU 端状态信息。
 */
__global__ void jacobi_cooperative_kernel(
    int n,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const double* __restrict__ values,
    const double* __restrict__ b,
    double* x0,
    double* x1,
    double* x_result,
    int max_iter,
    double tol,
    JacobiState* state)
{
    cg::grid_group grid = cg::this_grid();

    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    /*
     * ========================================================
     * 1. 初始化 GPU 状态
     * ========================================================
     */
    if (grid.thread_rank() == 0)
    {
        state->b_norm = 0.0;
        state->residual_sq = 0.0;
        state->relative_residual = CUDART_INF;

        state->iterations = 0;
        state->converged = 0;
        state->stop = 0;
        state->zero_diagonal = 0;
    }

    grid.sync();

    /*
     * ========================================================
     * 2. 计算 ||b||_2
     * ========================================================
     */
    double local_sum = 0.0;

    for (int i = global_tid; i < n; i += stride)
    {
        local_sum += b[i] * b[i];
    }

    double block_sum = block_reduce_sum(local_sum);

    if (tid == 0)
    {
        atomicAdd(&state->b_norm, block_sum);
    }

    grid.sync();

    if (grid.thread_rank() == 0)
    {
        state->b_norm = sqrt(state->b_norm);
    }

    grid.sync();

    /*
     * ========================================================
     * 3. 计算初始相对残差
     *
     * relative_residual =
     *     ||b - A*x0||_2 / ||b||_2
     * ========================================================
     */
    if (grid.thread_rank() == 0)
    {
        state->residual_sq = 0.0;
    }

    grid.sync();

    local_sum = 0.0;

    for (int i = global_tid; i < n; i += stride)
    {
        double Ax_i = 0.0;

        for (int jj = row_ptr[i]; jj < row_ptr[i + 1]; ++jj)
        {
            int col = col_idx[jj];
            Ax_i += values[jj] * x0[col];
        }

        double r_i = b[i] - Ax_i;
        local_sum += r_i * r_i;
    }

    block_sum = block_reduce_sum(local_sum);

    if (tid == 0)
    {
        atomicAdd(&state->residual_sq, block_sum);
    }

    grid.sync();

    if (grid.thread_rank() == 0)
    {
        /*
         * 当 b=0 时，不能除以 ||b||。
         * 此时使用绝对残差作为判断依据。
         */
        double denominator =
            state->b_norm > 0.0 ? state->b_norm : 1.0;

        state->relative_residual =
            sqrt(state->residual_sq) / denominator;

        state->iterations = 0;

        if (state->relative_residual <= tol)
        {
            state->converged = 1;
            state->stop = 1;
        }
    }

    grid.sync();

    /*
     * 初始解已经满足收敛条件。
     */
    if (state->stop)
    {
        for (int i = global_tid; i < n; i += stride)
        {
            x_result[i] = x0[i];
        }

        return;
    }

    /*
     * x_old 和 x_new 是每个线程的局部指针，
     * 但所有线程执行完全相同的交换操作。
     */
    double* x_old = x0;
    double* x_new = x1;
    double* x_latest = x0;

    /*
     * ========================================================
     * 4. Jacobi 主迭代
     * ========================================================
     */
    for (int iter = 1; iter <= max_iter; ++iter)
    {
        /*
         * ----------------------------------------------------
         * 4.1 Jacobi 更新
         *
         * x_new[i] =
         *   (b[i] - sum_{j != i} Aij*x_old[j]) / Aii
         * ----------------------------------------------------
         */
        for (int i = global_tid; i < n; i += stride)
        {
            double diagonal = 0.0;
            double off_diagonal_sum = 0.0;

            for (int jj = row_ptr[i];
                 jj < row_ptr[i + 1];
                 ++jj)
            {
                int col = col_idx[jj];
                double value = values[jj];

                if (col == i)
                {
                    diagonal = value;
                }
                else
                {
                    off_diagonal_sum += value * x_old[col];
                }
            }

            if (diagonal != 0.0)
            {
                x_new[i] =
                    (b[i] - off_diagonal_sum) / diagonal;
            }
            else
            {
                /*
                 * 防止 x_new 中留下未初始化的数据。
                 */
                x_new[i] = x_old[i];

                atomicExch(&state->zero_diagonal, 1);
            }
        }

        /*
         * 必须保证所有 x_new 都已经写完，
         * 才能使用 x_new 计算残差。
         */
        grid.sync();

        /*
         * 遇到零对角元，停止 Jacobi。
         */
        if (state->zero_diagonal)
        {
            if (grid.thread_rank() == 0)
            {
                state->relative_residual = CUDART_INF;
                state->iterations = iter - 1;
                state->converged = 0;
                state->stop = 1;
            }

            /*
             * 当前这一轮 Jacobi 结果无效，
             * 所以返回上一轮的 x_old。
             */
            x_latest = x_old;

            grid.sync();
            break;
        }

        /*
         * ----------------------------------------------------
         * 4.2 清零残差平方和
         * ----------------------------------------------------
         */
        if (grid.thread_rank() == 0)
        {
            state->residual_sq = 0.0;
        }

        grid.sync();

        /*
         * ----------------------------------------------------
         * 4.3 计算 r = b - A*x_new
         * ----------------------------------------------------
         */
        local_sum = 0.0;

        for (int i = global_tid; i < n; i += stride)
        {
            double Ax_i = 0.0;

            for (int jj = row_ptr[i];
                 jj < row_ptr[i + 1];
                 ++jj)
            {
                int col = col_idx[jj];

                Ax_i += values[jj] * x_new[col];
            }

            double r_i = b[i] - Ax_i;

            local_sum += r_i * r_i;
        }

        /*
         * block 内归约。
         */
        block_sum = block_reduce_sum(local_sum);

        /*
         * 每个 block 只执行一次 double atomicAdd。
         */
        if (tid == 0)
        {
            atomicAdd(&state->residual_sq, block_sum);
        }

        /*
         * 确保所有 block 的 atomicAdd 均已完成。
         */
        grid.sync();

        x_latest = x_new;

        /*
         * ----------------------------------------------------
         * 4.4 计算相对残差并判断收敛
         * ----------------------------------------------------
         */
        if (grid.thread_rank() == 0)
        {
            double denominator =
                state->b_norm > 0.0
                ? state->b_norm
                : 1.0;

            state->relative_residual =
                sqrt(state->residual_sq) / denominator;

            state->iterations = iter;

            if (state->relative_residual <= tol)
            {
                state->converged = 1;
                state->stop = 1;
            }
        }

        grid.sync();

        /*
         * 所有线程读取同一个停止标志，
         * 所以会一起退出循环。
         */
        if (state->stop)
        {
            break;
        }

        /*
         * 下一轮：
         * 当前 x_new 变成下一轮的 x_old。
         */
        double* temp = x_old;
        x_old = x_new;
        x_new = temp;
    }

    /*
     * ========================================================
     * 5. 把最后一次有效结果写到固定输出位置
     * ========================================================
     */
    for (int i = global_tid; i < n; i += stride)
    {
        x_result[i] = x_latest[i];
    }
}


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
    JacobiState* host_state)
{
    if (n <= 0 ||
        max_iter < 0 ||
        tol < 0.0 ||
        d_row_ptr == nullptr ||
        d_col_idx == nullptr ||
        d_values == nullptr ||
        d_b == nullptr ||
        d_x0 == nullptr ||
        d_x1 == nullptr ||
        d_x_result == nullptr ||
        host_state == nullptr)
    {
        return cudaErrorInvalidValue;
    }

    int device = 0;

    cudaError_t error = cudaGetDevice(&device);

    if (error != cudaSuccess)
    {
        return error;
    }

    /*
     * 检查 GPU 是否支持 cooperative launch。
     */
    int cooperative_supported = 0;

    error = cudaDeviceGetAttribute(
        &cooperative_supported,
        cudaDevAttrCooperativeLaunch,
        device);

    if (error != cudaSuccess)
    {
        return error;
    }

    if (!cooperative_supported)
    {
        std::cerr
            << "Current GPU does not support cooperative launch."
            << std::endl;

        return cudaErrorNotSupported;
    }

    constexpr int block_size = 256;

    /*
     * block_reduce_sum 使用的动态共享内存。
     */
    std::size_t shared_memory_size =
        block_size * sizeof(double);

    /*
     * 查询一个 SM 最多能够同时驻留多少个当前 kernel 的 block。
     */
    int active_blocks_per_sm = 0;

    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_per_sm,
        jacobi_cooperative_kernel,
        block_size,
        shared_memory_size);

    if (error != cudaSuccess)
    {
        return error;
    }

    if (active_blocks_per_sm <= 0)
    {
        return cudaErrorLaunchOutOfResources;
    }

    int sm_count = 0;

    error = cudaDeviceGetAttribute(
        &sm_count,
        cudaDevAttrMultiProcessorCount,
        device);

    if (error != cudaSuccess)
    {
        return error;
    }

    /*
     * cooperative kernel 的最大安全 block 数量。
     */
    int maximum_resident_blocks =
        active_blocks_per_sm * sm_count;

    int blocks_needed =
        (n + block_size - 1) / block_size;

    int grid_size =
        std::min(blocks_needed, maximum_resident_blocks);

    /*
     * 即使 grid_size 小于处理 n 所需的 block 数，
     * kernel 内部也使用 grid-stride loop 处理全部元素。
     */
    grid_size = std::max(grid_size, 1);

    JacobiState* d_state = nullptr;

    error = cudaMalloc(
        reinterpret_cast<void**>(&d_state),
        sizeof(JacobiState));

    if (error != cudaSuccess)
    {
        return error;
    }

    /*
     * cudaLaunchCooperativeKernel 要求使用 void* 参数数组。
     */
    void* kernel_arguments[] = {
        &n,
        &d_row_ptr,
        &d_col_idx,
        &d_values,
        &d_b,
        &d_x0,
        &d_x1,
        &d_x_result,
        &max_iter,
        &tol,
        &d_state
    };

    error = cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(jacobi_cooperative_kernel),
        dim3(grid_size),
        dim3(block_size),
        kernel_arguments,
        shared_memory_size,
        nullptr);

    if (error != cudaSuccess)
    {
        cudaFree(d_state);
        return error;
    }

    /*
     * CPU 只在整个 Jacobi 迭代结束后同步一次。
     */
    error = cudaDeviceSynchronize();

    if (error != cudaSuccess)
    {
        cudaFree(d_state);
        return error;
    }

    /*
     * 只复制一个很小的状态结构体回 CPU。
     */
    error = cudaMemcpy(
        host_state,
        d_state,
        sizeof(JacobiState),
        cudaMemcpyDeviceToHost);

    cudaError_t free_error = cudaFree(d_state);

    if (error != cudaSuccess)
    {
        return error;
    }

    return free_error;
}
