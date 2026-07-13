#include "head.cuh"
#include <cuda_runtime.h>

__global__ void vecscale_kernel(int n, double alpha, double* y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (; i < n; i += stride)
    {
        y[i] = alpha * y[i];
    }
}
__global__ void axpy_kernel(int n, double alpha, const double* x, double* y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (; i < n; i += stride)
    {
        y[i] = alpha * x[i] + y[i];
    }
}
__global__ void axpby_kernel(int n, double alpha, double beta, const double* x, double* y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (; i < n; i += stride)
    {
        y[i] = alpha * x[i] + beta * y[i];
    }
}
__global__ void dot_kernel(
    int n,
    const double* __restrict__ x,
    const double* __restrict__ y,
    double* partial_sums)
{
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    double local_sum = 0.0;

    for (; i < n; i += stride)
    {
        local_sum += x[i] * y[i];
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            sdata[tid] += sdata[tid + offset];
        }

        __syncthreads();
    }

    if (tid == 0)
    {
        partial_sums[blockIdx.x] = sdata[0];
    }
}
int VecScale_gpu(int n, double *y, double alpha, 
    int blockSize, int gridSize)
{
    double* d_y = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_y, n * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_y, y, n * sizeof(double), 
        cudaMemcpyHostToDevice));

    vecscale_kernel <<<gridSize, blockSize>>> (n, alpha, d_y);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(y, d_y, n * sizeof(double), 
        cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_y));
    return 0;
}
int AXPY_gpu(int n, const double *x, double *y, double alpha, 
    int blockSize, int gridSize)
{
    double* d_x = nullptr;
    double* d_y = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_x, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_y, n * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_x, x, n * sizeof(double), 
        cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, y, n * sizeof(double), 
        cudaMemcpyHostToDevice));

    axpy_kernel <<<gridSize, blockSize>>> (n, alpha, d_x, d_y);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(y, d_y, n * sizeof(double), 
        cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    return 0;
}
int AXPBY_gpu(int n, const double *x, double *y, double alpha, 
    double beta, int blockSize, int gridSize)
{
    double* d_x = nullptr;
    double* d_y = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_x, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_y, n * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_x, x, n * sizeof(double), 
        cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, y, n * sizeof(double), 
        cudaMemcpyHostToDevice));

    axpby_kernel <<<gridSize, blockSize>>> 
        (n, alpha, beta, d_x, d_y);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(y, d_y, n * sizeof(double), 
        cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    return 0;
}
int Dot_gpu(int n, const double *x, const double *y, double &result, 
    int blockSize, int gridSize)
{
    if ((blockSize & (blockSize - 1)) != 0)
    {
        std::cerr << "Error: blockSize must be power of two for Dot_gpu." << std::endl;
        return 1;
    }
    double* d_x = nullptr;
    double* d_y = nullptr;
    double *d_partial_sums = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_x, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_y, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_partial_sums, gridSize * sizeof(double)));

    std::vector<double> h_partial_sums(gridSize);

    CHECK_CUDA(cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, y, n * sizeof(double), cudaMemcpyHostToDevice));
    dot_kernel <<<gridSize, blockSize, blockSize * sizeof(double)>>> 
        (n, d_x, d_y, d_partial_sums);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_partial_sums.data(), d_partial_sums, 
        gridSize * sizeof(double), cudaMemcpyDeviceToHost));
    result = 0.0;
    for (int i = 0; i < gridSize; i++)
    {
        result += h_partial_sums[i];
    }
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUDA(cudaFree(d_partial_sums));
    return 0;
}
int Dot_device(int n, const double* d_x, const double* d_y,
    double* d_partial_sums, double& result, int blockSize,
    int gridSize) // 传入的是gpu上的数组
{
    if ((blockSize & (blockSize - 1)) != 0)
    {
        std::cerr
            << "Error: blockSize must be power of two."
            << std::endl;

        return 1;
    }

    /*
     * d_x 和 d_y 已经位于 GPU，
     * 直接启动点积核函数。
     */
    dot_kernel<<<
        gridSize,
        blockSize,
        blockSize * sizeof(double)>>>(
            n,
            d_x,
            d_y,
            d_partial_sums);

    CHECK_CUDA(cudaGetLastError());

    /*
     * 只把每个 block 的部分和复制回 CPU。
     */
    std::vector<double> h_partial_sums(gridSize);

    CHECK_CUDA(cudaMemcpy(
        h_partial_sums.data(),
        d_partial_sums,
        gridSize * sizeof(double),
        cudaMemcpyDeviceToHost));

    result = 0.0;

    for (int i = 0; i < gridSize; ++i)
    {
        result += h_partial_sums[i];
    }

    return 0;
}
void CheckCSR(const CSRMatrix& A)
{
    if (A.n <= 0)
    {
        throw std::invalid_argument(
            "A.n must be positive.");
    }
    const int n = A.n;
    const int nnz =
    static_cast<int>(A.values.size());
    if (static_cast<int>(A.row_ptr.size()) != n + 1)
    {
        throw std::invalid_argument(
            "A.row_ptr.size() must equal A.n + 1.");
    }
    if (static_cast<int>(A.col_idx.size()) != nnz)
    {
        throw std::invalid_argument(
            "A.col_idx.size() must equal A.values.size().");
    }
    if (A.row_ptr.front() != 0)
    {
        throw std::invalid_argument(
            "A.row_ptr[0] must equal zero.");
    }
    if (A.row_ptr.back() != nnz)
    {
        throw std::invalid_argument(
            "A.row_ptr[A.n] must equal nnz.");
    }
    for (int row = 0; row < n; ++row)
    {
        if (A.row_ptr[row] > A.row_ptr[row + 1])
        {
            throw std::invalid_argument(
                "A.row_ptr must be nondecreasing.");
        }
    }
    for (int row = 0; row <= n; ++row)
    {
        if (A.row_ptr[row] < 0 ||
            A.row_ptr[row] > nnz)
        {
            throw std::invalid_argument(
                "A.row_ptr contains an invalid offset.");
        }
    }
    for (int jj = 0; jj < nnz; ++jj)
    {
        if (A.col_idx[jj] < 0 ||
            A.col_idx[jj] >= n)
        {
            throw std::invalid_argument(
                "A.col_idx contains an invalid column index.");
        }
    }
}
