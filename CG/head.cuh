#pragma once
#include <cuda_runtime.h>
#include <library_types.h>
#include <cudss.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <map>
#include <sstream>
#include <iomanip>

struct CSRMatrix
{
    int n;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<double> values;
};

struct SolverOptions
{
    int max_iterations;
    int residual_check_interval;
    double relative_tolerance;
    int block_size;
    int grid_size;
};
#define CHECK_FUNC(call)                                      \
    do {                                                      \
        int status = (call);                                  \
        if (status != 0) {                                    \
            std::cerr << "Function error at " << __FILE__     \
                      << ":" << __LINE__                      \
                      << " : status = " << status             \
                      << std::endl;                           \
            return 1;                                         \
        }                                                     \
    } while (0)

#define CHECK_CUDA(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            std::cerr << "CUDA error at " << __FILE__ << ":"  \
                      << __LINE__ << " : "                    \
                      << cudaGetErrorString(err) << std::endl; \
            return 1;                                         \
        }                                                     \
    } while (0)

#define CHECK_CUDSS(call)                                     \
    do {                                                      \
        cudssStatus_t status = (call);                        \
        if (status != CUDSS_STATUS_SUCCESS) {                 \
            std::cerr << "cuDSS error at " << __FILE__ << ":" \
                      << __LINE__ << " : status = "           \
                      << static_cast<int>(status) << std::endl;\
            return 1;                                         \
        }                                                     \
    } while (0)

int VecScale_gpu(int n, double *y, double alpha, 
    int blockSize, int gridSize);
int AXPY_gpu(int n, const double *x, double *y, double alpha, 
    int blockSize, int gridSize);
int AXPBY_gpu(int n, const double *x, double *y, double alpha, 
    double beta, int blockSize, int gridSize);
int Dot_gpu(int n, const double *x, const double *y, double &result, 
    int blockSize, int gridSize);
int Dot_device(int n, const double* d_x, const double* d_y,
    double* d_partial_sums, double& result, int blockSize,
    int gridSize);
int SpMV_gpu(int n, int nnz, const int *row_ptr, const int *col_idx, 
    const double *values, const double *x, double *y, 
    int blockSize, int gridSize, int kernel_type);
int Readmtx(std::string filename, int &M, int &N, int &nnz, 
            std::vector<int> &COO_row,
            std::vector<int> &COO_col, 
            std::vector<double> &COO_values);
int COO_to_CSR(int M, int N, int nnz,
               const std::vector<int> &COO_row,
               const std::vector<int> &COO_col,
               const std::vector<double> &COO_values,
               std::vector<int> &CSR_row_ptr,
               std::vector<int> &CSR_col_idx,
               std::vector<double> &CSR_values);
void CheckCSR(const CSRMatrix& A);
