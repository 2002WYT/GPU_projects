#include "head.cuh"
#include "CG.cuh"

__global__ void axpy_kernel(
    int n,
    double alpha,
    const double* x,
    double* y);

__global__ void axpby_kernel(
    int n,
    double alpha,
    double beta,
    const double* x,
    double* y);

__global__ void dot_kernel(
    int n,
    const double* x,
    const double* y,
    double* partial_sums);

template <typename T>
__global__ void spmv_csr_scalar_kernel(int n, const int *row_ptr, const int *col_idx, 
    const T *values, const T *x, T *y);

int CG_gpu(
    const CSRMatrix& A,
    const std::vector<double>& b,
    std::vector<double>& x,
    const SolverOptions& options)
{
    CheckCSR(A);
    if (static_cast<int>(b.size()) != A.n) {
        throw std::invalid_argument("b.size() must equal A.n.");
    }
    if (x.empty()) {
        x.assign(A.n, 0.0);
    }
    if (static_cast<int>(x.size()) != A.n) {
        throw std::invalid_argument("x.size() must equal A.n.");
    }
    if (options.max_iterations <= 0 || options.residual_check_interval <= 0) {
        throw std::invalid_argument("Iteration counts must be positive.");
    }
    if (!(options.relative_tolerance > 0.0)) {
        throw std::invalid_argument("relative_tolerance must be positive.");
    }
    if (options.block_size <= 0 || options.block_size > 1024) {
        throw std::invalid_argument("block_size must be in [1, 1024].");
    }
    if ((options.block_size & (options.block_size - 1)) != 0)
    {
        throw std::invalid_argument("block_size must be a power of two.");
    }
    const int n = A.n;
    const int nnz = static_cast<int>(A.values.size());
    const int block_size = options.block_size;
    int grid_size = options.grid_size;
    int* d_row_ptr = nullptr;
    int* d_col_idx = nullptr;
    double* d_values = nullptr;
    double* d_b = nullptr;
    double* d_x = nullptr;
    double* d_r = nullptr;
    double* d_p = nullptr;
    double* d_Ap = nullptr;
    double* d_partial_sums = nullptr;

    auto cleanup = [&]()
    {
        if (d_row_ptr != nullptr) cudaFree(d_row_ptr);
        if (d_col_idx != nullptr) cudaFree(d_col_idx);
        if (d_values != nullptr) cudaFree(d_values);
        if (d_b != nullptr) cudaFree(d_b);
        if (d_x != nullptr) cudaFree(d_x);
        if (d_r != nullptr) cudaFree(d_r);
        if (d_p != nullptr) cudaFree(d_p);
        if (d_Ap != nullptr) cudaFree(d_Ap);
        if (d_partial_sums != nullptr) cudaFree(d_partial_sums);
    };
    #define CG_CHECK_CUDA(call)                                   \
    do                                                        \
    {                                                         \
        cudaError_t cg_error = (call);                        \
        if (cg_error != cudaSuccess)                          \
        {                                                     \
            std::cerr                                         \
                << "CUDA error at "                           \
                << __FILE__                                   \
                << ":"                                        \
                << __LINE__                                   \
                << " : "                                      \
                << cudaGetErrorString(cg_error)               \
                << std::endl;                                 \
            cleanup();                                        \
            return 1;                                         \
        }                                                     \
    } while (0)

    CG_CHECK_CUDA(cudaMalloc(
            reinterpret_cast<void**>(&d_row_ptr),
            (n + 1) * sizeof(int)));
    CG_CHECK_CUDA(cudaMalloc(
            reinterpret_cast<void**>(&d_col_idx),
            nnz * sizeof(int)));
    CG_CHECK_CUDA(cudaMalloc(
            reinterpret_cast<void**>(&d_values),
            nnz * sizeof(double)));
    CG_CHECK_CUDA(cudaMalloc(
            reinterpret_cast<void**>(&d_b),
            n * sizeof(double)));
    CG_CHECK_CUDA(cudaMalloc(
            reinterpret_cast<void**>(&d_x),
            n * sizeof(double)));
    CG_CHECK_CUDA(cudaMalloc(
            reinterpret_cast<void**>(&d_r),
            n * sizeof(double)));
    CG_CHECK_CUDA(cudaMalloc(
            reinterpret_cast<void**>(&d_p),
            n * sizeof(double)));
    CG_CHECK_CUDA(cudaMalloc(
            reinterpret_cast<void**>(&d_Ap),
            n * sizeof(double)));
    CG_CHECK_CUDA(cudaMalloc(
            reinterpret_cast<void**>(&d_partial_sums),
            grid_size * sizeof(double)));
    CG_CHECK_CUDA(cudaMemcpy(
            d_row_ptr, A.row_ptr.data(),
            (n + 1) * sizeof(int),
            cudaMemcpyHostToDevice));
    CG_CHECK_CUDA(cudaMemcpy(
            d_col_idx, A.col_idx.data(),
            nnz * sizeof(int),
            cudaMemcpyHostToDevice));
    CG_CHECK_CUDA(cudaMemcpy(
            d_values, A.values.data(),
            nnz * sizeof(double),
            cudaMemcpyHostToDevice));
    CG_CHECK_CUDA(cudaMemcpy(
            d_b, b.data(),
            n * sizeof(double),
            cudaMemcpyHostToDevice));
    CG_CHECK_CUDA(cudaMemcpy(
            d_x, x.data(),
            n * sizeof(double),
            cudaMemcpyHostToDevice));

    /*
     * ========================================================
     * 0.1. 计算初始残差 r = b - A*x
     * ========================================================
     */
    spmv_csr_scalar_kernel<<<grid_size, block_size>>>(
            n, d_row_ptr, d_col_idx,
            d_values, d_x, d_Ap); // d_Ap = A *d_x
    CG_CHECK_CUDA(cudaGetLastError());

    CG_CHECK_CUDA(cudaMemcpy(
            d_r, d_b,
            n * sizeof(double),
            cudaMemcpyDeviceToDevice)); // d_r = d_b

    axpy_kernel<<<grid_size, block_size>>>(
            n, -1.0, d_Ap, d_r); // d_r = -d_Ap + d_r = d_b - A*d_x
    CG_CHECK_CUDA(cudaGetLastError());

    /*
     * ========================================================
     * 0.2. 设置初始搜索方向 d_p = d_r
     * ========================================================
     */
    CG_CHECK_CUDA(cudaMemcpy(
            d_p, d_r,
            n * sizeof(double),
            cudaMemcpyDeviceToDevice));

    /*
     * ========================================================
     * 0.3. 计算 d_b^T*d_b
     * ========================================================
     */
    double bb = 0.0;
    if (Dot_device(n, d_b, d_b, d_partial_sums, bb, 
        block_size, grid_size) != 0)
    {
        cleanup();
        return 1;
    }
    if (bb < 0.0 || !std::isfinite(bb))
    {
        std::cerr
            << "Invalid b^T*b: "
            << bb
            << std::endl;
        cleanup();
        return 1;
    }
    const double b_norm = std::sqrt(bb);
    const double residual_normalizer = (b_norm > 0.0) ? b_norm : 1.0;

    /*
     * ========================================================
     * 0.4. 计算初始的 r^T*r
     * ========================================================
     */
    double rr_old = 0.0;
    if (Dot_device(n, d_r, d_r, d_partial_sums, rr_old, 
        block_size, grid_size) != 0)
    {
        cleanup();
        return 1;
    }
    if (rr_old < 0.0 || !std::isfinite(rr_old))
    {
        std::cerr
            << "Invalid r^T*r: "
            << rr_old
            << std::endl;
        cleanup();
        return 1;
    }
    double relative_residual = std::sqrt(rr_old) /
        residual_normalizer;
    std::cout << std::scientific << std::setprecision(12);
    std::cout << "CG initial relative residual = "
        << relative_residual << std::endl;

    /*
     * ========================================================
     * 0.5. 如果初始解已经收敛
     * ========================================================
     */
    if (relative_residual <= options.relative_tolerance)
    {
        std::cout
            << "CG converged at the initial guess."
            << std::endl;
        cleanup();
        return 0;
    }
    /*
     * ========================================================
     * 1. 开始 CG 迭代
     * ========================================================
     */
    bool converged = false;
    int final_iteration = 0;
    for (int iteration = 1; iteration <= options.max_iterations;
         ++iteration)
    {
        /*
         * ====================================================
         * 1.1. 计算 alpha
         *
         *                  r_k^T*r_k
         *      alpha_k = ----------------
         *                  p_k^T*A*p_k
         * ====================================================
         */
        spmv_csr_scalar_kernel<<<grid_size, block_size>>>
            (n, d_row_ptr, d_col_idx, 
            d_values, d_p, d_Ap); // d_Ap = A * d_p
        CG_CHECK_CUDA(cudaGetLastError());

        double pAp = 0.0;
        if (Dot_device(n, d_p, d_Ap, d_partial_sums, 
            pAp, block_size, 
            grid_size) != 0) // pAp = d_p^T * A * d_p
        {
            cleanup();
            return 1;
        }
        if (!(pAp > 0.0) ||
            !std::isfinite(pAp))
        {
            std::cerr
                << "CG breakdown at iteration "
                << iteration
                << ": p^T*A*p = "
                << pAp
                << std::endl;
            cleanup();
            return 1;
        }
        const double alpha =
            rr_old / pAp;
        if (!std::isfinite(alpha))
        {
            std::cerr
                << "Invalid alpha at iteration "
                << iteration
                << std::endl;
            cleanup();
            return 1;
        }

        /*
         * ====================================================
         * 1.2. 更新解 x = x + alpha*p
         * ====================================================
         */
        axpy_kernel<<<grid_size, block_size>>>(
                n, alpha, d_p, d_x);
        CG_CHECK_CUDA(cudaGetLastError());

        /*
         * ====================================================
         * 1.3. 更新残差 r = r - alpha*Ap
         * ====================================================
         */
        axpy_kernel<<<grid_size, block_size>>>(
                n, -alpha, d_Ap, d_r);
        CG_CHECK_CUDA(cudaGetLastError());
        
        /*
         * ====================================================
         * 1.4. 计算相对残差，检查是否收敛
         *
         *          sqrt(r^T*r)
         *      ------------------
         *              ||b||
         * ====================================================
         */
        double rr_new = 0.0;

        if (Dot_device(n, d_r, d_r, d_partial_sums, 
            rr_new, block_size, 
            grid_size) != 0) // rr_new = d_r^T * d_r
        {
            cleanup();
            return 1;
        }
        if (rr_new < 0.0 ||
            !std::isfinite(rr_new))
        {
            std::cerr
                << "Invalid r^T*r at iteration "
                << iteration
                << ": "
                << rr_new
                << std::endl;
            cleanup();
            return 1;
        }
        relative_residual = std::sqrt(rr_new) /
            residual_normalizer;
        if (iteration == 1 ||
            iteration %
                options.residual_check_interval == 0 ||
            relative_residual <=
                options.relative_tolerance ||
            iteration == options.max_iterations)
        {
            std::cout
                << "CG iteration "
                << std::setw(6)
                << iteration
                << ", alpha = "
                << alpha
                << ", relative residual = "
                << relative_residual
                << std::endl;
        }
        if (relative_residual <=
            options.relative_tolerance)
        {
            converged = true;
            final_iteration = iteration;
            break;
        }

        /*
         * ====================================================
         * 1.5. 计算 beta
         *
         *                 r_{k+1}^T*r_{k+1}
         *      beta_k = ---------------------
         *                    r_k^T*r_k
         * ====================================================
         */

        if (rr_old == 0.0)
        {
            std::cerr
                << "CG breakdown: rr_old is zero."
                << std::endl;
            cleanup();
            return 1;
        }
        const double beta = rr_new / rr_old;
        if (!std::isfinite(beta))
        {
            std::cerr
                << "Invalid beta at iteration "
                << iteration
                << std::endl;
            cleanup();
            return 1;
        }

        /*
         * ====================================================
         * 1.6. 更新搜索方向 p = r + beta*p
         * ====================================================
         */

        axpby_kernel<<<grid_size, block_size>>>(
                n, 1.0, beta, d_r, d_p);
        CG_CHECK_CUDA(cudaGetLastError());

        /*
         * ====================================================
         * 1.7. 保存新的 r^T*r
         *
         * 下一轮中：
         *
         *      rr_old = r_{k+1}^T*r_{k+1}
         * ====================================================
         */
        rr_old = rr_new;
        final_iteration = iteration;
    }

    /*
     * ========================================================
     * 2.1. 将最终解从 GPU 复制回 CPU
     * ========================================================
     */
    CG_CHECK_CUDA(cudaMemcpy(
            x.data(), d_x,
            n * sizeof(double),
            cudaMemcpyDeviceToHost));

    /*
     * ========================================================
     * 2.2. 输出迭代结果
     * ========================================================
     */
    if (converged)
    {
        std::cout
            << "CG converged after "
            << final_iteration
            << " iterations."
            << std::endl;
    }
    else
    {
        std::cout
            << "CG reached maximum iterations: "
            << options.max_iterations
            << std::endl;
    }

    std::cout
        << "Final recursive relative residual = "
        << relative_residual
        << std::endl;
    
    /*
     * ========================================================
     * 2.3.  重新计算真实残差
     *
     * 迭代中的残差使用递推公式：
     *
     *      r = r - alpha*Ap
     *
     * 由于浮点误差，递推残差可能与真实残差：
     *
     *      b - A*x
     *
     * 有少量区别。
     * ========================================================
     */
    spmv_csr_scalar_kernel<<<grid_size, block_size>>>(
            n, d_row_ptr, d_col_idx,
            d_values, d_x, d_Ap); // d_Ap = A * d_x
    CG_CHECK_CUDA(cudaGetLastError());
    CG_CHECK_CUDA(cudaMemcpy(
            d_r, d_b,
            n * sizeof(double),
            cudaMemcpyDeviceToDevice)); // d_r = d_b
    axpy_kernel<<<grid_size, block_size>>>(
            n, -1.0, d_Ap, d_r); // d_r = d_r - d_Ap = d_b - A * d_x
    CG_CHECK_CUDA(cudaGetLastError());

    double true_rr = 0.0;
    if (Dot_device(n, d_r, d_r, d_partial_sums, 
            true_rr, block_size, 
            grid_size) != 0)
    {
        cleanup();
        return 1;
    }
    const double true_relative_residual =
        std::sqrt(true_rr) /
        residual_normalizer;
    std::cout
        << "Final true relative residual "
        << "||b-Ax||/||b|| = "
        << true_relative_residual
        << std::endl;
    /*
     * ========================================================
     * 2.4.  释放所有显存
     * ========================================================
     */

    cleanup();
#undef CG_CHECK_CUDA
    return 0;
}
