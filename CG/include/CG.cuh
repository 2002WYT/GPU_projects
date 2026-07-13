#include "head.cuh"

int CG_gpu(
    const CSRMatrix& A,
    const std::vector<double>& b,
    std::vector<double>& x,
    const SolverOptions& options);
