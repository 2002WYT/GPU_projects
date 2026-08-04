#pragma once

#include <cuda_runtime.h>

#include <iostream>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        const cudaError_t error__ = (call);                                    \
        if (error__ != cudaSuccess) {                                          \
            std::cerr << "CUDA error at " << __FILE__ << ':' << __LINE__     \
                      << ": " << cudaGetErrorString(error__) << std::endl;    \
            return 1;                                                          \
        }                                                                      \
    } while (0)
