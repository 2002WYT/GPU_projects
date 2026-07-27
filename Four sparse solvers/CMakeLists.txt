cmake_minimum_required(VERSION 3.18)
if(DEFINED ENV{CUDA_HOME})
    set(CMAKE_CUDA_COMPILER "$ENV{CUDA_HOME}/bin/nvcc" CACHE FILEPATH "CUDA compiler")
    set(CUDAToolkit_ROOT "$ENV{CUDA_HOME}" CACHE PATH "CUDA Toolkit root")
endif()
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.6/bin/nvcc)
project(cuda_test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 70)

# ---- Install roots ----
set(CUDA_ROOT        /usr/local/cuda-12.6)
set(CUDSS_ROOT       /home/wangyitong/cudss_install/nvidia/cu12)
set(MPI_HOME         /usr/lib/x86_64-linux-gnu/openmpi)
set(SUPERLU_DIST_ROOT /home/wangyitong/soft/superlu_dist-install)
set(PANGULU_ROOT     /home/wangyitong/soft/PanguLU-5.0.0)
set(STRUMPACK_ROOT   /home/wangyitong/soft/strumpack-slate-install)
set(SCALAPACK_ROOT   /home/wangyitong/soft/scalapack-install)
set(OPENBLAS_ROOT    /home/wangyitong/soft/openblas-install)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CUDSS_ROOT}/include
    ${CUDA_ROOT}/include
    ${MPI_HOME}/include
    ${SUPERLU_DIST_ROOT}/include
    ${PANGULU_ROOT}/include
    ${STRUMPACK_ROOT}/include
)

link_directories(
    ${CUDSS_ROOT}/lib
    ${MPI_HOME}/lib
    ${SUPERLU_DIST_ROOT}/lib
    ${PANGULU_ROOT}/lib
    ${STRUMPACK_ROOT}/lib
    ${SCALAPACK_ROOT}/lib
    ${OPENBLAS_ROOT}/lib
)

find_package(CUDAToolkit REQUIRED)
aux_source_directory(src SRC_SOURCE)
add_library(SRC ${SRC_SOURCE})

# ---- SuperLU_DIST ----
set(EXE1 test_slu)
add_executable(${EXE1} test_superlu_dist.cpp)
target_link_libraries(${EXE1} PRIVATE
    SRC
    superlu_dist
    CUDA::cublas
    CUDA::cudart
    ${CUDSS_ROOT}/lib/libcudss.so.0
    mpi
    mpi_cxx
)

# ---- PanguLU ----
set(EXE2 test_pangulu)
add_executable(${EXE2} test_pangulu.cpp)
add_dependencies(${EXE2} ${EXE1})  # serialize to avoid GPU contention
target_link_libraries(${EXE2} PRIVATE
    SRC
    pangulu
    superlu_dist
    CUDA::cublas
    CUDA::cudart
    ${CUDSS_ROOT}/lib/libcudss.so.0
    mpi
    mpi_cxx
)

# ---- cuDSS (single GPU, no MPI) ----
set(EXE3 test_cudss)
add_executable(${EXE3} test_cudss.cpp)
target_link_libraries(${EXE3} PRIVATE
    SRC
    ${CUDSS_ROOT}/lib/libcudss.so.0
    CUDA::cublas
    CUDA::cudart
    mpi
    mpi_cxx
)

# ---- STRUMPACK ----
set(EXE4 test_strumpack)
add_executable(${EXE4} test_strumpack.cpp)
target_link_libraries(${EXE4} PRIVATE
    SRC
    strumpack
    scalapack
    openblas
    CUDA::cublas
    CUDA::cudart
    mpi
    mpi_cxx
)
