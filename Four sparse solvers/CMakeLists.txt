cmake_minimum_required(VERSION 3.24)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(GPUSparseSolverBenchmarks LANGUAGES CXX CUDA)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

option(GPU_SOLVERS_BUILD_SUPERLU_DIST "Build the SuperLU_DIST adapter" ON)
option(GPU_SOLVERS_BUILD_PANGULU "Build the PanguLU adapter" ON)
option(GPU_SOLVERS_BUILD_CUDSS "Build the NVIDIA cuDSS adapter" ON)
option(GPU_SOLVERS_BUILD_STRUMPACK "Build the STRUMPACK adapter" ON)

set(SUPERLU_DIST_ROOT "" CACHE PATH "SuperLU_DIST installation prefix")
set(PANGULU_ROOT "" CACHE PATH "PanguLU installation prefix")
set(CUDSS_ROOT "" CACHE PATH "NVIDIA cuDSS installation prefix")
set(STRUMPACK_ROOT "" CACHE PATH "STRUMPACK installation prefix")
set(SCALAPACK_ROOT "" CACHE PATH "ScaLAPACK installation prefix")
set(OPENBLAS_ROOT "" CACHE PATH "OpenBLAS installation prefix")

add_library(mtx_reader INTERFACE)
target_include_directories(mtx_reader INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

function(gpu_solver_target target)
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_compile_options(${target} PRIVATE -O3)
    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endfunction()

if(GPU_SOLVERS_BUILD_SUPERLU_DIST)
    find_path(SUPERLU_DIST_INCLUDE_DIR superlu_ddefs.h
        HINTS ${SUPERLU_DIST_ROOT} PATH_SUFFIXES include)
    find_library(SUPERLU_DIST_LIBRARY NAMES superlu_dist
        HINTS ${SUPERLU_DIST_ROOT} PATH_SUFFIXES lib lib64)
    if(NOT SUPERLU_DIST_INCLUDE_DIR OR NOT SUPERLU_DIST_LIBRARY)
        message(FATAL_ERROR
            "SuperLU_DIST was requested but not found. Set SUPERLU_DIST_ROOT.")
    endif()

    add_executable(test_slu test_superlu_dist.cpp)
    target_include_directories(test_slu PRIVATE ${SUPERLU_DIST_INCLUDE_DIR})
    target_link_libraries(test_slu PRIVATE
        mtx_reader MPI::MPI_CXX ${SUPERLU_DIST_LIBRARY})
    gpu_solver_target(test_slu)
endif()

if(GPU_SOLVERS_BUILD_PANGULU)
    find_path(PANGULU_INCLUDE_DIR pangulu.h
        HINTS ${PANGULU_ROOT} PATH_SUFFIXES include)
    find_library(PANGULU_LIBRARY NAMES pangulu
        HINTS ${PANGULU_ROOT} PATH_SUFFIXES lib lib64)
    if(NOT PANGULU_INCLUDE_DIR OR NOT PANGULU_LIBRARY)
        message(FATAL_ERROR
            "PanguLU was requested but not found. Set PANGULU_ROOT.")
    endif()

    add_executable(test_pangulu test_pangulu.cpp)
    target_include_directories(test_pangulu PRIVATE ${PANGULU_INCLUDE_DIR})
    target_link_libraries(test_pangulu PRIVATE
        mtx_reader MPI::MPI_CXX ${PANGULU_LIBRARY}
        CUDA::cublas CUDA::cudart)
    gpu_solver_target(test_pangulu)
endif()

if(GPU_SOLVERS_BUILD_CUDSS)
    find_path(CUDSS_INCLUDE_DIR cudss.h
        HINTS ${CUDSS_ROOT} PATH_SUFFIXES include)
    find_library(CUDSS_LIBRARY NAMES cudss cudss.so.0
        HINTS ${CUDSS_ROOT} PATH_SUFFIXES lib lib64)
    if(NOT CUDSS_INCLUDE_DIR OR NOT CUDSS_LIBRARY)
        message(FATAL_ERROR
            "cuDSS was requested but not found. Set CUDSS_ROOT.")
    endif()

    add_executable(test_cudss test_cudss.cpp)
    target_include_directories(test_cudss PRIVATE ${CUDSS_INCLUDE_DIR})
    target_link_libraries(test_cudss PRIVATE
        mtx_reader MPI::MPI_CXX ${CUDSS_LIBRARY}
        CUDA::cublas CUDA::cudart)
    gpu_solver_target(test_cudss)
endif()

if(GPU_SOLVERS_BUILD_STRUMPACK)
    find_path(STRUMPACK_INCLUDE_DIR StrumpackSparseSolverMPIDist.hpp
        HINTS ${STRUMPACK_ROOT} PATH_SUFFIXES include)
    find_library(STRUMPACK_LIBRARY NAMES strumpack
        HINTS ${STRUMPACK_ROOT} PATH_SUFFIXES lib lib64)
    find_library(SCALAPACK_LIBRARY NAMES scalapack
        HINTS ${SCALAPACK_ROOT} PATH_SUFFIXES lib lib64)
    find_library(OPENBLAS_LIBRARY NAMES openblas
        HINTS ${OPENBLAS_ROOT} PATH_SUFFIXES lib lib64)
    if(NOT STRUMPACK_INCLUDE_DIR OR NOT STRUMPACK_LIBRARY)
        message(FATAL_ERROR
            "STRUMPACK was requested but not found. Set STRUMPACK_ROOT.")
    endif()
    if(NOT SCALAPACK_LIBRARY OR NOT OPENBLAS_LIBRARY)
        message(FATAL_ERROR
            "STRUMPACK requires ScaLAPACK and OpenBLAS. Set SCALAPACK_ROOT and OPENBLAS_ROOT.")
    endif()

    add_executable(test_strumpack test_strumpack.cpp)
    target_include_directories(test_strumpack PRIVATE ${STRUMPACK_INCLUDE_DIR})
    target_link_libraries(test_strumpack PRIVATE
        mtx_reader MPI::MPI_CXX ${STRUMPACK_LIBRARY}
        ${SCALAPACK_LIBRARY} ${OPENBLAS_LIBRARY}
        CUDA::cublas CUDA::cudart)
    gpu_solver_target(test_strumpack)
endif()
