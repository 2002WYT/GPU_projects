cmake_minimum_required(VERSION 3.18)
project(spgemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# Default to the local GPU's arch; override with -DCMAKE_CUDA_ARCHITECTURES (build.sh uses 70;80;90).
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# --- AMGX (only needed for the amgx backend) ----------------------------------
# Source tree with the C++ internal headers (matrix.h, csr_multiply.h).
# Lib dir is AMGX_BUILD if given, else ${AMGX_ROOT}/build.
if(AMGX_ROOT)
  if(NOT AMGX_INCLUDE_DIR)
    set(AMGX_INCLUDE_DIR "${AMGX_ROOT}/include")
  endif()
  if(NOT AMGX_BUILD)
    set(AMGX_BUILD "${AMGX_ROOT}/build")
  endif()
  find_library(AMGX_LIBRARY amgxsh PATHS "${AMGX_BUILD}" NO_DEFAULT_PATH)
  if(NOT AMGX_LIBRARY)
    message(FATAL_ERROR "AMGX_ROOT set but libamgxsh.so not found under: ${AMGX_BUILD}")
  endif()
endif()

# --- targets -------------------------------------------------------------------
# bench_main.cpp includes <cuda_runtime.h>, so compile it as CUDA (nvcc always
# finds the CUDA headers). Each backend TU defines make_backend/backend_name.
set_source_files_properties(src/bench_main.cpp PROPERTIES LANGUAGE CUDA)

add_executable(bench_custom src/bench_main.cpp src/custom.cu)
target_include_directories(bench_custom PRIVATE include)

if(AMGX_LIBRARY)
  add_executable(bench_amgx src/bench_main.cpp src/amgx_backend.cu)
  target_include_directories(bench_amgx PRIVATE include "${AMGX_INCLUDE_DIR}"
                                                  "${AMGX_ROOT}/external/rapidjson/include")
  target_link_directories(bench_amgx PRIVATE "${AMGX_BUILD}")
  # amgxsh depends on cusparse/cublas/cusolver at link time.
  target_link_libraries(bench_amgx PRIVATE amgxsh cusparse cublas cusolver)
  # AMGX wraps CUDA's thrust into namespace `amgx` and needs extended lambdas,
  # matching the flags its own library was built with.
  target_compile_definitions(bench_amgx PRIVATE THRUST_CUB_WRAPPED_NAMESPACE=amgx)
  target_compile_options(bench_amgx PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
endif()

add_executable(test_input tests/test_input.cpp)
target_include_directories(test_input PRIVATE include)
