# GPU 多色 Gauss–Seidel 稀疏线性方程组求解器

本项目使用 CUDA 实现面向一般 CSR 稀疏矩阵的多色 Gauss–Seidel 迭代，用于求解线性系统

$$
A x = b.
$$

程序首先在 CPU 上对矩阵对应的无向冲突图进行贪心着色，然后按照颜色顺序在 GPU 上更新未知量。同一种颜色中的行不存在直接依赖，因此可以并行计算。

主要功能包括：

- 一般 CSR 稀疏矩阵输入；
- 对角元提取和倒数预计算；
- CPU 贪心图着色；
- 一个线程处理一行的 CUDA kernel；
- 一个 warp 处理一行的 CUDA kernel；
- CUB 并行残差归约；
- CUDA Graph 重放完整多色 sweep；
- 按固定间隔检查相对残差；
- 支持 Gauss–Seidel 和 SOR 类型更新。

---

## 1. 文件结构

建议将代码组织为：

```text
gauss_seidel/
├── CMakeLists.txt
├── include/
│   └── gs.cuh
├── src/
│   └── gs.cu
├── ex_gs.cu
```

各文件作用如下：

| 文件 | 作用 |
|---|---|
| `include/gs.cuh` | 定义 CSR 矩阵、求解参数、求解结果和公开函数接口 |
| `src/gs.cu` | 实现矩阵预处理、图着色、CUDA kernel、残差计算和求解器 |
| `ex_gs.cu` | 构造测试问题、设置参数、调用求解器并检查结果 |
| `CMakeLists.txt` | 配置 CUDA 编译、GPU 架构和链接方式 |

---

## 2. 数学原理

### 2.1 Gauss–Seidel 迭代

将矩阵分解为

$$
A = D + L + U,
$$

其中：

- \(D\) 是对角部分；
- \(L\) 是严格下三角部分；
- \(U\) 是严格上三角部分。

标准 Gauss–Seidel 迭代满足

$$
(D+L)x^{(k+1)} = b-Ux^{(k)}.
$$

按分量写为

$$
x_i^{(k+1)}=
\frac{1}{a_{ii}}
\left(
b_i-
\sum_{j<i}a_{ij}x_j^{(k+1)}-
\sum_{j>i}a_{ij}x_j^{(k)}
\right).
$$

本程序采用原地松弛更新。对于当前颜色中的行 \(i\)，更新公式为

$$
x_i
\leftarrow
(1-\omega)x_i
+
\frac{\omega}{a_{ii}}
\left(
b_i-\sum_{j\ne i}a_{ij}x_j
\right).
$$

其中，已经处理过的颜色使用本轮更新后的分量，尚未处理的颜色使用上一轮的分量。同一种颜色中的行之间不存在直接非零耦合，因此可以在 GPU 上并行更新。

当 \(\omega=1\) 时，该方法为多色 Gauss–Seidel；当 \(0<\omega<2\) 且 \(\omega\ne1\) 时，为多色 SOR 型迭代。

### 2.2 多色并行

一般 Gauss–Seidel 存在数据依赖，不能直接让所有行同时更新。

程序构造无向冲突图：如果 \(a_{ij}\ne0\) 或 \(a_{ji}\ne0\)，则行 \(i\) 与行 \(j\) 不能具有相同颜色。

设所有行被划分为颜色集合

$$
C_0,C_1,\ldots,C_{p-1}.
$$

一次完整迭代按照以下顺序执行：

```text
for color = 0, 1, ..., p - 1
    并行更新颜色集合 C_color 中的全部行
end for
```

同一种颜色中的行可以并行执行，不同颜色之间保持顺序。

多色 Gauss–Seidel 与固定自然行顺序的 Gauss–Seidel 更新顺序不同，因此两者的迭代过程和迭代次数可能不同，但收敛后求解的仍是同一个线性系统。

---

## 3. CSR 矩阵格式

矩阵使用 CSR 格式保存：

```cpp
struct CSRMatrix
{
    int n = 0;

    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<double> values;
};
```

其中：

- `n`：矩阵阶数；
- `row_ptr`：长度为 `n + 1`；
- `col_idx`：每个非零元的列号；
- `values`：每个非零元的数值；
- `row_ptr[i]` 到 `row_ptr[i + 1] - 1` 对应第 \(i\) 行。

非零元总数为

$$
\operatorname{nnz}(A)=
\texttt{row\_ptr[n]}.
$$

程序要求：

1. \(A\) 必须是方阵；
2. `row_ptr[0]` 必须为 `0`；
3. `row_ptr` 必须单调不减；
4. 所有列下标必须位于 `[0, n)`；
5. 每一行必须具有有限且非零的对角元；
6. `col_idx.size()` 必须等于 `values.size()`；
7. `row_ptr[n]` 必须等于非零元总数。

---

## 4. 公开数据结构

### 4.1 求解参数

```cpp
struct SolverOptions
{
    int max_iterations = 10000;
    int residual_check_interval = 20;
    double relative_tolerance = 1.0e-8;
    double omega = 1.0;
    int block_size = 256;
    int warp_row_threshold = 16;
};
```

参数说明：

| 参数 | 含义 |
|---|---|
| `max_iterations` | 最大迭代次数 |
| `residual_check_interval` | 每隔多少次迭代检查一次残差 |
| `relative_tolerance` | 相对残差收敛容差 |
| `omega` | 松弛参数，`1.0` 表示 Gauss–Seidel |
| `block_size` | CUDA block 中的线程数 |
| `warp_row_threshold` | 切换到 warp-per-row kernel 的平均行长度阈值 |

### 4.2 求解结果

```cpp
struct SolverResult
{
    int iterations = 0;
    int num_colors = 0;

    double relative_residual =
        std::numeric_limits<double>::infinity();

    float elapsed_ms = 0.0f;
};
```

返回信息包括：

- 实际迭代次数；
- 图着色使用的颜色数；
- 最终相对残差；
- GPU 迭代时间。

---

## 5. 矩阵预处理

`prepare_matrix` 会执行以下工作：

1. 检查 CSR 数据是否合法；
2. 合并同一行中的重复对角元；
3. 将对角元从迭代 CSR 中移除；
4. 预计算每个对角元的倒数。

定义

$$
d_i^{-1}=\frac{1}{a_{ii}}.
$$

kernel 中只需执行

```cpp
const double gs_value =
    rhs * diagonal_inverse[row];
```

不需要在每次迭代中重新搜索对角元，也不需要重复执行除法。

预处理后的内部结构为：

```cpp
struct PreparedMatrix
{
    int n = 0;

    std::vector<int> off_row_ptr;
    std::vector<int> off_col_idx;
    std::vector<double> off_values;
    std::vector<double> diagonal_inverse;
};
```

其中 `off_*` 只包含非对角元。

> `PreparedMatrix` 只在 `gs.cu` 内部使用，推荐将它放在 `gs.cu` 中，而不是放在公开头文件 `gs.cuh` 中。

---

## 6. 图着色

程序构造对称冲突图。

当矩阵中存在 \(a_{ij}\ne0\) 时，代码同时加入：

```cpp
adjacency[row].push_back(col);
adjacency[col].push_back(row);
```

随后：

1. 对邻接表排序；
2. 删除重复邻居；
3. 按节点度数从大到小排列；
4. 使用贪心算法分配颜色；
5. 将相同颜色的行连续存储在 `rows_by_color` 中。

颜色偏移由 `color_offsets` 保存。第 `color` 种颜色对应：

```cpp
rows_by_color[
    color_offsets[color]
    ...
    color_offsets[color + 1] - 1
]
```

---

## 7. CUDA kernel

### 7.1 一个线程处理一行

对于平均每行非对角元较少的矩阵，使用：

```cpp
colored_gauss_seidel_kernel
```

核心更新如下：

```cpp
double rhs = b[row];

for (int jj = row_ptr[row];
     jj < row_ptr[row + 1];
     ++jj)
{
    rhs -= values[jj] * x[col_idx[jj]];
}

const double gs_value =
    rhs * diagonal_inverse[row];

x[row] += omega * (gs_value - x[row]);
```

### 7.2 一个 warp 处理一行

当平均每行非对角元数量达到

```cpp
options.warp_row_threshold
```

时，程序使用：

```cpp
colored_gauss_seidel_warp_kernel
```

一个 warp 中的 32 个线程共同处理一行，并通过

```cpp
__shfl_down_sync
```

完成 warp 内归约。

使用该 kernel 时，`block_size` 必须是 32 的整数倍。

默认设置为：

```cpp
options.block_size = 256;
options.warp_row_threshold = 16;
```

---

## 8. 收敛判断

程序使用相对残差作为收敛标准：

$$
\frac{
\lVert b-Ax^{(k)}\rVert_2
}{
\lVert b\rVert_2
}
\le
\texttt{relative\_tolerance}.
$$

示例主程序中设置：

```cpp
options.relative_tolerance = 1.0e-10;
options.residual_check_interval = 20;
```

为了减少 CPU 与 GPU 同步，残差不会每次迭代都计算，而是每隔固定次数检查一次。

残差平方和通过 CUB 完成：

```cpp
cub::DeviceReduce::Sum(...)
```

---

## 9. CUDA Graph

一次完整的多色 sweep 包含多个按颜色顺序启动的 kernel：

```text
color 0 kernel
color 1 kernel
...
color p - 1 kernel
```

程序使用：

```cpp
cudaStreamBeginCapture(...)
cudaStreamEndCapture(...)
cudaGraphInstantiate(...)
```

将一次完整 sweep 捕获为 CUDA Graph。

每次迭代只需调用：

```cpp
cudaGraphLaunch(graph_exec, stream);
```

这样可以减少 CPU 为每种颜色重复启动 kernel 的开销。

---

## 10. 测试问题

示例主程序构造三对角矩阵

$$
A=
\operatorname{tridiag}(-1,4,-1).
$$

即

$$
A=
\begin{pmatrix}
4 & -1 & 0 & \cdots & 0\\
-1 & 4 & -1 & \ddots & \vdots\\
0 & -1 & 4 & \ddots & 0\\
\vdots & \ddots & \ddots & \ddots & -1\\
0 & \cdots & 0 & -1 & 4
\end{pmatrix}.
$$

精确解设置为

$$
x_i=i+1,
\qquad
i=0,1,\ldots,n-1.
$$

右端项通过

$$
b=Ax_{\mathrm{exact}}
$$

生成。

求解结束后，程序计算最大绝对误差：

$$
\max_i
\left|
x_i-x_{i,\mathrm{exact}}
\right|.
$$

---

## 11. 求解器调用方法

```cpp
CSRMatrix A;
std::vector<double> b;
std::vector<double> x(A.n, 0.0);

SolverOptions options;

options.max_iterations = 10000;
options.residual_check_interval = 20;
options.relative_tolerance = 1.0e-10;
options.omega = 1.0;
options.block_size = 256;
options.warp_row_threshold = 16;

const SolverResult result =
    solve_multicolor_gauss_seidel_gpu(
        A,
        b,
        x,
        options);
```

求解结束后，最终解保存在主机端向量 `x` 中。

结果输出示例：

```cpp
std::cout
    << "colors = "
    << result.num_colors
    << '\n'

    << "iterations = "
    << result.iterations
    << '\n'

    << "relative residual = "
    << result.relative_residual
    << '\n'

    << "GPU iteration time = "
    << result.elapsed_ms
    << " ms\n";
```

---

## 12. 编译前必须检查的头文件

当前 `gs.cuh` 中定义了 `CUDA_CHECK`，因此它应包含该宏所依赖的头文件：

```cpp
#pragma once

#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
```

否则在 `main.cu` 中展开 `CUDA_CHECK` 时，可能出现以下类型或函数未声明：

```text
cudaError_t
cudaSuccess
cudaGetErrorString
std::runtime_error
std::string
std::to_string
```

推荐将 `gs.cuh` 的开头修改为：

```cpp
#pragma once

#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
```

---

## 13. 推荐的函数声明与定义

头文件 `gs.cuh` 中只保留函数声明：

```cpp
SolverResult solve_multicolor_gauss_seidel_gpu(
    const CSRMatrix& A,
    const std::vector<double>& b,
    std::vector<double>& x,
    const SolverOptions& options);
```

源文件 `gs.cu` 中写函数定义：

```cpp
SolverResult solve_multicolor_gauss_seidel_gpu(
    const CSRMatrix& A,
    const std::vector<double>& b,
    std::vector<double>& x,
    const SolverOptions& options)
{
    // ...
}
```

不要在 `.cu` 文件定义中再次写：

```cpp
const SolverOptions& options = {}
```

如果需要默认参数，应只在头文件声明中添加一次：

```cpp
const SolverOptions& options = {}
```

---

## 14. CMake 配置

下面的配置适用于 NVIDIA Tesla V100，其计算能力为 7.0：

```cmake
cmake_minimum_required(VERSION 3.18)

project(multicolor_gauss_seidel
    LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 70)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)

add_executable(main
    src/main.cu
    src/gs.cu
)

target_include_directories(main
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(main
    PRIVATE
        CUDA::cudart
)
```

编译：

```bash
cmake -S . -B build
cmake --build build -j
```

运行默认规模：

```bash
./build/main
```

指定矩阵规模：

```bash
./build/main 1000000
```

---

## 15. 输出格式

程序输出格式如下：

```text
n                  = 1000000
colors             = 2
iterations          = ...
relative residual   = ...
max absolute error  = ...
GPU iteration time  = ... ms
```

对于当前三对角矩阵，对称冲突图通常只需要两种颜色。

---

## 16. 注意事项

### 16.1 收敛性

算法实现正确不代表任意矩阵都能收敛。常见的充分条件包括：

- \(A\) 对称正定；
- \(A\) 严格对角占优。

### 16.2 图着色开销

当前图着色在 CPU 上执行，并使用：

```cpp
std::vector<std::vector<int>>
```

保存邻接表。

对于超大规模矩阵，图着色可能消耗较多内存和预处理时间。如果矩阵结构固定并需要求解多个右端项，应复用预处理和着色结果。

### 16.3 多色更新顺序

多色 Gauss–Seidel 不等同于固定自然行顺序的 lexicographic Gauss–Seidel。两者的迭代矩阵、迭代轨迹和迭代次数可能不同。

### 16.4 CUB 的 clangd 红线

代码使用：

```cpp
#include <cub/device/device_reduce.cuh>
```

部分 clangd 版本可能在 CUDA 或 Thrust 头文件中显示：

```text
dynamic initialization is not supported
```

如果 `nvcc` 可以正常编译，则通常只是 clangd 对 CUDA 头文件的误报，不影响程序运行。

---
