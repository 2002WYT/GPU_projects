# GPU Jacobi 迭代求解器

本项目使用 CUDA 在 GPU 上实现 Jacobi 迭代法，用于求解稀疏线性方程组

$$
Ax=b.
$$

矩阵使用 CSR（Compressed Sparse Row）格式存储。Jacobi 更新、残差计算、收敛判断和停止控制均在 GPU 内完成，CPU 只负责：

1. 构造或读入矩阵；
2. 申请显存并复制数据；
3. 启动一次 cooperative kernel；
4. 在计算结束后获取最终解和迭代状态。

---

## 1. Jacobi 迭代公式

将矩阵 $A$ 分解为

$$
A=D+L+U,
$$

其中 $D$ 是对角部分，$L$ 和 $U$ 分别是严格下三角和严格上三角部分。

Jacobi 迭代格式为

$$
x_i^{(k+1)}
=
\frac{
b_i-\sum_{j\ne i}A_{ij}x_j^{(k)}
}{
A_{ii}
}.
$$

每个未知量 $x_i^{(k+1)}$ 只依赖上一轮的解 $x^{(k)}$，因此各行可以并行计算。

程序使用两个解向量：

```text
x_old：保存第 k 轮结果
x_new：保存第 k+1 轮结果
```

每轮结束后交换两个指针，避免额外的数据复制。

---

## 2. 收敛判断

程序使用相对残差作为停止条件：

$$
\frac{\|b-Ax^{(k)}\|_2}{\|b\|_2}
\leq \text{tol}.
$$

其中

$$
\|b\|_2
=
\sqrt{\sum_i b_i^2}.
$$

当 $b=0$ 时，为避免除以零，程序使用绝对残差进行判断。

---

## 3. 程序结构

```text
.
├── include
│   ├── head.cuh
│   └── jacobi.cuh
├── src
│   ├── jacobi.cu
│   └── main.cu
└── CMakeLists.txt
```

各文件作用如下：

| 文件 | 作用 |
|---|---|
| `head.cuh` | CUDA 错误检查宏及公共头文件 |
| `jacobi.cuh` | `JacobiState` 和求解函数声明 |
| `jacobi.cu` | GPU Jacobi kernel 和主机端封装 |
| `main.cu` | 构造测试矩阵、调用求解器并验证结果 |

---

## 4. GPU 状态结构

```cpp
struct JacobiState
{
    double b_norm;
    double residual_sq;
    double relative_residual;

    int iterations;
    int converged;
    int stop;
    int zero_diagonal;
};
```

成员含义：

| 成员 | 含义 |
|---|---|
| `b_norm` | $\|b\|_2$ |
| `residual_sq` | $\|b-Ax\|_2^2$ |
| `relative_residual` | 当前相对残差 |
| `iterations` | 实际完成的迭代次数 |
| `converged` | 是否满足收敛条件 |
| `stop` | GPU 内部停止标志 |
| `zero_diagonal` | 是否检测到零对角元 |

---

## 5. Block 内归约

为了计算向量范数和残差，程序先让每个线程计算局部平方和，再在 block 内使用共享内存归约。

```cpp
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
```

例如，每个线程先计算：

```cpp
double local_sum = 0.0;

for (int i = global_tid; i < n; i += stride)
{
    local_sum += b[i] * b[i];
}
```

然后在 block 内求和：

```cpp
double block_sum = block_reduce_sum(local_sum);
```

每个 block 只由线程 `tid == 0` 将结果累加到全局状态：

```cpp
if (tid == 0)
{
    atomicAdd(&state->b_norm, block_sum);
}
```

这样可以显著减少全局原子操作次数。

---

## 6. Cooperative Groups 全局同步

kernel 使用：

```cpp
namespace cg = cooperative_groups;
```

并在 kernel 内创建整个 grid 的线程组：

```cpp
cg::grid_group grid = cg::this_grid();
```

普通的

```cpp
__syncthreads();
```

只能同步同一个 block 内的线程，而

```cpp
grid.sync();
```

可以同步 cooperative kernel 中的所有 block。

例如，在计算完所有 block 的局部范数后：

```cpp
if (tid == 0)
{
    atomicAdd(&state->b_norm, block_sum);
}

grid.sync();
```

必须等待全部 block 完成累加后，才能安全执行：

```cpp
if (grid.thread_rank() == 0)
{
    state->b_norm = sqrt(state->b_norm);
}
```

---

## 7. 计算初始残差

在正式迭代前，程序先计算初始解 $x_0$ 的相对残差：

```cpp
if (grid.thread_rank() == 0)
{
    state->residual_sq = 0.0;
}

grid.sync();

double local_sum = 0.0;

for (int i = global_tid; i < n; i += stride)
{
    double Ax_i = 0.0;

    for (int jj = row_ptr[i];
         jj < row_ptr[i + 1];
         ++jj)
    {
        int col = col_idx[jj];
        Ax_i += values[jj] * x0[col];
    }

    double r_i = b[i] - Ax_i;
    local_sum += r_i * r_i;
}
```

之后通过 block 归约和原子累加得到：

$$
\|b-Ax_0\|_2^2.
$$

如果初始解已经满足容差：

```cpp
if (state->relative_residual <= tol)
{
    state->converged = 1;
    state->stop = 1;
}
```

则直接复制初始解并退出：

```cpp
if (state->stop)
{
    for (int i = global_tid; i < n; i += stride)
    {
        x_result[i] = x0[i];
    }

    return;
}
```

---

## 8. Jacobi 主迭代

核心迭代如下：

```cpp
for (int iter = 1; iter <= max_iter; ++iter)
{
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
            x_new[i] = x_old[i];
            atomicExch(&state->zero_diagonal, 1);
        }
    }

    grid.sync();
```

每个线程处理矩阵的一行或多行，使用 CSR 格式遍历该行的非零元素。

---

## 9. 零对角元检测

Jacobi 迭代要求：

$$
A_{ii}\neq 0.
$$

如果某个线程检测到零对角元：

```cpp
atomicExch(&state->zero_diagonal, 1);
```

`atomicExch` 会以原子方式将全局标志设置为 `1`。

同步后，所有线程检查：

```cpp
if (state->zero_diagonal)
{
    if (grid.thread_rank() == 0)
    {
        state->relative_residual = CUDART_INF;
        state->iterations = iter - 1;
        state->converged = 0;
        state->stop = 1;
    }

    x_latest = x_old;

    grid.sync();
    break;
}
```

此时当前轮结果无效，程序返回上一轮有效解。

---

## 10. 每轮残差计算

每次 Jacobi 更新完成后，程序计算：

$$
r=b-Ax_{\text{new}}.
$$

```cpp
if (grid.thread_rank() == 0)
{
    state->residual_sq = 0.0;
}

grid.sync();

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

block_sum = block_reduce_sum(local_sum);

if (tid == 0)
{
    atomicAdd(&state->residual_sq, block_sum);
}

grid.sync();
```

随后由整个 grid 的第一个线程计算相对残差并判断是否收敛：

```cpp
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
```

---

## 11. 交换新旧向量

当当前轮没有收敛时：

```cpp
double* temp = x_old;
x_old = x_new;
x_new = temp;
```

不需要复制整个向量，只交换指针。

例如：

```text
第 0 轮：
x_old -> x0
x_new -> x1

第 1 轮后：
x_old -> x1
x_new -> x0

第 2 轮后：
x_old -> x0
x_new -> x1
```

---

## 12. 主机端求解接口

```cpp
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
    JacobiState* host_state);
```

注意：该函数接收的矩阵、右端项和解向量必须都是 GPU 指针。

调用示例：

```cpp
JacobiState state{};

CHECK_CUDA(jacobi_gpu_solve(
    n,
    d_row_ptr,
    d_col_idx,
    d_values,
    d_b,
    d_x0,
    d_x1,
    d_x_result,
    max_iter,
    tol,
    &state));
```

---

## 13. Cooperative Kernel 启动

由于 kernel 内使用了 `grid.sync()`，不能使用普通的：

```cpp
kernel<<<grid_size, block_size>>>();
```

而需要使用：

```cpp
cudaLaunchCooperativeKernel(
    reinterpret_cast<void*>(jacobi_cooperative_kernel),
    dim3(grid_size),
    dim3(block_size),
    kernel_arguments,
    shared_memory_size,
    nullptr);
```

程序先检查 GPU 是否支持 cooperative launch：

```cpp
int cooperative_supported = 0;

cudaDeviceGetAttribute(
    &cooperative_supported,
    cudaDevAttrCooperativeLaunch,
    device);
```

然后通过 occupancy API 计算一个安全的 grid 大小：

```cpp
constexpr int block_size = 256;

int active_blocks_per_sm = 0;

cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &active_blocks_per_sm,
    jacobi_cooperative_kernel,
    block_size,
    shared_memory_size);
```

cooperative kernel 的所有 block 必须能够同时驻留在 GPU 上，因此：

```cpp
int maximum_resident_blocks =
    active_blocks_per_sm * sm_count;
```

最终网格大小为：

```cpp
int grid_size =
    std::min(blocks_needed, maximum_resident_blocks);
```

kernel 内部使用 grid-stride loop，因此即使 block 数少于处理全部元素所需的数量，也可以覆盖整个向量。

---

## 14. GPU 状态内存

GPU 端状态结构通过以下代码申请：

```cpp
JacobiState* d_state = nullptr;

cudaMalloc(
    reinterpret_cast<void**>(&d_state),
    sizeof(JacobiState));
```

含义是：

1. 在 GPU 显存中申请一块 `sizeof(JacobiState)` 大小的空间；
2. 将该显存地址保存到 `d_state`；
3. kernel 中所有线程都可以通过 `d_state` 访问迭代状态。

kernel 结束后，只需要复制一个很小的状态结构体回 CPU：

```cpp
cudaMemcpy(
    host_state,
    d_state,
    sizeof(JacobiState),
    cudaMemcpyDeviceToHost);
```

---

## 15. 测试矩阵

主程序构造严格对角占优的三对角矩阵：

$$
A=
\operatorname{tridiag}(-1,4,-1).
$$

即：

$$
A=
\begin{bmatrix}
4 & -1 \\
-1 & 4 & -1 \\
& \ddots & \ddots & \ddots \\
&& -1 & 4 & -1\\
&&& -1 & 4
\end{bmatrix}.
$$

真实解设置为：

$$
x_{\text{true}}
=
[1,2,\ldots,n]^T.
$$

然后在 CPU 上构造：

$$
b=Ax_{\text{true}}.
$$

Jacobi 求解完成后，通过最大绝对误差验证结果：

$$
\max_i
\left|
x_i-x_{\text{true},i}
\right|.
$$

---

## 16. 编译

示例 `CMakeLists.txt`：

```cmake
cmake_minimum_required(VERSION 3.18)

project(gpu_jacobi LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 70)

add_executable(main
    src/main.cu
    src/jacobi.cu
)

target_include_directories(main PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(main PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
```

对于 NVIDIA Tesla V100，使用：

```cmake
set(CMAKE_CUDA_ARCHITECTURES 70)
```

编译：

```bash
mkdir -p build
cd build

cmake ..
make -j
```

---

## 17. 运行

默认矩阵规模：

```bash
./main
```

指定矩阵规模：

```bash
./main 100000
```

---

## 18. 示例输出

输出格式类似：

```text
--------------------------------------------------
GPU Jacobi Example
Solving A x = b, A = tridiag(-1, 4, -1)
n = 10000, nnz = 29998
--------------------------------------------------
Jacobi iterations       = 21
Relative residual       = 7.8e-07
Converged               = Yes
Zero diagonal detected  = No
Maximum absolute error  = ...
--------------------------------------------------
First 10 entries of Jacobi solution:
x_jacobi[0] = ...
x_jacobi[1] = ...
...
```

实际迭代次数和误差与 `n`、`tol`、初始解及浮点计算环境有关。

---

## 19. 当前实现特点

- 使用 CSR 格式存储稀疏矩阵；
- Jacobi 更新完全在 GPU 上执行；
- 初始残差和每轮残差均在 GPU 上计算；
- 使用共享内存完成 block 内归约；
- 每个 block 只执行一次全局 `atomicAdd`；
- 使用 `atomicExch` 检测零对角元；
- 使用 cooperative groups 完成 grid 级同步；
- CPU 不参与每轮收敛判断；
- 新旧解通过交换指针切换，不进行整向量复制；
- 求解结束后只返回最终解和一个小型状态结构体。

---

## 20. 注意事项

1. GPU 必须支持 cooperative launch。
2. `block_size` 应为 2 的幂，以满足当前共享内存归约实现。
3. Jacobi 方法通常要求矩阵严格对角占优或满足其他收敛条件。
4. 对角元不能为零。
5. 当前实现每轮都计算一次完整残差，便于判断收敛，但会增加一次 SpMV 的开销。
6. 对于一般大规模稀疏矩阵，可以通过降低残差检查频率进一步提升性能。
