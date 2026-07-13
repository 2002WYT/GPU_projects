# CUDA 共轭梯度法（CG）求解器

本项目使用 CUDA 在 GPU 上实现共轭梯度法（Conjugate Gradient，CG），用于求解大型稀疏线性方程组

$$
Ax=b.
$$

代码使用 CSR 格式存储稀疏矩阵，并把计算量最大的操作放到 GPU 上完成，包括：

- 稀疏矩阵向量乘法 `SpMV`；
- 向量点积 `Dot`；
- 向量更新 `AXPY`；
- 搜索方向更新 `AXPBY`。

这份 README 按照“先理解算法，再阅读代码”的顺序介绍整个程序。

---

## 1. CG 能解决什么问题

普通 CG 要求矩阵 \(A\) 满足：

1. \(A\) 是方阵；
2. \(A\) 是对称矩阵；
3. \(A\) 是正定矩阵。

也就是对于任意非零向量 \(z\)，都有

$$
z^TAz>0.
$$

代码中的测试矩阵为

$$
A=
\begin{bmatrix}
4 & -1 \\
-1 & 4 & -1 \\
& -1 & 4 & -1 \\
& & \ddots & \ddots & \ddots \\
& & & -1 & 4
\end{bmatrix}.
$$

这是一个对称正定三对角矩阵，因此适合使用 CG。

如果矩阵不是对称正定矩阵，普通 CG 可能无法收敛，甚至会在计算

$$
p^TAp
$$

时出现非正数。对于一般非对称矩阵，可以使用 GMRES。

---

## 2. 项目中的主要文件

推荐按下面的方式组织代码：

```text
project/
├── include/
│   ├── head.cuh
│   └── CG.cuh
├── src/
│   ├── maths.cu
│   ├── spmv.cu
│   └── CG.cu
└── ex_CG.cu
```

各文件作用如下。

### `head.cuh`

保存整个项目都会使用的数据结构、错误检查宏和函数声明，例如：

- `CSRMatrix`；
- `SolverOptions`；
- `CHECK_CUDA`；
- `Dot_device`；
- `CheckCSR`；
- 各种向量运算和 SpMV 接口。

### `CG.cuh`

只声明 CG 求解器接口：

```cpp
int CG_gpu(
    const CSRMatrix& A,
    const std::vector<double>& b,
    std::vector<double>& x,
    const SolverOptions& options);
```

这里：

- `A` 是 CSR 稀疏矩阵；
- `b` 是右端向量；
- `x` 既是初始解，也是最终输出的近似解；
- `options` 保存最大迭代次数、收敛容差和 CUDA 启动参数。

### 向量运算源文件

实现以下 GPU 核函数：

```cpp
vecscale_kernel
axpy_kernel
axpby_kernel
dot_kernel
```

### SpMV 源文件

实现 CSR 稀疏矩阵向量乘法：

```cpp
spmv_csr_scalar_kernel
```

### `CG.cu`

实现完整 CG 迭代，是本项目的核心。

### `ex_CG.cu`

完成以下工作：

1. 初始化 CUDA；
2. 构造测试矩阵；
3. 构造精确解；
4. 计算右端项；
5. 设置求解参数；
6. 调用 `CG_gpu`；
7. 检查最终误差。

---

## 3. 主要数据结构

### 3.1 CSRMatrix

```cpp
struct CSRMatrix
{
    int n;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<double> values;
};
```

CSR 使用三个数组保存稀疏矩阵。

#### `row_ptr`

`row_ptr[i]` 表示第 `i` 行元素在 `col_idx` 和 `values` 中的起始位置。

第 `i` 行的非零元素范围是：

```cpp
for (int jj = row_ptr[i];
     jj < row_ptr[i + 1];
     ++jj)
```

#### `col_idx`

保存每个非零元素所在的列号。

#### `values`

保存每个非零元素的数值。

例如矩阵

$$
\begin{bmatrix}
4 & -1 & 0 \\
-1 & 4 & -1 \\
0 & -1 & 4
\end{bmatrix}
$$

可以写为：

```text
row_ptr = [0, 2, 5, 7]
col_idx = [0, 1, 0, 1, 2, 1, 2]
values  = [4, -1, -1, 4, -1, -1, 4]
```

---

### 3.2 SolverOptions

```cpp
struct SolverOptions
{
    int max_iterations;
    int residual_check_interval;
    double relative_tolerance;
    int block_size;
    int grid_size;
};
```

各参数含义如下：

| 参数 | 含义 |
|---|---|
| `max_iterations` | 最大 CG 迭代次数 |
| `residual_check_interval` | 每隔多少次输出一次残差 |
| `relative_tolerance` | 相对残差收敛容差 |
| `block_size` | 每个 CUDA block 的线程数 |
| `grid_size` | CUDA block 数量 |

代码要求 `block_size` 是 2 的幂，例如：

```text
128、256、512
```

这是因为 `dot_kernel` 使用二分规约计算点积。

---

## 4. CG 算法的核心思想

给定初始解 \(x_0\)，首先计算初始残差：

$$
r_0=b-Ax_0.
$$

把第一个搜索方向设置为：

$$
p_0=r_0.
$$

第 \(k\) 次迭代包含以下步骤。

### 第一步：计算矩阵向量乘法

$$
Ap_k=A p_k.
$$

### 第二步：计算步长

$$
\alpha_k=
\frac{r_k^Tr_k}
     {p_k^TAp_k}.
$$

### 第三步：更新解

$$
x_{k+1}=x_k+\alpha_kp_k.
$$

### 第四步：更新残差

$$
r_{k+1}=r_k-\alpha_kAp_k.
$$

### 第五步：计算搜索方向系数

$$
\beta_k=
\frac{r_{k+1}^Tr_{k+1}}
     {r_k^Tr_k}.
$$

### 第六步：更新搜索方向

$$
p_{k+1}=r_{k+1}+\beta_kp_k.
$$

不断重复，直到相对残差足够小：

$$
\frac{\lVert r_k\rVert_2}
     {\lVert b\rVert_2}
\leq \text{tolerance}.
$$

---

## 5. 数学变量与代码变量的对应关系

| 数学符号 | GPU 变量 | 含义 |
|---|---|---|
| \(A\) | `d_row_ptr`、`d_col_idx`、`d_values` | CSR 稀疏矩阵 |
| \(b\) | `d_b` | 右端向量 |
| \(x\) | `d_x` | 当前近似解 |
| \(r\) | `d_r` | 当前残差 |
| \(p\) | `d_p` | 当前搜索方向 |
| \(Ap\) | `d_Ap` | 矩阵向量乘积 |
| \(r^Tr\) | `rr_old`、`rr_new` | 残差平方 |
| \(p^TAp\) | `pAp` | CG 步长公式的分母 |
| \(\alpha\) | `alpha` | 解更新步长 |
| \(\beta\) | `beta` | 搜索方向更新系数 |

理解这张表以后，阅读 `CG.cu` 会容易很多。

---

## 6. `CG_gpu` 的完整代码流程

### 6.1 检查输入是否合法

函数开始调用：

```cpp
CheckCSR(A);
```

它会检查：

- `A.n` 是否为正数；
- `row_ptr.size()` 是否等于 `n + 1`；
- `col_idx.size()` 是否等于 `values.size()`；
- `row_ptr[0]` 是否为 0；
- `row_ptr[n]` 是否等于非零元个数；
- 行偏移是否单调不减；
- 列下标是否越界。

随后检查：

```cpp
b.size() == A.n
x.size() == A.n
```

如果 `x` 是空向量，代码会自动把它初始化为全 0：

```cpp
if (x.empty())
{
    x.assign(A.n, 0.0);
}
```

这相当于使用初始解

$$
x_0=0.
$$

---

### 6.2 分配 GPU 内存

代码为以下数据分配显存：

```cpp
d_row_ptr
d_col_idx
d_values
d_b
d_x
d_r
d_p
d_Ap
d_partial_sums
```

其中：

- 矩阵和向量在整个求解过程中一直保留在 GPU；
- `d_partial_sums` 用于保存每个 block 计算出的点积部分和。

这比每次运算都重新 `cudaMalloc` 和 `cudaMemcpy` 高效得多。

---

### 6.3 把矩阵和向量复制到 GPU

例如：

```cpp
cudaMemcpy(
    d_values,
    A.values.data(),
    nnz * sizeof(double),
    cudaMemcpyHostToDevice);
```

只在求解开始时复制一次。

最终解也只在求解结束后复制回 CPU：

```cpp
cudaMemcpy(
    x.data(),
    d_x,
    n * sizeof(double),
    cudaMemcpyDeviceToHost);
```

---

### 6.4 计算初始残差

首先计算：

```cpp
d_Ap = A * d_x
```

对应代码：

```cpp
spmv_csr_scalar_kernel<<<grid_size, block_size>>>(
    n,
    d_row_ptr,
    d_col_idx,
    d_values,
    d_x,
    d_Ap);
```

然后复制：

```cpp
d_r = d_b
```

再执行：

```cpp
d_r = d_r - d_Ap
```

对应：

```cpp
axpy_kernel<<<grid_size, block_size>>>(
    n,
    -1.0,
    d_Ap,
    d_r);
```

因为 AXPY 的计算形式是

$$
y=\alpha x+y,
$$

所以这里令：

```text
alpha = -1
x     = d_Ap
y     = d_r
```

得到：

$$
r=b-Ax.
$$

---

### 6.5 初始化搜索方向

```cpp
cudaMemcpy(
    d_p,
    d_r,
    n * sizeof(double),
    cudaMemcpyDeviceToDevice);
```

也就是：

$$
p_0=r_0.
$$

这里使用的是 GPU 到 GPU 的复制，不经过 CPU。

---

### 6.6 计算初始残差大小

调用：

```cpp
Dot_device(
    n,
    d_r,
    d_r,
    d_partial_sums,
    rr_old,
    block_size,
    grid_size);
```

得到：

$$
rr_{\text{old}}=r^Tr=\lVert r\rVert_2^2.
$$

相对残差为：

```cpp
relative_residual =
    std::sqrt(rr_old) / residual_normalizer;
```

即：

$$
\frac{\lVert r\rVert_2}
     {\lVert b\rVert_2}.
$$

---

### 6.7 每轮 CG 迭代

### 计算 `d_Ap = A*d_p`

```cpp
spmv_csr_scalar_kernel<<<grid_size, block_size>>>(
    n,
    d_row_ptr,
    d_col_idx,
    d_values,
    d_p,
    d_Ap);
```

这是每轮 CG 中计算量最大的步骤。

---

### 计算 `pAp`

```cpp
Dot_device(
    n,
    d_p,
    d_Ap,
    d_partial_sums,
    pAp,
    block_size,
    grid_size);
```

得到：

$$
p^TAp.
$$

对于对称正定矩阵，它应当严格大于 0。

代码因此检查：

```cpp
if (!(pAp > 0.0))
```

如果这个条件不满足，通常说明：

- 输入矩阵不是正定矩阵；
- 输入矩阵不是预期的对称矩阵；
- 数值计算发生严重异常。

---

### 计算 `alpha`

```cpp
const double alpha =
    rr_old / pAp;
```

对应：

$$
\alpha_k=
\frac{r_k^Tr_k}
     {p_k^TAp_k}.
$$

---

### 更新解

```cpp
axpy_kernel<<<grid_size, block_size>>>(
    n,
    alpha,
    d_p,
    d_x);
```

得到：

$$
x_{k+1}=x_k+\alpha_kp_k.
$$

---

### 更新残差

```cpp
axpy_kernel<<<grid_size, block_size>>>(
    n,
    -alpha,
    d_Ap,
    d_r);
```

得到：

$$
r_{k+1}=r_k-\alpha_kAp_k.
$$

---

### 计算新的残差平方

```cpp
Dot_device(
    n,
    d_r,
    d_r,
    d_partial_sums,
    rr_new,
    block_size,
    grid_size);
```

得到：

$$
rr_{\text{new}}=
r_{k+1}^Tr_{k+1}.
$$

---

### 判断是否收敛

```cpp
relative_residual =
    std::sqrt(rr_new) /
    residual_normalizer;
```

当

```cpp
relative_residual <=
    options.relative_tolerance
```

时停止迭代。

---

### 计算 `beta`

```cpp
const double beta =
    rr_new / rr_old;
```

对应：

$$
\beta_k=
\frac{r_{k+1}^Tr_{k+1}}
     {r_k^Tr_k}.
$$

---

### 更新搜索方向

```cpp
axpby_kernel<<<grid_size, block_size>>>(
    n,
    1.0,
    beta,
    d_r,
    d_p);
```

`axpby_kernel` 的形式为：

$$
y=\alpha x+\beta y.
$$

这里：

```text
x     = r
y     = p
alpha = 1
beta  = beta
```

因此得到：

$$
p_{k+1}=r_{k+1}+\beta_kp_k.
$$

---

## 7. GPU 核函数是做什么的

### 7.1 `axpy_kernel`

```cpp
y[i] = alpha * x[i] + y[i];
```

完成：

$$
y=\alpha x+y.
$$

CG 中用于：

- 更新解；
- 更新残差；
- 构造初始残差。

---

### 7.2 `axpby_kernel`

```cpp
y[i] =
    alpha * x[i] +
    beta * y[i];
```

完成：

$$
y=\alpha x+\beta y.
$$

CG 中用于更新搜索方向：

$$
p=r+\beta p.
$$

---

### 7.3 `dot_kernel`

每个线程计算若干个元素乘积：

```cpp
local_sum += x[i] * y[i];
```

然后同一个 block 中的线程通过共享内存规约：

```cpp
sdata[tid] += sdata[tid + offset];
```

每个 block 最后输出一个部分和：

```cpp
partial_sums[blockIdx.x] = sdata[0];
```

---

### 7.4 `Dot_device`

`Dot_device` 接收的 `d_x` 和 `d_y` 已经位于 GPU，因此不会再次复制完整向量。

它的流程是：

1. GPU 计算每个 block 的部分和；
2. 只把 `grid_size` 个部分和复制到 CPU；
3. CPU 对这些部分和求和。

因此，复制的数据量不再是 \(n\) 个元素，而只是 `grid_size` 个元素。

不过，每次调用仍会产生一次 GPU 到 CPU 的同步。这是当前实现的重要性能瓶颈之一。

---

### 7.5 `spmv_csr_scalar_kernel`

该核函数逐行计算：

$$
y_i=
\sum_{j=A.row\_ptr[i]}^{A.row\_ptr[i+1]-1}
A.values[j]\,
x_{A.col\_idx[j]}.
$$

也就是：

$$
y=Ax.
$$

在 CG 中，每轮只需要进行一次 SpMV：

$$
Ap_k=A p_k.
$$

---

## 8. 为什么最后还要重新计算真实残差

迭代过程中，残差使用递推公式更新：

$$
r_{k+1}=r_k-\alpha_kAp_k.
$$

这种做法比较快，因为不需要每轮重新计算 \(Ax\)。

但是经过很多次浮点运算后，递推残差可能与真正的

$$
b-Ax
$$

出现少量偏差。

因此程序结束前重新执行：

```cpp
d_Ap = A * d_x
d_r  = d_b - d_Ap
```

并输出：

```text
Final true relative residual ||b-Ax||/||b||
```

判断求解是否可靠时，最终的真实残差比递推残差更值得参考。

---

## 9. 主程序如何构造测试问题

主程序先设置精确解：

```cpp
x_exact[i] =
    static_cast<double>(i + 1);
```

也就是：

$$
x_{\text{exact}}=
[1,2,3,\ldots,n]^T.
$$

然后在 CPU 上计算：

$$
b=Ax_{\text{exact}}.
$$

求解器并不知道 `x_exact`，它只接收 \(A\)、\(b\) 和初始解 \(x_0\)。

求解结束后比较：

```cpp
x[i] - x_exact[i]
```

从而检查程序是否正确。

---

## 10. 如何设置求解参数

示例：

```cpp
SolverOptions options;

options.max_iterations = 1000;
options.residual_check_interval = 10;
options.relative_tolerance = 1e-10;
options.block_size = 256;

options.grid_size =
    (n + options.block_size - 1) /
    options.block_size;
```

### `max_iterations`

最大迭代次数。

如果矩阵条件数较大，CG 可能需要更多迭代。

### `relative_tolerance`

例如：

```cpp
options.relative_tolerance = 1e-10;
```

表示要求：

$$
\frac{\lVert b-Ax\rVert_2}
     {\lVert b\rVert_2}
\leq 10^{-10}.
$$

### `residual_check_interval`

只控制打印频率，不改变实际计算。

因为当前代码每一轮都必须计算 `rr_new` 来计算 \(\beta\)，所以即使设置为 10，点积仍然每轮执行。

### `block_size`

推荐先使用：

```cpp
options.block_size = 256;
```

### `grid_size`

目前代码使用：

```cpp
grid_size =
    (n + block_size - 1) /
    block_size;
```

由于核函数内部使用 grid-stride loop，即使适当限制 `grid_size`，也能遍历完整向量。

大规模问题中，可以尝试把 grid 数限制为 GPU SM 数量的若干倍，以减少点积部分和数量。

---

## 11. 输出结果怎么看

典型输出：

```text
CG initial relative residual = 1.000000000000e+00
CG iteration      1, alpha = ..., relative residual = ...
CG iteration     10, alpha = ..., relative residual = ...
CG converged after 14 iterations.
Final recursive relative residual = ...
Final true relative residual ||b-Ax||/||b|| = ...
```

### `initial relative residual`

初始解的相对残差。

当 \(x_0=0\) 时：

$$
r_0=b,
$$

所以初始相对残差通常为 1。

### `recursive relative residual`

由递推残差 `d_r` 得到。

### `true relative residual`

重新计算 \(b-Ax\) 得到，更适合判断最终求解质量。

### `Maximum absolute error`

$$
\max_i |x_i-x_{\text{exact},i}|.
$$

### `Relative solution error`

$$
\frac{\lVert x-x_{\text{exact}}\rVert_2}
     {\lVert x_{\text{exact}}\rVert_2}.
$$

残差小不一定意味着每个分量的绝对误差都特别小，特别是当精确解本身数值很大时。因此更推荐同时观察相对解误差。

---

## 12. 编译注意事项

`CG.cu` 只声明并调用下面的核函数：

```cpp
axpy_kernel
axpby_kernel
spmv_csr_scalar_kernel
```

因此编译目标必须同时链接真正定义这些核函数的 `.cu` 文件。

CMake 中可写成类似：

```cmake
add_executable(cg_main
    main.cu
    CG.cu
    vector_ops.cu
    spmv.cu
)
```

请把 `vector_ops.cu` 和 `spmv.cu` 替换成项目中的实际文件名。

如果使用多个 `.cu` 翻译单元，并且模板核函数定义只放在另一个 `.cu` 文件中，可能出现链接问题。更稳妥的做法是：

- 把模板核函数定义放入 `.cuh`；
- 或为 `double` 添加显式实例化；
- 或直接把该核函数改为只支持 `double` 的普通核函数。

---

## 13. 当前实现的性能瓶颈

这份代码已经避免了每轮复制完整向量，但仍有几个可以继续优化的地方。

### 13.1 点积在 CPU 上完成最终求和

每次 `Dot_device` 都会：

1. 等待 GPU 点积核函数完成；
2. 复制 `grid_size` 个部分和到 CPU；
3. 在 CPU 上完成最终求和。

CG 每轮至少需要两个点积：

$$
p^TAp,\qquad r^Tr.
$$

所以同步次数较多。

后续可以改为：

- 在 GPU 上继续做第二级规约；
- 使用 cuBLAS 的 `cublasDdot`；
- 使用 CUB 的 `DeviceReduce`；
- 使用单 kernel 融合多个向量操作和规约。

### 13.2 向量核函数数量较多

每轮依次启动：

1. SpMV；
2. Dot；
3. AXPY 更新 \(x\)；
4. AXPY 更新 \(r\)；
5. Dot；
6. AXPBY 更新 \(p\)。

对于内存带宽受限问题，可以研究 kernel fusion，减少向量重复读写。

### 13.3 没有预处理器

矩阵条件数较大时，CG 收敛会变慢。

后续可以实现 PCG，并加入：

- Jacobi 预处理；
- 不完全 Cholesky；
- 多重网格预处理。

---

## 14. CG 与 GMRES 的简单区别

| 项目 | CG | GMRES |
|---|---|---|
| 矩阵要求 | 对称正定 | 一般方阵，常用于非对称矩阵 |
| 每轮存储 | 少量向量 | 需要保存 Krylov 基向量 |
| 正交化 | 不需要显式保存全部基 | 需要 Arnoldi 正交化 |
| 内存占用 | 较小 | 较大 |
| 单轮成本 | 较低 | 随子空间维数增加 |
| 常见预处理版本 | PCG | PGMRES / FGMRES |

对于当前三对角对称正定测试矩阵，CG 通常比 GMRES 更合适。

---

## 15. 一句话理解整份代码

这份程序把矩阵 \(A\) 和主要向量长期保存在 GPU 上，每轮通过一次 SpMV、两次点积和若干向量更新完成 CG 迭代，直到

$$
\frac{\lVert b-Ax\rVert_2}
     {\lVert b\rVert_2}
$$

小于给定容差。
