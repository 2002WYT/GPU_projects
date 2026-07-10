# 五种 GPU SpMV 算法性能对比

本项目使用 CUDA 实现并比较五种 CSR 格式稀疏矩阵向量乘法（Sparse Matrix-Vector Multiplication，SpMV）算法：

1. **CSR Scalar**
2. **CSR Vector**
3. **CSR Adaptive**
4. **PCSR**
5. **LightSpMV-style 动态调度**

程序会自动生成一个行长不均匀的 CSR 稀疏矩阵，分别运行五种 GPU 实现，并输出：

- 平均运行时间 `avg_ms`
- 计算性能 `GFLOP/s`
- 近似有效带宽 `approx_GB/s`
- 相对于 CPU 参考结果的最大绝对误差
- 相对于 CPU 参考结果的最大相对误差
- 当前测试中运行最快的算法
- 总体正确性检查结果

该程序适合用于：

- 学习 CSR SpMV 的不同 GPU 并行化方式；
- 比较规则行长和不规则行长对算法性能的影响；
- 使用 Nsight Compute 分析访存、占用率和线程束停顿；
- 验证 CSR Adaptive、PCSR 和动态调度算法的基本实现思路。

---

## 1. 项目文件

```text
demo01/
├── CMakeLists.txt
├── spmv_compare.md
├── spmv_compare_complete.cu
└── spmv_result.md
```

其中：

- `CMakeLists.txt`：CMake 编译配置；
- `spmv_compare.md`：项目说明和使用方法。
- `spmv_compare_complete.cu`：完整程序，包括矩阵构造、CPU 参考计算、五种 GPU kernel、计时、误差检查和结果输出；
- `spmv_result.md`：程序运行的结果与性能分析

程序不依赖 cuSPARSE、cuDSS 或其他第三方数值库，只需要 CUDA Toolkit、支持 CUDA 的 NVIDIA GPU、CMake 和 C++17 编译环境。

---

## 2. SpMV 问题

程序计算：

```text
y = A x
```

其中：

- `A` 是 `n × n` 的稀疏矩阵；
- `x` 是长度为 `n` 的稠密向量；
- `y` 是长度为 `n` 的结果向量；
- `A` 使用 CSR（Compressed Sparse Row）格式保存。

CSR 使用三个数组：

```text
row_ptr : 每一行非零元在数组中的起止位置
col_idx : 每个非零元对应的列号
values  : 非零元数值
```

第 `i` 行的非零元范围为：

```cpp
row_ptr[i] <= jj < row_ptr[i + 1]
```

因此一行的 SpMV 计算为：

```cpp
y[i] += values[jj] * x[col_idx[jj]];
```

一个非零元通常按一次乘法和一次加法估计，因此一次 SpMV 的计算量近似为：

```text
2 × nnz FLOPs
```

这里的 `nnz` 表示矩阵非零元总数。

---

## 3. 五种算法简介

### 3.1 CSR Scalar

CSR Scalar 使用一个 CUDA 线程处理一整行：

```text
一个线程 → 一行
```

优点：

- 实现简单；
- 对很短且行长较均匀的矩阵通常有效；
- 不需要 block 内规约。

不足：

- 一行中的非零元由单个线程串行累加；
- 行长差异较大时容易出现线程负载不均衡；
- 长行不能充分利用 GPU 并行性。

---

### 3.2 CSR Vector

CSR Vector 使用一个 warp 处理一行：

```text
一个 warp（32 个线程）→ 一行
```

warp 内的线程共同读取该行非零元，并通过 `__shfl_down_sync` 完成规约。

优点：

- 一行内部可以并行计算；
- 对中等长度或较长的行通常优于 CSR Scalar；
- 不使用 block 级 `__syncthreads()`，避免最后一个 block 的条件同步问题。

不足：

- 对只有几个非零元的短行，大量 lane 会空闲；
- 不同行的行长差异仍可能造成 warp 间负载不均衡。

---

### 3.3 CSR Adaptive

CSR Adaptive 根据行长选择两种处理方式：

- 多个短行合并到同一个 CUDA block，使用类似 CSR Stream 的方式处理；
- 单个超长行由一个 CUDA block 的所有线程共同处理并规约。

程序在 CPU 上预先生成 `rowBlocks`，描述每个 CUDA block 负责的行区间。该预处理只执行一次，不放入 GPU kernel 计时区间。

优点：

- 能同时兼顾短行和长行；
- 对行长分布不规则、含少量超长行的矩阵更有针对性。

不足：

- 需要额外的行分块预处理；
- shared memory 使用量较大；
- 性能依赖分块阈值 `ADAPTIVE_NNZ_PER_BLOCK`；
- 本程序是用于学习和实验的简化实现，不等同于论文或成熟库中的完整优化版本。

当前代码中的阈值为：

```cpp
constexpr int ADAPTIVE_NNZ_PER_BLOCK = 1024;
```

---

### 3.4 PCSR

PCSR 在本程序中拆成两个 GPU kernel：

1. `spmv_pcsr_product_kernel`：计算所有非零元乘积；
2. `spmv_pcsr_sum_kernel`：按 CSR 行边界对乘积求和。

即先生成：

```text
products[j] = values[j] × x[col_idx[j]]
```

再计算每一行：

```text
y[i] = sum(products[row_ptr[i] : row_ptr[i + 1]])
```

优点：

- 乘法阶段具有较规则的并行访问；
- 将乘积计算和按行求和分开，便于研究不同阶段的性能瓶颈。

不足：

- 需要额外的 `products` 数组；
- 每个非零元会额外写入并读回一次 `double`；
- 需要启动两个 kernel；
- 第二阶段存在 shared memory 和同步开销。

因此 PCSR 的 `approx_GB/s` 会额外计入中间数组的一次写入和一次读取，不能直接套用普通 CSR 的字节数估计。

---

### 3.5 LightSpMV-style 动态调度

该实现采用 LightSpMV 风格的动态任务分配：

- 一个 warp 被划分为若干小 vector；
- 每个 vector 共同处理一行；
- warp 通过原子计数器动态领取下一批行；
- 使用 persistent blocks，避免为每一批行都启动大量 block。

程序根据平均每行非零元数量自动选择每行使用的线程数：

| 平均每行非零元 | `THREADS_PER_VECTOR` |
|---:|---:|
| `<= 4` | 2 |
| `<= 8` | 4 |
| `<= 16` | 8 |
| `<= 32` | 16 |
| `> 32` | 32 |

优点：

- 动态领取任务可以缓解行长不均匀造成的负载不均衡；
- 小 vector 能减少短行中的空闲线程；
- persistent blocks 可以让每个 warp 连续处理多批行。

不足：

- 原子计数器会产生额外开销；
- 仅根据全局平均行长选择线程数，不一定适合所有行；
- 不规则访存和动态调度可能增加控制开销。

每次 LightSpMV 执行前必须清零行计数器。这个 `cudaMemsetAsync` 是算法每次运行所必需的操作，因此本程序将它计入 LightSpMV 的计时区间。

---

## 4. 测试矩阵

程序生成一个确定性的非规则 CSR 矩阵，不需要外部矩阵文件。

列号使用非连续模式构造，以模拟较一般的间接访存：

```cpp
col_idx[begin + local] = raw_col % n;
```

测试向量 `x` 不是全 1，而是使用非恒定函数生成：

```cpp
x[i] = sin(0.001 * i)
     + 0.5 * cos(0.003 * i)
     + 0.25;
```

这样可以避免错误列访问在 `x[i] = 1` 时仍然碰巧得到正确结果。

程序提供两种矩阵模式。

### `matrix_mode = 0`：原始行长模式

每行非零元数量循环使用：

```text
3, 8, 16, 32, 64, 128
```

特点：

- 同时包含短行和中等长度行；
- 平均每行约有 41.8 个非零元；
- 适合比较 CSR Scalar、CSR Vector 和常规动态调度；
- 因为没有行长达到 1024，CSR Adaptive 的超长行分支通常不会触发。

### `matrix_mode = 1`：重尾行长模式

在原始模式基础上加入少量超长行：

```text
每 1000 行：一行长度为 4096
每 100 行：一行长度为 1024
其他行：仍使用 3, 8, 16, 32, 64, 128
```

特点：

- 同时包含短行、中等行和少量超长行；
- 能真正触发 CSR Adaptive 的长行处理路径；
- 更适合观察不同负载均衡策略的区别；
- 默认使用该模式。

当 `n` 小于指定行长时，实际行长会被限制为 `n`。

> 该矩阵是用于控制实验变量的合成矩阵，不能代表所有实际稀疏矩阵。正式论文实验还应增加 SuiteSparse Matrix Collection 或实际应用产生的矩阵。

---

## 5. 运行环境要求

建议环境：

- Linux；
- NVIDIA GPU；
- NVIDIA 驱动；
- CUDA Toolkit；
- CMake 3.18 或更高版本；
- 支持 C++17 的主机编译器。

检查基本环境：

```bash
nvidia-smi
nvcc --version
cmake --version
g++ --version
```

查看 CUDA 是否能识别 GPU：

```bash
nvidia-smi -L
```

如果安装了 CUDA Samples，也可以运行 `deviceQuery` 查看 GPU 的 Compute Capability。

---

## 6. 编译

### 6.1 V100 示例

NVIDIA V100 的 Compute Capability 为 7.0，因此使用：

```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=70

cmake --build build -j
```

生成的可执行文件为：

```text
build/spmv_compare
```

### 6.2 常见架构示例

| GPU 示例 | `CMAKE_CUDA_ARCHITECTURES` |
|---|---:|
| V100 | 70 |
| T4 | 75 |
| A100 | 80 |
| RTX 30 系列 | 86 |
| RTX 40 系列 | 89 |
| H100 | 90 |

如果没有在配置命令中指定架构，当前 `CMakeLists.txt` 默认使用：

```cmake
CMAKE_CUDA_ARCHITECTURES=70
```

因此在非 V100 GPU 上建议显式设置正确的架构。

### 6.3 重新配置架构

CMake 会缓存原来的架构设置。修改 GPU 架构后，建议删除旧构建目录：

```bash
rm -rf build

cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=80

cmake --build build -j
```

### 6.4 查看详细编译命令

```bash
cmake --build build --verbose
```

---

## 7. 运行方法

命令格式：

```bash
./build/spmv_compare [n] [repeats] [warmup] [matrix_mode] [block_size]
```

查看内置帮助：

```bash
./build/spmv_compare --help
```

### 参数说明

| 参数 | 含义 | 默认值 |
|---|---|---:|
| `n` | 方阵行数和列数 | `2000000` |
| `repeats` | 每种算法正式计时重复次数 | `10` |
| `warmup` | 每种算法预热次数 | `2` |
| `matrix_mode` | 测试矩阵模式，取 `0` 或 `1` | `1` |
| `block_size` | CUDA block 中线程数 | `256` |

`block_size` 必须满足：

- 大于 0 且不超过 1024；
- 是 32 的倍数；
- 是 2 的整数次幂；
- 不超过 `ADAPTIVE_NNZ_PER_BLOCK`。

推荐值：

```text
128, 256, 512
```

### 7.1 小规模快速验证

首次运行建议先用小矩阵：

```bash
./build/spmv_compare 10000 3 1 1 256
```

### 7.2 一般性能测试

```bash
./build/spmv_compare 200000 20 3 1 256
```

### 7.3 较大规模测试

```bash
./build/spmv_compare 2000000 20 3 1 256
```

大规模运行前应确认 CPU 内存和 GPU 显存充足。PCSR 会额外分配一个长度为 `nnz` 的 `double` 中间数组，因此比其他算法占用更多显存。

### 7.4 比较两种矩阵模式

原始行长模式：

```bash
./build/spmv_compare 200000 20 3 0 256
```

重尾行长模式：

```bash
./build/spmv_compare 200000 20 3 1 256
```

为了进行公平比较，除 `matrix_mode` 外应保持其他参数不变。

### 7.5 比较不同 block size

```bash
./build/spmv_compare 200000 20 3 1 128
./build/spmv_compare 200000 20 3 1 256
./build/spmv_compare 200000 20 3 1 512
```

不同算法的最优 block size 可能不同。当前程序为了方便统一比较，五种算法共享命令行传入的 `block_size`，但 PCSR 的第二个 kernel 固定使用 256 个线程。

---

## 8. 输出结果说明

程序首先输出设备和矩阵信息，例如：

```text
GPU                    = Tesla V100-SXM2-32GB
Compute capability     = 7.0
SM count               = 80
n                      = 200000
nnz                    = ...
average nnz per row    = ...
matrix mode            = 1 (heavy-tail)
block size             = 256
Adaptive row blocks    = ...
Light threads/vector   = 32
Light grid size        = ...
repeats                = 20
warmup                 = 3
```

随后输出五种算法的性能表：

```text
kernel                  avg_ms       GFLOP/s     approx_GB/s     max_abs_err     max_rel_err
--------------------------------------------------------------------------------------------
CSR Scalar             ...          ...         ...             ...             ...
CSR Vector             ...          ...         ...             ...             ...
CSR Adaptive           ...          ...         ...             ...             ...
PCSR                   ...          ...         ...             ...             ...
LightSpMV              ...          ...         ...             ...             ...
```

### `avg_ms`

一次 SpMV 的平均 GPU 时间，单位为毫秒。

程序使用 CUDA Event 计时：

```text
总计时时间 / repeats
```

数值越小越好。

### `GFLOP/s`

按照每个非零元一次乘法和一次加法估计：

```text
GFLOP/s = 2 × nnz / 时间
```

数值越大越好。

### `approx_GB/s`

根据算法需要读取和写入的数据量估算得到的有效带宽。

普通 CSR 近似计入：

- `values[j]`；
- `col_idx[j]`；
- `x[col_idx[j]]`；
- `row_ptr`；
- `y`。

PCSR 还额外计入：

- 中间数组 `products` 的一次写入；
- 中间数组 `products` 的一次读取。

该指标只是算法级近似值，忽略了缓存复用、缓存行粒度、写分配和实际内存事务。真实 DRAM/L2 带宽应以 Nsight Compute 为准。

### `max_abs_err`

GPU 结果和 CPU 参考结果之间的最大绝对误差：

```text
max |y_gpu[i] - y_cpu[i]|
```

### `max_rel_err`

GPU 结果和 CPU 参考结果之间的最大相对误差。

由于不同并行规约顺序会产生浮点舍入差异，各算法结果不一定逐位相同。程序默认正确性容差为：

```text
1e-10
```

当最大绝对误差和最大相对误差都超过容差时，该算法会被判定为不通过。

### `Fastest kernel`

程序会根据本次测试的 `avg_ms` 输出最快算法。这个结论只对当前 GPU、矩阵模式、矩阵规模、block size 和编译配置有效，不应理解为某个算法在所有矩阵上始终最快。

---

## 9. 计时范围

为了比较 GPU 算法本身，以下操作只执行一次，不计入每种 kernel 的 `avg_ms`：

- 主机端生成 CSR 矩阵；
- CPU 参考 SpMV；
- `cudaMalloc`；
- CSR 数据从 Host 到 Device 的复制；
- CSR Adaptive 的 `rowBlocks` 预处理；
- 最终结果从 Device 复制回 Host；
- GPU 内存释放。

计时区间内包含：

- CSR Scalar：一个 SpMV kernel；
- CSR Vector：一个 SpMV kernel；
- CSR Adaptive：一个 SpMV kernel；
- PCSR：乘积 kernel 和行求和 kernel；
- LightSpMV：计数器清零和动态调度 kernel。

其中：

- PCSR 必须完成两个 kernel 才构成一次完整 SpMV；
- LightSpMV 每次都必须重新设置动态行计数器，因此清零操作属于算法调用成本。

这种计时方式反映的是 GPU 已经持有矩阵和向量之后的一次 SpMV 成本，适合迭代求解器中重复执行 SpMV 的场景。

如果需要测量从 CPU 输入到 CPU 输出的端到端耗时，应另外把内存分配、H2D、D2H 和预处理也纳入计时，不要与当前结果混在一起比较。

---

## 10. 如何获得较稳定的性能数据

建议遵循以下原则：

1. 使用 Release 编译；
2. 正常计时时使用多次重复，例如 `repeats=20` 或更高；
3. 保留 2～5 次预热；
4. 正式对比时保持矩阵、GPU、block size 和编译选项一致；
5. 避免同时运行其他 GPU 程序；
6. 每组实验运行多次，记录平均值和波动范围；
7. 先确认所有算法正确性检查为 `PASS`；
8. NCU 分析时使用较小规模和较少重复次数，避免报告过大；
9. 不要把 NCU 插桩后的执行时间当作正常运行时间；
10. 对真实研究结论，应使用多种矩阵而不是只使用单个合成矩阵。

建议分别测试：

```text
矩阵规模：      10^4、10^5、10^6 或更多
矩阵模式：      mode 0、mode 1
block size：     128、256、512
重复次数：       至少 10～30 次
```

---

## 11. Nsight Compute 性能分析

### 11.1 生成基础报告

首次分析建议使用 `speedOfLight`，采集速度通常比 `full` 更快：

```bash
ncu --set speedOfLight \
    --target-processes all \
    -o spmv_speedoflight \
    ./build/spmv_compare 200000 1 0 1 256
```

生成：

```text
spmv_speedoflight.ncu-rep
```

打开报告：

```bash
ncu-ui spmv_speedoflight.ncu-rep
```

### 11.2 生成完整报告

```bash
ncu --set full \
    --target-processes all \
    -o spmv_five_kernels \
    ./build/spmv_compare 200000 1 0 1 256
```

`--set full` 会采集大量指标，运行时间可能明显增加。建议使用：

```text
repeats = 1
warmup  = 0
```

### 11.3 为什么报告里可能看到六个 kernel

程序比较五种算法，但 PCSR 由两个 kernel 组成：

```text
spmv_pcsr_product_kernel
spmv_pcsr_sum_kernel
```

因此一次完整测试通常会看到六种 kernel 名称：

```text
1 × CSR Scalar
1 × CSR Vector
1 × CSR Adaptive
2 × PCSR
1 × LightSpMV
```

分析 PCSR 时，应同时考虑两个 kernel 的耗时和瓶颈，不能只看其中一个。

### 11.4 命令行导出 CSV

不同 NCU 版本支持的指标名称可能略有区别。可以先查看可用指标：

```bash
ncu --query-metrics | less
```

一个常用导出示例：

```bash
ncu --csv --page raw \
    --metrics \
gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__t_sector_hit_rate.pct,\
sm__warps_active.avg.pct_of_peak_sustained_active \
    ./build/spmv_compare 200000 1 0 1 256 \
    > ncu_metrics.csv
```

如果某个指标在当前 NCU 版本不存在，请通过 `ncu --query-metrics` 查找对应名称。

### 11.5 建议重点观察的指标

#### 运行时间

```text
gpu__time_duration.sum
```

用于比较每个 kernel 的 GPU 持续时间。

#### SM 吞吐率

```text
sm__throughput.avg.pct_of_peak_sustained_elapsed
```

用于判断 SM 执行资源利用程度。

SpMV 通常不是计算密集型算法，因此 SM 吞吐率不一定很高。

#### DRAM 吞吐率

```text
dram__throughput.avg.pct_of_peak_sustained_elapsed
```

用于判断全局显存带宽使用程度。SpMV 通常受内存带宽和不规则访问限制，因此该指标很重要。

#### L2 命中率

```text
lts__t_sector_hit_rate.pct
```

用于观察对 `x[col_idx[j]]` 的间接访问是否从 L2 缓存中获益。

#### Occupancy

```text
sm__warps_active.avg.pct_of_peak_sustained_active
```

Occupancy 较低可能来自：

- 每个 block 使用较多 shared memory；
- 寄存器使用较多；
- block 数量不足；
- kernel 工作量太小。

Occupancy 高不代表性能一定高，还需要结合带宽、停顿原因和运行时间分析。

#### Warp Stall

建议在 NCU 的 `Warp State Statistics` 中观察：

- `Long Scoreboard`：常与全局内存等待有关；
- `Barrier`：常与 `__syncthreads()` 有关；
- `Not Selected`：有可运行 warp，但调度器选择了其他 warp；
- `Wait`：可能与流水线依赖或固定延迟有关。

### 11.6 各算法可能的关注点

| 算法 | 建议关注 |
|---|---|
| CSR Scalar | 行长不均衡、Long Scoreboard、较低并行度 |
| CSR Vector | 短行 lane 空闲、warp 执行效率、内存等待 |
| CSR Adaptive | shared memory、Barrier、长行与短行分支差异 |
| PCSR product | 全局内存吞吐、间接读取 `x`、中间数组写入 |
| PCSR sum | shared memory、Barrier、Occupancy、多个 chunk |
| LightSpMV | 原子调度、warp 工作分布、persistent blocks、内存等待 |

---

## 12. Compute Sanitizer 正确性检查

在正式做性能分析前，建议先运行 CUDA 检查工具。

### 12.1 检查越界和非法内存访问

```bash
compute-sanitizer --tool memcheck \
    ./build/spmv_compare 10000 1 0 1 256
```

### 12.2 检查同步问题

```bash
compute-sanitizer --tool synccheck \
    ./build/spmv_compare 10000 1 0 1 256
```

### 12.3 检查竞争条件

```bash
compute-sanitizer --tool racecheck \
    ./build/spmv_compare 10000 1 0 1 256
```

Sanitizer 会显著减慢程序，因此应使用小规模矩阵和一次重复测试。

---

## 13. 常见问题

### 13.1 `No CUDA device found`

说明 CUDA Runtime 没有检测到可用 GPU。检查：

```bash
nvidia-smi
```

并确认程序运行在有 NVIDIA GPU 的节点上。

---

### 13.2 `no kernel image is available for execution on the device`

通常说明编译时设置的 CUDA 架构与实际 GPU 不匹配。

删除构建目录并重新指定架构：

```bash
rm -rf build
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build -j
```

将 `70` 替换为实际 GPU 对应的架构。

---

### 13.3 `unsupported GNU version`

可能是 CUDA Toolkit 与系统 GCC 版本不兼容。查看：

```bash
nvcc --version
gcc --version
```

然后为 CMake 指定兼容的主机编译器，例如：

```bash
cmake -S . -B build \
    -DCMAKE_CUDA_HOST_COMPILER=/path/to/g++ \
    -DCMAKE_CUDA_ARCHITECTURES=70
```

---

### 13.4 `out of memory`

减小 `n`：

```bash
./build/spmv_compare 200000 10 2 1 256
```

PCSR 需要额外的：

```text
nnz × sizeof(double)
```

显存来保存中间乘积。

此外，主机端还需要保存 CSR、输入向量、CPU 参考结果和 GPU 返回结果。

---

### 13.5 NCU 报 `ERR_NVGPUCTRPERM`

说明当前用户没有访问 NVIDIA GPU Performance Counters 的权限。普通 CUDA 程序仍可能正常运行，但 NCU 无法采集硬件计数器。

需要管理员按照服务器的 NVIDIA 驱动配置开放 profiling 权限。权限修改后，应重新登录服务器会话并再次运行 NCU。

---

### 13.6 NCU 报 `No metrics to collect found in sections`

可能原因包括：

- NCU 版本与 GPU 或驱动不匹配；
- 选择的 section 在当前架构上没有可用指标；
- 指标名称在当前版本中发生变化；
- 实际程序没有成功启动 kernel。

先检查：

```bash
ncu --version
ncu --list-sets
ncu --list-sections
ncu --query-metrics | head
```

并先尝试：

```bash
ncu --set basic ./build/spmv_compare 10000 1 0 1 256
```

---

### 13.7 为什么程序运行时间比单个 NCU kernel 时间大

正常运行程序还包含：

- 矩阵构造；
- CPU 参考计算；
- GPU 内存分配；
- H2D/D2H 复制；
- 五种算法依次执行；
- 正确性验证。

而结果表中的 `avg_ms` 只表示对应 GPU 算法的一次 SpMV 时间。

NCU 中显示的是单个 kernel 的时间。尤其是 PCSR，需要把两个 kernel 作为一次完整算法共同分析。

---

### 13.8 为什么 `approx_GB/s` 可能高于或低于 NCU 的 DRAM 带宽

`approx_GB/s` 是根据算法数据结构估算的有效带宽，不是实际 DRAM 事务量。

差异可能来自：

- L1/L2 缓存复用；
- 合并或未合并的内存访问；
- 缓存行和 sector 粒度；
- 重复列号导致 `x` 被缓存；
- 写事务和中间数据；
- NCU 指标统计口径不同。

因此：

- `approx_GB/s` 适合在本程序内部做粗略比较；
- 实际硬件带宽和利用率应以 NCU 为准。

---

### 13.9 为什么最快算法会随矩阵改变

不同算法适合不同的行长分布：

- 大量短行可能更适合 CSR Scalar 或小 vector；
- 中长且较均匀的行可能更适合 CSR Vector；
- 含少量超长行的重尾矩阵可能更适合 Adaptive；
- 动态调度可能改善严重不均衡矩阵；
- PCSR 会增加中间数组流量，因此可能受显存带宽限制。

不存在对所有矩阵和所有 GPU 都绝对最快的 SpMV 格式。

---

## 14. 当前实现的实验限制

使用本程序撰写性能结论时，应明确以下限制：

1. 只测试了合成 CSR 矩阵；
2. `double` 是固定数据类型；
3. CSR 索引使用 32 位 `int`，因此 `nnz` 不能超过 `INT_MAX`；
4. LightSpMV 的线程数只根据全局平均行长选择；
5. CSR Adaptive 使用固定阈值 1024；
6. 五种算法没有分别搜索各自最优 block size；
7. PCSR 是两阶段学习型实现，并非成熟库的高度优化版本；
8. 没有与 cuSPARSE 的 `cusparseSpMV` 进行基准对比；
9. 没有读取 Matrix Market 文件；
10. 没有统计多次独立运行之间的标准差。

较完整的研究实验可以继续增加：

- Matrix Market 文件读取；
- SuiteSparse 实际矩阵；
- cuSPARSE 基线；
- `float`、`double` 两种精度；
- 每种算法独立调优 block size；
- 多次独立运行的均值、最小值、最大值和标准差；
- Roofline 分析；
- 预处理成本和端到端成本的单独统计。

---

## 15. 推荐实验流程

一个较规范的实验流程如下：

```text
1. Release 模式编译
2. 小规模运行，检查 Correctness check: PASS
3. 使用 memcheck、synccheck 检查代码
4. 在 mode 0 和 mode 1 下分别测试
5. 对 128、256、512 block size 做参数实验
6. 每组正常运行至少重复 10～30 次
7. 记录 avg_ms、GFLOP/s 和 approx_GB/s
8. 使用 NCU 分析最快和最慢算法
9. 对 PCSR 分别分析两个 kernel
10. 结合运行时间、DRAM、L2、Occupancy 和 Warp Stall 得出结论
```

记录表可以使用以下格式：

| GPU | 模式 | n | nnz | block | 算法 | avg_ms | GFLOP/s | approx_GB/s | DRAM % | SM % | Occupancy % |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| V100 | 0 | | | 256 | CSR Scalar | | | | | | |
| V100 | 0 | | | 256 | CSR Vector | | | | | | |
| V100 | 0 | | | 256 | CSR Adaptive | | | | | | |
| V100 | 0 | | | 256 | PCSR | | | | | | |
| V100 | 0 | | | 256 | LightSpMV | | | | | | |

---

## 16. 快速命令汇总

编译：

```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build -j
```

快速验证：

```bash
./build/spmv_compare 10000 3 1 1 256
```

正常测试：

```bash
./build/spmv_compare 200000 20 3 1 256
```

NCU：

```bash
ncu --set speedOfLight \
    -o spmv_speedoflight \
    ./build/spmv_compare 200000 1 0 1 256
```

打开报告：

```bash
ncu-ui spmv_speedoflight.ncu-rep
```

内存检查：

```bash
compute-sanitizer --tool memcheck \
    ./build/spmv_compare 10000 1 0 1 256
```

同步检查：

```bash
compute-sanitizer --tool synccheck \
    ./build/spmv_compare 10000 1 0 1 256
```
