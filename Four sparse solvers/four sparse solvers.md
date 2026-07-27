# SuperLU_DIST、PanguLU、NVIDIA cuDSS 与 STRUMPACK 对比及性能瓶颈分析

> 基于 16 个实测矩阵、配套测试程序及四个项目的官方文档  
> 文档日期：2026-07-24  
> 分析视角：使用者的性能体验、软件设计与 API 集成成本

> **本次修订：**全部实测表格和统计量已按 2026-07-24 的新数据重算。SuperLU_DIST 的 `total_ms` 现在是屏障同步后、包围整个 `pdgssvx()` 调用并对各 rank 取最大值的墙钟时间，完整覆盖该调用；新 CSV 没有导出其 reorder/factorization/solve 三个内部细分字段，因此本版不再沿用旧数据中“若干分项未覆盖总时长”的结论，也不再从不完整分项推断 SuperLU_DIST 的内部瓶颈。

## 1. 摘要

本次数据并不是“四个分布式求解器在集群上的横向扩展测试”，而是一次 **MPI singleton（1 个 rank）、默认只暴露 GPU 0、每个矩阵启动一个新进程的单进程/单 GPU 冷启动测试**。这一点决定了数据能回答的是“当前四个适配程序在这台机器上的单次求解体验”，不能回答“多 GPU 或多节点时谁扩展得更好”。

在严格只统计通过正确性检查的结果时，本次测试的主要结论如下：

1. **cuDSS 是本次有效结果中的总体最快者。**16 个矩阵中，cuDSS 在 13 个矩阵上既通过精度检查又取得最短主要计时值；STRUMPACK 在 `crystm03` 上获胜；SuperLU_DIST 仅在 3×3 的 `AA1` 上获胜；`denormal` 没有任何求解器通过。
2. 在非平凡矩阵、双方都通过检查的共同样本上：
   - cuDSS 相对 SuperLU_DIST 的主要计时值几何平均快 **3.62×**，冷启动外部墙钟时间快 **2.17×**；
   - STRUMPACK 相对 SuperLU_DIST 的主要计时值几何平均快 **1.93×**，冷启动外部墙钟时间快 **1.49×**；
   - cuDSS 相对 STRUMPACK 的主要计时值几何平均快 **1.88×**，冷启动外部墙钟时间快 **1.43×**。
3. **cuDSS 的主要瓶颈不是数值分解，而是 analysis。**在非 `AA1` 的已完成运行中，analysis 占核心阶段总时间的中位数为 **92.3%**；数值分解仅占 6.7%，solve 仅占 0.8%。这与 NVIDIA 官方说明一致：重排是难以在 GPU 上加速的图算法，目前主要在主机端执行。
4. **STRUMPACK 的主要瓶颈同样是重排。**其 reorder 占核心阶段总时间的中位数为 **80.8%**。本次程序固定使用串行 METIS、关闭 matching、关闭压缩、关闭迭代改进，并把所有输入当作一般非对称模式，因此没有展示 STRUMPACK 的低秩压缩和预条件器能力，也没有利用这批矩阵均为对称模式这一事实。
5. **SuperLU_DIST 的总计时已经完整，但新结果不支持内部阶段归因。**新代码用 `MPI_Wtime()` 包围整个 `pdgssvx()`，在调用前后执行进程网格屏障，并以各 rank 最大值作为 `total_ms`。因此整个专家驱动调用都已计时，不存在旧版由三个分项求和造成的时间缺口。与此同时，新 CSV 的 `superlu_reorder_ms`、`superlu_factorization_ms` 和 `superlu_solve_ms` 全部为空，所以当前只能可靠比较整个 `pdgssvx`，不能定量断言其瓶颈位于重排、符号处理还是数值分解。
6. **PanguLU 当前结果不能用于有效性能排名。**16 个矩阵中只有 3×3 的 `AA1` 通过；`Dubcova3` 发生段错误，其余 14 个矩阵均因精度失败退出。即使暂时忽略正确性，完成运行的分解阶段通常也是主要瓶颈（中位占比 81.3%）。但测试采用 1 个 MPI rank、`nthread=1`、固定 `nb=64`，且 CSV 没有保存 PanguLU 的编译开关，因此没有验证其面向分布式多 GPU 的设计优势。更重要的是，应先排查调用端与库的类型 ABI、MC64/METIS 构建开关和数值稳定性，再讨论速度。
7. `denormal` 对四个库都不是普通的“求解失败”。SuperLU_DIST 的相对残差约为 `1.82e-8`，但相对解误差接近 1；其他三个库的相对解误差也都接近 1。这通常意味着矩阵严重病态，使“小残差”不能保证“小的前向误差”。该矩阵应单独做缩放、条件数估计和高精度参考解分析。

如果用户只需要当前机器上的单次大矩阵求解，且可以接受 NVIDIA GPU 和 cuDSS 的 Preview/专有许可约束，**cuDSS 是当前最有吸引力的候选**。如果需要开放源码、跨 NVIDIA/AMD GPU、成熟的 MPI 分布式路径及可选的低秩近似/预条件器，**STRUMPACK 更均衡**。如果需要成熟、细粒度可控的分布式超节点 LU、重复结构复用及 2D/3D 进程网格，**SuperLU_DIST 仍是稳健基线**。PanguLU 则更适合在先解决本次正确性问题后，针对多 GPU、多节点和 `nb` 调优重新评估。

---

## 2. 数据、代码与分析口径

### 2.1 数据集

2026-07-24 的新粘贴数据没有换行，表头和 16 行记录被拼接为一个长字符串。本报告按每个矩阵的唯一 `matrix_path` 边界重建记录，重建后每行均为一致的 51 个字段，没有修改任何数值。与上一版相比，四个库的时间和多数误差指标都来自一次新的运行，不能只替换 SuperLU_DIST 的单个字段；因此本文重新计算了全部表格、加速比、范围和中位数。

数据集具有以下共同特征：

| 项目 | 值 |
|---|---:|
| 矩阵数 | 16 |
| Matrix Market 类型 | 全部为 `real coordinate symmetric` |
| 阶数范围 | 3 ～ 1,585,478 |
| 对称展开后的 nnz 范围 | 7 ～ 7,660,826 |
| 右端项 | 构造精确解 \(x^\ast=1\)，令 \(b=A x^\ast\) |
| 精度检查 | 同时检查相对残差与相对解误差 |

Matrix Market 中只存一半的对称矩阵会由读取逻辑展开为完整矩阵，因此 `nnz_expanded` 大于 `nnz_stored`。这种做法保证四个适配器看到相同的一般矩阵数据，但也意味着：

- 主机内存、PCIe 传输和稀疏索引流量接近只存一个三角部分时的两倍；
- cuDSS 和 STRUMPACK 可表达的“对称矩阵/对称模式”能力没有在当前测试中使用；
- 本报告比较的是“一般 LU 路径下处理完整对称矩阵”的表现，而不是专用 Cholesky/LDL 路径。

### 2.2 配套代码确认的实际测试方式

代码位于：

- [`test_superlu_dist.cpp`](direct_solver_benchmark_bundle/test_superlu_dist.cpp)
- [`test_pangulu.cpp`](direct_solver_benchmark_bundle/test_pangulu.cpp)
- [`test_cudss.cpp`](direct_solver_benchmark_bundle/test_cudss.cpp)
- [`test_strumpack.cpp`](direct_solver_benchmark_bundle/test_strumpack.cpp)
- [`batch_direct_solvers.py`](direct_solver_benchmark_bundle/batch_direct_solvers.py)

批处理脚本直接执行每个求解器，没有使用 `mpirun`，所以 MPI 程序以 singleton 方式运行。脚本还在环境变量未预设时执行：

```python
child_env.setdefault("CUDA_VISIBLE_DEVICES", "0")
```

因此当前数据的基准形态是：

```text
一个矩阵
  └─ 固定顺序启动 SuperLU_DIST → PanguLU → cuDSS → STRUMPACK
       └─ 每个求解器一个全新进程
            └─ 1 个 MPI rank，通常只看到 GPU 0
```

固定执行顺序还带来两个潜在偏差：

- 后运行的库可能受益于操作系统文件页缓存；
- GPU 温度、频率和后台负载会随顺序变化。

正式对比应随机化求解器顺序并重复多次。

### 2.3 四个适配器的关键配置

| 求解器 | 本次适配器的关键配置 |
|---|---|
| SuperLU_DIST | `set_default_options_dist()`；`IterRefine=SLU_DOUBLE`；`pdgssvx`；分布式行块 CSR；进程网格由 `MPI_Dims_create` 生成；GPU 是否使用由库的构建和 `get_acc_offload()` 决定 |
| PanguLU | 适配器按 5.0.0 API 编写；完整 CSR 转为 rank 0 上的完整 CSC；默认 `nb=64`；`nthread=1`；GPU warp 参数为 4；具体 GPU、METIS、MC64 和数值类型仍由编译开关决定 |
| cuDSS | 设备端 32 位 CSR 索引、FP64 数值；`CUDSS_MTYPE_GENERAL` + `CUDSS_MVIEW_FULL`；只用默认 config；没有打开 matching 或 iterative refinement；每个 MPI rank 复制并独立求解完整矩阵 |
| STRUMPACK | `StrumpackSparseSolverMPIDist<double,int>`；`MatchingJob::NONE`；`CompressionType::NONE`；`KrylovSolver::DIRECT`；`ReorderingStrategy::METIS`；GPU 可见时启用，默认 4 streams；`symmetric_pattern=false` |

### 2.4 正确性阈值并不完全一致

| 求解器 | 相对残差阈值 | 相对解误差阈值 |
|---|---:|---:|
| SuperLU_DIST | `1e-8` | `1e-8` |
| PanguLU | `1e-6` | `1e-6` |
| cuDSS | `1e-6` | `1e-6` |
| STRUMPACK | `1e-8` | `1e-8` |

四个程序都要求两个指标同时通过。PanguLU 和 cuDSS 的阈值反而宽松 100 倍，因此 PanguLU 的大量失败不能归因于“给它设了更严格的门槛”。不过，阈值不一致仍然使总体比较不够规范，后续应统一。

### 2.5 两种时间的含义

CSV 同时给出：

- `*_total_ms`：适配器内部打印的库调用或核心阶段时间；
- `*_wall_elapsed_ms`：批处理进程启动前到进程退出后的外部墙钟时间。

二者边界不同：

- SuperLU_DIST 的 `total_ms` 是整个 `pdgssvx()` 调用的墙钟；
- PanguLU 的 `total_ms` 是 `init + factor + solve`；
- cuDSS 的 `total_ms` 是 CUDA event 测得的 `analysis + factor + solve`，不含此前的设备分配和主机到设备复制；
- STRUMPACK 的 `total_ms` 是 `reorder + factor + solve`，不含此前设置分布式 CSR 的时间。

所以：

- `total_ms` 更适合比较适配器所定义的主要库调用，但四库的边界不是完全相同；只有在 CSV 提供内部阶段字段时，才能据此定位具体阶段瓶颈；
- `wall_elapsed_ms` 更接近用户一次冷启动实际等待时间，但包含环境脚本、动态链接、MPI/CUDA 初始化、矩阵读取、数据转换、验证和清理。

本报告同时给出两种口径，不把其中任意一种包装成绝对公平的单一排名。

---

## 3. 实测结果

### 3.1 成功率

| 求解器 | SUCCESS | 精度失败 | 运行失败 | 通过率 |
|---|---:|---:|---:|---:|
| SuperLU_DIST | 15 | 1 | 0 | 93.75% |
| PanguLU | 1 | 14 | 1（段错误） | 6.25% |
| cuDSS | 14 | 2 | 0 | 87.50% |
| STRUMPACK | 15 | 1 | 0 | 93.75% |

其中：

- `denormal`：四个求解器全部精度失败；
- `crystm03`：cuDSS 和 PanguLU 精度失败，SuperLU_DIST 与 STRUMPACK 通过；
- `Dubcova3`：PanguLU 发生 `SIGSEGV`；
- PanguLU 除 `AA1` 外没有通过任何矩阵。

### 3.2 每个矩阵的主要调用/核心阶段总时间

单位为秒。SuperLU_DIST 一列是完整 `pdgssvx()` 墙钟，其他三列是各适配器所列核心阶段之和。`✓` 表示通过精度检查，`✗精度` 表示程序完成但未通过，`✗崩溃` 表示没有有效计时。粗体表示该矩阵上 **通过检查的最短结果**。

| 矩阵 | n | 展开 nnz | SuperLU_DIST | PanguLU | cuDSS | STRUMPACK |
|---|---:|---:|---:|---:|---:|---:|
| AA1 | 3 | 7 | **0.029 ✓** | 0.034 ✓ | 0.036 ✓ | 0.039 ✓ |
| Dubcova3 | 146,689 | 3,636,649 | 3.592 ✓ | — ✗崩溃 | **0.708 ✓** | 2.022 ✓ |
| G3_circuit | 1,585,478 | 7,660,826 | 42.270 ✓ | 113.546 ✗精度 | **12.565 ✓** | 26.248 ✓ |
| apache2 | 715,176 | 4,817,870 | 31.229 ✓ | 206.589 ✗精度 | **7.314 ✓** | 13.731 ✓ |
| bundle1 | 10,581 | 770,901 | 0.391 ✓ | 0.358 ✗精度 | **0.075 ✓** | 0.554 ✓ |
| cfd1 | 70,656 | 1,828,364 | 3.794 ✓ | 15.030 ✗精度 | **1.103 ✓** | 1.675 ✓ |
| cfd2 | 123,440 | 3,087,898 | 6.826 ✓ | 40.820 ✗精度 | **1.909 ✓** | 2.806 ✓ |
| crystm03 | 24,696 | 583,770 | 1.010 ✓ | 1.557 ✗精度 | 0.348 ✗精度 | **0.495 ✓** |
| denormal | 89,400 | 1,156,224 | 2.367 ✗精度 | 6.258 ✗精度 | 0.782 ✗精度 | 1.142 ✗精度 |
| ecology2 | 999,999 | 4,995,991 | 17.826 ✓ | 41.277 ✗精度 | **5.959 ✓** | 11.375 ✓ |
| gas_sensor | 66,917 | 1,703,365 | 3.936 ✓ | 24.772 ✗精度 | **1.070 ✓** | 1.596 ✓ |
| offshore | 259,789 | 4,242,673 | 15.884 ✓ | 140.223 ✗精度 | **3.993 ✓** | 5.651 ✓ |
| parabolic_fem | 525,825 | 3,674,625 | 10.755 ✓ | 23.527 ✗精度 | **3.861 ✓** | 5.749 ✓ |
| qa8fm | 66,127 | 1,660,579 | 4.001 ✓ | 27.337 ✗精度 | **1.046 ✓** | 1.555 ✓ |
| thermomech_dM | 204,316 | 1,423,116 | 4.249 ✓ | 5.555 ✗精度 | **1.443 ✓** | 2.131 ✓ |
| tmt_sym | 726,713 | 5,080,961 | 15.700 ✓ | 32.054 ✗精度 | **5.478 ✓** | 8.446 ✓ |

不能把 `bundle1` 上 PanguLU 的 0.358 秒当作胜出，因为输出没有通过正确性检查。对于直接求解器，“更快地产生错误结果”没有用户价值。

### 3.3 有效结果的获胜次数

按上述主要计时值，在每个矩阵上仅比较 `SUCCESS`：

| 求解器 | 最快次数 |
|---|---:|
| cuDSS | 13 |
| STRUMPACK | 1（`crystm03`） |
| SuperLU_DIST | 1（3×3 的 `AA1`） |
| PanguLU | 0 |
| 无求解器通过 | 1（`denormal`） |

### 3.4 共同成功样本上的几何平均加速

排除 3×3 的 `AA1`，并且每一对只使用双方都通过的矩阵。几何平均比算术平均更适合跨度很大的运行时间和加速比。

| 比较 | 共同样本数 | 主要计时值加速 | 主要计时值范围 | 冷启动墙钟加速 | 冷启动范围 |
|---|---:|---:|---:|---:|---:|
| cuDSS / SuperLU_DIST | 13 | **3.62×** | 2.79×～5.24× | **2.17×** | 1.14×～3.44× |
| STRUMPACK / SuperLU_DIST | 14 | **1.93×** | 0.71×～2.81× | **1.49×** | 0.79×～2.15× |
| cuDSS / STRUMPACK | 13 | **1.88×** | 1.42×～7.43× | **1.43×** | 1.25×～1.90× |

这里的“加速”只成立于当前测试程序、当前机器和当前单 rank 配置。它不是四个项目在官方最佳配置下的通用性能比。

### 3.5 阶段占比与计时覆盖

以下是排除 `AA1` 后所有“已完成并具有阶段时间”的运行中，各阶段占内部总时间的中位数。阶段占比中包含精度失败的已完成运行，因为这里分析的是时间花在哪里，而不是把无效解纳入性能排名。

| 求解器 | 样本数 | 分析/初始化/重排 | 数值分解 | solve | 计时覆盖说明 |
|---|---:|---:|---:|---:|---|
| SuperLU_DIST | 15 | — | — | — | `total_ms` 完整覆盖 `pdgssvx()`；新 CSV 未导出内部三项，故不计算占比 |
| PanguLU | 14 | initialization 18.1% | **81.3%** | 0.6% | 三项之和 |
| cuDSS | 15 | **analysis 92.3%** | 6.7% | 0.8% | 三项之和 |
| STRUMPACK | 15 | **reorder 80.8%** | 14.8% | 2.7% | 三项之和 |

cuDSS 的 analysis、factorization 和 solve 三项构成其当前 `total_ms`。SuperLU_DIST 则是另一种口径：完整调用总时间已经覆盖，但阶段粒度缺失。后者意味着无法细分瓶颈，不意味着总计时存在缺口。

外部冷启动墙钟与内部主要计时值之比的中位数为：

| 求解器 | 外部墙钟 / 内部主要计时 |
|---|---:|
| SuperLU_DIST | 1.35× |
| PanguLU | 1.08× |
| cuDSS | 2.11× |
| STRUMPACK | 1.83× |

这说明 cuDSS 的 GPU 上核心阶段很快，但一次性设备分配、复制、handle/config/data/matrix 创建与销毁、程序启动和验证等成本对用户的冷启动等待仍然明显。对于短任务，进程常驻和对象复用会比继续优化 solve kernel 更重要。

---

## 4. 四个库的性能特性与软件设计差异

### 4.1 SuperLU_DIST

#### 定位与算法

SuperLU 项目包含串行 SuperLU、共享内存 SuperLU_MT 和分布式 SuperLU_DIST。本次使用的是 **SuperLU_DIST**。官方将其定位为面向大规模非对称稀疏线性系统的分布式超节点 LU 求解器，采用 MPI、OpenMP，并可通过 CUDA 或 HIP 使用 GPU；新版本还提供通信规避的 3D 算法和 GPU 三角求解。

SuperLU_DIST 为并行可扩展性采用静态主元策略。官方 FAQ 明确指出，静态主元比串行 SuperLU 的动态部分选主元更容易扩展，但对非常病态的矩阵可能不够稳定；`ReplaceTinyPivot` 会用与矩阵范数相关的阈值替换过小主元，也可由用户关闭。多进程异步流水执行还意味着默认不保证逐位可重复。

官方资料：

- [SuperLU 官方主页与 SuperLU_DIST 版本/特性](https://portal.nersc.gov/project/sparse/superlu/)
- [SuperLU FAQ：进程网格、静态主元、符号分解、GPU 与性能](https://portal.nersc.gov/project/sparse/superlu/faq.html)
- [`pdgssvx` 专家驱动接口](https://portal.nersc.gov/project/sparse/superlu/superlu_dist_code_html/pdgssvx_8c.html)

#### 性能特性

优势：

- 超节点方法把具有相似结构的列聚合成较密的块，可使用高性能 BLAS-3；
- 适合 MPI+OpenMP 混合并行，2D 进程网格成熟，新版还支持 3D 通信规避；
- 支持 NVIDIA CUDA 与 AMD HIP，硬件选择比 cuDSS 更开放；
- `SamePattern`、`SamePattern_SameRowPerm` 和已分解模式可以复用排列、消去树、行排列乃至因子；
- 专家驱动可做平衡、误差估计、迭代改进及统计采集。

代价：

- 超节点稠密化可能为结构不规则的矩阵引入无效计算；
- 性能高度依赖消去树、填充量、进程网格、BLAS、CPU/GPU 映射和通信；
- 静态主元是稳定性与可扩展性之间的折中；
- API 需要用户管理较多结构，学习和清理成本较高。

#### 本次表现与瓶颈

SuperLU_DIST 通过 15/16，数值稳健性总体良好。它相对 cuDSS 和 STRUMPACK 慢，但本次只用一个 MPI rank，没有发挥分布式设计。

修改后的计时逻辑是：

1. 在进程网格 communicator 上执行 `MPI_Barrier`；
2. 用 `MPI_Wtime()` 记录起点；
3. 执行完整 `pdgssvx()`；
4. 再次执行 `MPI_Barrier` 并记录终点；
5. 通过 `MPI_Reduce(..., MPI_MAX, ...)` 取最慢 rank 的调用墙钟。

这样得到的 `total_ms` 已经完整覆盖 `pdgssvx`，不会再因只求和 `ETREE/FACT/SOLVE` 而漏掉驱动器中的其他工作。排除 `AA1` 后，外部进程墙钟与 `pdgssvx` 墙钟之比的中位数为 1.35×；在 `G3_circuit` 和 `apache2` 两个最大样本上，该比值分别只有 1.09× 和 1.08×，说明大问题的用户等待时间主要发生在 `pdgssvx` 内，而不是进程启动、读入或验证。

但是，新 CSV 的三个 SuperLU_DIST 阶段列在全部 16 行中都为空。参考源码仍保留 `stat.utime[ETREE/FACT/SOLVE]` 的读取与打印，因此应核对实际运行的二进制、日志格式和批处理解析器是否与当前源码同步。在取得可验证的分项前，本报告不再声称 `FACT`、`ETREE` 或 solve 中任何一个是 SuperLU_DIST 的主瓶颈。可靠的结论仅限于：整个 `pdgssvx` 是大样本端到端时间的主要组成，并且在共同有效样本上，其完整调用时间分别是 cuDSS 当前核心三阶段的 3.62×、STRUMPACK 当前核心三阶段的 1.93×（几何平均）。

#### 用户层面评价

SuperLU_DIST 像一个“功能全面但低层”的 HPC 基础件。它适合愿意管理 MPI 网格、矩阵分布、多个状态结构和生命周期，并需要对重排、复用、稳定性进行精细控制的团队。对于只想传入一个 CSR 并立刻得到结果的单 GPU 用户，它的接入成本明显高于 cuDSS 或 STRUMPACK 的对象接口。

---

### 4.2 PanguLU

#### 定位与算法

PanguLU 是面向异构分布式平台的开源稀疏 LU 求解器，使用 C、MPI、OpenMP 和 CUDA。其设计不沿用传统的超节点/多波前稠密块路线，而是：

- 将矩阵划分为规则二维块；
- 块内保留稀疏存储；
- 对不同块自适应选择稀疏 BLAS kernel；
- 用二维块循环分布和任务映射做负载均衡；
- 采用无同步通信/调度策略减少依赖等待。

这一设计意图是避免超节点方法因强行构造稠密块而对零填充做无效浮点运算，并提高大规模多 GPU 的并行度。官方 SC'23 论文的主要结果来自最多 128 张 A100 或 MI50 的分布式平台，不能直接映射到本次 1 rank 测试。

官方资料：

- [PanguLU 官方软件页](https://www.ssslab.cn/software.html)
- [PanguLU 官方 GitHub 仓库](https://github.com/SuperScientificSoftwareLaboratory/PanguLU)
- [PanguLU 5.0.0 用户指南](https://www.ssslab.cn/assets/panguLU/PanguLU_Users_Guide.pdf)
- [PanguLU SC'23 论文](https://www.ssslab.cn/assets/papers/2023-fu-PanguLU.pdf)

#### 性能特性

优势：

- 规则 2D 稀疏块和稀疏 kernel 更适合某些不规则稀疏结构；
- 以多 GPU、多节点为核心目标，通信调度是设计重点；
- `nb` 直接控制块大小，可针对矩阵结构和 GPU 调优；
- 5.0.0 加入任务聚合，优化预处理和 GPU 内存布局；
- 支持实/复数、单/双精度，CPU/GPU 可通过构建开关选择。

代价：

- `nb` 是敏感调优参数，固定值难以适应全部矩阵；
- 很多重要行为由编译宏决定，而不是运行时 config；
- 官方 API 较窄，诊断、内存查询、条件估计、主元统计和算法切换能力不如另外三者；
- 官方函数返回 `void`，调用方不能像 cuDSS 的 `cudssStatus_t` 或 STRUMPACK 的 `ReturnCode` 一样逐调用处理错误；
- AGPL-3.0 对闭源产品和网络服务集成需要法务评估。

#### 本次表现与正确性风险

PanguLU 当前 1/16 的通过率远比速度问题严重。其失败类型具有系统性：

- `Dubcova3` 段错误；
- 14 个矩阵返回了解，但相对残差或相对解误差超阈值；
- 多个矩阵的相对解误差达到 `0.4～10`；
- 测试阈值是较宽松的 `1e-6`，仍然失败。

当前证据不足以断言“PanguLU 本身不稳定”，因为测试条件显著偏离其目标场景，而且关键构建信息缺失：

1. **ABI/类型一致性必须首先检查。**PanguLU 的实数/复数、单/双精度由 `CALCULATE_TYPE_*` 编译宏决定，官方说明要求库和示例编译命令同时设置。调用端定义了 FP64、32 位行索引和 64 位指针；如果预编译库采用了不同 typedef 或宏，可能造成错误结果甚至崩溃。
2. **MC64/METIS 开关未知。**官方说明 MC64 用于提高数值稳定性，METIS/项目重排影响填充和结构。本次 CSV 没有记录实际构建宏。
3. **运行形态不匹配。**本次是 1 rank、`nthread=1`，没有测试 PanguLU 的分布式任务映射和无同步通信优势。
4. **块大小未调优。**全部矩阵固定 `nb=64`，而规则块算法对局部密度和任务粒度敏感。
5. **段错误需要独立诊断。**应对 `Dubcova3` 使用 AddressSanitizer、CUDA Compute Sanitizer 和 PanguLU 的详细日志复现，并确认内存容量与索引范围。

在已完成的非 `AA1` 运行中，PanguLU 的 initialization、factorization、solve 中位占比分别为 18.1%、81.3%、0.6%。这表明一旦正确性修复，优化重点应是数值分解 kernel、块大小和并行任务分布，而不是单 RHS 的三角求解。

#### 用户层面评价

PanguLU 的 API 是四者中最简短的：`init → gstrf → gstrs → finalize`。这降低了最基本的调用门槛，但也把许多策略转移到编译期和内部实现。它适合能够控制部署环境、愿意按目标集群重新构建并调优、并且确实需要多 GPU/多节点稀疏 LU 的研究或 HPC 用户。对要求稳定 ABI、丰富运行时配置、结构化错误处理和宽松商业许可的产品团队，目前接入成本反而可能更高。

---

### 4.3 NVIDIA cuDSS

#### 定位与算法

cuDSS 是 NVIDIA 面向直接稀疏求解的 GPU 库。官方 0.8.0 文档把流程拆为：

```text
创建 handle/config/data/matrix
    → analysis（重排 + 符号分解）
    → factorization / refactorization
    → solve（可带迭代改进）
    → 销毁对象
```

它支持一般、对称、正定、实数和复数矩阵，单/多 RHS、批处理、单 GPU、多 GPU 以及多节点模式。API 使用 CUDA stream，默认模式下 factorization 和 solve 可以异步执行，但 analysis 仍是同步阶段。

官方资料：

- [cuDSS 官方概览与支持范围](https://docs.nvidia.com/cuda/cudss/)
- [cuDSS Getting Started](https://docs.nvidia.com/cuda/cudss/getting_started.html)
- [`cudssExecute` 与对象生命周期](https://docs.nvidia.com/cuda/cudss/functions.html)
- [配置、重排、matching、迭代改进和内存查询类型](https://docs.nvidia.com/cuda/cudss/types.html)
- [cuDSS 性能测量建议](https://docs.nvidia.com/cuda/cudss/tips_and_tricks.html)

#### 性能特性

优势：

- 数值分解和三角求解高度 GPU 化，本次数据中这两段非常快；
- C API 采用统一 opaque object + config/data 查询模式，扩展能力强；
- `cudssConfigSet/Get` 和 `cudssDataSet/Get` 可以控制或查询重排、matching、主元、迭代改进、内存估计和排列；
- analysis 结果可复用，矩阵数值变化但稀疏模式不变时只需重新分解和求解；
- 支持用户 permutation、ND partition tree、设备内存 handler、内存池、CUDA Graph，以及混合主机/设备内存模式；
- 每次调用返回 `cudssStatus_t`，错误处理比 PanguLU 更容易嵌入服务程序。

代价：

- 只支持 NVIDIA GPU，硬件供应商锁定最强；
- 官方仍标记为 Preview，API 可能变更；
- 使用 NVIDIA SDK 许可而不是开源许可，分发和产品合规需检查 EULA；
- 用户通常要自行管理设备内存、CUDA stream 和同步；
- analysis 的主机重排很容易成为端到端瓶颈。

#### 本次表现与瓶颈

cuDSS 在 13 个矩阵上获胜，并且相对 SuperLU_DIST 的有效共同样本主要调用/核心阶段几何平均快 3.62×。但它的阶段结构非常不均衡：

- analysis 中位占比 92.3%；
- factorization 中位占比 6.7%；
- solve 中位占比 0.8%。

NVIDIA 官方文档说明，重排是图算法，目前在主机上执行；默认重排类似基于 METIS 的 nested dissection。当前程序未启用多线程重排，也没有复用排列和 ND tree，所以每个新进程都完整支付 analysis 成本。

此外，冷启动外部墙钟/内部核心时间的中位比达到 2.11×。程序在计时前做了：

- `cudaMalloc`；
- CSR、数值和 RHS 的 H2D 复制；
- stream、handle、config、data 和矩阵 wrapper 创建。

这些操作虽然不在 `total_ms` 中，却是单次调用用户真实等待的一部分。对于 `bundle1`，核心阶段仅 0.075 秒，外部墙钟却为 1.274 秒；对于 3×3 的 `AA1`，核心阶段 0.036 秒，外部墙钟约 1.054 秒。小问题上固定开销完全压过计算。

#### `crystm03` 的精度失败

`crystm03` 上 cuDSS 的相对残差约 `1.96e-2`、相对解误差约 `1.46e-1`。当前程序完全使用默认 config：

- iterative refinement 默认步数为 0，即关闭；
- matching 默认关闭；
- 矩阵被声明为 `GENERAL/FULL`；
- 没有尝试 BTF+COLAMD、COLAMD、AMD 或不同主元阈值。

官方文档指出 matching 可把较大元素移到对角线，通常能改善坏缩放或病态问题的稳定性，但会增加 analysis 成本并可能改变填充；迭代改进也可通过 `CUDSS_CONFIG_IR_N_STEPS` 和 `CUDSS_CONFIG_IR_TOL` 启用。应对 `crystm03` 做“默认、matching、迭代改进、不同重排/主元策略”的精度—时间联合扫描，而不是直接判定 cuDSS 不支持该矩阵。

#### 用户层面评价

cuDSS 的接口复杂度处于中间：对象多于 PanguLU，但状态、配置、错误码和生命周期非常一致。对已经使用 CUDA、数据常驻 GPU、同一稀疏模式反复分解或多 RHS 求解的应用，它最容易把本次性能优势进一步放大。对 CPU-only、AMD GPU、需要完全开源或要求稳定非 Preview ABI 的项目，它则不是合适选择。

---

### 4.4 STRUMPACK

#### 定位与算法

STRUMPACK 是一个 C++ 线性代数库，既可作为标准多波前稀疏直接求解器，也可使用 HSS、BLR、HODLR、Butterfly/HODBF 等秩结构压缩填充块，形成近似直接求解器或预条件器。它还提供预条件 GMRES 和 BiCGStab。

无压缩时，STRUMPACK 是标准多波前精确 LU；启用压缩后，可用一定精度损失换取内存、分解时间和可扩展性收益。这是另外三个库没有同样完整暴露给用户的核心设计差异。

官方资料：

- [STRUMPACK 官方概览](https://portal.nersc.gov/project/sparse/strumpack/master/)
- [稀疏求解器类与分布式模式](https://portal.nersc.gov/project/sparse/strumpack/master/sparse.html)
- [CSR、分布式 CSR 与典型调用流程](https://portal.nersc.gov/project/sparse/strumpack/master/sparse_example_usage.html)
- [GPU 支持](https://portal.nersc.gov/project/sparse/strumpack/master/GPU_Support.html)
- [安装、METIS、MPI、OpenMP、CUDA/HIP 依赖](https://portal.nersc.gov/project/sparse/strumpack/master/installation.html)
- [C 与 Fortran 接口](https://portal.nersc.gov/project/sparse/strumpack/master/C_Interface.html)

#### 性能特性

优势：

- C++ 对象接口清晰，`set matrix → reorder → factor → solve` 与算法阶段一致；
- 提供单节点、MPI 和完全分布式三个主要类；
- 原生支持 0-based CSR 和 block-row distributed CSR；
- 精确直接法、低秩近似、预条件器和 Krylov 方法统一在一个 options 体系内；
- MPI+OpenMP，并支持 NVIDIA CUDA 和 AMD HIP；
- GPU 加速不改变 API，输入输出仍在主机端，集成较方便；
- C 和 Fortran wrapper 使非 C++ 项目也可调用；
- 返回 `ReturnCode`，比只靠日志或进程退出更适合库式集成。

代价：

- 可选依赖很多：METIS/ParMETIS、Scotch/PT-Scotch、BLAS/LAPACK、ScaLAPACK、CUDA/HIP、SLATE、ButterflyPACK、ZFP 等；
- 低秩压缩参数带来额外调优维度，不是“打开就一定更快”；
- 主机输入输出意味着 GPU 路径仍可能有隐藏的数据移动；
- 对象模板、依赖栈和构建选项比 cuDSS 更重。

#### 本次表现与瓶颈

STRUMPACK 通过 15/16，在 `crystm03` 上是唯一快于 SuperLU_DIST且通过的替代库；总体核心阶段相对 SuperLU_DIST 完整 `pdgssvx` 调用几何平均快 1.93×。

但本次配置刻意关闭了 STRUMPACK 的多项特色：

```cpp
set_matching(MatchingJob::NONE);
set_compression(CompressionType::NONE);
set_Krylov_solver(KrylovSolver::DIRECT);
set_reordering_method(ReorderingStrategy::METIS);
```

这意味着当前结果是“无 matching、无压缩、无迭代改进的精确 LU 基线”，并非 STRUMPACK 能达到的唯一性能点。

reorder 占核心阶段时间的中位数为 80.8%。官方文档说明：在 MPI 求解器中选择 METIS、SCOTCH 或 RCM 时，会把完整图收集到 root 再调用串行重排；大图可能产生 root 内存和串行瓶颈。当前只有一个 rank，不存在收集通信，但也完全没有测试 ParMETIS/PT-Scotch 的并行重排。

此外，本批矩阵的 sparsity pattern 都是对称的，而程序调用：

```cpp
set_distributed_csr_matrix(..., false);
```

官方文档指出，当 pattern 对称时传 `true` 可以节省 setup 工作。后续应单独测量这个安全且直接的优化。

#### 用户层面评价

STRUMPACK 是四者中“求解器平台”属性最强的一个，而不只是单一 LU 内核。它适合需要在精确直接法、近似直接法、预条件器和 Krylov 方法之间切换，或者希望同时支持 CPU、NVIDIA/AMD GPU 和 MPI 的项目。若应用只需要最短的 NVIDIA 单 GPU 精确 LU，它的依赖和重排成本可能不如 cuDSS；若应用更看重算法选择、开放源码和硬件可移植性，它更有长期灵活性。

---

## 5. API 接口对比

### 5.1 总览

| 维度 | SuperLU_DIST | PanguLU | cuDSS | STRUMPACK |
|---|---|---|---|---|
| 主接口风格 | C 专家驱动 + 多个显式结构 | C 函数 + `void*` handle | C opaque objects + config/data | C++ 模板对象；另有 C/Fortran |
| 典型流程 | init grid/structs → `pdgssvx` → 多结构清理 | `init → gstrf → gstrs → finalize` | create objects → `cudssExecute(phase)` → destroy | set matrix → `reorder → factor → solve` |
| 本次输入 | 各 rank 的 block-row CSR | rank 0 完整 CSC | 设备端 CSR | 各 rank 的 block-row CSR |
| 索引 | `int_t` 可构建为 32/64 位 | typedef/编译期决定 | 0.8 支持分离 offset/index 类型及 32/64 位 | C++ integer 模板，常用 `int`/`int64_t` |
| 多 RHS | `nrhs` 原生支持 | 文档建议重复调用单 RHS `gstrs` | dense B/X 原生单/多 RHS、batch | solve 接口及 dense RHS 能力 |
| 重用同一 pattern | `SamePattern`/`SamePattern_SameRowPerm` | handle 内重复 factor/solve 能力较少显式说明；同一因子可多次 `gstrs` | analysis 复用、refactorization、用户 permutation/tree | 重排、因子对象可保留并多次 solve |
| 配置方式 | 大型 `options` 结构 | 多为编译宏，少量 options struct | `cudssConfigSet/Get` | `SPOptions` 方法或命令行 |
| 状态与错误 | `info` + stats | 公共函数返回 `void`，主要依赖日志/进程行为 | 每调用 `cudssStatus_t`，另可查询 data info | `ReturnCode` |
| 内存可观测性 | statistics/内部结构，需较低层处理 | 公共 API 较少 | analysis 后可查询 host/device 峰值估计 | 提供多种统计/内存查询 |
| GPU | NVIDIA CUDA、AMD HIP | NVIDIA CUDA；论文还讨论 AMD 版本，但公开构建以 CUDA 为主 | NVIDIA CUDA only | NVIDIA CUDA、AMD HIP |
| 分布式 | MPI 2D/3D | MPI 2D block-cyclic | MG/MGMN，通信层可配置 | MPI 与 fully distributed 类 |
| 许可 | 宽松 BSD 类三条款 | AGPL-3.0 | NVIDIA SDK/EULA | 宽松 BSD 类三条款 |

许可来源：

- [SuperLU_DIST License](https://raw.githubusercontent.com/xiaoyeli/superlu_dist/master/License.txt)
- [PanguLU AGPL-3.0](https://raw.githubusercontent.com/SuperScientificSoftwareLaboratory/PanguLU/main/LICENSE)
- [cuDSS Software License Agreement](https://docs.nvidia.com/cuda/cudss/license.html)
- [STRUMPACK License](https://raw.githubusercontent.com/pghysels/STRUMPACK/master/LICENSE)

### 5.2 API 易用性

从“最少代码完成一次求解”看：

1. PanguLU 调用链最短，但输入必须转为特定 CSC，且编译期 ABI 和运行时诊断负担较大；
2. STRUMPACK 的 C++ API 最符合面向对象直觉，CSR 接入自然，选项可读性好；
3. cuDSS 对象数量较多，但模式高度一致，且状态码、数据查询和 CUDA stream 适合工程化；
4. SuperLU_DIST 最低层，调用和清理结构最多，但也提供最细控制。

从“长期维护大型应用”看，接口短不等于总代价低。结构化错误码、版本查询、内存估计、可复用对象和运行时配置通常比少写几行初始化代码更重要。

### 5.3 API 的状态复用能力决定真实吞吐

四个适配器当前都以新进程处理单矩阵，这恰好浪费了直接求解器最有价值的复用路径：

1. **同一个 A，多组 RHS**：只需重复 solve；
2. **相同 pattern、不同数值**：复用重排和符号分解，只重新数值分解；
3. **相同或近似 A 的时间步进问题**：可进一步使用 refactorization 或复用 row permutation；
4. **小矩阵批量**：cuDSS 的 batch、SuperLU_DIST 的 GPU batched path 可能比逐进程调用更合适。

当前数据中 solve 通常不足总时间的 1%～3%。因此，多 RHS 场景如果仍为每个 RHS 重启进程和重新 analysis，用户会浪费一个到两个数量级的潜在吞吐。

---

## 6. 性能瓶颈的根因分析

### 6.1 稀疏直接法不能只看 n 和输入 nnz

真正决定时间和内存的关键量通常是：

- 重排后的填充量 `nnz(L+U)`；
- 消去树高度与并行宽度；
- frontal/supernode/block 的尺寸分布；
- 浮点操作量；
- 主元替换、matching、缩放和迭代改进次数；
- CPU-GPU 数据移动与通信量；
- 峰值 host/device memory。

本次 CSV 只有输入 `n` 和 `nnz`，没有上述量，因此无法建立可靠的复杂度模型。例如，两个输入 nnz 接近的矩阵可能因填充量相差数倍而表现完全不同。后续应把 `nnz(L+U)`、估算 FLOPs、峰值内存和主元统计作为必填字段。

### 6.2 主机端重排已成为 GPU 求解器的主瓶颈

cuDSS 和 STRUMPACK 的分解/solve 已被 GPU 加速到很短，但重排仍主要是主机图算法：

- cuDSS analysis 中位 92.3%；
- STRUMPACK reorder 中位 80.8%。

这形成典型的 Amdahl 定律限制：即使把数值分解再加速 2 倍，端到端收益也非常小。更有效的方向是：

- 缓存并复用 permutation、elimination/ND tree；
- 在相同 sparsity pattern 的序列问题中不重复 analysis；
- 启用 cuDSS MT reordering；
- 多 rank 时为 STRUMPACK 使用 ParMETIS/PT-Scotch；
- 对已知网格问题使用几何重排；
- 把多个小任务聚合而不是逐个冷启动。

### 6.3 冷启动和数据生命周期

外部墙钟明显高于核心阶段，尤其是 cuDSS 和 STRUMPACK。固定成本包括：

- 动态库加载；
- MPI runtime singleton 初始化；
- CUDA context 和 stream；
- handle/config/data 创建；
- 主机/设备分配；
- Matrix Market 读取和对称展开；
- CSR/CSC 转换；
- H2D/D2H；
- 验证和对象销毁。

用户若在应用进程中常驻求解器、复用内存池和对象、让矩阵/向量驻留设备，并对多个 RHS 批量求解，真实吞吐会远优于当前脚本。

### 6.4 GPU 利用率受任务粒度和结构影响

稀疏 LU 的并行任务尺寸不均匀，存在依赖链：

- 太小的 supernode/front/block 无法充分占用 GPU；
- 太大的块可能引入零填充或降低并发；
- 规则块对稀疏密度不均的矩阵可能负载不平衡；
- 三角 solve 通常并行度低，受依赖和内存带宽限制；
- 多流只在存在足够独立任务时有效。

SuperLU_DIST/STRUMPACK 的超节点或多波前方法倾向于制造适合 dense BLAS 的块；PanguLU 倾向于保持块内稀疏并自适应选 kernel；cuDSS 隐藏内部选择。没有 Nsight Systems/Compute、rocprof、MPI trace 或 per-kernel 统计时，不能仅由总时间断言是哪一种 kernel 慢。

### 6.5 数值稳定性与性能不可分割

重排、matching、缩放、主元策略和迭代改进都会同时改变：

- 正确性；
- 填充量；
- analysis 时间；
- factorization 时间；
- solve 时间和迭代次数。

本次并非相同的数值策略：

- SuperLU_DIST 显式启用双精度迭代改进；
- cuDSS 默认关闭迭代改进和 matching；
- STRUMPACK 显式关闭 matching，并选择 `DIRECT`；
- PanguLU 的 MC64 构建开关未知。

因此“默认配置速度”与“达到相同精度后的速度”是两个不同问题。正式报告应分别给出：

1. 原厂/项目默认配置；
2. 统一正确性目标下的稳健配置；
3. 每个库调优后的最佳有效配置。

### 6.6 `denormal` 暴露的是条件数问题

`denormal` 的关键现象：

| 求解器 | 相对残差 | 相对解误差 | 结论 |
|---|---:|---:|---|
| SuperLU_DIST | `1.82e-8` | `≈1.0` | 残差很小但前向误差大 |
| PanguLU | `3.38e2` | `≈1.0` | 残差和解都不可接受 |
| cuDSS | `1.16e-7` | `≈1.0` | 残差不算极端，但解不可信 |
| STRUMPACK | `5.10e-8` | `≈1.0` | 同样表现出前向敏感性 |

若 \(\kappa(A)\) 很大，经典误差关系允许很小的后向误差被条件数放大成很大的前向误差。因此：

- 不能只用残差判断解是否可信；
- 真实应用没有已知 \(x^\*\) 时，应报告 scaled backward error、条件数估计和误差界；
- 应尝试行列缩放、matching、更稳健主元策略、迭代改进和更高精度参考；
- 如果矩阵本身接近奇异，应把它归类为鲁棒性测试，而不是普通性能样本。

---

## 7. 各库的针对性优化建议

### 7.1 SuperLU_DIST

优先级从高到低：

1. 保留当前完整覆盖 `pdgssvx`、带前后屏障并对各 rank 取最大值的总计时；在不改变该边界的前提下补充 scaling/matching、column ordering、symbolic、distribution、factor、solve、refinement 等细分；
2. 记录运行时版本、`GPU_ACC`、`get_acc_offload()`、BLAS 实现和线程数；
3. 对相同 pattern 使用 `SamePattern` 或 `SamePattern_SameRowPerm`；
4. 保留 `SOLVEstruct`，对多 RHS 重复 solve；
5. 单节点先扫描 MPI ranks × OpenMP threads，避免 rank 过多导致 GPU 争用；
6. 多节点比较接近方形的 2D 网格和新版 3D 通信规避算法；
7. 使用高性能 BLAS，而不是仅具功能性的参考 BLAS；
8. 对病态矩阵比较 `ReplaceTinyPivot`、row matching、equilibration 与 iterative refinement 组合。

### 7.2 PanguLU

必须先做正确性，再做性能：

1. 保存并核对库与调用端的 `CALCULATE_TYPE_R64`、索引 typedef、`sizeof_value`、复数标志；
2. 在输出中打印 PanguLU commit/version 和全部构建宏；
3. 确认是否启用 MC64；对失败矩阵分别测试 MC64 开/关；
4. 确认 METIS 或项目自带并行重排实际生效；
5. 用官方 example 对同一矩阵交叉验证适配器的 CSR→CSC 和 RHS/solution 语义；
6. 对 `Dubcova3` 使用 ASan、UBSan、Compute Sanitizer 和详细日志；
7. 正确性通过后扫描 `nb`，至少包含 32、64、128、256 及矩阵相关候选；
8. 将 `nthread` 从 1 扩展到物理核范围，并测试多 rank/多 GPU；
9. 分别报告单 GPU kernel 性能和多 GPU 通信扩展，避免用 singleton 结果评价其核心设计。

### 7.3 cuDSS

1. 保持 handle/config/data 和设备内存常驻，不要每个矩阵/每个 RHS 重建；
2. 对相同 sparsity pattern 复用 analysis，或导出/复用 permutation 与 ND partition tree；
3. 启用 MT reordering，并扫描 ND levels；
4. 对 `crystm03` 测试 matching、迭代改进、BTF+COLAMD/COLAMD、AMD、主元策略；
5. 通过 `CUDSS_DATA_MEMORY_ESTIMATES` 在 factorization 前记录永久/峰值 host/device memory；
6. 使用 device memory handler 或异步内存池，减少 `cudaMalloc/cudaFree`；
7. 若业务矩阵确实为对称或正定，使用准确的 matrix type/view，并验证只存三角部分的收益；
8. 对同模式数值更新使用 factorization/refactorization，而不是重新 analysis；
9. 分开报告冷启动、warm solve、H2D/D2H 和纯 GPU 阶段。

### 7.4 STRUMPACK

1. 对这批对称 pattern，把 `symmetric_pattern` 改为 `true` 后单独测量；
2. 多 rank 时将串行 METIS 与 ParMETIS/PT-Scotch 对比；
3. 若矩阵数值稳定性不足，恢复默认/稳健 matching，而不是固定 `NONE`；
4. 如果目标是精确解，可比较 `DIRECT` 与默认 iterative refinement；
5. 如果目标允许近似或预条件，评估 BLR/HSS/HODLR 等压缩的时间—内存—迭代次数三方权衡；
6. 记录 GPU 是否真正启用、SLATE 是否启用、每 rank stream 数和 BLAS/ScaLAPACK；
7. 多 RHS 时保留因子并重复 solve；
8. 记录 front rank、压缩率、factor nnz、峰值内存和 Krylov 迭代数。

---

## 8. 建议的下一轮基准设计

### 8.1 测试矩阵分组

至少按以下特征分组，而不是把所有矩阵混成一个平均值：

- 对称正定；
- 对称不定；
- 一般非对称；
- 对角占优；
- 坏缩放/病态；
- 2D PDE、3D PDE、图、电路和结构力学；
- 高填充与低填充；
- 小矩阵批量、中型单 GPU、大型多 GPU。

### 8.2 三种工作负载

每个库都应测试：

1. **Cold one-shot**：进程启动到结果；
2. **Same pattern, new values**：复用重排/符号结构；
3. **Same factors, multiple RHS**：只重复 solve。

这三种模式对应完全不同的用户成本。本次只覆盖第一种。

### 8.3 统一正确性指标

建议同时记录：

\[
r_{\mathrm{rel}}=\frac{\|Ax-b\|_2}{\|b\|_2}
\]

\[
\eta=\frac{\|Ax-b\|}{\|A\|\|x\|+\|b\|}
\]

以及：

- 有参考解时的相对前向误差；
- 库返回的 backward error；
- 条件数估计或 reciprocal condition number；
- NaN/Inf、零主元、替换主元数量；
- 迭代改进次数。

所有库使用相同阈值，或明确给出“默认阈值”和“统一阈值”两套结果。

### 8.4 统一计时边界

建议输出以下独立字段：

```text
process_startup
matrix_io
symmetry_expansion
format_conversion
host_allocation
device_allocation
h2d
reordering
symbolic
numeric_factorization
solve
iterative_refinement
d2h
verification
cleanup
end_to_end
```

GPU 阶段必须在边界处同步；MPI 阶段使用所有 rank 的最大值；外部墙钟应由统一 harness 测量。

### 8.5 重复、顺序与统计

- 每个配置至少 5～10 次；
- 第一次作为 cold，后续作为 warm；
- 随机化库顺序；
- 报告 median、P95 和最小值，不只报单次；
- 固定 CPU affinity、NUMA、GPU clocks 或至少记录；
- 记录系统后台负载和温度；
- 不在启用详细性能日志的构建与关闭日志的构建之间直接比较。

### 8.6 平台与版本元数据

每行结果至少保存：

- CPU 型号、socket、NUMA、物理核；
- GPU 型号、显存、驱动、CUDA/ROCm；
- MPI、OpenMP、编译器和优化选项；
- 求解器版本/commit；
- BLAS/LAPACK/ScaLAPACK；
- 32/64 位索引；
- MPI ranks、threads/rank、GPU/rank；
- 实际重排、matching、pivot、refinement、compression；
- 峰值 host/device memory；
- 可执行文件哈希。

当前 CSV 缺少这些信息，是限制结论可复现性的最大因素之一。

---

## 9. 选型建议

| 用户场景 | 首选建议 | 原因与注意点 |
|---|---|---|
| NVIDIA 单 GPU、一次大矩阵、追求当前最快 | cuDSS | 本次有效样本最快；但要接受 Preview API、NVIDIA 锁定和 EULA |
| NVIDIA GPU、同 pattern 多次更新 | cuDSS | analysis 可复用，数值分解很快；应常驻对象和内存 |
| 多 RHS | cuDSS / SuperLU_DIST / STRUMPACK | 三者都可保留分析和因子；不要逐 RHS 冷启动 |
| 开源、NVIDIA/AMD GPU、MPI | STRUMPACK 或 SuperLU_DIST | 二者许可宽松且支持 CUDA/HIP；需按矩阵结构实测 |
| 需要低秩压缩或直接法预条件器 | STRUMPACK | HSS/BLR/HODLR/Butterfly/ZFP 和 Krylov 组合是独特优势 |
| 需要成熟、细粒度分布式 LU 控制 | SuperLU_DIST | 2D/3D 网格、专家驱动、复用和误差控制成熟 |
| 大规模多 GPU 稀疏块研究、可接受重新构建调优 | PanguLU | 设计目标匹配，但必须先解决本次正确性和 ABI/构建问题 |
| 闭源商业产品 | SuperLU_DIST / STRUMPACK，或经法务确认的 cuDSS | PanguLU 为 AGPL-3.0；cuDSS 需核对 NVIDIA 分发条款 |
| CPU-only | SuperLU/STRUMPACK；PanguLU CPU 构建需另测 | cuDSS 不适用 |
| 病态矩阵、稳健性优先 | 先做条件分析，再比较稳健配置 | 默认速度排名不足以代表稳健配置下的表现 |

最终建议不是永久绑定一个库，而是建立统一 solver abstraction：

```text
MatrixDescriptor
SolverConfig
analyze(pattern)
factor(values)
solve(rhs)
query_stats()
reset()
```

在其后分别封装四个 backend。这样应用可以按硬件、矩阵类型、正确性要求和许可策略动态选择，也能让冷启动、复用和统计口径保持一致。

---

## 10. 最终结论

从当前用户可见的实测结果看，**cuDSS 提供了最强的单 GPU 核心计算性能，STRUMPACK 提供了更均衡的开放源码与算法灵活性，SuperLU_DIST 提供了最成熟的分布式低层控制和稳健基线，PanguLU 则展示了有针对性的分布式稀疏块设计，但当前适配结果尚未达到可用于性能决策的正确性门槛。**

当前最重要的三个行动不是继续扩大矩阵数量，而是：

1. **先修复和验证 PanguLU 的正确性及构建 ABI；**
2. **把测试改成常驻进程，分别测 cold、same-pattern 和 multi-RHS；**
3. **补齐版本、硬件、build flags、填充量、内存和完整阶段计时。**

完成这三步后，才适合进一步做多 rank、多 GPU 和多节点扩展曲线。届时很可能得到与本次 singleton 排名不同的结论，尤其是 PanguLU、SuperLU_DIST 和 STRUMPACK 的相对位置。

---

## 11. 官方资料索引

### SuperLU_DIST

- [SuperLU 官方主页](https://portal.nersc.gov/project/sparse/superlu/)
- [SuperLU FAQ](https://portal.nersc.gov/project/sparse/superlu/faq.html)
- [SuperLU_DIST Doxygen](https://portal.nersc.gov/project/sparse/superlu/superlu_dist_code_html/)
- [`pdgssvx` 文档](https://portal.nersc.gov/project/sparse/superlu/superlu_dist_code_html/pdgssvx_8c.html)
- [SuperLU_DIST License](https://raw.githubusercontent.com/xiaoyeli/superlu_dist/master/License.txt)

### PanguLU

- [SSSLab PanguLU 软件页](https://www.ssslab.cn/software.html)
- [PanguLU GitHub](https://github.com/SuperScientificSoftwareLaboratory/PanguLU)
- [PanguLU 5.0.0 用户指南](https://www.ssslab.cn/assets/panguLU/PanguLU_Users_Guide.pdf)
- [PanguLU SC'23 论文](https://www.ssslab.cn/assets/papers/2023-fu-PanguLU.pdf)
- [PanguLU License](https://raw.githubusercontent.com/SuperScientificSoftwareLaboratory/PanguLU/main/LICENSE)

### NVIDIA cuDSS

- [cuDSS 官方文档首页](https://docs.nvidia.com/cuda/cudss/)
- [Getting Started](https://docs.nvidia.com/cuda/cudss/getting_started.html)
- [Functions](https://docs.nvidia.com/cuda/cudss/functions.html)
- [Data Types 与配置项](https://docs.nvidia.com/cuda/cudss/types.html)
- [Advanced Features](https://docs.nvidia.com/cuda/cudss/advanced_features.html)
- [Tips and Tricks](https://docs.nvidia.com/cuda/cudss/tips_and_tricks.html)
- [Release Notes](https://docs.nvidia.com/cuda/cudss/release_notes.html)
- [Software License Agreement](https://docs.nvidia.com/cuda/cudss/license.html)

### STRUMPACK

- [STRUMPACK 官方概览](https://portal.nersc.gov/project/sparse/strumpack/master/)
- [Sparse Direct Solver](https://portal.nersc.gov/project/sparse/strumpack/master/sparse.html)
- [Sparse Example Usage](https://portal.nersc.gov/project/sparse/strumpack/master/sparse_example_usage.html)
- [GPU Support](https://portal.nersc.gov/project/sparse/strumpack/master/GPU_Support.html)
- [Installation and Requirements](https://portal.nersc.gov/project/sparse/strumpack/master/installation.html)
- [C and Fortran Interfaces](https://portal.nersc.gov/project/sparse/strumpack/master/C_Interface.html)
- [STRUMPACK GitHub](https://github.com/pghysels/STRUMPACK)
- [STRUMPACK License](https://raw.githubusercontent.com/pghysels/STRUMPACK/master/LICENSE)

## 12. 限制声明

本报告没有取得远端日志文件、实际可执行文件、环境脚本和运行机器，因此无法确认：

- SuperLU_DIST、cuDSS 和 STRUMPACK 的实际二进制版本；
- PanguLU 实际链接库是否确为适配器注释中的 5.0.0；
- 各库是否实际启用 GPU、使用何种 BLAS、使用多少 OpenMP 线程；
- PanguLU 是否启用 MC64/METIS，以及库与调用端 typedef 是否一致；
- CPU/GPU 型号、内存容量、驱动、编译器、MPI 和 CUDA 版本；
- 各矩阵的 `nnz(L+U)`、FLOPs、条件数和峰值内存。

因此所有性能数字都应理解为“当前 CSV 与适配器代码所代表的一次观测”，而不是四个软件项目在最佳配置下的官方性能结论。
