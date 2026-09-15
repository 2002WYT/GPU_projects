# SpGEMM Benchmark：custom CUDA vs AMGX

对称稀疏矩阵-矩阵乘法（A × Aᵀ）的双后端基准测试框架，对比自研 CUDA 实现（`custom`）与 NVIDIA AMGX 库（`amgx`）的性能与正确性。

## 简介

本项目计算 C = A · B，其中 B 由 A 按以下模式派生：

- `transpose`（默认）：B = Aᵀ，即计算 A · Aᵀ
- `self`：B = A（须方阵）
- `random`：以 `--seed` 生成与 A 等规模的随机稀疏 B

每个后端在同一 A 上独立运行：预热 + 重复计时，并在进程内对照 CPU 参考实现做正确性校验，无需落盘中间结果。

## 架构（custom 后端）

行按“输出行估计非零数（bound）”分入 5 个 bin，分别走不同路径：

| bin | bound 上界 | 路径 | 说明 |
|-----|-----------|------|------|
| 0 | ≤ 32 | `small_rows<64,8>` | 每 8 线程一行，shared 内存哈希表 |
| 1 | ≤ 128 | `small_rows<256,32>` | 每 warp 一行，shared 内存哈希表 |
| 2 | ≤ 512 | `medium_rows<1024>` | 每 block 一行，shared 内存哈希表 |
| 3 | ≤ 1024 | `medium_rows<2048>` | 每 block 一行，shared 内存哈希表 |
| 4 | > 1024 | 排序路径 | 分批 expand → 分段基数排序 → 合并 |

- **哈希路径**（bin 0–3）：shared 内存开放寻址哈希表累加，装载率 ≤ 0.5，不保证列有序，保留显式零值。
- **排序路径**（bin 4）：先展开所有标量乘积 `(j, aᵣⱼ·bⱼₖ)`，按列做分段基数排序（CUB `DeviceSegmentedRadixSort`），再扫描合并相同列。按空闲显存的 1/4 自适应分批，避免单批展开量超出显存。

### 设计原则：可移植，不针对硬件特调

- bin 划分完全由矩阵的行工作量决定，**不查询设备名或架构**。
- 显存预算通过 `cudaMemGetInfo` 在运行时查询，按空闲显存比例分配，与具体卡型无关。
- `CUDA_ARCHITECTURES` 仅用于编译覆盖，**内核从不在架构名上分派**。

## 目录结构

```
├── CMakeLists.txt
├── README.md
├── run_bench.sh                 # 一键跑套件（调用 run_suite.py）
├── include/
│   ├── backend.hpp              # Backend 接口、HostCSR、validate_product
│   ├── cpu_verify.hpp           # 进程内 CPU 参考校验
│   └── matrix_io.hpp            # Matrix Market 读取、转置、指纹、CPU 参考乘
├── src/
│   ├── custom.cu                # 自研 SpGEMM 后端
│   ├── amgx_backend.cu          # AMGX 参考后端
│   └── bench_main.cpp           # CLI 入口：计时 + 校验 + JSON 输出
├── scripts/
│   ├── run_suite.py             # 串行配对基准驱动（仅标准库）
│   └── build.sh                 # 构建 AMGX + 本项目
├── tests/
│   ├── test_input.cpp           # matrix_io.hpp 解析测试
│   ├── test_suite.py            # run_suite.py 逻辑单测
│   └── test_gpu.py              # 生成固定 fixture 并跑双后端自校验
├── test-artifacts/gpu/*.mtx     # test_gpu.py 使用的固定测试矩阵
├── test_input_fixture.mtx       # test_input.cpp 的小输入
└── docs/superpowers/specs/      # 设计文档
```

## 依赖

- **CUDA Toolkit**（含 CUB；CUB 自 CUDA 11 起随 toolkit 分发）
- **CMake ≥ 3.18**
- **NVIDIA AMGX 2.5.0**（仅 `amgx` 后端需要）—— 须可用的源码树（含 `include/`、`external/rapidjson/`）与构建出的 `libamgxsh.so`
- 支持 FP64 `atomicAdd` 的 GPU（sm_60+）

## 构建

```bash
# AMGX_ROOT        指向 AMGX 源码树（含 internal headers）
# AMGX_BUILD       指向 libamgxsh.so 所在目录（缺省为 ${AMGX_ROOT}/build）
# CUDA_ARCHITECTURES  编译目标架构（缺省 70；仅编译覆盖，不影响内核分派）
AMGX_ROOT=/path/to/AMGX-2.5.0 \
AMGX_BUILD=/path/to/AMGX-2.5.0/build \
CUDA_ARCHITECTURES=70 \
bash scripts/build.sh
```

`scripts/build.sh` 先构建 AMGX 的 `amgxsh` 目标，再用 CMake 构建本项目，产出 `build/bench_custom`、`build/bench_amgx`、`build/test_input`。

也可直接用 cmake：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=70 -DAMGX_ROOT=/path/to/AMGX-2.5.0
cmake --build build --parallel
```

> 说明：仓库内的 `scripts/build.sh` 与 `run_bench.sh` 含作者本地环境的绝对路径。复现时请用上述环境变量覆盖 AMGX 路径，并编辑 `run_bench.sh` 里的 `--matrix-dir` 指向你自己的矩阵目录。

## 运行

### 单矩阵快速检查

```bash
./build/bench_custom --a path/to/A.mtx --b-mode transpose --warmup 2 --repeat 5
./build/bench_amgx   --a path/to/A.mtx --b-mode transpose --warmup 2 --repeat 5
```

CLI 参数：

- `--a MATRIX`（必填）输入 A
- `--b MATRIX` 输入 B；省略时按 `--b-mode` 派生
- `--b-mode transpose|self|random`（默认 `transpose`）
- `--seed` 随机 B 种子（默认 20260910）
- `--warmup` 预热次数（默认 5）
- `--repeat` 计时次数（默认 20）

每个后端输出两行 JSON：

- `RESULT {...}` 计时与规模信息（rows/cols/nnz_a/nnz_b/nnz_c、first/median/mean/min_ms、逐次 times_ms）
- `VERIFY {"passed":...}` CPU 校验结果（max_error、sample_rows、identity_trials=3）

### 套件批跑

```bash
python3 scripts/run_suite.py \
    --matrix-dir /path/to/mtxs/ \
    --bin-dir build --results-dir results \
    --b-mode transpose --warmup 2 --repeat 5 --timeout 900
```

`run_suite.py` 对目录下每个 `.mtx`（递归、符号链接去重，每个物理矩阵只测一次）做配对基准：交替先后运行 amgx/custom，解析各自 `RESULT`/`VERIFY`，计算加速比 `amgx_median / custom_median`。

输出在 `results/<UTC时间戳>/`：

- `results.csv` —— 每矩阵的双后端中位时间 + 加速比
- `summary.json` —— 聚合：算术/几何平均加速比、覆盖率、失败统计
- `case_NNNNNN/` —— 每案例 stdout/stderr/迭代/校验日志
- `metadata.json` —— 平台、Python、SLURM 环境变量、源文件 SHA256、nvidia-smi

## 正确性校验

每个后端在进程内（无需落盘 .csr）对照 CPU 参考实现校验自身乘积：

1. **采样行精确比对**：对若干采样行，用堆归并的 CPU 参考行与后端输出逐项比对（列、值）。
2. **Freivalds 恒等式**（3 次）：随机 ±1 向量 x，检验 ‖C·x − A·(B·x)‖ 在容差内。
3. **相对容差**：`|a−b| ≤ 1e-9 + 1e-8·max(|a|,|b|,scale)`。

`run_suite.py` 把未通过 `VERIFY` 的案例标记为 `verify_failed`。

## 测试

```bash
# C++ 解析测试（需先 build）
./build/test_input

# Python 单测
python3 -m pytest tests/test_suite.py        # run_suite.py 逻辑
python3 tests/test_gpu.py --bin-dir build    # 生成 fixture 并跑双后端自校验
```

## 基准结果示例

在单一 V100（16 GB）上对 SuiteSparse 矩阵集（A × Aᵀ，warmup=2，repeat=5）的代表性结果：

| 矩阵 | amgx 中位 (ms) | custom 中位 (ms) | amgx / custom |
|------|---------------|-----------------|---------------|
| cage14 | 1673.8 | 68.2 | **24.5×** custom 快 |
| poisson3Db | 82.1 | 30.2 | 2.7× custom 快 |
| memplus | 14.0 | 5.5 | 2.6× custom 快 |
| add20 | 0.95 | 0.34 | 2.8× custom 快 |
| bundle1 | 87.7 | 420.8 | 0.21×（amgx 快） |
| nemeth26 | 24.2 | 207.6 | 0.12×（amgx 快） |

总体：在 39 个双后端均通过校验的矩阵中，custom 在约 21 个上更快，amgx 在约 18 个上更快。custom 在大行工作量、长输出的矩阵上优势明显（cage14 达 24.5×）；amgx 在部分 bin4 稠密展开的矩阵上仍更快。完整逐矩阵结果见运行后生成的 `results/`。
