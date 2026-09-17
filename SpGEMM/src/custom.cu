#include "backend.hpp"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <algorithm>
#include <array>
#include <climits>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

// 版本 A：小/中行 shared hash，大行分批 expand → sort → merge。
// 输入须为合法 int32 CSR；要求 FP64 atomicAdd 支持（sm_60+）。
// hash 路径不保证列有序；保留显式零值；所有操作使用默认流。
namespace spgemm {
namespace {
using U64 = unsigned long long;
constexpr int kWarp = 32, kBlock = 128, kMergeBlock = 256, kBins = 5;
constexpr unsigned kFullMask = 0xffffffffu;

struct ComplexValue { double re, im; };

template<class Value> struct ValueOps;
template<> struct ValueOps<double> {
    __host__ __device__ static double zero() { return 0.0; }
    __host__ __device__ static double multiply(double a, double b) { return a * b; }
    __device__ static void atomic_add(double* destination, double value) { atomicAdd(destination, value); }
};
template<> struct ValueOps<ComplexValue> {
    __host__ __device__ static ComplexValue zero() { return {0.0, 0.0}; }
    __host__ __device__ static ComplexValue multiply(ComplexValue a, ComplexValue b) {
        return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
    }
    __device__ static void atomic_add(ComplexValue* destination, ComplexValue value) {
        atomicAdd(&destination->re, value.re);
        atomicAdd(&destination->im, value.im);
    }
};

void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

unsigned blocks(size_t count, unsigned threads) {
    return static_cast<unsigned>((count + threads - 1) / threads);
}

template<class T>
struct Buffer {
    T* p = nullptr;
    size_t capacity = 0;
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { if (p) cudaFree(p); }

    // 扩容不保留旧内容；多次 multiply 复用已分配空间。
    void reserve(size_t count) {
        if (count <= capacity) return;
        if (count > std::numeric_limits<size_t>::max() / sizeof(T))
            throw std::overflow_error("workspace size overflow");
        T* next = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&next), count * sizeof(T)));
        if (p) {
            const auto error = cudaFree(p);
            if (error != cudaSuccess) { cudaFree(next); check(error); }
        }
        p = next;
        capacity = count;
    }

    void upload(const std::vector<T>& values) {
        reserve(values.size());
        if (!values.empty())
            check(cudaMemcpy(p, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
};

// 轻量视图统一传递 CSR 和临时输出，缩短 kernel 参数列表。
template<class Value>
struct CSRView { const int* row; const int* col; const Value* val; };
template<class Value>
struct RowOutput { const U64* offsets; int* col; Value* val; int* counts; };

template<class Value>
struct DeviceCSR {
    int rows, cols;
    Buffer<int> row, col;
    Buffer<Value> val;
    explicit DeviceCSR(const HostCSR& h) : rows(h.rows), cols(h.cols) {
        row.upload(h.row); col.upload(h.col);
        std::vector<Value> values(h.val.size());
        for (size_t i = 0; i < values.size(); ++i) {
            if constexpr (std::is_same<Value, double>::value) values[i] = h.val[i];
            else values[i] = {h.val[i], h.complex ? h.imag[i] : 0.0};
        }
        val.upload(values);
    }
    CSRView<Value> view() const { return {row.p, col.p, val.p}; }
};

__device__ unsigned hash_key(int key) {
    unsigned x = static_cast<unsigned>(key);
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15;
    return x;
}

template<class Value, int H>
__device__ void hash_add(int* keys, Value* vals, int col, Value value) {
    unsigned slot = hash_key(col) & (H - 1);
    while (true) {
        const int old = atomicCAS(keys + slot, -1, col);
        if (old == -1 || old == col) break;
        slot = (slot + 1) & (H - 1);
    }
    ValueOps<Value>::atomic_add(vals + slot, value);
}

// raw_count = Σ nnz(B[j])；upper = min(raw_count, B.cols)。
// upper 用于输出预留，排序展开必须使用未截断的 raw_count。
template<class Value>
__global__ void classify(int n, int ncols, CSRView<Value> a, CSRView<Value> b,
                         U64* upper, U64* raw_count, int* bins, int* sizes, int* counts) {
    const int r = blockIdx.x * (blockDim.x / kWarp) + threadIdx.x / kWarp;
    const int lane = threadIdx.x % kWarp;
    if (r >= n) return;
    U64 work = 0;
    for (long long p = static_cast<long long>(a.row[r]) + lane; p < a.row[r + 1]; p += kWarp)
        work += U64(b.row[a.col[p] + 1] - b.row[a.col[p]]);
    for (int step = kWarp / 2; step; step /= 2)
        work += __shfl_down_sync(kFullMask, work, step);
    if (lane == 0) {
        const U64 bound = work < U64(ncols) ? work : U64(ncols);
        upper[r] = bound; raw_count[r] = work; counts[r] = 0;
        const int bin = bound <= 32 ? 0 : bound <= 128 ? 1 : bound <= 512 ? 2 : bound <= 1024 ? 3 : 4;
        if (bound) bins[size_t(bin) * n + atomicAdd(sizes + bin, 1)] = r;
        if (r == 0) { upper[n] = 0; raw_count[n] = 0; counts[n] = 0; }
    }
}

// 每个 8/32 线程组处理一行；哈希表装载率上界为 0.5。
template<class Value, int H, int GROUP>
__global__ void small_rows(int num, const int* rows, CSRView<Value> a, CSRView<Value> b, RowOutput<Value> out) {
    constexpr int GROUPS = kBlock / GROUP;
    static_assert(GROUP == 8 || GROUP == kWarp, "unsupported group size");
    __shared__ int keys[GROUPS * H];
    __shared__ Value vals[GROUPS * H];
    const int group = threadIdx.x / GROUP, lane = threadIdx.x % GROUP;
    const int q = blockIdx.x * GROUPS + group;
    if (q >= num) return;
    const int r = rows[q], base = group * H;
    const unsigned mask = GROUP == kWarp ? kFullMask : (0xffu << (threadIdx.x % kWarp / GROUP * GROUP));
    for (int slot = lane; slot < H; slot += GROUP) { keys[base + slot] = -1; vals[base + slot] = ValueOps<Value>::zero(); }
    __syncwarp(mask);
    for (long long p = a.row[r]; p < a.row[r + 1]; ++p) {
        const int j = a.col[p];
        const Value value = a.val[p];
        for (long long k = static_cast<long long>(b.row[j]) + lane; k < b.row[j + 1]; k += GROUP)
            hash_add<Value, H>(keys + base, vals + base, b.col[k], ValueOps<Value>::multiply(value, b.val[k]));
    }
    __syncwarp(mask);
    int cursor = 0;
    for (int slot = lane; slot < H; slot += GROUP) {
        const int col = keys[base + slot];
        const unsigned active = __ballot_sync(mask, col != -1);
        const int rank = __popc(active & ((1u << (threadIdx.x % kWarp)) - 1u));
        if (col != -1) {
            const U64 at = out.offsets[r] + cursor + rank;
            out.col[at] = col; out.val[at] = vals[base + slot];
        }
        cursor += __popc(active);
    }
    if (lane == 0) out.counts[r] = cursor;
}

// 每个 block 处理一行，多个 warp 分担 A[r] 的非零项。
template<class Value, int H>
__global__ void medium_rows(int num, const int* rows, CSRView<Value> a, CSRView<Value> b, RowOutput<Value> out) {
    const int q = blockIdx.x;
    if (q >= num) return;
    const int r = rows[q], lane = threadIdx.x % kWarp, warp = threadIdx.x / kWarp;
    __shared__ int keys[H], cursor;
    __shared__ Value vals[H];
    for (int slot = threadIdx.x; slot < H; slot += kBlock) { keys[slot] = -1; vals[slot] = ValueOps<Value>::zero(); }
    if (threadIdx.x == 0) cursor = 0;
    __syncthreads();
    for (long long p = static_cast<long long>(a.row[r]) + warp; p < a.row[r + 1]; p += kBlock / kWarp) {
        const int j = a.col[p];
        const Value value = a.val[p];
        for (long long k = static_cast<long long>(b.row[j]) + lane; k < b.row[j + 1]; k += kWarp)
            hash_add<Value, H>(keys, vals, b.col[k], ValueOps<Value>::multiply(value, b.val[k]));
    }
    __syncthreads();
    for (int slot = threadIdx.x; slot < H; slot += kBlock) {
        const int col = keys[slot];
        const unsigned active = __ballot_sync(kFullMask, col != -1);
        int start = lane == 0 ? atomicAdd(&cursor, __popc(active)) : 0;
        start = __shfl_sync(kFullMask, start, 0);
        if (col != -1) {
            const U64 at = out.offsets[r] + start + __popc(active & ((1u << lane) - 1u));
            out.col[at] = col; out.val[at] = vals[slot];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) out.counts[r] = cursor;
}

__global__ void gather_counts(int num, const int* rows, const U64* raw_count, U64* batch_count) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size_t(num)) batch_count[i] = raw_count[rows[i]];
    if (i == size_t(num)) batch_count[i] = 0;
}

__global__ void relative_offsets(int num, const U64* offsets, int first, U64* local) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i <= size_t(num)) local[i] = offsets[size_t(first) + i] - offsets[first];
}

// 展开所有标量乘积；各 warp 原子申请互不重叠的区间。
template<class Value>
__global__ void expand_rows(int num, const int* rows, CSRView<Value> a, CSRView<Value> b,
                            const U64* offsets, int* col, Value* val) {
    const int q = blockIdx.x;
    if (q >= num) return;
    const int r = rows[q], lane = threadIdx.x % kWarp, warp = threadIdx.x / kWarp;
    __shared__ int cursor;
    if (threadIdx.x == 0) cursor = 0;
    __syncthreads();
    for (long long p = static_cast<long long>(a.row[r]) + warp; p < a.row[r + 1]; p += kBlock / kWarp) {
        const int j = a.col[p], length = b.row[j + 1] - b.row[j];
        int start = lane == 0 ? atomicAdd(&cursor, length) : 0;
        start = __shfl_sync(kFullMask, start, 0);
        for (long long k = static_cast<long long>(b.row[j]) + lane; k < b.row[j + 1]; k += kWarp) {
            const U64 at = offsets[q] + start + (k - b.row[j]);
            col[at] = b.col[k]; val[at] = ValueOps<Value>::multiply(a.val[p], b.val[k]);
        }
    }
}

// 分 tile 扫描相同列的连续段，last_col/runs 处理跨 tile 的重复列。
// 段首先赋值，其他成员再原子累加；同步防止赋值覆盖累加结果。
template<class Value, int BLOCK = kMergeBlock>
__global__ void merge_rows(int num, const int* rows, const U64* offsets,
                           const int* col, const Value* val, RowOutput<Value> out) {
    const int q = blockIdx.x;
    if (q >= num) return;
    const int r = rows[q], tid = threadIdx.x;
    const U64 base = offsets[q], length = offsets[q + 1] - base, dest = out.offsets[r];
    __shared__ int tile_col[BLOCK], scan[BLOCK], last_col, runs;
    if (tid == 0) { last_col = INT_MAX; runs = 0; }
    __syncthreads();
    for (U64 tile = 0; tile < length; tile += BLOCK) {
        const U64 index = tile + tid;
        const bool valid = index < length;
        const int key = valid ? col[base + index] : INT_MAX;
        const Value value = valid ? val[base + index] : ValueOps<Value>::zero();
        tile_col[tid] = key;
        __syncthreads();
        const bool start = valid && (key != (tid == 0 ? last_col : tile_col[tid - 1]));
        scan[tid] = start ? 1 : 0;
        __syncthreads();
        for (int step = 1; step < BLOCK; step *= 2) {
            const int add = tid >= step ? scan[tid - step] : 0;
            __syncthreads();
            scan[tid] += add;
            __syncthreads();
        }
        const int exclusive = scan[tid] - int(start);
        if (start) { out.col[dest + runs + exclusive] = key; out.val[dest + runs + exclusive] = value; }
        __syncthreads();
        if (valid && !start) ValueOps<Value>::atomic_add(out.val + dest + runs + exclusive - 1, value);
        __syncthreads();
        if (tid == BLOCK - 1) {
            const int last = int((length - tile < BLOCK ? length - tile : BLOCK) - 1);
            last_col = tile_col[last]; runs += scan[BLOCK - 1];
        }
        __syncthreads();
    }
    if (tid == 0) out.counts[r] = runs;
}

__global__ void widen_counts(int n, const int* counts, U64* wide) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i <= size_t(n)) wide[i] = U64(counts[i]);
}

template<class Value>
__global__ void compact(int n, RowOutput<Value> temp, const U64* exact, int* row, int* col, Value* val) {
    const int r = blockIdx.x * (blockDim.x / kWarp) + threadIdx.x / kWarp;
    const int lane = threadIdx.x % kWarp;
    if (r >= n) return;
    const U64 first = exact[r], end = exact[r + 1];
    for (U64 p = first + lane; p < end; p += kWarp) {
        const U64 src = temp.offsets[r] + p - first;
        col[p] = temp.col[src]; val[p] = temp.val[src];
    }
    if (lane == 0) { row[r] = int(first); if (r == n - 1) row[n] = int(end); }
}

template<class Value>
class Custom final : public Backend {
    DeviceCSR<Value> a, b;
    Buffer<U64> upper, raw_count, temp_offsets, wide_counts, exact_offsets;
    Buffer<U64> large_counts, large_offsets, batch_offsets;
    Buffer<int> bins, bin_sizes, row_counts, temp_col, expand_col, sort_col, out_row, out_col;
    Buffer<Value> temp_val, expand_val, sort_val, out_val;
    Buffer<unsigned char> scratch;
    int nnz = 0;

    RowOutput<Value> temporary() const { return {temp_offsets.p, temp_col.p, temp_val.p, row_counts.p}; }
    const int* bin_rows(int bin) const { return bins.p + size_t(bin) * a.rows; }

    void scan(const U64* input, U64* output, int count) {
        size_t bytes = 0;
        check(cub::DeviceScan::ExclusiveSum(nullptr, bytes, input, output, count));
        scratch.reserve(bytes);
        check(cub::DeviceScan::ExclusiveSum(scratch.p, bytes, input, output, count));
    }

    std::array<int, kBins> prepare_rows() {
        const int n = a.rows;
        const size_t size = size_t(n) + 1;
        upper.reserve(size); raw_count.reserve(size); temp_offsets.reserve(size);
        row_counts.reserve(size); wide_counts.reserve(size); exact_offsets.reserve(size);
        bins.reserve(size_t(n) * kBins); bin_sizes.reserve(kBins);
        check(cudaMemset(bin_sizes.p, 0, kBins * sizeof(int)));
        classify<Value><<<blocks(n, kBlock / kWarp), kBlock>>>(
            n, b.cols, a.view(), b.view(), upper.p, raw_count.p, bins.p, bin_sizes.p, row_counts.p);
        check(cudaGetLastError());
        scan(upper.p, temp_offsets.p, n + 1);
        U64 total;
        std::array<int, kBins> sizes{};
        check(cudaMemcpy(&total, temp_offsets.p + n, sizeof(total), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(sizes.data(), bin_sizes.p, sizeof(int) * kBins, cudaMemcpyDeviceToHost));
        temp_col.reserve(total); temp_val.reserve(total);
        return sizes;
    }

    void hash_rows(const std::array<int, kBins>& sizes) {
        const CSRView<Value> av = a.view(), bv = b.view();
        const RowOutput<Value> out = temporary();
        if (sizes[0]) small_rows<Value, 64, 8><<<blocks(sizes[0], 16), kBlock>>>(sizes[0], bin_rows(0), av, bv, out);
        if (sizes[1]) small_rows<Value, 256, 32><<<blocks(sizes[1], 4), kBlock>>>(sizes[1], bin_rows(1), av, bv, out);
        if (sizes[2]) medium_rows<Value, 1024><<<sizes[2], kBlock>>>(sizes[2], bin_rows(2), av, bv, out);
        if (sizes[3]) medium_rows<Value, 2048><<<sizes[3], kBlock>>>(sizes[3], bin_rows(3), av, bv, out);
        check(cudaGetLastError());
    }

    void sort_batch(int count, int tuples, const int* rows) {
        expand_rows<Value><<<count, kBlock>>>(count, rows, a.view(), b.view(), batch_offsets.p, expand_col.p, expand_val.p);
        check(cudaGetLastError());
        // CUB 指针接口要求输入/输出区间不重叠。
        size_t bytes = 0;
        check(cub::DeviceSegmentedRadixSort::SortPairs(nullptr, bytes,
            expand_col.p, sort_col.p, expand_val.p, sort_val.p,
            tuples, count, batch_offsets.p, batch_offsets.p + 1, 0, 32));
        scratch.reserve(bytes);
        check(cub::DeviceSegmentedRadixSort::SortPairs(scratch.p, bytes,
            expand_col.p, sort_col.p, expand_val.p, sort_val.p,
            tuples, count, batch_offsets.p, batch_offsets.p + 1, 0, 32));
        merge_rows<Value><<<count, kMergeBlock>>>(count, rows, batch_offsets.p, sort_col.p, sort_val.p, temporary());
        check(cudaGetLastError());
    }

    void large_rows(int count) {
        const int* rows = bin_rows(4);
        large_counts.reserve(size_t(count) + 1); large_offsets.reserve(size_t(count) + 1);
        batch_offsets.reserve(size_t(count) + 1);
        gather_counts<<<blocks(size_t(count) + 1, 256), 256>>>(count, rows, raw_count.p, large_counts.p);
        check(cudaGetLastError());
        scan(large_counts.p, large_offsets.p, count + 1);
        std::vector<U64> offsets(size_t(count) + 1);
        check(cudaMemcpy(offsets.data(), large_offsets.p, offsets.size() * sizeof(U64), cudaMemcpyDeviceToHost));

        // 两套 (column,value) 数组共用当前空闲显存的 1/4，CUB scratch 另计。
        // 复用已有容量，分配量不超过实际展开总量。
        size_t free_bytes = 0, total_bytes = 0;
        check(cudaMemGetInfo(&free_bytes, &total_bytes));
        const U64 budget = free_bytes / 4 / (2 * (sizeof(int) + sizeof(Value)));
        const U64 cached = std::min({expand_col.capacity, expand_val.capacity, sort_col.capacity, sort_val.capacity});
        const U64 cap = std::min({offsets.back(), std::max(budget, cached), U64(0x40000000u)});
        for (int i = 0; i < count; ++i)
            if (offsets[size_t(i) + 1] - offsets[i] > cap)
                throw std::runtime_error("single large-row expansion exceeds batch workspace or sort item limit");
        expand_col.reserve(cap); expand_val.reserve(cap); sort_col.reserve(cap); sort_val.reserve(cap);
        for (int first = 0; first < count;) {
            int end = first;
            while (end < count && offsets[size_t(end) + 1] - offsets[first] <= cap) ++end;
            const int batch = end - first;
            relative_offsets<<<blocks(size_t(batch) + 1, 256), 256>>>(batch, large_offsets.p, first, batch_offsets.p);
            check(cudaGetLastError());
            sort_batch(batch, int(offsets[end] - offsets[first]), rows + first);
            first = end;
        }
    }

    void finish_product() {
        const int n = a.rows;
        widen_counts<<<blocks(size_t(n) + 1, 256), 256>>>(n, row_counts.p, wide_counts.p);
        check(cudaGetLastError());
        scan(wide_counts.p, exact_offsets.p, n + 1);
        U64 total;
        check(cudaMemcpy(&total, exact_offsets.p + n, sizeof(total), cudaMemcpyDeviceToHost));
        if (total > INT_MAX) throw std::overflow_error("product nnz exceeds int32 CSR range");
        nnz = int(total); out_col.reserve(nnz); out_val.reserve(nnz);
        compact<Value><<<blocks(n, kBlock / kWarp), kBlock>>>(n, temporary(), exact_offsets.p, out_row.p, out_col.p, out_val.p);
        check(cudaGetLastError());
    }

public:
    Custom(const HostCSR& aa, const HostCSR& bb) : a(aa), b(bb) {
        if (a.cols != b.rows) throw std::runtime_error("A.cols must equal B.rows");
    }

    void multiply() override {
        if (a.rows == INT_MAX) throw std::overflow_error("row count plus sentinel exceeds scan index limit");
        out_row.reserve(size_t(a.rows) + 1);
        if (!a.rows) { check(cudaMemset(out_row.p, 0, sizeof(int))); nnz = 0; return; }
        const auto sizes = prepare_rows();
        hash_rows(sizes);
        if (sizes[4]) large_rows(sizes[4]);
        finish_product();
    }

    HostCSR download() override {
        HostCSR h;
        h.rows = a.rows; h.cols = b.cols; h.complex = !std::is_same<Value, double>::value;
        h.row.resize(size_t(h.rows) + 1); h.col.resize(nnz); h.val.resize(nnz);
        check(cudaMemcpy(h.row.data(), out_row.p, h.row.size() * sizeof(int), cudaMemcpyDeviceToHost));
        if (nnz) {
            check(cudaMemcpy(h.col.data(), out_col.p, size_t(nnz) * sizeof(int), cudaMemcpyDeviceToHost));
            if constexpr (std::is_same<Value, double>::value) {
                check(cudaMemcpy(h.val.data(), out_val.p, size_t(nnz) * sizeof(double), cudaMemcpyDeviceToHost));
            } else {
                std::vector<ComplexValue> values(nnz);
                check(cudaMemcpy(values.data(), out_val.p, size_t(nnz) * sizeof(ComplexValue), cudaMemcpyDeviceToHost));
                h.imag.resize(nnz);
                for (int i = 0; i < nnz; ++i) { h.val[i] = values[i].re; h.imag[i] = values[i].im; }
            }
        }
        return h;
    }
};
} // namespace

std::unique_ptr<Backend> make_backend(const HostCSR& a, const HostCSR& b) {
    if (a.complex || b.complex) return std::make_unique<Custom<ComplexValue>>(a, b);
    return std::make_unique<Custom<double>>(a, b);
}
const char* backend_name() { return "custom"; }
} // namespace spgemm
