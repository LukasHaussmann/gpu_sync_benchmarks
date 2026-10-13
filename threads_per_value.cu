// Threads per value: aggregating uint32 values with atomicAdd vs. a spinlock,
// shared vs. global memory, three ways of mapping threads to values.
//
// The whole GPU runs (blocks of 1024 threads, as many as are resident at once).
// Every thread repeatedly adds a value in [1, 8] to one value of a table; k
// threads share each value, from k = 1 (every thread has its own value) to
// k = all threads (one value). One op is
//   atomic:  atomicAdd(&value, v)                  on a packed uint32 array
//   lock:    acquire; value += v; release          on an array of {lock, value}
//            (test-and-test-and-set spinlock without backoff)
// Atomics and fences use the narrowest scope that contains all sharers: block
// scope for k <= 1024, device scope beyond.
//
// Fair comparison of the two memories: for k <= 1024 the k threads of a value
// are always in the same block, in both memories (a shared-memory table per
// block; in global memory, each block owns its own slice of one table). So the
// contenders are identical and only the memory differs. For k > 1024 a value is
// shared by threads of several blocks, which only global memory can do.
//
// Global layout: for k <= 1024 each block's slice starts on its own 128-byte
// cache line (values within it packed), so blocks never interfere in the L2,
// just as shared memory is private to the SM. For k > 1024 the GPU's values
// form one plain packed array, in all layouts.
//
// Which threads share a value matters, because lanes of one warp that hit the
// same address are handled within one instruction. Three layouts:
//   neighbors  consecutive threads share a value     (value = thread / k)
//   spread     sharers are as far apart as possible (value = thread % values)
//              -> the 32 lanes of a warp hit different values while k <= values
//   random     exactly k threads per value as well, but which threads share a
//              value is random: the thread-to-value assignment is a random
//              shuffle (fixed per run; each thread always uses its value)
// Up to k = 1024 all three work within the block's own table, beyond that on
// one table for the GPU.
//
// One source for NVIDIA (CUDA) and AMD (HIP) GPUs:
//   NVIDIA:  make / make test / make run              (nvcc)
//   AMD:     make hip / make test-hip / make run-hip  (hipcc)
// The only platform-specific code is the portability block below and the
// scoped atomic primitives.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Portability: the code is written against the CUDA runtime API; with hipcc
// these names map to their HIP equivalents.
// ---------------------------------------------------------------------------

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define PLATFORM "HIP"
#define cudaDeviceProp hipDeviceProp_t
#define cudaError_t hipError_t
#define cudaEventCreate hipEventCreate
#define cudaEventDestroy hipEventDestroy
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEvent_t hipEvent_t
#define cudaFree hipFree
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaMalloc hipMalloc
#define cudaMemcpy hipMemcpy
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemset hipMemset
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaSuccess hipSuccess
#else
#define PLATFORM "CUDA"
#endif

// ---------------------------------------------------------------------------
// Host helpers: error checking, kernel timing, calibrate-then-measure
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            std::fprintf(stderr, PLATFORM " error %s at %s:%d: %s\n", #call, __FILE__, \
                         __LINE__, cudaGetErrorString(err_));                     \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

constexpr uint32_t kMaxOpsPerThread = 1u << 20;

// Times one kernel launch (or any sequence of stream work) with GPU events.
struct Timer {
    cudaEvent_t start, stop;
    Timer() { CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop)); }
    ~Timer() { (void)cudaEventDestroy(start); (void)cudaEventDestroy(stop); }

    template <class F>
    float time_ms(F&& launch) {
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

inline float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct TimingConfig {
    int reps = 10;               // timed runs per point (median reported)
    double target_ms = 50;       // calibrate ops/thread so one run takes >= this
    uint32_t min_ops = 32;       // ... but never fewer ops per thread than this
    double max_seconds = 3;      // time budget per point; caps reps for slow points
};

// Calibrates ops/thread so that one run lasts at least `target_ms` (from
// `min_ops` up to kMaxOpsPerThread), then times `reps` runs at that size.
// The minimum keeps every thread in a steady state of repeated ops: with a
// single op per thread, lock results would be dominated by the start-up herd.
// `run(ops)` performs one complete, timed run (including untimed output reset).
template <class Run>
void calibrate_and_time(const TimingConfig& cfg, Run run, uint32_t& ops, std::vector<float>& times) {
    ops = cfg.min_ops;
    float ms = run(ops);
    while (ms < cfg.target_ms && ops < kMaxOpsPerThread) {
        const double scale = std::clamp(1.2 * cfg.target_ms / std::max(ms, 1e-3f), 2.0, 64.0);
        ops = (uint32_t)std::min<double>(kMaxOpsPerThread, std::ceil(ops * scale));
        ms = run(ops);
    }
    // Respect the time budget; a single run longer than the budget is reported as is.
    const double budget_ms = cfg.max_seconds * 1e3;
    times.clear();
    if (ms > budget_ms) {
        times.push_back(ms);
        return;
    }
    const int reps = std::min(cfg.reps, std::max(1, (int)(budget_ms / ms)));
    for (int r = 0; r < reps; ++r) times.push_back(run(ops));
}

// ---------------------------------------------------------------------------
// Device helpers: scoped atomics and fences, per-thread op stream
// ---------------------------------------------------------------------------

// Synchronization scope: the narrowest scope that contains all contenders.
// Block = CUDA block scope = HIP workgroup scope; Device = CUDA device scope =
// HIP agent scope. All atomics are relaxed; the lock orders its critical
// section with fences.
enum class Scope { Block, Device };

#if defined(__HIP_PLATFORM_AMD__)
template <Scope S>
constexpr int kHipScope = S == Scope::Block ? __HIP_MEMORY_SCOPE_WORKGROUP : __HIP_MEMORY_SCOPE_AGENT;

template <Scope S> __device__ __forceinline__ uint32_t atomic_add(uint32_t* p, uint32_t v) {
    return __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED, kHipScope<S>);
}
template <Scope S> __device__ __forceinline__ int atomic_exch(int* p, int v) {
    return __hip_atomic_exchange(p, v, __ATOMIC_RELAXED, kHipScope<S>);
}
template <Scope S> __device__ __forceinline__ int load_relaxed(int* p) {
    return __hip_atomic_load(p, __ATOMIC_RELAXED, kHipScope<S>);
}
#else
template <Scope S> __device__ __forceinline__ uint32_t atomic_add(uint32_t* p, uint32_t v) {
    if constexpr (S == Scope::Block) return atomicAdd_block(p, v); else return atomicAdd(p, v);
}
template <Scope S> __device__ __forceinline__ int atomic_exch(int* p, int v) {
    if constexpr (S == Scope::Block) return atomicExch_block(p, v); else return atomicExch(p, v);
}
template <Scope S> __device__ __forceinline__ int load_relaxed(int* p) {
    return *(volatile int*)p;  // never a stale register or L1 copy
}
#endif
template <Scope S> __device__ __forceinline__ void fence() {
    if constexpr (S == Scope::Block) __threadfence_block(); else __threadfence();
}

// Every thread draws its add values from its own xorshift32 generator, in
// registers, so no input is read from memory; the host replays the same streams.
__host__ __device__ inline uint32_t hash32(uint32_t x) {  // murmur3 finalizer
    x ^= x >> 16; x *= 0x85ebca6bu;
    x ^= x >> 13; x *= 0xc2b2ae35u;
    x ^= x >> 16;
    return x;
}

struct OpStream {
    uint32_t state;

    __host__ __device__ OpStream(uint32_t seed, uint32_t thread)
        : state(hash32(thread ^ hash32(seed)) | 1u) {}

    // Advances the stream; returns a slot in [0, num_slots) and a value in [1, 8].
    __host__ __device__ void next(uint32_t num_slots, uint32_t& slot, uint32_t& value) {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        slot = (uint32_t)(((uint64_t)state * num_slots) >> 32);  // uniform, no modulo
        value = 1u + (state & 7u);
    }
};

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

constexpr int kBlock = 1024;

struct LockedValue {
    int lock;
    uint32_t value;
};

// The two mechanisms. Elem is one table element, `value_word` the index of the
// value within the element, counted in uint32 words.
template <Scope S>
struct Atomic32 {
    using Elem = uint32_t;
    static constexpr int value_word = 0;
    __device__ static void add(Elem* e, uint32_t v) { atomic_add<S>(e, v); }
};

template <Scope S>
struct Lock32 {
    using Elem = LockedValue;
    static constexpr int value_word = 1;
    // Test-and-test-and-set spinlock without backoff.
    __device__ static void add(Elem* e, uint32_t v) {
#if defined(__HIP_PLATFORM_AMD__)
        // AMD: lanes of a wavefront run in lock-step. With "spin until acquired,
        // then work" (below), the winning lane would wait at the end of the spin
        // loop for wave-mates that wait for its lock -> deadlock. So this is a
        // try-lock loop: a lane that gets the lock runs the critical section and
        // releases the lock within the same loop iteration.
        for (bool done = false; !done;) {
            if (load_relaxed<S>(&e->lock) == 0 && atomic_exch<S>(&e->lock, 1) == 0) {
                critical_section(e, v);
                done = true;
            }
        }
#else
        // NVIDIA (sm_70+): independent thread scheduling lets the lock holder
        // progress while its warp-mates spin, so the usual loop is safe.
        while (load_relaxed<S>(&e->lock) != 0 || atomic_exch<S>(&e->lock, 1) != 0) {}
        critical_section(e, v);
#endif
    }

    __device__ static void critical_section(Elem* e, uint32_t v) {
        fence<S>();  // acquire
        volatile uint32_t* value = &e->value;
        *value = *value + v;
        fence<S>();  // release
        atomic_exch<S>(&e->lock, 0);
    }
};

template <class Elem>
constexpr uint32_t kLineElems = 128 / sizeof(Elem);  // elements per cache line

// Distance between two blocks' slices of the global table: whole cache lines.
template <class Elem>
__host__ __device__ inline uint32_t block_stride(uint32_t per_block) {
    return (per_block + kLineElems<Elem> - 1) / kLineElems<Elem> * kLineElems<Elem>;
}

enum class Mapping { Neighbors, Spread, Random };

// A thread works on one region of the global table: its block's slice for
// k <= kBlock, else the table for the whole GPU.
template <class Elem>
__host__ __device__ inline uint32_t region_base(uint32_t k, uint32_t gtid) {
    return k <= kBlock ? gtid / kBlock * block_stride<Elem>(kBlock / k) : 0;
}

__host__ __device__ inline uint32_t region_values(uint32_t k, uint32_t threads) {
    return k <= kBlock ? kBlock / k : threads / k;
}

// A thread's element, relative to its region. `assign` is the random layout's
// thread-to-value assignment (unused by the others).
template <class Elem>
__host__ __device__ inline uint32_t region_offset(Mapping m, uint32_t k, uint32_t threads, uint32_t gtid,
                                                  const uint32_t* assign) {
    if (m == Mapping::Random) return assign[gtid];
    const bool spread = m == Mapping::Spread;
    if (k <= kBlock) {
        const uint32_t tid = gtid % kBlock;
        return spread ? tid % (kBlock / k) : tid / k;
    }
    return spread ? gtid % (threads / k) : gtid / k;
}

// One thread's ops, all on its one element. `k` is a kernel argument, so the
// compiler cannot prove that a warp's lanes hit the same address and does not
// merge their atomics (no REDUX).
template <class Op, Mapping kMap>
__device__ __forceinline__ void add_ops(typename Op::Elem* region, uint32_t k, const uint32_t* assign,
                                        uint32_t ops, uint32_t seed, uint32_t gtid) {
    typename Op::Elem* e = &region[region_offset<typename Op::Elem>(kMap, k, gridDim.x * blockDim.x, gtid, assign)];
    OpStream stream(seed, gtid);
    for (uint32_t i = 0; i < ops; ++i) {
        uint32_t slot, v;
        stream.next(1, slot, v);
        Op::add(e, v);
    }
}

template <template <Scope> class M, bool kShared, Mapping kMap>
__global__ void threads_per_value_kernel(typename M<Scope::Block>::Elem* table, uint32_t k, const uint32_t* assign,
                                         uint32_t ops, uint32_t seed) {
    using Elem = typename M<Scope::Block>::Elem;
    const uint32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if (k > blockDim.x) {  // sharers in several blocks: one table for the GPU (global memory only)
        if constexpr (!kShared) add_ops<M<Scope::Device>, kMap>(table, k, assign, ops, seed, gtid);
        return;
    }
    // Sharers within one block: a table per block (in global memory, the block's slice).
    Elem* block_table = table + region_base<Elem>(k, gtid);
    if constexpr (kShared) {
        extern __shared__ uint64_t smem_words[];  // 8-byte aligned
        Elem* smem = reinterpret_cast<Elem*>(smem_words);
        const uint32_t per_block = blockDim.x / k;
        for (uint32_t i = threadIdx.x; i < per_block; i += blockDim.x) smem[i] = Elem{};
        __syncthreads();
        add_ops<M<Scope::Block>, kMap>(smem, k, assign, ops, seed, gtid);
        __syncthreads();  // write the table out (a few KB, negligible next to the ops)
        for (uint32_t i = threadIdx.x; i < per_block; i += blockDim.x) block_table[i] = smem[i];
    } else {
        add_ops<M<Scope::Block>, kMap>(block_table, k, assign, ops, seed, gtid);
    }
}

// Sum of all values every thread adds (mod 2^32, like the table).
__global__ void expected_total_kernel(uint32_t ops, uint32_t seed, uint32_t* out) {
    OpStream stream(seed, blockIdx.x * blockDim.x + threadIdx.x);
    uint32_t sum = 0;
    for (uint32_t i = 0; i < ops; ++i) {
        uint32_t slot, v;
        stream.next(1, slot, v);
        sum += v;
    }
    atomicAdd(out, sum);
}

// Sum of the value words of n elements of `words_per_elem` uint32 words each.
__global__ void table_total_kernel(const uint32_t* table, uint32_t n, uint32_t words_per_elem, uint32_t value_word,
                                   uint32_t* out) {
    uint32_t sum = 0;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        sum += table[i * words_per_elem + value_word];
    atomicAdd(out, sum);
}

// ---------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------

struct Variant {
    std::string mechanism, mapping, memory;
    uint32_t words_per_elem, value_word;  // element layout, in uint32 words
    void (*launch)(int grid, void* table, uint32_t k, const uint32_t* assign, uint32_t ops, uint32_t seed);
    const void* kernel;  // for the occupancy query
};

template <template <Scope> class M, bool kShared, Mapping kMap>
Variant variant(const char* mechanism, const char* mapping, const char* memory) {
    using Elem = typename M<Scope::Block>::Elem;
    return {mechanism, mapping, memory, sizeof(Elem) / sizeof(uint32_t), M<Scope::Block>::value_word,
            [](int grid, void* table, uint32_t k, const uint32_t* assign, uint32_t ops, uint32_t seed) {
                const size_t smem = kShared ? (kBlock / k) * sizeof(Elem) : 0;
                threads_per_value_kernel<M, kShared, kMap>
                    <<<grid, kBlock, smem>>>(static_cast<Elem*>(table), k, assign, ops, seed);
            },
            reinterpret_cast<const void*>(&threads_per_value_kernel<M, kShared, kMap>)};
}

template <template <Scope> class M>
static void add_variants(std::vector<Variant>& out, const char* mechanism) {
    out.push_back(variant<M, true, Mapping::Neighbors>(mechanism, "neighbors", "shared"));
    out.push_back(variant<M, false, Mapping::Neighbors>(mechanism, "neighbors", "global"));
    out.push_back(variant<M, true, Mapping::Spread>(mechanism, "spread", "shared"));
    out.push_back(variant<M, false, Mapping::Spread>(mechanism, "spread", "global"));
    out.push_back(variant<M, true, Mapping::Random>(mechanism, "random", "shared"));
    out.push_back(variant<M, false, Mapping::Random>(mechanism, "random", "global"));
}

static std::vector<Variant> variants() {
    std::vector<Variant> out;
    add_variants<Atomic32>(out, "atomic");
    add_variants<Lock32>(out, "lock");
    return out;
}

// k = 1, 2, 4, ... (dividing the thread count), then k = all threads.
static std::vector<uint32_t> threads_per_value(uint32_t threads) {
    std::vector<uint32_t> ks;
    for (uint32_t k = 1; k < threads && threads % k == 0; k *= 2) ks.push_back(k);
    ks.push_back(threads);
    return ks;
}

// The random layout's thread-to-value assignment: every value of a region
// exactly k times, shuffled (within each block for k <= kBlock, else over the GPU).
static std::vector<uint32_t> random_assignment(uint32_t k, uint32_t threads, uint32_t seed) {
    std::vector<uint32_t> assign(threads);
    std::mt19937 rng(seed * 1000003u + k);
    const uint32_t region = k <= (uint32_t)kBlock ? kBlock : threads;
    for (uint32_t start = 0; start < threads; start += region) {
        for (uint32_t i = 0; i < region; ++i) assign[start + i] = i / k;
        std::shuffle(assign.begin() + start, assign.begin() + start + region, rng);
    }
    return assign;
}

// Host replay of the op streams: the exact expected value of every element
// (mod 2^32), indexed like the table; padding stays 0.
static std::vector<uint32_t> expected_values(const Variant& v, int grid, uint32_t k, const uint32_t* assign,
                                             uint32_t ops, uint32_t seed) {
    const uint32_t threads = grid * kBlock;
    const Mapping m = v.mapping == "neighbors" ? Mapping::Neighbors
                    : v.mapping == "spread"    ? Mapping::Spread
                                               : Mapping::Random;
    const bool pair = v.words_per_elem == 2;
    std::vector<uint32_t> values(threads, 0);
    for (uint32_t t = 0; t < threads; ++t) {
        const uint32_t base = pair ? region_base<LockedValue>(k, t) : region_base<uint32_t>(k, t);
        const uint32_t offset = pair ? region_offset<LockedValue>(m, k, threads, t, assign)
                                     : region_offset<uint32_t>(m, k, threads, t, assign);
        OpStream stream(seed, t);
        for (uint32_t i = 0; i < ops; ++i) {
            uint32_t slot, value;
            stream.next(1, slot, value);
            values[base + offset] += value;
        }
    }
    return values;
}

int main(int argc, char** argv) {
    bool test = false;
    std::string out_dir = "results";
    TimingConfig timing;
    timing.reps = 5;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--test") test = true;
        else if (a == "--out" && i + 1 < argc) out_dir = argv[++i];
        else if (a == "--reps" && i + 1 < argc) timing.reps = std::max(1, std::stoi(argv[++i]));
        else if (a == "--min-ops" && i + 1 < argc) timing.min_ops = std::max(1, std::stoi(argv[++i]));
        else {
            std::printf("Usage: %s [--out DIR] [--reps R] [--min-ops N] [--test]\n", argv[0]);
            return a == "--help" || a == "-h" ? EXIT_SUCCESS : EXIT_FAILURE;
        }
    }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    // One full wave: as many 1024-thread blocks per SM (CU on AMD) as every
    // variant can keep resident at once.
    int blocks_per_sm = prop.maxThreadsPerMultiProcessor / kBlock;
    for (const Variant& v : variants()) {
        int n = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, v.kernel, kBlock, kBlock * sizeof(LockedValue)));
        blocks_per_sm = std::min(blocks_per_sm, n);
    }
    if (blocks_per_sm < 1) {
        std::fprintf(stderr, "A %d-thread block does not fit on this GPU.\n", kBlock);
        return EXIT_FAILURE;
    }
    const int grid = prop.multiProcessorCount * blocks_per_sm;
    const uint32_t threads = (uint32_t)grid * kBlock;
    std::printf("Threads per value (%s): uint32 atomicAdd vs. spinlock, shared vs. global\n"
                "Device: %s, warp size %d, %d SMs x %d blocks x %d threads = %u threads\n",
                PLATFORM, prop.name, prop.warpSize, prop.multiProcessorCount, blocks_per_sm, kBlock, threads);

    // Big enough for every layout: at most one element per thread, padding included.
    const size_t table_bytes = threads * sizeof(LockedValue);
    uint32_t *table, *d_total, *d_assign;
    CUDA_CHECK(cudaMalloc(&table, table_bytes));
    CUDA_CHECK(cudaMalloc(&d_assign, threads * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_total, sizeof(uint32_t)));
    auto table_total = [&](const Variant& v) {
        CUDA_CHECK(cudaMemset(d_total, 0, sizeof(uint32_t)));
        table_total_kernel<<<256, 256>>>(table, threads, v.words_per_elem, v.value_word, d_total);
        uint32_t h;
        CUDA_CHECK(cudaMemcpy(&h, d_total, sizeof(h), cudaMemcpyDeviceToHost));
        return h;
    };
    auto expected_total = [&](uint32_t ops, uint32_t seed) {
        CUDA_CHECK(cudaMemset(d_total, 0, sizeof(uint32_t)));
        expected_total_kernel<<<grid, kBlock>>>(ops, seed, d_total);
        uint32_t h;
        CUDA_CHECK(cudaMemcpy(&h, d_total, sizeof(h), cudaMemcpyDeviceToHost));
        return h;
    };

    FILE* csv = nullptr;
    if (!test) {
        csv = std::fopen((out_dir + "/threads_per_value.csv").c_str(), "w");
        if (!csv) { std::perror(out_dir.c_str()); return EXIT_FAILURE; }
        std::fprintf(csv, "mechanism,mapping,memory,threads_per_value,values,blocks,threads_per_block,threads,ops_per_thread,"
                          "reps,median_ms,mops_per_s,ns_per_op_per_thread,verified\n");
        if (FILE* dev = std::fopen((out_dir + "/device.txt").c_str(), "w")) {  // for the plot
            std::fprintf(dev, "%s\nwarp_size=%d\n", prop.name, prop.warpSize);
            std::fclose(dev);
        }
    }

    const uint32_t seed = 42;
    int checks = 0, failures = 0;
    Timer timer;
    for (const Variant& v : variants()) {
        for (uint32_t k : threads_per_value(threads)) {
            if (v.memory == "shared" && k > (uint32_t)kBlock) continue;  // shared memory is per block
            const uint32_t values = threads / k;
            const std::vector<uint32_t> assign = random_assignment(k, threads, seed);  // used by "random" only
            CUDA_CHECK(cudaMemcpy(d_assign, assign.data(), threads * sizeof(uint32_t), cudaMemcpyHostToDevice));
            if (test) {  // exact, value by value; every lock released
                for (uint32_t ops : {1u, 7u}) {
                    CUDA_CHECK(cudaMemset(table, 0, table_bytes));
                    v.launch(grid, table, k, d_assign, ops, seed);
                    CUDA_CHECK(cudaGetLastError());
                    std::vector<uint32_t> words(threads * v.words_per_elem);
                    CUDA_CHECK(cudaMemcpy(words.data(), table, words.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
                    const std::vector<uint32_t> want = expected_values(v, grid, k, assign.data(), ops, seed);
                    bool ok = true;
                    for (uint32_t i = 0; i < threads; ++i) {
                        ok &= words[i * v.words_per_elem + v.value_word] == want[i];
                        if (v.words_per_elem == 2) ok &= words[i * 2] == 0;  // lock word
                    }
                    ++checks;
                    if (!ok) {
                        ++failures;
                        std::printf("  FAIL %s %s %s k=%u ops=%u\n", v.mechanism.c_str(), v.mapping.c_str(),
                                    v.memory.c_str(), k, ops);
                    }
                }
                continue;
            }
            auto run_once = [&](uint32_t ops) {
                CUDA_CHECK(cudaMemset(table, 0, table_bytes));
                return timer.time_ms([&] { v.launch(grid, table, k, d_assign, ops, seed); });
            };
            uint32_t ops;
            std::vector<float> times;
            calibrate_and_time(timing, run_once, ops, times);
            const float ms = median(times);
            const bool ok = table_total(v) == expected_total(ops, seed);
            if (!ok) ++failures;
            const double mops = threads * (double)ops / (ms * 1e-3) / 1e6;
            std::printf("%-6s %-9s %-6s %6u threads/value %6u values %8u ops/thread %2zu reps %10.1f Mops/s %9.2f ns/op/thread  %s\n",
                        v.mechanism.c_str(), v.mapping.c_str(), v.memory.c_str(), k, values, ops, times.size(), mops, ms * 1e6 / ops,
                        ok ? "OK" : "FAILED");
            std::fflush(stdout);
            std::fprintf(csv, "%s,%s,%s,%u,%u,%d,%d,%u,%u,%zu,%.6f,%.6f,%.4f,%d\n", v.mechanism.c_str(),
                         v.mapping.c_str(), v.memory.c_str(),
                         k, values, grid, kBlock, threads, ops, times.size(), ms, mops, ms * 1e6 / ops, ok ? 1 : 0);
        }
    }

    if (test) {
        std::printf("%d checks, %d failures. %s\n", checks, failures, failures ? "TESTS FAILED." : "All tests passed.");
    } else {
        std::fclose(csv);
        std::printf("Wrote %s/threads_per_value.csv%s\n", out_dir.c_str(), failures ? " (WITH FAILURES)" : "");
    }
    CUDA_CHECK(cudaFree(table));
    CUDA_CHECK(cudaFree(d_total));
    CUDA_CHECK(cudaFree(d_assign));
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
