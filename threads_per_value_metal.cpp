// Threads per value, Metal port: the host side of threads_per_value.cu on Apple
// GPUs, written against metal-cpp. The kernels are in threads_per_value.metal
// (compiled to threads_per_value.metallib, loaded from next to the binary).
//
// Same benchmark, same output: the layouts, mechanisms, op streams, timing
// (calibrate-then-measure), verification and CSV are those of
// threads_per_value.cu; the host logic below is a copy of it with the CUDA
// runtime calls replaced by Metal ones. What differs is in the README
// ("Apple vs. NVIDIA / AMD").
//
//   make metal / make test-metal / make run-metal
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <mach-o/dyld.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define PLATFORM "Metal"

// ---------------------------------------------------------------------------
// Host helpers: error checking, kernel timing, calibrate-then-measure
// ---------------------------------------------------------------------------

[[noreturn]] static void fail(const char* what, NS::Error* err = nullptr) {
    std::fprintf(stderr, PLATFORM " error: %s%s%s\n", what, err ? ": " : "",
                 err ? err->localizedDescription()->utf8String() : "");
    std::exit(EXIT_FAILURE);
}

constexpr uint32_t kMaxOpsPerThread = 1u << 20;

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
// Op stream and layouts: identical to the device code (threads_per_value.metal)
// ---------------------------------------------------------------------------

inline uint32_t hash32(uint32_t x) {  // murmur3 finalizer
    x ^= x >> 16; x *= 0x85ebca6bu;
    x ^= x >> 13; x *= 0xc2b2ae35u;
    x ^= x >> 16;
    return x;
}

struct OpStream {
    uint32_t state;

    OpStream(uint32_t seed, uint32_t thread) : state(hash32(thread ^ hash32(seed)) | 1u) {}

    // Advances the stream; returns a slot in [0, num_slots) and a value in [1, 8].
    void next(uint32_t num_slots, uint32_t& slot, uint32_t& value) {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        slot = (uint32_t)(((uint64_t)state * num_slots) >> 32);  // uniform, no modulo
        value = 1u + (state & 7u);
    }
};

constexpr int kBlock = 1024;

struct LockedValue {
    int lock;
    uint32_t value;
};

template <class Elem>
constexpr uint32_t kLineElems = 128 / sizeof(Elem);  // elements per cache line

// Distance between two blocks' slices of the global table: whole cache lines.
template <class Elem>
inline uint32_t block_stride(uint32_t per_block) {
    return (per_block + kLineElems<Elem> - 1) / kLineElems<Elem> * kLineElems<Elem>;
}

enum class Mapping { Neighbors, Spread, Random };

// A thread works on one region of the global table: its block's slice for
// k <= kBlock, else the table for the whole GPU.
template <class Elem>
inline uint32_t region_base(uint32_t k, uint32_t gtid) {
    return k <= kBlock ? gtid / kBlock * block_stride<Elem>(kBlock / k) : 0;
}

// A thread's element, relative to its region. `assign` is the random layout's
// thread-to-value assignment (unused by the others).
template <class Elem>
inline uint32_t region_offset(Mapping m, uint32_t k, uint32_t threads, uint32_t gtid, const uint32_t* assign) {
    if (m == Mapping::Random) return assign[gtid];
    const bool spread = m == Mapping::Spread;
    if (k <= kBlock) {
        const uint32_t tid = gtid % kBlock;
        return spread ? tid % (kBlock / k) : tid / k;
    }
    return spread ? gtid % (threads / k) : gtid / k;
}

// ---------------------------------------------------------------------------
// Metal
// ---------------------------------------------------------------------------

struct Params {  // the kernels' `constant Params&`
    uint32_t k, ops, seed;
};

struct TotalParams {
    uint32_t n, words_per_elem, value_word;
};

struct Gpu {
    MTL::Device* device;
    MTL::CommandQueue* queue;
    MTL::Library* library;

    MTL::ComputePipelineState* pipeline(const char* name) const {
        NS::Error* err = nullptr;
        MTL::Function* fn = library->newFunction(NS::String::string(name, NS::UTF8StringEncoding));
        if (!fn) fail((std::string("kernel not found in the metallib: ") + name).c_str());
        MTL::ComputePipelineState* p = device->newComputePipelineState(fn, &err);
        fn->release();
        if (!p) fail(name, err);
        return p;
    }

    MTL::Buffer* buffer(size_t bytes) const {
        MTL::Buffer* b = device->newBuffer(bytes, MTL::ResourceStorageModeShared);
        if (!b) fail("buffer allocation failed");
        return b;
    }

    // Zeroes a buffer on the GPU (cudaMemset); ordered before later work on the queue.
    void clear(MTL::Buffer* b, size_t bytes) const {
        MTL::CommandBuffer* cb = queue->commandBuffer();
        MTL::BlitCommandEncoder* blit = cb->blitCommandEncoder();
        blit->fillBuffer(b, NS::Range(0, bytes), 0);
        blit->endEncoding();
        cb->commit();
    }

    // Encodes one compute dispatch with `encode`, runs it and waits; returns its GPU time in ms.
    template <class F>
    float run(F&& encode) const {
        MTL::CommandBuffer* cb = queue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        encode(enc);
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
        if (cb->status() != MTL::CommandBufferStatusCompleted) fail("command buffer failed", cb->error());
        return (float)((cb->GPUEndTime() - cb->GPUStartTime()) * 1e3);
    }
};

// Bytes of threadgroup memory: Metal wants a non-zero multiple of 16.
static size_t threadgroup_bytes(size_t bytes) { return std::max<size_t>(16, (bytes + 15) / 16 * 16); }

// GPU cores, from the IORegistry (Metal does not report them); 0 if unknown.
static int gpu_core_count() {
    int cores = 0;
    io_iterator_t it;
    if (IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("AGXAccelerator"), &it) != KERN_SUCCESS)
        return 0;
    for (io_object_t e; (e = IOIteratorNext(it)) && !cores; IOObjectRelease(e)) {
        CFTypeRef p = IORegistryEntrySearchCFProperty(e, kIOServicePlane, CFSTR("gpu-core-count"), kCFAllocatorDefault,
                                                      kIORegistryIterateRecursively);
        if (p && CFGetTypeID(p) == CFNumberGetTypeID()) CFNumberGetValue((CFNumberRef)p, kCFNumberIntType, &cores);
        if (p) CFRelease(p);
    }
    IOObjectRelease(it);
    return cores;
}

// Most blocks of kBlock threads (with `tg_bytes` of threadgroup memory) resident
// at once, measured with residency_kernel: Metal has no occupancy query.
static int resident_blocks(const Gpu& gpu, int cores, size_t tg_bytes) {
    MTL::ComputePipelineState* p = gpu.pipeline("residency_kernel");
    MTL::Buffer* state = gpu.buffer(3 * sizeof(uint32_t));
    const uint32_t launched = (uint32_t)std::max(cores, 1) * 32;  // far more than fit at once
    int most = 0;
    for (uint32_t spin : {1u << 14, 1u << 16}) {  // long enough that every resident block overlaps
        std::memset(state->contents(), 0, 3 * sizeof(uint32_t));
        gpu.run([&](MTL::ComputeCommandEncoder* enc) {
            enc->setComputePipelineState(p);
            enc->setBuffer(state, 0, 0);
            enc->setBytes(&spin, sizeof(spin), 1);
            enc->setThreadgroupMemoryLength(threadgroup_bytes(tg_bytes), 0);
            enc->dispatchThreadgroups(MTL::Size(launched, 1, 1), MTL::Size(kBlock, 1, 1));
        });
        most = std::max(most, (int)static_cast<uint32_t*>(state->contents())[1]);
    }
    state->release();
    p->release();
    return most;
}

// ---------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------

struct Variant {
    std::string mechanism, mapping, memory;
    uint32_t words_per_elem, value_word;  // element layout, in uint32 words
    MTL::ComputePipelineState* pipeline;

    bool shared() const { return memory == "shared"; }
    void encode(MTL::ComputeCommandEncoder* enc, int grid, MTL::Buffer* table, const Params& p, MTL::Buffer* assign) const {
        enc->setComputePipelineState(pipeline);
        enc->setBuffer(table, 0, 0);
        enc->setBytes(&p, sizeof(p), 1);
        enc->setBuffer(assign, 0, 2);
        const size_t smem = shared() ? (kBlock / p.k) * words_per_elem * sizeof(uint32_t) : 0;
        enc->setThreadgroupMemoryLength(threadgroup_bytes(smem), 0);
        enc->dispatchThreadgroups(MTL::Size(grid, 1, 1), MTL::Size(kBlock, 1, 1));
    }
};

static std::vector<Variant> variants(const Gpu& gpu) {
    std::vector<Variant> out;
    for (const char* mechanism : {"atomic", "lock"})
        for (const char* mapping : {"neighbors", "spread", "random"})
            for (const char* memory : {"shared", "global"}) {
                const bool lock = std::string(mechanism) == "lock";
                const std::string name = std::string(mechanism) + "_" + mapping + "_" + memory;
                out.push_back({mechanism, mapping, memory, lock ? 2u : 1u, lock ? 1u : 0u, gpu.pipeline(name.c_str())});
            }
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

// The metallib next to this executable.
static std::string metallib_path() {
    char buf[4096];
    uint32_t size = sizeof(buf);
    std::string exe = _NSGetExecutablePath(buf, &size) == 0 ? buf : "./threads_per_value_metal";
    return exe.substr(0, exe.find_last_of('/') + 1) + "threads_per_value.metallib";
}

int main(int argc, char** argv) {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    bool test = false;
    std::string out_dir = "results_metal";
    TimingConfig timing;
    timing.reps = 5;
    int grid_override = 0;
    double max_run_s = 10;  // skip points whose single run would take longer
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--test") test = true;
        else if (a == "--out" && i + 1 < argc) out_dir = argv[++i];
        else if (a == "--reps" && i + 1 < argc) timing.reps = std::max(1, std::stoi(argv[++i]));
        else if (a == "--min-ops" && i + 1 < argc) timing.min_ops = std::max(1, std::stoi(argv[++i]));
        else if (a == "--blocks" && i + 1 < argc) grid_override = std::max(1, std::stoi(argv[++i]));
        else if (a == "--max-run-s" && i + 1 < argc) max_run_s = std::stod(argv[++i]);
        else {
            std::printf("Usage: %s [--out DIR] [--reps R] [--min-ops N] [--blocks B] [--max-run-s S] [--test]\n", argv[0]);
            return a == "--help" || a == "-h" ? EXIT_SUCCESS : EXIT_FAILURE;
        }
    }
    Gpu gpu;
    gpu.device = MTL::CreateSystemDefaultDevice();
    if (!gpu.device) fail("no Metal device");
    gpu.queue = gpu.device->newCommandQueue();
    {
        NS::Error* err = nullptr;
        const std::string path = metallib_path();
        gpu.library = gpu.device->newLibrary(NS::String::string(path.c_str(), NS::UTF8StringEncoding), &err);
        if (!gpu.library) fail(path.c_str(), err);
    }
    const std::vector<Variant> all = variants(gpu);
    int warp_size = 0;
    for (const Variant& v : all) {
        if (v.pipeline->maxTotalThreadsPerThreadgroup() < (NS::UInteger)kBlock) {
            std::fprintf(stderr, "A %d-thread block does not fit kernel %s_%s_%s.\n", kBlock, v.mechanism.c_str(),
                         v.mapping.c_str(), v.memory.c_str());
            return EXIT_FAILURE;
        }
        warp_size = (int)v.pipeline->threadExecutionWidth();
    }
    // One full wave: as many 1024-thread blocks as are resident at once (measured,
    // with the largest threadgroup memory of any variant), unless --blocks is given.
    const int cores = gpu_core_count();
    const int grid = grid_override ? grid_override : resident_blocks(gpu, cores, kBlock * sizeof(LockedValue));
    if (grid < 1) fail("could not measure how many blocks are resident; pass --blocks");
    const uint32_t threads = (uint32_t)grid * kBlock;
    const char* name = gpu.device->name()->utf8String();
    std::printf("Threads per value (%s): uint32 atomic add vs. spinlock, threadgroup (shared) vs. device (global)\n"
                "Device: %s, SIMD width %d, %d GPU cores, %d blocks x %d threads = %u threads (%s)\n",
                PLATFORM, name, warp_size, cores, grid, kBlock, threads,
                grid_override ? "--blocks" : "measured residency");
    std::fflush(stdout);

    // Big enough for every layout: at most one element per thread, padding included.
    const size_t table_bytes = threads * sizeof(LockedValue);
    MTL::Buffer* table = gpu.buffer(table_bytes);
    MTL::Buffer* d_assign = gpu.buffer(threads * sizeof(uint32_t));
    MTL::Buffer* d_total = gpu.buffer(sizeof(uint32_t));
    MTL::ComputePipelineState* table_total_pipeline = gpu.pipeline("table_total_kernel");
    MTL::ComputePipelineState* expected_total_pipeline = gpu.pipeline("expected_total_kernel");
    auto total = [&] { return *static_cast<uint32_t*>(d_total->contents()); };
    auto table_total = [&](const Variant& v) {
        *static_cast<uint32_t*>(d_total->contents()) = 0;
        const TotalParams p{threads, v.words_per_elem, v.value_word};
        gpu.run([&](MTL::ComputeCommandEncoder* enc) {
            enc->setComputePipelineState(table_total_pipeline);
            enc->setBuffer(table, 0, 0);
            enc->setBytes(&p, sizeof(p), 1);
            enc->setBuffer(d_total, 0, 3);
            enc->dispatchThreadgroups(MTL::Size(256, 1, 1), MTL::Size(256, 1, 1));
        });
        return total();
    };
    auto expected_total = [&](uint32_t ops, uint32_t seed) {
        *static_cast<uint32_t*>(d_total->contents()) = 0;
        const Params p{0, ops, seed};
        gpu.run([&](MTL::ComputeCommandEncoder* enc) {
            enc->setComputePipelineState(expected_total_pipeline);
            enc->setBytes(&p, sizeof(p), 1);
            enc->setBuffer(d_total, 0, 3);
            enc->dispatchThreadgroups(MTL::Size(grid, 1, 1), MTL::Size(kBlock, 1, 1));
        });
        return total();
    };

    FILE* csv = nullptr;
    if (!test) {
        csv = std::fopen((out_dir + "/threads_per_value.csv").c_str(), "w");
        if (!csv) { std::perror(out_dir.c_str()); return EXIT_FAILURE; }
        std::fprintf(csv, "mechanism,mapping,memory,threads_per_value,values,blocks,threads_per_block,threads,ops_per_thread,"
                          "reps,median_ms,mops_per_s,ns_per_op_per_thread,verified\n");
        if (FILE* dev = std::fopen((out_dir + "/device.txt").c_str(), "w")) {  // for the plot
            std::fprintf(dev, "%s\nwarp_size=%d\n", name, warp_size);
            std::fclose(dev);
        }
    }

    const uint32_t seed = 42;
    int checks = 0, failures = 0, skipped = 0;
    for (const Variant& v : all) {
        // Skip points whose first run would take longer than max_run_s: projected
        // from the previous k of this variant, with run time per op growing like k
        // (the k threads of a value take turns). Only device-scope spinlocks with
        // many sharers get there: their lock hand-offs cost tens of microseconds.
        uint32_t prev_k = 0;
        double prev_ms_per_op = 0;
        for (uint32_t k : threads_per_value(threads)) {
            if (v.shared() && k > (uint32_t)kBlock) continue;  // shared memory is per block
            NS::AutoreleasePool* point_pool = NS::AutoreleasePool::alloc()->init();
            const uint32_t values = threads / k;
            const uint32_t first_ops = test ? 7 : timing.min_ops;
            const double projected_s = prev_k ? prev_ms_per_op * k / prev_k * first_ops * 1e-3 : 0;
            if (projected_s > max_run_s) {
                ++skipped;
                std::printf("%-6s %-9s %-6s %6u threads/value  SKIPPED: one run projected to take %.0f s (--max-run-s %.0f)\n",
                            v.mechanism.c_str(), v.mapping.c_str(), v.memory.c_str(), k, projected_s, max_run_s);
                std::fflush(stdout);
                point_pool->release();
                continue;
            }
            const std::vector<uint32_t> assign = random_assignment(k, threads, seed);  // used by "random" only
            std::memcpy(d_assign->contents(), assign.data(), threads * sizeof(uint32_t));
            prev_k = k;
            if (test) {  // exact, value by value; every lock released
                for (uint32_t ops : {1u, 7u}) {
                    gpu.clear(table, table_bytes);
                    const float ms = gpu.run([&](MTL::ComputeCommandEncoder* enc) { v.encode(enc, grid, table, {k, ops, seed}, d_assign); });
                    prev_ms_per_op = ms / ops;
                    const uint32_t* words = static_cast<const uint32_t*>(table->contents());
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
                        std::fflush(stdout);
                    }
                }
                point_pool->release();
                continue;
            }
            auto run_once = [&](uint32_t ops) {
                gpu.clear(table, table_bytes);
                return gpu.run([&](MTL::ComputeCommandEncoder* enc) { v.encode(enc, grid, table, {k, ops, seed}, d_assign); });
            };
            uint32_t ops;
            std::vector<float> times;
            calibrate_and_time(timing, run_once, ops, times);
            const float ms = median(times);
            prev_ms_per_op = ms / ops;
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
            point_pool->release();
        }
    }

    if (test) {
        std::printf("%d checks, %d failures, %d points skipped. %s\n", checks, failures, skipped,
                    failures ? "TESTS FAILED." : "All tests passed.");
    } else {
        std::fclose(csv);
        std::printf("Wrote %s/threads_per_value.csv%s, %d points skipped\n", out_dir.c_str(),
                    failures ? " (WITH FAILURES)" : "", skipped);
    }
    for (const Variant& v : all) v.pipeline->release();
    table_total_pipeline->release();
    expected_total_pipeline->release();
    table->release();
    d_assign->release();
    d_total->release();
    gpu.library->release();
    gpu.queue->release();
    gpu.device->release();
    pool->release();
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
