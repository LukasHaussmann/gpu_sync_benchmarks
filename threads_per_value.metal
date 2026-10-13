// Threads per value, Metal port: the kernels of threads_per_value.cu in the
// Metal Shading Language (4.1, for its scoped atomics). The host side is
// threads_per_value_metal.cpp; see the header of threads_per_value.cu for what
// is measured. Everything here mirrors the CUDA/HIP device code one to one:
// the same op stream, layouts, mechanisms, scopes and fences.
//
// Tables are handled as arrays of 32-bit words, `Op::words` words per element:
//   atomic:  1 word,  the value
//   lock:    2 words, {lock, value}
#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Scoped atomics and fences
// ---------------------------------------------------------------------------

// Synchronization scope: the narrowest scope that contains all contenders.
// Block = CUDA block scope = Metal threadgroup scope; Device = CUDA device
// scope = Metal device scope. All atomics are relaxed; the lock orders its
// critical section with fences.
enum Scope { kBlockScope, kDeviceScope };

template <int S>
using scoped_uint = atomic<uint, S == kBlockScope ? thread_scope_threadgroup : thread_scope_device>;

// Fence over the memory the table lives in (CUDA __threadfence_block /
// __threadfence order all memory; here a table is in one address space).
inline void fence(device scoped_uint<kBlockScope>*) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_threadgroup);
}
inline void fence(device scoped_uint<kDeviceScope>*) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
}
inline void fence(threadgroup scoped_uint<kBlockScope>*) {
    atomic_thread_fence(mem_flags::mem_threadgroup, memory_order_seq_cst, thread_scope_threadgroup);
}

// Every thread draws its add values from its own xorshift32 generator, in
// registers, so no input is read from memory; the host replays the same streams.
inline uint hash32(uint x) {  // murmur3 finalizer
    x ^= x >> 16; x *= 0x85ebca6bu;
    x ^= x >> 13; x *= 0xc2b2ae35u;
    x ^= x >> 16;
    return x;
}

struct OpStream {
    uint state;

    OpStream(uint seed, uint thread_id) : state(hash32(thread_id ^ hash32(seed)) | 1u) {}

    // Advances the stream; returns a slot in [0, num_slots) and a value in [1, 8].
    void next(uint num_slots, thread uint& slot, thread uint& value) {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        slot = (uint)(((ulong)state * num_slots) >> 32);  // uniform, no modulo
        value = 1u + (state & 7u);
    }
};

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

constant uint kBlock = 1024;

// The two mechanisms. `words` is the element size, `value_word` the index of
// the value within the element, both in uint32 words.
struct Atomic32 {
    enum { words = 1, value_word = 0 };
    template <int S, class P> static void add(P e, uint v) { atomic_fetch_add_explicit(e, v, memory_order_relaxed); }
};

struct Lock32 {
    enum { words = 2, value_word = 1 };
    // Test-and-test-and-set spinlock without backoff. Apple GPUs run the lanes
    // of a SIMD-group in lock-step, as AMD wavefronts do: with "spin until
    // acquired, then work", the winning lane would wait at the end of the spin
    // loop for SIMD-group mates that wait for its lock -> deadlock. So this is
    // the same try-lock loop as the HIP build: a lane that gets the lock runs
    // the critical section and releases the lock within the same iteration.
    template <int S, class P> static void add(P e, uint v) {
        for (bool done = false; !done;) {
            if (atomic_load_explicit(e, memory_order_relaxed) == 0 &&
                atomic_exchange_explicit(e, 1u, memory_order_relaxed) == 0) {
                fence(e);  // acquire
                // The value as a relaxed atomic: always read from and written to
                // the coherent level, as the CUDA build's volatile access.
                const uint x = atomic_load_explicit(e + 1, memory_order_relaxed);
                atomic_store_explicit(e + 1, x + v, memory_order_relaxed);
                fence(e);  // release
                atomic_exchange_explicit(e, 0u, memory_order_relaxed);
                done = true;
            }
        }
    }
};

// Elements per 128-byte cache line, for an element of `words` uint32 words.
inline uint line_elems(uint words) { return 128 / (4 * words); }

// Distance between two blocks' slices of the global table: whole cache lines.
inline uint block_stride(uint per_block, uint words) {
    const uint line = line_elems(words);
    return (per_block + line - 1) / line * line;
}

enum Mapping { kNeighbors, kSpread, kRandom };

// A thread works on one region of the global table: its block's slice for
// k <= kBlock, else the table for the whole GPU. In elements.
inline uint region_base(uint k, uint gtid, uint words) {
    return k <= kBlock ? gtid / kBlock * block_stride(kBlock / k, words) : 0;
}

// A thread's element, relative to its region. `assign` is the random layout's
// thread-to-value assignment (unused by the others).
inline uint region_offset(int m, uint k, uint threads, uint gtid, device const uint* assign) {
    if (m == kRandom) return assign[gtid];
    const bool spread = m == kSpread;
    if (k <= kBlock) {
        const uint tid = gtid % kBlock;
        return spread ? tid % (kBlock / k) : tid / k;
    }
    return spread ? gtid % (threads / k) : gtid / k;
}

struct Params {
    uint k, ops, seed;
};

// One thread's ops, all on its one element. `k` is a kernel argument, so the
// compiler cannot prove that a SIMD-group's lanes hit the same address and
// does not merge their atomics.
template <class Op, int S, int kMap, class P>
inline void add_ops(P region, uint k, device const uint* assign, uint ops, uint seed, uint gtid, uint threads) {
    P e = region + region_offset(kMap, k, threads, gtid, assign) * Op::words;
    OpStream stream(seed, gtid);
    for (uint i = 0; i < ops; ++i) {
        uint slot, v;
        stream.next(1, slot, v);
        Op::template add<S>(e, v);
    }
}

template <class Op, bool kShared, int kMap>
kernel void threads_per_value_kernel(device uint* table [[buffer(0)]], constant Params& p [[buffer(1)]],
                                     device const uint* assign [[buffer(2)]],
                                     threadgroup uint* smem [[threadgroup(0)]],
                                     uint gtid [[thread_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                     uint block_dim [[threads_per_threadgroup]], uint threads [[threads_per_grid]]) {
    const uint k = p.k;
    if (k > block_dim) {  // sharers in several blocks: one table for the GPU (global memory only)
        if (!kShared)
            add_ops<Op, kDeviceScope, kMap>((device scoped_uint<kDeviceScope>*)table, k, assign, p.ops, p.seed, gtid, threads);
        return;
    }
    // Sharers within one block: a table per block (in global memory, the block's slice).
    device uint* block_table = table + region_base(k, gtid, Op::words) * Op::words;
    if (kShared) {
        threadgroup scoped_uint<kBlockScope>* tg = (threadgroup scoped_uint<kBlockScope>*)smem;
        const uint words = block_dim / k * Op::words;
        for (uint i = tid; i < words; i += block_dim) atomic_store_explicit(&tg[i], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        add_ops<Op, kBlockScope, kMap>(tg, k, assign, p.ops, p.seed, gtid, threads);
        threadgroup_barrier(mem_flags::mem_threadgroup);  // write the table out (a few KB, negligible next to the ops)
        for (uint i = tid; i < words; i += block_dim) block_table[i] = atomic_load_explicit(&tg[i], memory_order_relaxed);
    } else {
        add_ops<Op, kBlockScope, kMap>((device scoped_uint<kBlockScope>*)block_table, k, assign, p.ops, p.seed, gtid,
                                       threads);
    }
}

#define VARIANT(Op, shared, map, name)                                                                            \
    template [[host_name(name)]] kernel void threads_per_value_kernel<Op, shared, map>(                           \
        device uint*, constant Params&, device const uint*, threadgroup uint*, uint, uint, uint, uint);

VARIANT(Atomic32, true, kNeighbors, "atomic_neighbors_shared")
VARIANT(Atomic32, false, kNeighbors, "atomic_neighbors_global")
VARIANT(Atomic32, true, kSpread, "atomic_spread_shared")
VARIANT(Atomic32, false, kSpread, "atomic_spread_global")
VARIANT(Atomic32, true, kRandom, "atomic_random_shared")
VARIANT(Atomic32, false, kRandom, "atomic_random_global")
VARIANT(Lock32, true, kNeighbors, "lock_neighbors_shared")
VARIANT(Lock32, false, kNeighbors, "lock_neighbors_global")
VARIANT(Lock32, true, kSpread, "lock_spread_shared")
VARIANT(Lock32, false, kSpread, "lock_spread_global")
VARIANT(Lock32, true, kRandom, "lock_random_shared")
VARIANT(Lock32, false, kRandom, "lock_random_global")

// ---------------------------------------------------------------------------
// Verification and residency
// ---------------------------------------------------------------------------

// Sum of all values every thread adds (mod 2^32, like the table).
kernel void expected_total_kernel(constant Params& p [[buffer(1)]], device atomic_uint* out [[buffer(3)]],
                                  uint gtid [[thread_position_in_grid]]) {
    OpStream stream(p.seed, gtid);
    uint sum = 0;
    for (uint i = 0; i < p.ops; ++i) {
        uint slot, v;
        stream.next(1, slot, v);
        sum += v;
    }
    atomic_fetch_add_explicit(out, sum, memory_order_relaxed);
}

struct TotalParams {
    uint n, words_per_elem, value_word;
};

// Sum of the value words of n elements of `words_per_elem` uint32 words each.
kernel void table_total_kernel(device const uint* table [[buffer(0)]], constant TotalParams& p [[buffer(1)]],
                               device atomic_uint* out [[buffer(3)]], uint gtid [[thread_position_in_grid]],
                               uint threads [[threads_per_grid]]) {
    uint sum = 0;
    for (uint i = gtid; i < p.n; i += threads) sum += table[i * p.words_per_elem + p.value_word];
    atomic_fetch_add_explicit(out, sum, memory_order_relaxed);
}

// Metal has no occupancy query. This kernel measures how many blocks are
// resident at once: each block registers itself, records the highest count
// seen, stays resident for `spin` rounds, then leaves. It uses the benchmark's
// block size and the largest threadgroup memory any variant allocates.
// state: [0] blocks resident now, [1] most blocks resident at once, [2] sink.
kernel void residency_kernel(device atomic_uint* state [[buffer(0)]], constant uint& spin [[buffer(1)]],
                             threadgroup atomic_uint* smem [[threadgroup(0)]],
                             uint tid [[thread_index_in_threadgroup]]) {
    if (tid == 0) {
        atomic_store_explicit(smem, 0u, memory_order_relaxed);
        const uint now = atomic_fetch_add_explicit(&state[0], 1u, memory_order_relaxed) + 1;
        atomic_fetch_max_explicit(&state[1], now, memory_order_relaxed);
        uint sink = 0;
        for (uint i = 0; i < spin; ++i) sink += atomic_load_explicit(&state[1], memory_order_relaxed);
        atomic_fetch_add_explicit(&state[2], sink, memory_order_relaxed);
        atomic_fetch_sub_explicit(&state[0], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
