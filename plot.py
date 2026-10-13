#!/usr/bin/env python3
"""Plot: contention = threads per value, uint32 atomicAdd vs. spinlock, shared vs. global memory.

Usage: plot.py RESULTS_DIR     (reads RESULTS_DIR/threads_per_value.csv and device.txt,
                                writes RESULTS_DIR/threads_per_value.png)

Three panels, one per layout (which threads share a value): neighbors, spread, random.
Color = memory (green shared, blue global), line style = mechanism (solid
atomicAdd, dashed spinlock). The top axis shows how many lanes of one warp hit
the same value, which drives the cost of atomics. The shaded range (more than one block per value)
exists only in global memory.
"""
import csv
import random
import sys
from pathlib import Path

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullLocator  # noqa: E402

INK, MUTED, GRID, SURFACE, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#f0efec"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "axes.titleweight": "bold", "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})


def count_label(x, _pos):
    x = int(round(x))
    return f"{x // 1024}K" if x >= 2048 and x % 1024 == 0 else f"{x:,}"


def log2_count_axis(ax, ticks):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(FuncFormatter(count_label))
    ax.set_xticks(sorted(set(ticks)))
    ax.tick_params(axis="x", labelrotation=45)


def plain_log_axis(ax):
    """Plain tick numbers on a log axis; label minor ticks when the range is short."""
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.10g}"))
    lo, hi = ax.get_ylim()

    def minor(v, _):
        lead = round(v / 10 ** math.floor(math.log10(v)), 6)
        return f"{v:,.10g}" if hi / lo < 12 or (hi / lo < 100 and lead in (2, 5)) else ""
    ax.yaxis.set_minor_formatter(FuncFormatter(minor))
    ax.tick_params(axis="y", which="minor", labelsize=8)


COLOR = {"shared": "#1baf7a", "global": "#2a78d6"}
MARKER = {"shared": "^", "global": "o"}
NAMES = {"atomic": "atomicAdd", "lock": "spinlock"}
BLOCK = 1024
PANELS = [("neighbors", "a) Neighbors share a value  (value = thread / k)"),
          ("spread", "b) Sharers spread out  (value = thread mod number of values)"),
          ("random", "c) Random sharers  (random thread-to-value assignment, k threads each)")]


def lanes_per_value(mapping, k, threads, warp):
    """Most lanes of one warp that hit the same value (the cost driver for atomics).
    Random: average over simulated shuffles of the warp's most-hit value."""
    if mapping == "neighbors":
        return str(min(k, warp))
    region = BLOCK if k <= BLOCK else threads  # threads whose assignment is shuffled together
    values = region // k
    if mapping == "spread":
        return str(-(-warp // values))  # ceil(warp / values)
    # Random: a warp's lanes are distinct threads of a region in which
    # every value occurs exactly k times (drawn without replacement).
    rng = random.Random(1)
    trials = 4000
    total = 0
    for _ in range(trials):
        hits = {}
        for t in rng.sample(range(region), warp):
            hits[t // k] = hits.get(t // k, 0) + 1
        total += max(hits.values())
    return f"{total / trials:.1f}"


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    d = Path(sys.argv[1])
    rows = list(csv.DictReader(open(d / "threads_per_value.csv")))
    for r in rows:
        r["k"] = int(r["threads_per_value"])
        r["mops"] = float(r["mops_per_s"])
    threads = int(rows[0]["threads"])
    # device.txt: GPU name, then "warp_size=N" (older files: name only, warp size 32)
    lines = (d / "device.txt").read_text().splitlines() if (d / "device.txt").exists() else []
    gpu = f" ({lines[0].strip()})" if lines else ""
    warp = next((int(x.split("=")[1]) for x in lines[1:] if x.startswith("warp_size=")), 32)

    fig, axes = plt.subplots(1, 3, figsize=(23, 8.4), sharey=True)
    for ax, (mapping, title) in zip(axes, PANELS):
        for mechanism in ("atomic", "lock"):
            for memory in ("shared", "global"):
                pts = sorted((r["k"], r["mops"]) for r in rows
                             if r["mechanism"] == mechanism and r["mapping"] == mapping and r["memory"] == memory)
                dashed = mechanism == "lock"
                ax.plot([x for x, _ in pts], [y for _, y in pts], color=COLOR[memory], marker=MARKER[memory],
                        markersize=5, linewidth=2, linestyle="--" if dashed else "-",
                        markerfacecolor=SURFACE if dashed else COLOR[memory],
                        label=f"{NAMES[mechanism]}, {memory} memory", zorder=3)
        ks = sorted({r["k"] for r in rows})
        ax.axvspan(BLOCK * 2 ** 0.5, ks[-1] * 1.5, color=BAND, zorder=0, linewidth=0)
        ax.text(BLOCK * 2 ** 0.5 * 1.15, 0.97, "value shared by several blocks\n(global memory only)",
                transform=ax.get_xaxis_transform(), va="top", color=MUTED, fontsize=8.5)
        ax.set_yscale("log")
        log2_count_axis(ax, ks)
        ax.set_xlim(ks[0] / 1.5, ks[-1] * 1.5)
        ax.set_xlabel(f"threads per value k  (1 = own value per thread … {threads:,} = all threads on one value)")
        top = ax.secondary_xaxis("top")
        top.set_xticks(ks, [lanes_per_value(mapping, k, threads, warp) for k in ks])
        top.tick_params(axis="x", colors=MUTED, labelsize=8.5)
        top.xaxis.set_minor_locator(NullLocator())
        top.set_xlabel(f"lanes of a warp ({warp} lanes) on the same value", color=MUTED, fontsize=9)
        ax.set_title(title, loc="left")
        ax.legend(frameon=False, fontsize=9, loc="lower left", ncol=2)
    axes[0].set_ylabel("throughput, whole GPU [million ops/s, log]")
    plain_log_axis(axes[0])

    fig.suptitle(f"Aggregating uint32 values with atomicAdd vs. spinlock: contention = threads per value, shared vs. global memory{gpu}",
                 x=0.01, y=0.985, ha="left", fontweight="bold", fontsize=13)
    fig.text(0.01, 0.945,
             "Storage.  atomicAdd: packed array of uint32 values (4 B each).  Spinlock: array of {int lock; uint32 value} "
             "(8 B each, every value next to its own lock).\n"
             "Shared memory: one table per block.  Global memory, k \u2264 1,024: one table, each block's slice starts on its own "
             "128-B cache line, values within a slice packed.\n"
             f"Global memory, k > 1,024: one packed table for the GPU (all layouts): {threads // (2 * BLOCK):,} values "
             f"at k = 2,048 down to 1 at k = {threads:,}.",
             va="top", color=INK, fontsize=9.5)
    fig.text(0.01, 0.005,
             f"{threads:,} threads (one full wave of 1,024-thread blocks). Up to k = 1,024 the threads of a value are in the "
             "same block in both memories, so only the memory differs. Each thread adds to its value over and over;\n"
             "the compiler cannot merge a warp's atomics (k is a run-time argument). Spinlock: TTAS without backoff. "
             "Scope (atomics, fences): block up to k = 1,024, device beyond.\n"
             "Spread and random: within the block's table up to k = 1,024, the GPU's table beyond. "
             "Top axis for random: average over random assignments.",
             color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.075, 1, 0.915))
    fig.savefig(d / "threads_per_value.png", dpi=130)
    print(f"Wrote {d}/threads_per_value.png")


if __name__ == "__main__":
    main()
