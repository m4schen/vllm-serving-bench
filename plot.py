#!/usr/bin/env python3
"""Turn results/*.json (written by `vllm bench serve --save-result`) into plots/ and README tables.

Usage:
    python plot.py            # vLLM sweep + vLLM prefix-caching figures
    python plot.py --table    # also print the markdown tables used in the README

    # SGLang, once its results exist:
    python plot.py --tag t4_sglang --prefix-on t4_sglang_prefix_on --prefix-off t4_sglang_prefix_off

Only needs matplotlib. Figures: plots/sweep_<tag>.png and plots/prefix_caching_<tag>.png.
"""
import argparse
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator  # noqa: E402

# Colors: the first three slots of a colorblind-checked categorical palette (light mode).
SERIES = ["#257de0", "#f47230", "#00b894"]
INK, INK_2, GRID, AXIS, SURFACE = "#191919", "#666666", "#e7e7e7", "#c2c2c2", "#ffffff"
TPOT_SLO_MS = 50  # the latency target used for the "capacity" numbers in the README


def load_sweep(tag):
    """results/<tag>/c<conc>_in<len>.json -> {(input_len, concurrency): result}"""
    runs = {}
    for path in glob.glob(os.path.join("results", tag, "c*_in*.json")):
        conc, in_len = map(int, re.search(r"c(\d+)_in(\d+)\.json$", path).groups())
        runs[(in_len, conc)] = json.load(open(path))
    if not runs:
        raise SystemExit(f"No sweep results in results/{tag}/")
    return runs


def load_prefix(folder):
    """results/<folder>/c<conc>.json -> {concurrency: result}"""
    runs = {}
    for path in glob.glob(os.path.join("results", folder, "c*.json")):
        conc = int(re.search(r"c(\d+)\.json$", path).group(1))
        runs[conc] = json.load(open(path))
    return runs


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_sweep(tag, out_path):
    runs = load_sweep(tag)
    in_lens = sorted({k[0] for k in runs})
    concs = sorted({k[1] for k in runs})
    fig, (ax_thr, ax_tpot) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)

    for color, in_len in zip(SERIES, in_lens):
        xs = [c for c in concs if (in_len, c) in runs]
        thr = [runs[(in_len, c)]["output_throughput"] for c in xs]
        tpot = [runs[(in_len, c)]["median_tpot_ms"] for c in xs]
        label = f"input {in_len}"
        for ax, ys in ((ax_thr, thr), (ax_tpot, tpot)):
            ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
            ax.annotate(label, (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK_2)

    for ax in (ax_thr, ax_tpot):
        ax.set_xscale("log", base=2)
        ax.set_xticks(concs)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlim(concs[0] / 1.3, concs[-1] * 2.6)  # room for the end labels
        ax.set_xlabel("concurrency (requests in flight)", color=INK_2, fontsize=9)

    style(ax_thr, "Output throughput", "tokens / s (all requests)")
    ax_thr.set_ylim(bottom=0)
    style(ax_tpot, "P50 time per output token", "ms (log scale)")
    ax_tpot.set_yscale("log")
    ax_tpot.yaxis.set_major_locator(FixedLocator([10, 20, 50, 100, 200, 500, 1000]))
    ax_tpot.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax_tpot.yaxis.set_minor_formatter(NullFormatter())
    ax_tpot.yaxis.set_minor_locator(NullLocator())
    ax_tpot.axhline(TPOT_SLO_MS, color=INK_2, linewidth=1, zorder=2)
    ax_tpot.annotate(f"{TPOT_SLO_MS} ms", (concs[0] / 1.3, TPOT_SLO_MS), xytext=(2, 4),
                     textcoords="offset points", fontsize=8, color=INK_2)

    model = next(iter(runs.values())).get("model_id", "")
    ax_thr.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper left")
    fig.suptitle(f"{model} on a T4 · {tag} · output 128 tokens per request",
                 x=0.01, ha="left", color=INK_2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_prefix(on_folder, off_folder, out_path):
    on, off = load_prefix(on_folder), load_prefix(off_folder)
    concs = sorted(set(on) & set(off))
    if not concs:
        print(f"skip prefix plot: no matching runs in results/{on_folder} and results/{off_folder}")
        return
    panels = [
        ("Output throughput", "tokens / s", lambda d: d["output_throughput"]),
        ("Mean time to first token", "s", lambda d: d["mean_ttft_ms"] / 1000),
        ("Mean time per output token", "ms", lambda d: d["mean_tpot_ms"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), facecolor=SURFACE)
    width = 0.38
    for ax, (title, unit, get) in zip(axes, panels):
        xs = range(len(concs))
        for offset, (name, runs, color) in zip(
                (-width / 2, width / 2), (("prefix caching ON", on, SERIES[0]), ("OFF", off, SERIES[1]))):
            ax.bar([x + offset for x in xs], [get(runs[c]) for c in concs], width,
                   color=color, edgecolor=SURFACE, linewidth=2, label=name, zorder=3)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([f"c = {c}" for c in concs])
        style(ax, title, unit)
    # Selective label: the ON/OFF throughput ratio at the highest concurrency only.
    top = concs[-1]
    ratio = on[top]["output_throughput"] / off[top]["output_throughput"]
    axes[0].annotate(f"{ratio:.1f}×", (len(concs) - 1 - width / 2, on[top]["output_throughput"]),
                     xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, color=INK)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper left")
    fig.suptitle("1,024-token shared prefix + 128 unique tokens per request, output 128 · same requests ON and OFF",
                 x=0.01, ha="left", color=INK_2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out_path}")


def print_tables(tag, on_folder, off_folder):
    runs = load_sweep(tag)
    in_lens = sorted({k[0] for k in runs})
    concs = sorted({k[1] for k in runs})
    for title, key, fmt in (("Output throughput (tok/s)", "output_throughput", "{:.0f}"),
                            ("P50 TPOT (ms)", "median_tpot_ms", "{:.1f}"),
                            ("P99 TTFT (ms)", "p99_ttft_ms", "{:,.0f}")):
        print(f"\n**{title}**\n")
        print("| Input | " + " | ".join(f"c={c}" for c in concs) + " |")
        print("|---|" + "---|" * len(concs))
        for in_len in in_lens:
            cells = [fmt.format(runs[(in_len, c)][key]) if (in_len, c) in runs else "–" for c in concs]
            print(f"| {in_len} | " + " | ".join(cells) + " |")

    on, off = load_prefix(on_folder), load_prefix(off_folder)
    concs = sorted(set(on) & set(off))
    if concs:
        print("\n**Prefix caching ON / OFF**\n")
        print("| Concurrency | Throughput ON / OFF (tok/s) | Mean TTFT ON / OFF (ms) | Mean TPOT ON / OFF (ms) |")
        print("|---|---|---|---|")
        for c in concs:
            a, b = on[c], off[c]
            r = a["output_throughput"] / b["output_throughput"]
            print(f"| {c} | {a['output_throughput']:.0f} / {b['output_throughput']:.0f} ({r:.2f}×) | "
                  f"{a['mean_ttft_ms']:,.0f} / {b['mean_ttft_ms']:,.0f} | "
                  f"{a['mean_tpot_ms']:.1f} / {b['mean_tpot_ms']:.1f} |")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="t4_qwen1.5b", help="sweep folder under results/")
    parser.add_argument("--prefix-on", default="t4_prefix_on", help="prefix-caching ON folder under results/")
    parser.add_argument("--prefix-off", default="t4_prefix_off", help="prefix-caching OFF folder under results/")
    parser.add_argument("--table", action="store_true", help="print the markdown tables used in the README")
    args = parser.parse_args()

    os.makedirs("plots", exist_ok=True)
    plot_sweep(args.tag, os.path.join("plots", f"sweep_{args.tag}.png"))
    plot_prefix(args.prefix_on, args.prefix_off, os.path.join("plots", f"prefix_caching_{args.tag}.png"))
    if args.table:
        print_tables(args.tag, args.prefix_on, args.prefix_off)


if __name__ == "__main__":
    main()
