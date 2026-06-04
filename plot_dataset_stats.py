"""Render a composition graph for a Wordmaze config, for the dataset card.

    uv run plot_dataset_stats.py [data/wordmaze-m4-6/metadata.json] [-o dataset_stats.png]

Reads the metadata.json a generation run writes and produces two panels: puzzle counts
across (word length x move count), and the vocabulary -> ladder-graph sizes per length.
"""
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.7", "numpy>=1.24"]
# ///
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

INDIGO = "#6366f1"
INDIGO_DK = "#4f46e5"
INDIGO_LT = "#a5a6f6"
GRAY = "#9ca3af"
INK = "#1a1a1a"


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Plot Wordmaze dataset composition.")
    ap.add_argument("metadata", nargs="?", default=str(here / "data" / "wordmaze-m4-6" / "metadata.json"))
    ap.add_argument("-o", "--out", default=None, help="Output PNG (default: next to metadata.json).")
    args = ap.parse_args()

    meta = json.loads(Path(args.metadata).read_text())
    out = Path(args.out) if args.out else Path(args.metadata).with_name("dataset_stats.png")
    slug = meta.get("move_count_slug", "wordmaze")

    # examples_by_combo keys look like "4_letters/5_moves"
    combo = meta.get("examples_by_combo", {})
    lengths = sorted({int(k.split("_")[0]) for k in combo})
    moves = sorted({int(k.split("/")[1].split("_")[0]) for k in combo})
    wbl = {int(k): v for k, v in meta.get("words_by_length", {}).items()}
    nodes = {int(k): v for k, v in meta.get("graph_nodes_by_length", {}).items()}
    splits = meta.get("splits", {})
    total = meta.get("generated_examples", sum(combo.values()))

    rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": "#ececec",
        "axes.facecolor": "white", "figure.facecolor": "white",
        "xtick.color": "#6b7280", "ytick.color": "#6b7280",
        "xtick.major.size": 0, "ytick.major.size": 0,
    })
    shades = {lengths[i]: [INDIGO_LT, INDIGO, INDIGO_DK][i % 3] for i in range(len(lengths))}

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.6))

    # panel A: puzzles per (length x moves)
    x = np.arange(len(moves))
    w = 0.8 / max(1, len(lengths))
    for i, L in enumerate(lengths):
        vals = [combo.get(f"{L}_letters/{m}_moves", 0) for m in moves]
        axA.bar(x + (i - (len(lengths) - 1) / 2) * w, vals, w, color=shades[L],
                edgecolor="white", label=f"{L} letters", zorder=3)
    axA.set_xticks(x)
    axA.set_xticklabels([f"{m} moves" for m in moves])
    axA.set_ylabel("puzzles")
    split_txt = " · ".join(f"{k} {v}" for k, v in sorted(splits.items()))
    axA.set_title(f"{total:,} puzzles, balanced across length x moves", fontsize=11, fontweight="bold", color=INK, pad=10)
    axA.legend(fontsize=8.5, loc="lower center", ncol=len(lengths), frameon=False)
    axA.set_ylim(0, max(combo.values()) * 1.25 if combo else 1)

    # panel B: vocabulary -> graph nodes per length
    xb = np.arange(len(lengths))
    axB.bar(xb - 0.2, [wbl.get(L, 0) for L in lengths], 0.4, color=GRAY, edgecolor="white", label="words in dictionary", zorder=3)
    axB.bar(xb + 0.2, [nodes.get(L, 0) for L in lengths], 0.4, color=INDIGO, edgecolor="white", label="words with a neighbor", zorder=3)
    axB.set_xticks(xb)
    axB.set_xticklabels([f"{L} letters" for L in lengths])
    axB.set_ylabel("count")
    axB.set_title("Vocabulary → ladder graph", fontsize=11, fontweight="bold", color=INK, pad=10)
    axB.legend(fontsize=8.5, loc="upper right", frameon=False)

    fig.suptitle(f"Wordmaze {slug}  ·  wordfreq top-10k, proper nouns filtered",
                 fontsize=12.5, fontweight="bold", color=INK, y=1.02)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
