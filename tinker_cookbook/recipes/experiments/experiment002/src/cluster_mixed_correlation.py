#!/usr/bin/env python3
"""
Hierarchical clustering of the mixed-model topic-correlation matrix.

Reuses the Spearman ρ matrix from topic_correlation_heatmap.py, converts
it to a distance matrix d = 1 − ρ, runs average-linkage clustering, then
produces a clustermap-style figure: dendrogram on top of the reordered
correlation heatmap.

Also prints the cluster decomposition at a few cut levels and the top
intra-/inter-cluster comparisons.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from topic_correlation_heatmap import (  # noqa: E402
    TOPIC_LABELS,
    correlation_matrix,
    load_deltas,
)

_EXP_DIR = _HERE.parent
_MIXED_DIR = _EXP_DIR / "evaluations" / "narrow_political_calibration" / "results"
_OUT_DIR = _EXP_DIR / "evaluations" / "narrow_political_QA" / "plots_judged"


def to_distance(corr: np.ndarray) -> np.ndarray:
    """Convert a correlation matrix in [-1, 1] to a non-negative distance.

    d(i, j) = 1 - ρ(i, j); diagonal forced to 0; symmetrized.
    """
    d = 1.0 - corr
    d = (d + d.T) / 2.0
    np.fill_diagonal(d, 0.0)
    d = np.clip(d, 0.0, None)
    return d


def reorder(mat: np.ndarray, order: list[int]) -> np.ndarray:
    return mat[np.ix_(order, order)]


def plot_clustermap(
    corr: np.ndarray,
    Z: np.ndarray,
    leaf_order: list[int],
    topic_labels: list[str],
    cluster_ids: np.ndarray,
    out: Path,
    title: str,
) -> None:
    n = len(leaf_order)
    reordered = reorder(corr, leaf_order)
    labels_ordered = [topic_labels[i] for i in leaf_order]
    clusters_ordered = [int(cluster_ids[i]) for i in leaf_order]

    # Lower-triangle only: mask upper triangle and diagonal
    masked = reordered.astype(float).copy()
    for i in range(n):
        for j in range(n):
            if j >= i:
                masked[i, j] = np.nan

    fig = plt.figure(figsize=(12.5, 13.0))
    outer = fig.add_gridspec(2, 1, height_ratios=[3.4, 10.4], hspace=0.55)
    ax_dendro = fig.add_subplot(outer[0, 0])
    inner = outer[1, 0].subgridspec(2, 1, height_ratios=[0.30, 10.0],
                                    hspace=0.04)
    ax_strip = fig.add_subplot(inner[0, 0])
    ax_heat = fig.add_subplot(inner[1, 0])

    # Dendrogram on top, colored by the same palette as the cluster strip
    palette = ["#4477AA", "#EE6677", "#228833", "#CCBB44",
               "#66CCEE", "#AA3377", "#BBBBBB"]

    # Build link_color mapping so dendrogram colors match cluster ids
    cluster_to_color = {
        cid: palette[(cid - 1) % len(palette)]
        for cid in sorted(set(int(c) for c in cluster_ids))
    }
    leaf_color = {leaf_order[i]: cluster_to_color[int(cluster_ids[leaf_order[i]])]
                  for i in range(n)}

    # scipy color_threshold can't easily honor a custom mapping, so
    # post-process: assign each linkage node the color of its members if
    # they all agree, else gray.
    n_leaves = n
    node_color = {i: leaf_color[i] for i in range(n_leaves)}
    node_members = {i: {i} for i in range(n_leaves)}
    for k_idx, (a, b, _, _) in enumerate(Z):
        a, b = int(a), int(b)
        members = node_members[a] | node_members[b]
        node_id = n_leaves + k_idx
        node_members[node_id] = members
        cset = {leaf_color[m] for m in members if m < n_leaves}
        node_color[node_id] = next(iter(cset)) if len(cset) == 1 else "#999999"

    def _link_color(k: int) -> str:
        return node_color.get(k, "#999999")

    dendrogram(
        Z,
        ax=ax_dendro,
        labels=topic_labels,  # original index order; scipy reorders into leaves
        leaf_rotation=40,
        leaf_font_size=9,
        link_color_func=_link_color,
    )
    ax_dendro.set_ylabel("distance (1 − ρ)", fontsize=9)
    ax_dendro.tick_params(axis="y", labelsize=8)
    ax_dendro.tick_params(axis="x", which="both", length=0, pad=2)
    # Color each leaf label by its cluster
    for tick_label, leaf_idx in zip(ax_dendro.get_xticklabels(), leaf_order):
        cid = int(cluster_ids[leaf_idx])
        tick_label.set_color(palette[(cid - 1) % len(palette)])
        tick_label.set_ha("right")
        tick_label.set_fontweight("semibold")
    for spine in ("top", "right", "bottom"):
        ax_dendro.spines[spine].set_visible(False)

    # Cluster strip (horizontal): one colored block per leaf
    strip_img = np.zeros((1, n, 3))
    for col, cid in enumerate(clusters_ordered):
        strip_img[0, col, :] = matplotlib.colors.to_rgb(
            palette[(cid - 1) % len(palette)]
        )
    ax_strip.imshow(strip_img, aspect="auto")
    ax_strip.set_xticks([])
    ax_strip.set_yticks([])
    for spine in ax_strip.spines.values():
        spine.set_visible(False)
    # Cluster numerals centered on each block
    for col, cid in enumerate(clusters_ordered):
        ax_strip.text(col, 0, str(cid), ha="center", va="center",
                      fontsize=9, fontweight="bold", color="white")

    # Lower-triangle heatmap with NaN cells transparent (white background)
    cmap = matplotlib.cm.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="white")
    im = ax_heat.imshow(masked, cmap=cmap, vmin=-1, vmax=1, aspect="equal")
    ax_heat.set_xticks(np.arange(n))
    ax_heat.set_xticklabels(labels_ordered, rotation=40, ha="right", fontsize=9)
    ax_heat.set_yticks(np.arange(n))
    ax_heat.set_yticklabels(labels_ordered, fontsize=9)
    ax_heat.tick_params(axis="both", which="both", length=0)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)

    # Annotate lower triangle only
    for i in range(n):
        for j in range(n):
            if j >= i:
                continue
            v = masked[i, j]
            if np.isnan(v):
                continue
            tc = "white" if abs(v) > 0.55 else "#222222"
            ax_heat.text(j, i, f"{v:+.2f}", ha="center", va="center",
                         fontsize=8.5, color=tc, fontweight="semibold")

    # Draw faint cluster boundary boxes inside the lower triangle
    boundaries: list[int] = []
    for k_idx in range(1, n):
        if clusters_ordered[k_idx] != clusters_ordered[k_idx - 1]:
            boundaries.append(k_idx)
    starts = [0] + boundaries + [n]
    for s, e in zip(starts[:-1], starts[1:]):
        # Box around the within-cluster lower-triangle block
        if e - s < 2:
            continue
        ax_heat.plot([s - 0.5, e - 0.5], [e - 0.5, e - 0.5],
                     color="black", lw=1.4)
        ax_heat.plot([s - 0.5, s - 0.5], [s - 0.5, e - 0.5],
                     color="black", lw=1.4)
        ax_heat.plot([s - 0.5, e - 0.5], [s - 0.5, s - 0.5],
                     color="black", lw=1.4, linestyle=":")  # along diagonal

    # Colorbar to the right of the heatmap
    cbar = fig.colorbar(im, ax=ax_heat, shrink=0.65, pad=0.015,
                        ticks=[-1, -0.5, 0, 0.5, 1])
    cbar.set_label("Spearman ρ across fine-tunes", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.985)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def cluster_summary(
    corr: np.ndarray,
    cluster_ids: np.ndarray,
    topics: list[str],
    label: str,
) -> dict:
    """Print intra-cluster and inter-cluster mean ρ; return a dict."""
    n = len(topics)
    by_cluster: dict[int, list[int]] = {}
    for idx, cid in enumerate(cluster_ids):
        by_cluster.setdefault(int(cid), []).append(idx)

    intra: dict[int, float] = {}
    intra_pairs: dict[int, int] = {}
    for cid, members in by_cluster.items():
        vals = []
        for i in members:
            for j in members:
                if i < j:
                    vals.append(corr[i, j])
        intra[cid] = float(np.mean(vals)) if vals else float("nan")
        intra_pairs[cid] = len(vals)

    inter_vals = []
    for i in range(n):
        for j in range(i + 1, n):
            if cluster_ids[i] != cluster_ids[j]:
                inter_vals.append(corr[i, j])
    inter_mean = float(np.mean(inter_vals)) if inter_vals else float("nan")

    print(f"\n=== {label} — {len(by_cluster)} clusters ===")
    for cid in sorted(by_cluster):
        members = [topics[i] for i in by_cluster[cid]]
        ip = intra_pairs[cid]
        ip_str = f"{intra[cid]:+.3f}" if ip > 0 else "  n/a (singleton)"
        print(f"  cluster {cid} ({len(members):2d} topics, {ip:2d} pairs):"
              f"  mean intra ρ = {ip_str}")
        for m in members:
            print(f"      - {m}")
    print(f"  mean inter-cluster ρ = {inter_mean:+.3f}  "
          f"(n_pairs={len(inter_vals)})")

    return {
        "k": len(by_cluster),
        "clusters": {int(cid): [topics[i] for i in members]
                     for cid, members in by_cluster.items()},
        "intra_mean_rho": {int(cid): intra[cid] for cid in by_cluster},
        "inter_mean_rho": inter_mean,
    }


def main() -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading mixed (dual-ideology) deltas...")
    mixed = load_deltas(_MIXED_DIR)
    print(f"  {len(mixed)} fine-tunes")

    corr, topics = correlation_matrix(mixed, method="spearman")
    n = len(topics)
    print(f"  Correlation matrix: {n} topics")

    # Hierarchical clustering on the distance matrix
    dist = to_distance(corr)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")  # UPGMA
    leaf_order = list(leaves_list(Z))

    # Cut at a few candidate k values; persist k=4 as the headline
    summaries = {}
    for k in (2, 3, 4, 5):
        cluster_ids = fcluster(Z, t=k, criterion="maxclust")
        summaries[k] = cluster_summary(corr, cluster_ids, topics, f"k={k}")

    # Headline figure with k=4 cluster colour-bar
    headline_k = 4
    cluster_ids_h = fcluster(Z, t=headline_k, criterion="maxclust")
    title = (
        "Hierarchical clustering of mixed-model topic correlations\n"
        f"average linkage on 1 − Spearman ρ; cluster bar shows k={headline_k} cut"
    )
    out_png = _OUT_DIR / "topic_correlation_mixed_clustered.png"
    plot_clustermap(
        corr=corr,
        Z=Z,
        leaf_order=leaf_order,
        topic_labels=[TOPIC_LABELS[t] for t in topics],
        cluster_ids=cluster_ids_h,
        out=out_png,
        title=title,
    )

    # JSON sidecar so the report can quote exact numbers
    sidecar = {
        "n_models": len(mixed),
        "topics_in_dendrogram_order": [topics[i] for i in leaf_order],
        "linkage_method": "average",
        "distance": "1 - spearman_rho",
        "cuts": summaries,
    }
    out_json = _OUT_DIR / "topic_correlation_mixed_clustered.json"
    with open(out_json, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"\nSidecar: {out_json}")


if __name__ == "__main__":
    main()
