"""
附录图 S2 — BC2 (A) 与 seqFISH (B) 空间先验得分 raincloud 合并图。

合并自 02_plot_spatial_prior_bc2.py（面板 A）与 02_plot_spatial_prior_seqfish.py
（面板 B，四子行按中位数分 quarter）。

Output: 02_spatial_prior_bc2_seqfish_S2.png / .pdf / .eps
"""

import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import (
    CT_COLORS,
    CT_COLORS_SEQFISH,
    CT_CHORD_ABBREV_SEQFISH,
    FS_FIG678 as FS,
    apply_paper_style,
)

apply_paper_style(fs=FS)

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR
STEM = "02_spatial_prior_bc2_seqfish_S2"

# Raincloud geometry (shared)
VIOLIN_MAX = 0.40
BOX_CENTER_OFFSET = 0.05
BOX_HALF_WIDTH = 0.06
STRIP_LEFT = 0.18
STRIP_WIDTH = 0.22


def _draw_strip(
    ax,
    combo_scores: pd.DataFrame,
    ct_pair_order: list,
    *,
    sender_colors: dict,
    short_pair,
    max_n: int,
    overall_med: float,
    title: str,
    x_rotation: float = 0.0,
    x_ha: str = "center",
    show_ylabel: bool = True,
):
    rng = np.random.default_rng(42)

    for i, cp_name in enumerate(ct_pair_order):
        sub = combo_scores.loc[
            combo_scores["ct_pair"] == cp_name, "log_score"
        ].values
        if len(sub) == 0:
            continue
        sender = cp_name.split(" → ")[0]
        col = sender_colors.get(sender, "#888888")
        n = len(sub)

        if n >= 4 and np.std(sub) > 1e-6:
            kde = stats.gaussian_kde(sub, bw_method=0.4)
            yg = np.linspace(sub.min() - 0.4, sub.max() + 0.4, 220)
            dens = kde(yg)
            dens = dens / dens.max() * VIOLIN_MAX * np.sqrt(n / max_n)
            ax.fill_betweenx(
                yg, i - dens, i,
                facecolor=col, alpha=0.55,
                edgecolor=col, linewidth=1.2, zorder=2,
            )

        x_jit = i + STRIP_LEFT + rng.uniform(0, STRIP_WIDTH, size=n)
        marker_size = 22 if n > 60 else (34 if n > 10 else 60)
        ax.scatter(
            x_jit, sub, s=marker_size, color=col, alpha=0.55,
            edgecolor="black", linewidth=0.4, zorder=3,
        )

        q1, med, q3 = np.percentile(sub, [25, 50, 75])
        iqr = q3 - q1
        lo = max(sub.min(), q1 - 1.5 * iqr)
        hi = min(sub.max(), q3 + 1.5 * iqr)
        cx = i + BOX_CENTER_OFFSET

        ax.add_patch(mpatches.Rectangle(
            (cx - BOX_HALF_WIDTH, q1), 2 * BOX_HALF_WIDTH, q3 - q1,
            facecolor="white", edgecolor="black", linewidth=1.4, zorder=4))
        ax.plot([cx - BOX_HALF_WIDTH, cx + BOX_HALF_WIDTH], [med, med],
                color="black", linewidth=2.4, zorder=5)
        ax.plot([cx, cx], [lo, q1], color="black", linewidth=1.2, zorder=4)
        ax.plot([cx, cx], [q3, hi], color="black", linewidth=1.2, zorder=4)
        cap = 0.035
        for y in (lo, hi):
            ax.plot([cx - cap, cx + cap], [y, y],
                    color="black", linewidth=1.2, zorder=4)

    positions = list(range(len(ct_pair_order)))
    labels = [
        short_pair(cp) + f"\n$n$={(combo_scores['ct_pair'] == cp).sum()}"
        for cp in ct_pair_order
    ]
    ax.set_xticks(positions)
    ax.set_xticklabels(
        labels, fontsize=FS.TICK_SMALL, fontweight="bold",
        rotation=x_rotation, ha=x_ha,
        rotation_mode="anchor" if x_rotation else "default",
    )
    ax.tick_params(axis="x", pad=4)
    ax.set_xlim(-0.6, len(ct_pair_order) - 0.35)

    if show_ylabel:
        ax.set_ylabel(r"$\log_{10}$(mean\_score)",
                      fontsize=FS.AXIS_LABEL, fontweight="bold")
    ax.tick_params(axis="y", labelsize=FS.TICK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    ax.axhline(overall_med, color="#333333", linestyle=":",
               linewidth=1.6, alpha=0.7, zorder=1)
    ax.text(len(ct_pair_order) - 0.5, overall_med + 0.20,
            f"overall median = {overall_med:.2f}",
            fontsize=FS.ANNOT_SMALL, color="#333333",
            ha="right", style="italic", fontweight="bold")

    if title:
        ax.set_title(title, fontsize=FS.TITLE, fontweight="bold", pad=14)


def _load_bc2():
    root = SCRIPT_DIR.parent / "BreastCancer2"
    scores = pd.read_csv(root / "3.LR_Scoring" / "LR_scores_all_pairs_V0_meta_gpu_A2.csv")
    combo = pd.read_csv(root / "3.LR_Scoring" / "combo_only-A2.csv")
    combo_set = set(combo["combo"].dropna().str.strip())
    df = scores[scores["pair"].isin(combo_set)].copy()
    df["ct_pair"] = df["sender_cluster"] + " → " + df["receiver_cluster"]
    df = df[df["mean_score_recv"] > 0].copy()
    df["log_score"] = np.log10(df["mean_score_recv"] + 1e-15)
    order = (df.groupby("ct_pair")["log_score"].median()
             .sort_values(ascending=False).index.tolist())
    return df, order


def _load_seqfish():
    root = SCRIPT_DIR.parent / "giotto_seqfish"
    scores = pd.read_csv(root / "3.LR_Scoring" / "LR_scores_all_pairs_V0_meta_gpu_A3.csv")
    combo = pd.read_csv(
        root / "2.LR_Screening" / "3.Identify sensitive genes and gene combinations"
        / "3.Subnetwork exploration" / "combo_only-A3.csv"
    )
    combo_set = set(combo["combo"].dropna().str.strip())
    df = scores[scores["pair"].isin(combo_set)].copy()
    df["ct_pair"] = df["sender_cluster"] + " → " + df["receiver_cluster"]
    df = df[df["mean_score_recv"] > 0].copy()
    df["log_score"] = np.log10(df["mean_score_recv"] + 1e-15)
    order = (df.groupby("ct_pair")["log_score"].median()
             .sort_values(ascending=False).index.tolist())
    return df, order


def main():
    # ── BC2 ────────────────────────────────────────────────────────────────
    combo_bc2, order_bc2 = _load_bc2()
    abbrev_bc2 = {"Bcell": "Bce", "Macrophage": "Mac", "Malignant": "Mal",
                  "Stroma": "Str", "Endothelial": "End", "Tcell": "Tce"}

    def short_pair_bc2(cp):
        s, r = cp.split(" → ")
        return f"{abbrev_bc2.get(s, s[:3])}\n→ {abbrev_bc2.get(r, r[:3])}"

    max_n_bc2 = max((combo_bc2["ct_pair"] == cp).sum() for cp in order_bc2)
    med_bc2 = combo_bc2["log_score"].median()

    # ── seqFISH ────────────────────────────────────────────────────────────
    combo_sf, order_sf = _load_seqfish()
    abbrev_sf = CT_CHORD_ABBREV_SEQFISH

    def short_pair_sf(cp):
        s, r = cp.split(" → ")
        return f"{abbrev_sf.get(s, s[:4])}→{abbrev_sf.get(r, r[:4])}"

    n_quarters = 4
    q_step = (len(order_sf) + n_quarters - 1) // n_quarters
    quarters = [order_sf[i * q_step:(i + 1) * q_step]
                for i in range(n_quarters)]
    quarters = [q for q in quarters if len(q) > 0]
    max_n_sf = max((combo_sf["ct_pair"] == cp).sum() for cp in order_sf)
    med_sf = combo_sf["log_score"].median()

    # Figure size: width from longest seq quarter; height = BC2 strip + 4 seq strips
    fig_w = min(40, 8 + 0.65 * max(1, q_step))
    n_sf_rows = len(quarters)
    fig_h = 11.0 + 9.5 * n_sf_rows

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300, facecolor="white")
    # Row 0 = A (BC2); rows 1.. = B panels
    height_ratios = [1.0] + [1.05] * n_sf_rows
    gs = GridSpec(
        1 + n_sf_rows, 1, figure=fig,
        height_ratios=height_ratios,
        left=0.07, right=0.97,
        # Extra room at top; larger hspace so panel A legends do not cover B's title
        top=0.96, bottom=0.04,
        hspace=1.08,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    _draw_strip(
        ax_a, combo_bc2, order_bc2,
        sender_colors=CT_COLORS,
        short_pair=short_pair_bc2,
        max_n=max_n_bc2,
        overall_med=med_bc2,
        title=(
            f"A. Spatial prior score (BreastCancer2) — "
            f"N={len(combo_bc2)} L-R pairs, {len(order_bc2)} cluster pairs"
        ),
        x_rotation=0.0,
        x_ha="center",
        show_ylabel=True,
    )
    # Panel A: legends (match standalone BC2)
    leg_send_a = sorted(set(cp.split(" → ")[0] for cp in order_bc2))
    ct_handles_a = [
        mpatches.Patch(facecolor=CT_COLORS.get(ct, "#888"),
                       edgecolor="black", alpha=0.55,
                       label=f"{abbrev_bc2.get(ct, ct[:3])} = {ct}")
        for ct in leg_send_a
    ]
    violin_p = mpatches.Patch(facecolor="#888", alpha=0.35,
                              edgecolor="#888", label="KDE density")
    box_p = mpatches.Patch(facecolor="white", edgecolor="black",
                           label="median / IQR / whiskers")
    dot_p = plt.Line2D([0], [0], marker="o", linestyle="",
                       markerfacecolor="#888", markeredgecolor="black",
                       markersize=8, alpha=0.7, label="individual L-R pair")
    gleg_a = ax_a.legend(
        handles=[violin_p, box_p, dot_p],
        loc="upper center", ncol=3,
        fontsize=FS.LEGEND, frameon=False,
        bbox_to_anchor=(0.5, -0.18),
        columnspacing=2.5, handlelength=1.5,
    )
    ax_a.add_artist(gleg_a)
    ax_a.legend(
        handles=ct_handles_a,
        loc="upper center", ncol=len(leg_send_a),
        fontsize=FS.LEGEND, frameon=False,
        bbox_to_anchor=(0.5, -0.30),
        columnspacing=2.0, handlelength=1.3,
        title="Sender cell type",
        title_fontsize=FS.LEGEND_TITLE,
    )

    panel_titles = [
        (
            f"B. Spatial prior score (seqFISH) — top {len(quarters[0])} cluster pairs "
            f"[ranked by median, N={len(combo_sf)} L-R pairs total]"
        ),
    ] + [
        (
            f"B (cont.). seqFISH — panel {i+1}/{n_sf_rows} "
            f"[ranks {i*q_step+1}–{min((i+1)*q_step, len(order_sf))}]"
        )
        for i in range(1, n_sf_rows)
    ]

    axes_b = []
    for i, (pairs, ptitle) in enumerate(zip(quarters, panel_titles)):
        ax_b = fig.add_subplot(gs[1 + i, 0])
        axes_b.append(ax_b)
        _draw_strip(
            ax_b, combo_sf, pairs,
            sender_colors=CT_COLORS_SEQFISH,
            short_pair=short_pair_sf,
            max_n=max_n_sf,
            overall_med=med_sf,
            title=ptitle,
            x_rotation=45.0,
            x_ha="right",
            show_ylabel=True,
        )

    ax_last = axes_b[-1]
    legend_senders = sorted(set(cp.split(" → ")[0] for cp in order_sf))
    ct_handles = [
        mpatches.Patch(facecolor=CT_COLORS_SEQFISH.get(ct, "#888"),
                       edgecolor="black", alpha=0.55,
                       label=f"{abbrev_sf.get(ct, ct[:4])} = {ct}")
        for ct in legend_senders
    ]
    ncol_leg = min(6, max(1, len(legend_senders)))

    violin_proxy = mpatches.Patch(facecolor="#888", alpha=0.35,
                                  edgecolor="#888", label="KDE density")
    box_proxy = mpatches.Patch(facecolor="white", edgecolor="black",
                               label="median / IQR / whiskers")
    dot_proxy = plt.Line2D([0], [0], marker="o", linestyle="",
                           markerfacecolor="#888", markeredgecolor="black",
                           markersize=8, alpha=0.7, label="individual L-R pair")

    glyph_leg = ax_last.legend(
        handles=[violin_proxy, box_proxy, dot_proxy],
        loc="upper center", ncol=3,
        fontsize=FS.LEGEND, frameon=False,
        bbox_to_anchor=(0.5, -0.28),
        columnspacing=2.5, handlelength=1.5,
    )
    ax_last.add_artist(glyph_leg)

    ax_last.legend(
        handles=ct_handles,
        loc="upper center", ncol=ncol_leg,
        fontsize=FS.LEGEND, frameon=False,
        bbox_to_anchor=(0.5, -0.40),
        columnspacing=1.4, handlelength=1.2,
        title="Sender cell type",
        title_fontsize=FS.LEGEND_TITLE,
    )

    out_png = OUT_DIR / f"{STEM}.png"
    out_pdf = OUT_DIR / f"{STEM}.pdf"
    out_eps = OUT_DIR / f"{STEM}.eps"
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(out_eps, bbox_inches="tight", facecolor="white", format="eps")
    plt.close()

    print("=== BC2 (A) ===")
    print(f"L-R pairs with score>0: {len(combo_bc2)}")
    print(f"Cluster pair groups:   {len(order_bc2)}")
    print(f"max group n: {max_n_bc2}")
    print("=== seqFISH (B) ===")
    print(f"L-R pairs with score>0: {len(combo_sf)}")
    sizes = " / ".join(str(len(q)) for q in quarters)
    print(f"Cluster pair groups:    {len(order_sf)}  ({sizes} per B-panel row)")
    print(f"max group n: {max_n_sf}")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_eps}")


if __name__ == "__main__":
    main()