"""
图类 02 — 空间先验得分 raincloud（Section 3.3），数据集 BC1。
Figure for Section 3.3 — Spatial prior score distribution (panel C, upgraded).

Raincloud-style plot replacing the plain boxplot:
    half-violin (KDE) | inner mini-box (median / IQR / whiskers) | jittered strip
All three layers are aligned on the same x-tick, sorted by median across
sender->receiver cluster pairs.  The strip makes every individual L-R pair
visible (important because n ranges from 3 to 390 across groups), while the
KDE violin shows distribution shape and the inner box anchors the summary
statistics.

Output: 与本脚本同目录 — 02_spatial_prior_bc1.png / .pdf / .eps
"""

import sys
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import CT_COLORS, FS, apply_paper_style

apply_paper_style()

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
BC1_ROOT = SCRIPT_DIR.parent / "BreastCancer1"
SCORES_CSV = BC1_ROOT / "3.LR_Scoring" / "LR_scores_all_pairs_V0_meta_gpu.csv"
COMBO_CSV = BC1_ROOT / "4.Model_Training" / "combo_only.csv"
OUT_DIR = SCRIPT_DIR
FIG_TYPE_ID = "02"

scores_df = pd.read_csv(SCORES_CSV)
combo_df = pd.read_csv(COMBO_CSV)
combo_set = set(combo_df["combo"].dropna().str.strip())

combo_scores = scores_df[scores_df["pair"].isin(combo_set)].copy()
combo_scores["ct_pair"] = combo_scores["sender_cluster"] + " → " + combo_scores["receiver_cluster"]
combo_scores = combo_scores[combo_scores["mean_score_recv"] > 0].copy()
combo_scores["log_score"] = np.log10(combo_scores["mean_score_recv"] + 1e-15)

ct_pair_order = (combo_scores.groupby("ct_pair")["log_score"].median()
                 .sort_values(ascending=False).index.tolist())

CT_ABBREV = {"Bcell": "Bce", "Macrophage": "Mac", "Malignant": "Mal",
             "Stroma": "Str", "Endothelial": "End", "Tcell": "Tce"}


def short_pair(cp):
    s, r = cp.split(" → ")
    return f"{CT_ABBREV.get(s, s[:3])}\n→ {CT_ABBREV.get(r, r[:3])}"


fig = plt.figure(figsize=(22, 10), dpi=300, facecolor="white")
ax = fig.add_axes([0.06, 0.22, 0.89, 0.68])

rng = np.random.default_rng(42)

# Area of each half-violin is proportional to sqrt(n), so big groups look
# visually dominant but small groups are still readable.
max_n = max((combo_scores["ct_pair"] == cp).sum() for cp in ct_pair_order)
VIOLIN_MAX = 0.40        # left-half violin occupies up to 0.40 on the x-axis
BOX_CENTER_OFFSET = 0.05  # inner box sits just right of the tick
BOX_HALF_WIDTH = 0.06
STRIP_LEFT = 0.18
STRIP_WIDTH = 0.22

for i, cp_name in enumerate(ct_pair_order):
    sub = combo_scores.loc[combo_scores["ct_pair"] == cp_name, "log_score"].values
    if len(sub) == 0:
        continue
    sender = cp_name.split(" → ")[0]
    col = CT_COLORS.get(sender, "#888888")
    n = len(sub)

    # --- 1. half-violin (KDE) on left of the tick ---
    if n >= 4 and np.std(sub) > 1e-6:
        kde = stats.gaussian_kde(sub, bw_method=0.4)
        yg = np.linspace(sub.min() - 0.4, sub.max() + 0.4, 220)
        dens = kde(yg)
        dens = dens / dens.max() * VIOLIN_MAX * np.sqrt(n / max_n)
        ax.fill_betweenx(yg, i - dens, i, facecolor=col, alpha=0.55,
                         edgecolor=col, linewidth=1.2, zorder=2)

    # --- 2. jittered strip (raw points) on right of the tick ---
    x_jit = i + STRIP_LEFT + rng.uniform(0, STRIP_WIDTH, size=n)
    marker_size = 22 if n > 60 else (34 if n > 10 else 60)
    ax.scatter(x_jit, sub, s=marker_size, color=col, alpha=0.55,
               edgecolor="black", linewidth=0.4, zorder=3)

    # --- 3. inner mini-box (summary stats) ---
    q1, med, q3 = np.percentile(sub, [25, 50, 75])
    iqr = q3 - q1
    lo = max(sub.min(), q1 - 1.5 * iqr)
    hi = min(sub.max(), q3 + 1.5 * iqr)
    cx = i + BOX_CENTER_OFFSET

    ax.add_patch(mpatches.Rectangle((cx - BOX_HALF_WIDTH, q1),
                                    2 * BOX_HALF_WIDTH, q3 - q1,
                                    facecolor="white", edgecolor="black",
                                    linewidth=1.4, zorder=4))
    ax.plot([cx - BOX_HALF_WIDTH, cx + BOX_HALF_WIDTH], [med, med],
            color="black", linewidth=2.4, zorder=5)
    ax.plot([cx, cx], [lo, q1], color="black", linewidth=1.2, zorder=4)
    ax.plot([cx, cx], [q3, hi], color="black", linewidth=1.2, zorder=4)
    cap = 0.035
    for y in (lo, hi):
        ax.plot([cx - cap, cx + cap], [y, y],
                color="black", linewidth=1.2, zorder=4)

positions = list(range(len(ct_pair_order)))
labels = []
for cp in ct_pair_order:
    n = (combo_scores["ct_pair"] == cp).sum()
    labels.append(short_pair(cp) + f"\n$n$={n}")

ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=FS.TICK_SMALL, fontweight="bold")
ax.tick_params(axis="x", pad=4)
ax.set_xlim(-0.6, len(ct_pair_order) - 0.35)

ax.set_ylabel("$\\log_{10}$(mean\\_score)",
              fontsize=FS.AXIS_LABEL, fontweight="bold")
ax.set_title(
    f"Spatial prior score by sender→receiver cluster pair "
    f"(N={len(combo_scores)} L-R pairs, {len(ct_pair_order)} cluster pairs)",
    fontsize=FS.TITLE, fontweight="bold", pad=14)
ax.tick_params(axis="y", labelsize=FS.TICK)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.25)

overall_med = combo_scores["log_score"].median()
ax.axhline(overall_med, color="#333333", linestyle=":",
           linewidth=1.6, alpha=0.7, zorder=1)
ax.text(len(ct_pair_order) - 0.5, overall_med + 0.20,
        f"overall median = {overall_med:.2f}",
        fontsize=FS.ANNOT_SMALL, color="#333333",
        ha="right", style="italic", fontweight="bold")

legend_senders = sorted(set(cp.split(" → ")[0] for cp in ct_pair_order))
ct_handles = [mpatches.Patch(facecolor=CT_COLORS.get(ct, "#888"),
                             edgecolor="black", alpha=0.55,
                             label=f"{CT_ABBREV.get(ct, ct[:3])} = {ct}")
              for ct in legend_senders]

violin_proxy = mpatches.Patch(facecolor="#888", alpha=0.35,
                              edgecolor="#888", label="KDE density")
box_proxy = mpatches.Patch(facecolor="white", edgecolor="black",
                           label="median / IQR / whiskers")
dot_proxy = plt.Line2D([0], [0], marker="o", linestyle="",
                       markerfacecolor="#888", markeredgecolor="black",
                       markersize=8, alpha=0.7, label="individual L-R pair")
glyph_leg = ax.legend(handles=[violin_proxy, box_proxy, dot_proxy],
                      loc="upper center", ncol=3,
                      fontsize=FS.LEGEND, frameon=False,
                      bbox_to_anchor=(0.5, -0.22),
                      columnspacing=2.5, handlelength=1.5)
ax.add_artist(glyph_leg)

ax.legend(handles=ct_handles, loc="upper center",
          ncol=len(legend_senders),
          fontsize=FS.LEGEND, frameon=False,
          bbox_to_anchor=(0.5, -0.32),
          columnspacing=2.0, handlelength=1.3,
          title="Sender cell type",
          title_fontsize=FS.LEGEND_TITLE)

out_png = OUT_DIR / f"{FIG_TYPE_ID}_spatial_prior_bc1.png"
out_pdf = OUT_DIR / f"{FIG_TYPE_ID}_spatial_prior_bc1.pdf"
out_eps = OUT_DIR / f"{FIG_TYPE_ID}_spatial_prior_bc1.eps"
fig.savefig(out_png, bbox_inches="tight", facecolor="white")
fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
fig.savefig(out_eps, bbox_inches="tight", facecolor="white", format="eps")
plt.close()
print(f"L-R pairs with score>0: {len(combo_scores)}")
print(f"Cluster pair groups:   {len(ct_pair_order)}")
print(f"max group n: {max_n}")
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
print(f"Saved: {out_eps}")
