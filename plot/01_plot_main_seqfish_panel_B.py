"""
图类 01 — seqFISH 主文图 B（气泡图）单独大图。

逻辑与 01_plot_main_seqfish.py 的 B 一致（top L-R + top CT 对）；
单独输出便于排版与审阅。

Input : ../giotto_seqfish/4.Model_Training/pred_scores.csv
Output: 01_main_figure_seqfish_panel_B.png / .eps
"""

import importlib.util
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import FS_FIG8 as FS, apply_paper_style

apply_paper_style(fs=FS)

ROOT = pathlib.Path(__file__).resolve().parent
FIG_TYPE_ID = "01"


def _load_main_module():
    path = ROOT / "01_plot_main_seqfish.py"
    spec = importlib.util.spec_from_file_location("plot_main_seqfish", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    mod = _load_main_module()

    # Mirror draw_dotplot's internal filter to get the exact ordered CT-pair list
    _df = mod.df.copy()
    _df["LRPair"] = _df["Ligand"] + " - " + _df["Receptor"]
    _df["CTPair"] = _df["Sender"] + " \u2192 " + _df["Receiver"]
    _top_lr = _df.groupby("LRPair")["Score"].max().nlargest(mod.TOP_N_LR).index
    _top = _df[_df["LRPair"].isin(_top_lr)]
    ct_pairs_ordered = (
        _top.groupby("CTPair")["Score"].sum()
        .sort_values(ascending=False).index.tolist()
    )
    n_ct = len(ct_pairs_ordered)

    fig_w = max(22.0, n_ct * 1.35 + 8.0)
    fig_h = 28.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300, facecolor="white")
    mod.draw_dotplot(ax, mod.df, top_n=mod.TOP_N_LR)

    # Override x-axis: show every CT-pair label (draw_dotplot may sample to ~10)
    ax.set_xticks(range(n_ct))
    ax.set_xticklabels(
        ct_pairs_ordered, rotation=90, ha="center", va="top",
        fontsize=FS.TICK, fontweight="bold",
    )
    ax.tick_params(axis="x", which="major", length=4, pad=6)

    ax.set_title(
        "High-Confidence Spatial LR Pairs",
        fontsize=FS.TITLE, fontweight="bold", pad=28, loc="left", x=0.0,
    )
    ax.yaxis.set_label_coords(-0.14, 0.58)
    fig.subplots_adjust(left=0.14, right=0.92, top=0.90, bottom=0.30)

    out_png = ROOT / f"{FIG_TYPE_ID}_main_figure_seqfish_panel_B.png"
    out_eps = ROOT / f"{FIG_TYPE_ID}_main_figure_seqfish_panel_B.eps"
    fig.savefig(
        out_png, bbox_inches="tight", facecolor="white", dpi=300, pad_inches=0.35
    )
    fig.savefig(
        out_eps, format="eps", bbox_inches="tight", facecolor="white", pad_inches=0.22
    )
    plt.close(fig)
    print(f"[DONE] {out_png}  (all {n_ct} CT-pair x labels shown)")
    print(f"[DONE] {out_eps}")


if __name__ == "__main__":
    main()
