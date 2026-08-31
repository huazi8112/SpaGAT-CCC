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

    # 列间距压到刚好容纳竖排标签，画布长宽比从 ~4.4:1 收到 ~2:1，
    # 同时字号相对更大（不再是又长又扁的窄条）
    fig_w = max(30.0, n_ct * 1.25 + 24.0)
    fig_h = 46.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300, facecolor="white")
    mod.draw_dotplot(ax, mod.df, top_n=mod.TOP_N_LR)

    # 放大气泡（draw_dotplot 内固定为 1000，这里仅对本单独大图加倍）
    for coll in ax.collections:
        sizes = coll.get_sizes()
        if len(sizes):
            coll.set_sizes(sizes * 2.0)
            coll.set_linewidths(2.0)

    # Override x-axis: show every CT-pair label (draw_dotplot may sample to ~10)
    _TICK_FS = FS.TICK + 32
    ax.set_xticks(range(n_ct))
    ax.set_xticklabels(
        ct_pairs_ordered, rotation=90, ha="center", va="top",
        fontsize=_TICK_FS, fontweight="bold",
    )
    ax.tick_params(axis="x", which="major", length=8, pad=10)

    # 放大 y 轴刻度（L-R 配体-受体对名称）与坐标轴标题
    for t in ax.get_yticklabels():
        t.set_fontsize(_TICK_FS)
    ax.xaxis.label.set_fontsize(FS.TITLE + 24)
    ax.xaxis.label.set_fontfamily("Microsoft YaHei")
    ax.yaxis.label.set_fontsize(FS.AXIS_LABEL + 28)
    ax.yaxis.label.set_fontfamily("Microsoft YaHei")

    # 放大右侧颜色条（draw_dotplot 内部创建，未返回，取图上最后一个新增 axes）
    cbar_ax = [a for a in fig.axes if a is not ax][-1]
    cbar_ax.yaxis.label.set_fontsize(FS.LEGEND_TITLE + 28)
    cbar_ax.yaxis.label.set_fontfamily("Microsoft YaHei")
    cbar_ax.yaxis.labelpad = 46
    cbar_ax.tick_params(labelsize=_TICK_FS - 8)

    ax.set_title(
        "高置信度空间配体-受体对",
        fontsize=FS.TITLE + 42, fontweight="bold", pad=40, loc="left", x=0.0,
        fontfamily="Microsoft YaHei",
    )
    # 纵坐标标题更靠近刻度文字
    ax.yaxis.set_label_coords(-0.125, 0.58)
    fig.subplots_adjust(left=0.17, right=0.92, top=0.92, bottom=0.42)

    # 颜色条整体创建时 pad 在超宽图上会拉出很大间隙，重新贴近主图右边
    ax_pos = ax.get_position()
    cbar_pos = cbar_ax.get_position()
    cbar_ax.set_position([ax_pos.x1 + 0.008, cbar_pos.y0, cbar_pos.width, cbar_pos.height])

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
