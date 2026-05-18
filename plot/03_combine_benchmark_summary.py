"""
图类 03 — Benchmark 汇总（workflow + 性能柱 + 方法表）。
生成三张子图并拼合为一张：
  A: benchmark_workflow（顶部独占整行）
  B: performance_comparison_bar（左下）
  C: methods_comparison_table（右下）

所有绘图直接在同一 figure 上完成。

输出：与本脚本同目录 — 03_benchmark_summary.png（同一主干名）。
"""

import sys
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import FS, apply_paper_style

apply_paper_style()
# Over-ride: this script uses smaller local fonts because every element
# is tightly packed inside schematic boxes, so we keep its own tuning.
plt.rcParams["font.weight"] = "normal"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

out_dir = pathlib.Path(__file__).resolve().parent
FIG_TYPE_ID = "03"
OUT_STEM = f"{FIG_TYPE_ID}_benchmark_summary"

# ═══════════════════════════════════════════════════════════════════════════════
#  A — Benchmark Workflow（画在传入的 ax 上）
# ═══════════════════════════════════════════════════════════════════════════════

def draw_benchmark_workflow(ax):
    HEADER_BG    = "#d9d9d9"
    SECTION_BG   = "#f7f7f7"
    BORDER_CLR   = "#888888"
    ARROW_CLR    = "#444444"
    ACCENT_RED   = "#c0392b"
    HEATMAP_CMAP = plt.cm.Reds
    METHOD_CLR   = "#333333"

    # A 图字体整体放大（workflow 示意图）
    _A_FS = 1.75

    def sz(pt):
        return int(round(float(pt) * _A_FS))

    ax.set_xlim(0, 17)
    ax.set_ylim(0, 7.5)
    ax.set_aspect("equal")
    ax.axis("off")

    def draw_section_box(x, y, w, h, title):
        rect = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.06",
            linewidth=1.3, edgecolor=BORDER_CLR,
            facecolor=SECTION_BG, linestyle="--", zorder=1)
        ax.add_patch(rect)
        header = FancyBboxPatch(
            (x, y + h - 0.46), w, 0.46, boxstyle="round,pad=0.02",
            linewidth=0, facecolor=HEADER_BG, zorder=2)
        ax.add_patch(header)
        ax.text(x + w / 2, y + h - 0.23, title, ha="center", va="center",
                fontsize=sz(16), fontweight="bold", zorder=3)

    def big_arrow(x1, y1, x2, y2, text=None, dy_text=0.18):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=ARROW_CLR,
                                    lw=2.2, mutation_scale=int(15 * _A_FS)), zorder=5)
        if text:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy_text, text,
                    ha="center", va="bottom", fontsize=sz(11),
                    fontstyle="italic", color="#555")

    def small_arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=ARROW_CLR,
                                    lw=1.5, mutation_scale=int(11 * _A_FS)), zorder=5)

    # ── SECTION 1: Input ──
    draw_section_box(0.3, 0.3, 4.0, 6.9, "Input")

    ax.text(0.65, 6.45, "Gene expression matrix", fontsize=sz(12), fontweight="bold")
    hm_x, hm_y, hm_s = 0.85, 4.8, 0.30
    np.random.seed(42)
    hm = np.random.rand(5, 5)
    hm[0, 2] = 0.96; hm[1, 1] = 0.90; hm[2, 0] = 0.93
    hm[3, 3] = 0.87; hm[4, 4] = 0.91; hm[0, 0] = 0.12; hm[4, 1] = 0.08
    for i in range(5):
        for j in range(5):
            ax.add_patch(plt.Rectangle(
                (hm_x + j * hm_s, hm_y + (4 - i) * hm_s), hm_s, hm_s,
                facecolor=HEATMAP_CMAP(hm[i, j] * 0.85),
                edgecolor="white", lw=0.6, zorder=3))

    ax.text(2.15, 4.45, "+", fontsize=sz(21), fontweight="bold", ha="center", va="center")

    ax.text(0.65, 4.05, "Coordinate information", fontsize=sz(12), fontweight="bold")
    cx0, cy0, cw, ch = 0.85, 3.2, 0.48, 0.33
    for r, lab in enumerate(["x", "y"]):
        ax.add_patch(plt.Rectangle((cx0, cy0 + (1 - r) * ch), cw * 0.65, ch,
                     facecolor="#e0e0e0", edgecolor=BORDER_CLR, lw=0.5, zorder=3))
        ax.text(cx0 + cw * 0.325, cy0 + (1 - r) * ch + ch / 2, lab,
                ha="center", va="center", fontsize=sz(11), fontweight="bold", zorder=4)
        for c in range(4):
            ax.add_patch(plt.Rectangle(
                (cx0 + cw * 0.65 + c * cw * 0.65, cy0 + (1 - r) * ch), cw * 0.65, ch,
                facecolor="white", edgecolor=BORDER_CLR, lw=0.5, zorder=3))

    ax.text(2.15, 2.85, "+", fontsize=sz(21), fontweight="bold", ha="center", va="center")

    ax.text(0.65, 2.45, "Cell type metadata", fontsize=sz(12), fontweight="bold")
    tm_x, tm_y = 0.85, 1.5
    for r in range(2):
        for c in range(5):
            ax.add_patch(plt.Rectangle(
                (tm_x + c * 0.42, tm_y + (1 - r) * 0.33), 0.42, 0.33,
                facecolor="white", edgecolor=BORDER_CLR, lw=0.5, zorder=3))
    cell_colors = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854"]
    for idx in range(5):
        ccx = tm_x + idx * 0.42 + 0.21
        shapes = ["o", "s", "D", "^", "v"]
        ax.plot(ccx, tm_y - 0.22, marker=shapes[idx], markersize=sz(11),
                color=cell_colors[idx], markeredgecolor="white", markeredgewidth=0.6, zorder=4)

    ax.text(2.3, 0.65, "ST dataset", fontsize=sz(14), fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.18", facecolor=HEADER_BG,
                      edgecolor=BORDER_CLR, lw=0.8))

    # ── SECTION 2: Methods ──
    draw_section_box(5.1, 0.3, 3.3, 6.9, "Methods")

    ax.add_patch(FancyBboxPatch(
        (5.4, 6.15), 2.7, 0.42, boxstyle="round,pad=0.04",
        facecolor="#e0e0e0", edgecolor=BORDER_CLR, lw=0.6, zorder=3))
    ax.text(6.75, 6.36, "LR inference", ha="center", va="center",
            fontsize=sz(13), fontweight="bold", zorder=4)

    methods_list = ["CellChatV2", "COMMOT", "CytoSignal", "stMLnet", "SpaGAT-CCC"]
    m_colors = ["#66c2a5", "#e78ac3", "#a6d854", "#8da0cb", "#fc8d62"]
    for i, (m, mc) in enumerate(zip(methods_list, m_colors)):
        yy = 5.55 - i * 0.85
        ax.add_patch(FancyBboxPatch(
            (5.55, yy - 0.22), 2.4, 0.5, boxstyle="round,pad=0.06",
            facecolor="white", edgecolor=BORDER_CLR, lw=0.6,
            linestyle="--", zorder=3))
        ax.plot(5.8, yy + 0.02, marker="s", markersize=sz(11),
                color=mc, markeredgecolor="white", markeredgewidth=0.6, zorder=4)
        is_ours = m == "SpaGAT-CCC"
        ax.text(6.12, yy + 0.02, m, fontsize=sz(14), va="center",
                fontweight="bold" if is_ours else "normal",
                color=ACCENT_RED if is_ours else METHOD_CLR, zorder=4)

    # ── SECTION 3: Benchmark ──
    draw_section_box(9.2, 0.3, 7.5, 6.9, "Benchmark")

    ax.text(10.0, 6.3, "Validated LR pairs", fontsize=sz(12), fontweight="bold", color="#c62828")
    bar_x, bar_y = 10.0, 5.35
    n_cells = 14
    for c in range(n_cells):
        is_pos = c < 5
        ax.add_patch(plt.Rectangle(
            (bar_x + c * 0.38, bar_y), 0.36, 0.7,
            facecolor="#4caf50" if is_pos else "#e0e0e0",
            edgecolor="white", lw=0.8, zorder=3))
    ax.text(bar_x + 1.0, bar_y + 0.82, "positives", fontsize=sz(10),
            ha="center", color="#2e7d32", fontweight="bold")
    ax.text(bar_x + 3.6, bar_y + 0.82, "negatives", fontsize=sz(10),
            ha="center", color="#757575", fontweight="bold")

    ax.text(10.0, 4.85, "Predicted LR scores", fontsize=sz(12), fontweight="bold", color="#1565c0")
    score_y = 3.95
    score_vals = np.array([0.95, 0.88, 0.82, 0.75, 0.68, 0.60, 0.52, 0.45,
                           0.38, 0.30, 0.22, 0.15, 0.08, 0.03])
    for c in range(n_cells):
        intensity = score_vals[c]
        ax.add_patch(plt.Rectangle(
            (bar_x + c * 0.38, score_y), 0.36, 0.7,
            facecolor=plt.cm.Blues(intensity * 0.8 + 0.15),
            edgecolor="white", lw=0.8, zorder=3))
    ax.text(bar_x + n_cells * 0.38 / 2, score_y - 0.2,
            "ranked by score (descending)", fontsize=sz(10),
            ha="center", color="#555", fontstyle="italic")

    small_arrow(bar_x + n_cells * 0.38 + 0.2, 5.0, 14.35, 5.0)

    metric_names = ["ROC-AUC", "PR-AUC", "Precision@k", "Recall@k"]
    metric_colors = ["#1b5e20", "#0d47a1", "#e65100", "#6a1b9a"]
    metric_bg = ["#e8f5e9", "#e3f2fd", "#fff3e0", "#f3e5f5"]
    for i, (mn, mc, mb) in enumerate(zip(metric_names, metric_colors, metric_bg)):
        my = 6.0 - i * 0.72
        ax.add_patch(FancyBboxPatch(
            (14.4, my - 0.22), 2.0, 0.5,
            boxstyle="round,pad=0.06", facecolor=mb,
            edgecolor=mc, lw=1.0, zorder=3))
        ax.text(15.4, my + 0.02, mn, ha="center", va="center",
                fontsize=sz(12), fontweight="bold", color=mc, zorder=4)

    cm_x, cm_y = 10.3, 0.65
    cm_s = 0.65
    cm_labels = [["TP", "FP"], ["FN", "TN"]]
    cm_fc = [["#c8e6c9", "#ffe0b2"], ["#ffcdd2", "#bbdefb"]]
    cm_ec = [["#2e7d32", "#ef6c00"], ["#c62828", "#1565c0"]]
    ax.text(cm_x + cm_s, cm_y + 2 * cm_s + 0.12, "Confusion matrix",
            fontsize=sz(11), ha="center", fontweight="bold", color="#555")
    for r in range(2):
        for c in range(2):
            ax.add_patch(plt.Rectangle(
                (cm_x + c * cm_s, cm_y + (1 - r) * cm_s), cm_s - 0.04, cm_s - 0.04,
                facecolor=cm_fc[r][c], edgecolor=cm_ec[r][c], lw=1.2, zorder=3))
            ax.text(cm_x + c * cm_s + (cm_s - 0.04) / 2,
                    cm_y + (1 - r) * cm_s + (cm_s - 0.04) / 2,
                    cm_labels[r][c], ha="center", va="center",
                    fontsize=sz(13), fontweight="bold", color=cm_ec[r][c], zorder=4)

    small_arrow(cm_x + 2 * cm_s + 0.15, cm_y + cm_s, cm_x + 2 * cm_s + 0.7, cm_y + cm_s)

    formula_x = cm_x + 2 * cm_s + 0.8
    formulas = [
        ("ROC-AUC", "= AUC(FPR, TPR)", "#1b5e20"),
        ("PR-AUC", "= AUC(Recall, Prec)", "#0d47a1"),
        ("Precision", "= TP / (TP+FP)", "#e65100"),
        ("Recall", "= TP / (TP+FN)", "#6a1b9a"),
    ]
    for i, (fn, fv, fc) in enumerate(formulas):
        fy = cm_y + 1.35 - i * 0.42
        ax.text(formula_x, fy, f"{fn} {fv}", fontsize=sz(11),
                fontfamily="monospace", color=fc, fontweight="bold", zorder=4)

    big_arrow(4.3, 3.75, 5.1, 3.75)
    big_arrow(8.4, 3.75, 9.2, 3.75, text="LR scores")


# ═══════════════════════════════════════════════════════════════════════════════
#  B — Performance Comparison Bar（画在传入的 subfig 上）
#  数值来自各数据集 5.Results/.../eval_other_methods.py 当前运行输出（与 performance_comparison_bar 可对齐更新）
# ═══════════════════════════════════════════════════════════════════════════════

# 与各数据集 eval_other_methods.py 终端输出一致（顺序：stMLnet, CellChatV2, COMMOT, CytoSignal, SpaGAT-CCC）
PERFORMANCE_BENCH_DATA = {
        "ROC-AUC": {
            "BC1":     [0.0057, 0.0061, 0.2448, 0.0067, 0.7958],
            "BC2":     [0.0008, 0.0121, 0.1798, 0.0197, 0.1032],
            "seqFISH": [0.0029, 0.0144, 0.5697, 0.0073, 0.6311],
        },
        "PR-AUC": {
            "BC1":     [0.4167, 0.4187, 0.5281, 0.4162, 0.9139],
            "BC2":     [0.7181, 0.7108, 0.8868, 0.7121, 0.9047],
            "seqFISH": [0.5306, 0.7317, 0.9066, 0.5266, 0.9340],
        },
        "Precision": {
            "BC1":     [0.0080, 0.0060, 0.2680, 0.0100, 0.6360],
            "BC2":     [0.0033, 0.1833, 0.7467, 0.1867, 0.8400],
            "seqFISH": [0.0100, 0.6733, 0.7300, 0.0300, 0.7700],
        },
        "Recall": {
            "BC1":     [0.0111, 0.0083, 0.3712, 0.0139, 0.8809],
            "BC2":     [0.0013, 0.0719, 0.2928, 0.0732, 0.3294],
            "seqFISH": [0.0088, 0.5906, 0.6404, 0.0263, 0.6754],
        },
}


def draw_performance_bar(subfig):
    methods  = ["stMLnet", "CellChatV2", "COMMOT", "CytoSignal", "SpaGAT-CCC"]
    datasets = ["BC1", "BC2", "seqFISH"]
    metrics  = ["ROC-AUC", "PR-AUC", "Precision", "Recall"]

    data = PERFORMANCE_BENCH_DATA

    colors = {
        "stMLnet":    "#8da0cb",
        "CellChatV2": "#66c2a5",
        "COMMOT":     "#e78ac3",
        "CytoSignal": "#a6d854",
        "SpaGAT-CCC": "#fc8d62",
    }

    n_methods  = len(methods)
    n_datasets = len(datasets)
    bar_width  = 0.14
    group_gap  = 0.40

    axes_b = subfig.subplots(2, 2)
    subfig.subplots_adjust(
        left=0.06, right=0.99, bottom=0.07, top=0.90,
        hspace=0.22, wspace=0.16,
    )
    axes_b = axes_b.flatten()

    for col, metric in enumerate(metrics):
        ax = axes_b[col]
        for d_idx, ds in enumerate(datasets):
            group_center = d_idx * (n_methods * bar_width + group_gap)
            best_val = max(data[metric][ds])
            for m_idx, method in enumerate(methods):
                x   = group_center + (m_idx - n_methods / 2 + 0.5) * bar_width
                val = data[metric][ds][m_idx]
                is_best = abs(val - best_val) < 1e-9
                ax.bar(x, val, width=bar_width * 0.88,
                       color=colors[method],
                       edgecolor="black", linewidth=0.6, zorder=3)
                # 所有方法均标注完整三位小数（与 performance_comparison_bar 一致），最优仅加红
                lbl = f"{val:.3f}"
                ax.text(
                    x, val + 0.018, lbl,
                    ha="center", va="bottom",
                    fontsize=12 if is_best else 11,
                    rotation=90,
                    color="#c0392b" if is_best else "#222222",
                    fontweight="bold",
                )

        group_centers = [d * (n_methods * bar_width + group_gap) for d in range(n_datasets)]
        ax.set_xticks(group_centers)
        ax.set_xticklabels(ds_xtick_labels, fontsize=14, fontweight="bold")
        ax.set_title(metric, fontsize=18, fontweight="bold", pad=8)
        ax.set_ylabel("Score", fontsize=13)
        ax.set_ylim(0, 1.25)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)

    legend_handles = [Patch(facecolor=colors[m], edgecolor="black", label=m) for m in methods]
    subfig.legend(handles=legend_handles, loc="upper center", ncol=n_methods,
                  fontsize=13, frameon=False, bbox_to_anchor=(0.5, 0.995),
                  handlelength=1.5, handletextpad=0.45, columnspacing=1.35)


# ═══════════════════════════════════════════════════════════════════════════════
#  C — Methods Comparison Table（画在传入的 ax 上）
# ═══════════════════════════════════════════════════════════════════════════════

def draw_methods_table(ax):
    methods = ["CellChatV2", "COMMOT", "CytoSignal", "stMLnet", "SpaGAT-CCC"]
    functionalities = [
        "Spatial\ninformation",
        "Intercellular\nsignaling",
        "Intracellular\nsignaling",
        "Multilayer",
        "Feedback\nloop",
    ]
    table = {
        "CellChatV2":  [True,  True,  False, False, False],
        "COMMOT":      [True,  True,  False, True,  False],
        "CytoSignal":  [True,  True,  False, False, False],
        "stMLnet":     [True,  True,  True,  True,  True],
        "SpaGAT-CCC":  [True,  True,  True,  True,  True],
    }

    n_methods = len(methods)
    n_funcs = len(functionalities)

    col0_w = 3.05
    col_w = 2.38
    row_h = 1.12
    header_h = 1.55
    total_w = col0_w + n_funcs * col_w
    total_h = header_h + n_methods * row_h
    x0, y0 = 0.35, 0.35

    fig_w = total_w + 0.75
    fig_h = total_h + 0.85
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.set_aspect("equal")
    ax.axis("off")

    HEADER_BG = "#c0392b"
    HEADER_BG2 = "#d9534f"
    HEADER_TEXT = "white"
    ROW_EVEN = "#ffffff"
    ROW_ODD = "#f5f5f5"
    BORDER = "#cccccc"
    HIGHLIGHT_BG = "#fff8e1"

    top_y = y0 + total_h

    ax.add_patch(plt.Rectangle(
        (x0, top_y - header_h), col0_w, header_h,
        facecolor=HEADER_BG, edgecolor="white", lw=1.5, zorder=3))
    ax.text(x0 + col0_w / 2, top_y - header_h / 2, "Methods",
            ha="center", va="center", fontsize=19, fontweight="bold",
            color=HEADER_TEXT, zorder=4)

    func_total_w = n_funcs * col_w
    ax.add_patch(plt.Rectangle(
        (x0 + col0_w, top_y - header_h * 0.4), func_total_w, header_h * 0.4,
        facecolor=HEADER_BG, edgecolor="white", lw=1.5, zorder=3))
    ax.text(x0 + col0_w + func_total_w / 2, top_y - header_h * 0.2,
            "Functionalities", ha="center", va="center",
            fontsize=19, fontweight="bold", color=HEADER_TEXT, zorder=4)

    for j, func in enumerate(functionalities):
        fx = x0 + col0_w + j * col_w
        ax.add_patch(plt.Rectangle(
            (fx, top_y - header_h), col_w, header_h * 0.6,
            facecolor=HEADER_BG2, edgecolor="white", lw=1.0, zorder=3))
        ax.text(fx + col_w / 2, top_y - header_h * 0.4 - header_h * 0.3,
                func, ha="center", va="center", fontsize=15,
                fontweight="bold", color=HEADER_TEXT, zorder=4,
                linespacing=1.15)

    for i, method in enumerate(methods):
        ry = top_y - header_h - (i + 1) * row_h
        is_ours = method == "SpaGAT-CCC"
        row_bg = HIGHLIGHT_BG if is_ours else (ROW_ODD if i % 2 else ROW_EVEN)

        ax.add_patch(plt.Rectangle(
            (x0, ry), col0_w, row_h,
            facecolor=row_bg, edgecolor=BORDER, lw=0.6, zorder=2))
        ax.text(x0 + 0.2, ry + row_h / 2, method,
                ha="left", va="center", fontsize=17,
                fontweight="bold" if is_ours else "normal",
                color="#c0392b" if is_ours else "#333333", zorder=4)

        vals = table[method]
        for j, val in enumerate(vals):
            fx = x0 + col0_w + j * col_w
            ax.add_patch(plt.Rectangle(
                (fx, ry), col_w, row_h,
                facecolor=row_bg, edgecolor=BORDER, lw=0.6, zorder=2))
            cx = fx + col_w / 2
            cy = ry + row_h / 2
            r = 0.31
            if val:
                circle = plt.Circle((cx, cy), r, facecolor="#4CAF50",
                                    edgecolor="white", lw=1.2, zorder=4)
            else:
                circle = plt.Circle((cx, cy), r, facecolor="#e0e0e0",
                                    edgecolor="#cccccc", lw=1.0, zorder=4)
            ax.add_patch(circle)

    ax.add_patch(plt.Rectangle(
        (x0, top_y - header_h - n_methods * row_h), total_w, header_h + n_methods * row_h,
        facecolor="none", edgecolor="#999999", lw=1.5, zorder=5))
    ax.plot([x0, x0 + total_w],
            [top_y - header_h, top_y - header_h],
            color="#999999", lw=1.2, zorder=5)


# ═══════════════════════════════════════════════════════════════════════════════
#  组合：使用 subfigures 直接绘制，输出真矢量
# ═══════════════════════════════════════════════════════════════════════════════

print("Drawing combined figure (true vector)...")

fig = plt.figure(figsize=(22, 20), dpi=300)
fig.patch.set_facecolor("white")

# 上行 A 占更高比例；下行 B/C 更紧凑（用于 B/C 角标 y 位置）
_H_TOP, _H_BOT = 1.48, 0.88
_Y_SPLIT = _H_BOT / (_H_TOP + _H_BOT)

subfigs_row = fig.subfigures(2, 1, height_ratios=[_H_TOP, _H_BOT], hspace=0.028)

# ── A：顶行，单个 axes（subplots_adjust 贴边，占满 panel）──
ax_A = subfigs_row[0].subplots(1, 1)
subfigs_row[0].subplots_adjust(left=0.02, right=0.995, top=0.98, bottom=0.02)
draw_benchmark_workflow(ax_A)

# ── 底行：B/C 间距收紧，宽度接近 1:1 以占满宽度 ──
subfigs_bottom = subfigs_row[1].subfigures(1, 2, wspace=0.008, width_ratios=[1.02, 1.0])
subfigs_row[1].subplots_adjust(left=0.01, right=0.995, top=0.99, bottom=0.01)

# ── B：左下，2×2 柱状图 ──
draw_performance_bar(subfigs_bottom[0])

# ── C：右下，方法对比表 ──
ax_C = subfigs_bottom[1].subplots(1, 1)
subfigs_bottom[1].subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
draw_methods_table(ax_C)

# ── ABC 标签：A 贴顶行；B/C 贴底行上沿（随 height_ratios 对齐）──
fig.text(0.015, 0.985, "A", fontsize=44, fontweight="bold", va="top", ha="left")
fig.text(0.015, _Y_SPLIT + 0.012, "B", fontsize=44, fontweight="bold", va="bottom", ha="left")
fig.text(0.515, _Y_SPLIT + 0.012, "C", fontsize=44, fontweight="bold", va="bottom", ha="left")

# ── 保存（png）──
out_png = out_dir / f"{OUT_STEM}.png"
fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"Saved: {out_png}")
