"""
图类 01 — 主文综合图（Chord / Dot / Edge bundling），数据集 BC1。
Figure 6: SpaGAT-CCC Reveals Complex and Spatially Organized Communication in the TME (BC1)
  A (top-left)    : Chord diagram — CT-level global strength (top 100)
  C (bottom-left) : Hierarchical Edge Bundling — gene-level connections (top 60)
  B (right, full) : Dot plot — top 30 L-R pairs across CT pairs

Input : ../BreastCancer1/4.Model_Training/pred_scores.csv
Output: 01_main_figure_bc1.png / .eps
        01_main_figure_bc1_ABC_genes_values.csv（A/B 全量；C 仅图上标注的 hub，不含边表）
"""

import sys
import pathlib
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pycirclize import Circos

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import CT_COLORS, FS_FIG678 as FS, apply_paper_style

apply_paper_style(fs=FS)

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT.parent / "BreastCancer1" / "4.Model_Training" / "pred_scores.csv"
OUT = ROOT
# 与 01_plot_main_bc2.py / 01_plot_main_seqfish.py 同属「图类 01」
FIG_TYPE_ID = "01"


def load_data():
    pred = pd.read_csv(DATA)
    pred["Ligand"]     = pred["Cell1"].str.split("__").str[0]
    pred["SenderCT"]   = pred["Cell1"].str.split("__").str[1]
    pred["Receptor"]   = pred["Cell2"].str.split("__").str[0]
    pred["ReceiverCT"] = pred["Cell2"].str.split("__").str[1]
    pred["CTPair"]     = pred["SenderCT"] + " -> " + pred["ReceiverCT"]
    return pred


# ════════════════════════════════════════════════════════════════════════
# Panel A : Chord diagram  (pycirclize → 极坐标子图，EPS/PDF 中为纯矢量)
# ════════════════════════════════════════════════════════════════════════
def build_chord_circos_bc1(top100):
    """返回 Circos 与用于标注 Strongest flow 的 flow 表（不写栅格图）。"""
    flow = top100.groupby(["SenderCT", "ReceiverCT"])["att"].sum().reset_index()
    all_cts = sorted(set(flow["SenderCT"]) | set(flow["ReceiverCT"]))
    matrix_df = pd.DataFrame(0.0, index=all_cts, columns=all_cts)
    for _, r in flow.iterrows():
        matrix_df.loc[r["SenderCT"], r["ReceiverCT"]] += r["att"]

    color_dict = {ct: CT_COLORS.get(ct, "#999999") for ct in all_cts}
    circos = Circos.initialize_from_matrix(
        matrix_df, space=5, cmap=color_dict,
        label_kws=dict(size=50, weight="bold"),
        link_kws=dict(direction=1, ec="black", lw=0.5),
    )
    return circos, flow


def place_strongest_flow_between_a_c(fig, ax_a, ax_c, flow):
    """Strongest flow 放在 A/C 两子图之间的缝里，避免 transAxes 负 y 压到 C 标题。"""
    best = flow.loc[flow["att"].idxmax()]
    txt = f"最强通讯流：{best['SenderCT']} \u2192 {best['ReceiverCT']}"
    pa = ax_a.get_position()
    pc = ax_c.get_position()
    x = pa.x0 + pa.width / 2
    y = (pa.y0 + pc.y1) / 2
    fig.text(
        x, y, txt,
        transform=fig.transFigure,
        ha="center", va="center",
        fontsize=40, fontstyle="italic", fontweight="bold",
        fontfamily="Microsoft YaHei",
    )


# ════════════════════════════════════════════════════════════════════════
# Panel B : Dot plot (top 30)
# ════════════════════════════════════════════════════════════════════════
def draw_dotplot(ax, pred_all, top_n=30):
    """Dot plot: top_n *unique* LR pairs (by best att), all CT pairs shown."""
    pred_all = pred_all.copy()
    pred_all["LRPair"] = pred_all["Ligand"] + " - " + pred_all["Receptor"]

    # Select top_n unique LR pairs by their single best att score
    top_lr = (
        pred_all.groupby("LRPair")["att"].max()
        .nlargest(top_n).index.tolist()
    )
    top_df = pred_all[pred_all["LRPair"].isin(top_lr)].copy()

    lr_order = (
        top_df.groupby("LRPair")["att"].max()
        .sort_values(ascending=False).index.tolist()
    )
    lr_idx = {lr: i for i, lr in enumerate(lr_order)}

    ct_pairs_ordered = (
        top_df.groupby("CTPair")["att"].sum()
        .sort_values(ascending=False).index.tolist()
    )
    cp_idx = {cp: i for i, cp in enumerate(ct_pairs_ordered)}

    # Color scale: full min→max often stretches the bar when a few outliers sit far
    # from the bulk; use 5–95%ile for the *displayed* range so mid-range contrast
    # is readable. Values outside clip to the ends; colorbar caps show clipping.
    vals = top_df["att"].to_numpy(dtype=float)
    att_lo = float(np.percentile(vals, 5))
    att_hi = float(np.percentile(vals, 95))
    if (not np.isfinite(att_lo)) or (not np.isfinite(att_hi)) or att_hi <= att_lo:
        att_lo, att_hi = float(vals.min()), float(vals.max())
    norm = Normalize(vmin=att_lo, vmax=att_hi, clip=True)
    cmap = plt.cm.RdYlBu_r

    # Fixed marker size (points^2); attention strength is encoded only by color.
    _DOT_S = 1600

    for _, row in top_df.iterrows():
        x = cp_idx[row["CTPair"]]
        y = len(lr_order) - 1 - lr_idx[row["LRPair"]]
        ax.scatter(
            x, y, s=_DOT_S,
            c=[cmap(norm(row["att"]))],
            edgecolors="black", linewidths=0.9, zorder=3,
        )

    ax.set_xticks(range(len(ct_pairs_ordered)))
    ax.set_xticklabels(
        ct_pairs_ordered, rotation=90, ha="center", va="top",
        fontsize=FS.TICK, fontweight="bold",
    )
    ax.set_xlabel(
        "细胞类型对（发送者 \u2192 接收者）",
        fontsize=FS.TITLE, fontweight="bold", labelpad=30,
        fontfamily="Microsoft YaHei",
    )
    ax.tick_params(axis="x", pad=6)

    ax.set_yticks(range(len(lr_order)))
    ylabs = ax.set_yticklabels(
        list(reversed(lr_order)),
        fontsize=FS.TICK, fontweight="bold",
    )
    for t in ylabs:
        t.set_ha("right")
    ax.set_ylabel(
        f"前 {top_n} 个配体-受体对",
        fontsize=FS.AXIS_LABEL, fontweight="bold", labelpad=12,
        fontfamily="Microsoft YaHei",
    )
    # 轴坐标：y 轴标题再向左一点，避免与左上角标题/刻度标签发生遮挡
    # 第二个坐标：0.5=垂直居中；略大于 0.5 则整段竖排标题视觉上上移
    ax.yaxis.set_label_coords(-0.27, 0.62)
    ax.tick_params(axis="y", pad=10)
    ax.set_title(
        "B. 高置信度空间配体-受体对",
        fontsize=FS.TITLE + 10, fontweight="bold", pad=36, loc="left", x=0.08, y=1.04,
        fontfamily="Microsoft YaHei",
    )
    ax.grid(True, linestyle="--", alpha=0.25, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = ax.figure.colorbar(
        sm, ax=ax, shrink=0.45, pad=0.025, aspect=22, extend="both",
    )
    cbar.set_label(
        "注意力分数（5–95百分位范围）",
        fontsize=FS.LEGEND_TITLE, fontweight="bold",
        fontfamily="Microsoft YaHei",
    )
    cbar.ax.tick_params(labelsize=FS.TICK)


# ════════════════════════════════════════════════════════════════════════
# Panel C : Hierarchical Edge Bundling (top 60, gene-level)
# ════════════════════════════════════════════════════════════════════════
def draw_edge_bundling(ax, top60):
    nodes_info = {}
    for _, row in top60.iterrows():
        for col in ["Cell1", "Cell2"]:
            node = row[col]
            gene, ct = node.split("__", 1)
            nodes_info[node] = {"gene": gene, "ct": ct}

    ct_groups = {}
    for node, info in nodes_info.items():
        ct_groups.setdefault(info["ct"], []).append(node)
    for ct in ct_groups:
        ct_groups[ct] = sorted(ct_groups[ct])

    sorted_cts = sorted(ct_groups.keys())
    total_nodes = sum(len(v) for v in ct_groups.values())
    gap_angle = 0.18
    usable_angle = 2 * np.pi - gap_angle * len(sorted_cts)

    node_angles = {}
    cur = 0.0
    for ct in sorted_cts:
        group = ct_groups[ct]
        span = len(group) / total_nodes * usable_angle
        for i, node in enumerate(group):
            node_angles[node] = cur + (i + 0.5) / len(group) * span
        cur += span + gap_angle

    radius = 1.0
    node_pos = {n: (radius * np.cos(a), radius * np.sin(a))
                for n, a in node_angles.items()}

    att_min, att_max = top60["att"].min(), top60["att"].max()
    edge_cmap = plt.cm.Reds

    beta = 0.82
    for _, row in top60.iterrows():
        src, tgt = row["Cell1"], row["Cell2"]
        if src not in node_pos or tgt not in node_pos:
            continue
        x0, y0 = node_pos[src]
        x1, y1 = node_pos[tgt]
        cx = (1 - beta) * (x0 + x1) / 2
        cy = (1 - beta) * (y0 + y1) / 2

        att_norm = (row["att"] - att_min) / (att_max - att_min + 1e-9)
        verts = [(x0, y0), (cx, cy), (x1, y1)]
        codes = [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3]
        path = MPath(verts, codes)
        patch = PathPatch(
            path, facecolor="none",
            edgecolor=edge_cmap(0.3 + 0.7 * att_norm),
            lw=1.2 + 4.2 * att_norm, alpha=0.65, zorder=2,
        )
        ax.add_patch(patch)

    G = nx.Graph()
    for _, row in top60.iterrows():
        G.add_edge(row["Cell1"], row["Cell2"], weight=row["att"])
    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 1

    centrality = nx.degree_centrality(G)
    top_nodes = set(sorted(centrality, key=centrality.get, reverse=True)[:15])

    for ct in sorted_cts:
        for node in ct_groups[ct]:
            x, y = node_pos[node]
            deg = degrees.get(node, 0)
            is_top = node in top_nodes
            s = (550 + 1600 * (deg / max_deg)) if is_top else (180 + 700 * (deg / max_deg))
            ec = "gold" if is_top else "white"
            ew = 3.0 if is_top else 1.0
            ax.scatter(x, y, s=s, c=CT_COLORS.get(ct, "#999"),
                       edgecolors=ec, linewidths=ew, zorder=5)
            if is_top:
                angle = node_angles[node]
                rot = np.degrees(angle)
                ha = "left"
                if np.cos(angle) < 0:
                    rot += 180
                    ha = "right"
                lr = radius * 1.09
                gene_label = f"{nodes_info[node]['gene']} ({deg})"
                ax.text(lr * np.cos(angle), lr * np.sin(angle),
                        gene_label,
                        ha=ha, va="center",
                        fontsize=FS.NODE_LABEL, fontweight="bold",
                        color="black",
                        rotation=rot, rotation_mode="anchor")

    for ct in sorted_cts:
        group = ct_groups[ct]
        a_start = node_angles[group[0]] - 0.04
        a_end   = node_angles[group[-1]] + 0.04
        theta = np.linspace(a_start, a_end, 60)
        arc_r = radius * 1.32
        ax.plot(arc_r * np.cos(theta), arc_r * np.sin(theta),
                color=CT_COLORS.get(ct, "#999"),
                lw=8, solid_capstyle="round", zorder=1)
        mid_a = (a_start + a_end) / 2
        label_r = radius * 1.48
        rot = np.degrees(mid_a)
        ha = "left"
        if np.cos(mid_a) < 0:
            rot += 180
            ha = "right"
        ax.text(label_r * np.cos(mid_a), label_r * np.sin(mid_a), ct,
                ha=ha, va="center",
                fontsize=FS.AXIS_LABEL, fontweight="bold",
                color=CT_COLORS.get(ct, "#333"),
                rotation=rot, rotation_mode="anchor")

    # 外圈细胞类型字在 1.48r 附近，略放大显示范围避免底部/侧边被裁切
    ax.set_xlim(-2.12, 2.12)
    ax.set_ylim(-2.12, 2.12)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("C. 关键分子枢纽（前60，标注中心性前15）",
                 fontsize=FS.TITLE + 10, fontweight="bold", pad=14,
                 fontfamily="Microsoft YaHei")

    handles = [mpatches.Patch(color=CT_COLORS[ct], label=ct)
               for ct in sorted_cts]
    leg = ax.legend(handles=handles, loc="lower center",
                    bbox_to_anchor=(0.5, -0.14), ncol=3,
                    fontsize=FS.LEGEND, title="细胞类型",
                    title_fontsize=FS.LEGEND_TITLE,
                    framealpha=0.95, edgecolor="gray")
    leg.get_title().set_fontweight("bold")
    leg.get_title().set_fontfamily("Microsoft YaHei")


def export_main_figure_bc1_genes_values(
    pred, flow_chord, top60, top_n_dotplot=30,
):
    """仅导出 A/B/C 面板上出现的基因（A 为细胞型对流量，无单基因）及对应数值，合并为一个 CSV。
    C 子图仅导出图上标注了基因名的 hub（度中心性 top15）；不写 bundling_edge 明细。"""
    path = OUT / f"{FIG_TYPE_ID}_main_figure_bc1_ABC_genes_values.csv"
    rows = []

    for _, r in flow_chord.iterrows():
        rows.append({
            "panel": "A",
            "row_kind": "chord_flow_top100",
            "sender_ct": r["SenderCT"],
            "receiver_ct": r["ReceiverCT"],
            "ligand": np.nan,
            "receptor": np.nan,
            "gene_symbol": np.nan,
            "gene_cell_type": np.nan,
            "att": float(r["att"]),
            "degree_in_top60_subgraph": np.nan,
            "degree_centrality": np.nan,
            "is_hub_labeled": np.nan,
        })

    pred_all = pred.copy()
    pred_all["LRPair"] = pred_all["Ligand"] + " - " + pred_all["Receptor"]
    top_lr = (
        pred_all.groupby("LRPair")["att"].max()
        .nlargest(top_n_dotplot).index.tolist()
    )
    top_df = pred_all[pred_all["LRPair"].isin(top_lr)].copy()
    for _, r in top_df.iterrows():
        rows.append({
            "panel": "B",
            "row_kind": "dotplot_lr",
            "sender_ct": r["SenderCT"],
            "receiver_ct": r["ReceiverCT"],
            "ligand": r["Ligand"],
            "receptor": r["Receptor"],
            "gene_symbol": np.nan,
            "gene_cell_type": np.nan,
            "att": float(r["att"]),
            "degree_in_top60_subgraph": np.nan,
            "degree_centrality": np.nan,
            "is_hub_labeled": np.nan,
        })

    nodes_info = {}
    for _, row in top60.iterrows():
        for col in ["Cell1", "Cell2"]:
            node = row[col]
            gene, ct = node.split("__", 1)
            nodes_info[node] = {"gene": gene, "cell_type": ct}

    G = nx.Graph()
    for _, row in top60.iterrows():
        G.add_edge(row["Cell1"], row["Cell2"], weight=row["att"])
    degrees = dict(G.degree())
    centrality = nx.degree_centrality(G)
    hub_nodes = set(sorted(centrality, key=centrality.get, reverse=True)[:15])
    hub_ordered = sorted(hub_nodes, key=lambda n: centrality.get(n, 0.0), reverse=True)

    for node in hub_ordered:
        g = nodes_info[node]["gene"]
        ct = nodes_info[node]["cell_type"]
        deg = int(degrees.get(node, 0))
        rows.append({
            "panel": "C",
            "row_kind": "bundling_node_labeled",
            "sender_ct": np.nan,
            "receiver_ct": np.nan,
            "ligand": np.nan,
            "receptor": np.nan,
            "gene_symbol": g,
            "gene_cell_type": ct,
            "att": np.nan,
            "degree_in_top60_subgraph": deg,
            "degree_centrality": float(centrality.get(node, 0.0)),
            "is_hub_labeled": True,
        })

    out = pd.DataFrame(rows)
    col_order = [
        "panel", "row_kind",
        "sender_ct", "receiver_ct",
        "ligand", "receptor",
        "gene_symbol", "gene_cell_type",
        "att", "degree_in_top60_subgraph", "degree_centrality", "is_hub_labeled",
    ]
    out = out[col_order]
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[DONE] Table: {path}")


# ════════════════════════════════════════════════════════════════════════
def main():
    pred = load_data()
    top100 = pred.head(100).copy()
    top30  = pred.head(30).copy()
    top60  = pred.head(60).copy()

    print(f"Data: {len(pred)} edges")

    print("Building combined figure (A: vector chord on polar axes) ...")
    circos_a, flow_chord = build_chord_circos_bc1(top100)
    # Layout: 2 rows x 2 cols
    #   Left  col 0: row 0 = A (chord),    row 1 = C (edge bundling)
    #   Right col 1: row 0+1 = B (dot plot, full-height)
    fig = plt.figure(figsize=(36, 34), dpi=300, facecolor="white")
    gs = GridSpec(
        2, 2, figure=fig,
        width_ratios=[1.0, 1.15],
        height_ratios=[1, 1],
        hspace=0.24, wspace=0.30,
        left=0.05, right=0.97, top=0.95, bottom=0.10,
    )

    # A : chord (top left) — 极坐标矢量，非位图 imshow
    ax_a = fig.add_subplot(gs[0, 0], projection="polar")
    circos_a.plotfig(ax=ax_a, dpi=300)
    ax_a.set_title(
        "A. 细胞间通讯全局强度（前100）",
        fontsize=FS.TITLE + 10, fontweight="bold", pad=14,
        fontfamily="Microsoft YaHei",
    )

    # C : edge bundling (bottom left)
    ax_c = fig.add_subplot(gs[1, 0])
    draw_edge_bundling(ax_c, top60)
    place_strongest_flow_between_a_c(fig, ax_a, ax_c, flow_chord)

    # B : dot plot (right, full height) — pass full pred so top_n dedup is correct
    ax_b = fig.add_subplot(gs[:, 1])
    draw_dotplot(ax_b, pred, top_n=30)

    out_eps = OUT / f"{FIG_TYPE_ID}_main_figure_bc1.eps"
    out_png = OUT / f"{FIG_TYPE_ID}_main_figure_bc1.png"
    fig.savefig(
        out_eps, format="eps", bbox_inches="tight", pad_inches=0.18,
        facecolor="white",
    )
    fig.savefig(
        out_png, bbox_inches="tight", pad_inches=0.18,
        facecolor="white", dpi=300,
    )
    plt.close(fig)
    print(f"\n[DONE] Saved: {out_png}")
    print(f"[DONE] Saved: {out_eps}")

    export_main_figure_bc1_genes_values(
        pred, flow_chord, top60, top_n_dotplot=30,
    )


if __name__ == "__main__":
    main()
