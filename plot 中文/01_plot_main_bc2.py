"""
图类 01 — 主文综合图（Heatmap / Sankey / Edge bundling），数据集 BC2。
Figure 7: BreastCancer2 — Alternative Visualizations
  A. Heatmap           (top left)     — CT-pair communication strength (top 100)
  B. Sankey / Alluvial (right, full)  — Flow from sender to receiver CT (top 30)
  C. Hierarchical Edge Bundling (bottom left) — Gene-level connections (top 30)

Input : ../BreastCancer2/4.Model_Training/pred_scores.csv  (same pattern as BC1)
Output: 01_main_figure_bc2.png / .eps
        01_main_figure_bc2_ABC_genes_values.csv（A/B/C 与图一致；C 仅标注 hub，无边表）
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
from matplotlib.patches import PathPatch, FancyBboxPatch
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib import patheffects

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import CT_COLORS, FS_FIG678 as FS, apply_paper_style

apply_paper_style(fs=FS)

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT.parent / "BreastCancer2" / "4.Model_Training" / "pred_scores.csv"
OUT = ROOT
FIG_TYPE_ID = "01"


def load_data():
    pred = pd.read_csv(DATA)
    pred["Ligand"]     = pred["Cell1"].str.split("__").str[0]
    pred["SenderCT"]   = pred["Cell1"].str.split("__").str[1]
    pred["Receptor"]   = pred["Cell2"].str.split("__").str[0]
    pred["ReceiverCT"] = pred["Cell2"].str.split("__").str[1]
    pred["CTPair"]     = pred["SenderCT"] + " -> " + pred["ReceiverCT"]
    return pred


# ====================================================================
# Panel A : Heatmap  (CT x CT aggregated attention)
# ====================================================================
def draw_heatmap(ax, top100):
    """矢量热图：pcolormesh 四边形填色，EPS/PDF 中无数格点栅格（非 imshow）。"""
    flow = top100.groupby(["SenderCT", "ReceiverCT"]).agg(
        total_att=("att", "sum"), n_pairs=("att", "count")
    ).reset_index()

    all_cts = sorted(set(flow["SenderCT"]) | set(flow["ReceiverCT"]))
    matrix = pd.DataFrame(0.0, index=all_cts, columns=all_cts)
    count_matrix = pd.DataFrame(0, index=all_cts, columns=all_cts)
    for _, r in flow.iterrows():
        matrix.loc[r["SenderCT"], r["ReceiverCT"]] = r["total_att"]
        count_matrix.loc[r["SenderCT"], r["ReceiverCT"]] = int(r["n_pairs"])

    Z = matrix.values
    n = len(all_cts)
    vmax = float(Z.max())
    if vmax <= 0:
        vmax = 1e-9
    x_edges = np.arange(-0.5, n + 0.5, 1.0)
    y_edges = np.arange(-0.5, n + 0.5, 1.0)

    pcm = ax.pcolormesh(
        x_edges,
        y_edges,
        Z,
        cmap="YlOrRd",
        vmin=0.0,
        vmax=vmax,
        shading="flat",
        edgecolors="0.35",
        linewidths=0.2,
    )
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("auto")

    for i in range(n):
        for j in range(n):
            val = Z[i, j]
            cnt = count_matrix.values[i, j]
            if val > 0:
                text_color = "white" if val > Z.max() * 0.6 else "black"
                ax.text(j, i, f"{val:.2f}\n({cnt})",
                        ha="center", va="center", fontsize=FS.DATA_LABEL,
                        fontweight="bold", color=text_color)

    ax.set_xticks(range(len(all_cts)))
    ax.set_xticklabels(all_cts, rotation=45, ha="right",
                       fontsize=FS.TICK, fontweight="bold")
    ax.set_yticks(range(len(all_cts)))
    ax.set_yticklabels(all_cts, fontsize=FS.TICK, fontweight="bold")
    ax.set_xlabel("接收方细胞类型",
                  fontsize=FS.AXIS_LABEL, fontweight="bold", labelpad=8,
                  fontfamily="Microsoft YaHei")
    ax.set_ylabel("发送方细胞类型",
                  fontsize=FS.AXIS_LABEL, fontweight="bold",
                  fontfamily="Microsoft YaHei")
    ax.set_title("A. 细胞类型间通讯强度（前100）",
                 fontsize=FS.TITLE + 10, fontweight="bold", pad=14,
                 fontfamily="Microsoft YaHei")

    cbar = ax.figure.colorbar(pcm, ax=ax, shrink=0.7, pad=0.03, aspect=18)
    cbar.set_label("累计注意力值",
                   fontsize=FS.LEGEND_TITLE, fontweight="bold",
                   fontfamily="Microsoft YaHei")
    cbar.ax.tick_params(labelsize=FS.TICK)


# ====================================================================
# Panel B : Sankey with individual L-R ribbons and gene pair labels
# ====================================================================
def draw_sankey(ax, top30):
    top30 = top30.copy()
    top30["LRLabel"] = top30["Ligand"] + "-" + top30["Receptor"]
    top30 = top30.sort_values("att", ascending=False).reset_index(drop=True)

    n = len(top30)
    row_h = 0.028
    row_gap = 0.004
    ct_gap = 0.018

    x_left, x_right = 0.0, 1.0
    bar_w = 0.08

    def build_blocks(col):
        order = []
        for ct in top30[col]:
            if ct not in order:
                order.append(ct)
        counts = top30[col].value_counts()
        pos, cursors = {}, {}
        y = 1.0
        for ct in order:
            cnt = int(counts[ct])
            h = cnt * row_h + max(0, cnt - 1) * row_gap
            pos[ct] = (y - h, y)
            cursors[ct] = y
            y -= h + ct_gap
        return pos, cursors

    s_pos, s_cur = build_blocks("SenderCT")
    r_pos, r_cur = build_blocks("ReceiverCT")

    for ct, (y0, y1) in s_pos.items():
        ax.add_patch(FancyBboxPatch(
            (x_left, y0), bar_w, y1 - y0, boxstyle="round,pad=0.004",
            facecolor=CT_COLORS.get(ct, "#999"), edgecolor="black", lw=1.5, zorder=5))
        ax.text(x_left - 0.02, (y0 + y1) / 2, ct,
                ha="right", va="center",
                fontsize=FS.AXIS_LABEL, fontweight="bold",
                color=CT_COLORS.get(ct, "#333"))

    for ct, (y0, y1) in r_pos.items():
        ax.add_patch(FancyBboxPatch(
            (x_right - bar_w, y0), bar_w, y1 - y0, boxstyle="round,pad=0.004",
            facecolor=CT_COLORS.get(ct, "#999"), edgecolor="black", lw=1.5, zorder=5))
        ax.text(x_right + 0.02, (y0 + y1) / 2, ct,
                ha="left", va="center",
                fontsize=FS.AXIS_LABEL, fontweight="bold",
                color=CT_COLORS.get(ct, "#333"))

    x0 = x_left + bar_w
    x1 = x_right - bar_w
    xm = (x0 + x1) / 2

    for _, row in top30.iterrows():
        sct, rct = row["SenderCT"], row["ReceiverCT"]

        sy_top = s_cur[sct]
        sy_bot = sy_top - row_h
        s_cur[sct] = sy_bot - row_gap

        ry_top = r_cur[rct]
        ry_bot = ry_top - row_h
        r_cur[rct] = ry_bot - row_gap

        verts = [
            (x0, sy_top), (xm, sy_top), (xm, ry_top), (x1, ry_top),
            (x1, ry_bot), (xm, ry_bot), (xm, sy_bot), (x0, sy_bot),
            (x0, sy_top),
        ]
        codes = [
            MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
            MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
            MPath.CLOSEPOLY,
        ]
        path = MPath(verts, codes)
        patch = PathPatch(path, facecolor=CT_COLORS.get(sct, "#aaa"),
                          alpha=0.50, edgecolor=CT_COLORS.get(sct, "#666"),
                          lw=0.4, zorder=2)
        ax.add_patch(patch)

        mid_y = (sy_top + sy_bot + ry_top + ry_bot) / 4
        label = f"{row['LRLabel']} ({row['att']:.2f})"
        ax.text(xm, mid_y, label,
                ha="center", va="center",
                fontsize=FS.TICK_SMALL, fontweight="bold",
                color="#222", zorder=6,
                path_effects=[patheffects.withStroke(linewidth=3, foreground="white")])

    ax.text(x_left + bar_w / 2, 1.055, "发送方\n细胞类型",
            ha="center", fontsize=FS.SUBTITLE, fontweight="bold",
            fontfamily="Microsoft YaHei")
    ax.text(xm, 1.055, "配体-受体（注意力分数）",
            ha="center", fontsize=FS.SUBTITLE, fontweight="bold",
            fontfamily="Microsoft YaHei")
    ax.text(x_right - bar_w / 2, 1.055, "接收方\n细胞类型",
            ha="center", fontsize=FS.SUBTITLE, fontweight="bold",
            fontfamily="Microsoft YaHei")

    all_y = [v[0] for v in list(s_pos.values()) + list(r_pos.values())]
    ax.set_xlim(-0.22, 1.22)
    ax.set_ylim(min(all_y) - 0.03, 1.12)
    ax.axis("off")
    ax.set_title("B. 细胞类型通讯流向（桑基图，前30）",
                 fontsize=FS.TITLE + 10, fontweight="bold", pad=18,
                 fontfamily="Microsoft YaHei")


# ====================================================================
# Panel C : Hierarchical Edge Bundling  (top 60, gene-level)
# ====================================================================
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
            lw=1.0 + 4.0 * att_norm, alpha=0.6, zorder=2,
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
            s = (400 + 1200 * (deg / max_deg)) if is_top else (150 + 600 * (deg / max_deg))
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
                color=CT_COLORS.get(ct, "#999"), lw=8,
                solid_capstyle="round", zorder=1)
        mid_a = (a_start + a_end) / 2
        label_r = radius * 1.48
        rot = np.degrees(mid_a)
        ha = "left"
        if np.cos(mid_a) < 0:
            rot += 180
            ha = "right"
        tx = label_r * np.cos(mid_a)
        ty = label_r * np.sin(mid_a)
        # 仅 Macrophage：更贴近外圈弧、略向右下，避免与 C 标题叠字
        if ct == "Macrophage":
            label_r_m = radius * 1.33
            tx = label_r_m * np.cos(mid_a)
            ty = label_r_m * np.sin(mid_a)
            tx += 0.22
            ty -= 0.16
        ax.text(tx, ty, ct,
                ha=ha, va="center",
                fontsize=FS.AXIS_LABEL, fontweight="bold",
                color=CT_COLORS.get(ct, "#333"),
                rotation=rot, rotation_mode="anchor")

    ax.set_xlim(-1.85, 1.85)
    ax.set_ylim(-1.85, 1.85)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("C. 层级边捆绑图（前60）",
                 fontsize=FS.TITLE + 10, fontweight="bold", pad=14,
                 fontfamily="Microsoft YaHei")

    handles = [mpatches.Patch(color=CT_COLORS[ct], label=ct) for ct in sorted_cts]
    leg = ax.legend(handles=handles, loc="lower center",
                    bbox_to_anchor=(0.5, -0.10), ncol=3,
                    fontsize=FS.LEGEND,
                    title="细胞类型", title_fontsize=FS.LEGEND_TITLE,
                    framealpha=0.95, edgecolor="gray")
    leg.get_title().set_fontweight("bold")
    leg.get_title().set_fontfamily("Microsoft YaHei")


def export_main_figure_bc2_genes_values(top100, top30, top60):
    """导出 A 热图细胞型对、B Sankey top30 L-R、C bundling 标注 hub（top15），列与 BC1 CSV 一致。"""
    path = OUT / f"{FIG_TYPE_ID}_main_figure_bc2_ABC_genes_values.csv"
    rows = []

    flow_a = top100.groupby(["SenderCT", "ReceiverCT"]).agg(
        total_att=("att", "sum"),
        n_pairs=("att", "count"),
    ).reset_index()
    for _, r in flow_a.iterrows():
        rows.append({
            "panel": "A",
            "row_kind": "heatmap_ct_pair_top100",
            "sender_ct": r["SenderCT"],
            "receiver_ct": r["ReceiverCT"],
            "ligand": np.nan,
            "receptor": np.nan,
            "gene_symbol": np.nan,
            "gene_cell_type": np.nan,
            "att": float(r["total_att"]),
            "degree_in_top60_subgraph": np.nan,
            "degree_centrality": np.nan,
            "is_hub_labeled": np.nan,
        })

    top30_vis = top30.sort_values("att", ascending=False)
    for _, r in top30_vis.iterrows():
        rows.append({
            "panel": "B",
            "row_kind": "sankey_lr_top30",
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


# ====================================================================
def main():
    pred = load_data()
    top100 = pred.head(100).copy()
    top30  = pred.head(30).copy()
    top60  = pred.head(60).copy()

    print(f"Data: {len(pred)} edges")
    print("Top100 CT pair counts:")
    print(top100.groupby("CTPair").size().sort_values(ascending=False).to_string())

    fig = plt.figure(figsize=(34, 32), dpi=300, facecolor="white")
    gs = GridSpec(
        2, 2, figure=fig,
        width_ratios=[1.0, 1.15],
        height_ratios=[1, 1],
        hspace=0.26, wspace=0.20,
        left=0.04, right=0.97, top=0.96, bottom=0.05,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    draw_heatmap(ax_a, top100)

    ax_c = fig.add_subplot(gs[1, 0])
    draw_edge_bundling(ax_c, top60)

    ax_b = fig.add_subplot(gs[:, 1])
    draw_sankey(ax_b, top30)

    out_eps = OUT / f"{FIG_TYPE_ID}_main_figure_bc2.eps"
    out_png = OUT / f"{FIG_TYPE_ID}_main_figure_bc2.png"
    fig.savefig(out_eps, format="eps", bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, bbox_inches="tight", facecolor="white", dpi=300)
    plt.close(fig)
    print(f"\n[DONE] Saved: {out_png}")
    print(f"[DONE] Saved: {out_eps}")

    export_main_figure_bc2_genes_values(top100, top30, top60)


if __name__ == "__main__":
    main()
