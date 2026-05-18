"""
图类 01 — 主文综合图（Chord / Dot / Network），数据集 seqFISH。
Input : ../giotto_seqfish/4.Model_Training/pred_scores.csv  (same pattern as BC1)
Output: 01_main_figure_seqfish.png / .eps
        01_main_figure_seqfish1_ABC_genes_values.csv（A/B/C 与图一致；C 仅标注 hub）
"""

import sys
import pathlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.text as mtext
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import networkx as nx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import (
    CT_CHORD_ABBREV_SEQFISH,
    CT_COLORS_SEQFISH,
    FS_FIG8 as FS,
    apply_paper_style,
)

apply_paper_style(fs=FS)

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT.parent / "giotto_seqfish" / "4.Model_Training" / "pred_scores.csv"
OUT = ROOT
FIG_TYPE_ID = "01"


def load_data() -> pd.DataFrame:
    """Match BC1: read pred_scores.csv under dataset / 4.Model_Training."""
    pred = pd.read_csv(DATA)
    score_col = "att" if "att" in pred.columns else None
    if score_col is None and "score" in pred.columns:
        score_col = "score"
    if score_col is None:
        raise ValueError(f"{DATA} needs column 'att' or 'score'")
    pred = pred.rename(columns={score_col: "Score"})
    pred["Ligand"] = pred["Cell1"].astype(str).str.split("__").str[0]
    pred["Sender"] = pred["Cell1"].astype(str).str.split("__").str[1]
    pred["Receptor"] = pred["Cell2"].astype(str).str.split("__").str[0]
    pred["Receiver"] = pred["Cell2"].astype(str).str.split("__").str[1]
    pred["Score"] = pd.to_numeric(pred["Score"], errors="coerce").fillna(0.0)
    pred = pred.sort_values("Score", ascending=False).reset_index(drop=True)
    return pred[["Sender", "Receiver", "Ligand", "Receptor", "Score"]]


# ==========================================
# 1. 数据集（与 BC1 相同：repo 根目录下 giotto_seqfish/.../pred_scores.csv）
# ==========================================
df = load_data()


# ==========================================
# 2. 图 B：气泡图 (Bubble Plot) — 原始逻辑
# ==========================================
def draw_dotplot(ax, df_all, top_n=25):
    """Dot plot: top_n *unique* L-R pairs (by best score), all CT pairs shown."""
    df_all = df_all.copy()
    df_all["LRPair"] = df_all["Ligand"] + " - " + df_all["Receptor"]
    df_all["CTPair"] = df_all["Sender"] + " \u2192 " + df_all["Receiver"]

    # Select top_n unique LR pairs by their single best score
    top_lr = (
        df_all.groupby("LRPair")["Score"].max()
        .nlargest(top_n).index.tolist()
    )
    top = df_all[df_all["LRPair"].isin(top_lr)].copy()

    lr_order = (
        top.groupby("LRPair")["Score"].max()
        .sort_values(ascending=False).index.tolist()
    )
    lr_idx = {lr: i for i, lr in enumerate(lr_order)}

    ct_pairs_ordered = (
        top.groupby("CTPair")["Score"].sum()
        .sort_values(ascending=False).index.tolist()
    )
    cp_idx = {cp: i for i, cp in enumerate(ct_pairs_ordered)}

    score_min, score_max = top["Score"].min(), top["Score"].max()
    norm = Normalize(vmin=score_min, vmax=score_max)
    cmap = plt.cm.RdYlBu_r

    # Uniform marker size; score is encoded by color only.
    _S_DOT = 1000

    for _, row in top.iterrows():
        x = cp_idx[row["CTPair"]]
        y = len(lr_order) - 1 - lr_idx[row["LRPair"]]
        ax.scatter(
            x, y, s=_S_DOT,
            c=[cmap(norm(row["Score"]))],
            edgecolors="black", linewidths=0.9, zorder=3,
        )

    ax.set_xticks(range(len(ct_pairs_ordered)))
    ax.set_xticklabels(
        ct_pairs_ordered, rotation=90, ha="center", va="top",
        fontsize=FS.TICK, fontweight="bold",
    )
    ax.set_xlabel(
        "Cell Type Pairs (Sender \u2192 Receiver)",
        fontsize=FS.TITLE, fontweight="bold", labelpad=44,
    )
    ax.tick_params(axis="x", pad=6)

    ax.set_yticks(range(len(lr_order)))
    ytl = ax.set_yticklabels(
        list(reversed(lr_order)),
        fontsize=FS.TICK, fontweight="bold",
    )
    for t in ytl:
        t.set_ha("right")
    ax.set_ylabel(
        f"Top {top_n} L-R pairs",
        fontsize=FS.AXIS_LABEL, fontweight="bold", labelpad=22,
    )
    # 略往右、往上，减少与 y 刻度叠字
    ax.yaxis.set_label_coords(-0.26, 0.58)
    ax.tick_params(axis="y", pad=10)
    ax.set_title(
        "B. High-Confidence Spatial LR Pairs",
        fontsize=FS.TITLE, fontweight="bold", pad=60,
        loc="left", x=0.085,
    )
    ax.grid(True, linestyle="--", alpha=0.25, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax, shrink=0.45, pad=0.06, aspect=22)
    cbar.set_label("Attention Score",
                   fontsize=FS.LEGEND_TITLE, fontweight="bold")
    cbar.ax.tick_params(labelsize=FS.TICK)



# ==========================================
# 3. 图 C：网络图 (Network Plot) — 原始逻辑
# ==========================================
def plot_network_chart(ax, df):
    G = nx.Graph()

    for _, row in df.head(60).iterrows():
        sender_node = f"{row['Ligand']}\n({row['Sender']})"
        receiver_node = f"{row['Receptor']}\n({row['Receiver']})"
        G.add_node(sender_node, cell_type=row['Sender'])
        G.add_node(receiver_node, cell_type=row['Receiver'])
        G.add_edge(sender_node, receiver_node, weight=row['Score'])

    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 1

    unique_cells = sorted(set(df['Sender'].unique()) | set(df['Receiver'].unique()))
    color_map = {ct: CT_COLORS_SEQFISH.get(ct, "#888888") for ct in unique_cells}

    edges = G.edges(data=True)
    edge_widths = [d['weight'] * 15 for u, v, d in edges]

    pos = dict(nx.spring_layout(G, k=0.8, seed=42))

    # Nudge the degree-11 hub (e.g. TGFB1) downward to separate from overlapping neighbors.
    deg11 = [n for n in G.nodes() if degrees[n] == 11]
    if deg11:
        if len(deg11) == 1:
            n_down = deg11[0]
        else:
            xy_mean = np.mean([pos[n] for n in G.nodes()], axis=0)
            n_down = min(
                deg11,
                key=lambda nn: np.linalg.norm(np.asarray(pos[nn], dtype=float) - xy_mean),
            )
        x0, y0 = pos[n_down]
        pos[n_down] = (x0, y0 - 0.16)

    # Manual pixel-space nudges (layout coords ~ [-1, 1]).
    _GENE_NUDGES = {
        "ITGB1": (0.0, -0.14),      # down
        "COL4A2": (-0.14, 0.0),    # left
        "SEMA7A": (0.12, 0.12),    # upper-right
    }
    for n in G.nodes():
        gene_key = n.split("\n")[0].upper()
        if gene_key in _GENE_NUDGES:
            dx, dy = _GENE_NUDGES[gene_key]
            x0, y0 = pos[n]
            pos[n] = (x0 + dx, y0 + dy)

    nx.draw_networkx_edges(G, pos, ax=ax,
                           width=edge_widths, alpha=0.3, edge_color='grey')

    centrality = nx.degree_centrality(G)
    top_nodes = set(sorted(centrality, key=centrality.get, reverse=True)[:15])

    for n in G.nodes():
        deg = degrees[n]
        is_top = n in top_nodes
        s = (400 + 1200 * (deg / max_deg)) if is_top else (150 + 500 * (deg / max_deg))
        fc = color_map.get(G.nodes[n].get('cell_type', 'Unknown'), 'grey')
        ec = "gold" if is_top else "white"
        ew = 3.0 if is_top else 1.0
        ax.scatter(*pos[n], s=s, c=[fc], edgecolors=ec,
                   linewidths=ew, zorder=5, alpha=0.9)

    for n in top_nodes:
        deg = degrees[n]
        gene = n.split("\n")[0]
        ax.text(pos[n][0], pos[n][1], f"{gene}\n({deg})",
                ha="center", va="center", fontsize=FS.NODE_LABEL,
                fontweight="bold", color="#111111", zorder=6)

    ax.set_title('C. Key Molecular Hubs and Signaling Networks (Top 60)',
                 fontsize=FS.TITLE, pad=18, fontweight='bold')
    ax.axis('off')

    legend_handles = [mpatches.Patch(color=color_map[c], label=c)
                      for c in unique_cells]
    # 下移图例，避免与网络节点标签（如 DLL4/TGFB1）及 B 列视觉区重叠
    leg_c = ax.legend(
        handles=legend_handles, title="Cell Types",
        loc="upper center", bbox_to_anchor=(0.5, -0.20),
        ncol=3, frameon=False,
        fontsize=FS.LEGEND, title_fontsize=FS.LEGEND_TITLE,
    )
    leg_c.get_title().set_fontweight("bold")
    for t in leg_c.get_texts():
        t.set_fontweight("bold")


# ==========================================
# 4. Chord diagram generation (inline, no external PNG)
# ==========================================
def build_chord_circos_seqfish(df):
    """top-100 聚合流量 → Circos（不写栅格图，供极坐标子图纯矢量绘制）。

    外圈已用缩写，与 initialize_from_matrix 相同单半径标签即可，不再多半径分层。
    """
    from pycirclize import Circos

    top100 = df.head(100)
    flow = top100.groupby(["Sender", "Receiver"])["Score"].sum().reset_index()
    all_cts = sorted(set(flow["Sender"]) | set(flow["Receiver"]))
    matrix_df = pd.DataFrame(0.0, index=all_cts, columns=all_cts)
    for _, r in flow.iterrows():
        matrix_df.loc[r["Sender"], r["Receiver"]] += r["Score"]

    color_dict_full = {ct: CT_COLORS_SEQFISH.get(ct, "#999999") for ct in all_cts}
    _ab = {ct: CT_CHORD_ABBREV_SEQFISH.get(ct, ct) for ct in all_cts}
    matrix_df = matrix_df.rename(index=_ab, columns=_ab)
    color_dict = {_ab[ct]: color_dict_full[ct] for ct in all_cts}

    circos = Circos.initialize_from_matrix(
        matrix_df,
        space=12,
        cmap=color_dict,
        label_kws=dict(size=max(30.0, float(FS.TICK)), weight="bold"),
        link_kws=dict(direction=1, ec="black", lw=0.5),
        r_lim=(88, 100),
    )
    return circos, flow


def place_strongest_flow_between_a_c_seqfish(fig, ax_a, ax_c, flow):
    """Strongest flow 放在 A/C 竖缝中间；细胞类型用全称（与弦图外圈缩写区分）。"""
    best = flow.loc[flow["Score"].idxmax()]
    txt = f"Strongest flow: {best['Sender']} \u2192 {best['Receiver']}"
    pa = ax_a.get_position()
    pc = ax_c.get_position()
    x = pa.x0 + pa.width / 2
    y = (pa.y0 + pc.y1) / 2
    fig.text(
        x, y, txt,
        transform=fig.transFigure,
        ha="center", va="center",
        fontsize=max(36.0, float(FS.SUBTITLE)),
        fontstyle="italic", fontweight="bold",
    )


def _unclip_chord_outside_text(ax_polar):
    """弦图扇区名在极坐标轴外，默认会被轴域裁切；关闭文字 clip 以显示完整。"""
    for art in ax_polar.get_children():
        if isinstance(art, mtext.Text):
            art.set_clip_on(False)


def export_main_figure_seqfish_genes_values(df, flow_chord, top_n_dotplot=25):
    """导出 A 弦图细胞型对、B 点图 top_n L-R、C 网络图标注 hub（top15）；列与 BC1 CSV 一致，数值列统一为 att。"""
    path = OUT / f"{FIG_TYPE_ID}_main_figure_seqfish1_ABC_genes_values.csv"
    rows = []

    for _, r in flow_chord.iterrows():
        rows.append({
            "panel": "A",
            "row_kind": "chord_flow_top100",
            "sender_ct": r["Sender"],
            "receiver_ct": r["Receiver"],
            "ligand": np.nan,
            "receptor": np.nan,
            "gene_symbol": np.nan,
            "gene_cell_type": np.nan,
            "att": float(r["Score"]),
            "degree_in_top60_subgraph": np.nan,
            "degree_centrality": np.nan,
            "is_hub_labeled": np.nan,
        })

    df_b = df.copy()
    df_b["LRPair"] = df_b["Ligand"] + " - " + df_b["Receptor"]
    top_lr = (
        df_b.groupby("LRPair")["Score"].max()
        .nlargest(top_n_dotplot).index.tolist()
    )
    top_df = df_b[df_b["LRPair"].isin(top_lr)].copy()
    for _, r in top_df.iterrows():
        rows.append({
            "panel": "B",
            "row_kind": "dotplot_lr",
            "sender_ct": r["Sender"],
            "receiver_ct": r["Receiver"],
            "ligand": r["Ligand"],
            "receptor": r["Receptor"],
            "gene_symbol": np.nan,
            "gene_cell_type": np.nan,
            "att": float(r["Score"]),
            "degree_in_top60_subgraph": np.nan,
            "degree_centrality": np.nan,
            "is_hub_labeled": np.nan,
        })

    # C：与 plot_network_chart 相同 — top60 边、图上 top15 度中心性标注节点
    top60 = df.head(60).copy()
    G = nx.Graph()
    for _, row in top60.iterrows():
        sender_node = f"{row['Ligand']}\n({row['Sender']})"
        receiver_node = f"{row['Receptor']}\n({row['Receiver']})"
        G.add_node(sender_node, cell_type=row["Sender"])
        G.add_node(receiver_node, cell_type=row["Receiver"])
        G.add_edge(sender_node, receiver_node, weight=row["Score"])

    degrees = dict(G.degree())
    centrality = nx.degree_centrality(G)
    hub_nodes = set(sorted(centrality, key=centrality.get, reverse=True)[:15])
    hub_ordered = sorted(hub_nodes, key=lambda n: centrality.get(n, 0.0), reverse=True)

    for n in hub_ordered:
        gene = n.split("\n")[0]
        ct = G.nodes[n].get("cell_type", "")
        deg = int(degrees.get(n, 0))
        rows.append({
            "panel": "C",
            "row_kind": "network_node_labeled",
            "sender_ct": np.nan,
            "receiver_ct": np.nan,
            "ligand": np.nan,
            "receptor": np.nan,
            "gene_symbol": gene,
            "gene_cell_type": ct,
            "att": np.nan,
            "degree_in_top60_subgraph": deg,
            "degree_centrality": float(centrality.get(n, 0.0)),
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


# ==========================================
# 5. Combined figure: A(chord) + B(dotplot) + C(network)
# ==========================================
def main():
    print("Building figure (A: vector chord on polar axes) ...")
    circos_a, flow_chord = build_chord_circos_seqfish(df)

    # seqFISH has many CT pairs on x-axis: widen B column and leave room for vertical labels.
    fig = plt.figure(figsize=(48, 34), dpi=300, facecolor="white")
    gs = GridSpec(
        2, 2, figure=fig,
        width_ratios=[1.0, 2.15],
        height_ratios=[1, 1],
        hspace=0.38, wspace=0.44,
        left=0.065, right=0.965, top=0.91, bottom=0.32,
    )

    # A: Chord diagram (top-left) — 极坐标矢量
    ax_a = fig.add_subplot(gs[0, 0], projection="polar")
    circos_a.plotfig(ax=ax_a, dpi=300)
    _unclip_chord_outside_text(ax_a)
    ax_a.set_title(
        "A. Cell-Cell Communication Global Strength",
        fontsize=FS.TITLE, fontweight="bold", pad=14,
    )

    # B: Dot plot (right, full height)
    ax_b = fig.add_subplot(gs[:, 1])
    draw_dotplot(ax_b, df)

    # C: 网络图 (bottom-left)
    ax_c = fig.add_subplot(gs[1, 0])
    plot_network_chart(ax_c, df)
    place_strongest_flow_between_a_c_seqfish(fig, ax_a, ax_c, flow_chord)

    out_png = OUT / f"{FIG_TYPE_ID}_main_figure_seqfish1.png"
    out_eps = OUT / f"{FIG_TYPE_ID}_main_figure_seqfish1.eps"
    fig.savefig(
        out_png, bbox_inches="tight", facecolor="white", dpi=300, pad_inches=0.38
    )
    fig.savefig(
        out_eps, format="eps", bbox_inches="tight", facecolor="white", pad_inches=0.22
    )
    plt.close(fig)
    print(f"[DONE] {out_png}")
    print(f"[DONE] {out_eps}")

    export_main_figure_seqfish_genes_values(df, flow_chord, top_n_dotplot=25)


if __name__ == "__main__":
    main()
