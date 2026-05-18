"""
图类 04 — 六格合并图（GRN / 扰动 / 阈值等），数据集 BC1。
BreastCancer1 (BC1) 矢量版 — 将 6 个 panel 合并为真正的矢量 EPS。

原版 combine_panels() 经由 PNG/imshow 光栅化，导致 EPS 并非矢量。
本文件使用 matplotlib subfigures，直接将所有元素以矢量方式绘制到子图，
不再经过任何 PNG 中间步骤，保证最终 EPS 为真正矢量图。

输出：与本脚本同目录 — 04_combined_panels_bc1.eps / .png
"""

import matplotlib
matplotlib.use("Agg")

import pathlib
import numpy as np
import pandas as pd
import scipy.io as sio
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import patheffects
from collections import deque
import random

matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['axes.unicode_minus'] = False

# ── 路径：数据在仓库根下，矢量图输出在本脚本所在目录（plot/）────────────────
PLOT_DIR = pathlib.Path(__file__).resolve().parent
BASE = PLOT_DIR.parent
LR_DIR      = BASE / 'BreastCancer1' / '2.LR_Screening'
ENTROPY_DIR = LR_DIR / '3.Identify sensitive genes and gene combinations' / \
              '1.Initial network entropy index calculation'
DISTURB_DIR = LR_DIR / '3.Identify sensitive genes and gene combinations' / \
              '2.Disturbance handling'
SUBNET_DIR  = LR_DIR / '3.Identify sensitive genes and gene combinations' / \
              '3.Subnetwork exploration'
DGRN_DIR = PLOT_DIR / 'dgrn-se'
DGRN_DIR.mkdir(parents=True, exist_ok=True)
FIG_TYPE_ID = "04"

CT_COLORS = {
    "Bcell": "#4e79a7", "Macrophage": "#e15759", "Stroma": "#76b7b2",
    "Endothelial": "#59a14f", "Tcell": "#f28e2b", "Malignant": "#9c755f",
}
THRESHOLD = 0.26
# 与 threshold_scan.py 输出一致（敏感基因列表）
_NEW_NAME_L_MAT = f"new_name-L{THRESHOLD:.2f}.mat"
_NEW_NAME_R_MAT = f"new_name-R{THRESHOLD:.2f}.mat"


# ============================================================================
# Panel A — GRN hub-burst 网络（直接绘制到 subfig）
# ============================================================================
def _draw_A(subfig, top_k: int = 10):
    print("[A] Building GRN network …")
    prewavelet    = sio.loadmat(str(ENTROPY_DIR / 'prewavelet-L.mat'))
    new_name_data = sio.loadmat(str(SUBNET_DIR / _NEW_NAME_L_MAT))
    adj_df        = pd.read_excel(
        str(LR_DIR / '2.Build a gene network' / 'L-averaged_final_adj-.xlsx'),
        header=None)
    init_idx  = sio.loadmat(str(ENTROPY_DIR / 'init_indext_wavelet-L.mat'))
    H_init    = np.array(init_idx['H'], dtype=float)
    entropy_all = np.nanmean(H_init, axis=1)

    all_names = [str(prewavelet['name'][i, 0][0])
                 for i in range(prewavelet['name'].shape[0])]
    sens_set  = set(str(new_name_data['new_name'][i, 0][0])
                    for i in range(new_name_data['new_name'].shape[0]))
    adj_matrix = adj_df.values.astype(int)

    G = nx.Graph()
    for i, name in enumerate(all_names):
        G.add_node(i, label=name.split('__')[0], full_name=name,
                   is_sensitive=name in sens_set,
                   entropy=float(entropy_all[i]) if i < len(entropy_all) else 0.0)
    n = len(all_names)
    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i, j] == 1:
                G.add_edge(i, j)

    G.remove_nodes_from([v for v in G.nodes() if G.degree(v) == 0])
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    if len(comps) > 1:
        G = G.subgraph(comps[0]).copy()

    sens_nodes = [v for v in G.nodes() if G.nodes[v]['is_sensitive']]
    top_k = max(1, min(top_k, len(sens_nodes)))

    def bfs_dist(src):
        d = {src: 0}; q = deque([src])
        while q:
            u = q.popleft()
            for w in G.neighbors(u):
                if w not in d:
                    d[w] = d[u] + 1; q.append(w)
        return d

    def hub_score(v):
        return G.degree(v) * max(G.nodes[v]['entropy'], 1e-3)

    ranked = sorted(sens_nodes, key=lambda v: -hub_score(v))
    hubs = [ranked[0]]; hub_bfs = {hubs[0]: bfs_dist(hubs[0])}
    while len(hubs) < top_k:
        best_v, best_key = None, None
        for v in sens_nodes:
            if v in hubs: continue
            min_d = min(hub_bfs[h].get(v, 10**9) for h in hubs)
            key = (min_d, hub_score(v))
            if best_key is None or key > best_key:
                best_key, best_v = key, v
        if best_v is None: break
        hubs.append(best_v); hub_bfs[best_v] = bfs_dist(best_v)

    hub_set = set(hubs)
    n_total = G.number_of_nodes()
    target_size = max(2.0, (n_total - len(hubs)) / len(hubs) + 1.0)
    hub_nbrs   = {h: set(G.neighbors(h)) for h in hubs}
    cell_members = {h: {h} for h in hubs}
    non_hub_nodes = sorted([v for v in G.nodes() if v not in hub_set],
                           key=lambda v: -G.degree(v))
    node_hub = {h: h for h in hubs}
    for v in non_hub_nodes:
        v_nbrs = set(G.neighbors(v)); best_h, best_score = hubs[0], None
        for h in hubs:
            d = hub_bfs[h].get(v, 10**9)
            shared = len(v_nbrs & hub_nbrs[h])
            overfill = max(0.0, len(cell_members[h]) - 1 - target_size + 1.0)
            score = d * 10.0 - shared * 1.0 + 0.40 * overfill
            if best_score is None or score < best_score:
                best_score, best_h = score, h
        node_hub[v] = best_h; cell_members[best_h].add(v)

    rng = random.Random(7)
    tree_edges = set()
    for h in hubs:
        cell_m = {v for v, hh in node_hub.items() if hh == h}
        parent = {h: None}; q = deque([h])
        while q:
            u = q.popleft()
            for w in G.neighbors(u):
                if w in cell_m and w not in parent:
                    parent[w] = u; q.append(w)
        for v, p in parent.items():
            if p is not None: tree_edges.add(tuple(sorted((v, p))))

    intra_extra, cross_edges = [], []
    for u, v in G.edges():
        e = tuple(sorted((u, v)))
        if e in tree_edges: continue
        if node_hub[u] == node_hub[v]:
            if rng.random() < 0.04: intra_extra.append(e)
        else:
            if rng.random() < 0.025: cross_edges.append(e)

    G_vis = nx.Graph()
    G_vis.add_nodes_from(G.nodes(data=True))
    G_vis.add_edges_from(tree_edges)
    G_vis.add_edges_from(intra_extra)
    G_vis.add_edges_from(cross_edges)

    n_hubs = len(hubs)
    if n_hubs == 1:
        pin_coords = np.array([[0.0, 0.0]])
    elif n_hubs == 2:
        pin_coords = np.array([[-0.35, 0.0], [0.35, 0.0]])
    else:
        R_hub = 0.45 if n_hubs <= 4 else 0.55
        angles = np.linspace(np.pi/2, np.pi/2 - 2*np.pi*(n_hubs-1)/n_hubs, n_hubs)
        pin_coords = np.column_stack([R_hub*np.cos(angles), R_hub*np.sin(angles)])
    pin_pos = {h: np.array(pin_coords[i], dtype=float) for i, h in enumerate(hubs)}

    rng_np = np.random.default_rng(7)
    init_pos = dict(pin_pos)
    for v in G_vis.nodes():
        if v in init_pos: continue
        hp  = pin_pos[node_hub[v]]
        ang = rng_np.uniform(0, 2*np.pi)
        r   = 0.35 + 0.65 * rng_np.random()
        init_pos[v] = hp + r * np.array([np.cos(ang), np.sin(ang)])

    pos = nx.spring_layout(G_vis, pos=init_pos, fixed=hubs,
                           k=1.6/np.sqrt(max(G_vis.number_of_nodes(), 2)),
                           iterations=600, seed=7, scale=None)
    for h in hubs: pos[h] = pin_pos[h]

    hub_palette = ['#D62728','#2CA02C','#1F77B4','#7F7F7F','#9467BD',
                   '#FF7F0E','#17BECF','#8C564B','#E377C2','#E6B800']
    hub_color = {h: hub_palette[i % len(hub_palette)] for i, h in enumerate(hubs)}

    ent_arr = np.array([G.nodes[v]['entropy'] for v in G.nodes()])
    e_min, e_max = float(ent_arr.min()), float(ent_arr.max())
    e_span = (e_max - e_min) or 1.0
    def size_of(v):
        t = (G.nodes[v]['entropy'] - e_min) / e_span
        if v in hub_set: return 900 + t * 1600
        if G.nodes[v]['is_sensitive']: return 90 + t * 220
        return 18 + t * 95

    subfig.set_facecolor('white')
    ax = subfig.add_subplot(1, 1, 1)

    intra_by_color = {}
    for u, v in list(tree_edges) + intra_extra:
        hu, hv = node_hub[u], node_hub[v]
        if hu != hv: continue
        intra_by_color.setdefault(hub_color[hu], []).append((u, v))

    nx.draw_networkx_edges(G_vis, pos, edgelist=cross_edges, ax=ax,
                           edge_color='#B5B9C0', width=0.45, alpha=0.55)
    for color, elist in intra_by_color.items():
        nx.draw_networkx_edges(G_vis, pos, edgelist=elist, ax=ax,
                               edge_color=[color], width=0.9, alpha=0.72)

    non_sens = [v for v in G.nodes() if v not in hub_set and not G.nodes[v]['is_sensitive']]
    nx.draw_networkx_nodes(G, pos, nodelist=non_sens, ax=ax,
                           node_color='white', node_shape='o',
                           node_size=[size_of(v) for v in non_sens],
                           edgecolors='#555a63', linewidths=0.5, alpha=0.95)

    other_sens = [v for v in sens_nodes if v not in hub_set]
    if other_sens:
        nx.draw_networkx_nodes(G, pos, nodelist=other_sens, ax=ax,
                               node_color=[hub_color[node_hub[v]] for v in other_sens],
                               node_shape='o', node_size=[size_of(v) for v in other_sens],
                               edgecolors='white', linewidths=0.9, alpha=0.92)

    nx.draw_networkx_nodes(G, pos, nodelist=hubs, ax=ax,
                           node_color=[hub_color[h] for h in hubs],
                           node_shape='o', node_size=[size_of(h) for h in hubs],
                           edgecolors='white', linewidths=3.2, alpha=1.0)

    hub_stroke = [patheffects.withStroke(linewidth=4.5, foreground='white')]
    small_labels = {v: G.nodes[v]['label'] for v in G.nodes() if v not in hub_set}
    lp = {v: (pos[v][0], pos[v][1] + 0.018) for v in small_labels}
    nx.draw_networkx_labels(G, lp, small_labels, ax=ax,
                            font_size=6, font_color='#333333', font_family='Arial')

    for h in hubs:
        x, y = pos[h]
        ax.text(x, y, G.nodes[h]['label'], fontsize=26, fontweight='bold',
                color=hub_color[h], ha='center', va='center',
                fontfamily='Arial', path_effects=hub_stroke, zorder=12)

    legend_handles = [mpatches.Patch(facecolor=hub_color[h], edgecolor='white',
                                     label=G.nodes[h]['label']) for h in hubs]
    ax.legend(handles=legend_handles, loc='lower right', ncol=1, fontsize=17,
              frameon=True, framealpha=0.9, edgecolor='#B5B9C0',
              title='Hub (sensitive gene)', title_fontsize=17)

    ax.axis('off'); ax.set_aspect('equal')
    xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
    pad = 0.08
    x_lo, x_hi = min(xs)-pad, max(xs)+pad
    y_lo, y_hi = min(ys)-pad, max(ys)+pad
    tr = 18/15
    w, h = x_hi-x_lo, y_hi-y_lo
    if w/h < tr:
        extra = (tr*h - w)/2; x_lo -= extra; x_hi += extra
    else:
        extra = (w/tr - h)/2; y_lo -= extra; y_hi += extra
    ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
    subfig.text(0.01, 0.97, 'A', transform=subfig.transSubfigure,
                fontsize=44, fontweight='bold', fontfamily='Arial', va='top', ha='left')
    print("  Panel A done.")


# ============================================================================
# Panel B — 预测误差均值折线
# ============================================================================
def _draw_B(subfig):
    print("[B] Plotting mean prediction error …")
    d  = sio.loadmat(str(ENTROPY_DIR / 'error-L.mat'))
    pw = sio.loadmat(str(ENTROPY_DIR / 'prewavelet-L.mat'))

    delta = d['delta_record'].T
    n_gene, n_time = delta.shape
    names = [str(pw['name'][i, 0][0]) for i in range(pw['name'].shape[0])]
    assert len(names) == n_gene

    delta_clean = delta.copy(); delta_clean[delta_clean > 10] = 0
    t = np.arange(1, n_time + 1)
    mean_t = np.nanmean(delta_clean, axis=0)
    std_t  = np.nanstd(delta_clean,  axis=0)

    subfig.set_facecolor('white')
    ax = subfig.add_subplot(1, 1, 1)
    ax.fill_between(t, mean_t - std_t, mean_t + std_t, alpha=0.25, color='#E53935')
    ax.plot(t, mean_t, color='#E53935', linewidth=1.5)
    ax.set_xlabel('Time window', fontsize=22)
    ax.set_ylabel('Mean Δ', fontsize=22)
    ax.set_title('Mean prediction error across all genes  —  L ligands',
                 fontsize=24, fontweight='bold', pad=12)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=18)
    subfig.text(0.01, 0.97, 'B', transform=subfig.transSubfigure,
                fontsize=44, fontweight='bold', fontfamily='Arial', va='top', ha='left')
    print("  Panel B done.")


# ============================================================================
# Panel C — 蝴蝶图（使用专用中间坐标轴，无需 canvas.draw() + 坐标变换）
# ============================================================================
def _draw_C(subfig):
    print("[C] Plotting butterfly chart …")
    pre  = sio.loadmat(str(DISTURB_DIR / 'prewavelet-L.mat'))
    addu = sio.loadmat(str(DISTURB_DIR / 'adduwavelet-100-(all)-L-merged.mat'))

    names = [str(pre["name"][i, 0][0]) for i in range(pre["name"].shape[0])]
    H = addu["H_point_value"]

    n_show = 30
    gene_names = names[:n_show]
    X_left  = H[:n_show, 0:3]
    X_right = H[:n_show, 3:6]

    CT_ABBREV = {"Bcell": "Bce", "Macrophage": "Mac", "Malignant": "Mal",
                 "Stroma": "Str", "Endothelial": "End", "Tcell": "Tce"}

    def shorten_label(full_name):
        parts = full_name.split("__", 1)
        if len(parts) == 2:
            gene, ct = parts
            return f"{gene}_{CT_ABBREV.get(ct, ct[:3])}"
        return full_name

    labels = [shorten_label(n) for n in gene_names]
    C_left  = [[0.647,0.000,0.149],[0.843,0.200,0.153],[0.961,0.459,0.278]]
    C_right = [[0.220,0.298,0.624],[0.345,0.549,0.749],[0.557,0.757,0.867]]

    subfig.set_facecolor('white')
    # 三列布局：左侧条形 | 中间基因标签 | 右侧条形
    ax_l = subfig.add_axes([0.03, 0.10, 0.36, 0.76])
    ax_c = subfig.add_axes([0.39, 0.10, 0.20, 0.76])   # 中间标签轴
    ax_r = subfig.add_axes([0.59, 0.10, 0.36, 0.76])

    Y = np.arange(n_show)
    left_bottom = np.zeros(n_show)
    for j in range(3):
        ax_l.barh(Y, X_left[:, j], left=left_bottom, height=0.6,
                  color=C_left[j], edgecolor="none")
        left_bottom += X_left[:, j]

    right_bottom = np.zeros(n_show)
    for j in range(3):
        ax_r.barh(Y, X_right[:, j], left=right_bottom, height=0.6,
                  color=C_right[j], edgecolor="none")
        right_bottom += X_right[:, j]

    ax_l.set_yticks(Y); ax_l.set_yticklabels([])
    ax_l.invert_xaxis(); ax_l.invert_yaxis()
    ax_l.set_xlim(3, 0)
    ax_l.spines["top"].set_visible(False); ax_l.spines["right"].set_visible(False)
    ax_l.tick_params(axis="y", length=0); ax_l.tick_params(axis="x", labelsize=24)
    ax_l.grid(axis="x", linestyle="-", alpha=0.15)

    ax_r.set_yticks(Y); ax_r.set_yticklabels([])
    ax_r.set_ylim(ax_l.get_ylim()); ax_r.set_xlim(0, 2.5)
    ax_r.spines["top"].set_visible(False); ax_r.spines["left"].set_visible(False)
    ax_r.tick_params(axis="y", length=0); ax_r.tick_params(axis="x", labelsize=24)
    ax_r.grid(axis="x", linestyle="-", alpha=0.15)
    ax_r.yaxis.set_ticks_position("none")

    # 中间轴：直接用数据坐标放置标签，无需 canvas 变换
    ax_c.axis('off')
    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(-0.5, n_show - 0.5)
    ax_c.invert_yaxis()
    # 与 run() 中 fig 高 27" 三行均分、add_axes 高 0.76 子图高一致；字号 ≤ 行高，避免纵重叠
    _row_h_in = (27.0 / 3.0) * 0.76
    _c_fs = max(9.0, min(13.0, _row_h_in * 72.0 / n_show * 0.88))
    for i, lbl in enumerate(labels):
        ax_c.text(0.5, i, lbl, ha='center', va='center',
                  fontsize=_c_fs, fontfamily='Arial', color='k')

    legend_l = [mpatches.Patch(color=C_left[j],  label=f"-{30-j*5}%") for j in range(3)]
    legend_r = [mpatches.Patch(color=C_right[j], label=f"{30-j*5}%")  for j in range(3)]
    ax_l.legend(handles=legend_l, loc="upper center", bbox_to_anchor=(0.5, 1.08),
                ncol=3, frameon=False, fontsize=18,
                handlelength=1.2, handleheight=0.7, handletextpad=0.4)
    ax_r.legend(handles=legend_r, loc="upper center", bbox_to_anchor=(0.5, 1.08),
                ncol=3, frameon=False, fontsize=18,
                handlelength=1.2, handleheight=0.7, handletextpad=0.4)

    subfig.text(0.01, 0.97, 'C', transform=subfig.transSubfigure,
                fontsize=44, fontweight='bold', fontfamily='Arial', va='top', ha='left')
    print("  Panel C done.")


# ============================================================================
# Panel D — 阈值敏感性曲线
# ============================================================================
def _draw_D(subfig):
    print("[D] Plotting threshold sensitivity …")

    def load_names(mat_path):
        mat = sio.loadmat(str(mat_path))
        for key in ["new_name", "name"]:
            if key in mat:
                arr = mat[key].squeeze()
                return [str(x[0]) if hasattr(x, "__len__") else str(x) for x in arr]
        return []

    def threshold_scan(layer):
        hp  = sio.loadmat(str(SUBNET_DIR / f"adduwavelet-100-(all)-{layer}-merged.mat"))
        pre = sio.loadmat(str(SUBNET_DIR / f"prewavelet-{layer}.mat"))
        H      = np.array(hp["H_point_value"])
        maprho = np.array(pre["maprho"])
        thresholds = np.round(np.arange(0.20, 0.401, 0.003), 3)
        counts = []
        for t in thresholds:
            idx = np.where(np.mean(H, axis=1) < t)[0]
            sub = maprho[np.ix_(idx, idx)]
            counts.append(int(sub.any(axis=1).sum()))
        return thresholds, np.array(counts)

    names_L = load_names(SUBNET_DIR / _NEW_NAME_L_MAT)
    names_R = load_names(SUBNET_DIR / _NEW_NAME_R_MAT)
    thresh_L, counts_L = threshold_scan("L")
    thresh_R, counts_R = threshold_scan("R")

    subfig.set_facecolor('white')
    ax = subfig.add_subplot(1, 1, 1)
    ax.plot(thresh_L, counts_L, "o-", color="#4e79a7", linewidth=2, markersize=3,
            label=f"L-layer / Ligand (n={len(names_L)} @ {THRESHOLD})")
    ax.plot(thresh_R, counts_R, "s-", color="#e15759", linewidth=2, markersize=3,
            label=f"R-layer / Receptor (n={len(names_R)} @ {THRESHOLD})")
    ax.fill_between(thresh_L, counts_L, alpha=0.06, color="#4e79a7")
    ax.fill_between(thresh_R, counts_R, alpha=0.06, color="#e15759")

    y_max = max(counts_L.max(), counts_R.max())
    ax.axvline(THRESHOLD, color="#333333", linestyle="--", linewidth=1.4, alpha=0.7)
    ax.annotate(f"threshold = {THRESHOLD}", xy=(THRESHOLD, y_max + 3),
                fontsize=18, fontweight="bold", color="#333333",
                ha="center", va="bottom")

    yL = counts_L[np.argmin(np.abs(thresh_L - THRESHOLD))]
    yR = counts_R[np.argmin(np.abs(thresh_R - THRESHOLD))]
    ax.plot(THRESHOLD, yL, "o", color="#4e79a7", markersize=8, zorder=5,
            markeredgecolor="white", markeredgewidth=1.5)
    ax.plot(THRESHOLD, yR, "s", color="#e15759", markersize=8, zorder=5,
            markeredgecolor="white", markeredgewidth=1.5)
    ax.annotate(f"  {yL}", xy=(THRESHOLD, yL), fontsize=16, color="#4e79a7",
                fontweight="bold", va="center")
    ax.annotate(f"  {yR}", xy=(THRESHOLD + 0.005, yR), fontsize=16, color="#e15759",
                fontweight="bold", va="center")
    ax.axhspan(yL - 0.5, yL + 0.5, color="#4e79a7", alpha=0.08)
    ax.axhspan(yR - 0.5, yR + 0.5, color="#e15759", alpha=0.08)

    ax.set_xlabel("Sensitivity threshold", fontsize=22)
    ax.set_ylabel("Number of retained genes", fontsize=22)
    ax.set_title("Threshold sensitivity (real data)",
                 fontsize=24, fontweight="bold", pad=12)
    ax.legend(fontsize=20, loc="center right", framealpha=0.9)
    ax.tick_params(labelsize=18)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_ylim(30, y_max + 12)
    subfig.text(0.01, 0.97, 'D', transform=subfig.transSubfigure,
                fontsize=44, fontweight='bold', fontfamily='Arial', va='top', ha='left')
    print("  Panel D done.")


# ============================================================================
# Panel E — L/R 敏感基因数量按细胞类型柱状图
# ============================================================================
def _load_sensitive_names(mat_path):
    m = sio.loadmat(str(mat_path))
    arr = m['new_name']
    return [str(arr[i, 0][0]) for i in range(arr.shape[0])]


def _count_by_ct(names):
    counts = {}
    for n in names:
        parts = n.split('__', 1)
        ct = parts[1] if len(parts) == 2 else ''
        counts[ct] = counts.get(ct, 0) + 1
    return counts


def _draw_E(subfig):
    print("[E] Plotting gene count bars …")
    names_L = _load_sensitive_names(SUBNET_DIR / _NEW_NAME_L_MAT)
    names_R = _load_sensitive_names(SUBNET_DIR / _NEW_NAME_R_MAT)
    cnt_L = _count_by_ct(names_L); cnt_R = _count_by_ct(names_R)

    preferred = ["Bcell", "Tcell", "Macrophage", "Endothelial", "Stroma", "Malignant"]
    cts = [c for c in preferred if c in cnt_L or c in cnt_R]
    cts += sorted(set(list(cnt_L) + list(cnt_R)) - set(cts))

    L_vals = [cnt_L.get(c, 0) for c in cts]
    R_vals = [cnt_R.get(c, 0) for c in cts]
    colors = [CT_COLORS.get(c, '#888888') for c in cts]

    subfig.set_facecolor('white')
    ax = subfig.add_subplot(1, 1, 1)
    x = np.arange(len(cts)); w = 0.38
    bars_L = ax.bar(x - w/2, L_vals, w, color=colors, edgecolor='white', linewidth=1.0)
    bars_R = ax.bar(x + w/2, R_vals, w, color=colors, edgecolor='white', linewidth=1.0,
                    hatch='//', alpha=0.85)
    for bars, vals in [(bars_L, L_vals), (bars_R, R_vals)]:
        for b, v in zip(bars, vals):
            if v > 0:
                ax.text(b.get_x() + b.get_width()/2, v + 0.5, str(v),
                        ha='center', va='bottom', fontsize=15,
                        fontweight='bold', color='#333333')
    ax.set_xticks(x); ax.set_xticklabels(cts, fontsize=17)
    ax.set_ylabel('Number of sensitive genes', fontsize=20)
    ax.set_title(f'Sensitive gene counts by cell type  '
                 f'(L: {len(names_L)},  R: {len(names_R)})',
                 fontsize=22, fontweight='bold', pad=12)
    ax.tick_params(axis='y', labelsize=16)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_ylim(0, max(max(L_vals), max(R_vals)) * 1.18)
    legend_handles = [
        mpatches.Patch(facecolor='#bbbbbb', edgecolor='white', label='L (ligand)'),
        mpatches.Patch(facecolor='#bbbbbb', edgecolor='white', hatch='//', label='R (receptor)'),
    ]
    ax.legend(handles=legend_handles, fontsize=16, loc='upper right', framealpha=0.9)
    subfig.text(0.01, 0.97, 'E', transform=subfig.transSubfigure,
                fontsize=44, fontweight='bold', fontfamily='Arial', va='top', ha='left')
    print("  Panel E done.")


# ============================================================================
# Panel F — L/R 敏感基因平均表达量折线
# ============================================================================
def _load_prewavelet(mat_path):
    m = sio.loadmat(str(mat_path))
    names = [str(m['name'][i, 0][0]) for i in range(m['name'].shape[0])]
    return names, np.array(m['Rec_time'])


def _draw_F(subfig):
    print("[F] Plotting expression curves …")
    names_L_all, Rec_L = _load_prewavelet(ENTROPY_DIR / 'prewavelet-L.mat')
    names_R_all, Rec_R = _load_prewavelet(ENTROPY_DIR / 'prewavelet-R.mat')
    sens_L = set(_load_sensitive_names(SUBNET_DIR / _NEW_NAME_L_MAT))
    sens_R = set(_load_sensitive_names(SUBNET_DIR / _NEW_NAME_R_MAT))

    idx_L = [i for i, n in enumerate(names_L_all) if n in sens_L]
    idx_R = [i for i, n in enumerate(names_R_all) if n in sens_R]
    expr_L = Rec_L[idx_L, :]; expr_R = Rec_R[idx_R, :]

    t = np.arange(1, expr_L.shape[1] + 1)
    mean_L = np.nanmean(expr_L, axis=0); std_L = np.nanstd(expr_L, axis=0)
    mean_R = np.nanmean(expr_R, axis=0); std_R = np.nanstd(expr_R, axis=0)

    subfig.set_facecolor('white')
    ax_L, ax_R = subfig.subplots(1, 2)

    for color, ax, expr, mean, std, n in [
        ('#4e79a7', ax_L, expr_L, mean_L, std_L, len(idx_L)),
        ('#e15759', ax_R, expr_R, mean_R, std_R, len(idx_R)),
    ]:
        for i in range(expr.shape[0]):
            ax.plot(t, expr[i], color=color, alpha=0.12, linewidth=0.6)
        ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.25, label='±1 std')
        ax.plot(t, mean, color=color, linewidth=2.4, label=f'Mean (n={n})')
        ax.set_xlabel('Time window', fontsize=18); ax.set_ylabel('Expression', fontsize=18)
        ax.tick_params(labelsize=15)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.25)
        ax.legend(fontsize=15, loc='best', framealpha=0.9)

    for ax, title in zip([ax_L, ax_R], ['L-layer', 'R-layer']):
        ax.set_title(title, fontsize=20, fontweight='bold', pad=10)

    subfig.text(0.01, 0.97, 'F', transform=subfig.transSubfigure,
                fontsize=44, fontweight='bold', fontfamily='Arial', va='top', ha='left')
    print("  Panel F done.")


# ============================================================================
# 矢量合并主函数
# ============================================================================
def run():
    print("\n=== Generating fully-vector combined figure (BC1) ===")
    fig = plt.figure(figsize=(22, 27), dpi=300, facecolor='white')
    sfs = fig.subfigures(3, 2, wspace=0.04, hspace=0.06)

    _draw_A(sfs[0, 0])
    _draw_B(sfs[0, 1])
    _draw_C(sfs[1, 0])
    _draw_D(sfs[1, 1])
    _draw_E(sfs[2, 0])
    _draw_F(sfs[2, 1])

    out_eps = PLOT_DIR / f'{FIG_TYPE_ID}_combined_panels_bc1.eps'
    out_png = PLOT_DIR / f'{FIG_TYPE_ID}_combined_panels_bc1.png'
    fig.savefig(str(out_eps), format='eps', bbox_inches='tight', facecolor='white')
    fig.savefig(str(out_png), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n  Saved EPS (vector): {out_eps}")
    print(f"  Saved PNG (preview): {out_png}")
    print("=== Done (BreastCancer1) ===")


if __name__ == "__main__":
    run()
