"""
诊断脚本：不训练，只验证 LR 先验是否真的被 corr_target 用上。

复用 run_demo_2layers_multihead.py 的全部预处理逻辑，跑到
Train_STAGATE 内部构造 corr_target 的那一步，把以下指标打出来：

  A. cross_df: 跨层边数
  B. LR score map: 多少 (lig, rec) 在表里能查到分数 (lr_weight>0 的比例)
  C. expr_sim 分布: 表达相似度的分位数
  D. prior = expr_sim * lr_weights 的非零比例 / 分位数
  E. corr_target 对齐到 G_tf 边集后:
        - 总边数 / 非零位置数 / 是否与 cross_df 对应
        - 与 cross_df 走 (src, dst) 是否一一对上
  F. 反向 key (dst, src) 命中数 (=0 才说明 bug 已修)

只读不训练。
"""

import os
os.environ.setdefault("PYTHONHASHSEED", "2021")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import sys
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent
SCORING_DIR = ROOT.parent / "3.LR_Scoring"
sys.path.insert(0, str(ROOT))

from STAGATE.Train_STAGATE import (
    _load_lr_score_map,
    _build_lr_weights,
    prepare_graph_data,
)


SEED = 2020
random.seed(SEED)
np.random.seed(SEED)


def build_symmetric_knn_edges(expr, node_names, label, k=3, weight=1.0):
    if expr.shape[0] < 2:
        return pd.DataFrame(columns=["Cell1", "Cell2", "Distance", "SNN", "Weight"])
    nn = NearestNeighbors(n_neighbors=min(k + 1, expr.shape[0]), metric="euclidean")
    nn.fit(expr)
    _, indices = nn.kneighbors(expr, return_distance=True)
    edge_set = set()
    for i in range(expr.shape[0]):
        for j in indices[i]:
            if i == j:
                continue
            edge_set.add((i, j))
            edge_set.add((j, i))
    if not edge_set:
        return pd.DataFrame(columns=["Cell1", "Cell2", "Distance", "SNN", "Weight"])
    src, dst = zip(*edge_set)
    return pd.DataFrame({
        "Cell1": node_names[list(src)],
        "Cell2": node_names[list(dst)],
        "Distance": weight,
        "SNN": label,
        "Weight": weight,
    })


def main():
    lig_df = pd.read_csv(SCORING_DIR / "ligand_expr_by_cell_filtered-A2.csv", index_col=0)
    rec_df = pd.read_csv(SCORING_DIR / "receptor_expr_by_cell_filtered-A2.csv", index_col=0)

    X_lig = lig_df.astype(np.float32)
    X_rec = rec_df.astype(np.float32)

    all_vars = sorted(set(X_lig.columns) | set(X_rec.columns))
    X_lig = X_lig.reindex(columns=all_vars, fill_value=0.0)
    X_rec = X_rec.reindex(columns=all_vars, fill_value=0.0)

    X = pd.concat([X_lig, X_rec], axis=0)
    section_labels = ["s1"] * X_lig.shape[0] + ["s2"] * X_rec.shape[0]

    if X.index.duplicated().any():
        keep_mask = ~X.index.duplicated(keep="first")
        X = X.loc[keep_mask]
        section_labels = [lab for keep, lab in zip(keep_mask, section_labels) if keep]

    adata = sc.AnnData(X.values)
    adata.obs_names = X.index
    adata.var_names = X.columns
    adata.obs["Section_id"] = section_labels
    adata.X = sp.csr_matrix(adata.X)

    s1_nodes = adata.obs_names[adata.obs["Section_id"] == "s1"]
    s2_nodes = adata.obs_names[adata.obs["Section_id"] == "s2"]

    expr_s1 = np.asarray(adata[s1_nodes, :].X.todense())
    expr_s2 = np.asarray(adata[s2_nodes, :].X.todense())

    intra_s1 = build_symmetric_knn_edges(expr_s1, s1_nodes, "s1", k=3)
    intra_s2 = build_symmetric_knn_edges(expr_s2, s2_nodes, "s2", k=3)

    combo_path = (
        ROOT.parent
        / "2.LR_Screening"
        / "3.Identify sensitive genes and gene combinations"
        / "3.Subnetwork exploration"
        / "combo_only-A2.csv"
    )
    combo_pairs = []
    with combo_path.open("r", encoding="utf-8") as f:
        _ = f.readline()
        for line in f:
            combo = line.strip()
            if not combo or "|" not in combo:
                continue
            lig, rec = combo.split("|", 1)
            if lig in set(s1_nodes) and rec in set(s2_nodes):
                combo_pairs.append((lig, rec))
    cross_df = pd.DataFrame(combo_pairs, columns=["Cell1", "Cell2"])
    cross_df["Distance"] = 1.0
    cross_df["SNN"] = "s1-s2"
    cross_df["Weight"] = 1.0

    adata.uns["Spatial_Net_Zaxis"] = cross_df
    adata.uns["Spatial_Net"] = pd.concat([intra_s1, intra_s2, cross_df], ignore_index=True)

    print("=" * 70)
    print("A. 图规模")
    print("=" * 70)
    print(f"  节点数（基因）          : {adata.n_obs}")
    print(f"    s1 配体节点           : {len(s1_nodes)}")
    print(f"    s2 受体节点           : {len(s2_nodes)}")
    print(f"  层内边（对称 kNN）       : {len(intra_s1) + len(intra_s2)}")
    print(f"  跨层边（cross_df）       : {len(cross_df)}")

    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(len(cells))))

    Spatial_Net = adata.uns["Spatial_Net"].copy()
    Spatial_Net["Cell1"] = Spatial_Net["Cell1"].map(cells_id_tran)
    Spatial_Net["Cell2"] = Spatial_Net["Cell2"].map(cells_id_tran)
    edge_weights = Spatial_Net["Weight"].values.astype(np.float32)
    G = sp.coo_matrix(
        (edge_weights, (Spatial_Net["Cell1"], Spatial_Net["Cell2"])),
        shape=(adata.n_obs, adata.n_obs),
    )
    G_tf = prepare_graph_data(G)
    print(f"  G_tf 总边数 (含自环)     : {G_tf[0].shape[0]}")

    print()
    print("=" * 70)
    print("B. LR score map 命中率")
    print("=" * 70)
    lr_score_map = _load_lr_score_map(
        SCORING_DIR / "LR_scores_all_pairs_V0_meta_gpu_A2.csv",
        score_column="mean_score_recv",
    )
    print(f"  LR score map 条目数      : {len(lr_score_map)}")

    section_lookup = adata.obs["Section_id"]
    lr_weights = _build_lr_weights(cross_df, section_lookup, "s1", "s2", lr_score_map)
    n_hit = int(np.sum(lr_weights > 0))
    print(f"  cross 边数                : {len(cross_df)}")
    print(f"    在 LR map 中查到 (lr_weight>0): {n_hit} ({100*n_hit/len(cross_df):.2f}%)")
    if n_hit > 0:
        nz = lr_weights[lr_weights > 0]
        q = np.quantile(nz, [0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
        print(f"  lr_weight 非零分位数      : min/25/50/75/90/99/max = {q}")

    print()
    print("=" * 70)
    print("C. expr_sim (sigma_expr=30)")
    print("=" * 70)
    expr = np.asarray(adata.X.todense())
    rows = cross_df["Cell1"].map(cells_id_tran).values
    cols = cross_df["Cell2"].map(cells_id_tran).values
    diff = expr[rows] - expr[cols]
    expr_dist_sq = np.sum(diff * diff, axis=1)
    sigma = 30
    expr_sim = np.exp(-expr_dist_sq / (2.0 * sigma * sigma))
    q = np.quantile(expr_sim, [0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
    print(f"  expr_sim 分位数           : min/25/50/75/90/99/max = {q}")
    print(f"  expr_sim < 1e-3 比例      : {100*np.mean(expr_sim < 1e-3):.2f}%")

    print()
    print("=" * 70)
    print("D. prior = expr_sim * lr_weights (归一化前/后)")
    print("=" * 70)
    prior_raw = expr_sim * lr_weights
    n_nz_prior = int(np.sum(prior_raw > 0))
    print(f"  prior 非零数              : {n_nz_prior} / {len(prior_raw)} ({100*n_nz_prior/len(prior_raw):.2f}%)")
    if n_nz_prior > 0:
        q = np.quantile(prior_raw[prior_raw > 0], [0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
        print(f"  prior_raw 非零分位数      : min/25/50/75/90/99/max = {q}")
    max_prior = float(np.max(prior_raw))
    prior = prior_raw / (max_prior + 1e-6) if max_prior > 0 else np.zeros_like(prior_raw)
    print(f"  归一化后 prior max        : {prior.max():.4f}, mean (含零): {prior.mean():.4f}")

    print()
    print("=" * 70)
    print("E. corr_target 对齐 (使用修正后的 (src,dst))")
    print("=" * 70)
    prior_map = {(int(r), int(c)): float(v) for r, c, v in zip(rows, cols, prior)}
    aligned = np.zeros(G_tf[0].shape[0], dtype=np.float32)
    for i in range(G_tf[0].shape[0]):
        r = int(G_tf[0][i, 0])
        c = int(G_tf[0][i, 1])
        aligned[i] = prior_map.get((r, c), 0.0)
    n_nz_aligned = int(np.sum(aligned > 0))
    print(f"  corr_target 长度          : {len(aligned)}")
    print(f"  corr_target 非零位置数    : {n_nz_aligned}")
    print(f"  vs cross_df 中 prior 非零 : {n_nz_prior}  (应一致)")
    if n_nz_aligned > 0:
        q = np.quantile(aligned[aligned > 0], [0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
        print(f"  corr_target 非零分位数    : min/25/50/75/90/99/max = {q}")

    print()
    print("=" * 70)
    print("F. 反向 key 命中数 (验证 bug 是否真的修好)")
    print("=" * 70)
    bad_aligned = np.zeros(G_tf[0].shape[0], dtype=np.float32)
    for i in range(G_tf[0].shape[0]):
        r = int(G_tf[0][i, 1])  # 故意反一下
        c = int(G_tf[0][i, 0])
        bad_aligned[i] = prior_map.get((r, c), 0.0)
    n_nz_bad = int(np.sum(bad_aligned > 0))
    print(f"  反向 (dst,src) 查 prior_map 非零数: {n_nz_bad}")
    print(f"    若远小于正向 ({n_nz_aligned}), 说明先验有方向且 bug 修复有效")

    print()
    print("=" * 70)
    print("结论提示")
    print("=" * 70)
    if n_nz_aligned == 0:
        print("  ✗ corr_target 全 0，先验没进训练。请检查 cross_df 与 LR map 名字匹配。")
    else:
        ratio = n_nz_aligned / len(aligned)
        print(f"  ✓ corr_target 中 {n_nz_aligned} / {len(aligned)} ({100*ratio:.3f}%) 是非零监督信号。")
        print(f"  ✓ 模型的 corr loss 只在这 {n_nz_aligned} 条边上计算 Pearson 相关。")
        print(f"    （model_multihead.py 里 mask=tf.not_equal(corr_target,0) 决定参与计算的边）")


if __name__ == "__main__":
    main()
