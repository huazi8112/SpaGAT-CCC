import os
os.environ["PYTHONHASHSEED"] = "2021"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"

import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
from pathlib import Path
from typing import Optional
from sklearn.neighbors import NearestNeighbors
from STAGATE.Train_STAGATE import train_STAGATE

SEED = 2021
random.seed(SEED)
np.random.seed(SEED)

# -----------------------------
# Load ligand/receptor CSV (no transpose; genes当作细胞)
# -----------------------------
root = Path(__file__).resolve().parent
scoring_dir = root.parent / "3.LR_Scoring"

lig_df = pd.read_csv(scoring_dir / "ligand_expr_by_cell_filtered-A2.csv", index_col=0)
rec_df = pd.read_csv(scoring_dir / "receptor_expr_by_cell_filtered-A2.csv", index_col=0)

# 保持原方向：行=feature(基因)，列=细胞。将行视作节点。
X_lig = lig_df.astype(np.float32)
X_rec = rec_df.astype(np.float32)

# 对齐列名（基因当节点，列名作为特征名）；不一致时并集填0
all_vars = sorted(set(X_lig.columns) | set(X_rec.columns))
X_lig = X_lig.reindex(columns=all_vars, fill_value=0.0)
X_rec = X_rec.reindex(columns=all_vars, fill_value=0.0)

X = pd.concat([X_lig, X_rec], axis=0)
section_labels = ["s1"] * X_lig.shape[0] + ["s2"] * X_rec.shape[0]
section_ids = ["s1", "s2"]

# 去重重叠节点名（配体/受体各一条时保留第一条，标签同步裁剪）
if X.index.duplicated().any():
    keep_mask = ~X.index.duplicated(keep="first")
    X = X.loc[keep_mask]
    section_labels = [lab for keep, lab in zip(keep_mask, section_labels) if keep]

adata = sc.AnnData(X.values)
adata.obs_names = X.index
adata.var_names = X.columns
adata.obs["Section_id"] = section_labels

adata.X = sp.csr_matrix(adata.X)

def build_symmetric_knn_edges(expr: np.ndarray, node_names: pd.Index, label: str, k: int = 15, weight: float = 1.0) -> pd.DataFrame:
    """用表达矩阵的欧氏距离构建对称化 kNN 边."""

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
            edge_set.add((j, i))  # 对称化

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


def build_second_order_edges(first_edges: pd.DataFrame, max_second: Optional[int], weight: float, label: str) -> pd.DataFrame:
    """基于一阶邻居生成二阶边，max_second=None 表示不设上限."""

    if first_edges.empty:
        return pd.DataFrame(columns=["Cell1", "Cell2", "Distance", "SNN", "Weight"])

    adj = {}
    for c1, c2 in zip(first_edges["Cell1"], first_edges["Cell2"]):
        adj.setdefault(c1, set()).add(c2)

    new_edges = []
    for src, nbrs in adj.items():
        candidates = set()
        for n in nbrs:
            candidates.update(adj.get(n, set()))
        candidates.discard(src)
        candidates -= nbrs
        if not candidates:
            continue
        chosen = sorted(candidates) if max_second is None else list(candidates)[:max_second]
        for dst in chosen:
            new_edges.append((src, dst))
            new_edges.append((dst, src))  # 对称化二阶

    if not new_edges:
        return pd.DataFrame(columns=["Cell1", "Cell2", "Distance", "SNN", "Weight"])

    src, dst = zip(*new_edges)
    return pd.DataFrame({
        "Cell1": list(src),
        "Cell2": list(dst),
        "Distance": weight,
        "SNN": label,
        "Weight": weight,
    })


s1_nodes = adata.obs_names[adata.obs["Section_id"] == "s1"]
s2_nodes = adata.obs_names[adata.obs["Section_id"] == "s2"]

# 使用与 obs 对齐的表达矩阵，避免去重后索引不一致
expr_s1 = np.asarray(adata[s1_nodes, :].X.todense()) if sp.isspmatrix(adata.X) else np.asarray(adata[s1_nodes, :].X)
expr_s2 = np.asarray(adata[s2_nodes, :].X.todense()) if sp.isspmatrix(adata.X) else np.asarray(adata[s2_nodes, :].X)

# 一阶对称 kNN 边（k=3，权重1.0）
intra_s1_first = build_symmetric_knn_edges(expr_s1, s1_nodes, "s1", k=3, weight=1.0)
intra_s2_first = build_symmetric_knn_edges(expr_s2, s2_nodes, "s2", k=3, weight=1.0)


# 不使用二阶边，仅保留一阶对称 kNN
intra_s1 = intra_s1_first
intra_s2 = intra_s2_first

# print("s1 first:", len(intra_s1_first), "s1 second:", len(intra_s1_second))
# print("s2 first:", len(intra_s2_first), "s2 second:", len(intra_s2_second))

print("s1 first:", len(intra_s1_first))
print("s2 first:", len(intra_s2_first))

# 片间边：combo_only.csv 的配体->受体
combo_only_path = (
    root.parent
    / "2.LR_Screening"
    / "3.Identify sensitive genes and gene combinations"
    / "3.Subnetwork exploration"
    / "combo_only-A2.csv"
)
combo_pairs = []
with combo_only_path.open("r", encoding="utf-8") as f:
    _ = f.readline()
    for line in f:
        combo = line.strip()
        if not combo or "|" not in combo:
            continue
        lig, rec = combo.split("|", 1)
        if lig in s1_nodes and rec in s2_nodes:
            combo_pairs.append((lig, rec))

cross_df = pd.DataFrame(combo_pairs, columns=["Cell1", "Cell2"])
if not cross_df.empty:
    cross_df["Distance"] = 1.0
    cross_df["SNN"] = "s1-s2"
    cross_df["Weight"] = 1.0
else:
    cross_df = pd.DataFrame(columns=["Cell1", "Cell2", "Distance", "SNN", "Weight"])



# 训练前：表达相似度分布（log1p + Gaussian auto-sigma，与 Train_STAGATE 逻辑一致）
if not cross_df.empty:
    _expr_log = np.log1p(np.maximum(X.values, 0))
    _idx_map = {name: i for i, name in enumerate(X.index)}
    _rows_idx = [_idx_map[c] for c in cross_df["Cell1"]]
    _cols_idx = [_idx_map[c] for c in cross_df["Cell2"]]
    _diff = _expr_log[_rows_idx] - _expr_log[_cols_idx]
    _dist_sq = np.sum(_diff * _diff, axis=1)
    _median_d = float(np.median(np.sqrt(_dist_sq)))
    _sigma_auto = _median_d if _median_d > 0 else 1.0
    expr_sim = np.exp(-_dist_sq / (2.0 * _sigma_auto * _sigma_auto))
    q = np.quantile(expr_sim, [0, 0.25, 0.5, 0.75, 0.9, 0.99])
    near0_1e3 = np.mean(expr_sim < 1e-3) * 100
    near0_1e6 = np.mean(expr_sim < 1e-6) * 100
    print(f"[诊断] log1p+Gaussian auto sigma={_sigma_auto:.2f}: min/25/50/75/90/99% = {q}")
    print(f"[诊断] <1e-3: {near0_1e3:.2f}%  <1e-6: {near0_1e6:.2f}%")
else:
    print("无跨层边，跳过表达相似度统计")


adata.uns["Spatial_Net_2D"] = pd.concat([intra_s1, intra_s2], ignore_index=True)
adata.uns["Spatial_Net_Zaxis"] = cross_df
adata.uns["Spatial_Net"] = pd.concat([adata.uns["Spatial_Net_2D"], adata.uns["Spatial_Net_Zaxis"]], ignore_index=True)
print(
    f"Custom graph: intra edges={adata.uns['Spatial_Net_2D'].shape[0]}, "
    f"cross edges={adata.uns['Spatial_Net_Zaxis'].shape[0]}, total={adata.uns['Spatial_Net'].shape[0]}"
)

# Train STAGATE (short epochs for demo)
adata = train_STAGATE(
    adata,
    hidden_dims=[128, 30],
    num_heads=4,
    alpha=0,
    random_seed=SEED,
    n_epochs=500,
    lr=1e-4,
    #lr=5e-5,
    verbose=True,
    save_attention=True,
    save_loss=True,
    use_corr_loss=True,
    sigma_expr='auto',
    lr_score_rds=scoring_dir / "LR_scores_all_pairs_V0_meta_gpu_A2.csv",
    ligand_section="s1",
    receptor_section="s2",
    lr_score_column="mean_score_recv",
)

print("2-layer Embedding shape:", adata.obsm["STAGATE"].shape)
print("obs columns:", adata.obs.columns.tolist())
print("uns keys:", list(adata.uns.keys()))




# 按注意力权重选跨层最优通路（s1-s2）
att_list = adata.uns.get('STAGATE_attention', None)
if att_list is None or len(att_list) == 0:
    print("未保存注意力权重，请在 train_STAGATE 中设置 save_attention=True。")
else:
    att0 = att_list[0].tocoo()
    id2cell = pd.Index(adata.obs_names)
    att_df = pd.DataFrame({
        "Cell1": id2cell[att0.row],
        "Cell2": id2cell[att0.col],
        "att": att0.data,
    })

    cross = adata.uns['Spatial_Net_Zaxis'][['Cell1', 'Cell2']].copy()
    cross['pair'] = cross[['Cell1', 'Cell2']].apply(lambda x: tuple(sorted(x)), axis=1)
    att_df['pair'] = att_df[['Cell1', 'Cell2']].apply(lambda x: tuple(sorted(x)), axis=1)

    # 保存所有跨层边的预测分数，便于后续算 AUC/PR 等指标
    pred_all = (
        att_df
        .merge(cross[['pair']].drop_duplicates(), on='pair')
        .sort_values('att', ascending=False)
    )
    pred_path = root / "pred_scores.csv"
    pred_all[['Cell1', 'Cell2', 'att']].to_csv(pred_path, index=False)

    print(f"已保存全部跨层边打分到: {pred_path}")

    top = (
        att_df
        .merge(cross[['pair']].drop_duplicates(), on='pair')
        .sort_values('att', ascending=False)
        .head(500)
    )

    # 统计重复（按无序 pair）
    dup_mask = top.duplicated(subset='pair', keep=False)
    num_pairs = top['pair'].nunique()
    num_rows = len(top)
    num_dups = num_rows - num_pairs
    print(f"top 列表行数={num_rows}, 去重后唯一配对数={num_pairs}, 重复条数={num_dups}")
    if dup_mask.any():
        print("出现重复的配对示例：")
        print(top.loc[dup_mask, ['Cell1', 'Cell2', 'att']].head(10))



    print(f"\n按注意力排序的跨层前{len(top)}条：\n")
    print(f"{'序号':<4} {'细胞1':<10} {'细胞2':<10} {'注意力':<12}")
    print("-" * 500)
    for i, row in enumerate(top.itertuples(index=False), 1):
        print(f"{i:<4} {row.Cell1:<10} {row.Cell2:<10} {row.att:<12.4f}")

