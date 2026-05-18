"""
run_ablation_v5.py — 变体命名为 A–E，对照 run_ablation_v3.py（无字母前缀）：
  A <- MeanPooling    B <- Full    C <- ShuffledPrior    D <- Sigma60    E <- RawExpr
输出：ablation_v5/pred_{A..E}.csv, summary.csv 中 v3_variant 为上述短名。
其余逻辑与 run_ablation_v3.py 相同。
"""
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
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, average_precision_score
from STAGATE.Train_STAGATE import train_STAGATE

SEED = 2021
root = Path(__file__).resolve().parent
scoring_dir = root.parent / "3.LR_Scoring"
output_dir = root / "ablation_v5"
output_dir.mkdir(exist_ok=True)

COMBO_FILE = "combo_only-A2.csv"
COMBO_PATH = scoring_dir / COMBO_FILE
LIG_FILE = "ligand_expr_by_cell_filtered-A2.csv"
REC_FILE = "receptor_expr_by_cell_filtered-A2.csv"
LR_SCORE_FILE = "LR_scores_all_pairs_V0_meta_gpu_A2.csv"
KNOWN_NROWS = 766

EVAL_TOPK = 800

LR_SCORE_PATH = scoring_dir / LR_SCORE_FILE

VARIANTS = {
    "A": dict(
        _v3_name="MeanPooling",
        skip_train=True,
    ),
    "B": dict(
        _v3_name="Full",
        num_heads=4, use_corr_loss=True, sigma_expr="auto",
        lr_score_rds=LR_SCORE_PATH, shuffle_prior=False,
    ),
    "C": dict(
        _v3_name="ShuffledPrior",
        num_heads=4, use_corr_loss=True, sigma_expr="auto",
        lr_score_rds=LR_SCORE_PATH, shuffle_prior=True,
    ),
    "D": dict(
        _v3_name="Sigma60",
        num_heads=4, use_corr_loss=True, sigma_expr=60,
        lr_score_rds=LR_SCORE_PATH, shuffle_prior=False,
    ),
    "E": dict(
        _v3_name="RawExpr",
        num_heads=4, use_corr_loss=True, sigma_expr="auto",
        lr_score_rds=LR_SCORE_PATH, shuffle_prior=False, log_expr=False,
    ),
}


def _cfg_for_train(cfg):
    """供 train_STAGATE 使用，去掉元信息键。"""
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


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
        "Cell1": node_names[list(src)], "Cell2": node_names[list(dst)],
        "Distance": weight, "SNN": label, "Weight": weight,
    })


def load_known_pairs(path, nrows):
    raw = pd.read_csv(path, nrows=nrows, header=None)
    if raw.shape[1] == 1:
        col = raw.iloc[:, 0].astype(str)
        mask = col.str.contains("|", regex=False)
        split = col[mask].str.split("|", n=1, expand=True)
    else:
        split = raw.iloc[:, :2].copy()
        split.columns = [0, 1]
    known = split.rename(columns={0: "Cell1", 1: "Cell2"})
    known = known.dropna(subset=["Cell1", "Cell2"])
    known["Cell1"] = known["Cell1"].astype(str).str.strip()
    known["Cell2"] = known["Cell2"].astype(str).str.strip()
    known["label"] = 1
    known["key_ud"] = known.apply(lambda r: "||".join(sorted([r.Cell1, r.Cell2])), axis=1)
    return known


def evaluate_pred(pred_path, known, topk=500):
    pred = pd.read_csv(pred_path)
    score_col = "att" if "att" in pred.columns else "score"
    pred = pred.rename(columns={score_col: "score"})
    pred["Cell1"] = pred["Cell1"].astype(str).str.strip()
    pred["Cell2"] = pred["Cell2"].astype(str).str.strip()
    pred["key_ud"] = pred.apply(lambda r: "||".join(sorted([r.Cell1, r.Cell2])), axis=1)
    pred = pred.sort_values("score", ascending=False).head(topk).reset_index(drop=True)
    all_keys = pd.unique(pd.concat([pred["key_ud"], known["key_ud"]], ignore_index=True))
    labels = pd.DataFrame({"key_ud": all_keys})
    labels = labels.merge(known[["key_ud", "label"]], on="key_ud", how="left")
    labels["label"] = labels["label"].fillna(0)
    labels = labels.merge(pred[["key_ud", "score"]], on="key_ud", how="left").fillna(0)
    y_true, y_score = labels["label"].values, labels["score"].values
    hit = known[known["key_ud"].isin(pred["key_ud"])]
    m = {"hit": len(hit), "hit_rate": len(hit) / len(known) if len(known) else 0}
    if len(pd.unique(y_true)) >= 2:
        m["ROC_AUC"] = roc_auc_score(y_true, y_score)
        m["PR_AUC"] = average_precision_score(y_true, y_score)
    else:
        m["ROC_AUC"] = m["PR_AUC"] = 0.0
    topk_df = labels.sort_values("score", ascending=False).head(topk)
    tp = topk_df["label"].sum()
    pk, rk = f"P@{topk}", f"R@{topk}"
    m[pk] = tp / topk if topk else 0.0
    m[rk] = tp / len(known) if len(known) else 0.0
    return m


def prepare_base_data():
    lig_df = pd.read_csv(scoring_dir / LIG_FILE, index_col=0)
    rec_df = pd.read_csv(scoring_dir / REC_FILE, index_col=0)
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
    return X, section_labels


def load_combo_cross_edges(s1_nodes, s2_nodes, combo_path):
    combo_pairs = []
    with combo_path.open("r", encoding="utf-8") as f:
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
    return cross_df


def build_adata(X, section_labels, intra_s1, intra_s2, cross_df):
    adata = sc.AnnData(X.values)
    adata.obs_names = X.index
    adata.var_names = X.columns
    adata.obs["Section_id"] = section_labels
    adata.X = sp.csr_matrix(adata.X)
    adata.uns["Spatial_Net_2D"] = pd.concat([intra_s1, intra_s2], ignore_index=True)
    adata.uns["Spatial_Net_Zaxis"] = cross_df
    adata.uns["Spatial_Net"] = pd.concat(
        [adata.uns["Spatial_Net_2D"], adata.uns["Spatial_Net_Zaxis"]], ignore_index=True
    )
    return adata


def extract_pred(adata, save_path):
    att_list = adata.uns.get("STAGATE_attention", None)
    if att_list is None or len(att_list) == 0:
        pd.DataFrame(columns=["Cell1", "Cell2", "att"]).to_csv(save_path, index=False)
        return
    att0 = att_list[0].tocoo()
    id2cell = pd.Index(adata.obs_names)
    att_df = pd.DataFrame({"Cell1": id2cell[att0.row], "Cell2": id2cell[att0.col], "att": att0.data})
    cross = adata.uns["Spatial_Net_Zaxis"][["Cell1", "Cell2"]].copy()
    cross["pair"] = cross[["Cell1", "Cell2"]].apply(lambda x: tuple(sorted(x)), axis=1)
    att_df["pair"] = att_df[["Cell1", "Cell2"]].apply(lambda x: tuple(sorted(x)), axis=1)
    pred_all = att_df.merge(cross[["pair"]].drop_duplicates(), on="pair").sort_values("att", ascending=False)
    pred_all[["Cell1", "Cell2", "att"]].to_csv(save_path, index=False)


def run_mean_pooling(X, cross_df, save_path):
    scores = []
    for _, row in cross_df.iterrows():
        lig = X.loc[row["Cell1"]].values if row["Cell1"] in X.index else np.zeros(X.shape[1])
        rec = X.loc[row["Cell2"]].values if row["Cell2"] in X.index else np.zeros(X.shape[1])
        scores.append(float(np.mean(lig) * np.mean(rec)))
    result = cross_df[["Cell1", "Cell2"]].copy()
    result["att"] = scores
    result = result.sort_values("att", ascending=False)
    result.to_csv(save_path, index=False)


def main():
    X, section_labels = prepare_base_data()
    s1_nodes = X.index[[lab == "s1" for lab in section_labels]]
    s2_nodes = X.index[[lab == "s2" for lab in section_labels]]

    intra_s1 = build_symmetric_knn_edges(X.loc[s1_nodes].values, s1_nodes, "s1", k=3)
    intra_s2 = build_symmetric_knn_edges(X.loc[s2_nodes].values, s2_nodes, "s2", k=3)
    cross_df = load_combo_cross_edges(s1_nodes, s2_nodes, COMBO_PATH)

    known = load_known_pairs(COMBO_PATH, nrows=KNOWN_NROWS)
    print(f"Known positives: {len(known)}, Total combos: {len(cross_df)}, "
          f"Positive ratio: {len(known)/len(cross_df):.1%}\n")

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
        print(f"[诊断] <1e-3: {near0_1e3:.2f}%  <1e-6: {near0_1e6:.2f}%\n")
    else:
        print("无跨层边，跳过表达相似度统计\n")

    results = []
    for name, cfg in VARIANTS.items():
        v3n = cfg.get("_v3_name", "")
        print(f"\n{'='*60}\n  {name} (V3: {v3n})\n{'='*60}")

        pred_path = output_dir / f"pred_{name}.csv"
        cfg_train = _cfg_for_train(cfg)

        if cfg_train.get("skip_train"):
            run_mean_pooling(X, cross_df, pred_path)
        else:
            random.seed(SEED)
            np.random.seed(SEED)
            adata = build_adata(X, section_labels, intra_s1, intra_s2, cross_df.copy())
            score_col = "mean_score_recv" if cfg_train.get("lr_score_rds") else "mean_nonzero"
            adata = train_STAGATE(
                adata, hidden_dims=[512, 30],
                num_heads=cfg_train["num_heads"], alpha=0,
                random_seed=SEED, n_epochs=500, lr=1e-4,
                verbose=True, save_attention=True, save_loss=True,
                use_corr_loss=cfg_train["use_corr_loss"],
                sigma_expr=cfg_train["sigma_expr"],
                lr_score_rds=cfg_train.get("lr_score_rds"),
                ligand_section="s1", receptor_section="s2",
                lr_score_column=score_col,
                shuffle_prior=cfg_train.get("shuffle_prior", False),
                invert_prior=cfg_train.get("invert_prior", False),
                log_expr=cfg_train.get("log_expr", True),
            )
            extract_pred(adata, pred_path)

        m = evaluate_pred(pred_path, known, topk=EVAL_TOPK)
        m["variant"] = name
        m["v3_variant"] = v3n
        results.append(m)
        pk, rk = f"P@{EVAL_TOPK}", f"R@{EVAL_TOPK}"
        print(f"  hit={m['hit']}, ROC={m['ROC_AUC']:.4f}, PR={m['PR_AUC']:.4f}, "
              f"{pk}={m[pk]:.4f}, {rk}={m[rk]:.4f}")

    df = pd.DataFrame(results)[
        ["variant", "v3_variant", "hit", "hit_rate", "ROC_AUC", "PR_AUC", f"P@{EVAL_TOPK}", f"R@{EVAL_TOPK}"]
    ]
    df.to_csv(output_dir / "summary.csv", index=False)
    print(f"\n{'='*60}\nAblation v5 Summary\n{'='*60}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
