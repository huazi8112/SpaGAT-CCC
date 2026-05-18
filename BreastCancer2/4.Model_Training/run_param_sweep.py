"""Parameter sweep for STAGATE multihead training.

围绕 baseline (sigma_expr=30, k=3, num_heads=4, lr=1e-4, n_epochs=500)
对 sigma_expr / k / num_heads / lr / n_epochs 做 one-at-a-time 扫描，
并对 sigma × k 做一个小网格（最影响先验质量的两个超参）。

评估指标沿用 run_ablation_v4.py：ROC_AUC / PR_AUC / P@500 / R@500 / hit。
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

import time
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, average_precision_score
from STAGATE.Train_STAGATE import train_STAGATE

SEED = 2020
root = Path(__file__).resolve().parent
scoring_dir = root.parent / "3.LR_Scoring"
output_dir = root / "param_sweep"
output_dir.mkdir(exist_ok=True)

COMBO_FILE_NAME = "combo_only-A2.csv"
COMBO_FILE = (
    root.parent
    / "2.LR_Screening"
    / "3.Identify sensitive genes and gene combinations"
    / "3.Subnetwork exploration"
    / COMBO_FILE_NAME
)
LIG_FILE = "ligand_expr_by_cell_filtered-A2.csv"
REC_FILE = "receptor_expr_by_cell_filtered-A2.csv"
LR_SCORE_FILE = "LR_scores_all_pairs_V0_meta_gpu_A2.csv"
LR_SCORE_PATH = scoring_dir / LR_SCORE_FILE
KNOWN_NROWS = 766
TOPK = 800

BASELINE = dict(sigma_expr=30, k=3, num_heads=4, lr=1e-4, n_epochs=500)


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


def evaluate_pred(pred_path, known, topk=TOPK):
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
    topk_df = labels.sort_values("score", ascending=False).head(500)
    tp = topk_df["label"].sum()
    m["P@500"] = tp / 500
    m["R@500"] = tp / len(known) if len(known) else 0.0
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


def load_combo_cross_edges(s1_nodes, s2_nodes):
    combo_pairs = []
    with COMBO_FILE.open("r", encoding="utf-8") as f:
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


def run_one(name, X, section_labels, s1_nodes, s2_nodes, cross_df, known,
            sigma_expr, k, num_heads, lr, n_epochs):
    pred_path = output_dir / f"pred_{name}.csv"
    intra_s1 = build_symmetric_knn_edges(X.loc[s1_nodes].values, s1_nodes, "s1", k=k)
    intra_s2 = build_symmetric_knn_edges(X.loc[s2_nodes].values, s2_nodes, "s2", k=k)

    random.seed(SEED)
    np.random.seed(SEED)
    adata = build_adata(X, section_labels, intra_s1, intra_s2, cross_df.copy())

    t0 = time.time()
    adata = train_STAGATE(
        adata, hidden_dims=[512, 30],
        num_heads=num_heads, alpha=0,
        random_seed=SEED, n_epochs=n_epochs, lr=lr,
        verbose=False, save_attention=True, save_loss=True,
        use_corr_loss=True, sigma_expr=sigma_expr,
        lr_score_rds=LR_SCORE_PATH,
        ligand_section="s1", receptor_section="s2",
        lr_score_column="mean_score_recv",
        shuffle_prior=False,
    )
    train_sec = time.time() - t0

    extract_pred(adata, pred_path)
    m = evaluate_pred(pred_path, known)
    m["variant"] = name
    m["sigma_expr"] = sigma_expr
    m["k"] = k
    m["num_heads"] = num_heads
    m["lr"] = lr
    m["n_epochs"] = n_epochs
    m["train_sec"] = round(train_sec, 2)
    final_loss = float(adata.uns.get("STAGATE_loss", [np.nan])[-1]) if "STAGATE_loss" in adata.uns else np.nan
    m["final_loss"] = final_loss
    print(f"  [{name:<22}] sigma={sigma_expr} k={k} heads={num_heads} lr={lr} ep={n_epochs} "
          f"| hit={m['hit']:>3} ROC={m['ROC_AUC']:.4f} PR={m['PR_AUC']:.4f} "
          f"P@500={m['P@500']:.4f} R@500={m['R@500']:.4f} loss={final_loss:.4f} t={train_sec:.1f}s")
    return m


def build_configs():
    """One-at-a-time sweep + sigma×k grid."""
    configs = []

    # 1) baseline
    configs.append(("baseline_s30_k3_h4_lr1e-4_ep500", BASELINE.copy()))

    # 2) sigma_expr sweep (固定 k=3, heads=4, lr=1e-4, ep=500)
    for s in [10, 15, 20, 25, 35, 40, 50, 60]:
        cfg = BASELINE.copy(); cfg["sigma_expr"] = s
        configs.append((f"sigma_{s}", cfg))

    # 3) k sweep (固定 sigma=30, heads=4, lr=1e-4, ep=500)
    for k in [6, 10, 15]:
        cfg = BASELINE.copy(); cfg["k"] = k
        configs.append((f"k_{k}", cfg))

    # 4) num_heads sweep
    for h in [1, 2, 8]:
        cfg = BASELINE.copy(); cfg["num_heads"] = h
        configs.append((f"heads_{h}", cfg))

    # 5) lr sweep
    for lr in [5e-5, 2e-4, 5e-4]:
        cfg = BASELINE.copy(); cfg["lr"] = lr
        configs.append((f"lr_{lr:.0e}", cfg))

    # 6) n_epochs sweep
    for ep in [300, 1000, 2000]:
        cfg = BASELINE.copy(); cfg["n_epochs"] = ep
        configs.append((f"epochs_{ep}", cfg))

    # 7) sigma × k 小网格（先验质量与图密度联合影响）
    for s in [25, 30, 35]:
        for k in [6, 10]:
            cfg = BASELINE.copy(); cfg["sigma_expr"] = s; cfg["k"] = k
            configs.append((f"grid_s{s}_k{k}", cfg))

    return configs


def main():
    print("Loading base data...")
    X, section_labels = prepare_base_data()
    s1_nodes = X.index[[lab == "s1" for lab in section_labels]]
    s2_nodes = X.index[[lab == "s2" for lab in section_labels]]
    cross_df = load_combo_cross_edges(s1_nodes, s2_nodes)
    known = load_known_pairs(COMBO_FILE, nrows=KNOWN_NROWS)
    print(f"Known positives: {len(known)}, Total cross edges: {len(cross_df)}, "
          f"Positive ratio: {len(known)/len(cross_df):.1%}")

    configs = build_configs()
    print(f"Total configs: {len(configs)}\n")

    results = []
    for i, (name, cfg) in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {name}")
        try:
            m = run_one(name, X, section_labels, s1_nodes, s2_nodes, cross_df, known, **cfg)
            results.append(m)
            pd.DataFrame(results).to_csv(output_dir / "summary.csv", index=False)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    df = pd.DataFrame(results)
    cols = ["variant", "sigma_expr", "k", "num_heads", "lr", "n_epochs",
            "hit", "hit_rate", "ROC_AUC", "PR_AUC", "P@500", "R@500",
            "final_loss", "train_sec"]
    df = df[cols]
    df.to_csv(output_dir / "summary.csv", index=False)

    print(f"\n{'='*100}\nParameter Sweep Summary (sorted by PR_AUC desc)\n{'='*100}")
    print(df.sort_values("PR_AUC", ascending=False).to_string(index=False))
    print(f"\n{'='*100}\nTop 5 by ROC_AUC:\n{'='*100}")
    print(df.sort_values("ROC_AUC", ascending=False).head(5).to_string(index=False))


if __name__ == "__main__":
    main()
