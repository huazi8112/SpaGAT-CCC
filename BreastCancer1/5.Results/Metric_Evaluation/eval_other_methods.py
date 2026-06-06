import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
from pathlib import Path


# ------------------ DeLong test for ROC-AUC (Sun & Xu 2014 fast algorithm) ------------------

def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Return mid-ranks (1-indexed) for array x."""
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1)
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks + 1
    return out


def _fast_delong(predictions_T: np.ndarray, n_pos: int):
    """
    Compute AUC estimates and their covariance matrix for k classifiers.

    predictions_T : shape (k, n_pos + n_neg), positives in the first n_pos columns
                    (i.e. samples are sorted with positive labels first)
    n_pos         : number of positive samples
    Returns (aucs, cov)  — aucs shape (k,), cov shape (k, k)
    """
    m, n = n_pos, predictions_T.shape[1] - n_pos
    k = predictions_T.shape[0]
    pos = predictions_T[:, :m]
    neg = predictions_T[:, m:]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)
    for r in range(k):
        tx[r] = _compute_midrank(pos[r])
        ty[r] = _compute_midrank(neg[r])
        tz[r] = _compute_midrank(predictions_T[r])

    aucs = (tz[:, :m].sum(axis=1) - tx.sum(axis=1)) / (m * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m

    sx = np.cov(v01) if k > 1 else np.atleast_2d(np.var(v01, ddof=1))
    sy = np.cov(v10) if k > 1 else np.atleast_2d(np.var(v10, ddof=1))
    cov = sx / m + sy / n
    return aucs, cov


def delong_roc_test(
    y_true: np.ndarray,
    scores_ref: np.ndarray,
    scores_cmp: np.ndarray,
) -> tuple[float, float, float, float]:
    """
    DeLong test for comparing two correlated ROC-AUCs on the *same* test set.

    Returns
    -------
    z        : z-statistic (positive → ref > cmp)
    p_value  : two-sided p-value
    auc_ref  : AUC of the reference classifier
    auc_cmp  : AUC of the comparison classifier
    """
    y_true = np.asarray(y_true, dtype=int)
    scores_ref = np.asarray(scores_ref, dtype=float)
    scores_cmp = np.asarray(scores_cmp, dtype=float)

    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan, np.nan, np.nan, np.nan

    order = np.argsort(-y_true)
    pst = np.vstack([scores_ref[order], scores_cmp[order]])

    aucs, cov = _fast_delong(pst, n_pos)
    diff = aucs[0] - aucs[1]
    se = np.sqrt(max(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1], 1e-20))
    z = diff / se
    p = float(2 * stats.norm.sf(abs(z)))
    return float(z), p, float(aucs[0]), float(aucs[1])


def bootstrap_aupr_test(
    y_true: np.ndarray,
    scores_ref: np.ndarray,
    scores_cmp: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float, float, float, float]:
    """
    Non-parametric bootstrap test for comparing two PR-AUC values.

    H₀: AUPR(ref) == AUPR(cmp)

    Returns
    -------
    obs_diff  : observed AUPR(ref) - AUPR(cmp)
    p_value   : two-sided bootstrap p-value
    ci_low    : 2.5th percentile of bootstrap differences
    ci_high   : 97.5th percentile of bootstrap differences
    aupr_ref  : observed AUPR of reference
    aupr_cmp  : observed AUPR of comparison
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=int)
    scores_ref = np.asarray(scores_ref, dtype=float)
    scores_cmp = np.asarray(scores_cmp, dtype=float)
    n = len(y_true)

    aupr_ref = average_precision_score(y_true, scores_ref)
    aupr_cmp = average_precision_score(y_true, scores_cmp)
    obs_diff = aupr_ref - aupr_cmp

    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        d = average_precision_score(yt, scores_ref[idx]) - average_precision_score(yt, scores_cmp[idx])
        diffs.append(d)

    diffs = np.array(diffs)
    shifted = diffs - diffs.mean()
    p_value = float(np.mean(np.abs(shifted) >= abs(obs_diff)))
    p_value = max(p_value, 1.0 / n_bootstrap)
    ci_low = float(np.percentile(diffs, 2.5))
    ci_high = float(np.percentile(diffs, 97.5))
    return obs_diff, p_value, ci_low, ci_high, aupr_ref, aupr_cmp


# ------------------ Common helpers (from carculate-score.py) ------------------
def load_known_pairs(path: str, nrows: int = 361) -> pd.DataFrame:
    """Load validated positives from combo_only file (first nrows rows).
    Accepts either single-column "Ligand|Receptor" or two columns.
    """
    raw = pd.read_csv(path, nrows=nrows, header=None)

    if raw.shape[1] == 1:
        col = raw.iloc[:, 0].astype(str)
        split = col.str.split("|", n=1, expand=True)
    else:
        split = raw.iloc[:, :2].copy()
        split.columns = [0, 1]

    if split.shape[1] != 2:
        raise ValueError("无法拆分出 Cell1/Cell2，请检查 combo_only 文件格式")

    known = split.rename(columns={0: "Cell1", 1: "Cell2"})
    known = known.dropna(subset=["Cell1", "Cell2"])
    known["Cell1"] = known["Cell1"].astype(str).str.strip()
    known["Cell2"] = known["Cell2"].astype(str).str.strip()
    known["label"] = 1
    known["key_dir"] = known["Cell1"] + "||" + known["Cell2"]
    known["key_ud"] = known.apply(lambda r: "||".join(sorted([r.Cell1, r.Cell2])), axis=1)
    known = known.drop_duplicates(subset=["key_dir"]).reset_index(drop=True)
    return known


def attach_labels_and_score(pred: pd.DataFrame, known: pd.DataFrame) -> pd.DataFrame:
    # Evaluation set = each method's Top-K predictions only (no union with positives).
    # A row's label is 1 iff it appears in the validated positives, else 0.
    labels = pred[["key_dir", "score"]].copy()
    known_keys = set(known["key_dir"].tolist())
    labels["label"] = labels["key_dir"].isin(known_keys).astype(int)
    return labels


def attach_labels_union(pred: pd.DataFrame, known: pd.DataFrame) -> pd.DataFrame:
    """Union of predicted keys and known positives; missing scores filled with 0."""
    all_keys = pd.unique(pd.concat([pred["key_dir"], known["key_dir"]], ignore_index=True))
    labels = pd.DataFrame({"key_dir": all_keys})
    labels = labels.merge(known[["key_dir", "label"]], on="key_dir", how="left")
    labels["label"] = labels["label"].fillna(0)
    labels = labels.merge(pred[["key_dir", "score"]], on="key_dir", how="left").fillna(0)
    return labels


def report_metrics(name: str, labels: pd.DataFrame, known: pd.DataFrame, pred: pd.DataFrame, topk: int = 500) -> None:
    # All metrics are computed on the method's Top-K predictions only.
    pos_total = len(known)
    pred_total = len(pred)
    overlap = int(labels["label"].sum())
    print(f"[{name}] 已验证正例={pos_total}, Top-{pred_total} 预测边={pred_total}, 命中的正例数={overlap}")
    if overlap == 0:
        missing = known.loc[~known["key_dir"].isin(pred["key_dir"])]
        print(f"[{name}] 警告：Top-{pred_total} 未命中任何正例，可能是命名或方向不一致。示例前5条未命中：")
        print(missing.head(5))

    y_true = labels["label"].values
    y_score = labels["score"].values

    if len(pd.unique(y_true)) < 2:
        print(f"[{name}] 警告：Top-{pred_total} 内只有单一类别，无法计算 ROC/PR-AUC")
    else:
        roc = roc_auc_score(y_true, y_score)
        aupr = average_precision_score(y_true, y_score)
        print(f"[{name}] ROC-AUC = {roc:.4f}, PR-AUC = {aupr:.4f}  (computed on Top-{pred_total})")

    k = len(labels)
    precision_k = overlap / k if k else 0.0
    recall_k = overlap / pos_total if pos_total else 0.0
    print(f"[{name}] precision@{k} = {precision_k:.4f}, recall@{k} = {recall_k:.4f}")
    print("-")


def _fmt_p(p: float) -> str:
    if np.isnan(p):
        return "NA"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def significance_report(
    ref_name: str,
    comparisons: list[dict],
    n_bootstrap: int = 2000,
) -> None:
    """
    Pairwise DeLong (ROC-AUC) and bootstrap (PR-AUC) tests comparing *ref*
    against every baseline on a shared candidate universe (ref's Top-K only).
    """
    print("=" * 70)
    print(f"Significance tests — reference: {ref_name}")
    print("(shared universe: reference method's Top-K predictions only)")
    print("=" * 70)

    for cmp in comparisons:
        cmp_name = cmp["name"]
        y_true = cmp["y_true"]
        y_score_ref = cmp["y_score_ref"]
        y_score_cmp = cmp["y_score_cmp"]

        z, p_delong, auc_ref, auc_cmp = delong_roc_test(y_true, y_score_ref, y_score_cmp)
        direction_roc = ">" if auc_ref >= auc_cmp else "<"

        obs_diff, p_boot, ci_lo, ci_hi, aupr_ref, aupr_cmp = bootstrap_aupr_test(
            y_true, y_score_ref, y_score_cmp, n_bootstrap=n_bootstrap
        )
        direction_pr = ">" if obs_diff >= 0 else "<"

        print(f"\n  {ref_name}  vs  {cmp_name}")
        if "universe" in cmp:
            print(f"    Universe: {cmp['universe']}")
        print(
            f"    ROC-AUC : {auc_ref:.4f} {direction_roc} {auc_cmp:.4f}  "
            f"(DeLong z = {z:+.3f}, {_fmt_p(p_delong)})"
        )
        print(
            f"    PR-AUC  : {aupr_ref:.4f} {direction_pr} {aupr_cmp:.4f}  "
            f"(bootstrap {_fmt_p(p_boot)}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}])"
        )

        sig_roc = "*" if not np.isnan(p_delong) and p_delong < 0.05 else "ns"
        sig_pr = "*" if p_boot < 0.05 else "ns"
        if not np.isnan(p_delong) and p_delong < 0.001:
            sig_roc = "***"
        elif not np.isnan(p_delong) and p_delong < 0.01:
            sig_roc = "**"
        if p_boot < 0.001:
            sig_pr = "***"
        elif p_boot < 0.01:
            sig_pr = "**"
        print(f"    Significance: ROC-AUC {sig_roc}, PR-AUC {sig_pr}")

    print("=" * 70)


def normalize_pred(df: pd.DataFrame, col1: str, col2: str, score_col: str) -> pd.DataFrame:
    pred = df[[col1, col2, score_col]].copy()
    pred.columns = ["Cell1", "Cell2", "score"]
    pred["Cell1"] = pred["Cell1"].astype(str).str.strip()
    pred["Cell2"] = pred["Cell2"].astype(str).str.strip()
    pred["key_dir"] = pred["Cell1"] + "||" + pred["Cell2"]
    pred["key_ud"] = pred.apply(lambda r: "||".join(sorted([r.Cell1, r.Cell2])), axis=1)
    return pred


def load_pred_scores(path: Path, topk: int = 500) -> pd.DataFrame:
    pred = pd.read_csv(path)
    score_col = "score" if "score" in pred.columns else "att" if "att" in pred.columns else None
    if score_col is None:
        raise ValueError("pred_scores.csv 需包含 score 或 att 列")
    pred = pred.rename(columns={score_col: "score"})
    pred["Cell1"] = pred["Cell1"].astype(str).str.strip()
    pred["Cell2"] = pred["Cell2"].astype(str).str.strip()
    pred["key_dir"] = pred["Cell1"] + "||" + pred["Cell2"]
    pred["key_ud"] = pred.apply(lambda r: "||".join(sorted([r.Cell1, r.Cell2])), axis=1)
    pred = pred.sort_values("score", ascending=False).head(topk).reset_index(drop=True)
    return pred


def normalize_pred_with_celltype(df: pd.DataFrame, sender_col: str, receiver_col: str,
                                 ligand_col: str, receptor_col: str, score_col: str) -> pd.DataFrame:
    """Build keys that include celltype suffix: Ligand__Sender || Receptor__Receiver."""
    pred = df[[sender_col, receiver_col, ligand_col, receptor_col, score_col]].copy()
    pred.columns = ["Sender", "Receiver", "Ligand", "Receptor", "score"]
    pred = pred.astype({"Sender": str, "Receiver": str, "Ligand": str, "Receptor": str})
    pred["Cell1"] = pred.apply(lambda r: f"{r.Ligand}__{r.Sender}", axis=1)
    pred["Cell2"] = pred.apply(lambda r: f"{r.Receptor}__{r.Receiver}", axis=1)
    pred["key_dir"] = pred["Cell1"] + "||" + pred["Cell2"]
    pred["key_ud"] = pred.apply(lambda r: "||".join(sorted([r.Cell1, r.Cell2])), axis=1)
    return pred


def _scores_on_frame(frame: pd.DataFrame, pred: pd.DataFrame) -> np.ndarray:
    """Return a score vector aligned to *frame* (0 for keys absent in pred)."""
    pred_dedup = (
        pred[["key_dir", "score"]]
        .groupby("key_dir", as_index=False)["score"]
        .max()
    )
    merged = frame[["key_dir"]].merge(pred_dedup, on="key_dir", how="left")
    assert len(merged) == len(frame), (
        f"_scores_on_frame: merge expanded ({len(merged)} vs {len(frame)}). "
        "Check for duplicate key_dir in pred after dedup."
    )
    return merged["score"].fillna(0).values


def main():
    root = Path(__file__).resolve().parent
    methods_root = root / "OtherMethods"
    topk = 500
    combo_path = (
        root.parent.parent
        / "2.LR_Screening"
        / "3.Identify sensitive genes and gene combinations"
        / "3.Subnetwork exploration"
        / "combo_only.csv"
    )
    known = load_known_pairs(combo_path, nrows=361)

    pred_map: dict[str, pd.DataFrame] = {}

    # COMMOT (raw LR mean — commented out; uncomment if file available)
    # commot_df = pd.read_csv(methods_root / "COMMOT" / "result" / "result_lr_mean.csv")
    # pred_map["COMMOT_raw"] = normalize_pred(commot_df, "Ligand", "Receptor", "score_mean")

    stmlnet_df = pd.read_csv(methods_root / "Stmlnet" / "result_deconv" / "LR_activity_scores.csv")
    pred_map["Stmlnet"] = normalize_pred(stmlnet_df, "ligand", "receptor", "mean_score")

    cellcha_df = pd.read_csv(methods_root / "CellChatV2" / "result" / "result_combo_only.csv")
    pred_map["CellChatV2"] = normalize_pred_with_celltype(
        cellcha_df, "Sender", "Receiver", "Ligand", "Receptor", "LRscore"
    )

    newconm_df = pd.read_csv(methods_root / "COMMOT" / "result" / "result_combo_only.csv")
    pred_map["COMMOT"] = normalize_pred_with_celltype(
        newconm_df, "Sender", "Receiver", "Ligand", "Receptor", "LRscore"
    )

    cytosignal_df = pd.read_csv(methods_root / "CytoSignal" / "result" / "CytoSignal_result.csv")
    pred_map["CytoSignal"] = normalize_pred_with_celltype(
        cytosignal_df, "Sender", "Receiver", "Ligand", "Receptor", "LRscore"
    )

    pred_map["Spagat-ccc"] = load_pred_scores(
        root.parent.parent / "4.Model_Training" / "pred_scores.csv", topk=topk
    )

    pred_topk: dict[str, pd.DataFrame] = {
        name: p.sort_values("score", ascending=False).head(topk).reset_index(drop=True)
        for name, p in pred_map.items()
        if name != "Spagat-ccc"
    }
    pred_topk["Spagat-ccc"] = pred_map["Spagat-ccc"]

    print("\n======== Per-method metrics ========")
    for name, pred in pred_topk.items():
        labels = attach_labels_and_score(pred, known)
        report_metrics(name, labels, known, pred, topk=topk)

    spagat_pred = pred_topk.get("Spagat-ccc")
    if spagat_pred is not None:
        sig_frame = attach_labels_and_score(spagat_pred, known)
        if len(np.unique(sig_frame["label"])) < 2:
            print("\n警告：Spagat-ccc Top-K 内只有单一类别，跳过显著性检验")
        else:
            comparisons: list[dict] = []
            y_true = sig_frame["label"].values
            y_score_ref = sig_frame["score"].values
            universe_desc = f"Spagat-ccc top-{topk} (n={len(sig_frame)})"
            for name, pred_full in pred_map.items():
                if name == "Spagat-ccc":
                    continue
                comparisons.append({
                    "name": name,
                    "y_true": y_true,
                    "y_score_ref": y_score_ref,
                    "y_score_cmp": _scores_on_frame(sig_frame, pred_full),
                    "universe": universe_desc,
                })
            if comparisons:
                significance_report("Spagat-ccc", comparisons, n_bootstrap=2000)


if __name__ == "__main__":
    main()
