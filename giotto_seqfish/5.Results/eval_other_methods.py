import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from pathlib import Path


def canonical_lr_node(node: str) -> str:
    text = str(node).strip()
    if "__" in text:
        gene, celltype = text.split("__", 1)
        return f"{gene.upper()}__{celltype}"
    return text.upper()


def build_key_dir(cell1: str, cell2: str) -> str:
    return f"{canonical_lr_node(cell1)}||{canonical_lr_node(cell2)}"


# ------------------ Common helpers (from carculate-score.py) ------------------
def load_known_pairs(path: str | Path, nrows: int = 342) -> pd.DataFrame:
    """Load validated positives from combo_only file (first nrows rows).
    Accepts either single-column "Ligand|Receptor" or two columns.
    """
    raw = pd.read_csv(path, nrows=nrows, header=None)

    if raw.shape[1] == 1:
        col = raw.iloc[:, 0].astype(str)
        mask = col.str.contains("|", regex=False)
        split = col[mask].str.split("|", n=1, expand=True)
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
    known["key_dir"] = known.apply(lambda r: build_key_dir(r.Cell1, r.Cell2), axis=1)
    known = known.drop_duplicates(subset=["key_dir"]).reset_index(drop=True)
    return known[["Cell1", "Cell2", "label", "key_dir"]]


def attach_labels_and_score(pred: pd.DataFrame, known: pd.DataFrame) -> pd.DataFrame:
    # Direction-aware evaluation on key_dir; duplicate predicted keys keep max score
    pred_best = pred.groupby("key_dir", as_index=False)["score"].max()
    known_unique = known[["key_dir", "label"]].drop_duplicates(subset=["key_dir"])

    all_keys = pd.unique(pd.concat([pred_best["key_dir"], known_unique["key_dir"]], ignore_index=True))
    labels = pd.DataFrame({"key_dir": all_keys})
    labels = labels.merge(known_unique, on="key_dir", how="left")
    labels["label"] = labels["label"].fillna(0)
    labels = labels.merge(pred_best, on="key_dir", how="left").fillna(0)
    return labels


def report_metrics(name: str, labels: pd.DataFrame, known: pd.DataFrame, pred: pd.DataFrame, topk: int = 200) -> None:
    pos_total = len(known)
    pred_total = pred["key_dir"].nunique()
    hit = known[known["key_dir"].isin(pred["key_dir"])]
    overlap = len(hit)
    print(f"[{name}] 已验证正例={pos_total}, 预测边={pred_total}, 命中的正例数={int(overlap)}")
    if overlap == 0:
        missing = known.loc[~known["key_dir"].isin(pred["key_dir"])]
        print(f"[{name}] 警告：预测表未命中任何正例，可能是命名或方向不一致。示例前5条未命中：")
        print(missing.head(5))

    y_true = labels["label"].to_numpy(dtype=int)
    y_score = labels["score"].to_numpy(dtype=float)

    if len(pd.unique(y_true)) < 2:
        print(f"[{name}] 警告：y_true 只有单一类别，无法计算 ROC/PR")
    else:
        roc = roc_auc_score(y_true, y_score)
        aupr = average_precision_score(y_true, y_score)
        print(f"[{name}] ROC-AUC = {roc:.4f}, PR-AUC = {aupr:.4f}")

    k = min(topk, len(labels))
    topk_df = labels.sort_values("score", ascending=False).head(k)
    tp = topk_df["label"].sum()
    precision_k = tp / k if k else 0.0
    recall_k = tp / pos_total if pos_total else 0.0
    print(f"[{name}] precision@{k} = {precision_k:.4f}, recall@{k} = {recall_k:.4f}")
    print("-")


def normalize_pred(df: pd.DataFrame, col1: str, col2: str, score_col: str) -> pd.DataFrame:
    pred = df[[col1, col2, score_col]].copy()
    pred.columns = ["Cell1", "Cell2", "score"]
    pred["Cell1"] = pred["Cell1"].astype(str).str.strip()
    pred["Cell2"] = pred["Cell2"].astype(str).str.strip()
    pred["score"] = pd.to_numeric(pred["score"], errors="coerce").fillna(0)
    pred["key_dir"] = pred.apply(lambda r: build_key_dir(r.Cell1, r.Cell2), axis=1)
    return pred


def load_pred_scores(path: Path, topk: int = 200) -> pd.DataFrame:
    pred = pd.read_csv(path)
    score_col = "score" if "score" in pred.columns else "att" if "att" in pred.columns else None
    if score_col is None:
        raise ValueError("pred_scores.csv 需包含 score 或 att 列")
    pred = pred.rename(columns={score_col: "score"})
    pred["Cell1"] = pred["Cell1"].astype(str).str.strip()
    pred["Cell2"] = pred["Cell2"].astype(str).str.strip()
    pred["score"] = pd.to_numeric(pred["score"], errors="coerce").fillna(0)
    pred["key_dir"] = pred.apply(lambda r: build_key_dir(r.Cell1, r.Cell2), axis=1)
    pred = pred.sort_values("score", ascending=False).head(topk).reset_index(drop=True)
    return pred


def normalize_pred_with_celltype(df: pd.DataFrame, sender_col: str, receiver_col: str,
                                 ligand_col: str, receptor_col: str, score_col: str) -> pd.DataFrame:
    """Build keys that include celltype suffix: Ligand__Sender || Receptor__Receiver."""
    pred = df[[sender_col, receiver_col, ligand_col, receptor_col, score_col]].copy()
    pred.columns = ["Sender", "Receiver", "Ligand", "Receptor", "score"]
    pred = pred.astype({"Sender": str, "Receiver": str, "Ligand": str, "Receptor": str})
    pred["score"] = pd.to_numeric(pred["score"], errors="coerce").fillna(0)
    pred["Cell1"] = pred.apply(lambda r: f"{r.Ligand}__{r.Sender}", axis=1)
    pred["Cell2"] = pred.apply(lambda r: f"{r.Receptor}__{r.Receiver}", axis=1)
    pred["key_dir"] = pred.apply(lambda r: build_key_dir(r.Cell1, r.Cell2), axis=1)
    return pred


def main():
    root = Path(__file__).resolve().parent
    pkg_dir = root
    methods_root = root / "OtherMethods"
    topk = 300
    known_nrows = 343
    combo_path = (
        root.parent
        / "2.LR_Screening"
        / "3.Identify sensitive genes and gene combinations"
        / "3.Subnetwork exploration"
        / "combo_only-A3.csv"
    )
    known = load_known_pairs(combo_path, nrows=known_nrows)
    print(f"[Config] known positives use first nrows={known_nrows} from: {combo_path}")

 

    # Stmlnet
    stmlnet_df = pd.read_csv(methods_root / "Stmlnet" / "result_deconv" / "LR_activity_scores.csv")
    stmlnet_pred = normalize_pred(stmlnet_df, "ligand", "receptor", "mean_score")
    stmlnet_pred = stmlnet_pred.sort_values("score", ascending=False).head(topk).reset_index(drop=True)
    stmlnet_labels = attach_labels_and_score(stmlnet_pred, known)
    report_metrics("Stmlnet", stmlnet_labels, known, stmlnet_pred, topk=topk)

    # CellChatV2 (result_combo_only.csv) — needs sender/receiver appended to ligand/receptor
    cellcha_df = pd.read_csv(methods_root / "CellChatV2" / "result" / "result_combo_only.csv")
    cellcha_pred = normalize_pred_with_celltype(cellcha_df, "Sender", "Receiver", "Ligand", "Receptor", "LRscore")
    cellcha_pred = cellcha_pred.sort_values("score", ascending=False).head(topk).reset_index(drop=True)
    cellcha_labels = attach_labels_and_score(cellcha_pred, known)
    report_metrics("CellChatV2", cellcha_labels, known, cellcha_pred, topk=topk)

    # # Conm (result_combo_only copy.csv) — same structure
    newconm_df = pd.read_csv(methods_root / "COMMOT" / "result" / "result_combo_only.csv")
    newconm_pred = normalize_pred_with_celltype(newconm_df, "Sender", "Receiver", "Ligand", "Receptor", "LRscore")
    newconm_pred = newconm_pred.sort_values("score", ascending=False).head(topk).reset_index(drop=True)
    newconm_labels = attach_labels_and_score(newconm_pred, known)
    report_metrics("COMMOT", newconm_labels, known, newconm_pred, topk=topk)

    # CytoSignal (CytoSignal_result.csv) — same structure
    cytosignal_df = pd.read_csv(methods_root / "CytoSignal" / "result" / "CytoSignal_result.csv")
    cytosignal_pred = normalize_pred_with_celltype(cytosignal_df, "Sender", "Receiver", "Ligand", "Receptor", "LRscore")
    cytosignal_pred = cytosignal_pred.sort_values("score", ascending=False).head(topk).reset_index(drop=True)
    cytosignal_labels = attach_labels_and_score(cytosignal_pred, known)
    report_metrics("CytoSignal", cytosignal_labels, known, cytosignal_pred, topk=topk)

    # Spagat-ccc (pred_scores.csv)
    spagat_pred = load_pred_scores(root.parent / "4.Model_Training" / "pred_scores.csv", topk=topk)
    spagat_labels = attach_labels_and_score(spagat_pred, known)
    report_metrics("Spagat-ccc", spagat_labels, known, spagat_pred, topk=topk)


if __name__ == "__main__":
    main()
