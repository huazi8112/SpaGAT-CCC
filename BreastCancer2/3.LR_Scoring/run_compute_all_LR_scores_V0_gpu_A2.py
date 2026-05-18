#!/usr/bin/env python3
"""
GPU-accelerated LR scoring (A2 version)

默认输入路径（相对仓库 BreastCancer2 结构解析）：
- BreastCancer2/1.Preprocessing/ligand_expr_by_cell_A2.csv
- BreastCancer2/1.Preprocessing/receptor_expr_by_cell_A2.csv
- BreastCancer2/1.Preprocessing/de_coords.csv

默认输出文件：
- LR_scores_all_pairs_V0_meta_gpu_A2.csv（列含 mean_score_recv = 分数列和 / 全部细胞数，按其降序；不再写 mean_score）
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np

os.environ.setdefault("CUPY_ACCELERATORS", "")

import cupy as cp
import cupyx.scipy.sparse as csp

_SCRIPT_DIR = Path(__file__).resolve().parent
_PREPROCESSING_DIR = _SCRIPT_DIR.parent / "1.Preprocessing"
DEFAULT_LIGAND_CSV = _PREPROCESSING_DIR / "ligand_expr_by_cell_A2.csv"
DEFAULT_RECEPTOR_CSV = _PREPROCESSING_DIR / "receptor_expr_by_cell_A2.csv"
DEFAULT_COORDS_CSV = _PREPROCESSING_DIR / "de_coords.csv"


def parse_args():
    p = argparse.ArgumentParser(description="Compute LR scores with GPU (V0 weights, A2)")
    p.add_argument("--output-dir", default=None, help="Output directory. Defaults to this script directory.")
    p.add_argument(
        "--ligand-path",
        type=Path,
        default=DEFAULT_LIGAND_CSV,
        help=f"Ligand expression CSV (default: {DEFAULT_LIGAND_CSV})",
    )
    p.add_argument(
        "--receptor-path",
        type=Path,
        default=DEFAULT_RECEPTOR_CSV,
        help=f"Receptor expression CSV (default: {DEFAULT_RECEPTOR_CSV})",
    )
    p.add_argument(
        "--coords-path",
        type=Path,
        default=DEFAULT_COORDS_CSV,
        help=f"Coordinates CSV (default: {DEFAULT_COORDS_CSV})",
    )
    p.add_argument("--output-file", default="LR_scores_all_pairs_V0_meta_gpu_A2.csv", help="Output CSV filename")
    p.add_argument("--alpha", type=float, default=0.5, help="Ligand competition factor alpha_const")
    p.add_argument("--w-near", type=float, default=0.000106, help="Weight for d < d1")
    p.add_argument("--d1", type=float, default=5700.0, help="Distance threshold 1")
    p.add_argument("--d2", type=float, default=14100.0, help="Distance threshold 2")
    p.add_argument("--kappa", type=float, default=0.0000877, help="Decay parameter for mid distances")
    p.add_argument("--lambda_", type=float, default=0.000765, help="Decay parameter for far distances")
    p.add_argument("--device", default="0", help="CUDA device id, e.g., '0' or '0,1' (first id used).")
    return p.parse_args()


def parse_feature(feat: str):
    parts = feat.split("__", 1)
    if len(parts) != 2:
        raise ValueError(f"Feature name should look like GENE__Cluster, got: {feat}")
    return parts[0], parts[1]


def piecewise_weight(d, params):
    w = cp.empty_like(d)
    mask_near = d < params["d1"]
    w[mask_near] = params["w_near"]
    mask_mid = (d >= params["d1"]) & (d < params["d2"])
    w[mask_mid] = cp.exp(-params["kappa"] * d[mask_mid]) / d[mask_mid]
    mask_far = ~mask_near & ~mask_mid
    w[mask_far] = cp.exp(-params["lambda_"] * d[mask_far])
    return w


def load_inputs(lig_path: Path, rec_path: Path, coords_path: Path):
    lig = pd.read_csv(lig_path)
    rec = pd.read_csv(rec_path)
    coords = pd.read_csv(coords_path)
    coords = coords.rename(columns={coords.columns[0]: "Barcode"})

    cell_cols = list(set(lig.columns[1:]) & set(rec.columns[1:]) & set(coords["Barcode"]))
    if not cell_cols:
        raise ValueError("No shared cell barcodes across ligand/receptor/coords.")

    lig = lig[["feature"] + cell_cols]
    rec = rec[["feature"] + cell_cols]
    coords = coords.set_index("Barcode").loc[cell_cols]

    return lig, rec, coords, cell_cols


def main():
    args = parse_args()
    cp.cuda.Device(int(args.device.split(",")[0])).use()

    script_dir = Path(__file__).resolve().parent
    lig_path = args.ligand_path.expanduser().resolve()
    rec_path = args.receptor_path.expanduser().resolve()
    coords_path = args.coords_path.expanduser().resolve()
    output_dir = Path(args.output_dir) if args.output_dir else script_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    params = dict(w_near=args.w_near, d1=args.d1, d2=args.d2, kappa=args.kappa, lambda_=args.lambda_)
    alpha_const = args.alpha

    print(f"Ligand CSV:    {lig_path}")
    print(f"Receptor CSV: {rec_path}")
    print(f"Coords CSV:    {coords_path}")
    print(f"Output dir:    {output_dir}")
    print("Reading inputs...")

    lig, rec, coords, cell_cols = load_inputs(lig_path, rec_path, coords_path)
    n_cells = len(cell_cols)

    xy = cp.asarray(coords[["x", "y"]].to_numpy(), dtype=cp.float32)
    clusters = coords["Cluster"].to_numpy()
    idx_by_cluster = {c: np.where(clusters == c)[0] for c in np.unique(clusters)}

    print(f"Cells: {n_cells}; Lig features: {len(lig)}; Rec features: {len(rec)}; Clusters: {len(idx_by_cluster)}")

    weight_cache = {}

    def get_weight(c_sender, c_recv):
        key = (c_sender, c_recv)
        if key in weight_cache:
            return weight_cache[key]
        s_idx = idx_by_cluster[c_sender]
        r_idx = idx_by_cluster[c_recv]
        block = cp.linalg.norm(xy[s_idx][:, None, :] - xy[r_idx][None, :, :], axis=2)
        block = cp.maximum(block, 1.0)
        w = piecewise_weight(block, params)
        weight_cache[key] = w
        return w

    lig_feat = lig["feature"].to_numpy()
    rec_feat = rec["feature"].to_numpy()
    lig_mat = cp.asarray(lig.iloc[:, 1:].to_numpy(), dtype=cp.float32)
    rec_mat = cp.asarray(rec.iloc[:, 1:].to_numpy(), dtype=cp.float32)

    rows = []
    cols = []
    vals = []
    pair_meta = []
    pair_idx = 0

    print("Scoring ligand/receptor pairs on GPU...")
    for i, lf in enumerate(lig_feat):
        lg, lc = parse_feature(lf)
        if lc not in idx_by_cluster:
            continue
        s_idx = idx_by_cluster[lc]
        lig_vec = lig_mat[i, s_idx]
        if not np.any(cp.asnumpy(lig_vec)):
            continue

        for j, rf in enumerate(rec_feat):
            rg, rc = parse_feature(rf)
            if rc not in idx_by_cluster or lc == rc:
                continue
            r_idx = idx_by_cluster[rc]
            rec_vec = rec_mat[j, r_idx]
            if not np.any(cp.asnumpy(rec_vec)):
                continue

            w_sr = get_weight(lc, rc)
            lig_signal = w_sr.T @ lig_vec
            scores = rec_vec * lig_signal * alpha_const
            nz = cp.nonzero(scores)[0]
            if nz.size == 0:
                continue

            rows.append(cp.asarray(r_idx)[nz])
            cols.append(cp.full(nz.shape, pair_idx, dtype=cp.int64))
            vals.append(scores[nz])

            pair_meta.append({
                "pair": f"{lf}|{rf}",
                "ligand_gene": lg,
                "receptor_gene": rg,
                "sender_cluster": lc,
                "receiver_cluster": rc,
                "n_sender": len(s_idx),
                "n_receiver": len(r_idx),
            })
            pair_idx += 1

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} / {len(lig_feat)} ligands; pairs so far: {pair_idx}")

    if not rows:
        raise RuntimeError("No non-zero scores were generated.")

    rows = cp.concatenate(rows)
    cols = cp.concatenate(cols)
    vals = cp.concatenate(vals)
    score_mat = csp.csr_matrix((vals, (rows, cols)), shape=(n_cells, pair_idx))

    # mean_score_recv：该对在各接收细胞上的得分列和 ÷ n_cells（全部细胞数）
    col_sums = cp.asnumpy(score_mat.sum(axis=0)).ravel()
    mean_scores_recv = col_sums / float(n_cells)

    meta_df = pd.DataFrame(pair_meta)
    meta_df["mean_score_recv"] = mean_scores_recv
    meta_df = meta_df.sort_values("mean_score_recv", ascending=False)

    meta_path = output_dir / args.output_file
    meta_df.to_csv(meta_path, index=False, encoding="utf-8-sig")
    print(f"Saved pair metadata table to {meta_path}")
    print(f"Done. Total pairs: {pair_idx}")


if __name__ == "__main__":
    main()
