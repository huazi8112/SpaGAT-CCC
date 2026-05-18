#!/usr/bin/env python3
"""
GPU-accelerated reimplementation of run_compute_all_LR_scores_V0并行.R
- Uses CuPy to keep distance and core math on GPU.
- Preserves V0 piecewise distance weights and alpha_const semantics.
- Inputs: filtered expr matrices and de_coords.csv under BreastCancer1/1.Preprocessing by default.
- Output CSV: mean_score_recv = sparse column sum of per-cell scores / n_cells (all cells); sorted by mean_score_recv descending.

Requirements (example for CUDA 11.x):
    pip install pandas numpy cupy-cuda11x cupyx
Optional for writing RDS directly: pip install pyreadr
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np

# CUDA 13.1 + CuPy 在 CUB reduction 路径可能触发 NVRTC 头文件编译错误，
# 关闭 CUB 加速器以保证论文流程可稳定运行。
os.environ.setdefault("CUPY_ACCELERATORS", "")

import cupy as cp
import cupyx.scipy.sparse as csp


def parse_args():
    p = argparse.ArgumentParser(description="Compute LR scores with GPU (V0 weights)")
    p.add_argument("--input-dir", default=None,
                help="Directory with ligand_expr_by_cell_filtered.csv and receptor_expr_by_cell_filtered.csv "
                    "(defaults to BreastCancer1/1.Preprocessing; de_coords.csv is always read there).")
    p.add_argument("--output-dir", default=None,
                help="Directory to write outputs. Defaults to this script directory (3.LR_Scoring).")
    p.add_argument("--alpha", type=float, default=0.5, help="Ligand competition factor alpha_const")
    p.add_argument("--w-near", type=float, default=0.000114, help="Weight for d < d1")
    p.add_argument("--d1", type=float, default=5300.0, help="Distance threshold 1")
    p.add_argument("--d2", type=float, default=13000.0, help="Distance threshold 2")
    p.add_argument("--kappa", type=float, default=0.0000943, help="Decay parameter for mid distances")
    p.add_argument("--lambda_", type=float, default=0.000823, help="Decay parameter for far distances")
    p.add_argument("--max-cells-for-full-dist", type=int, default=25000,
                   help="If cell count exceeds this, distance weights are computed per cluster-pair block to save memory.")
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


def load_inputs(input_dir: Path, preproc_dir: Path):
    lig = pd.read_csv(input_dir / "ligand_expr_by_cell_filtered.csv")
    rec = pd.read_csv(input_dir / "receptor_expr_by_cell_filtered.csv")
    coords = pd.read_csv(preproc_dir / "de_coords.csv")
    coords = coords.rename(columns={coords.columns[0]: "Barcode"})

    cell_cols = list(set(lig.columns[1:]) & set(rec.columns[1:]) & set(coords["Barcode"]))
    if not cell_cols:
        raise ValueError("No shared cell barcodes across ligand/receptor/coords.")

    lig = lig[["feature"] + cell_cols]
    rec = rec[["feature"] + cell_cols]
    coords = coords.set_index("Barcode").loc[cell_cols]

    return lig, rec, coords, cell_cols


def compute_full_distance(xy: cp.ndarray):
    # xy: (n_cells, 2)
    dist = cp.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    dist = cp.maximum(dist, 1.0)
    return dist


def main():
    args = parse_args()
    cp.cuda.Device(int(args.device.split(",")[0])).use()

    script_dir = Path(__file__).resolve().parent          # 3.LR_Scoring/
    bc1_root = script_dir.parent
    preproc_dir = bc1_root / "1.Preprocessing"
    default_in = script_dir                               # filtered CSV 由本步骤生成，首次出现在此
    default_out = script_dir
    input_dir = Path(args.input_dir) if args.input_dir else default_in
    output_dir = Path(args.output_dir) if args.output_dir else default_out
    output_dir.mkdir(parents=True, exist_ok=True)

    params = dict(w_near=args.w_near, d1=args.d1, d2=args.d2, kappa=args.kappa, lambda_=args.lambda_)
    alpha_const = args.alpha

    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print("Reading inputs...")
    lig, rec, coords, cell_cols = load_inputs(input_dir, preproc_dir)

    n_cells = len(cell_cols)
    xy = cp.asarray(coords[["x", "y"]].to_numpy(), dtype=cp.float32)

    # Prepare cluster indices on CPU for cheap masking
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
        # Compute distance block between sender and receiver sets on the fly
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
            if rc not in idx_by_cluster:
                continue
            if lc == rc:
                continue
            r_idx = idx_by_cluster[rc]
            rec_vec = rec_mat[j, r_idx]
            if not np.any(cp.asnumpy(rec_vec)):
                continue

            w_sr = get_weight(lc, rc)  # (n_sender, n_receiver)
            lig_signal = w_sr.T @ lig_vec  # (n_receiver,)
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
    meta_path = output_dir / "LR_scores_all_pairs_V0_meta_gpu.csv"
    meta_df.to_csv(meta_path, index=False, encoding="utf-8-sig")
    print(f"Saved pair metadata table to {meta_path}")

    print(f"Done. Total pairs: {pair_idx}")


if __name__ == "__main__":
    main()
