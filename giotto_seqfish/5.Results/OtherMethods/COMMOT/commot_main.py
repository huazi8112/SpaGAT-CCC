import os
import time
from pathlib import Path

import psutil
import csv
import gc
import ot
import pickle
import anndata
import scanpy as sc
import pandas as pd
import numpy as np
from scipy import sparse
from scipy.stats import spearmanr, pearsonr
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import commot as ct

current_dir = os.getcwd()
print("current path:", current_dir)

def show_info(start):
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    memory = info.uss/1024/1024/1024
    return memory

start = show_info('strat')
start_time = time.time()

base_dir = Path(__file__).resolve().parent
benchmark_dir = base_dir.parent.parent
dataset_root = benchmark_dir.parent  # giotto_seqfish
data_dir = base_dir.parent / "InputData"

seqfish_dir = data_dir / "seqfish"
combo_subset_dir = seqfish_dir / "combo_subset_A3"

expression_file = combo_subset_dir / "cortex_svz_expression_combo.txt"
coord_file = combo_subset_dir / "cortex_svz_centroids_coord_combo.txt"
annot_file = combo_subset_dir / "cortex_svz_centroids_annot_combo.txt"
de_coords_file = dataset_root / "1.Preprocessing" / "de_coords_matched.csv"

ligand_filtered_path = dataset_root / "3.LR_Scoring" / "ligand_expr_by_cell_filtered-A3.csv"
receptor_filtered_path = dataset_root / "3.LR_Scoring" / "receptor_expr_by_cell_filtered-A3.csv"

required_paths = [
    expression_file,
    coord_file,
    annot_file,
    de_coords_file,
    ligand_filtered_path,
    receptor_filtered_path,
]
for fp in required_paths:
    if not fp.exists():
        raise FileNotFoundError(f"Required file not found: {fp}")

res_path = base_dir / "result"
res_path.mkdir(parents=True, exist_ok=True)

print(f"Loading combo subset expression from {expression_file}")
expression_df = pd.read_csv(expression_file, sep="\t")
if expression_df.columns[0].lower() != "gene":
    raise ValueError(f"Expected first column to be 'gene' in {expression_file}")

gene_names = expression_df.iloc[:, 0].astype(str).tolist()
barcode_names = expression_df.columns[1:].astype(str).tolist()
expression_values = expression_df.iloc[:, 1:].to_numpy(dtype=float).T  # cells x genes

adata = anndata.AnnData(X=sparse.csr_matrix(expression_values))
adata.obs_names = barcode_names
adata.var_names = [str(g) for g in gene_names]
adata.var_names_make_unique()

coord_df = pd.read_csv(coord_file, sep="\t")
annot_df = pd.read_csv(annot_file, sep="\t")
de_coords = pd.read_csv(de_coords_file)

coord_df = coord_df.rename(columns={"ID": "Barcode", "X": "x_raw", "Y": "y_raw"})
annot_df = annot_df.rename(columns={"ID": "Barcode", "cell_types": "celltype_raw"})
de_coords = de_coords.rename(columns={"Barcode": "Barcode", "x": "x", "y": "y", "Cluster": "celltype"})

meta_df = pd.DataFrame({"Barcode": adata.obs_names})
meta_df = meta_df.merge(coord_df[["Barcode", "x_raw", "y_raw"]], on="Barcode", how="left")
meta_df = meta_df.merge(annot_df[["Barcode", "celltype_raw"]], on="Barcode", how="left")
meta_df = meta_df.merge(de_coords[["Barcode", "x", "y", "celltype"]], on="Barcode", how="left")

missing_coords = meta_df[["x", "y"]].isna().any(axis=1)
if missing_coords.any():
    fallback_count = int(missing_coords.sum())
    print(f"Using raw combo_subset coordinates for {fallback_count} cells missing in de_coords_matched.csv")
    meta_df.loc[missing_coords, "x"] = meta_df.loc[missing_coords, "x_raw"]
    meta_df.loc[missing_coords, "y"] = meta_df.loc[missing_coords, "y_raw"]

missing_celltype = meta_df["celltype"].isna()
if missing_celltype.any():
    fallback_ct = int(missing_celltype.sum())
    print(f"Using annot combo cell types for {fallback_ct} cells missing in de_coords_matched.csv")
    meta_df.loc[missing_celltype, "celltype"] = meta_df.loc[missing_celltype, "celltype_raw"]

valid_mask = ~(meta_df[["x", "y", "celltype"]].isna().any(axis=1))
if not valid_mask.all():
    drop_count = int((~valid_mask).sum())
    print(f"Dropping {drop_count} cells without complete coordinates/celltype metadata")
    adata = adata[valid_mask.values].copy()
    meta_df = meta_df.loc[valid_mask].reset_index(drop=True)

adata.obsm['spatial'] = meta_df[["x", "y"]].to_numpy(dtype=float)
adata.obs['celltype'] = pd.Categorical(meta_df['celltype'].astype(str).values)

# Preprocessing the data
adata.var_names_make_unique()
adata.raw = adata
sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)

adata_dis500 = adata.copy()
print(adata_dis500)

db_name = 'custom'

# load custom LR list (A3 combos): format Ligand__Sender|Receptor__Receiver
combo_path = (dataset_root / "2.LR_Screening"
              / "3.Identify sensitive genes and gene combinations"
              / "3.Subnetwork exploration" / "combo_only-A3.csv")
if not combo_path.exists():
    raise FileNotFoundError(f"Custom LR list not found: {combo_path}")
combo_df = pd.read_csv(combo_path)
if "combo" not in combo_df.columns:
    combo_df = pd.read_csv(combo_path, header=None, names=["combo"])
if len(combo_df) and str(combo_df.iloc[0]["combo"]).strip().lower() == "combo":
    combo_df = combo_df.iloc[1:].reset_index(drop=True)
combo_df = combo_df[combo_df["combo"].astype(str).str.contains("|", na=False)].reset_index(drop=True)
ligands = []
receptors = []
for combo in combo_df['combo']:
    ligand_part, receptor_part = str(combo).split('|', 1)
    ligands.append(ligand_part.split('__')[0])
    receptors.append(receptor_part.split('__')[0])

df_custom = pd.DataFrame({
    'ligand': ligands,
    'receptor': receptors,
    'pathway_name': ['CUSTOM'] * len(ligands),
    'annotation': ['Custom'] * len(ligands),
})

# map combo genes to expression genes using case-insensitive matching
expr_gene_upper_to_original = {}
for gene in adata_dis500.var_names.astype(str):
    expr_gene_upper_to_original.setdefault(gene.upper(), gene)

df_custom['ligand'] = df_custom['ligand'].astype(str).str.upper().map(expr_gene_upper_to_original)
df_custom['receptor'] = df_custom['receptor'].astype(str).str.upper().map(expr_gene_upper_to_original)
df_custom = df_custom.dropna(subset=['ligand', 'receptor']).drop_duplicates().reset_index(drop=True)

print('custom LR database shape:', df_custom.shape)

# spatial communication inference using custom database (no filtering to keep all pairs)
ct.tl.spatial_communication(adata_dis500, database_name=db_name,
                            df_ligrec=df_custom,
                            dis_thr=500,
                            heteromeric=True,
                            pathway_sum=True)

ct.tl.communication_direction(adata_dis500, database_name=db_name, pathway_name=None, k=5)  # type: ignore[arg-type]
adata_dis500.write(res_path / 'adata_pw.h5ad')
lr_keys = list(df_custom[['ligand', 'receptor']].itertuples(index=False, name=None))

result = []
for lr in lr_keys:
    print('calculate the communication score of', lr)
    ct.tl.cluster_communication(adata_dis500, lr_pair=lr, database_name=db_name, pathway_name=None,  # type: ignore[arg-type]
                                clustering='celltype',
                                n_permutations=100)
    cluster_key = f'commot_cluster-celltype-{db_name}-{lr[0]}-{lr[1]}'
    if cluster_key not in adata_dis500.uns:
        print(f"skip {lr}: missing {cluster_key}")
        continue

    comm_mtx = adata_dis500.uns[cluster_key]['communication_matrix']
    comm_mtx = comm_mtx.reset_index()
    comm_mtx.rename(columns={'index': 'Sender'}, inplace=True)
    comm_mtx_df = comm_mtx.melt(id_vars='Sender', var_name='Receiver', value_name='score')
    comm_mtx_df['Ligand'] = lr[0]
    comm_mtx_df['Receptor'] = lr[1]
    result.append(comm_mtx_df)

if len(result) == 0:
    raise RuntimeError('No communication matrices were produced for mapped LR pairs.')

result = pd.concat(result, ignore_index=True)
end = show_info('end')
end_time = time.time()
run_time = (end_time - start_time) / 60
print(f"Training time is: {run_time} mins")
print('total memory used '+str(end-start) + 'GB')
adata_dis500.write(res_path / 'adata_pw_new.h5ad')
result.to_csv(res_path / 'result.csv', index=False)

summary_name = 'commot-'+db_name+'-sum-'+'receiver'
summary_abrv = 'r'
lr_pair: tuple = ('total','total')
comm_sum = np.asarray(
    adata_dis500.obsm[summary_name][summary_abrv+'-'+lr_pair[0]+'-'+lr_pair[1]]
).reshape(-1,1)
cell_weight = np.ones_like(comm_sum).reshape(-1,1)

np.savetxt(res_path / 'comm_sum.csv', comm_sum, delimiter=',', fmt='%.6f')
np.savetxt(res_path / 'cell_weight.csv', cell_weight, delimiter=',', fmt='%d')


