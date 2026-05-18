# SpaGAT-CCC

**Spatially-aware Graph Attention Network for Cell-Cell Communication inference**

SpaGAT-CCC is a computational framework for identifying ligand-receptor (LR) interactions between distinct cell types in spatial transcriptomics data. It combines cell-type deconvolution, dynamical sensitivity screening (DGRN-SE), spatial distance-weighted scoring, and a dual-layer multi-head graph attention autoencoder to predict high-confidence cross-cell-type LR pairs.

---

## Datasets

| Dataset | Folder | Type | Notes |
|---|---|---|---|
| BreastCancer1 | `BreastCancer1/` | 10x Visium | Breast Cancer Block A Section 1 |
| BreastCancer2 | `BreastCancer2/` | 10x Visium | Breast Cancer Block A Section 2 |
| seqFISH | `giotto_seqfish/` | seqFISH+ | Mouse cortex spatial transcriptomics |

The same pipeline (Steps 1–9 below) is applied independently to each dataset.

---

## Pipeline Overview

```
Spatial transcriptomics data (10x Visium / seqFISH)
         │
         ▼
Step 1.  make_expr_tables.R
         Build gene__celltype expression matrices (ligand & receptor)
         │
         ▼
Step 2.  preprocess_and_preWAVE.m
         Raw count → wavelet-transformed time-series tensors
         │
         ▼
Step 3.  GCNtest.py
         GCN → averaged gene-gene adjacency matrix
         │
         ▼
Step 4.  c_index0.m
         ARNN network entropy index (parallel, parpool)
         │
         ▼
Step 5.  raodong_point.m
         Perturbation / sensitivity estimation per gene node
         │
         ▼
Step 6.  final_test.m
         Select sensitive gene__celltype nodes → subnetwork
         │
         ▼
Step 7.  make_combo_only.py
         Generate candidate LR pairs → combo_only.csv
         │
         ▼
Step 8.  filter_expr_with_matrix-select-ligand_expr_by_cell_filtered.py
         Filter expression matrices; compute spatial LR prior scores
         │
         ▼
Step 9.  run_demo_2layers_multihead.py
         Dual-layer multi-head GAT autoencoder → pred_scores.csv
         │
         ├──▶ Step 10. eval_other_methods.py      (metric evaluation vs baselines)
         │
         └──▶ Step 11. GCNtest_threshold_eval.py  (threshold sensitivity evaluation)
```

---

## Directory Structure

```
SpaGAT-CCC/
├── BreastCancer1/                         # Full pipeline — BreastCancer1
│   ├── 1.Preprocessing/
│   │   ├── make_expr_tables.R             ★ Step 1
│   │   ├── ligand_expr_by_cell.csv        (generated)
│   │   ├── receptor_expr_by_cell.csv      (generated)
│   │   └── de_coords.csv                  (generated)
│   ├── 2.LR_Screening/
│   │   ├── 1.Data preprocessing/
│   │   │   └── preprocess_and_preWAVE.m   ★ Step 2
│   │   ├── 2.Build a gene network/
│   │   │   └── GCNtest.py                 ★ Step 3
│   │   └── 3.Identify sensitive genes and gene combinations/
│   │       ├── 1.Initial network entropy index calculation/
│   │       │   └── c_index0.m             ★ Step 4
│   │       ├── 2.Disturbance handling/
│   │       │   └── raodong_point.m        ★ Step 5
│   │       └── 3.Subnetwork exploration/
│   │           ├── final_test.m           ★ Step 6
│   │           └── make_combo_only.py     ★ Step 7
│   ├── 3.LR_Scoring/
│   │   └── filter_expr_with_matrix-select-ligand_expr_by_cell_filtered.py  ★ Step 8
│   ├── 4.Model_Training/
│   │   ├── run_demo_2layers_multihead.py  ★ Step 9
│   │   └── STAGATE/
│   │       ├── model_multihead.py
│   │       ├── STAGATE_multihead.py
│   │       ├── Train_STAGATE.py
│   │       └── utils.py
│   └── 5.Results/
│       ├── Metric_Evaluation/
│       │   └── eval_other_methods.py      ★ Step 10 — benchmark metric evaluation
│       └── Threshold_Selection/
│           └── GCNtest_threshold_eval.py  ★ Step 11 — threshold sensitivity evaluation
├── BreastCancer2/                         # Same structure as BreastCancer1
├── giotto_seqfish/                        # Same structure — seqFISH dataset
├── plot/                                  # Figure generation scripts
│   ├── _common.py
│   ├── 01_plot_main_bc1.py
│   ├── 01_plot_main_bc2.py
│   ├── 01_plot_main_seqfish.py
│   ├── 02_plot_spatial_prior_bc1.py
│   ├── 02_plot_spatial_prior_bc2.py
│   ├── 02_plot_spatial_prior_seqfish.py
│   ├── 03_combine_benchmark_summary.py
│   ├── 04_combine_panels_bc1.py
│   ├── 04_combine_panels_bc2.py
│   ├── 04_combine_panels_seqfish.py
│   └── S05_plot_threshold_sensitivity.py
├── simu/                                  # Simulation experiments
├── requirements.txt
└── README.md
```

`★` marks the scripts you run directly.

---

## Execution Order

Run Steps 1–9 independently for each dataset (BC1, BC2, seqFISH), then Steps 10–11 for evaluation.

### Step 1 — Build Expression Tables (R)

```r
# In {Dataset}/1.Preprocessing/
Rscript make_expr_tables.R
```

Outputs: `ligand_expr_by_cell.csv`, `receptor_expr_by_cell.csv`, `de_coords.csv`

---

### Step 2 — Preprocessing & Wavelet Transform (MATLAB)

```matlab
% In {Dataset}/2.LR_Screening/1.Data preprocessing/
run preprocess_and_preWAVE.m
% Raw counts → MATLAB workspace → wavelet-transformed tensors (t_wavelet-all-*.mat)
```

---

### Step 3 — Gene Regulatory Network (Python)

```bash
# In {Dataset}/2.LR_Screening/2.Build a gene network/
python GCNtest.py
# Output: L-averaged_final_adj-.xlsx  (gene-gene adjacency matrix)
```

---

### Step 4 — Network Entropy Index (MATLAB)

```matlab
% In {Dataset}/2.LR_Screening/3.../1.Initial network entropy index calculation/
run c_index0.m
% Uses parpool (parallel); Output: entropy index matrices
```

---

### Step 5 — Perturbation Sensitivity (MATLAB)

```matlab
% In {Dataset}/2.LR_Screening/3.../2.Disturbance handling/
run raodong_point.m
% Output: H_point_value_concat-*.mat
```

---

### Step 6 — Select Sensitive Gene Nodes (MATLAB)

```matlab
% In {Dataset}/2.LR_Screening/3.../3.Subnetwork exploration/
run final_test.m
% Output: adduwavelet-*-merged.mat  (sensitive gene__celltype subnetwork)
```

---

### Step 7 — Generate Candidate LR Pairs (Python)

```bash
# In {Dataset}/2.LR_Screening/3.../3.Subnetwork exploration/
python make_combo_only.py
# Output: combo_only.csv  (LigGene__CellType | RecGene__CellType)
```

---

### Step 8 — Expression Filtering & Spatial LR Scoring (Python)

```bash
# In {Dataset}/3.LR_Scoring/
python filter_expr_with_matrix-select-ligand_expr_by_cell_filtered.py
# Outputs:
#   ligand_expr_by_cell_filtered.csv
#   receptor_expr_by_cell_filtered.csv
#   LR_scores_all_pairs_V0_meta_gpu.csv  (piecewise distance-weighted prior scores)
```

---

### Step 9 — Model Training (Python)

```bash
# In {Dataset}/4.Model_Training/
python run_demo_2layers_multihead.py
# Output: pred_scores.csv  (top-500 predicted LR pairs with attention scores)
```

---

### Step 10 — Metric Evaluation vs Baselines (Python)

Located at `{Dataset}/5.Results/Metric_Evaluation/eval_other_methods.py`.

Computes **AUROC** and **AUPR** for SpaGAT-CCC and baseline methods (COMMOT, Stmlnet, etc.) against validated positive LR pairs from `combo_only.csv`.

```bash
# In {Dataset}/5.Results/Metric_Evaluation/
python eval_other_methods.py
```

---

### Step 11 — Threshold Sensitivity Evaluation (Python)

Located at `{Dataset}/5.Results/Threshold_Selection/GCNtest_threshold_eval.py`.

Scans candidate synchronous thresholds (`[0.4, 0.6, 0.8]`) and scores each threshold configuration on four metrics (stability, density match, connectivity, non-isolated ratio) using a GCN trained on wavelet features from `t_wavelet-all-L.mat`.

```bash
# In {Dataset}/5.Results/Threshold_Selection/
python GCNtest_threshold_eval.py
```

---

### Figure Generation (Python)

After all pipeline steps complete, generate paper figures from `plot/`:

```bash
cd plot

# Per-dataset main figures
python 01_plot_main_bc1.py
python 01_plot_main_bc2.py
python 01_plot_main_seqfish.py

# Spatial prior score raincloud plots
python 02_plot_spatial_prior_bc1.py
python 02_plot_spatial_prior_bc2.py
python 02_plot_spatial_prior_seqfish.py

# Threshold sensitivity supplementary figure
python S05_plot_threshold_sensitivity.py

# Multi-panel composite figures (run after upstream panels exist)
python 04_combine_panels_bc1.py
python 04_combine_panels_bc2.py
python 04_combine_panels_seqfish.py

# Benchmark summary (all datasets)
python 03_combine_benchmark_summary.py
```

---

## Method Description

### Step 1: Data Preprocessing

10x Visium / seqFISH spatial transcriptomics data is taken as input. Cell-type deconvolution (RCTD) maps a single-cell reference atlas to spatial spots. For each ligand/receptor gene, a **cell-type-specific expression matrix** is built using `gene__celltype` composite row keys: only spots assigned to the matching cell type carry real expression values. This prevents signal confusion across cell types.

### Step 2: Wavelet Transform

Raw per-gene expression across spots is treated as a time-series and transformed using discrete wavelet decomposition (`preprocess_and_preWAVE.m`), producing multi-scale tensors (`t_wavelet-all-*.mat`) that capture both local and global expression variation.

### Step 3: Gene Regulatory Network

A Graph Convolutional Network (`GCNtest.py`) learns pairwise gene correlations from the wavelet-transformed expression. The averaged adjacency matrix (`L-averaged_final_adj-.xlsx`) encodes the regulatory topology used by subsequent MATLAB steps.

### Steps 4–6: DGRN-SE Sensitivity Screening

The DGRN-SE algorithm identifies which `gene__celltype` nodes are dynamically sensitive:

- `c_index0.m` — computes ARNN-based network entropy indices under a sliding window (parallelised via `parpool`).
- `raodong_point.m` — estimates each node's response to perturbation.
- `final_test.m` — selects nodes exceeding sensitivity thresholds, forming a subnetwork of candidate senders/receivers.

### Step 7: Candidate LR Pair Generation

`make_combo_only.py` enumerates cross-cell-type pairs from the sensitive subnetwork and removes self-pairings (same sender and receiver cell type), producing `combo_only.csv` with columns `LigGene__CellType | RecGene__CellType`.

### Step 8: Spatial-Distance & Competition-Factor Scoring

For each candidate LR pair, a **piecewise distance-weighted score** is computed:

| Distance range | Weight function |
|---|---|
| d < d₁ (5 000 µm) | constant w_near |
| d₁ ≤ d < d₂ (13 000 µm) | exp(−κ·d) / d |
| d ≥ d₂ | exp(−λ·d) |

A competition factor α = 0.5 attenuates scores to reflect multi-ligand competition for the same receptor. Mean scores across all receiver spots serve as spatial prior strengths for model training.

### Step 9: Dual-Layer Multi-Head GAT Autoencoder

The model (`model_multihead.py`, `STAGATE_multihead.py`, `Train_STAGATE.py`) treats ligand nodes and receptor nodes as two separate graph layers connected by directed cross-layer edges from `combo_only.csv`.

- **Intra-layer edges**: symmetric kNN graph (k = 3) built on expression vector distances.
- **Cross-layer prior weight**: combination of Gaussian expression similarity (σ = 60) and the spatial-competition score, normalised per node.
- **Architecture**: 2-layer GAT encoder (hidden dims [512, 30], 4 attention heads per layer, mean aggregation), symmetric decoder.
- **Loss**: `1 − Pearson(attention, prior_target)` + L2 weight decay (λ = 10⁻⁴). The model is driven entirely to align attention distributions with biologically informed spatial priors.

After 500 training epochs, mean attention weights across 4 heads from the first encoder layer are extracted for all cross-layer (LR) edges. The top 500 pairs sorted by descending score are saved to `pred_scores.csv`.

### Step 10: Metric Evaluation (`eval_other_methods.py`)

Computes AUROC and AUPR for SpaGAT-CCC and competing methods against validated positive LR pairs. Predictions from baseline tools (COMMOT, Stmlnet, etc.) stored under `5.Results/Metric_Evaluation/OtherMethods/` are compared with the same ground-truth set derived from `combo_only.csv`.

### Step 11: Threshold Sensitivity Evaluation (`GCNtest_threshold_eval.py`)

Trains a lightweight GCN on wavelet features across candidate synchronous thresholds (`[0.4, 0.6, 0.8]`) and scores each configuration on four network-quality metrics:

| Metric | Weight |
|---|---|
| Stability | 0.20 |
| Density match | 0.35 |
| Connectivity | 0.25 |
| Non-isolated ratio | 0.20 |

The threshold with the highest composite score is selected for the final pipeline run.

---

## Key Design Choices

| Choice | Rationale |
|---|---|
| `gene__celltype` feature naming | Distinguishes same-gene signals from different senders/receivers without building separate subgraphs |
| Pearson correlation loss | Aligns attention distribution shape with the prior rather than absolute values |
| Piecewise distance weight | Models short-range contact, medium diffusion, and long-range decay separately |
| Dual-layer heterogeneous graph | Explicitly separates ligand space and receptor space; cross-layer edges encode candidate interactions |
| 4-head attention | Captures multiple interaction patterns in parallel; averaging reduces head-specific noise |

---

## Output Files (per dataset)

| File | Location | Description |
|---|---|---|
| `ligand_expr_by_cell.csv` | `1.Preprocessing/` | gene__celltype × spot expression matrix (ligands) |
| `receptor_expr_by_cell.csv` | `1.Preprocessing/` | gene__celltype × spot expression matrix (receptors) |
| `de_coords.csv` | `1.Preprocessing/` | Spatial coordinates + cell-type labels |
| `t_wavelet-all-*.mat` | `2.LR_Screening/1.Data preprocessing/` | Wavelet-transformed expression tensors |
| `L-averaged_final_adj-.xlsx` | `2.LR_Screening/2.Build a gene network/` | GCN-learned gene adjacency matrix |
| `adduwavelet-*-merged.mat` | `2.LR_Screening/3.…/3.Subnetwork exploration/` | Sensitive gene__celltype subnetwork |
| `combo_only.csv` | `2.LR_Screening/3.…/3.Subnetwork exploration/` | Candidate LR pairs |
| `ligand_expr_by_cell_filtered.csv` | `3.LR_Scoring/` | Filtered ligand expression matrix |
| `receptor_expr_by_cell_filtered.csv` | `3.LR_Scoring/` | Filtered receptor expression matrix |
| `LR_scores_all_pairs_V0_meta_gpu.csv` | `3.LR_Scoring/` | Spatial-competition prior scores |
| `pred_scores.csv` | `4.Model_Training/` | Final predicted LR pair scores (top 500) |

---

## Requirements

### R
- `spacexr` (RCTD)
- `Seurat`

### MATLAB
- Parallel Computing Toolbox (`parpool` used in `c_index0.m`)
- Signal Processing Toolbox (wavelet functions)

### Python

```
torch >= 1.13
tensorflow >= 2.10   # runs in TF1 compatibility mode
scanpy
anndata
scikit-learn
scipy
pandas
numpy
cupy-cuda11x         # match your CUDA version
tqdm
```

Install:

```bash
pip install torch tensorflow scanpy anndata scikit-learn scipy pandas numpy tqdm
pip install cupy-cuda11x   # adjust suffix to your CUDA version (e.g. cupy-cuda12x)
```

---

## Figure Scripts (`plot/`)

| Script | Datasets | Output |
|---|---|---|
| `01_plot_main_bc1.py` | BC1 | `01_main_figure_bc1.png/.eps` |
| `01_plot_main_bc2.py` | BC2 | `01_main_figure_bc2.png/.eps` |
| `01_plot_main_seqfish.py` | seqFISH | `01_main_figure_seqfish.png/.eps` |
| `02_plot_spatial_prior_bc1.py` | BC1 | `02_spatial_prior_bc1.png/.pdf/.eps` |
| `02_plot_spatial_prior_bc2.py` | BC2 | `02_spatial_prior_bc2.png/.pdf/.eps` |
| `02_plot_spatial_prior_seqfish.py` | seqFISH | `02_spatial_prior_seqfish.png/.pdf/.eps` |
| `S05_plot_threshold_sensitivity.py` | all | `S05_threshold_sensitivity.png` |
| `04_combine_panels_bc1.py` | BC1 | `04_combined_panels_bc1.png/.eps` |
| `04_combine_panels_bc2.py` | BC2 | `04_combined_panels_bc2.png/.eps` |
| `04_combine_panels_seqfish.py` | seqFISH | `04_combined_panels_seqfish.png/.eps` |
| `03_combine_benchmark_summary.py` | all | `03_benchmark_summary.png/.pdf/.eps` |

`_common.py` provides shared color palettes and style settings; keep it in the same `plot/` directory.

---

## Citation

If you use SpaGAT-CCC in your work, please cite this repository.
