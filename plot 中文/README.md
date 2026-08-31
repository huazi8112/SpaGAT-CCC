# plot/

Figure generation scripts for SpaGAT-CCC.  
Run all Python scripts from this directory (`cd plot`).

---

## Scripts

### Figure class 01 — Main composite figures (one per dataset)

| Script | Dataset | Panels | Output |
|--------|---------|--------|--------|
| `01_plot_main_bc1.py` | BreastCancer1 | A: Chord diagram (top 100 CT-level strength, pycirclize) · B: Dot plot (top 30 LR pairs across CT pairs) · C: Hierarchical Edge Bundling (top 60 gene-level connections) | `01_main_figure_bc1.png/.eps`<br>`01_main_figure_bc1_ABC_genes_values.csv` |
| `01_plot_main_bc2.py` | BreastCancer2 | A: Heatmap (CT×CT aggregated attention, top 100) · B: Sankey/Alluvial flow (top 30, sender→receiver CT) · C: Hierarchical Edge Bundling (top 30 gene-level) | `01_main_figure_bc2.png/.eps`<br>`01_main_figure_bc2_ABC_genes_values.csv` |
| `01_plot_main_seqfish.py` | seqFISH | A: Chord diagram · B: Dot plot · C: Network graph (top predicted LR pairs) | `01_main_figure_seqfish.png/.eps`<br>`01_main_figure_seqfish1_ABC_genes_values.csv` |

Input for all three: `{Dataset}/4.Model_Training/pred_scores.csv`  
All three import `_common.py` for shared cell-type colors and style (`CT_COLORS`, `apply_paper_style`).

---

### Figure class 02 — Spatial prior score raincloud plots (Section 3.3)

| Script | Dataset | Description | Output |
|--------|---------|-------------|--------|
| `02_plot_spatial_prior_bc1.py` | BreastCancer1 | Raincloud plot (half-violin KDE + inner mini-box + jittered strip) of spatial prior score distributions across sender→receiver CT pairs, sorted by median. | `02_spatial_prior_bc1.png/.pdf/.eps` |
| `02_plot_spatial_prior_bc2_seqfish_S2.py` | BC2 + seqFISH | **Supplementary Figure S2.** Combined two-panel figure: Panel A = BC2 raincloud, Panel B = seqFISH raincloud (four sub-rows split by median quartile). Merges the separate BC2 and seqFISH raincloud scripts into a single layout. | `02_spatial_prior_bc2_seqfish_S2.png/.pdf/.eps` |

Inputs: `{Dataset}/3.LR_Scoring/LR_scores_all_pairs_V0_meta_gpu.csv` and `{Dataset}/4.Model_Training/combo_only.csv`  
Both import `_common.py` for `CT_COLORS`, `CT_COLORS_SEQFISH`, `CT_CHORD_ABBREV_SEQFISH`.

---

### Figure class 04 — Multi-panel DGRN-SE composite figures

| Script | Dataset | Panels | Output |
|--------|---------|--------|--------|
| `04_combine_panels_bc1.py` | BreastCancer1 | Six-panel vectorized layout: GRN structure · entropy index · perturbation sensitivity · subnetwork · threshold scan · sensitive-gene summary. All elements drawn as true vector (matplotlib subfigures, no PNG intermediate). | `04_combined_panels_bc1.eps/.png` |
| `04_combine_panels_bc2.py` | BreastCancer2 | Same six-panel layout as BC1. | `04_combined_panels_bc2.eps/.png` |
| `04_combine_panels_seqfish.py` | seqFISH | Same six-panel layout as BC1. | `04_combined_panels_seqfish.eps/.png` |

Inputs read from `{Dataset}/2.LR_Screening/` (entropy, perturbation, subnetwork `.mat` files).

---

### Supplementary Figure S05 — Threshold sensitivity analysis

| Script | Description | Output |
|--------|-------------|--------|
| `S05_plot_threshold_sensitivity.py` | Reproduces the composite-score bar chart (Panel A) and six-axis radar charts (Panels B/C/D) for BreastCancer1 L100 vs R100 features. Evaluates synchronous thresholds τ ∈ {0.4, 0.6, 0.8} (3 runs each) on four metrics: Stability 0.20 · Density-Match 0.35 · Connectivity 0.25 · Non-isolated 0.20. | `S05_threshold_sensitivity.png` |

Required inputs (must exist on disk):
- `BreastCancer1/2.LR_Screening/1.Data preprocessing/t_wavelet-all-L.mat`
- `BreastCancer1/2.LR_Screening/1.Data preprocessing/t_wavelet-all-R.mat`

---

### Benchmark summary — Figure class 03

| Script | Output |
|--------|--------|
| `03_combine_benchmark_summary.py` | `03_benchmark_summary.png/.pdf/.eps` |

Single-page layout combining benchmark workflow diagram, performance bar chart, and methods comparison table across all datasets. Imports `_common.py`.

---

## Execution order

```bash
cd plot

# Step 1 — Main figures (per dataset)
python 01_plot_main_bc1.py
python 01_plot_main_bc2.py
python 01_plot_main_seqfish.py

# Step 2 — Spatial prior raincloud
python 02_plot_spatial_prior_bc1.py          # BC1 (main figure)
python 02_plot_spatial_prior_bc2_seqfish_S2.py  # BC2 + seqFISH (Supplementary S2)

# Step 3 — Threshold sensitivity (Supplementary S05, BC1 only)
python S05_plot_threshold_sensitivity.py

# Step 4 — DGRN-SE composite panels (depends on LR_Screening .mat files)
python 04_combine_panels_bc1.py
python 04_combine_panels_bc2.py
python 04_combine_panels_seqfish.py

# Step 5 — Benchmark summary (depends on all evaluation results)
python 03_combine_benchmark_summary.py
```

---

## Simulation experiments (`simu/`)

| Script | Language | Description |
|--------|----------|-------------|
| `simu/main_for_simulation_data1 win` | R | Simulation-data validation. Trains a **Random Forest** classifier (via `ranger` + `caret`, parallelised with `doParallel`/`doSNOW`) on synthetic LR pair features. Evaluates with AUROC/AUPR (`ROCR`, `Metrics`), generates PIM (permutation importance) figures and model comparison plots (`ggplot2`, `cowplot`, `ggsci`). Outputs written to `simu/runModel/`, `simu/getPIM/`, `simu/figure/`. |

Run from the `simu/` directory:
```r
# In RStudio: open the file and click Source
# From command line:
Rscript "simu/main_for_simulation_data1 win"
```

---

## Shared utilities

- `_common.py` — cell-type color palettes (`CT_COLORS`, `CT_COLORS_SEQFISH`, `CT_CHORD_ABBREV_SEQFISH`), font-size presets (`FS`, `FS_FIG678`, `FS_FIG8`), and `apply_paper_style()` used by all figure scripts.
- `ccc_plots/` — supplementary panel scripts for additional figures.
- `11/` — alternative version of `01_plot_main_seqfish.py` (development copy).
