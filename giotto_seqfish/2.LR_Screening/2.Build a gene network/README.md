# Gene Network Building Module - hESC Dataset

## Overview
This module implements Graph Convolutional Network (GCN) based gene regulatory network reconstruction for human embryonic stem cell (hESC) data. The module processes wavelet-transformed time series data and clustering results to build predictive gene interaction networks.

## Folder Structure

### Input Data (`data/` folder)
- `t_wavelet-hESC.mat`: Wavelet-processed time series gene expression data
- `kmeans_wavelet-hESC.mat`: K-means clustering results from previous step
- `merged_data-hESC.mat`: Integrated dataset combining multiple data sources
- `hESC-averaged_final_adj-k*.xlsx`: Ground truth adjacency matrices for each cluster
- `hESC-maprho-k*.xlsx`: Correlation-based connectivity matrices for each cluster
- Supporting Excel files (1.xlsx to 8.xlsx, merge.xlsx): Additional data matrices

### Output Results (`results/` folder)
- `prewavelet-L-A3.mat`: Written by `GCNtest.py` for downstream `c_index0.m` (with `L-averaged_final_adj-.xlsx`)
- `hESC-averaged_final_adj-k*.xlsx`: GCN-predicted adjacency matrices
- `final_score_k*.pth`: Trained PyTorch GCN model checkpoints

## Workflow

### Step 1: GCN Network Training (Python)
```python
# Configure cluster number and run GCN training
python GCNtest.py
```
To process different clusters:
1. Edit the `K_VALUE` parameter in the script (valid range: 1-8)
2. Execute the script for GCN model training

The GCN training process:
- Loads preprocessed data from `data/` folder
- Implements graph autoencoder architecture
- Performs multiple training runs with different initializations
- Saves trained models and predictions to `results/` folder
- Generates performance metrics and validation results

## Files Description

- `GCNtest.py`: Graph Convolutional Network training; writes `L-averaged_final_adj-.xlsx` and `prewavelet-L-A3.mat`
- `README.md`: This documentation file

## Configuration

### GCNtest.py Parameters
- `K_VALUE`: Cluster number to process (1-8)
- `GENE_SIZE`: Number of genes (500)
- `EPOCHS`: Training epochs (3000)
- `LEARNING_RATE`: Learning rate (0.001)
- `THRESHOLD`: Binarization threshold (0.6)