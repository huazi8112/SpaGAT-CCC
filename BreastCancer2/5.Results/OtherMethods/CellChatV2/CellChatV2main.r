library(Seurat)
library(ggplot2)
library(patchwork)
library(dplyr)
library(CellChat)
library(readr)
library(tibble)
library(tidyr)
library(lobstr)

rm(list = ls())
gc()

# Allow large globals to be shipped to future workers (CellChat exports a big Seurat object)
options(future.globals.maxSize = 16 * 1024^3)

script_dir <- local({
  cmd <- commandArgs(trailingOnly = FALSE)
  idx <- grep("^--file=", cmd)
  file_arg <- if (length(idx)) sub("^--file=", "", cmd[idx[1]]) else ""
  if (nzchar(file_arg)) {
    dirname(normalizePath(file_arg))
  } else {
    src_file <- tryCatch(
      normalizePath(sys.frames()[[1]]$ofile),
      error = function(e) ""
    )
    if (nzchar(src_file)) dirname(src_file) else normalizePath(getwd())
  }
})
dataset_root <- normalizePath(file.path(script_dir, "..", "..", ".."))
benchmark_root <- file.path(dataset_root, "5.Results")

visium_dir <- file.path(benchmark_root, "OtherMethods", "InputData", "Parent_Visium_Human_BreastCancer")
preprocess_dir <- file.path(benchmark_root, "OtherMethods", "InputData", "preprocess")

ptm = Sys.time()

### load st data
st_count <- Read10X(file.path(visium_dir, "filtered_feature_bc_matrix_combo"),
                    gene.column = 2)

st_decomp <- read_csv(file.path(preprocess_dir, "celltype_predictions.csv"), show_col_types = FALSE) %>%
  column_to_rownames(var = "...1") %>%
  as.data.frame()

st_coef <- as.data.frame(t(st_decomp))
colnames(st_coef) <- gsub('\\.', '_', colnames(st_coef))

common_cells <- intersect(colnames(st_count), rownames(st_coef))
if (length(common_cells) == 0) {
  stop("No overlapping spot barcodes between Visium matrix and celltype_predictions.csv")
}
missing_in_deconv <- setdiff(colnames(st_count), rownames(st_coef))
if (length(missing_in_deconv) > 0) {
  warning(
    "Dropping ", length(missing_in_deconv),
    " spot(s) present in the matrix but absent from celltype_predictions.csv"
  )
}
st_count <- st_count[, common_cells, drop = FALSE]
st_coef <- st_coef[common_cells, , drop = FALSE]

# BC2 combo 子矩阵特有：部分 spot 在 274 个 combo 基因上 count 全为 0，
# SCTransform 的 log_umi = log10(colSums) 会得到 -Inf 并报错（BC1 矩阵无此问题）。
nonzero_spots <- colSums(st_count) > 0
if (any(!nonzero_spots)) {
  warning(sprintf(
    "%d spot(s) removed: zero total counts in combo-gene matrix (required for SCTransform).",
    sum(!nonzero_spots)
  ))
  st_count <- st_count[, nonzero_spots, drop = FALSE]
  st_coef <- st_coef[colnames(st_count), , drop = FALSE]
}

predicted_labels <- colnames(st_coef)[apply(st_coef, 1, which.max)]
meta <- data.frame(labels = predicted_labels, row.names = rownames(st_coef))
unique(meta$labels) # check the cell labels

st_metadata <- st_coef
st_metadata$predicted_label <- predicted_labels

st.se <- CreateSeuratObject(st_count, meta.data = st_metadata, assay = 'Spatial')
st.se <- SCTransform(st.se, assay = "Spatial", verbose = FALSE) 

if ("LayerData" %in% getNamespaceExports("SeuratObject")) {
  data.input <- SeuratObject::LayerData(st.se, assay = "SCT", layer = "data")
} else if ("layer" %in% names(formals(SeuratObject::GetAssayData))) {
  data.input <- SeuratObject::GetAssayData(st.se, layer = "data", assay = "SCT")
} else {
  data.input <- Seurat::GetAssayData(st.se, slot = "data", assay = "SCT")
}


scale.factors = jsonlite::fromJSON(txt = file.path(visium_dir, 'spatial', 'scalefactors_json.json'))
scale.factors = list(spot.diameter = 65, spot = scale.factors$spot_diameter_fullres, # these two information are required
                     fiducial = scale.factors$fiducial_diameter_fullres, hires = scale.factors$tissue_hires_scalef, lowres = scale.factors$tissue_lowres_scalef # these three information are not required
)

positions_ls <- read_csv(file.path(visium_dir, 'spatial', 'tissue_positions_list.csv'),
                         col_names = FALSE) 
spatial.locs <- data.frame(spot=positions_ls$X1, imagerow=positions_ls$X5,imagecol=positions_ls$X6)
spatial.locs <- spatial.locs[which(spatial.locs$spot %in% colnames(st_count)),]
rownames(spatial.locs) <- NULL
spatial.locs <- column_to_rownames(spatial.locs, var = 'spot')
spatial.locs <- as.matrix(spatial.locs)

### Creat a CellChat object
cellchat <- createCellChat(object = data.input, meta = meta, group.by = "labels",
                           datatype = "spatial", coordinates = spatial.locs, scale.factors = scale.factors)

# Set the ligand-receptor interaction database：仅使用 combo_only-A2.csv 中的组合
combo_path <- normalizePath(file.path(
  dataset_root,
  "2.LR_Screening",
  "3.Identify sensitive genes and gene combinations",
  "3.Subnetwork exploration",
  "combo_only-A2.csv"
), mustWork = FALSE)
if (!file.exists(combo_path)) {
  stop("combo_only file not found: ", combo_path)
}
combo_df <- read_csv(combo_path, col_names = "combo", show_col_types = FALSE)
combo_parsed <- combo_df %>%
  tidyr::separate(combo, into = c("ligand_part", "receptor_part"), sep = "\\|") %>%
  mutate(
    ligand   = sub("__.*", "", ligand_part),
    receptor = sub("__.*", "", receptor_part),
    sender   = sub(".*__", "", ligand_part),
    receiver = sub(".*__", "", receptor_part)
  )

interaction_df <- combo_parsed %>%
  transmute(
    interaction_name   = paste0(ligand, "_", receptor),
    pathway_name       = "CUSTOM",
    ligand             = ligand,
    receptor           = receptor,
    agonist            = "",
    antagonist         = "",
    co_A_receptor      = "",
    co_I_receptor      = "",
    evidence           = "custom list",
    annotation         = "Custom",
    interaction_name_2 = paste0(ligand, " - (", receptor, ")")
  )

gene_symbols <- unique(c(combo_parsed$ligand, combo_parsed$receptor))
gene_info <- data.frame(
  Symbol = gene_symbols,
  Name = gene_symbols,
  EntrezGene.ID = NA_integer_,
  Ensembl.Gene.ID = NA_character_,
  MGI.ID = NA_character_,
  Gene.group.name = NA_character_,
  stringsAsFactors = FALSE
)

CellChatDB.use <- list(
  interaction = interaction_df,
  complex     = data.frame(),
  cofactor    = data.frame(),
  geneInfo    = gene_info
)

cellchat@DB <- CellChatDB.use

### Preprocessing the expression data for cell-cell communication analysis
# subset the expression data of signaling genes for saving computation cost
cellchat <- subsetData(cellchat) # This step is necessary even if using the whole database
available_cores <- parallel::detectCores()
future::plan("multisession", workers = 12)     
# available_cores
cellchat <- identifyOverExpressedGenes(cellchat)
cellchat <- identifyOverExpressedInteractions(cellchat)

### Inference of cell-cell communication network

compute_args <- list(object = cellchat,
                     type = "truncatedMean",
                     trim = 0.1,
                     distance.use = TRUE,
                     scale.distance = 0.01)
if ("interaction.range" %in% names(formals(CellChat::computeCommunProb))) {
  compute_args$interaction.range <- 250
}
cellchat <- do.call(CellChat::computeCommunProb, compute_args)
# Filter out the cell-cell communication if there are only few number of cells in certain cell groups
cellchat <- filterCommunication(cellchat, min.cells = 1)
# Infer the cell-cell communication at a signaling pathway level
# Skipping computeCommunProbPathway because our custom DB only has one pathway; CellChat 2.x drops the
# pathway dimension in that case and aperm() fails with "'perm' length 3 (!= 2)".
# If multiple pathways are added later, re-enable the call below.
# cellchat <- computeCommunProbPathway(cellchat)
# Calculate the aggregated cell-cell communication network
# cellchat <- aggregateNet(cellchat)
result_raw <- subsetCommunication(cellchat, slot.name = "net", thresh = 1)

# 构造 key 并左连接 combo 列表，缺失补 0 分
result_raw$key_combo <- paste0(result_raw$ligand, "__", result_raw$source, "|", result_raw$receptor, "__", result_raw$target)
combo_table <- combo_parsed %>%
  transmute(Sender = sender, Receiver = receiver, Ligand = ligand, Receptor = receptor,
            key_combo = combo_df$combo)

result_join <- combo_table %>%
  left_join(result_raw[, c("key_combo", "prob")], by = "key_combo") %>%
  rename(LRscore = prob)

result_join$LRscore[is.na(result_join$LRscore)] <- 0
result <- result_join

used.time = Sys.time() - ptm
used.memory <- mem_used()
print(as.numeric(used.time, units = "secs"))

result <- result %>% select(Sender, Receiver, Ligand, Receptor, LRscore)
result <- result[which(result$Sender != result$Receiver),]
result$Ligand <- gsub('_', '&', result$Ligand)
result$Receptor <- gsub('_', '&', result$Receptor)
result$all <- paste(result$Sender, result$Ligand, result$Receiver, result$Receptor, sep = '_')
result <- dplyr::distinct(result, all, .keep_all = TRUE)

result_record <- list(result=result, 
                      used_time = paste0(round(used.time/60,3), ' min'), 
                      used_memory =  round(used.memory/1024/1024/1024,3))

result_dir <- file.path(script_dir, "result")
if (!dir.exists(result_dir)) {
  dir.create(result_dir, recursive = TRUE)
}

saveRDS(result_record, file = file.path(result_dir, "result.rds"))
result_record <- result
