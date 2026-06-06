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

cmd <- commandArgs(trailingOnly = FALSE)
idx <- grep("^--file=", cmd)
file_arg <- if (length(idx)) sub("^--file=", "", cmd[idx[1]]) else ""
script_dir <- if (nzchar(file_arg)) {
  dirname(normalizePath(file_arg))
} else {
  # Detect path when loaded via source()
  src_file <- tryCatch(
    normalizePath(sys.frames()[[1]]$ofile),
    error = function(e) ""
  )
  if (nzchar(src_file)) dirname(src_file) else normalizePath(getwd())
}
metric_root <- normalizePath(file.path(script_dir, "..", ".."))

input_data <- file.path(metric_root, "OtherMethods", "InputData")
visium_dir <- file.path(input_data, "Breast_Cancer_Block_A_Section_1")
preprocess_dir <- file.path(input_data, "preprocess")

ptm = Sys.time()

### load st data
st_count <- Read10X(file.path(visium_dir, "filtered_feature_bc_matrix_combo"),
                    gene.column = 2)

st_decomp <- read_csv(file.path(preprocess_dir, "celltype_predictions.csv"), show_col_types = FALSE) %>%
  as.data.frame()
id_col <- names(st_decomp)[1]
st_decomp <- column_to_rownames(st_decomp, var = id_col)

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

### load spatial imaging information
scale.factors = jsonlite::fromJSON(txt = file.path(visium_dir, 'spatial', 'scalefactors_json.json'))
scale.factors = list(spot.diameter = 65, spot = scale.factors$spot_diameter_fullres, # these two information are required
                     fiducial = scale.factors$fiducial_diameter_fullres, hires = scale.factors$tissue_hires_scalef, lowres = scale.factors$tissue_lowres_scalef # these three information are not required
)

positions_ls <- read_csv(file.path(visium_dir, 'spatial', 'tissue_positions_list.csv'),
                         col_names = FALSE,
                         show_col_types = FALSE) 
spatial.locs <- data.frame(spot=positions_ls$X1, imagerow=positions_ls$X5,imagecol=positions_ls$X6)
spatial.locs <- spatial.locs[which(spatial.locs$spot %in% colnames(st_count)),]
rownames(spatial.locs) <- NULL
spatial.locs <- column_to_rownames(spatial.locs, var = 'spot')
spatial.locs <- as.matrix(spatial.locs)

### Creat a CellChat object
cellchat <- createCellChat(object = data.input, meta = meta, group.by = "labels",
                           datatype = "spatial", coordinates = spatial.locs, scale.factors = scale.factors)

# Set the ligand-receptor interaction database：仅使用 combo_only.csv 中的 1610 对
dataset_root <- normalizePath(file.path(script_dir, "..", "..", "..", ".."))
combo_path <- file.path(dataset_root, "2.LR_Screening",
                        "3.Identify sensitive genes and gene combinations",
                        "3.Subnetwork exploration", "combo_only.csv")
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
cellchat <- filterCommunication(cellchat, min.cells = 10)
# Infer the cell-cell communication at a signaling pathway level
# Skipping computeCommunProbPathway because our custom DB only has one pathway; CellChat 2.x drops the
# pathway dimension in that case and aperm() fails with "'perm' length 3 (!= 2)".
# If multiple pathways are added later, re-enable the call below.
# cellchat <- computeCommunProbPathway(cellchat)
# Calculate the aggregated cell-cell communication network
# cellchat <- aggregateNet(cellchat)
result_raw <- subsetCommunication(cellchat, slot.name = "net")

# 构造 key 并左连接 1610 组合，缺失补 0 分
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
### visualize

# ptm = Sys.time()
# 
# groupSize <- as.numeric(table(cellchat@idents))
# par(mfrow = c(1,2), xpd=TRUE)
# 
# netVisual_circle(cellchat@net$count, vertex.weight = rowSums(cellchat@net$count), weight.scale = T, label.edge= F, title.name = "Number of interactions")
# netVisual_circle(cellchat@net$weight, vertex.weight = rowSums(cellchat@net$weight), weight.scale = T, label.edge= F, title.name = "Interaction weights/strength")
# 
# netVisual_heatmap(cellchat, measure = "count", color.heatmap = "Blues")
# 
# pathways.show <- c("IGF") 
# # Circle plot
# par(mfrow=c(1,1))
# netVisual_aggregate(cellchat, signaling = pathways.show, layout = "circle")
# 
# par(mfrow=c(1,1))
# netVisual_aggregate(cellchat, signaling = pathways.show, layout = "spatial", edge.width.max = 2, vertex.size.max = 1, alpha.image = 0.2, vertex.label.cex = 3.5)




