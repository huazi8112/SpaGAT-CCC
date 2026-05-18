inp <- tryCatch({
  src <- sys.frame(1)$ofile
  if (!is.null(src)) dirname(normalizePath(src)) else getwd()
}, error = function(...) getwd())

env <- new.env()
load(file.path(inp, "giotto_seqfish_output.rda"), envir = env)

cnt <- env[["df_count"]]
norm_expr <- env[["df_norm"]]
anno <- env[["df_anno"]]
ligs <- env[["Ligs_expr_list"]]
recs <- env[["Recs_expr_list"]]
loca <- env[["df_loca"]]

cat("loaded giotto_seqfish_output.rda from:", inp, "\n")
cat("df_count dim:", paste(dim(cnt), collapse = " x "), "\n")
cat("df_norm dim:", paste(dim(norm_expr), collapse = " x "), "\n")

ct <- anno[, c("Barcode", "Cluster")]
ct_map <- split(ct$Barcode, ct$Cluster)
ct_map <- lapply(ct_map, function(x) intersect(x, colnames(cnt)))
ct_map <- ct_map[lengths(ct_map) > 0]

count_out <- as.matrix(norm_expr)
write.csv(count_out, file.path(inp, "de_count.csv"), row.names = TRUE)
cat("written de_count.csv (normalized) with", nrow(count_out), "genes and", ncol(count_out), "cells\n")

make_cell_expr <- function(gene_list, out_name) {
  map <- stack(gene_list)
  colnames(map) <- c("gene", "celltype")
  map <- unique(map)
  map <- map[map$gene %in% rownames(cnt) & map$celltype %in% names(ct_map), , drop = FALSE]

  cells_all <- colnames(cnt)
  if (nrow(map) == 0) {
    cat("no matched rows for", out_name, "- skipped\n")
    return(invisible(NULL))
  }

  expr_mat <- matrix(0,
                     nrow = nrow(map),
                     ncol = length(cells_all),
                     dimnames = list(paste(map$gene, map$celltype, sep = "__"), cells_all))

  for (cty in names(ct_map)) {
    idx <- which(map$celltype == cty)
    if (length(idx) == 0) next
    genes_cty <- map$gene[idx]
    cells_cty <- ct_map[[cty]]
    expr_mat[idx, cells_cty] <- as.matrix(cnt[genes_cty, cells_cty, drop = FALSE])
  }

  write.csv(cbind(feature = rownames(expr_mat), expr_mat),
            file.path(inp, out_name), row.names = FALSE)
  cat("written", out_name, "with", nrow(expr_mat), "features and", ncol(expr_mat), "cells\n")
}

make_cell_expr_norm <- function(gene_list, out_name) {
  map <- stack(gene_list)
  colnames(map) <- c("gene", "celltype")
  map <- unique(map)
  map <- map[map$gene %in% rownames(norm_expr) & map$celltype %in% names(ct_map), , drop = FALSE]

  cells_all <- colnames(norm_expr)
  if (nrow(map) == 0) {
    cat("no matched rows for", out_name, "- skipped\n")
    return(invisible(NULL))
  }

  expr_mat <- matrix(0,
                     nrow = nrow(map),
                     ncol = length(cells_all),
                     dimnames = list(paste(map$gene, map$celltype, sep = "__"), cells_all))

  for (cty in names(ct_map)) {
    idx <- which(map$celltype == cty)
    if (length(idx) == 0) next
    genes_cty <- map$gene[idx]
    cells_cty <- ct_map[[cty]]
    expr_mat[idx, cells_cty] <- as.matrix(norm_expr[genes_cty, cells_cty, drop = FALSE])
  }

  write.csv(cbind(feature = rownames(expr_mat), expr_mat),
            file.path(inp, out_name), row.names = FALSE)
  cat("written", out_name, "with", nrow(expr_mat), "features and", ncol(expr_mat), "cells\n")
}

make_cell_expr_norm(ligs, "ligand_expr_by_cell.csv")
make_cell_expr_norm(recs, "receptor_expr_by_cell.csv")

if (!is.null(loca)) {
  barcodes <- colnames(cnt)
  coords_out <- data.frame(
    Barcode = barcodes,
    x = loca[barcodes, 1],
    y = loca[barcodes, 2],
    Cluster = anno$Cluster[match(barcodes, anno$Barcode)],
    stringsAsFactors = FALSE
  )
  write.csv(coords_out, file.path(inp, "de_coords_matched.csv"), row.names = FALSE)
  cat("written de_coords_matched.csv with", nrow(coords_out), "cells\n")
}
