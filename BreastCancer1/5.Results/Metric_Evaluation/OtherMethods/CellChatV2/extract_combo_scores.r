library(readr)
library(dplyr)
library(tidyr)

# Resolve paths from this script (works with Rscript --file= and with source())
resolve_this_script <- function() {
  ca <- commandArgs(trailingOnly = FALSE)
  fa <- grep("^--file=", ca, value = TRUE)
  if (length(fa)) {
    return(normalizePath(sub("^--file=", "", fa[1])))
  }
  for (i in rev(seq_len(sys.nframe()))) {
    of <- sys.frame(i)$ofile
    if (is.null(of)) next
    fp <- if (inherits(of, "srcfile")) of$filename else as.character(of)
    if (length(fp) == 1L && nzchar(fp)) return(normalizePath(fp))
  }
  if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
    p <- rstudioapi::getActiveDocumentContext()$path
    if (length(p) == 1L && nzchar(p)) return(normalizePath(p))
  }
  stop(
    "Cannot resolve path to extract_combo_scores.R (source() did not set ofile).\n",
    "Run from CellChatV2 folder:  Rscript extract_combo_scores.R"
  )
}

script_cellchat_dir <- dirname(resolve_this_script())
other_methods_dir <- normalizePath(file.path(script_cellchat_dir, ".."))
# BreastCancer1 根目录：OtherMethods -> Metric_Evaluation -> 5.Results -> BreastCancer1
dataset_root <- normalizePath(file.path(other_methods_dir, "..", "..", ".."))

combo_path <- file.path(dataset_root, "2.LR_Screening",
                        "3.Identify sensitive genes and gene combinations",
                        "3.Subnetwork exploration", "combo_only.csv")
# CellChatV2/result 与 COMMOT/result（与 Metric_Evaluation/OtherMethods 下目录一致）
cellchat_result_dir <- file.path(script_cellchat_dir, "result")
commot_result_dir <- file.path(other_methods_dir, "COMMOT", "result")
cellchat_rds_path <- file.path(cellchat_result_dir, "result.rds")
commot_csv_path <- file.path(commot_result_dir, "result.csv")

# Output paths（写入各自 result 子目录）
cellchat_out <- file.path(cellchat_result_dir, "result_combo_only.csv")
commot_out <- file.path(commot_result_dir, "result_combo_only.csv")

# 1) Load combo definitions (1610 rows)
combo_df <- read_csv(combo_path, col_names = "combo", show_col_types = FALSE) %>%
  separate(combo, into = c("ligand_part", "receptor_part"), sep = "\\|") %>%
  mutate(
    Ligand   = sub("__.*", "", ligand_part),
    Receptor = sub("__.*", "", receptor_part),
    Sender   = sub(".*__", "", ligand_part),
    Receiver = sub(".*__", "", receptor_part)
  ) %>%
  select(Sender, Receiver, Ligand, Receptor)

stopifnot(nrow(combo_df) == 1610)

# Helper to ensure full combo coverage
fill_missing <- function(df, score_col) {
  df %>%
    right_join(combo_df, by = c("Sender", "Receiver", "Ligand", "Receptor")) %>%
    mutate({{ score_col }} := replace_na({{ score_col }}, 0)) %>%
    # Preserve original combo order
    mutate(.combo_id = row_number()) %>%
    arrange(.combo_id) %>%
    select(-.combo_id)
}

# 2) CellChat extraction
if (!file.exists(cellchat_rds_path)) {
  stop("CellChat result.rds not found at ", cellchat_rds_path)
}
cellchat_record <- readRDS(cellchat_rds_path)
cellchat_df <- cellchat_record$result %>%
  select(Sender, Receiver, Ligand, Receptor, LRscore)

cellchat_combo <- fill_missing(cellchat_df, LRscore)
write_csv(cellchat_combo, cellchat_out)
cat("Saved CellChat combo scores to: ", cellchat_out, "\n")
cat("Rows: ", nrow(cellchat_combo), "  Missing combos filled: ", sum(cellchat_combo$LRscore == 0), "\n")

# 3) COMMOT extraction（需先跑完 COMMOT/commot_main.py 生成 result.csv；未生成则跳过）
if (!file.exists(commot_csv_path)) {
  warning(
    "跳过 COMMOT：未找到 ", commot_csv_path, "\n",
    "请先运行 Python：BreastCancer1/.../OtherMethods/COMMOT/commot_main.py，",
    "成功后再运行本脚本以生成 result_combo_only.csv。"
  )
} else {
  commot_df <- read_csv(commot_csv_path, show_col_types = FALSE, col_types = cols()) %>%
    rename(LRscore = score) %>%
    group_by(Sender, Receiver, Ligand, Receptor) %>%
    summarise(LRscore = max(LRscore, na.rm = TRUE), .groups = "drop")

  commot_combo <- fill_missing(commot_df, LRscore)
  write_csv(commot_combo, commot_out)
  cat("Saved COMMOT combo scores to: ", commot_out, "\n")
  cat("Rows: ", nrow(commot_combo), "  Missing combos filled: ", sum(commot_combo$LRscore == 0), "\n")
}
