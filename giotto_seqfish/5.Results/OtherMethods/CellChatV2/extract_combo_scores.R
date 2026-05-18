library(readr)
library(dplyr)
library(tidyr)

# Resolve paths
dataset_root <- "D:/GitHub Code/SpaGAT-CCC/giotto_seqfish"
methods_root <- file.path(dataset_root, "5.Results/OtherMethods")
cellchat_result_dir <- file.path(methods_root, "CellChatV2", "result")

combo_path <- file.path(dataset_root, "2.LR_Screening",
                        "3.Identify sensitive genes and gene combinations",
                        "3.Subnetwork exploration", "combo_only-A3.csv")
cellchat_rds_path <- file.path(cellchat_result_dir, "result.rds")
commot_csv_path <- file.path(methods_root, "COMMOT", "result", "result.csv")

# Output paths (CellChat under CellChatV2/result)
cellchat_out <- file.path(cellchat_result_dir, "result_combo_only.csv")
commot_out <- file.path(methods_root, "COMMOT", "result", "result_combo_only.csv")

# 1) Load combo definitions (row count follows combo_only-A3.csv)
combo_df <- read_csv(combo_path, show_col_types = FALSE, col_types = cols()) %>%
  separate(combo, into = c("ligand_part", "receptor_part"), sep = "\\|") %>%
  mutate(
    Ligand   = sub("__.*", "", ligand_part),
    Receptor = sub("__.*", "", receptor_part),
    Sender   = sub(".*__", "", ligand_part),
    Receiver = sub(".*__", "", receptor_part)
  ) %>%
  select(Sender, Receiver, Ligand, Receptor)

if (nrow(combo_df) == 0) {
  stop("combo_df is empty after parsing: ", combo_path)
}
cat("Loaded ", nrow(combo_df), " L-R combos from combo_only\n")

# Helper to ensure full combo coverage (case-insensitive gene matching)
fill_missing <- function(df, score_col) {
  # Add uppercase keys for joining
  df_keyed <- df %>%
    mutate(.Ligand_up = toupper(Ligand), .Receptor_up = toupper(Receptor))

  combo_keyed <- combo_df %>%
    mutate(.Ligand_up = toupper(Ligand), .Receptor_up = toupper(Receptor))

  joined <- combo_keyed %>%
    left_join(df_keyed, by = c("Sender", "Receiver", ".Ligand_up", ".Receptor_up")) %>%
    mutate({{ score_col }} := replace_na({{ score_col }}, 0)) %>%
    # Use combo_df's Ligand/Receptor names as canonical output
    select(Sender, Receiver, Ligand = Ligand.x, Receptor = Receptor.x, {{ score_col }})
  joined
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

# 3) COMMOT extraction
if (!file.exists(commot_csv_path)) {
  stop("COMMOT result.csv not found at ", commot_csv_path)
}
commot_df <- read_csv(commot_csv_path, show_col_types = FALSE, col_types = cols()) %>%
  rename(LRscore = score) %>%
  group_by(Sender, Receiver, Ligand, Receptor) %>%
  summarise(LRscore = max(LRscore, na.rm = TRUE), .groups = "drop")

commot_combo <- fill_missing(commot_df, LRscore)
write_csv(commot_combo, commot_out)
cat("Saved COMMOT combo scores to: ", commot_out, "\n")
cat("Rows: ", nrow(commot_combo), "  Missing combos filled: ", sum(commot_combo$LRscore == 0), "\n")
