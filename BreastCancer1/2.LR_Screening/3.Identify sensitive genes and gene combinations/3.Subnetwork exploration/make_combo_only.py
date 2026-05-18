"""A3版本：生成 combo_only-A3.csv

来源1：
- new_name-L0.26.mat
- new_name-R0.26.mat
按笛卡尔积组合，并去掉同细胞类型配对。

来源2：
- BreastCancer1/5.Results/Metric_Evaluation/OtherMethods/InputData/runscMLnet/**/scMLnet.rds
从每个 rds 的 LigRec(source,target) 提取基因对，并按目录名 Sender_Receiver 补细胞类型。

输出：
- combo_only.csv
  单列 combo，格式：LigGene__SenderCT|RecGene__ReceiverCT
"""

import os
import subprocess
import sys
import tempfile

import pandas as pd
import scipy.io as sio

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 与脚本同项目：BreastCancer1/…/Metric_Evaluation/OtherMethods/InputData/runscMLnet（内含 **/scMLnet.rds）
RUNSCMLNET_DIR = os.path.normpath(
    os.path.join(
        SCRIPT_DIR,
        "..",
        "..",
        "..",
        "5.Results",
        "Metric_Evaluation",
        "OtherMethods",
        "InputData",
        "runscMLnet",
    )
)

MAT_L = os.path.join(SCRIPT_DIR, "new_name-L0.26.mat")
MAT_R = os.path.join(SCRIPT_DIR, "new_name-R0.26.mat")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "combo_only.csv")


def load_mat_names(path: str) -> list[str]:
    mat = sio.loadmat(path)
    if "new_name" not in mat:
        raise KeyError(f"MAT文件缺少变量 new_name: {path}")
    arr = mat["new_name"].squeeze()
    return [str(x[0]) if hasattr(x, "__len__") else str(x) for x in arr]


def split_gene_ct(val: str) -> tuple[str, str]:
    parts = val.split("__", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (val, "")


def extract_combos_from_runscmlnet(base_dir: str) -> set[str]:
    r_code_lines = [
        'base <- "{}"'.format(base_dir.replace("\\", "/")),
        'files <- list.files(base, pattern="scMLnet\\\\.rds$", recursive=TRUE, full.names=TRUE)',
        'rows <- character(0)',
        'for(f in files){',
        '  ct_pair <- basename(dirname(f))',
        '  parts <- strsplit(ct_pair, "_")[[1]]',
        '  if(length(parts) < 2) next',
        '  sender_ct <- parts[1]',
        '  receiver_ct <- paste(parts[2:length(parts)], collapse="_")',
        '  d <- readRDS(f)',
        '  if(!("LigRec" %in% names(d))) next',
        '  lr <- d[["LigRec"]]',
        '  if(is.null(lr) || nrow(lr) == 0) next',
        '  src_idx <- which(colnames(lr) == "source")',
        '  tgt_idx <- which(colnames(lr) == "target")',
        '  if(length(src_idx) < 1 || length(tgt_idx) < 1) next',
        '  src_idx <- src_idx[1]',
        '  tgt_idx <- tgt_idx[1]',
        '  for(i in seq_len(nrow(lr))){',
        '    lig <- as.character(lr[i, src_idx])',
        '    rec <- as.character(lr[i, tgt_idx])',
        '    if(is.na(lig) || is.na(rec) || lig == "" || rec == "") next',
        '    combo <- paste0(lig, "__", sender_ct, "|", rec, "__", receiver_ct)',
        '    rows <- c(rows, combo)',
        '  }',
        '}',
        'rows <- unique(rows)',
    ]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as tmp:
        tmp_csv = tmp.name

    r_code_lines.append(
        f'write.csv(data.frame(combo=rows), "{tmp_csv.replace(chr(92), "/")}", row.names=FALSE)'
    )
    r_script = "\n".join(r_code_lines)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".R", delete=False, encoding="utf-8"
    ) as rf:
        rf.write(r_script)
        r_script_path = rf.name

    try:
        result = subprocess.run(
            ["Rscript", "--vanilla", r_script_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            print("Rscript 错误输出:", result.stderr)
            sys.exit(1)
        rds_df = pd.read_csv(tmp_csv)
        if rds_df.empty and len(rds_df.columns) == 0:
            return set()

        cols = [str(c).strip() for c in rds_df.columns]
        colmap = {c.lower(): c for c in cols}

        combo_col = None
        if "combo" in colmap:
            combo_col = colmap["combo"]
        else:
            for c in cols:
                if not c.lower().startswith("unnamed"):
                    combo_col = c
                    break

        if combo_col is None:
            raise KeyError(f"runscMLnet 导出结果缺少可用列，当前列: {list(rds_df.columns)}")

        return set(rds_df[combo_col].dropna().astype(str).tolist())
    finally:
        if os.path.exists(tmp_csv):
            os.unlink(tmp_csv)
        if os.path.exists(r_script_path):
            os.unlink(r_script_path)


def main() -> None:
    print("=== A3 来源1: 阈值MAT笛卡尔积 ===")
    l_names = load_mat_names(MAT_L)
    r_names = load_mat_names(MAT_R)
    print(f"  L 节点数: {len(l_names)},  R 节点数: {len(r_names)}")

    mat_combos: set[str] = set()
    for l_name in l_names:
        for r_name in r_names:
            _, lct = split_gene_ct(l_name)
            _, rct = split_gene_ct(r_name)
            if lct != rct:
                mat_combos.add(f"{l_name}|{r_name}")

    print(f"  笛卡尔积（去同类）: {len(mat_combos)} 对")

    print("\n=== A3 来源2: runscMLnet/scMLnet.rds ===")
    rds_combos = extract_combos_from_runscmlnet(RUNSCMLNET_DIR)
    print(f"  runscMLnet 提取唯一 combo: {len(rds_combos)} 对")

    print("\n=== 合并 ===")
    overlap = mat_combos & rds_combos
    only_mat = mat_combos - rds_combos
    only_rds = rds_combos - mat_combos
    all_combos = mat_combos | rds_combos

    print(f"  仅来自 mat:    {len(only_mat)}")
    print(f"  仅来自 RDS:    {len(only_rds)}")
    print(f"  两者重叠:      {len(overlap)}")
    print(f"  合并唯一总计:  {len(all_combos)}")

    # 输出顺序：先RDS选出（仅RDS + 重叠），再仅MAT
    ordered = sorted(only_rds) + sorted(overlap) + sorted(only_mat)
    pd.DataFrame({"combo": ordered}).to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ 已写出: {OUTPUT_CSV}  ({len(ordered)} 行)")


if __name__ == "__main__":
    main()
