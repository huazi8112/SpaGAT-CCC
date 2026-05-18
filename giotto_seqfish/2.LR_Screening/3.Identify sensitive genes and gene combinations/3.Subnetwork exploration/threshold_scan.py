import csv
import numpy as np
from scipy.io import loadmat, savemat
from pathlib import Path

script_dir = Path(__file__).resolve().parent

# L / R 使用不同均值阈值筛选敏感基因（与均值 p 阈值规则一致：< t 纳入）
_disturbance_dir = script_dir.parent / "2.Disturbance handling"
_network_dir = script_dir.parent.parent / "2.Build a gene network"

SOURCES = [
    {
        "tag": "L",
        "hp": _disturbance_dir / "adduwavelet-100-(all)-L-merged-A3.mat",
        "pre": _network_dir / "prewavelet-L-A3.mat",
        "t0": 0.21,
        # 命名风格对齐 new_name-L0.26.mat
        "out": script_dir / "new_name-L0.21.mat",
    },
    {
        "tag": "R",
        "hp": _disturbance_dir / "adduwavelet-100-(all)-R-merged-A3.mat",
        "pre": _network_dir / "prewavelet-R-A3.mat",
        "t0": 0.002,
        "out": script_dir / "new_name-R0.002.mat",
    },
]

def _load_hp_pre(hp_path: Path, pre_path: Path):
    hp = loadmat(hp_path)
    pre = loadmat(pre_path)
    H_point_value = np.array(hp["H_point_value"])
    name = np.array(pre["name"]).squeeze()
    maprho = np.array(pre["maprho"])
    return H_point_value, name, maprho


def _matlab_cell_strings(name_arr) -> list[str]:
    """将 loadmat 读出的 name（cell 或字符数组）展平为 str 列表。"""
    arr = np.asarray(name_arr).ravel()
    out: list[str] = []
    for x in arr:
        s = np.asarray(x).squeeze()
        if s.dtype == object and s.size == 1:
            s = np.asarray(s.item()).squeeze()
        if s.size == 0:
            out.append("")
        elif s.dtype.kind in ("U", "S"):
            out.append(str(s.item()))
        else:
            out.append(str(np.asarray(s).flat[0]))
    return out


def filter_sensitive_names(H_point_value, name, maprho, t):
    """与 final_test.m / 旧版脚本一致：均值阈值 + maprho 非零行过滤."""
    idx = np.mean(H_point_value, axis=1) < t
    new_point = np.where(idx)[0]
    new_name = name[new_point]

    new_maprho = maprho[np.ix_(new_point, new_point)]
    nonzero_rows = new_maprho.any(axis=1)
    new_maprho = new_maprho[np.ix_(nonzero_rows, nonzero_rows)]
    new_point = new_point[nonzero_rows]
    new_name = new_name[nonzero_rows]
    return new_name


def print_threshold_scan(tag, H_point_value, name, maprho, t0: float):
    thresh = np.round(np.arange(0, 0.4, 0.0005), 3)
    counts_raw = []
    counts_nz = []
    for t in thresh:
        idx = np.mean(H_point_value, axis=1) < t
        new_point = np.where(idx)[0]
        counts_raw.append(len(new_point))

        new_maprho = maprho[np.ix_(new_point, new_point)]
        nonzero_rows = new_maprho.any(axis=1)
        new_point_nz = new_point[nonzero_rows]
        counts_nz.append(len(new_point_nz))

    print(f"\n===== 阈值扫描 ({tag}) =====")
    print("thresh\traw\tnonzero_filtered")
    for t, c1, c2 in zip(thresh, counts_raw, counts_nz):
        print(f"{t:.3f}\t{c1}\t{c2}")

    new_name = filter_sensitive_names(H_point_value, name, maprho, t0)
    raw_n = int(np.sum(np.mean(H_point_value, axis=1) < t0))
    t0_repr = f"{t0:.2f}" if t0 >= 0.01 else f"{t0:.3f}"
    print(f"\n筛选阈值 {t0_repr} ({tag}): raw={raw_n}, 过滤后={len(new_name)}")
    return new_name, thresh, counts_raw, counts_nz


def _save_threshold_scan_csv(path: Path, thresh, counts_raw, counts_nz) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["thresh", "raw", "nonzero_filtered"])
        for t, c1, c2 in zip(thresh, counts_raw, counts_nz):
            w.writerow([f"{t:.3f}", c1, c2])


def _save_genes_csv(path: Path, new_name_arr) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    genes = _matlab_cell_strings(new_name_arr)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gene"])
        for g in genes:
            w.writerow([g])


def main():
    for cfg in SOURCES:
        hp_path, pre_path = cfg["hp"], cfg["pre"]
        if not hp_path.is_file():
            print(f"[跳过 {cfg['tag']}] 缺少文件: {hp_path}")
            continue
        if not pre_path.is_file():
            print(f"[跳过 {cfg['tag']}] 缺少文件: {pre_path}")
            continue

        H_point_value, name, maprho = _load_hp_pre(hp_path, pre_path)
        new_name, thresh, counts_raw, counts_nz = print_threshold_scan(
            cfg["tag"], H_point_value, name, maprho, cfg["t0"]
        )

        out_path = cfg["out"]
        savemat(out_path, {"new_name": new_name.reshape(-1, 1)})
        print(f"已保存 → {out_path}")

        scan_csv = out_path.with_name(f"threshold_scan-{cfg['tag']}-A3.csv")
        genes_csv = out_path.with_suffix(".csv")
        _save_threshold_scan_csv(scan_csv, thresh, counts_raw, counts_nz)
        _save_genes_csv(genes_csv, new_name)
        print(f"已保存 → {scan_csv}")
        print(f"已保存 → {genes_csv}")


if __name__ == "__main__":
    main()
