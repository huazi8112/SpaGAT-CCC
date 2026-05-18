import numpy as np
from scipy.io import loadmat, savemat
from pathlib import Path

script_dir = Path(__file__).resolve().parent

# 与 04_combine_panels_bc1.py、run_demo 等下游一致：阈值 0.26 → new_name-L0.26.mat / new_name-R0.26.mat
T0 = 0.26

# L/R 各自与 final_test.m、adduwavelet 管线对齐的输入
_disturbance_dir = script_dir.parent / "2.Disturbance handling"
_network_dir = script_dir.parent.parent / "2.Build a gene network"

SOURCES = [
    {
        "tag": "L",
        "hp": _disturbance_dir / "adduwavelet-100-(all)-L-merged.mat",
        "pre": _network_dir / "prewavelet-L.mat",
        "out": script_dir / f"new_name-L{T0:.2f}.mat",
    },
    {
        "tag": "R",
        "hp": _disturbance_dir / "adduwavelet-100-(all)-R-merged.mat",
        "pre": _network_dir / "prewavelet-R.mat",
        "out": script_dir / f"new_name-R{T0:.2f}.mat",
    },
]


def _load_hp_pre(hp_path: Path, pre_path: Path):
    hp = loadmat(hp_path)
    pre = loadmat(pre_path)
    H_point_value = np.array(hp["H_point_value"])
    name = np.array(pre["name"]).squeeze()
    maprho = np.array(pre["maprho"])
    return H_point_value, name, maprho


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


def print_threshold_scan(tag, H_point_value, name, maprho):
    thresh = np.round(np.arange(0.20, 0.551, 0.003), 3)
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

    new_name = filter_sensitive_names(H_point_value, name, maprho, T0)
    raw_n = int(np.sum(np.mean(H_point_value, axis=1) < T0))
    print(f"\n示例阈值 {T0:.2f} ({tag}): raw={raw_n}, 过滤后={len(new_name)}")
    return new_name


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
        new_name = print_threshold_scan(cfg["tag"], H_point_value, name, maprho)

        out_path = cfg["out"]
        savemat(out_path, {"new_name": new_name.reshape(-1, 1)})
        print(f"已保存 → {out_path}")


if __name__ == "__main__":
    main()
