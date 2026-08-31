"""
Figure S05 – Prior-graph construction threshold sensitivity analysis
Reproduces the composite-score bar chart (Panel A) and six-axis radar charts
(Panels B / C / D) for the BreastCancer1 dataset (L100 vs R100 features).

Upstream features (required on disk for reproducibility of the evaluation pipeline):
  BreastCancer1/2.LR_Screening/1.Data preprocessing/t_wavelet-all-L.mat
  BreastCancer1/2.LR_Screening/1.Data preprocessing/t_wavelet-all-R.mat
  (variable `t_data`; see GCN threshold scripts that load these MAT files)

Panel numbers embed hard-coded metrics aligned with:
  GCNtest 测试阈值指标.py  (τ ∈ {0.4, 0.6, 0.8}, 3 independent runs each)
Weights: Stability 0.20 | Density-Match 0.35 | Connectivity 0.25 | Non-isolated 0.20

Output (same directory as this script): S05_threshold_sensitivity.png
"""

from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
_DATA_PREPROC = (
    REPO_ROOT / "BreastCancer1" / "2.LR_Screening" / "1.Data preprocessing"
)
MAT_WAVELET_L = _DATA_PREPROC / "t_wavelet-all-L.mat"
MAT_WAVELET_R = _DATA_PREPROC / "t_wavelet-all-R.mat"
OUT_FIG = SCRIPT_DIR / "S05_threshold_sensitivity.png"
OUT_EPS = SCRIPT_DIR / "S05_threshold_sensitivity.eps"

for _p in (MAT_WAVELET_L, MAT_WAVELET_R):
    if not _p.is_file():
        raise FileNotFoundError(
            f"Required data file missing: {_p}\n"
            "Expected t_wavelet-all-L.mat and t_wavelet-all-R.mat under "
            "BreastCancer1/2.LR_Screening/1.Data preprocessing/"
        )

# Global font scale (all figure text — bar chart, radar, legends, annotations)
FS = {
    'panel_main': 28,      # A. title
    'section': 26,         # middle section title
    'axis_label': 22,      # bar ylabel / xlabel
    'tick': 21,            # bar x-ticks
    'bar_value': 19,       # numbers on bars
    'note': 18,            # italic note
    'legend': 19,
    'radar_title': 24,     # B/C/D panel titles
    'radar_axis': 17,      # six axis labels on radar
    'radar_ann': 16,       # value labels on radar
    'fig_legend': 19,
}

matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif'],
    'axes.unicode_minus': False,
    'font.size': 19,
})

# ══════════════════════════════════════════════════════════════════════════════
#  RAW EXPERIMENTAL RESULTS
# ══════════════════════════════════════════════════════════════════════════════

THRESHOLDS = [0.4, 0.6, 0.8]

# Composite scores  (Rank in parentheses)
COMP = {
    'L100': [0.8970, 0.9689, 0.2488],
    'R100': [0.8984, 0.9658, 0.3040],
}
RANKS = {
    'L100': [2, 1, 3],
    'R100': [2, 1, 3],
}

# ── Four raw quality metrics per (threshold × feature) ────────────────────
#   Verified: 0.20·J + 0.35·DM + 0.25·LCC + 0.20·NI  ≈  composite score
#
#   τ=0.4  L100 check: 0.20×1.000 + 0.35×0.7059 + 0.25×1.000 + 0.20×1.000
#                    = 0.200 + 0.247 + 0.250 + 0.200 = 0.8971 ✓
#   τ=0.6  L100 check: 0.20×0.970 + 0.35×0.9441 + 0.25×0.990 + 0.20×0.990
#                    = 0.194 + 0.330 + 0.248 + 0.198 = 0.9700 ✓
#   τ=0.8  L100 check: 0.20×0.000 + 0.35×0.6990 + 0.25×0.0095 + 0.20×0.0090
#                    = 0.000 + 0.245 + 0.002 + 0.002 = 0.2488 ✓
#   τ=0.8  R100 check: 0.20×0.3333 + 0.35×0.6641 + 0.25×0.0076 + 0.20×0.0135
#                    = 0.067 + 0.232 + 0.002 + 0.003 = 0.3037 ✓

RAW = {
    # axis 0: Jaccard (stability)
    'jaccard': {
        'L100': [1.0000, 0.9700, 0.0000],
        'R100': [1.0000, 0.9700, 0.3333],
    },
    # axis 1: LCC ratio (connectivity)
    'lcc': {
        'L100': [1.0000, 0.9900, 0.0095],
        'R100': [1.0000, 0.9750, 0.0076],
    },
    # axis 2: Non-isolated ratio
    'niso': {
        'L100': [1.0000, 0.9900, 0.0090],
        'R100': [1.0000, 0.9750, 0.0135],
    },
    # axis 3: Density-match score  = 1 − |pred_density − prior_density|
    'dm': {
        'L100': [0.7059, 0.9441, 0.6990],
        'R100': [0.7097, 0.9604, 0.6641],
    },
}

# ── Axes 4 & 5: normalized density-match variants ─────────────────────────
#   Axis 4 – Global min-max normalisation across all 6 (threshold × feature) points
#   Axis 5 – Z-score + min-max normalisation (yields virtually identical result)

_dm_all = np.array(RAW['dm']['L100'] + RAW['dm']['R100'])  # shape (6,)
_dm_min, _dm_max = _dm_all.min(), _dm_all.max()

def _minmax(v: float) -> float:
    return (v - _dm_min) / (_dm_max - _dm_min)

DM_MM = {
    'L100': [_minmax(v) for v in RAW['dm']['L100']],
    'R100': [_minmax(v) for v in RAW['dm']['R100']],
}

_z = (_dm_all - _dm_all.mean()) / _dm_all.std()
_z_min, _z_max = _z.min(), _z.max()

def _zminmax(z_val: float) -> float:
    return (z_val - _z_min) / (_z_max - _z_min)

DM_ZMM = {
    'L100': [_zminmax(z) for z in _z[:3]],
    'R100': [_zminmax(z) for z in _z[3:]],
}

# ── Pack all 6 axis values for each (threshold-index, feature) ─────────────
def radar_vals(feat: str, t_idx: int) -> list:
    return [
        RAW['jaccard'][feat][t_idx],
        RAW['lcc'][feat][t_idx],
        RAW['niso'][feat][t_idx],
        RAW['dm'][feat][t_idx],
        DM_MM[feat][t_idx],
        DM_ZMM[feat][t_idx],
    ]


# ══════════════════════════════════════════════════════════════════════════════
#  COLOUR SCHEME
# ══════════════════════════════════════════════════════════════════════════════

COL_L = '#1f4e79'   # navy-blue  (L100)
COL_R = '#c55a11'   # burnt-orange (R100)
GRID_COL = '#aaaaaa'

# ══════════════════════════════════════════════════════════════════════════════
#  RADAR-CHART CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

CATS = [
    'Jaccard Mean\nRatio [0,1]',          # top  (12 o'clock)
    'LCC Ratio\nRatio [0,1]',             # upper-right
    'Non-Isolated\nRatio\nRatio [0,1]',   # lower-right
    'Normalized\nDensity Gap\n0-1 Score', # bottom
    'Normalized Density Gap\n[Min-Max z-score,\n0-1 Score]',  # lower-left
    'Normalized\nZ-Score Density\n[Min-Max,\n0-1 Score]',     # upper-left
]
N_AXES = len(CATS)

# Clockwise from the top (12 o'clock = Jaccard)
_angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, N_AXES, endpoint=False)
ANGLES = _angles
ANGLES_CLOSED = np.append(_angles, _angles[0])


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: draw one radar panel
# ══════════════════════════════════════════════════════════════════════════════

def draw_radar(ax, t_idx: int, panel_label: str):
    """Plot one spider chart for THRESHOLDS[t_idx]."""
    vL = radar_vals('L100', t_idx)
    vR = radar_vals('R100', t_idx)
    vL_c = vL + [vL[0]]
    vR_c = vR + [vR[0]]

    # ── spider grid ──────────────────────────────────────────────────────────
    for r in np.arange(0.2, 1.01, 0.2):
        ax.plot(ANGLES_CLOSED, [r] * (N_AXES + 1),
                color=GRID_COL, linewidth=0.45, alpha=0.35, zorder=1)

    for ang in ANGLES:
        ax.plot([ang, ang], [0, 1.0],
                color=GRID_COL, linewidth=0.5, alpha=0.4, zorder=1)

    # ── fill + line: L100 ────────────────────────────────────────────────────
    ax.fill(ANGLES_CLOSED, vL_c, color=COL_L, alpha=0.10, zorder=3)
    ax.plot(ANGLES_CLOSED, vL_c, 'o-',
            color=COL_L, linewidth=2.8, markersize=8, zorder=5, label='L100')

    # ── fill + line: R100 ────────────────────────────────────────────────────
    ax.fill(ANGLES_CLOSED, vR_c, color=COL_R, alpha=0.10, zorder=3)
    ax.plot(ANGLES_CLOSED, vR_c, 's--',
            color=COL_R, linewidth=2.8, markersize=8, zorder=5, label='R100')

    # ── value annotations: EVERY axis always gets a visible label ────────────
    #   Strategy: place the value just OUTSIDE the data-point marker, with a
    #   white bounding-box so it is readable over the polygon fill.
    #   • L100 and R100 share one label (formatted as "L / R") when they differ
    #     by ≤ 0.03; otherwise each is placed at a slight angular offset.
    #   • Even val = 1.0 is shown ("1.00") so no axis appears blank.
    ann_fs = FS['radar_ann']
    bbox_kw = dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.82)

    for k, ang in enumerate(ANGLES):
        vl, vr = vL[k], vR[k]
        similar = abs(vl - vr) <= 0.03

        if similar:
            # Single combined label centred on the axis
            val = (vl + vr) / 2
            label = f'{val:.4f}' if val >= 0.0001 else '0'
            r_lbl = max(0.18, val + 0.16) if val < 0.80 else val - 0.18
            r_lbl = min(r_lbl, 0.92)
            ax.annotate(label, xy=(ang, r_lbl),
                        ha='center', va='center',
                        fontsize=ann_fs, color=COL_L,
                        zorder=8, clip_on=False, bbox=bbox_kw)
        else:
            # Two labels: L100 slightly left, R100 slightly right on the axis spoke
            ANG_OFF = 0.22   # radians offset for separation
            for val, col, aoff in [(vl, COL_L, -ANG_OFF), (vr, COL_R, +ANG_OFF)]:
                label = f'{val:.4f}' if val >= 0.0001 else '0'
                r_lbl = max(0.18, val + 0.16) if val < 0.80 else val - 0.18
                r_lbl = min(r_lbl, 0.92)
                ax.annotate(label, xy=(ang + aoff, r_lbl),
                            ha='center', va='center',
                            fontsize=ann_fs, color=col,
                            zorder=8, clip_on=False, bbox=bbox_kw)

    # ── polar axis cosmetics ─────────────────────────────────────────────────
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([])
    ax.set_xticks(ANGLES)
    ax.set_xticklabels(CATS, fontsize=FS['radar_axis'])
    ax.tick_params(axis='x', pad=30)
    ax.spines['polar'].set_linewidth(0.6)
    ax.spines['polar'].set_color(GRID_COL)
    ax.grid(False)

    thr_val = THRESHOLDS[t_idx]
    ax.set_title(f'{panel_label}. Threshold {thr_val}',
                 fontsize=FS['radar_title'], fontweight='bold', pad=38, y=1.22)


# ══════════════════════════════════════════════════════════════════════════════
#  BUILD FIGURE
# ══════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(26, 18), facecolor='white')
gs = GridSpec(2, 3, figure=fig,
              height_ratios=[1.0, 2.2],
              hspace=0.80, wspace=0.50)

ax_bar = fig.add_subplot(gs[0, :])           # Panel A – full width
ax_B   = fig.add_subplot(gs[1, 0], polar=True)  # τ = 0.6
ax_C   = fig.add_subplot(gs[1, 1], polar=True)  # τ = 0.4
ax_D   = fig.add_subplot(gs[1, 2], polar=True)  # τ = 0.8

# ── Panel A: grouped bar chart ───────────────────────────────────────────────
x = np.arange(len(THRESHOLDS))
W = 0.32

bars_L = ax_bar.bar(x - W / 2, COMP['L100'], W,
                    color=COL_L, label='L100',
                    zorder=3, edgecolor='white', linewidth=0.5)
bars_R = ax_bar.bar(x + W / 2, COMP['R100'], W,
                    color=COL_R, label='R100',
                    zorder=3, edgecolor='white', linewidth=0.5)

for i, (bl, br) in enumerate(zip(bars_L, bars_R)):
    for bar, vals, ranks_ in [(bl, COMP['L100'], RANKS['L100']),
                               (br, COMP['R100'], RANKS['R100'])]:
        h = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                    f'{vals[i]:.4f}\n(Rank {ranks_[i]})',
                    ha='center', va='bottom',
                    fontsize=FS['bar_value'], color='black', linespacing=1.4)

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(['0.4', '0.6', '0.8'], fontsize=FS['tick'])
ax_bar.set_xlabel('Threshold\n[z-score normalized 0-1 Range Score]',
                  fontsize=FS['axis_label'], labelpad=12)

# Section title sits in the hspace gap – rendered as figure text at a safe y
# We draw it after the subplots so we know the axes bounding box.
_SECTION_TITLE = 'Standardized Quality Metrics Across Thresholds'
ax_bar.set_ylabel('Standardized Composite Score', fontsize=FS['axis_label'], labelpad=14)
ax_bar.set_ylim(0, 1.32)
ax_bar.set_xlim(-0.65, 2.65)
ax_bar.set_title('A.  Composite Score Performance Across Thresholds',
                 fontsize=FS['panel_main'], fontweight='bold', loc='left', pad=18)
ax_bar.tick_params(axis='both', labelsize=FS['tick'])
ax_bar.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.45, zorder=0)
ax_bar.spines[['top', 'right']].set_visible(False)

# Italicised note – placed in top-LEFT area so it never overlaps the legend
ax_bar.annotate(
    'Standardized values DS and DG\n[z-score normalized 0-1 Range Score]',
    xy=(2.62, 1.28), fontsize=FS['note'], ha='right', va='top',
    color='#555555', fontstyle='italic',
)
# Legend inside bar chart – upper-right, clear of the note
ax_bar.legend(
    handles=[mpatches.Patch(color=COL_L, label='L100'),
             mpatches.Patch(color=COL_R, label='R100')],
    loc='upper right', bbox_to_anchor=(0.72, 0.97),
    fontsize=FS['legend'], framealpha=0.92, edgecolor='#cccccc',
)

# ── Radar panels ─────────────────────────────────────────────────────────────
draw_radar(ax_B, t_idx=1, panel_label='B')   # τ = 0.6  (best)
draw_radar(ax_C, t_idx=0, panel_label='C')   # τ = 0.4
draw_radar(ax_D, t_idx=2, panel_label='D')   # τ = 0.8  (worst)

# Margins: large fonts need extra left space for rotated y-label, extra top for subplot title
fig.subplots_adjust(left=0.14, right=0.98, top=0.93, bottom=0.06)

# Place section title in the hspace gap between the two rows
fig.canvas.draw()          # forces layout engine so .get_position() is final
bar_y0  = ax_bar.get_position().y0   # bottom of bar chart row
radar_y1 = ax_C.get_position().y1   # top of radar subplot area (excl. title)
mid_y = (bar_y0 + radar_y1) / 2
fig.text(0.5, mid_y, _SECTION_TITLE,
         ha='center', va='center',
         fontsize=FS['section'], fontweight='bold')

# ── Shared radar legend (bottom-right of figure) ─────────────────────────────
leg_lines = [
    plt.Line2D([0], [0], color=COL_L, marker='o',
               linewidth=2.6, markersize=10, label='L100'),
    plt.Line2D([0], [0], color=COL_R, marker='s', linestyle='--',
               linewidth=2.6, markersize=10, label='R100'),
]
fig.legend(handles=leg_lines,
           loc='lower right', bbox_to_anchor=(0.995, 0.015),
           fontsize=FS['fig_legend'], framealpha=0.92, edgecolor='#cccccc')

# ── Save & show ───────────────────────────────────────────────────────────────
plt.savefig(
    OUT_FIG, dpi=220, bbox_inches='tight',
    pad_inches=0.55, facecolor='white',
)
print(f'Figure saved → {OUT_FIG}')
plt.savefig(
    OUT_EPS, format='eps', bbox_inches='tight',
    pad_inches=0.55, facecolor='white',
)
print(f'Figure saved → {OUT_EPS}')
plt.show()
