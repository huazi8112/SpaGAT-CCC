"""
Shared plotting style constants for all paper figures.

Usage
-----
    # At the top of each plotting script:
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from _common import CT_COLORS, CT_COLORS_SEQFISH, FS, apply_paper_style

    apply_paper_style()

Design rationale
----------------
All paper figures are generated at ~22–42 inch figsize and typically reproduced at
~7 inches (2-column) in the final PDF.  That means the effective scale factor is
roughly 0.17–0.32, so we pick *large* original font sizes so the final print size
stays readable (≥ 6 pt final).
"""

import matplotlib
import matplotlib.pyplot as plt


# ───────────────────────────────────────────────────────────────────────────
# 1. Cell-type color palette (authoritative, single source of truth)
# ───────────────────────────────────────────────────────────────────────────
CT_COLORS = {
    "Bcell":       "#4E79A7",   # blue
    "Tcell":       "#F28E2B",   # orange
    "Macrophage":  "#E15759",   # red
    "Stroma":      "#76B7B2",   # teal
    "Endothelial": "#59A14F",   # green
    "Malignant":   "#9C755F",   # brown
}

# seqFISH (mouse cortex) cell types — reuse BC palette and extend
CT_COLORS_SEQFISH = {
    "astrocytes":     "#4E79A7",
    "microglia":      "#E15759",
    "mural":          "#9C755F",
    "OPC":            "#59A14F",
    "endothelial":    "#F28E2B",
    "Olig":           "#76B7B2",
    "L2-L3-eNeuron":  "#3C5488",
    "L4-eNeuron":     "#8491B4",
    "L5-eNeuron":     "#91D1C2",
    "L6-eNeuron":     "#DC0000",
    "Lhx6-iNeuron":   "#F39B7F",
    "Adarb2-iNeuron": "#B09C85",
}

# 弦图外圈标签用缩写，避免弧长不足时长名叠压（图 8 等；配色键仍为全名）
CT_CHORD_ABBREV_SEQFISH = {
    "Adarb2-iNeuron": "Ad2-iN",
    "L2-L3-eNeuron": "L2/3-eN",
    "L4-eNeuron": "L4-eN",
    "L5-eNeuron": "L5-eN",
    "L6-eNeuron": "L6-eN",
    "Lhx6-iNeuron": "Lhx6-iN",
    "OPC": "OPC",
    "Olig": "Olig",
    "astrocytes": "Astro",
    "endothelial": "Endo",
    "microglia": "Micro",
    "mural": "Mural",
}


# ───────────────────────────────────────────────────────────────────────────
# 2. Font-size constants (in pt, for LARGE-figsize paper figures)
# ───────────────────────────────────────────────────────────────────────────
class FS:
    """Paper-ready font sizes, tuned for figsize ~20-24 in."""

    PANEL_LABEL   = 42   # "A", "B", "C" tags on each panel
    TITLE         = 26
    SUBTITLE      = 22
    AXIS_LABEL    = 22
    TICK          = 18
    TICK_SMALL    = 14
    LEGEND        = 18
    LEGEND_TITLE  = 20
    DATA_LABEL    = 16   # numbers above bars, inside heatmap cells
    ANNOT         = 16
    ANNOT_SMALL   = 13
    NODE_LABEL    = 15   # labels on centrality-top network nodes
    NODE_LABEL_SM = 11   # labels on non-hub network nodes


class FS_FIG678:
    """Larger fonts for combined multi-panel figures 6–8 (BC1/BC2/seqFISH)."""

    PANEL_LABEL   = 60
    TITLE         = 38
    SUBTITLE      = 32
    AXIS_LABEL    = 32
    TICK          = 26
    TICK_SMALL    = 20
    LEGEND        = 26
    LEGEND_TITLE  = 30
    DATA_LABEL    = 24
    ANNOT         = 24
    ANNOT_SMALL   = 19
    NODE_LABEL    = 23
    NODE_LABEL_SM = 17


class FS_FIG8:
    """seqFISH 图 8 专用。

    图 8 整图 `figsize` 宽约 48 in，图 6/7 约 36 in。同屏宽查看 PNG 时图 8 会被
    缩小更多，**相同 pt 的字会显得更细更小**。因此在 FS_FIG678 上乘以约 1.30
    （≈48/36），使视觉体量与图 6/7 接近；图 6/7 仍用 FS_FIG678。
    """

    PANEL_LABEL   = 78
    TITLE         = 50
    SUBTITLE      = 42
    AXIS_LABEL    = 42
    TICK          = 34
    TICK_SMALL    = 26
    LEGEND        = 34
    LEGEND_TITLE  = 39
    DATA_LABEL    = 31
    ANNOT         = 31
    ANNOT_SMALL   = 25
    NODE_LABEL    = 30
    NODE_LABEL_SM = 22


# ───────────────────────────────────────────────────────────────────────────
# 3. Consistent rcParams
# ───────────────────────────────────────────────────────────────────────────
def apply_paper_style(fs=None):
    """Apply the shared matplotlib rcParams for all paper figures.

    Parameters
    ----------
    fs : type, optional
        Font-size namespace (defaults to :class:`FS`). Use :class:`FS_FIG678`
        for figures 6–7；seqFISH 图 8 用 :class:`FS_FIG8`（更宽画布需更大 pt）。
    """
    f = fs if fs is not None else FS
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": f.TICK,
        "font.weight": "bold",

        "axes.titlesize": f.TITLE,
        "axes.labelsize": f.AXIS_LABEL,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "axes.linewidth": 1.8,
        "axes.edgecolor": "black",
        "axes.unicode_minus": False,

        "xtick.labelsize": f.TICK,
        "ytick.labelsize": f.TICK,
        "xtick.major.width": 1.4,
        "ytick.major.width": 1.4,

        "legend.fontsize": f.LEGEND,
        "legend.title_fontsize": f.LEGEND_TITLE,

        "pdf.fonttype": 42,
        "ps.fonttype": 3,
        "svg.fonttype": "none",
    })


# ───────────────────────────────────────────────────────────────────────────
# 4. Small helpers shared across scripts
# ───────────────────────────────────────────────────────────────────────────
def panel_label(fig_or_ax, letter, x=0.0, y=1.02, fontsize=None):
    """Draw a consistent 'A'/'B'/... panel-letter label."""
    if fontsize is None:
        fontsize = FS.PANEL_LABEL
    if hasattr(fig_or_ax, "text") and hasattr(fig_or_ax, "transAxes"):
        fig_or_ax.text(x, y, letter, transform=fig_or_ax.transAxes,
                       fontsize=fontsize, fontweight="bold",
                       fontfamily="Arial", va="bottom", ha="left")
    else:  # Figure object
        fig_or_ax.text(x, y, letter, fontsize=fontsize, fontweight="bold",
                       fontfamily="Arial", va="top", ha="left")


def hide_axes(ax):
    """Remove ticks and spines (for schematic / network panels)."""
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def style_spines(ax, keep=("bottom", "left")):
    """Keep only the requested spines visible."""
    for name, sp in ax.spines.items():
        sp.set_visible(name in keep)
