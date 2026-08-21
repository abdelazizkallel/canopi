#!/usr/bin/env python3
"""
plot_jbs_tree_positions.py
==========================
Map of the JBS (Jarvselja Birch Stand) tree positions, coloured by species and
marked by within-species size class (18 groups total).

Reads the per-group ``JBS_treelocations/{GROUP}_treelocations.dat`` files.
Marker area is proportional to the group mean tree height.

Outputs ``jbs_tree_positions_18groups.{eps,pdf}`` in the article directory.

Usage
-----
    python plot_jbs_tree_positions.py [OUTDIR]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from canopi.scenes.jbs_loader import JBS_GROUPS, GROUP_TO_SPECIES, LX, LY
from canopi.paths import figure_dir, scene_dir

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOC_DIR = Path(scene_dir(required=True))
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else figure_dir()
OUT_STEM = "jbs_tree_positions_18groups"

# Species display order (as in the published figure) and Latin binomials
SPECIES_ORDER = ["BEPE", "ALGL", "TICO", "POTR", "PIAB", "FREX", "ACPL"]
LATIN = {
    "BEPE": "B.~pendula",
    "ALGL": "A.~glutinosa",
    "TICO": "T.~cordata",
    "POTR": "P.~tremuloides",
    "PIAB": "P.~abies",
    "FREX": "F.~excelsior",
    "ACPL": "A.~platanoides",
}
COLOR = {
    "BEPE": "#2166ac",
    "ALGL": "#ff7f00",
    "TICO": "#4daf4a",
    "POTR": "#984ea3",
    "PIAB": "#009688",
    "FREX": "#e41a1c",
    "ACPL": "#795548",
}
# Marker per within-species size class (1..5)
CLASS_MARKER = ["o", "s", "^", "D", "v"]

# Font sizes (enlarged w.r.t. the first version of the figure)
FS_TICK = 15
FS_LABEL = 17
FS_LEG = 14
FS_LEG_TITLE = 15

SIZE_PER_METRE = 1.6  # marker area (pt^2) per metre of mean tree height


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.linewidth": 0.9,
        }
    )

    fig, ax = plt.subplots(figsize=(8.2, 8.0))

    # ---- scatter, one call per group -------------------------------------
    for grp, params in JBS_GROUPS.items():
        sp = GROUP_TO_SPECIES[grp]
        # within-species class index: trailing digit, or 1 for single-class species
        cls = int(grp[-1]) if grp[-1].isdigit() else 1

        path = LOC_DIR / f"{grp}_treelocations.dat"
        xy = np.loadtxt(path)
        xy = np.atleast_2d(xy)

        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=SIZE_PER_METRE * params.mean_height_m,
            c=COLOR[sp],
            marker=CLASS_MARKER[cls - 1],
            linewidths=0,
            zorder=3,
        )

    # ---- scene footprint --------------------------------------------------
    ax.add_patch(
        plt.Rectangle(
            (0, 0), LX, LY, fill=False, edgecolor="black", linewidth=1.4, zorder=4
        )
    )

    ax.set_xlim(-4, LX + 4)
    ax.set_ylim(-4, LY + 4)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$ (m)", fontsize=FS_LABEL)
    ax.set_ylabel(r"$y$ (m)", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, color="0.85", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    # ---- legend 1: species ------------------------------------------------
    sp_handles = []
    for sp in SPECIES_ORDER:
        n_cls = sum(1 for g in JBS_GROUPS if GROUP_TO_SPECIES[g] == sp)
        n_tree = sum(
            JBS_GROUPS[g].n_trees for g in JBS_GROUPS if GROUP_TO_SPECIES[g] == sp
        )
        sp_handles.append(
            Line2D(
                [],
                [],
                color=COLOR[sp],
                marker="s",
                linestyle="none",
                markersize=9,
                label=rf"$\mathit{{{LATIN[sp]}}}$ ({n_cls} cl., $n\,{{=}}\,{n_tree}$)",
            )
        )

    leg1 = ax.legend(
        handles=sp_handles,
        title="Species (classes, trees)",
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.175),
        fontsize=FS_LEG,
        title_fontsize=FS_LEG_TITLE,
        handletextpad=0.5,
        columnspacing=1.4,
        borderpad=0.6,
        labelspacing=0.5,
        framealpha=1.0,
    )
    ax.add_artist(leg1)

    # ---- legend 2: within-species class (second line, below legend 1) -----
    cls_handles = [
        Line2D(
            [],
            [],
            color="0.30",
            marker=CLASS_MARKER[i],
            linestyle="none",
            markersize=9,
            label=f"Class {i + 1}",
        )
        for i in range(5)
    ]
    cls_handles.append(
        Line2D(
            [],
            [],
            color="0.30",
            marker="o",
            linestyle="none",
            markersize=5,
            label=r"size $\propto H$",
        )
    )

    ax.legend(
        handles=cls_handles,
        title="Within-species class",
        ncol=6,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        fontsize=FS_LEG,
        title_fontsize=FS_LEG_TITLE,
        handletextpad=0.4,
        columnspacing=0.9,
        borderpad=0.5,
        framealpha=1.0,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("eps", "pdf"):
        out = OUT_DIR / f"{OUT_STEM}.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
