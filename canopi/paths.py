"""
Path resolution for figure output and RAMI-5 scene data.

Both resolvers are environment-overridable so that scripts never carry a
machine-specific absolute path:

    CANOPI_FIG_DIR      where figures are written   (default: ./figures)
    CANOPI_SCENE_DIR    RAMI-5 tree-location files  (default: ./JBS_treelocations)
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

__all__ = ["figure_dir", "scene_dir", "REPO_ROOT"]

REPO_ROOT = Path(__file__).resolve().parent.parent


def figure_dir(create: bool = True) -> Path:
    """Directory figures are written to. Honours ``$CANOPI_FIG_DIR``."""
    d = Path(os.environ.get("CANOPI_FIG_DIR", Path.cwd() / "figures"))
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def scene_dir(required: bool = False) -> str | None:
    """
    Locate the RAMI-5 ``*_treelocations.dat`` files.

    Search order: ``$CANOPI_SCENE_DIR``, ``./JBS_treelocations``,
    ``<repo>/data/JBS_treelocations``.

    Returns ``None`` when the data is absent, which makes the loaders fall
    back to *synthetic* tree placement. That fallback runs fine but does
    **not** reproduce the published figures, so a warning is emitted.
    Pass ``required=True`` to raise instead.
    """
    candidates = []
    env = os.environ.get("CANOPI_SCENE_DIR")
    if env:
        candidates.append(Path(env))
    candidates += [Path.cwd() / "JBS_treelocations",
                   REPO_ROOT / "data" / "JBS_treelocations"]

    for c in candidates:
        if (c / "ACPL_treelocations.dat").exists():
            return str(c)

    msg = ("RAMI-5 tree locations not found (looked in: "
           + ", ".join(str(c) for c in candidates)
           + "). Run:  python scripts/download_jbs_data.py data/JBS_treelocations")
    if required:
        raise FileNotFoundError(msg)
    warnings.warn(msg + "\n  -> falling back to SYNTHETIC tree placement; "
                  "results will NOT match the published figures.", stacklevel=2)
    return None
