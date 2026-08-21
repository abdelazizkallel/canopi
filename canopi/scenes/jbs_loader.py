"""
canopi.scenes.jbs_loader.py
======================
RAMI-5 HET09_JBS_SUM — full heterogeneous scene loader.

Replaces the homogenised loader (rami_jbs_loader.py) with an explicit
tree-by-tree placement that mirrors the RAMI-5 scene description:

  • Wood structure     — triangle mesh (OBJ file per species per object)
  • Foliage           — disk leaves (r = 0.05 m, fixed)
                        placed from explicit leaf-position files  -OR-
                        sampled from per-species crown + LAD profile
  • Tree instances    — read from an ASCII positions file with
                        translation (tx, ty, tz) and rotation angle (φ)

RAMI-5 file conventions
-----------------------
The RAMI-5 download for HET09_JBS_SUM produces a directory tree like:

    JBS_SUM/
    ├── scene.def           ← master list of object instances
    ├── BEPE/
    │   ├── BEPE_001.obj    ← wood mesh for tree instance 001 (local coords)
    │   ├── BEPE_001.lp     ← leaf positions for instance 001  (local coords)
    │   ├── BEPE_002.obj
    │   ├── BEPE_002.lp
    │   └── ...
    ├── TICO/ ...
    └── ...

scene.def  (one line per tree):
    BEPE  BEPE/BEPE_001  tx  ty  tz  rot_deg
    BEPE  BEPE/BEPE_002  tx  ty  tz  rot_deg
    ...

.lp file (one leaf per line, local tree coords):
    x  y  z  nx  ny  nz

.obj file (Wavefront OBJ, triangles, local tree coords):
    v  x  y  z
    f  i  j  k      ← triangle face (1-based vertex indices)

When these files are NOT present the loader falls back to a fully
parametric synthetic scene (crown ellipsoid + vertical LAD profile),
which reproduces the same stand-level statistics.

Public API
----------
    scene, wood = load_jbs_scene(
        scene_dir   = "JBS_SUM/",      # path to downloaded RAMI dir
        band        = "PUR",
        month       = "Jul",
        P_LAI       = 0.5,             # thinning for derivative
        seed        = 42,
    )
    # scene  → canopi.scene.Scene  (leaf disks, voxel grid)
    # wood   → WoodMesh              (all triangle vertices, face material)

    # Drop-in with GPU renderer:
    from canopi.gpu import render_gpu, GPU_AVAILABLE
    result = render_gpu_het(scene, wood, cfg_obs)

Usage (synthetic fallback, no RAMI files needed):
    scene, wood = load_jbs_scene(scene_dir=None, band="O08", month="Jul")
"""

from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Physical / geometric constants for JBS stand
# ---------------------------------------------------------------------------

#: Plot dimensions [m] — from RAMI-5 scene description
LX = 105.5115
LY = 106.1535
H  = 30.513          #: Canopy Zmax [m]
LEAF_RADIUS = 0.05   #: Fallback disk radius when species not in SPECIES_LEAF_AREA_M2
LAI_EFF     = 3.442  #: Total one-sided LAI (RAMI-5 doc); metadata sums to 2.914
VOXEL_SIZE  = 1.0    #: 1 m voxels → 105 × 106 × 30 grid

# One-sided leaf area per species [m²] from RAMI-5 HET09_JBS_SUM inventory.
# Used to set per-leaf disk radii r = sqrt(area / π).
SPECIES_LEAF_AREA_M2: Dict[str, float] = {
    "BEPE": 20.7e-4,   # Betula pendula
    "TICO": 52.5e-4,   # Tilia cordata
    "ALGL": 23.4e-4,   # Alnus glutinosa
    "POTR": 18.8e-4,   # Populus tremuloides
    "FREX": 176.6e-4,  # Fraxinus excelsior
    "ACPL": 139.6e-4,  # Acer platanoides
    "PIAB": 83.2e-4,   # Picea abies
}


# ---------------------------------------------------------------------------
# Per-species crown and LAD parameters (synthetic fallback)
# ---------------------------------------------------------------------------
# Crown shape: ellipsoid with semi-axes (Rx, Ry, Rz) and centre offset (dz)
# above ground.  Vertical LAD profile ~ Beta(a, b) rescaled to [0, H_tree].
# Foliage area per tree: from the area-weight table in the original loader.

@dataclass
class SpeciesParams:
    name: str
    n_trees: int
    mean_foliage_area_m2: float   #: one-sided leaf area per tree [m²]
    mean_height_m: float          #: mean tree height [m]
    crown_rx_m: float             #: crown x/y semi-axis [m]
    crown_rz_frac: float          #: crown z semi-axis as fraction of height
    crown_base_frac: float        #: crown base height as fraction of height
    lad_alpha: float              #: Beta shape a (LAD profile in crown)
    lad_beta:  float              #: Beta shape b
    # Optical properties per species per band are in the spectral table below


# RAMI-5 species table — kept for file-based path (scene.def uses 7 codes)
# and optical property lookup.  Mean foliage areas updated to per-species
# weighted averages from the 18-group inventory (see JBS_GROUPS below).
JBS_SPECIES: Dict[str, SpeciesParams] = {
    "BEPE": SpeciesParams("BEPE", 465, 19.0, 23.0, 3.0, 0.55, 0.55, 2.0, 3.0),
    "TICO": SpeciesParams("TICO", 205, 24.0, 14.0, 2.0, 0.50, 0.50, 2.0, 3.0),
    "ALGL": SpeciesParams("ALGL", 196, 39.9, 18.0, 2.8, 0.52, 0.48, 1.8, 2.5),
    "POTR": SpeciesParams("POTR",  78, 67.7, 24.0, 2.8, 0.58, 0.45, 1.5, 2.0),
    "PIAB": SpeciesParams("PIAB",  39, 92.0, 14.0, 2.0, 0.70, 0.35, 3.0, 2.0),
    "FREX": SpeciesParams("FREX",  30,121.0, 17.0, 2.8, 0.50, 0.50, 2.2, 3.0),
    "ACPL": SpeciesParams("ACPL",  16,  9.9, 14.0, 2.0, 0.48, 0.52, 2.0, 3.0),
}

# ---------------------------------------------------------------------------
# 18-group table from RAMI-5 HET09_JBS_SUM inventory
# ---------------------------------------------------------------------------
# 7 species split into size classes: BEPE×4, TICO×5, ALGL×4, POTR×2,
# PIAB×1, FREX×1, ACPL×1  (total 1029 trees).
# Foliage areas from RAMI-5 metadata (sum → LAI=2.914; RAMI doc: 3.442;
# the .lp files are scaled to 3.442 — synthetic path will give 2.914).
# Heights: from RAMI-5 inventory ranges (BEPE 19.9–30.5 m, TICO 5.9–20.7 m,
# POTR max ~28 m).  Crown radii: BEPE max 3.56 m, TICO max 2.65 m,
# POTR max 3.15 m (RAMI-5 documentation).
JBS_GROUPS: Dict[str, SpeciesParams] = {
    # name        n_trees  FA/tree(m2)    H(m)   Rx(m)  rz_f  base_f   α    β
    # H, Rx, base_f from RAMI-5 HET09_JBS_SUM inventory
    # base_f = height_to_live_crown / tree_height
    # rz_f kept at species level (no per-group vertical shape data)
    "ACPL":  SpeciesParams("ACPL",  16,   9.9413, 10.90, 1.60, 0.48, 0.1670, 2.0, 3.0),
    "BEPE1": SpeciesParams("BEPE1",115,  13.6225, 15.37, 2.23, 0.55, 0.4925, 2.0, 3.0),
    "BEPE2": SpeciesParams("BEPE2",246,  10.2282, 19.86, 1.95, 0.55, 0.6511, 2.0, 3.0),
    "BEPE3": SpeciesParams("BEPE3", 92,  32.2435, 25.49, 2.19, 0.55, 0.6446, 2.0, 3.0),
    "BEPE4": SpeciesParams("BEPE4", 12,  75.3714, 27.99, 3.11, 0.55, 0.5441, 2.0, 3.0),
    "ALGL1": SpeciesParams("ALGL1", 50,  93.9876, 30.51, 3.56, 0.52, 0.6018, 1.8, 2.5),
    "ALGL2": SpeciesParams("ALGL2", 97,  12.9206, 19.35, 1.92, 0.52, 0.5240, 1.8, 2.5),
    "ALGL3": SpeciesParams("ALGL3", 38,  29.3120, 22.58, 2.31, 0.52, 0.5580, 1.8, 2.5),
    "ALGL4": SpeciesParams("ALGL4", 11,  69.6790, 25.76, 2.78, 0.52, 0.5613, 1.8, 2.5),
    "FREX":  SpeciesParams("FREX",  30, 120.9830, 27.09, 3.60, 0.50, 0.6257, 2.2, 3.0),
    "TICO1": SpeciesParams("TICO1", 17,  20.2880, 13.72, 2.25, 0.50, 0.4810, 2.0, 3.0),
    "TICO2": SpeciesParams("TICO2", 43,   8.8554,  5.91, 2.55, 0.50, 0.3130, 2.0, 3.0),
    "TICO3": SpeciesParams("TICO3", 63,  18.0730, 11.27, 1.93, 0.50, 0.5262, 2.0, 3.0),
    "TICO4": SpeciesParams("TICO4", 62,  27.5280, 14.41, 2.38, 0.50, 0.5337, 2.0, 3.0),
    "TICO5": SpeciesParams("TICO5", 20,  58.3190, 18.34, 2.47, 0.50, 0.4264, 2.0, 3.0),
    "PIAB":  SpeciesParams("PIAB",  39,  92.0224, 20.70, 2.65, 0.70, 0.3652, 3.0, 2.0),
    "POTR1": SpeciesParams("POTR1", 61,  37.7483, 25.27, 2.65, 0.58, 0.7269, 1.5, 2.0),
    "POTR2": SpeciesParams("POTR2", 17, 143.3690, 30.49, 3.15, 0.58, 0.5418, 1.5, 2.0),
}

# Maps each group name → parent RAMI species code (for optical properties)
GROUP_TO_SPECIES: Dict[str, str] = {
    "ACPL":  "ACPL",
    "BEPE1": "BEPE", "BEPE2": "BEPE", "BEPE3": "BEPE", "BEPE4": "BEPE",
    "ALGL1": "ALGL", "ALGL2": "ALGL", "ALGL3": "ALGL", "ALGL4": "ALGL",
    "FREX":  "FREX",
    "TICO1": "TICO", "TICO2": "TICO", "TICO3": "TICO", "TICO4": "TICO", "TICO5": "TICO",
    "PIAB":  "PIAB",
    "POTR1": "POTR", "POTR2": "POTR",
}

# Area weights per species (aggregated from 18 groups, used for band-weighted rho/tau)
_TOTAL_FA = sum(g.n_trees * g.mean_foliage_area_m2 for g in JBS_GROUPS.values())
_SPECIES_FA: Dict[str, float] = {sp: 0.0 for sp in JBS_SPECIES}
for _grp, _g in JBS_GROUPS.items():
    _SPECIES_FA[GROUP_TO_SPECIES[_grp]] += _g.n_trees * _g.mean_foliage_area_m2
SPECIES_WEIGHT = {sp: fa / _TOTAL_FA for sp, fa in _SPECIES_FA.items()}


# ---------------------------------------------------------------------------
# Spectral tables (from rami_jbs_loader.py — kept for completeness)
# ---------------------------------------------------------------------------

BANDS = ["O03", "O04", "O06", "O08", "O10", "O11", "O12",
         "M08", "O17", "MD5", "M11", "MD7", "M12","PUR"]
_BAND_IDX = {b: i for i, b in enumerate(BANDS)}

_REFL = {
    "BEPE": [0.067742,0.075458,0.153929,0.076337,0.082215,0.246577,0.465246,
             0.462741,0.462010,0.436240,0.343284,0.201318,0.217508,0.5],
    "TICO": [0.040220,0.040526,0.066341,0.040675,0.040686,0.142183,0.430280,
             0.445613,0.446692,0.428975,0.329044,0.171912,0.192439,0.5],
    "ALGL": [0.059213,0.063115,0.107666,0.063289,0.062262,0.182928,0.484569,
             0.503541,0.505644,0.485144,0.364154,0.183467,0.200996,0.5],
    "POTR": [0.043842,0.047760,0.089848,0.045882,0.044595,0.159085,0.438141,
             0.450154,0.450595,0.425297,0.319139,0.155997,0.174128,0.5],
    "PIAB": [0.051368,0.054265,0.116617,0.051810,0.056129,0.211462,0.479058,
             0.486675,0.489264,0.423740,0.218632,0.080345,0.092002,0.5],
    "FREX": [0.048851,0.052006,0.105193,0.052954,0.051725,0.182420,0.498911,
             0.520089,0.521675,0.509786,0.420409,0.253921,0.269462,0.5],
    "ACPL": [0.040937,0.041766,0.068426,0.041567,0.041398,0.130351,0.387067,
             0.403283,0.404864,0.392522,0.315679,0.178553,0.193879,0.5],
}
_TRAN = {
    "BEPE": [0.011442,0.019563,0.113968,0.026235,0.025053,0.225646,0.463566,
             0.476427,0.480278,0.483955,0.404630,0.253746,0.280706,0.5],
    "TICO": [0.004512,0.009109,0.076875,0.013200,0.010814,0.180153,0.437522,
             0.456572,0.461335,0.464821,0.394604,0.263924,0.292740,0.5],
    "ALGL": [0.003947,0.007969,0.051025,0.012330,0.010671,0.123064,0.361185,
             0.383281,0.388923,0.398364,0.324149,0.193113,0.215621,0.5],
    "POTR": [0.012178,0.021111,0.092588,0.028822,0.025309,0.181357,0.422574,
             0.441467,0.446961,0.460377,0.402581,0.272775,0.298380,0.5],
    "PIAB": [0.010413,0.014699,0.051778,0.019004,0.018988,0.121429,0.346279,
             0.364396,0.369234,0.328729,0.156433,0.050045,0.060182,0.5],
    "FREX": [0.004224,0.008045,0.052475,0.011376,0.009375,0.121968,0.352701,
             0.375521,0.380813,0.392213,0.341348,0.228792,0.249032,0.5],
    "ACPL": [0.011058,0.018117,0.101102,0.023203,0.020152,0.198078,0.443856,
             0.464076,0.468546,0.480676,0.429995,0.308153,0.334505,0.5],
}
_SOIL = [0.014842,0.018151,0.045298,0.027397,0.030822,0.090073,
         0.191457,0.208704,0.216508,0.184635,0.098196,0.041594,0.047973,1.0]

ILLUMINATION = {
    "Apr": dict(sun_zenith_deg=56.0, sun_azimuth_deg=0.0),
    "Jul": dict(sun_zenith_deg=41.0, sun_azimuth_deg=0.0),
}

BAND_WAVELENGTH_NM = {
    "O03":  442.95, "O04":  490.45, "O06":  560.40, "O08":  665.25,
    "O10":  681.55, "O11":  709.10, "O12":  754.15, "M08":  833.50,
    "O17":  865.60, "MD5": 1242.50, "M11": 1610.50, "MD7": 2120.00,
    "M12": 2199.00,"PUR": 550.00,
}


def _weighted_band(table: Dict[str, List[float]], band: str) -> float:
    i = _BAND_IDX[band]
    return sum(SPECIES_WEIGHT[sp] * table[sp][i] for sp in SPECIES_WEIGHT)


def get_band_params(band: str) -> dict:
    if band not in _BAND_IDX:
        raise ValueError(f"Unknown band '{band}'. Choose from: {BANDS}")
    return dict(
        rho=_weighted_band(_REFL, band),
        tau=_weighted_band(_TRAN, band),
        rsoil=_SOIL[_BAND_IDX[band]],
        wavelength_nm=BAND_WAVELENGTH_NM[band],
    )


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class TreeInstance:
    """One tree instance in the scene."""
    species:   str
    stem_id:   str           #: e.g. "BEPE_042"
    tx: float                #: world x of tree base [m]
    ty: float                #: world y of tree base [m]
    tz: float                #: world z of tree base (usually 0) [m]
    rot_deg: float           #: rotation around z-axis [°]
    height_m: float = 0.0   #: tree height (read from file or species mean)

    group: str = ""  # size-class group name, e.g. "BEPE3" (synthetic path only)

    @property
    def rot_matrix_xy(self) -> np.ndarray:
        """2D rotation matrix (x/y plane) for this instance."""
        c = math.cos(math.radians(self.rot_deg))
        s = math.sin(math.radians(self.rot_deg))
        return np.array([[c, -s], [s, c]])


@dataclass
class WoodMesh:
    """
    All wood triangles for the whole stand.

    vertices : (M, 3) float64   world coordinates [m]
    faces    : (F, 3) int32     indices into vertices
    species  : (F,)  str array  species tag per face (for per-species ρ_wood)
    """
    vertices: np.ndarray          # (M, 3)
    faces:    np.ndarray          # (F, 3)  int32
    face_species: np.ndarray      # (F,)    object (str)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    def triangle_vertices(self, face_idx: int) -> np.ndarray:
        """Return the 3 × 3 vertex array for one triangle."""
        return self.vertices[self.faces[face_idx]]


# ---------------------------------------------------------------------------
# RAMI-5 file parsers
# ---------------------------------------------------------------------------

def parse_treelocations(path: str) -> np.ndarray:
    """
    Parse a RAMI-5 *_treelocations.dat file.

    Format: two whitespace-separated columns per line (no header):
        tx [m]   ty [m]
    Returns (N, 2) float64 array of (tx, ty) positions.
    """
    rows = []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                rows.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue
    return np.array(rows, dtype=np.float64) if rows else np.zeros((0, 2))


def _instances_from_treelocations(
    scene_dir: str,
    rng: np.random.Generator,
) -> List[TreeInstance]:
    """
    Build TreeInstance list from per-group *_treelocations.dat files.

    For each group in JBS_GROUPS, tries to read
    ``{scene_dir}/{group_name}_treelocations.dat``.
    Falls back to uniform random positions for groups whose file is missing.
    Heights: group mean ± 20% Gaussian scatter.
    Rotations: uniform random (not provided in the files).
    """
    instances: List[TreeInstance] = []
    for grp_name, grp in JBS_GROUPS.items():
        sp_code = GROUP_TO_SPECIES[grp_name]
        loc_path = Path(scene_dir, f"{grp_name}_treelocations.dat")
        if loc_path.exists():
            xy = parse_treelocations(str(loc_path))
        else:
            # Synthetic fallback for this group
            xy = np.column_stack([
                rng.uniform(0, LX, grp.n_trees),
                rng.uniform(0, LY, grp.n_trees),
            ])

        for k, (tx, ty) in enumerate(xy):
            h = float(rng.normal(grp.mean_height_m, grp.mean_height_m * 0.20))
            h = max(2.0, h)
            rot = float(rng.uniform(0, 360))
            instances.append(TreeInstance(
                species=sp_code,
                stem_id=f"{grp_name}_{k+1:03d}",
                tx=float(tx), ty=float(ty), tz=0.0,
                rot_deg=rot,
                height_m=h,
                group=grp_name,
            ))
    return instances


def parse_scene_def(path: str) -> List[TreeInstance]:
    """
    Parse the RAMI-5 scene.def (or equivalent) instance file.

    Expected format (whitespace-separated, # = comment):
        species  stem_id  tx  ty  tz  rot_deg  [height_m]
    Example:
        BEPE  BEPE_001  12.34  45.67  0.0  23.5  18.2

    The last column (height) is optional; if absent the species mean is used.
    """
    instances = []
    with open(path) as f:
        for raw in f:
            line = raw.split("#")[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            species  = parts[0].upper()
            stem_id  = parts[1]
            tx, ty, tz = float(parts[2]), float(parts[3]), float(parts[4])
            rot_deg    = float(parts[5])
            height_m   = float(parts[6]) if len(parts) > 6 else 0.0
            instances.append(TreeInstance(species, stem_id, tx, ty, tz,
                                          rot_deg, height_m))
    return instances


def parse_leaf_positions(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse a RAMI-5 .lp leaf-position file (local tree coordinates).

    Expected format per line (whitespace-separated):
        x  y  z  nx  ny  nz
    or with an optional 7th column (area, ignored — we use LEAF_RADIUS):
        x  y  z  nx  ny  nz  [area]

    Returns
    -------
    pos    : (N, 3) float64   leaf centres [m, local frame]
    normal : (N, 3) float64   unit leaf normals [local frame]
    """
    pos_list, nrm_list = [], []
    with open(path) as f:
        for raw in f:
            line = raw.split("#")[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                nx, ny, nz = float(parts[3]), float(parts[4]), float(parts[5])
            except ValueError:
                continue
            pos_list.append([x, y, z])
            # normalise the normal vector
            n = math.sqrt(nx*nx + ny*ny + nz*nz)
            if n < 1e-9:
                n = 1.0
            nrm_list.append([nx/n, ny/n, nz/n])
    if not pos_list:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.array(pos_list, dtype=np.float64), np.array(nrm_list, dtype=np.float64)


def parse_obj_mesh(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse a Wavefront OBJ file (local tree coordinates, triangles only).

    Returns
    -------
    vertices : (M, 3) float64   vertex positions [m, local frame]
    faces    : (F, 3) int32     triangle face indices (0-based)
    """
    verts, faces = [], []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                # face entries may be "i", "i/j", or "i/j/k" — take vertex index only
                idx = [int(p.split("/")[0]) - 1 for p in parts]   # 0-based
                # triangulate (fan from first vertex if polygon)
                for k in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[k], idx[k+1]])
    if not verts:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32)
    return (np.array(verts, dtype=np.float64),
            np.array(faces, dtype=np.int32).reshape(-1, 3))


# ---------------------------------------------------------------------------
# Coordinate transform helpers
# ---------------------------------------------------------------------------

def _rot_z(angle_deg: float) -> np.ndarray:
    """3×3 rotation matrix around z-axis."""
    c = math.cos(math.radians(angle_deg))
    s = math.sin(math.radians(angle_deg))
    return np.array([[c, -s, 0.],
                     [s,  c, 0.],
                     [0., 0., 1.]])


def _apply_tree_transform(
    local_pos:    np.ndarray,    # (N, 3)  local coords
    local_normal: np.ndarray,    # (N, 3)  local normals (rotation only)
    inst:         TreeInstance,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply  R(φ) × local + [tx, ty, tz]  to positions;
    apply  R(φ) to normals (no translation).
    """
    R = _rot_z(inst.rot_deg)
    world_pos = (R @ local_pos.T).T + np.array([inst.tx, inst.ty, inst.tz])
    world_nrm = (R @ local_normal.T).T
    return world_pos, world_nrm


# ---------------------------------------------------------------------------
# Synthetic crown generator (fallback when RAMI files absent)
# ---------------------------------------------------------------------------

_RNG_LAD = np.random.default_rng(0)   # shared for reproducibility

def _beta_sample(a: float, b: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample from Beta(a,b) using numpy."""
    return rng.beta(a, b, n)


def _synthetic_crown_leaves(
    inst:   TreeInstance,
    sp:     SpeciesParams,
    rng:    np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate leaf positions for one tree using crown ellipsoid + LAD profile.

    Crown model (local z-up tree frame):
      - Crown base at z = sp.crown_base_frac × height
      - Crown top  at z = height
      - Crown semi-axes: Rx, Ry = sp.crown_rx_m in x/y
      - Crown semi-axis  Rz = sp.crown_rz_frac × (height − base) / 2  in z

    Vertical LAD distribution within the crown:
      P(z in [z_base, z_top]) ∝ Beta(lad_alpha, lad_beta)
      (Beta(2,3) ≈ upper-concentrated, Beta(3,2) ≈ lower-concentrated)

    Leaf normals: uniform hemisphere (ALA='UNI') in the RAMI-5 reference.
    """
    height = sp.mean_height_m if inst.height_m == 0.0 else inst.height_m
    leaf_area = SPECIES_LEAF_AREA_M2.get(inst.species, math.pi * LEAF_RADIUS**2)
    n_leaf_tree = max(1, round(sp.mean_foliage_area_m2 / leaf_area))

    z_base = sp.crown_base_frac * height
    z_top  = height
    Rz     = sp.crown_rz_frac * (z_top - z_base) / 2.0
    z_mid  = (z_base + z_top) / 2.0

    # Vertical position within crown via Beta distribution
    u_z    = _beta_sample(sp.lad_alpha, sp.lad_beta, n_leaf_tree * 4, rng)
    z_local = z_base + u_z * (z_top - z_base)   # in [z_base, z_top]

    # Radial position: sample in ellipsoid cross-section at each z
    # Ellipsoid: (x/Rx)² + (y/Rx)² + ((z-z_mid)/Rz)² ≤ 1
    Rx = sp.crown_rx_m
    r_frac = np.sqrt(np.maximum(0, 1.0 - ((z_local - z_mid) / Rz) ** 2))
    r_max  = Rx * r_frac
    # Uniform disk sampling
    phi_r = rng.uniform(0, 2 * math.pi, n_leaf_tree * 4)
    r_samp = r_max * np.sqrt(rng.uniform(0, 1, n_leaf_tree * 4))
    x_local = r_samp * np.cos(phi_r)
    y_local = r_samp * np.sin(phi_r)

    # Keep only points strictly inside ellipsoid (reject those with r_frac=0)
    inside = r_frac > 1e-3
    x_local = x_local[inside]
    y_local = y_local[inside]
    z_local = z_local[inside]

    # Trim or top-up to exactly n_leaf_tree
    if len(x_local) >= n_leaf_tree:
        idx = rng.choice(len(x_local), n_leaf_tree, replace=False)
        x_local = x_local[idx]; y_local = y_local[idx]; z_local = z_local[idx]
    else:
        # Extra uniform fallback to hit exact count
        extra = n_leaf_tree - len(x_local)
        r_ex = Rx * np.sqrt(rng.uniform(0, 1, extra))
        p_ex = rng.uniform(0, 2*math.pi, extra)
        z_ex = z_base + rng.uniform(0, 1, extra) * (z_top - z_base)
        x_local = np.concatenate([x_local, r_ex * np.cos(p_ex)])
        y_local = np.concatenate([y_local, r_ex * np.sin(p_ex)])
        z_local = np.concatenate([z_local, z_ex])

    local_pos = np.column_stack([x_local, y_local, z_local])

    # Uniform spherical leaf normals (ALA = UNI)
    cos_z = rng.uniform(-1, 1, n_leaf_tree)
    phi_n = rng.uniform(0, 2 * math.pi, n_leaf_tree)
    sin_z = np.sqrt(np.maximum(0, 1 - cos_z**2))
    local_nrm = np.column_stack([
        sin_z * np.cos(phi_n),
        sin_z * np.sin(phi_n),
        cos_z,
    ])
    # Enforce upward hemisphere for normal (convention: nz ≥ 0)
    local_nrm[local_nrm[:, 2] < 0] *= -1

    return local_pos, local_nrm


def _synthetic_tree_wood(
    inst:   TreeInstance,
    sp:     SpeciesParams,
    rng:    np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a simplified parametric wood mesh for one tree (local frame).

    Model: trunk cylinder + main branches (cones/cylinders).
    Returns (vertices, faces) in LOCAL tree coordinates.

    The mesh is coarse but sufficient for shadow computation.
    """
    height = sp.mean_height_m if inst.height_m == 0.0 else inst.height_m
    verts_list: List[np.ndarray] = []
    faces_list: List[np.ndarray] = []
    v_offset = 0

    def _add_cylinder(
        x0: float, y0: float, z0: float,
        x1: float, y1: float, z1: float,
        r0: float, r1: float, n_sides: int = 6,
    ):
        nonlocal v_offset
        ax = np.array([x1-x0, y1-y0, z1-z0], dtype=float)
        length = np.linalg.norm(ax)
        if length < 1e-4:
            return
        ax /= length

        # Build orthonormal frame
        ref = np.array([0., 0., 1.]) if abs(ax[2]) < 0.9 else np.array([1., 0., 0.])
        u   = np.cross(ax, ref); u /= np.linalg.norm(u)
        v_  = np.cross(ax, u)

        angles = np.linspace(0, 2*math.pi, n_sides, endpoint=False)
        cos_a  = np.cos(angles); sin_a = np.sin(angles)

        bottom = np.array([x0, y0, z0]) + np.outer(r0*cos_a, u) + np.outer(r0*sin_a, v_)
        top    = np.array([x1, y1, z1]) + np.outer(r1*cos_a, u) + np.outer(r1*sin_a, v_)

        ring_v = np.vstack([bottom, top])     # 2*n_sides vertices
        verts_list.append(ring_v)

        ring_f = []
        for k in range(n_sides):
            k1 = (k + 1) % n_sides
            b0, b1 = v_offset + k,  v_offset + k1
            t0, t1 = v_offset + n_sides + k, v_offset + n_sides + k1
            ring_f.append([b0, t0, b1])
            ring_f.append([b1, t0, t1])
        faces_list.append(np.array(ring_f, dtype=np.int32))
        v_offset += 2 * n_sides

    # ── Trunk ──────────────────────────────────────────────────────────────
    trunk_r_base = max(0.05, 0.01 * height)      # radius at base
    trunk_r_top  = trunk_r_base * 0.4             # taper
    crown_base   = sp.crown_base_frac * height
    _add_cylinder(0, 0, 0, 0, 0, crown_base, trunk_r_base, trunk_r_top)

    # ── 4 main branches (randomly rotated) ────────────────────────────────
    n_branch = 4
    branch_len = (height - crown_base) * 0.6
    branch_elev = math.radians(40)               # elevation above horizontal
    for k in range(n_branch):
        phi_b = k * math.pi / 2.0 + rng.uniform(-0.3, 0.3)
        dx = branch_len * math.cos(branch_elev) * math.cos(phi_b)
        dy = branch_len * math.cos(branch_elev) * math.sin(phi_b)
        dz = branch_len * math.sin(branch_elev)
        zb = crown_base + (height - crown_base) * rng.uniform(0.1, 0.4)
        _add_cylinder(0, 0, zb, dx, dy, zb + dz,
                      trunk_r_top * 0.8, trunk_r_top * 0.2)

    if not verts_list:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32)
    return (np.vstack(verts_list),
            np.vstack(faces_list).reshape(-1, 3))


# ---------------------------------------------------------------------------
# Main scene builder
# ---------------------------------------------------------------------------

def load_jbs_scene(
    scene_dir: Optional[str],
    band:  str = "PUR",
    month: str = "Jul",
    P_LAI: float = 0.5,
    leaf_radius_std_mul: float = 0.5,
    voxel_size: float = VOXEL_SIZE,
    seed:  int = 42,
    max_leaves: Optional[int] = None,
    verbose: bool = True,
) -> Tuple["Scene", WoodMesh]:  # type: ignore[name-defined]
    """
    Load (or synthesise) the JBS stand and return (scene, wood).

    Parameters
    ----------
    scene_dir : str or None
        Path to the RAMI-5 JBS download directory (containing scene.def).
        If None (or the directory does not exist) the synthetic fallback is used.
    band : str
        RAMI-5 spectral band tag (default 'O08').
    month : str
        'Apr' or 'Jul' — controls sun geometry.
    P_LAI : float
        Bernoulli thinning factor for the derivative estimator.
        For BRF-only validation use P_LAI=1.0.
    leaf_radius_std_mul : float
        ``std_LAI = std_mul * LAI_r`` for the path-derivative model (matches
        ``SceneConfig.leaf_radius_std_mul``). Use ~0.001 for nearly deterministic
        leaf size; default 0.5.
    voxel_size : float
        DDA grid cell size [m]. Default 1.0 m.
    seed : int
        RNG seed.
    max_leaves : int or None
        Cap total leaf count (useful for quick tests). None = no cap.
    verbose : bool
        Print progress.

    Returns
    -------
    scene  : canopi.scene.Scene
        Leaf disks + voxel grid ready for the path tracer.
    wood   : WoodMesh
        All wood triangles for the stand (for shadow / wood BRF computation).
    """
    from canopi.scene import Scene, _assign_leaves_fast

    rng = np.random.default_rng(seed)
    bp  = get_band_params(band)
    illum = ILLUMINATION[month]

    use_treelocations = (scene_dir is not None and
                         Path(scene_dir, "ACPL_treelocations.dat").exists())
    use_scene_def     = (scene_dir is not None and not use_treelocations and
                         Path(scene_dir, "scene.def").exists())
    use_files = use_treelocations or use_scene_def

    if verbose:
        if use_treelocations:
            src = f"*_treelocations.dat files in '{scene_dir}'"
        elif use_scene_def:
            src = f"scene.def in '{scene_dir}'"
        else:
            src = "synthetic model"
        print(f"[JBS] Loading HET09_JBS_SUM from {src}")
        print(f"[JBS] Band={band} ({bp['wavelength_nm']:.0f} nm)  "
              f"Month={month}  SZA={illum['sun_zenith_deg']}°  P_LAI={P_LAI}")

    # ── 1. Tree instances ───────────────────────────────────────────────────
    if use_treelocations:
        instances = _instances_from_treelocations(scene_dir, rng)
        if verbose:
            print(f"[JBS]   {len(instances)} tree instances from treelocations files")
    elif use_scene_def:
        instances = parse_scene_def(str(Path(scene_dir, "scene.def")))
        if verbose:
            print(f"[JBS]   {len(instances)} tree instances from scene.def")
    else:
        instances = _synthetic_instances(rng)
        if verbose:
            print(f"[JBS]   {len(instances)} synthetic tree instances")

    # ── 2. Collect leaf positions and wood triangles ────────────────────────
    all_pos:  List[np.ndarray] = []
    all_nrm:  List[np.ndarray] = []
    all_rad:  List[np.ndarray] = []   # per-leaf disk radius (species-specific)
    all_vtx:  List[np.ndarray] = []
    all_fcs:  List[np.ndarray] = []
    all_fsp:  List[np.ndarray] = []
    vtx_offset = 0

    for inst in instances:
        # Use group params (18-class) when available (synthetic path),
        # fall back to species-level params (file-based path fallback).
        sp = JBS_GROUPS.get(inst.group) or JBS_SPECIES.get(inst.species)
        if sp is None:
            continue

        # Per-species leaf disk radius from RAMI inventory
        leaf_area_sp = SPECIES_LEAF_AREA_M2.get(inst.species,
                                                  math.pi * LEAF_RADIUS**2)
        r_sp = math.sqrt(leaf_area_sp / math.pi)

        # ── Foliage ──────────────────────────────────────────────────────
        if use_files:
            lp_path = Path(scene_dir, inst.species,
                           f"{inst.stem_id}.lp")
            if lp_path.exists():
                loc_pos, loc_nrm = parse_leaf_positions(str(lp_path))
            else:
                loc_pos, loc_nrm = _synthetic_crown_leaves(inst, sp, rng)
        else:
            loc_pos, loc_nrm = _synthetic_crown_leaves(inst, sp, rng)

        if len(loc_pos) == 0:
            continue

        all_rad.append(np.full(len(loc_pos), r_sp, dtype=np.float64))

        w_pos, w_nrm = _apply_tree_transform(loc_pos, loc_nrm, inst)

        # Wrap x/y into periodic domain [0, LX) × [0, LY)
        w_pos[:, 0] %= LX
        w_pos[:, 1] %= LY
        # Clamp z into [0, H]  (soil is z<0 in canopi → negate)
        w_pos[:, 2] = np.clip(w_pos[:, 2], 0.0, H)

        # Convert to canopi convention: z increases downward from sky
        # canopi uses z ∈ [−H, 0], sky at 0, soil at −H
        w_pos[:, 2] = -w_pos[:, 2]   # flip: world z=0(soil)→−H; z=H(sky)→0

        all_pos.append(w_pos)
        all_nrm.append(w_nrm)

        # ── Wood ─────────────────────────────────────────────────────────
        if use_files:
            obj_path = Path(scene_dir, inst.species,
                            f"{inst.stem_id}.obj")
            if obj_path.exists():
                loc_vtx, loc_fcs = parse_obj_mesh(str(obj_path))
            else:
                loc_vtx, loc_fcs = _synthetic_tree_wood(inst, sp, rng)
        else:
            loc_vtx, loc_fcs = _synthetic_tree_wood(inst, sp, rng)

        if len(loc_vtx) > 0:
            w_vtx, _ = _apply_tree_transform(
                loc_vtx,
                np.zeros_like(loc_vtx),   # normals not needed for vertices
                inst,
            )
            w_vtx[:, 0] %= LX
            w_vtx[:, 1] %= LY
            w_vtx[:, 2] = np.clip(w_vtx[:, 2], 0.0, H)
            w_vtx[:, 2] = -w_vtx[:, 2]

            all_vtx.append(w_vtx)
            global_fcs = loc_fcs + vtx_offset
            all_fcs.append(global_fcs)
            all_fsp.append(np.full(len(loc_fcs), inst.species, dtype=object))
            vtx_offset += len(w_vtx)

    # ── 3. Concatenate ──────────────────────────────────────────────────────
    if not all_pos:
        raise RuntimeError("No leaves generated — check species table / files.")

    leaf_pos = np.vstack(all_pos)
    leaf_nrm = np.vstack(all_nrm)
    leaf_rad = np.concatenate(all_rad)   # per-leaf species-specific radii

    if max_leaves is not None and len(leaf_pos) > max_leaves:
        idx = rng.choice(len(leaf_pos), max_leaves, replace=False)
        leaf_pos = leaf_pos[idx]
        leaf_nrm = leaf_nrm[idx]
        leaf_rad = leaf_rad[idx]
        if verbose:
            print(f"[JBS]   Leaves capped to {max_leaves:,}")

    n_leaf_rami = len(leaf_pos)

    # ── Over-populate by 1/P_LAI ─────────────────────────────────────────
    # Resample existing leaves with replacement to add (1/P - 1) × N extra
    # leaves, so Bernoulli thinning recovers the original RAMI LAI.
    # Small random jitter avoids exact overlap of duplicated disks.
    n_leaf_target = max(n_leaf_rami, round(n_leaf_rami / P_LAI))
    n_extra = n_leaf_target - n_leaf_rami
    if n_extra > 0:
        idx_extra = rng.choice(n_leaf_rami, n_extra, replace=True)
        r_mean    = float(leaf_rad.mean())
        jitter    = rng.normal(0.0, r_mean * 0.1, (n_extra, 3))
        jitter[:, 2] = 0.0
        extra_pos = leaf_pos[idx_extra] + jitter
        extra_pos[:, 0] %= LX
        extra_pos[:, 1] %= LY
        extra_pos[:, 2] = np.clip(extra_pos[:, 2], -H, 0.0)
        leaf_pos = np.vstack([leaf_pos, extra_pos])
        leaf_nrm = np.vstack([leaf_nrm, leaf_nrm[idx_extra]])
        leaf_rad = np.concatenate([leaf_rad, leaf_rad[idx_extra]])
        if verbose:
            print(f"[JBS]   Over-populated: {n_leaf_rami:,} → {len(leaf_pos):,} "
                  f"leaves  (× 1/P_LAI = {1/P_LAI:.1f})")

    n_leaf = len(leaf_pos)
    if verbose:
        print(f"[JBS]   Total leaves: {n_leaf:,}")

    # ── 4. Leaf radii — per-species from RAMI inventory ───────────────────
    # LAI_r = mean one-sided leaf area across all leaves (used by derivative)
    # std_LAI = 0: each leaf has a fixed area determined by its species
    LAI_r   = float(np.mean(np.pi * leaf_rad**2))
    std_LAI = 0.0
    r_max_scalar = float(leaf_rad.max())
    leaf_radius     = leaf_rad.copy()
    leaf_radius_max = leaf_rad.copy()   # fixed radii → r_max = r

    # n_leaf_norm and LAI_Max from inflated leaf count and mean leaf area
    surf = LX * LY
    LAI_Max     = float(np.sum(np.pi * leaf_rad**2)) / surf
    n_leaf_norm = n_leaf / surf

    if verbose:
        actual_LAI_eff = LAI_Max * P_LAI
        print(f"[JBS]   LAI_r (mean): {LAI_r:.6g} m²  "
              f"r range: [{leaf_rad.min()*100:.2f}, {leaf_rad.max()*100:.2f}] cm")
        print(f"[JBS]   LAI_Max: {LAI_Max:.3f}  "
              f"E[LAI_seen]={actual_LAI_eff:.3f}  (target: {LAI_EFF})")

    # ── 5. Voxel grid ──────────────────────────────────────────────────────
    pos_max  = np.array([LX, LY, -H], dtype=np.float64)
    max_voxel = np.maximum(
        (np.abs(pos_max) / voxel_size).astype(int), 1
    )
    n_vox = int(max_voxel[0] * max_voxel[1] * max_voxel[2])

    if verbose:
        print(f"[JBS]   Voxel grid: {max_voxel[0]}×{max_voxel[1]}×{max_voxel[2]}"
              f"  ({n_vox:,} voxels)")

    # Fast vectorised assignment (primary voxel only — valid since r=0.05 << 1.0 m)
    voxel_count, voxel_leaves = _assign_leaves_fast(leaf_pos, voxel_size, max_voxel)

    if verbose:
        occupied = int((voxel_count > 0).sum())
        print(f"[JBS]   Occupied voxels: {occupied:,} / {n_vox:,}  "
              f"({100*occupied/n_vox:.1f}%)")

    # ── 6. Build Scene ──────────────────────────────────────────────────────
    # leaf zenith/azimuth derived from normals (not used by GPU path tracer
    # directly, but required by Scene dataclass)
    nz = np.clip(leaf_nrm[:, 2], -1.0, 1.0)
    leaf_zenith  = np.arccos(nz)
    leaf_azimuth = np.arctan2(leaf_nrm[:, 1], leaf_nrm[:, 0])

    scene = Scene(
        leaf_pos=leaf_pos,
        leaf_normal=leaf_nrm,
        leaf_zenith=leaf_zenith,
        leaf_azimuth=leaf_azimuth,
        leaf_radius=leaf_radius.copy(),
        leaf_radius_max=leaf_radius_max,
        pos_max=pos_max,
        voxel_size=float(voxel_size),
        max_voxel=max_voxel,
        voxel_count=voxel_count,
        voxel_leaves=voxel_leaves,
        rho=float(bp["rho"]),
        tau=float(bp["tau"]),
        rsoil=float(bp["rsoil"]),
        LAI_r=float(LAI_r),
        std_LAI=float(std_LAI),
        LAI_Max=float(LAI_Max),
        P_LAI=float(P_LAI),
        n_leaf_norm=float(n_leaf_norm),
    )

    # ── 7. WoodMesh ─────────────────────────────────────────────────────────
    if all_vtx:
        wood = WoodMesh(
            vertices=np.vstack(all_vtx),
            faces=np.vstack(all_fcs).reshape(-1, 3).astype(np.int32),
            face_species=np.concatenate(all_fsp),
        )
    else:
        wood = WoodMesh(
            vertices=np.zeros((0, 3)),
            faces=np.zeros((0, 3), dtype=np.int32),
            face_species=np.array([], dtype=object),
        )

    if verbose:
        print(f"[JBS]   Wood mesh: {wood.n_vertices:,} vertices, "
              f"{wood.n_faces:,} triangles")
        print(f"[JBS]   rho={scene.rho:.5f}  tau={scene.tau:.5f}  "
              f"rsoil={scene.rsoil:.5f}")
        print(f"[JBS]   Scene ready.")

    return scene, wood


# ---------------------------------------------------------------------------
# Synthetic instance generator
# ---------------------------------------------------------------------------

def _synthetic_instances(rng: np.random.Generator) -> List[TreeInstance]:
    """
    Place trees uniformly at random in [0,LX)×[0,LY) with random rotation.
    Iterates over the 18 RAMI size-class groups (JBS_GROUPS).
    """
    instances = []
    for grp_name, grp in JBS_GROUPS.items():
        sp_code = GROUP_TO_SPECIES[grp_name]
        for k in range(grp.n_trees):
            tx  = float(rng.uniform(0, LX))
            ty  = float(rng.uniform(0, LY))
            rot = float(rng.uniform(0, 360))
            h   = float(rng.normal(grp.mean_height_m, grp.mean_height_m * 0.20))
            h   = max(2.0, h)
            instances.append(TreeInstance(
                species=sp_code,
                stem_id=f"{grp_name}_{k+1:03d}",
                tx=tx, ty=ty, tz=0.0,
                rot_deg=rot,
                height_m=h,
                group=grp_name,
            ))
    return instances


# ---------------------------------------------------------------------------
# Observation config helper
# ---------------------------------------------------------------------------

def jbs_obs_config(
    band:  str = "O08",
    month: str = "Jul",
    obs_zenith_max_deg: float = 70.0,
    obs_step_deg: float = 2.0,
) -> dict:
    """
    Return a dict with all parameters needed by the renderer except the scene.

    Use this with render_gpu_scene() (below) or unpack into your own call.
    """
    bp    = get_band_params(band)
    illum = ILLUMINATION[month]
    return dict(
        sun_zenith_deg=illum["sun_zenith_deg"],
        sun_azimuth_deg=illum["sun_azimuth_deg"],
        obs_zenith_max_deg=obs_zenith_max_deg,
        obs_step_deg=obs_step_deg,
        **bp,
    )


# ---------------------------------------------------------------------------
# Renderer integration
# ---------------------------------------------------------------------------

def render_jbs(
    scene,
    band:  str = "PUR",
    month: str = "Jul",
    n_photons: int = 500_000,
    max_depth: int = 50,
    obs_zenith_max_deg: float = 75.0,
    obs_step_deg: float = 2.0,
    compute_derivative: bool = False,
    seed: int = 42,
    force_cpu: bool = False,
):
    """
    Run the GPU (or CPU) path tracer on a JBS scene built by load_jbs_scene().

    Parameters
    ----------
    compute_derivative : bool
        If True, also compute dBRF/dLAI (fictive, edge, optimal).
        Requires the scene to have been built with P_LAI < 1.0 (e.g. 0.5).

    Returns a RenderResult (same structure as renderer.render()).
    """
    from canopi.geometry import polar_vect
    from canopi.renderer import _build_obs_dirs, RenderResult
    from canopi.derivative import optimal_weight
    import math
    import numpy as np

    illum = ILLUMINATION[month]
    deg   = math.pi / 180

    # Class body assigns to these names, so RHS must not repeat them (unbound local).
    _oz_max = obs_zenith_max_deg
    _oz_step = obs_step_deg
    _md = max_depth
    _sd = seed
    _np = n_photons

    # Build observation directions using a minimal SceneConfig-like object
    class _FakeCfg:
        sun_zenith_deg     = illum["sun_zenith_deg"]
        sun_azimuth_deg    = illum["sun_azimuth_deg"]
        obs_zenith_max_deg = _oz_max
        obs_step_deg       = _oz_step
        image_size         = (64, 64)
        compute_derivative = False
        max_depth          = _md
        rr_depth           = 999
        rr_min             = 0.5
        energy_threshold   = 1e-5
        edge_threshold     = 2.5
        seed               = _sd
        n_photons          = _np
        P_LAI              = scene.P_LAI

    obs_deg, obs_dirs, n_pp = _build_obs_dirs(_FakeCfg())
    n_obs = len(obs_dirs)
    tha_s   = (180.0 - illum["sun_zenith_deg"]) * deg
    phi_s   = illum["sun_azimuth_deg"] * deg
    vec_sun = polar_vect(tha_s, phi_s)

    try:
        from canopi.gpu import (compute_brf_gpu, compute_deriv_gpu,
                                     GPU_AVAILABLE)
        _gpu_ok = GPU_AVAILABLE and not force_cpu
    except ImportError:
        _gpu_ok = False

    # ── GPU path ──────────────────────────────────────────────────────────
    if _gpu_ok:
        if compute_derivative:
            lum, D_fic, D_edg = compute_deriv_gpu(
                scene, obs_dirs, vec_sun, n_photons,
                max_depth=max_depth, seed=seed,
            )
            brf       = math.pi * lum.sum(axis=0) / n_photons
            deriv_fic = D_fic.mean(axis=0)
            deriv_edg = D_edg.mean(axis=0)
            w         = optimal_weight(D_fic, D_edg)
            deriv_opt = w * deriv_fic + (1.0 - w) * deriv_edg
        else:
            lum = compute_brf_gpu(
                scene, obs_dirs, vec_sun, n_photons,
                max_depth=max_depth, seed=seed,
            )
            brf       = math.pi * lum.sum(axis=0) / n_photons
            deriv_fic = np.zeros(n_obs)
            deriv_edg = np.zeros(n_obs)
            deriv_opt = np.zeros(n_obs)
            w         = np.full(n_obs, 0.5)

    # ── CPU fallback ──────────────────────────────────────────────────────
    else:
        from canopi.intersection import warmup
        from canopi.nee import NEEEngine
        from canopi.path_tracer import trace_photon
        from canopi.derivative import compute_photon_derivative

        warmup(scene)
        collect = compute_derivative
        nee = NEEEngine(scene, collect_deriv=collect)
        le  = np.ones(scene.n_leaf, dtype=np.int8)
        nee.warmup(obs_dirs, le)

        rng_r   = np.random.default_rng(seed)
        lum     = np.zeros(n_obs)
        max_c   = max_depth + 5
        All_fic = np.zeros((n_photons, max_c, n_obs))
        All_edg = np.zeros((n_photons, max_c, n_obs))

        for n_pt in range(n_photons):
            if collect:
                le = (rng_r.random(scene.n_leaf) < scene.P_LAI).astype(np.int8)
                scene.resample_radii(rng_r)
            P = np.array([
                rng_r.uniform(0, scene.pos_max[0]),
                rng_r.uniform(0, scene.pos_max[1]),
                0.0,
            ])
            ref, contacts, nc = trace_photon(
                scene, nee, P, vec_sun, obs_dirs,
                6.77e-5, 1e-12, le, max_depth, 1e-5,
                2.5, rng_r, collect, rr_depth=999,
            )
            lum += ref[:nc].sum(axis=0)

            if collect and nc > 0:
                df, de = compute_photon_derivative(
                    contacts, ref, nc,
                    scene.LAI_r, scene.std_LAI, scene.LAI_Max,
                    scene.P_LAI, scene.n_leaf_norm, n_obs,
                )
                nc_ = min(nc, max_c)
                All_fic[n_pt, :nc_] = df[:nc_]
                All_edg[n_pt, :nc_] = de[:nc_]

        brf = math.pi * lum / n_photons
        if collect:
            D_fic     = All_fic.sum(axis=1)   # (n_photons, n_obs)
            D_edg     = All_edg.sum(axis=1)
            deriv_fic = D_fic.mean(axis=0)
            deriv_edg = D_edg.mean(axis=0)
            w         = optimal_weight(D_fic, D_edg)
            deriv_opt = w * deriv_fic + (1.0 - w) * deriv_edg
        else:
            deriv_fic = np.zeros(n_obs)
            deriv_edg = np.zeros(n_obs)
            deriv_opt = np.zeros(n_obs)
            w         = np.full(n_obs, 0.5)

    return RenderResult(
        obs_zenith_deg=obs_deg,
        obs_dirs=obs_dirs,
        n_pp=n_pp,
        brf=brf,
        image=np.zeros((64, 64)),
        deriv_fic=deriv_fic,
        deriv_edg=deriv_edg,
        deriv_opt=deriv_opt,
        n_photons=n_photons,
        opt_weights=w,
    )


# ---------------------------------------------------------------------------
# Stand statistics diagnostic
# ---------------------------------------------------------------------------

def print_stand_stats(scene, wood: WoodMesh) -> None:
    """Print key stand statistics for verification."""
    import numpy as np

    n   = scene.n_leaf
    LAI_r = scene.LAI_r
    surf  = scene.pos_max[0] * scene.pos_max[1]
    actual_LAI = n * LAI_r / abs(surf)

    print()
    print("═" * 50)
    print("  JBS stand statistics")
    print("═" * 50)
    print(f"  Leaves        : {n:,}")
    print(f"  Leaf radius   : {math.sqrt(LAI_r/math.pi)*100:.1f} cm")
    print(f"  Actual LAI    : {actual_LAI:.3f}  (target {LAI_EFF})")
    print(f"  Plot area     : {abs(surf):.1f} m²  ({scene.pos_max[0]:.1f}×{scene.pos_max[1]:.1f})")
    print(f"  Canopy height : {abs(scene.pos_max[2]):.1f} m")
    print(f"  Voxel size    : {scene.voxel_size} m")
    grid = scene.max_voxel
    print(f"  Voxel grid    : {grid[0]}×{grid[1]}×{grid[2]} = {grid[0]*grid[1]*grid[2]:,}")
    print(f"  ρ_leaf        : {scene.rho:.5f}")
    print(f"  τ_leaf        : {scene.tau:.5f}")
    print(f"  ρ+τ           : {scene.rho+scene.tau:.5f}")
    print(f"  ρ_soil        : {scene.rsoil:.5f}")
    print(f"  P_LAI         : {scene.P_LAI}")
    print(f"  LAI_Max       : {scene.LAI_Max:.3f}")
    print()
    print(f"  Wood mesh     : {wood.n_vertices:,} vertices, {wood.n_faces:,} faces")
    print("═" * 50)
    print()


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time, sys

    print("JBS heterogeneous loader — self-test (synthetic mode)")
    t0 = time.time()

    scene, wood = load_jbs_scene(
        scene_dir=None,
        band="O08",
        month="Jul",
        P_LAI=1.0,
        max_leaves=500_000,  # cap for quick test
        verbose=True,
    )
    print(f"Load time: {time.time()-t0:.1f}s")
    print_stand_stats(scene, wood)

    # Verify leaf z-range
    import numpy as np
    z = scene.leaf_pos[:, 2]
    print(f"Leaf z range: [{z.min():.2f}, {z.max():.2f}]  "
          f"(should be in [{-H:.1f}, 0.0])")

    # Verify leaf normal convention (upward hemisphere)
    nz = scene.leaf_normal[:, 2]
    frac_up = (nz >= 0).mean()
    print(f"Fraction with nz ≥ 0: {frac_up:.3f}  (should be 1.0 — upward convention)")

    # Species breakdown (approx from synthetic model)
    total_trees = sum(sp.n_trees for sp in JBS_SPECIES.values())
    print(f"\nExpected tree counts (synthetic, {total_trees} total):")
    for sp_name, sp in JBS_SPECIES.items():
        w = SPECIES_WEIGHT[sp_name]
        print(f"  {sp_name}: {sp.n_trees} trees  weight={w:.3f}")
