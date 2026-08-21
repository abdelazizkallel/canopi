"""
Canopy RT — Leaf Angle Distribution (LAD)

Bin centres match MATLAB code exactly:
    t_l = pi/180 * [5,15,25,35,45,55,64,72,77,79,81,83,85,87,89]

Key fix: bin WIDTHS are irregular (especially the last 6 bins are very narrow).
The probability must weight each bin by its solid-angle width:
    w_i = cos(theta_{i-1/2}) - cos(theta_{i+1/2})  (solid-angle element)
so that uniform (UNI) gives a flat f(Omega) over the sphere.
"""
import numpy as np

# Bin centres [rad] — identical to MATLAB t_l
BIN_CENTRES_DEG = np.array(
    [5, 15, 25, 35, 45, 55, 64, 72, 77, 79, 81, 83, 85, 87, 89],
    dtype=float,
)
BIN_CENTRES = BIN_CENTRES_DEG * np.pi / 180

# Bin edges: midpoints between consecutive centres, plus 0 and pi/2
_edges_deg = np.concatenate([
    [0.0],
    (BIN_CENTRES_DEG[:-1] + BIN_CENTRES_DEG[1:]) / 2,
    [90.0],
])
BIN_EDGES = _edges_deg * np.pi / 180

# Solid-angle weights:  w_i = cos(edge_lo) - cos(edge_hi)
#   = integral of sin(theta) dtheta over the bin
BIN_WEIGHTS = np.cos(BIN_EDGES[:-1]) - np.cos(BIN_EDGES[1:])   # (15,)


def _lad_shape(theta: np.ndarray, ala: str) -> np.ndarray:
    """
    Unnormalised leaf-inclination density g(theta) at bin centres.
    This is the SHAPE before solid-angle weighting.
    For UNI: g(theta) = 1  (uniform over sphere)
    For HOR: all leaves horizontal  (delta at theta=0, approximated by first bin only)
    For PLA: g(theta) proportional to planophile
    etc.
    """
    ala = ala.upper()
    if ala in ('UNI', 'SPH'):
        return np.ones_like(theta)            # uniform over solid angle
    elif ala == 'HOR':                        # all horizontal: only first bin
        p = np.zeros_like(theta)
        p[0] = 1.0
        return p
    elif ala == 'PLA':                        # planophile — mostly horizontal
        return (2 / np.pi) * (1 + np.cos(2 * theta))
    elif ala == 'ERE':                        # erectophile — mostly vertical
        return (2 / np.pi) * (1 - np.cos(2 * theta))
    elif ala == 'EXT':                        # extremophile
        return (2 / np.pi) * (1 - np.cos(4 * theta))
    else:
        return np.ones_like(theta)


def get_lad_probs(ala: str) -> np.ndarray:
    """
    Normalised discrete probability over the 15 bins.
    p_i = g(theta_i) * w_i / sum(g * w)
    where w_i = solid-angle weight of bin i.
    This ensures E[cos(theta)] = 0.5 for UNI.
    """
    g = _lad_shape(BIN_CENTRES, ala)
    g = np.maximum(g, 0.0)
    p = g * BIN_WEIGHTS
    return p / p.sum()


def sample_leaf_orientations(n: int, ala: str, rng: np.random.Generator):
    """
    Sample n leaf orientations from the LAD.
    Zenith angles are drawn uniformly within each bin's edge range
    (not ±5° jitter) to avoid bias at bin boundaries.

    Returns
    -------
    zenith   : (n,)   leaf zenith angles [rad] in [0, pi/2]
    azimuth  : (n,)   leaf azimuth angles [rad] in [0, 2*pi]
    normals  : (n, 3) unit normal vectors
    """
    probs = get_lad_probs(ala)
    bins  = rng.choice(len(BIN_CENTRES), size=n, p=probs)

    # Sample uniformly within each bin (correct: inverse-CDF on cos)
    # cos is uniform between cos(edge_hi) and cos(edge_lo)
    lo = np.cos(BIN_EDGES[bins + 1])   # cos(upper edge)  — smaller value
    hi = np.cos(BIN_EDGES[bins])       # cos(lower edge)  — larger value
    cos_z = lo + (hi - lo) * rng.random(n)
    zenith = np.arccos(np.clip(cos_z, 0.0, 1.0))

    azimuth = rng.uniform(0.0, 2 * np.pi, n)

    normals = np.column_stack([
        np.sin(zenith) * np.cos(azimuth),
        np.sin(zenith) * np.sin(azimuth),
        np.cos(zenith),
    ])
    return zenith, azimuth, normals
