"""
Canopy RT — Geometry helpers
"""
import numpy as np


def polar_vect(theta, phi):
    """
    Spherical → Cartesian unit vector(s).
    theta : polar angle  (0 = +z, pi = -z)
    phi   : azimuth angle
    Returns shape (3,) for scalars, (N, 3) for arrays.
    """
    scalar = np.isscalar(theta)
    theta = np.atleast_1d(np.asarray(theta, float))
    phi   = np.atleast_1d(np.asarray(phi,   float))
    vecs  = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])
    return vecs[0] if scalar else vecs


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def propagation_limit_face(P: np.ndarray, vec_P: np.ndarray,
                            pos_max: np.ndarray):
    """
    Find where ray (P, vec_P) exits the scene bounding box.

    Box:  x ∈ [0, Lx],  y ∈ [0, Ly],  z ∈ [-H, 0]
    where pos_max = [Lx, Ly, -H]  (pos_max[2] < 0)

    Face encoding (matches MATLAB convention):
        1 → top   (z = 0)        — escape to sky
        2 → x = 0
        3 → x = Lx
        4 → y = 0
        5 → y = Ly
        6 → bottom (z = pos_max[2]) — soil

    Returns: (p_boundary [3,], face_id int)
    """
    Lx, Ly = pos_max[0], pos_max[1]
    z_bot   = pos_max[2]   # negative value

    # t values for each face (only positive t considered)
    EPS = 1e-12
    t_candidates = []

    def add(t, face):
        if t > EPS:
            t_candidates.append((t, face))

    if abs(vec_P[2]) > EPS:
        add((0.0     - P[2]) / vec_P[2], 1)   # z = 0  (top)
        add((z_bot   - P[2]) / vec_P[2], 6)   # z = -H (soil)

    if abs(vec_P[0]) > EPS:
        add((0.0 - P[0]) / vec_P[0], 2)   # x = 0
        add((Lx  - P[0]) / vec_P[0], 3)   # x = Lx

    if abs(vec_P[1]) > EPS:
        add((0.0 - P[1]) / vec_P[1], 4)   # y = 0
        add((Ly  - P[1]) / vec_P[1], 5)   # y = Ly

    if not t_candidates:
        return P.copy(), -1

    t_min, face = min(t_candidates, key=lambda x: x[0])
    p_boundary  = P + t_min * vec_P
    return p_boundary, face


def cyclic_point(p: np.ndarray, face: int, pos_max: np.ndarray) -> np.ndarray:
    """
    Apply cyclic (periodic) boundary condition on x / y faces.
    The photon re-enters from the opposite face.
    """
    EPS = 1e-9
    p_cyc = p.copy()
    if   face == 2:  p_cyc[0] = pos_max[0] - EPS   # exited x=0   → enter x=Lx
    elif face == 3:  p_cyc[0] = EPS                  # exited x=Lx  → enter x=0
    elif face == 4:  p_cyc[1] = pos_max[1] - EPS   # exited y=0   → enter y=Ly
    elif face == 5:  p_cyc[1] = EPS                  # exited y=Ly  → enter y=0
    return p_cyc


def pos3d_index(vijk: np.ndarray, max_voxel: np.ndarray) -> int:
    """Flat voxel index from 0-based (i,j,k) triple."""
    return int(vijk[0] + vijk[1] * max_voxel[0] + vijk[2] * max_voxel[0] * max_voxel[1])
