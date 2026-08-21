"""
Canopy RT — Scattering / Phase functions.

Lambertian leaf BRDF + transmittance, Lambertian soil.
Stokes matrix formalism (4×4), first component = intensity.
"""
import numpy as np


# ── Lambertian Stokes scatter matrix ──────────────────────────────────────────
def lambertian_matrix(rho: float) -> np.ndarray:
    """4×4 Mueller/Stokes matrix for a Lambertian surface with reflectance rho."""
    M = np.zeros((4, 4))
    M[0, 0] = rho
    return M


def phase_function(vec_in: np.ndarray, vec_out: np.ndarray,
                   scat_matrix: np.ndarray) -> np.ndarray:
    """
    Compute phase function value (scalar for Lambertian).
    For Lambertian: phase = scat_matrix[0,0] / pi
    Returns 4×4 Mueller matrix scaled by the phase value.
    """
    # Lambertian: isotropic in the hemisphere → f = rho/pi
    M = scat_matrix.copy()
    # The normalization factor is already in scat_matrix for leaf (rho or tau)
    # For soil: rsoil/pi is applied in Soil_Observ
    return M


def sample_scatter_direction(
    rng:       np.random.Generator,
    leaf_nr:   np.ndarray,   # leaf normal
    vec_in:    np.ndarray,   # incoming direction (toward leaf)
    rho:       float,
    tau:       float,
) -> tuple:
    """
    Sample a scattered direction from a leaf.

    Returns
    -------
    vec_out   : (3,) unit vector  — scattered direction
    is_refl   : bool              — True=reflection, False=transmission
    weight    : float             — energy weight = (rho+tau)
    """
    # Choose reflection or transmission
    p_refl = rho / (rho + tau) if (rho + tau) > 0 else 0.5
    is_refl = rng.random() < p_refl

    # Determine the hemisphere for the scattered ray
    cos_in = np.dot(-vec_in, leaf_nr)
    if cos_in < 0:
        # incoming from below
        if is_refl:
            normal_used = -leaf_nr   # reflect into lower hemisphere
        else:
            normal_used =  leaf_nr   # transmit into upper hemisphere
    else:
        if is_refl:
            normal_used =  leaf_nr   # reflect into upper hemisphere
        else:
            normal_used = -leaf_nr   # transmit into lower hemisphere

    # Cosine-weighted (Lambertian) hemisphere sample
    vec_out = _cos_sample_hemisphere(rng, normal_used)
    # weight = selected albedo (rho or tau) for throughput tracking (Russian Roulette)
    weight  = rho if is_refl else tau
    return vec_out, is_refl, weight


def sample_soil_direction(rng: np.random.Generator) -> np.ndarray:
    """Lambertian scatter off soil: cosine-weighted in upper hemisphere (z>0)."""
    return _cos_sample_hemisphere(rng, np.array([0.0, 0.0, 1.0]))


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cos_sample_hemisphere(rng: np.random.Generator, n: np.ndarray) -> np.ndarray:
    """Sample direction cosine-weighted around normal n."""
    # Build orthonormal basis
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(n, up)) > 0.99:
        up = np.array([1.0, 0.0, 0.0])
    t = np.cross(n, up); t /= np.linalg.norm(t)
    b = np.cross(n, t)

    # Malley's method
    r1  = rng.random()
    r2  = rng.random()
    r   = np.sqrt(r1)
    phi = 2 * np.pi * r2
    x   = r * np.cos(phi)
    y   = r * np.sin(phi)
    z   = np.sqrt(max(0.0, 1.0 - r1))

    return x * t + y * b + z * n


def leaf_nee_weight(
    vec_in:  np.ndarray,   # incoming direction at leaf
    vec_obs: np.ndarray,   # direction toward sensor (unit, pointing up)
    leaf_nr: np.ndarray,   # leaf normal
    rho:     float,
    tau:     float,
) -> float:
    """
    NEE weight for a leaf interaction.
    For Lambertian BRDF: f = rho/π (reflection) or tau/π (transmission).
    The cos(θ_obs) factor is NOT included — it is absorbed in the BRF definition.
    BRF = π * <f * T>  →  f = rho/π or tau/π.
    """
    cos_in_nr  = float(np.dot(-vec_in,  leaf_nr))
    cos_obs_nr = float(np.dot( vec_obs, leaf_nr))

    # transmission when incoming and outgoing on opposite leaf sides
    if cos_in_nr * cos_obs_nr < 0:
        return tau / np.pi
    else:
        return rho / np.pi


def soil_nee_weight(
    vec_obs: np.ndarray,
    rsoil:   float,
) -> float:
    """
    NEE weight for Lambertian soil.
    f = rsoil/π  (no cos_obs — absorbed in BRF definition).
    BRF = π * <rsoil/π> = rsoil  ✓  (isotropic, all directions)
    """
    return rsoil / np.pi
