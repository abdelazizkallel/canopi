# The CANOPI model

A short guide to what the code computes and where each piece lives.
Full derivations are in the paper; this document maps them onto modules.

## 1. Probabilistic scene

A canopy realisation is drawn by combining two independent sources of randomness:

- **Structural** — every leaf is *active* with probability `P_LAI` (Bernoulli thinning).
  Inactive leaves are still traversed by rays but do not scatter; they are the *fictive* contacts.
- **Geometric** — leaf radius is perturbed, `r ~ (r₀, σ_r)`, controlled by
  `SceneConfig.leaf_radius_std_mul` and surfaced on the scene as `scene.std_LAI`.

Both act on the rendering equation through one **path-activation probability**, so the expected
render has a closed form. The effective leaf area index is

    LAI_eff = P_LAI · LAI_max

Code: `canopi/config.py` (`SceneConfig`), `canopi/scene.py` (voxel grid, `_assign_leaves_fast`),
`canopi/lad.py` (leaf angles), `canopi/scattering.py` (leaf and soil BRDF).

## 2. Two derivative estimators

Differentiating the expected render with respect to LAI gives two unbiased estimators.

**Structural (fictive-contact) term.** Along a path, count real leaf contacts `n_H` and accumulate
the fictive-contact weights `q_k`:

    D_fic = BRF / LAI_max · ( n_H / P_LAI − Σ_k q_k )

**Geometric (edge) term.** `D_edg` differentiates through the leaf-radius distribution; each
interaction contributes a score term weighted by how the perturbation moves the leaf boundary.

Both estimate the same quantity with very different variance depending on the regime — hence:

**Minimum-variance combination.**

    D_opt = w · D_fic + (1 − w) · D_edg

with `w` from `canopi.derivative.optimal_weight(D_fic, D_edg)`.

> **Degenerate case.** With fixed leaf radius (`scene.std_LAI = 0`, which is how the JBS scene is
> configured — all leaves are 0.05 m) the geometric term vanishes identically, so `Var(D_edg) = 0`.
> `optimal_weight` detects this (`Var(D_edg) < 1e-30`) and returns `w = 1`, i.e. use the structural
> estimator alone. Returning `w = 0` there would silently discard the whole derivative.
>
> In this regime `Σ_k q_k` reduces to `n_fic / (1 − P_LAI)` — a step function — and the estimator
> coincides with the classical Bernoulli-thinning form used by the Matlab VRT reference.

Code: `canopi/derivative.py`, plus the CUDA kernels `_trace_deriv_kernel` and
`_trace_bwd_deriv_species_kernel` in `canopi/gpu.py`.

## 3. Per-species Jacobian

Each leaf carries a species id, so contacts are binned per species and one backward pass returns a
derivative image per species. `render_image_bwd_deriv_species_gpu` returns

    brf   (H, W)               reflectance image
    dfic  (N_sp, H, W)         structural estimator
    dedg  (N_sp, H, W)         geometric estimator
    dopt  (N_sp, H, W)         minimum-variance combination
    drho  (N_sp, H, W)         ∂BRF/∂ρ_s, via a per-species interaction counter

Chain rule to the retrieved parameters — activation probability `P_s` and leaf reflectance `ρ_s`:

    J[:, s]     = π · LAI_max_s · dopt[s].ravel()
    J_rho[:, s] = drho[s].ravel() / rho_s

The factor π converts the directional derivative to a BRF derivative; `LAI_max_s` converts
∂/∂LAI_eff,s to ∂/∂P_s since `LAI_eff,s = P_s · LAI_max_s`.

## 4. Fixed-scene parameterisation

This is the single most important implementation detail.

Reloading the scene at a different `P_LAI` rescales the leaf count as `n_leaf ∝ 1/P_LAI`, which
leaves `LAI_eff = P_LAI · LAI_max` **unchanged** — useless for both finite differences and
retrieval.

Instead: load the geometry **once** at the true `P_LAI`, then vary the per-species activation
probability `P_LAI_sp` between renders. Leaf activation uses a deterministic hash of
`(photon_id, leaf_id, seed)`, so the active leaf sets are *nested* as `P` increases. With the same
seed and `N_spp` for the observation and the forward renders, the residual is **exactly zero** at
the true parameter — no Monte Carlo floor.

`P_LAI_sp` is an optional argument of `render_image_bwd_deriv_species_gpu` and defaults to a
uniform `P_LAI`, so existing calls are unaffected.

## 5. Inversion

Because one render yields BRF and the full Jacobian, Gauss–Newton / Levenberg–Marquardt costs one
render per iteration whatever the parameter count. The column-scaled LM step is

    ΔP[s] = −(JᵀJ + λ·diag(JᵀJ))⁻¹ Jᵀ r / LAI_max_s

followed by an **ascending line search**: try amplifications 2^k for k = k_max … 0 and accept the
largest that reduces the cost. BRF is concave in LAI, so the Gauss–Newton step systematically
undershoots and amplification above 1 is the norm.

For the joint two-band problem, RED resolves LAI (low reflectance ⇒ strong LAI sensitivity) and
NIR resolves ρ; the retrieval alternates band-wise LM steps within each iteration.

Worked implementations: `examples/inversion_joint_gaussnewton.py` (LAI only, 7 species) and
`paper/fig_inversion_species.py` (joint LAI + ρ, 21 parameters, 2 bands).

## 6. Kernel limits

The derivative kernels store per-path contact lists in thread-local arrays with compile-time caps,
set at the top of `canopi/gpu.py`:

    _MAX_C   = 60     max real-leaf contacts per photon (the H set)
    _MAX_FIC = 200    max fictive / edge leaves per set
    _N_SP    = 18     JBS size-class groups

These are a deliberate trade-off. Every thread allocates several arrays of `_MAX_FIC` doubles, so
the caps multiply directly into per-thread register and local-memory pressure — pushing them to
very large values exhausts GPU memory. If you see derivatives that look truncated at very high LAI,
raise them and watch memory; the estimator degrades gracefully rather than failing outright.
