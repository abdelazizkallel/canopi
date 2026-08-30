# CANOPI

[![CI](https://github.com/abdelazizkallel/canopi/actions/workflows/ci.yml/badge.svg)](https://github.com/abdelazizkallel/canopi/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**CAN**opy pr**O**babilistic rendering and **I**nversion — a unified probabilistic rendering
model for canopy reflectance **and its derivatives**, with an inversion engine built into the
renderer itself.

Reference implementation for:

> Kallel, Abdelaziz, *CANOPI: A Unified Probabilistic Rendering Model for Canopy Reflectance
> and its Derivatives — Bridging the Gap between Forward and Inverse Modeling*
> (August 27, 2026). SSRN: <https://ssrn.com/abstract=7365079> ·
> doi:[10.2139/ssrn.7365079](https://doi.org/10.2139/ssrn.7365079)

---

**[▶ Try the interface in your browser](https://abdelazizkallel.github.io/canopi/demo/)** — a recorded
session of CANOPI Studio: three stands, BRF and derivative images, and LAI retrieval running
iteration by iteration. No install, no GPU, and the page makes no network requests.

## What the model does

Most canopy radiative-transfer models return reflectance. CANOPI returns reflectance **and its
exact Jacobian from the same Monte Carlo pass**.

Leaves are **discrete disks of finite radius** everywhere in CANOPI — there is no turbid-medium
formulation. "Homogeneous" and "heterogeneous" below describe how those disks are distributed in
space (uniform over the slab, or grouped into crowns), not what they are made of. The finite leaf
size is precisely what the geometric estimator `D_edg` differentiates. A turbid formulation may be
added later.

Randomness is injected at the scene level by combining two mechanisms that earlier work treated
separately:

| Mechanism | Physical meaning | Derivative term |
|---|---|---|
| **Structural variability** | Bernoulli thinning — each leaf is active with probability P | `D_fic` (fictive-contact) |
| **Geometric variability** | random leaf-size perturbation | `D_edg` (edge) |

Both act on the rendering equation through a **single path-activation probability**, which gives a
closed-form expected-rendering expression. Differentiating it yields two unbiased estimators of
∂BRF/∂LAI whose **minimum-variance combination is available in closed form**:

    D_opt = w · D_fic + (1 − w) · D_edg,     w = argmin Var(D_opt)

with the structural term

    D_fic = BRF / LAI_max · ( n_H / P_LAI − Σ_k q_k )

where `n_H` counts real leaf contacts along the path and `Σ_k q_k` accumulates the fictive
(inactive-leaf) contacts. When leaf radius is fixed, `q_k` degenerates to a step function and the
estimator reduces to the classical thinning form — CANOPI recovers it as a special case.

**Why this matters for inversion.** One backward path-tracing pass returns the BRF image together
with the complete Jacobian with respect to per-species LAI *and* per-species leaf reflectance. A
Gauss–Newton / Levenberg–Marquardt retrieval therefore costs **one render per iteration regardless
of how many parameters are retrieved** — the paper demonstrates 21 parameters (7 species × LAI,
7 × ρ_RED, 7 × ρ_NIR) at one render per iteration.

---

## Install

```bash
git clone https://github.com/abdelazizkallel/canopi.git
cd canopi
pip install -e .
```

Python ≥ 3.9 with NumPy, SciPy, Matplotlib and Numba.

**GPU.** The 3-D stand renderer is a Numba CUDA kernel and needs an NVIDIA GPU with a working CUDA
toolkit. Check with:

```python
from canopi.gpu import GPU_AVAILABLE; print(GPU_AVAILABLE)
```

Without a GPU the pure-Python tracer (`canopi.render`) still works and is fine for homogeneous
stands, but the 2.5-million-leaf JBS stand is not practical on CPU. Development and all
published figures used an **NVIDIA A40 (46 GB)**; ~8 GB is enough for the 64×64 inversions.

---

## Scene data

The RAMI-5 `HET09_JBS_SUM` tree locations are **not redistributed here**. Download them once:

```bash
python scripts/download_jbs_data.py data/JBS_treelocations
```

That fetches 18 `*_treelocations.dat` files (~120 KB total) from the JRC RAMI-5 server. Everything
else — 2.5 M leaves, leaf normals, the voxel grid — is **rebuilt deterministically from `seed=42`**,
so no multi-gigabyte scene cache needs to be shipped or downloaded.

Override the location with `CANOPI_SCENE_DIR`; send figures elsewhere with `CANOPI_FIG_DIR`
(default `./figures`).

> If the tree-location files are missing, the loaders fall back to **synthetic** tree placement and
> emit a warning. That path runs, but it does *not* reproduce the published figures.

---

## Quick start

No scene download needed — a homogeneous canopy, BRF and derivative in one pass:

```bash
python examples/quickstart.py
```

```
device    : GPU   photons=100,000   LAI_eff=2.0   (71.5s)
BRF       : mean=0.021750  min=0.013830  max=0.040250
dBRF/dLAI : mean=-0.000151  (optimal estimator)
```

Most of that time is Numba JIT compilation of the CUDA kernels — the first call in a process is
slow, subsequent renders are not. In code:

```python
from canopi import SceneConfig
from canopi.gpu import render_gpu

cfg = SceneConfig(LAI_eff=2.0, P_LAI=0.5, rho=0.0632, tau=0.0213, rsoil=0.0274,
                  ALA='UNI', sun_zenith_deg=41.0, n_photons=200_000,
                  compute_derivative=True, seed=42)
res = render_gpu(cfg)      # use canopi.render(cfg) for the CPU tracer

res.brf          # BRF over the viewing hemisphere
res.deriv_opt    # minimum-variance dBRF/dLAI
res.deriv_fic    # structural estimator
res.deriv_edg    # geometric estimator
res.opt_weights  # the blend weight w
```

Explicit 3-D stand with per-species derivatives, on GPU:

```python
import numpy as np
from canopi.paths import scene_dir
from canopi.geometry import polar_vect
from canopi.scenes.jbs_loader import ILLUMINATION
from canopi.scenes.jbs_species_loader import load_jbs_scene_groups
from canopi.gpu import render_image_bwd_deriv_species_gpu

scene, wood, leaf_gid, stats = load_jbs_scene_groups(scene_dir(), band='O08', P_LAI=0.5)
scene.std_LAI = 0.0                                   # fixed leaf radius (0.05 m)

illum = ILLUMINATION['Jul']
deg   = np.pi / 180
vec_sun = polar_vect((180.0 - illum['sun_zenith_deg']) * deg,
                     (illum['sun_azimuth_deg'] + 180.0) * deg)

brf, dfic, dedg, dopt, drho = render_image_bwd_deriv_species_gpu(
    scene, obs_dir=polar_vect(0.0, 0.0), vec_sun=vec_sun,
    leaf_species_id=leaf_gid, species_stats=stats,
    img_H=64, img_W=64, N_spp=8)
# brf  : (H, W)            reflectance image
# dopt : (N_species, H, W) min-variance ∂BRF/∂LAI_s per pixel
# drho : (N_species, H, W) ∂BRF/∂ρ_s per pixel
```

---

## The API

### Layer 1 — scenes and forward rendering

| Function | Returns |
|---|---|
| `canopi.SceneConfig(...)` | dataclass holding every scene and sampling parameter (`LAI_eff`, `P_LAI`, `rho`, `tau`, `rsoil`, `ALA`, `sun_zenith_deg`, `n_photons`, `max_depth`, `seed`, …) |
| `canopi.render(cfg)` | CPU forward Monte Carlo render → `RenderResult` (BRF over the hemisphere, plus derivative fields when `compute_derivative=True`) |
| `canopi.gpu.render_gpu(cfg)` | same, on CUDA |
| `canopi.render_backward_image(cfg, …)` | backward (camera-side) BRF image |
| `canopi.render_backward_image_deriv(cfg, …)` | backward BRF image **+ derivative image** |

### Layer 2 — derivative estimators

| Function | Role |
|---|---|
| `canopi.derivative.optimal_weight(D_fic, D_edg)` | closed-form minimum-variance weight *w*; returns `w = 1` when the geometric term is degenerate (fixed leaf radius ⇒ `Var(D_edg) = 0`) |
| `canopi.gpu.compute_deriv_gpu(...)` | hemisphere-resolved `D_fic` / `D_edg` for homogeneous stands |
| `canopi.gpu.compute_deriv_bwd_species_gpu(...)` | per-species derivative, backward, angularly resolved |

### Layer 3 — explicit 3-D stands (RAMI-5 JBS)

| Function | Role |
|---|---|
| `canopi.scenes.spheres.load_sphere_scene(...)` | heterogeneous stand of spherical crowns (the paper's heterogeneous figures); needs no download |
| `canopi.scenes.jbs_loader.load_jbs_scene(...)` | single-species JBS stand |
| `canopi.scenes.jbs_species_loader.load_jbs_scene_groups(scene_dir, band, P_LAI, seed)` | JBS stand with 18 size-groups → 7 species; returns `(scene, wood, leaf_group_id, group_stats)` |
| `canopi.scenes.jbs_species_loader.build_scene_for_lai(...)` | rebuild a stand at a prescribed per-species LAI |
| `canopi.gpu.render_image_bwd_deriv_species_gpu(...)` | **the workhorse** — one pass → `(brf, dfic, dedg, dopt, drho)`; `species_shadow=True` also makes the sun-visibility ray read `P_LAI_sp` |
| `canopi.paths.scene_dir()` / `figure_dir()` | environment-overridable path resolution |

**The `P_LAI_sp` argument is the key to both validation and inversion.** Reloading the scene at a
different `P_LAI` rescales `n_leaf ∝ 1/P_LAI` and leaves `LAI_eff = P_LAI × LAI_max` *unchanged*.
Instead, load the geometry **once** and vary the per-species activation probability `P_LAI_sp`
between renders. A deterministic hash on `(photon_id, leaf_id, seed)` makes the leaf sets nested,
so reusing one seed across renders acts as *common random numbers*: the residual then falls to zero
at the true parameter instead of onto a Monte Carlo floor. This is not a correctness requirement —
a fresh seed per render converges too — but it is what keeps finite-difference checks and line
searches reliable at a few rays per pixel, which is far cheaper than averaging the noise away.

Jacobian assembly (chain rule through `P_s`):

    J[:, s]     = π · LAI_max_s · dopt[s].ravel()      ∂BRF/∂P_s
    J_rho[:, s] = drho[s].ravel() / rho_s              ∂BRF/∂ρ_s

---

## Reproducing the paper figures

All commands are run from the repository root, after `pip install -e .` and the data download.
Figures land in `$CANOPI_FIG_DIR` (default `./figures`). Timings are for one A40.

| Paper figure(s) | Command | Runtime |
|---|---|---|
| `jbs_tree_positions_18groups` | `python paper/fig_tree_positions.py` | 5 s, CPU |
| `bias_vs_LAI_*` | `python paper/fig_bias_vs_lai.py` | minutes, GPU |
| `convergence_fic_edg_*` | `python paper/fig_convergence_fic_edg.py` | minutes, GPU |
| `multiang_{angular,lai,rho,err}` | `python paper/fig_multiang_toy.py` | minutes, GPU |
| `jbs_O17_{nadir,hotspot}_{brf,dopt_*,contrib_*}` (22 figures) | `python paper/fig_jbs_species_images.py --band O17 --res 512 --n-spp 32` | ~1 h, GPU |
| `lai_rho_species_p035_k0_*` (16 figures) | step 1 below | hours, GPU |
| `lai_species_truerho_p035_k0_*` (11 figures) | step 2 below | hours, GPU |

`paper/run_all.py` drives all of these: `--list` shows the stages, `--only NAME` runs one,
`--fast` runs just the ones that finish in minutes, and `--quick` runs the heavy stages at reduced
resolution to exercise the pipeline (it does *not* reproduce the published figures).

The two inversion figure families come from the same script run twice, then a shared replot pass:

```bash
# 1. joint LAI + rho retrieval (21 parameters)
python paper/fig_inversion_species.py --init-p-lai 0.35 --k-max 0 \
       --out-prefix lai_rho_species_p035_k0

# 2. LAI-only retrieval, reflectance held at truth
python paper/fig_inversion_species.py --init-p-lai 0.35 --k-max 0 --true-rho \
       --out-prefix lai_species_truerho_p035_k0

# 3. render all 27 publication EPS from the two .pkl result files
python paper/fig_inversion_replot.py
```

Steps 1 and 2 each write a `.pkl` of the full iteration history; re-run either with `--plot-only`
to replot without re-simulating. Step 3 sets `text.usetex=True` and therefore needs a LaTeX
installation — the results themselves do not.

Three paper figures (`model_3D_Unified`, `Global_PRF`,
`Discret_xBRDF_Deriv_Hem_Hom_var_LAI`) are hand-drawn schematics and Matlab reference plots, and
are **not** generated by this repository.

See [docs/figures.md](docs/figures.md) for the complete figure-to-script map, including the
published filenames each script writes.

---

## Validating the derivative

The trust anchor is a finite-difference check of the per-species analytic derivative:

```bash
python examples/derivative_fd_check.py --res 64 --n-spp 64
```

It renders the reference BRF plus the analytic derivative once, then two forward-only renders at
`P ± δ`, and compares

    FD = [BRF(P+δ) − BRF(P−δ)] / (2δ)   against   AN = Σ_s π · dopt[s].mean() · LAI_max_s

At 64×64 with 64 samples per pixel this gives `FD/AN ≈ 0.95` with sign agreement across all seven
species — the residual gap is Monte Carlo noise at that sampling, and closes as `--n-spp` grows.

Other worked examples:

| Script | What it shows |
|---|---|
| `examples/quickstart.py` | homogeneous canopy, BRF + all three derivative estimators, no data needed |
| `examples/brf_deriv_images.py` | spatially resolved BRF and per-species ∂BRF/∂LAI images |
| `examples/inversion_single_species.py` | single-species golden-section retrieval |
| `examples/inversion_joint_gaussnewton.py` | joint 7-species Gauss–Newton, one render per iteration |

---

## Tests

```bash
pip install -e ".[dev]"
pytest -q                  # ~40 s, no GPU needed
pytest -q -m "not slow"    # skip the end-to-end CPU render
```

The suite runs entirely on CPU and is what CI executes on every push. It covers the
`optimal_weight` degenerate branches (where a bug once silenced the whole derivative), scene
construction invariants, the RAMI-5 stand metadata, path resolution, and a small end-to-end CPU
render. It also parses every script in `paper/` and `examples/` and checks that each symbol they
import from `canopi` exists — those scripts need a GPU and hours of compute, so this is the guard
against a refactor breaking them silently.

CI additionally downloads the RAMI-5 tree locations and reproduces the stand-map figure on a
GPU-less runner, publishing it as a build artifact.

---

## Repository layout

```
canopi/            the model
  config.py          SceneConfig
  scene.py           voxel grid, leaf assignment
  geometry.py        directions, polar_vect
  lad.py             leaf angle distributions
  scattering.py      leaf/soil BRDF
  intersection.py    ray–leaf intersection
  path_tracer.py     CPU tracer
  renderer.py        render / render_backward_image*
  derivative.py      D_fic, D_edg, optimal_weight
  gpu.py             CUDA kernels (forward, backward, per-species derivative)
  paths.py           CANOPI_SCENE_DIR / CANOPI_FIG_DIR resolution
  scenes/            RAMI-5 JBS loaders
paper/             one script per paper figure family + run_all.py
examples/          validation and worked examples
scripts/           data download
tests/             CPU test suite (what CI runs)
docs/              model notes, API notes, figure map
```

---

## Citation

If you use CANOPI, please cite the paper:

> Kallel, Abdelaziz, Canopi: A Unified Probabilistic Rendering Model for Canopy Reflectance and
> its Derivatives-Bridging the Gap between Forward and Inverse Modeling (August 27, 2026).
> Available at SSRN: <https://ssrn.com/abstract=7365079> or
> <http://dx.doi.org/10.2139/ssrn.7365079>

```bibtex
@article{kallel2026canopi,
  author  = {Kallel, Abdelaziz},
  title   = {{CANOPI}: A Unified Probabilistic Rendering Model for Canopy Reflectance and its
             Derivatives---Bridging the Gap between Forward and Inverse Modeling},
  journal = {SSRN Electronic Journal},
  year    = {2026},
  month   = aug,
  doi     = {10.2139/ssrn.7365079},
  url     = {https://ssrn.com/abstract=7365079}
}
```

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

Scene geometry comes from the **RAMI-5** benchmark (`HET09_JBS_SUM`) hosted by the European
Commission Joint Research Centre. Please cite RAMI when you use the JBS scene, and consult the
[RAMI website](https://rami-benchmark.jrc.ec.europa.eu/) for its terms of use.
