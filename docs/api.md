# API reference

Signatures below are taken from the code. `from canopi import ...` gives the four top-level
entry points; everything else is imported from its submodule.

## canopi.config

### `SceneConfig`

Dataclass holding the complete description of a homogeneous / turbid scene and its sampling.

| Group | Fields |
|---|---|
| Geometry | `scene_xy`, `scene_z`, `voxel_size`, `leaf_radius`, `leaf_radius_std_mul` |
| Optics | `rho`, `tau`, `rsoil` |
| Structure | `LAI_eff`, `P_LAI`, `ALA`, `voxel_lai_stats` |
| Illumination / view | `sun_zenith_deg`, `sun_azimuth_deg`, `obs_zenith_max_deg`, `obs_step_deg` |
| Sampling | `n_photons`, `photon_start_xy_min_frac`, `photon_start_xy_max_frac`, `max_depth`, `rr_depth`, `rr_min`, `image_size`, `seed` |
| Derivative | `compute_derivative`, `energy_threshold`, `edge_threshold` |

`leaf_radius_std_mul` sets the geometric variability: `std_LAI = leaf_radius_std_mul · LAI_r`.
Set it near zero (or assign `scene.std_LAI = 0.0` after building) for a fixed-radius canopy — this
is what the JBS scene uses, and it makes the geometric estimator degenerate by construction.

`ALA` selects the leaf-angle distribution (`'UNI'` = spherical); see `canopi/lad.py`.

## canopi.renderer

```python
render(cfg: SceneConfig) -> RenderResult
render_backward_image(cfg, obs_zenith_deg=0.0, obs_azimuth_deg=0.0,
                      img_H=64, img_W=64, N_spp=1)
render_backward_image_deriv(cfg, obs_zenith_deg=0.0, obs_azimuth_deg=0.0,
                            img_H=64, img_W=64, N_spp=1)
```

`RenderResult` carries `brf`, `obs_dirs`, `obs_zenith_deg`, `n_photons`, `n_pp`, `image`, and —
when `compute_derivative=True` — `deriv_fic`, `deriv_edg`, `deriv_opt`, `deriv_rho`, `opt_weights`.

## canopi.derivative

```python
optimal_weight(D_fic: np.ndarray, D_edg: np.ndarray) -> np.ndarray
```

Per-observation minimum-variance weight *w* for `D_opt = w·D_fic + (1−w)·D_edg`.

Degenerate handling (threshold `1e-30` on the variances):

- `Var(D_edg) ≈ 0` → `w = 1` — geometric term is degenerate, use the structural estimator alone.
  This is the fixed-leaf-radius case.
- `Var(D_fic) ≈ 0` → `w = 0`.
- otherwise `w = clip((Var(Y) − Cov) / (Var(X) + Var(Y) − 2·Cov), 0, 1)`.

## canopi.gpu

Import-time probe sets `GPU_AVAILABLE`. Compile-time kernel caps `_MAX_C = 60`,
`_MAX_FIC = 200`, `_N_SP = 18` live at the top of the module.

```python
render_gpu(cfg, threads_per_block=128, scene=None)

compute_brf_gpu(scene, obs_dirs, vec_sun, N, ...)
compute_deriv_gpu(scene, obs_dirs, vec_sun, N, ...)
compute_deriv_bwd_species_gpu(scene, obs_dirs, vec_sun, N,
                              leaf_species_id, species_stats,
                              max_depth=99, threads_per_block=128, seed=42)

render_image_bwd_gpu(scene, obs_dir, vec_sun, ...)
render_image_bwd_deriv_gpu(scene, obs_dir, vec_sun, ...)
render_image_bwd_deriv_species_gpu(scene, obs_dir, vec_sun,
                                   leaf_species_id, species_stats,
                                   img_H=64, img_W=64, N_spp=4,
                                   max_depth=99, threads_per_block=128, seed=42,
                                   P_LAI_sp=None, rho_sp_arr=None,
                                   _scene_gpu=None, _rng_states=None)

upload_scene_gpu(scene, leaf_species_id=None, species_stats=None)
create_rng_states(N_total, seed=42)
benchmark(scene, obs_dirs, vec_sun, cfg, N_list=None)
```

### `render_image_bwd_deriv_species_gpu` — the workhorse

Returns `(brf, dfic, dedg, dopt, drho)` with `brf` shaped `(H, W)` and the four derivative
arrays shaped `(N_species, H, W)`.

Two optional arguments carry the retrieval parameters:

- `P_LAI_sp` — per-species leaf activation probability. Defaults to a uniform `scene.P_LAI`, so
  existing calls are unchanged. **Vary this, not the scene, when you change LAI** (see
  [model.md §4](model.md)).
- `rho_sp_arr` — per-species leaf reflectance, enabling the `drho` output.

Pass `_scene_gpu` and `_rng_states` (from `upload_scene_gpu` and `create_rng_states`) across
iterations of a retrieval loop to avoid re-uploading 2.5 M leaves every render.

## canopi.scenes.jbs_loader

Single-species RAMI-5 `HET09_JBS_SUM`.

Module constants: `LX = 105.5115`, `LY = 106.1535`, `H = 30.513` (metres), `LAI_EFF = 3.442`,
`JBS_GROUPS` (18 size classes), `GROUP_TO_SPECIES`, `ILLUMINATION` (`'Apr'`: SZA 56°,
`'Jul'`: SZA 41°), and the band tables reached through `get_band_params(band)`. Note both loaders default to `band='PUR'` (the synthetic conservative band); pass `band='O08'` for RED.

```python
load_jbs_scene(scene_dir, band='PUR', month='Jul', P_LAI=0.5,
               leaf_radius_std_mul=0.5, voxel_size=1.0, seed=42,
               max_leaves=None, verbose=True) -> (scene, wood)
get_band_params(band) -> dict          # rho, tau, rsoil for a RAMI band
parse_treelocations(path) -> np.ndarray
jbs_obs_config(...)
render_jbs(...)
print_stand_stats(scene, wood)
```

## canopi.scenes.jbs_species_loader

Per-species / per-group version. Constants: `SPECIES_LIST`
(`['BEPE','TICO','ALGL','POTR','PIAB','FREX','ACPL']`), `N_SPECIES = 7`, `GROUP_LIST` (18),
`N_GROUPS = 18`, `GROUP_TO_SPECIES`.

```python
load_jbs_scene_species(scene_dir, band='PUR', month='Jul', P_LAI=0.5,
                       leaf_radius_std_mul=0.5, voxel_size=1.0, seed=42,
                       max_leaves=None, verbose=True, return_pools=False)

load_jbs_scene_groups(scene_dir, band='O08', month='Jul', P_LAI=0.5,
                      voxel_size=1.0, seed=42, max_leaves=None, verbose=True)
    -> (scene, wood, leaf_group_id, group_stats)

build_scene_for_lai(leaf_pools, LAI_eff_sp, P_LAI, band_params,
                    voxel_size=1.0, seed=0, verbose=False, nested_species=None)
```

`group_stats` holds `LAI_Max_sp` (per-group LAI including fictive leaves), `n_leaf_norm_sp`,
`rho_sp_arr` and `species_list` — `LAI_Max_sp` is what converts a derivative with respect to
`LAI_eff,s` into one with respect to the activation probability `P_s`.

## canopi.paths

```python
figure_dir(create=True) -> Path        # $CANOPI_FIG_DIR, default ./figures
scene_dir(required=False) -> str|None  # $CANOPI_SCENE_DIR, then ./JBS_treelocations,
                                       # then <repo>/data/JBS_treelocations
```

`scene_dir()` returns `None` with a warning when the RAMI files are absent — the loaders then fall
back to synthetic tree placement, which runs but does not reproduce the paper. Pass
`required=True` to raise instead.
