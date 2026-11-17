# Changelog

## Unreleased

Checked against DART-Lux 5.11 on the JBS stand (identical leaves exported as OBJ): nadir BRF
agrees to -0.21 % (665 nm) and -0.04 % (866 nm), and to about 1 % over the principal and cross
planes with the exact hot spot.

### Fixed

- **JBS stand upside down.** `jbs_loader` and `jbs_species_loader` mapped a height h above the
  soil to z = -h while the soil is at z = -H, so the crown tops stood on the soil. Leaves and wood
  now go to z = h - H. Every JBS result changes.
- **Vertical voxel grid stopped above the soil.** The grid had floor(H / voxel_size) layers; the
  ray march clamps its start to the grid, so a ray leaving the soil restarted up to one voxel
  higher at the same (x, y). This breaks the sun-view correlation of the soil (JBS, H = 30.513 m,
  1 m voxels: hot-spot BRF 14 % low against DART; exact after the fix). The grid now has
  ceil(H / voxel_size) layers (`canopi.scene.voxel_grid_dims`); scenes pickled before are extended
  when loaded and again before every GPU upload (`ensure_soil_in_grid`).
- **Species kernel used the wrong species after a shadowed hit.** In
  `_trace_bwd_deriv_species_kernel` the species of the hit leaf was set only when the sun was
  visible from it, so the energy weight of a shadowed bounce used the previous sunlit leaf's
  reflectance. Bites when species have different optics.
- `compute_deriv_bwd_species_gpu` called the species kernel with an out-of-date argument list
  and failed at launch.

### Added

- **Leaf reflectance != transmittance per species** (`tau_sp_arr` in
  `render_image_bwd_deriv_species_gpu` and `compute_deriv_bwd_species_gpu`): reflection or
  transmission is chosen with rho_s / (rho_s + tau_s), the sun connection uses rho_s when sun and
  viewer are on the same side of the leaf and tau_s otherwise. `drho` is the derivative with
  tau_s / rho_s fixed (central differences: ratio 0.999-1.000 at 665 and 866 nm).
- **Exact periodic geometry, on by default in the scene builders.** `periodic_voxel_size` chooses
  the voxel size whose grid period equals the stand (the kernels wrap with n * voxel_size: with
  1 m voxels JBS was wrapped with 105 x 106 m and a 0.5 m strip carried leaves twice), and
  `_assign_leaves_bbox` registers each leaf in every voxel its disk overlaps (the centre-voxel
  assignment missed the part of a disk sticking into a neighbour voxel). Each moves the JBS BRF by
  about 0.4 %. `exact_geometry=False, voxel_size=1.0` restores v1.0.
- Species-aware shadow ray (`species_shadow`, `CANOPI_SPECIES_SHADOW`), default off.
- H∩B blocking distance merged with min (Eq. 28), CPU and GPU; edge estimator only.
- `canopi.scenes.spheres`: the spherical-crown stand of the paper.

## 1.0.1 — 2026-09-24

### Fixed

- **GPU derivatives: fictive leaves lying beyond the hit were kept when the hit fell in a
  boundary voxel layer.** The ray-marches `_dda_fwd_deriv` and `_dda_fwd_deriv_sp` trimmed
  the fictive and edge sets to the leaves in front of the hit only when the next step passed
  the hit. A hit in the top voxel layer of an upward ray (or the bottom layer of a downward
  one) leaves the grid on that step, and the loop exited before trimming. Leaves in a voxel
  are visited in list order, so a fictive leaf behind the hit could already be in the set;
  it stayed there and was charged `-1/(1-P)` at every later vertex of the photon. The trim
  now runs after the loop, on every exit — which is what the CPU tracer
  (`canopi/intersection.py`) has always done.

  *Effect.* dBRF/dLAI from the GPU read low in multiple scattering; single scattering was
  exact. On a 4 m × 4 m × 1 m test canopy with 0.25 m voxels and fixed leaf size, checked
  against paired finite differences in `P_LAI` on the same photons, `compute_deriv_bwd_gpu`
  read 7 % low at LAI 1.2 rising to 71 % low at LAI 7 (pooled −8.6σ over seven LAI values);
  after the fix every point agrees within ±2σ (pooled −0.4σ). The size of the error depends
  on the grid — the fewer vertical voxel layers, the larger the share of hits in a boundary
  layer — so scenes with many layers are affected less.

  *Affected:* every GPU derivative entry point — `compute_deriv_bwd_gpu`,
  `render_image_bwd_deriv_gpu`, `compute_deriv_bwd_species_gpu`,
  `render_image_bwd_deriv_species_gpu`, and the forward `compute_deriv_gpu` family. The fix
  was measured on `compute_deriv_bwd_gpu`; the other paths run through the same two
  functions. With a leaf-size spread (`leaf_radius_std_mul > 0`) the edge set is trimmed the
  same way, so `deriv_edg` was affected as well; that case was not measured.

  *Not affected:* BRF and every luminance output (bit-identical before and after), the CPU
  tracer, and the finite-difference validations.

## 1.0.0 — 2026-08-21

Initial public release.
