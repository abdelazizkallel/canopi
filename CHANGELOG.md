# Changelog

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
