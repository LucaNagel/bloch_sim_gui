# Changelog

All notable changes to BlochSimulator are documented in this file.

The project follows [Semantic Versioning](https://semver.org/).

## [2.4.0] - 2026-08-16

### Added

- Added a private, macOS-only Metal numerical-feasibility probe and expanded
  dynamic precision validator with `.blochproj` loading, species/phase/
  low-signal/error-growth metrics, deterministic-repeat checks, memory
  enforcement, and machine-readable Apple M3 gate evidence. The Metal probe
  is not a selectable simulation backend because it failed the provisional
  accuracy gate.
- Added an opt-in CPU/Metal subvoxel hybrid to the validator and GUI dynamic
  kernel setting. It runs disjoint centre-symmetric Float64 calibration and
  validation samples concurrently with bounded GPU spin chunks, applies a
  complex phase/amplitude correction, and returns the untouched Float64 CPU
  result whenever the held-out accuracy check fails or Metal cannot be used.

- Added spatial transmit (B1+) and receive (B1-) field support to phantom and
  sequence simulations, including multi-channel receive sensitivity maps.
- Added a B1 field workspace for generating presets, importing field maps,
  transforming and previewing fields, and applying them to phantoms.
- Added self-contained `.blochproj` project files for saving and restoring the
  workspace state, phantom, B1 fields, sequence program, and simulation results.
- Added physical RF and gradient waveform data, effective B1 maps, and related
  metadata to sequence results and exports.
- Added advanced 3D multi-echo and spectral-selective bSSFP sequence builders,
  together with improved acquisition-frame and encoding metadata.
- Added additional sequence controls, result views, probe visualizations, and
  notebook export content.
- Added rotatable cylinder geometry in the phantom designer.
- Added phantom-voxel-referenced SS-bSSFP end-volume spoilers, validation of
  named receiver offsets against spectral-phantom peaks, exact 0° target
  pulses, and in-GUI FLASH/SS-bSSFP coherence and regular-grid alias checks.
- Static spectral sequence results now retain component-resolved signals for
  metabolite-specific reconstruction. A Python validator and Jupyter tutorial
  reproduce spoiler convergence and crosstalk checks from saved projects.

### Changed

- Dynamic pyruvate/lactate simulations now cache concentration coefficients,
  fuse coupled inflow/concentration kinetics in a native voxel kernel, and
  execute repeated RF raster intervals in persistent OpenMP waveform blocks.
  Single-coil ADC observation also avoids an all-ones weighting temporary.
- Sequence ETA calculation now excludes import/compilation time and uses a
  smoothed measured solver throughput after the first progress sample.
- Generated spoilers are now treated as ideal transverse crushers: at the
  explicitly marked spoiler end, transverse magnetization (`Mx` and `My`) is
  set to zero while longitudinal magnetization (`Mz`) and the remaining
  simulation dynamics are preserved. This avoids implying unresolved
  intravoxel dephasing when gradient-waveform subvoxel simulation is not
  selected.
- Expanded sequence result export with physical units and improved Cartesian,
  spiral, and multi-receive-channel acquisition handling.
- Refined the sequence simulation workspace, phantom tools, volume viewer, and
  application layout.

### Compatibility notes

- CPU Float64 remains the reference and default. The unchecked Metal probe is
  not exposed as a simulation backend. The GUI offers only the experimental
  checked CPU/Metal hybrid with automatic exact-CPU fallback; CPU-only imports
  and builds remain supported.

- The ideal spoiler behavior is applied only to spoiler endpoints explicitly
  identified by generated sequence metadata; arbitrary gradients are not
  automatically classified as spoilers.
- Existing projects and simulations without spatial B1 fields continue to use
  uniform transmit and receive sensitivity.

[2.4.0]: https://github.com/LucaNagel/bloch_sim_gui/compare/v2.3.0...v2.4.0
