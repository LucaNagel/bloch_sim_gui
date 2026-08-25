# Changelog

All notable changes to BlochSimulator are documented in this file.

The project follows [Semantic Versioning](https://semver.org/).

## [2.5.0] - 2026-08-25

### Added

- Added train-wide gradient-spoiler coherence analysis, automatic FLASH spoiler
  moments, and train-safe subvoxel-grid recommendations based on the actual ADC
  moment history.
- Added deterministic stratified subvoxel sampling as an alternative to the
  regular midpoint grid, together with memory-limit checks and convergence
  tests for large subvoxel simulations.
- Added a FLASH spoiler-train benchmark, flip-angle sweep tooling, and an
  expanded SS-bSSFP spoiling-validation tutorial.
- Added direct RF Pulse Designer and waveform-file input to all generated
  sequence families, including retained complex phase, carrier offset, and
  reference-flip-angle scaling.

### Changed

- Unified analytic RF generation across Free Mode, the GUI sequence builders,
  and standalone scripts. Sinc, SLR, Gaussian, and block pulses now report TBW
  and bandwidth from the completed pulse shape; SLR sharpness also controls the
  temporal lobe structure.
- Standardized generated-sequence RF fields, sampling-bandwidth handling, and
  scanner-raster rounding across Cartesian, radial, spectral, and multi-echo
  builders.
- Improved sequence-workspace memory guidance, B1 controls, slice exploration,
  and reconstruction contrast controls, including a series-wide contrast range
  with display headroom.

### Compatibility notes

- RF time-bandwidth product and bandwidth metadata for analytic pulses now
  describe the generated waveform rather than echoing a legacy construction
  parameter. Code that compares these fields to user input should use the
  reported values from the generated sequence.
- Existing midpoint subvoxel sampling remains the default; deterministic
  stratified sampling is opt-in.

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

[2.5.0]: https://github.com/LucaNagel/bloch_sim_gui/compare/v2.4.0...v2.5.0
[2.4.0]: https://github.com/LucaNagel/bloch_sim_gui/compare/v2.3.0...v2.4.0
