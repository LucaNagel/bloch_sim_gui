# Changelog

All notable changes to BlochSimulator are documented in this file.

The project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [2.9.0] - 2026-10-17

### Added

- Added a strict Float64 spin-major dynamic executor with bounded work buffers,
  deterministic thread reduction, and cached affine propagators for repeated
  RF segments. Two-pool models support spatial complex Tx sensitivity,
  multiple receive coils, subvoxel sampling, and existing inflow modes.
- "Plan metabolite excitation…" in the RF settings of generated sequences
  designs one metabolite-selective spectral-spatial pulse per target
  (`design_metabolite_selective_pulse`) from a metabolite list. For FLASH,
  the ordered steps (for example pyruvate 10°, then lactate 30°) become the
  pulse schedule. A flip-angle matrix shows how much each pulse excites
  the other peaks.

### Changed

- The spectral-spatial pulse designer opens without simulating. Its excitation
  map runs in a background thread and is cached, so reopening the designer
  for an unchanged pulse is immediate.

- The settings dialog offers only Optimized and Strict reference sequence
  kernels and Native automatic (`native_parallel`, now the default) and
  Optimized NumPy dynamic kernels. Saved choices that are no longer offered
  fall back to the default.
- The kernel benchmark compares Sequence Bloch kernels against
  `strict_reference` instead of `reference`.
- `scripts/validate_dynamic_precision.py` compares dynamic kernels against the
  optimized Float64 result instead of a Float32 shadow run.

### Fixed

- Dynamic precision validation now executes requested subvoxel sampling and
  checks every pool, final state, checkpoint, and available reconstruction
  against explicit Float64 tolerances. Failed comparisons return a failing
  exit status; native-path variants and timestep refinement can be compared.

### Removed

- Removed the experimental Metal backend, including the `metal_hybrid`
  dynamic kernel and the Metal feasibility probe.
- Removed Float32 dynamic simulation. Dynamic state is always integrated in
  Float64; `BlochSimulator(dynamic_sequence_precision=...)` and
  `simulate_dynamic_sequence(simulation_precision=...)` no longer exist.
- Removed the Python-only `sequence_kernel="reference"`. Use
  `"strict_reference"`, which runs the same general rotation-matrix path
  without fast math.
- Removed the unused `simulate_phantom_grouped` extension function and the
  ineffective `use_grouped` argument of `BlochSimulator.simulate_phantom`.

## [2.8.4] - 2026-10-16

### Added

- Added manually entered, comma-separated parameter-loop values. The entered
  order and repeated values are retained, with a preview plot beside the
  existing formula workflow.
- Added export of parameter-loop results directly to separate project files to
  limit memory use, including configurable, collision-safe file names.
- Added configurable batch export names for simulations retained in the
  session.

### Changed

- Project archives are written atomically and stream compressed arrays directly
  into the archive, avoiding duplicate in-memory magnetization arrays.
- Added `ipympl` to the notebook dependencies and documented the Conda setup
  for notebooks and the Python API.

### Fixed

- Prevent switching the displayed session simulation while a parameter loop is
  running.

## [2.8.3] - 2026-10-14

### Added

- Disable the slice-selection gradient in RF + Slice Selection to inspect
  the full spatial magnetization after RF alone. Slice controls and rephasing
  become inactive, and the viewer labels the result After RF excitation.

### Fixed

- Update sequence progress and remaining-time estimates after each spectral
  component chunk, including single-component phantoms. Start timing the
  simulation rate after compilation and allocation.

## [2.8.2] - 2026-10-12

### Added

- Added RF + Slice Selection in Sequence Mode for 3D magnetization volumes
  after one excitation, with selectable slice normal and offset, shared RF
  pulse controls, optional slice rephasing, and Pulseq/notebook export.
- Automatically retain the state before optional rephasing and label both
  states in Spatial Magnetization. Results and controls persist in projects.
- Select individual parameter-loop runs directly in Spatial Magnetization,
  preserving the chosen map, observation stage, and pool when available.

### Fixed

- Use the sequence's declared nucleus when exporting physical RF and
  gradient waveforms for spectral component phantoms.

## [2.8.1] - 2026-10-11

### Added

- Added formula-based parameter loops with an editable expression, start and
  end values, point count, and a preview of the generated values. Expressions
  such as `sin(x)` and `x^2` are applied to evenly spaced input points.
- Added a simulation selector to Reconstruction Explorer so individual loop
  iterations can be inspected directly, with parameter values in their names.
- Added adaptive AHP/AFP integration for Free Induction Decay in Free Mode,
  with independent waveform sampling, internal integration, and saved output
  spacing, plus numerical validation and usage guidance.

### Changed

- Parameter loops accept finite values outside the sequence form's default
  ranges, including negative RF scales and scales above 10. Form controls
  expand to preserve requested values instead of silently clipping them.
- Reduced Free Mode memory use by storing a configurable output raster,
  bounding plot and animation samples, and releasing previous result views
  before a new simulation starts.
- Improved cancellation, progress reporting, and UI responsiveness during
  Free Mode simulations, including releasing the Python interpreter lock
  while the serial native kernel computes.

### Fixed

- Kept Reconstruction Explorer's simulation list synchronized with renamed
  and deleted session runs.
- Preserved exact final magnetization when the saved output raster does not
  include the last integration point.
- Corrected Free Induction Decay tail timing and kept RF waveform timing
  independent of saved result sampling in plots and notebook exports.
- Included the strict reference kernel's Cython source in source distributions
  so the package can be rebuilt from the published source archive.
- Made the commit test hook use the active Python environment instead of a
  machine-specific interpreter path.

### Compatibility notes

- Integer loop parameters remain rounded to the form's integer representation;
  the sequence builders still validate physical requirements such as positive
  durations and matrix sizes.
- Adaptive Free Mode integration retains the existing proton-based kernel
  convention. It does not correct non-proton nucleus scaling in that legacy
  path; see [AHP/AFP validation and limitations](docs/free_mode_ahp_performance.md).

## [2.8.0] - 2026-09-30

### Added

- Added spectral-spatial RF pulse design with two-dimensional excitation-map
  validation, metabolite-target presets, Pulseq export, and reusable pulse
  libraries with deterministic cyclic schedules for generated sequences.
- Added a strict Float64 sequence-simulation reference kernel, explicit
  numerical-contract metadata, and a shared capability registry for selecting
  and reporting available simulation backends.
- Added a bounded native execution plan for dynamic spectral simulations,
  supporting reproducible parent-voxel chunking, receive coils, spatial
  transmit fields, dynamic $B_0$, delayed kinetics, multi-pool conversion, and
  controlled fallback to the established NumPy path.
- Added expanded mouse-perfusion authoring and preview controls, including
  independent physical anatomy size and position, live perfusion views, and
  voxel-specific spectra.

### Changed

- Improved dynamic-simulation throughput with persistent native RF blocks,
  bounded coefficient tables, deterministic signal reduction, and automatic
  OpenMP parallelization for suitably large simulations.
- Improved sequence-simulation and reconstruction interfaces with clearer
  backend status, kernel availability, and retained project display state.
- Extended benchmarks, validation utilities, and developer documentation for
  native kernels, numerical contracts, and acceleration evidence.

### Compatibility notes

- The optimized dynamic native paths have a `float64-close` numerical
  contract; use the strict reference kernel when reproducible validation is
  required. Unsupported capabilities, constrained memory, and unavailable
  experimental hardware continue to fall back to the compatible CPU path.
- Spectral-spatial schedules are stored separately from pulse definitions.
  Existing projects containing the former single pulse setting are migrated to
  a deterministic legacy schedule.

## [2.6.3] - 2026-08-31

### Added

- Added an explicit sequence reference frequency for spectral and dynamic
  simulations, including Pulseq metadata, frequency validation, visualization,
  and Bruker export propagation.
- Added per-shape pyruvate-to-lactate conversion rates with optional spatial
  overrides, linearly interpolated inflow controls, and selectable set-value or
  simulated-value kinetics previews.
- Added the `T1_multiphantom.npz` spectral phantom as a downloadable GitHub
  Release asset.

### Changed

- Unified analytic RF pulse sampling between Free Mode and Sequence Mode and
  made pulse samples symmetric on the RF raster.
- Improved GUI layout, group-title styling, B1 preview synchronization,
  reconstruction contrast handling across slices and repetitions, and result
  tab guidance.
- Limited subvoxel sampling and physical spoiler diagnostics to
  gradient-waveform spoiling; ideal spoilers consistently use one spin per
  voxel.

### Fixed

- Corrected spectral reference offsets across static and dynamic phantoms and
  exports.
- Fixed spoiler-control visibility and consistency across generated and loaded
  sequence sources.

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

[2.6.3]: https://github.com/LucaNagel/bloch_sim_gui/compare/v2.6.2...v2.6.3
[2.8.0]: https://github.com/LucaNagel/bloch_sim_gui/compare/v2.7.0...v2.8.0
[2.5.0]: https://github.com/LucaNagel/bloch_sim_gui/compare/v2.4.0...v2.5.0
[2.4.0]: https://github.com/LucaNagel/bloch_sim_gui/compare/v2.3.0...v2.4.0
