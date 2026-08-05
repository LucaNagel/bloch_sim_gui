# Changelog

All notable changes to BlochSimulator are documented in this file.

The project follows [Semantic Versioning](https://semver.org/).

## [2.2.0] - Unreleased

### Added

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

### Changed

- Generated spoilers are now treated as ideal transverse crushers: at the
  explicitly marked spoiler end, transverse magnetization (`Mx` and `My`) is
  set to zero while longitudinal magnetization (`Mz`) and the remaining
  simulation dynamics are preserved. This avoids implying unresolved
  intravoxel dephasing before subvoxel simulation is available.
- Expanded sequence result export with physical units and improved Cartesian,
  spiral, and multi-receive-channel acquisition handling.
- Refined the sequence simulation workspace, phantom tools, volume viewer, and
  application layout.

### Compatibility notes

- The ideal spoiler behavior is applied only to spoiler endpoints explicitly
  identified by generated sequence metadata; arbitrary gradients are not
  automatically classified as spoilers.
- Existing projects and simulations without spatial B1 fields continue to use
  uniform transmit and receive sensitivity.

[2.2.0]: https://github.com/LucaNagel/bloch_sim_gui/compare/v2.1.2...v2.2.0
