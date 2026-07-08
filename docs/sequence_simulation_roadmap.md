# Sequence simulation roadmap

This file is the cross-session handoff. Update status, validation commands, and
the next concrete action before ending work on the sequence simulator.

The original detailed implementation plan is preserved in
[sequence_simulation_plan.md](sequence_simulation_plan.md). Stable technical
decisions are documented in
[sequence_simulation_architecture.md](sequence_simulation_architecture.md).

## Milestones

| Milestone | Status | Notes |
| --- | --- | --- |
| 0. Baseline and units | Complete | Phantom import fixed; split offset maps, voxel centres, conversions tested |
| 1. Sequence IR/compiler | Complete | Immutable events, legacy adapter, sparse RF-aware compiler |
| 2. Streaming C/Cython core | Complete | Chunked ADC/checkpoint/final output; OpenMP ADC reduction |
| 3. Python API/object integration | Complete | `simulate_sequence`, memory checks, HDF5/xarray exports |
| 4. Pulseq 1.5.0 import | Complete | Optional PyPulseq; RF/grad/ADC, trigger metadata, version gate |
| 5. Integrated GUI | Complete | Lazy desktop tab with loader, object controls, run/cancel, plots |
| 6. Single-Tx and Multi-Rx | Complete | Complex voxelwise B1+ scaling and coil-resolved ADC reduction; compiler timing unchanged |
| 7. Cartesian acquisition/reconstruction | Complete | ADC moments, bandwidth/layout model, EPI builder, 2D FFT/coil combination, GUI controls and views |
| 8. Spectral phantom designer/viewers | Complete | Shape ROIs, Lorentz peaks, persistence, independent-component simulation, orthogonal/3D views |
| 9. CSI acquisition/export and physical phantom views | Complete | Explicit ky-kx-FID layout, spatial/spectral FFTs, structured export, mm axes, analytic 2D/3D B0 maps |
| 10. Dynamic coupled species | Designed | Requires a Bloch-McConnell state kernel; implementation deferred |
| 11. Multi-Tx and WASM | Deferred | Separate follow-up milestones |

## Repository baseline

- Branch: `main`.
- User-owned changes present at start: modified `README.md`, untracked
  `test_sweep.ipynb`, and `test_sweep_data.npz`. Do not overwrite them.
- Targeted input/RF/frequency tests: 12 passed before implementation.
- The initial full local suite aborted in the Qt export path. Lazy creation of
  the new heavy pyqtgraph workspace removed the extra lifecycle pressure; the
  final complete suite now passes with the offscreen Qt platform.
- `simulate_phantom` initially failed because of an absolute `phantom` import.

## Validation log

- 2026-07-03: repository and C/Python architecture inspected; plan decisions
  fixed; no sequence implementation existed.
- 2026-07-03: milestones 0-5 implemented. Final suite: 84 tests, including
  legacy parity, Pulseq extended gradients, and a multiblock EPI readout.
  Native extension rebuilt successfully; Pulseq 1.5.0 reference sequence
  imported and simulated; wheel built with `--no-isolation`; example returned
  256 ADC samples.
- 2026-07-04: official Pulseq references checked from `pulseq/pulseq` tag
  `v1.5.0` (`7d249a0`) and `imr-framework/pypulseq` (`b1d574b`). The full GRE
  reference imported, compiled to 275,824 intervals/4,096 ADC samples, and ran
  end-to-end with finite output. The official PyPulseq EPI example imported as
  1.5.0, matched all 128 PyPulseq ADC times, and ran end-to-end. This exposed
  and fixed triangular trapezoids with zero flat time.
- 2026-07-04: compiler event accumulation changed from interval-by-event
  traversal to overlap-only accumulation. The official GRE compile fell to
  about 1.6 s locally without changing the compiled timing contract.
- 2026-07-04: complex single-Tx and multi-Rx maps implemented through Phantom,
  Cython, and the streaming C/OpenMP kernel. Singleton receive output remains
  backward-compatible; multi-coil output is coil-resolved.
- 2026-07-04: native extension rebuilt successfully; complete offscreen suite:
  89 tests passed.
- 2026-07-04: baseline committed as `464cbae5` before acquisition work; user
  README/notebook/data changes remained outside the commit.
- 2026-07-04: Cartesian acquisition/reconstruction milestone implemented.
  Results now expose ADC gradient moments; `CartesianAcquisition` validates
  dwell/grid ordering, reverses EPI lines, and reconstructs voxel-centred 2D
  images. `make_cartesian_epi` derives gradients and ADC timing from matrix,
  FOV, and dwell. Optional 3D voxel-volume signal weighting was added without
  changing the default signal scale. Complete offscreen suite: 94 tests passed;
  the 8x8 example reconstructed unit peak amplitude end-to-end.
- 2026-07-04: the Sequence Simulation GUI gained a Cartesian EPI source with
  independent read/phase matrices and sampling bandwidth, derived dwell/pixel
  bandwidth, signal-weighting selection, and k-space/IFFT result tabs. Complete
  offscreen suite: 96 tests passed.
- 2026-07-05: the Sequence Simulation control column was made vertically
  scrollable after validation at reduced window height, keeping Run/Cancel
  reachable with the expanded acquisition controls. Complete offscreen suite:
  97 tests passed.
- 2026-07-05: imported single-acquisition 2D Pulseq EPI gained strict Cartesian
  inference from ADC gradient moments and FOV, including fractional grid
  offsets. The example ADC aperture was centred correctly, numerically duplicate
  compiler boundaries are coalesced, Pulseq FOV synchronizes independent
  in-plane/through-plane object controls, and final-Mz display levels use the
  complete volume. Corrected 16x16 EPI was loaded, simulated, displayed, and
  reconstructed offscreen. Complete suite: 101 tests passed.
- 2026-07-06: added an interactive multi-shape spectral phantom designer with
  draggable ellipsoid/box ROIs, z extent, T1/B0 properties, and arbitrary
  amplitude/frequency/T2* Lorentz peaks. Spectral phantoms persist to NPZ/HDF5,
  load in the Phantom tab, and run in Sequence Simulation as independent
  components. Phantom and sequence-result workspaces gained orthogonal-slice,
  OpenGL 3D, voxel-spectrum, and checkpoint views with explicit slice-selection
  versus kz-IFFT semantics. The previously hard-disabled Phantom tab was
  re-enabled with lazy initialization; OpenGL is skipped only on the offscreen
  test platform. Complete offscreen suite: 108 tests passed. The shared volume
  viewer now also normalizes 1D/2D masks together with their data and resets
  stale slice indices when dimensions change.
- 2026-07-07: added an explicit 2D CSI acquisition model and validated the
  checked-in `csi_2d_centric.seq` end-to-end as 16x16 spatial encodings with
  256-point FIDs. The GUI now shows selected-FID k-space, spatial IFFT, and a
  voxel spectrum. NetCDF/NPZ/HDF5 exports contain sorted CSI k-space, spatial
  FIDs, spectra, and physical k/time/frequency coordinates. Phantom views use
  physical mm coordinates and a clearer 3D FOV frame; the designer gained
  linear/radial 2D/3D B0 maps and disables controls owned by external phantom
  editors. All 150 tests passed when test modules were run in isolated
  processes. A single monolithic offscreen run remains nondeterministic on
  macOS because PyQtGraph occasionally crashes natively after repeated widget
  construction; affected modules pass independently.
- 2026-07-08: added a CSI split view with reconstruction/k-space and
  spectrum/FID toggles, click-to-select voxel synchronization, and coupled
  multidimensional sliders. The main GUI now switches between Free Mode and a
  focused Sequence Mode that removes legacy single-spin, global playback, and
  run controls while retaining Sequence Simulation and Phantom workspaces.
- 2026-07-08: collapsed EPI and built-in phantom controls when another source
  owns their settings, repaired masked/constant spectral property-map display,
  and added in-memory reopening of the current spectral shape design.

## Current limitations

- Pulseq 1.5.1, soft delays, multiple transmit channels, coupled species,
  exchange, diffusion, flow, motion, and WASM are intentionally deferred.
- Spectral peaks are independent Lorentzian/T2* components. Coupled spin
  systems, exchange, J-coupling, and density-matrix evolution remain deferred.
- Dynamic pyruvate-to-lactate conversion is not approximated by rescaling
  already simulated static signals. Correct conversion requires coupled
  species state evolution during RF, relaxation, exchange, and ADC sampling.
- Receiver aperture/filtering and oversampling/decimation are not yet modeled;
  ADC values are instantaneous observations at dwell centres.
- Cartesian reshaping currently targets one 2D acquisition. Slice, echo,
  repetition, and Pulseq label dimensions remain chronological metadata work.
  Automatic inference is limited to x-read/y-phase ADC blocks with a Pulseq FOV
  definition and rejects unequal per-line kx grids.
- Pulseq triggers do not influence magnetization; they are retained as metadata.
- Checkpoints intentionally allocate `Ncheckpoint x Nvoxel x 3`; memory policy
  rejects unsafe requests before native execution.
- The integrated GUI is a native desktop workspace; the legacy Phantom and
  K-space tabs remain disabled as they were at the start of this work.

## Next action

Implement a coupled-species Bloch-McConnell kernel as the next spectral
milestone. Start with irreversible two-pool pyruvate-to-lactate exchange,
voxelwise initial concentrations and rate maps, then validate zero-exchange,
mass-balance, no-RF analytic, and static-phantom limits before exposing kinetic
controls in the Phantom designer. Radial UTE, true 3D Cartesian kz encoding,
receiver filtering, Pulseq 1.5.1/soft delays, and Multi-Tx remain separate
versioned milestones.
