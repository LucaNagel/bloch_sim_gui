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
| 7. Multi-Tx, multi-species, WASM | Deferred | Separate follow-up milestones |

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

## Current limitations

- Pulseq 1.5.1, soft delays, multiple transmit channels, multiple species,
  exchange, diffusion, flow, motion, and WASM are intentionally deferred.
- Pulseq triggers do not influence magnetization; they are retained as metadata.
- Checkpoints intentionally allocate `Ncheckpoint x Nvoxel x 3`; memory policy
  rejects unsafe requests before native execution.
- The integrated GUI is a native desktop workspace; the legacy Phantom and
  K-space tabs remain disabled as they were at the start of this work.

## Next action

Implement Pulseq 1.5.1 extensions and soft-delay handling as a separately
versioned importer milestone, then design Multi-Tx without changing the
compiled sequence timing contract.
