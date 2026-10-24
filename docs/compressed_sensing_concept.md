# Compressed sensing: module and integration concept

Compressed sensing (CS) changes two things: which k-space points a sequence
acquires, and how the image is reconstructed from the incomplete grid. The
Bloch kernel does not change at all. An undersampled sequence is simply a
sequence with fewer phase-encoding steps. CS therefore lives in its own
module, `blochsimulator.sequence.compressed_sensing`. It touches the builders
only through a sampling mask and the reconstruction only through one extra
method.

## Current state: retrospective module

The module is pure numpy and independent of Qt and of the acquisition
classes. Its grids use the layout and FFT convention of
`CartesianAcquisition.reconstruct`:

- spatial axes last;
- DC at index `n // 2`;
- image `fftshift(ifftn(ifftshift(kspace)))`;
- `norm` handled exactly like numpy.

| Function | Purpose |
| --- | --- |
| `variable_density_mask(shape, R, centre_size=, density_power=, seed=)` | Exactly `round(N/R)` points. The centre is fully sampled; the rest is drawn without replacement with probability `(1 - r)^p`. `shape` is `(ky,)` for 2D or `(ky, kz)` for 3D. |
| `kt_variable_density_mask(ky, frames, R, ...)` | Independent line draw per frame, for k-t sparsity. |
| `expand_phase_mask(mask, read)` | Repeats a line mask along the fully sampled read axis. |
| `mask_point_spread_function(mask)` | Transform PSF and its peak sidelobe, a measure of incoherence. |
| `zero_filled_reconstruction(kspace, mask)` | Reference: inverse FFT with the missing points set to zero. |
| `reconstruct_compressed_sensing(kspace, mask, wavelet_weight=, tv_weight=, temporal_weight=, temporal_axis=, ...)` | Primal-dual CS reconstruction. |
| `HaarWavelet`, `normalized_rmse` | Orthonormal multi-level Haar transform; error metric. |

`reconstruct_compressed_sensing` solves

    min_x 0.5 ||M F x - y||^2 + l_w ||W x||_1 + l_tv TV(x) + l_t ||D_t x||_1

with the Chambolle–Pock primal-dual algorithm (DOI 10.1007/s10851-010-0251-1)
and the sparse-MRI model of Lustig, Donoho and Pauly (DOI 10.1002/mrm.21391).
The terms are:

- `W`: Haar wavelet transform, zero-padded to a multiple of `2^levels`;
- `TV`: isotropic total variation over the spatial axes;
- `D_t`: finite difference along a leading frame axis.

Solver details:

- **Data consistency** has a closed-form proximal step for the masked
  unitary FFT, so one iteration costs two FFTs plus the sparsifying
  operators. On a 64×64 grid, 200 iterations take about 0.05 s.
- **Weights** are relative to the peak of the zero-filled image, so they do
  not depend on the signal scale. TV around 0.002 and wavelet around 0.003
  are good starting values.
- **Leading axes** that are not marked as temporal (coils, slices, echoes)
  are reconstructed independently.

### What the tests show

`tests/test_compressed_sensing.py` checks:

- masks: exact counts, a fully sampled centre, reproducible seeds, and
  frames that differ;
- operators: the Haar transform is an isometry with a matching adjoint (also
  on padded odd sizes), and the difference operators have correct adjoints;
- convention: a fully sampled grid reproduces the Cartesian inverse FFT
  exactly;
- solver: the objective decreases, and the reconstruction cases below stay
  under fixed error bounds.

During development, the solver was also compared with an independent ISTA
implementation of the same objective; both reach the same minimum.

The cases also measure how well CS works for each sampling pattern. nRMSE
values compared with the true image (for FLASH: compared with the fully
sampled reconstruction):

| Case | Zero-filled | CS |
| --- | --- | --- |
| 64², line mask R = 3, TV | 0.16 | 0.010 |
| 64², line mask R = 3, Haar only | 0.16 | 0.12 |
| 64², (ky, kz) point mask R = 3, Haar only | 0.19 | 0.013 |
| 48², 10 frames, k-t line mask R = 3, spatial TV | 0.22 | 0.021 |
| same, plus temporal term | 0.22 | 0.011 |
| same, identical lines in every frame | 0.23 | 0.063 |
| Bloch-simulated FLASH 32², R = 2.5, TV | 0.30 | 0.13 |

What follows from this:

1. **Incoherence along both axes matters.** A 2D line mask aliases only
   along ky. The Haar prior can barely separate that aliasing from the
   image, while TV still can. CS works best in 3D with (ky, kz) masks or in
   k-t with masks that change from frame to frame.
2. **k-t needs masks that differ per frame.** The temporal term halves the
   error, but only if the missing lines are different in every frame.
3. **Bloch data is harder than the ideal phantom.** The simulated FLASH
   signal carries T1/T2 transients along ky and a smooth image phase. TV on
   complex images penalizes phase gradients as well as edges.

   The current FLASH reconstruction shows a linear phase of about π/N per
   pixel along read. The k-space centre sits half a sample away from index
   `n // 2`. Magnitude images are unaffected, but phase-constrained CS
   (a low-resolution phase estimate from the fully sampled centre) would
   help here. That is a sensible next extension.

### Retrospective use

```python
import numpy as np
from blochsimulator.sequence import (
    expand_phase_mask, infer_cartesian_acquisition, load_pulseq,
    reconstruct_compressed_sensing, variable_density_mask,
)
from blochsimulator.simulator import BlochSimulator

program = load_pulseq("flash.seq")            # fully sampled sequence
acquisition = infer_cartesian_acquisition(program)
signal = BlochSimulator().simulate_sequence(program, phantom).signal

# (phase, read) grid with the voxel-centred phase of acquisition.reconstruct()
dx = acquisition.fov_m[0] / acquisition.read_matrix
dy = acquisition.fov_m[1] / acquisition.phase_matrix
kspace = acquisition.reshape_signal(signal) * np.exp(2j * np.pi * (
    np.asarray(acquisition.ky_cyc_per_m)[:, None] * dy / 2
    + np.asarray(acquisition.kx_cyc_per_m)[None, :] * dx / 2
))
mask = expand_phase_mask(
    variable_density_mask((acquisition.phase_matrix,), 2.5, seed=0),
    acquisition.read_matrix,
)
result = reconstruct_compressed_sensing(kspace, mask, tv_weight=0.002)
image, reference = result.image, result.zero_filled
```

Retrospective undersampling discards lines of a fully sampled run, so it
ignores one effect of a real CS scan: with fewer RF pulses, the
magnetization evolves differently. The prospective integration below covers
that effect.

## Prospective integration

### Stage 1: sampling masks in the builders

Builders gain an optional `phase_sampling_mask` (or `acceleration`, `seed`,
`centre_lines` and `density_power`, from which the builder draws the mask
itself) and acquire only the selected points:

| Builder | Mask | Remarks |
| --- | --- | --- |
| `make_pulseq_flash` | `(ky,)`, optionally `(repetitions, ky)` for k-t | Line mask; limited gain in 2D. Acquire lines in ascending ky so phase-encoding steps stay small and TE_eff stays at the centre line. For IR, TI keeps pointing at the centre-line excitation. |
| `make_pulseq_bssfp`, Cartesian ME-bSSFP | `(ky, kz)` | Best case, because incoherence is 2D. Large jumps in the phase-encoding area cause eddy-current artefacts in real scanners (not simulated). Order the points along a radius or spiral. TR stays constant, so the steady state is preserved. |
| `make_pulseq_csi` | `(ky, kx)` points | One FID per point. For hyperpolarized (HP) 13C, every omitted encoding step saves an RF pulse. The spectral axis offers further sparsity: few metabolite peaks, so an l1 penalty along the spectrum. |
| `make_pulseq_epi` | across repetitions only | Single-shot EPI cannot skip lines randomly within one shot without uneven blips, which changes echo spacing and ghosting. CS therefore only makes sense as k-t across repetitions with interleaved, segmented shots that are shifted per frame. That needs multi-shot EPI. |
| `make_pulseq_press` | – | Single voxel; no spatial encoding. |

For HP sequences, the variable flip angle (VFA) schedule must be computed
from the number of excitations actually played
(`variable_flip_angle_schedule(num_excitations=N_acquired)`). Otherwise the last lines see too large or too small an
angle. The signal decay along the acquisition order is a k-space modulation
that CS has to absorb. Centre-first ordering puts that decay at the edge of
k-space.

The Pulseq definitions record the mask in a reproducible form:

- `SamplingMask` (index list per frame);
- `SamplingAcceleration`;
- `SamplingSeed`;
- `SamplingDensityPower`;
- `SamplingCentre`.

`LIN`/`PAR` labels keep their meaning as grid indices.

### Stage 2: acquisition models with a subset of the grid

`CartesianAcquisition.phase_indices` currently has to be a permutation of all
phase rows. The extension:

- allows a subset of rows and exposes `sampling_mask`;
- has `reshape_signal` fill the missing rows with zeros and return the mask
  alongside;
- has `infer_cartesian_acquisition` allow missing rows only when a
  `SamplingMask` definition says so. Without it, inference stays
  conservative and rejects incomplete grids, as it does now.

`CartesianAcquisitionVolumes` (missing kz/ky cells) and
`SpectroscopicAcquisition` (missing CSI points) get the same treatment. The
gradient-moment validation stays unchanged for every acquired line.

### Stage 3: reconstruction hook

- `reconstruct(..., method="zero_filled" | "cs", cs_options=...)` on the
  acquisition classes calls the module. The default stays the zero-filled
  FFT, so existing results do not change.
- Reconstruction Explorer:
  - a zero-filled/CS switch, weights and iterations;
  - the mask with its PSF sidelobe;
  - nRMSE against an optional fully sampled reference run of the same
    phantom.
- Notebook export: one cell with the mask parameters and the CS call.
- xarray/NPZ/HDF5 export: the mask, the zero-filled and CS images and the
  solver objective.

### Stage 4: GUI controls

An "Undersampling" section in the FLASH, 3D and CSI forms holds:

- acceleration, centre lines, density power and seed;
- a k-t toggle for repetitions;
- a mask preview with PSF sidelobe;
- the saved scan time.

Scan time and ETA come from the duration of the generated program, so they
reflect the shorter scan without further changes.

## Validation of the prospective integration

1. **Same sequence, same mask:** the retrospective result from a fully
   sampled run and the prospective undersampled run agree when T1, T2 → ∞
   and transients are switched off. This checks stage 1 and stage 2.
2. **Real dynamics:** with finite T1/T2 or HP depletion, the difference
   between retrospective and prospective runs measures the
   signal-evolution effect of the shorter scan. That is exactly the point
   that only a Bloch simulation can show.
3. **Metrics:** nRMSE, PSF sidelobe, and, with the receiver noise model,
   SNR per unit of scan time. That noise model also gives a starting point
   for choosing the weights (discrepancy principle on the noise level).

## Open points

- **Coil sensitivities:** the simulated receive profiles
  (`rx_sensitivity_maps`) are known, so SENSE+CS (an `F S` operator instead
  of `F`) can be added directly without estimating maps.
- **Phase-constrained or real-valued CS** for smooth image phase (see the
  FLASH observation above).
- **Spectral sparsity** for CSI/EPSI (l1 along the spectrum) and k-f
  methods.
- **Performance:** a 64³ volume with TV and wavelet takes about 10 s for 300
  iterations with numpy FFTs. Larger cases can move the same algorithm to
  `scipy.fft` with `workers`.
