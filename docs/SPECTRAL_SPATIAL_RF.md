# Spectral-spatial RF pulse design

The Bloch Simulator contains an independent native-Python implementation of a
flyback spectral-spatial RF pulse design method based on the work of Adam B.
Kerr and Peder E. Z. Larson. It does not start MATLAB, require MATLAB, or claim
numerical equivalence to the reference toolbox.

The reference implementation is the
[LarsonLab Spectral-Spatial RF Pulse Design toolbox](https://github.com/LarsonLab/Spectral-Spatial-RF-Pulse-Design).

## Supported design path

The Python designer currently performs the following steps:

1. It converts the requested spatial time-bandwidth product, slice or slab
   thickness, and RF subpulse duration into a physical slice gradient using
   the selected nucleus-specific gyromagnetic ratio.
2. It creates a slew-limited positive excitation lobe and a separately
   optimized, shortest raster-compatible negative rewinder with equal and
   opposite area.
3. It searches raster-compatible spectral sampling periods and rejects
   pass/pass or explicit pass/stop alias conflicts.
4. It designs an equiripple spatial subpulse. Spatial pass-band ripple,
   stop-band ripple, and transition width all affect the resulting waveform.
5. It designs complex multiband spectral weights. Linear phase uses a weighted
   complex least-squares FIR design. Minimum phase uses causal per-band
   spectral factorization followed by a complex center-frequency fit.
6. It scales the complex RF waveform to the requested reference flip angle and
   checks gradient, slew, maximum B1, and total-duration limits.
7. It records the complete input prescription, selected alias period,
   constraints, waveform extrema, residual gradient moment, and sampled FIR
   errors in the pulse metadata.

RF samples are stored as nutation frequency in Hz. Gradient samples are stored
as Hz/m. The physical fields are

$$
B_1 = \frac{f_1}{\gamma}, \qquad G = \frac{g}{\gamma},
$$

where $\gamma$ is in Hz/T, so $B_1$ is in T and $G$ is in T/m. Time is stored
in seconds. Pulseq export converts and checks these quantities against the
scanner system on its own RF and gradient rasters.

## Comparison with the reference toolbox

| Capability | Larson/Kerr MATLAB reference | Native Python implementation |
| --- | --- | --- |
| Flyback SPSP | Whole- and half-pulse variants | Whole-pulse flyback |
| Bipolar/echo-planar SPSP | Whole, half, and opposite-null variants | API-reserved, explicitly not implemented |
| Spectral alias handling | Candidate search with optional frequency offset | Raster-period search at the fixed RF-carrier reference; explicit conflict report |
| Spectral FIR | Convex optimization, spectral factorization, linear/minimum/maximum phase | Weighted complex linear phase and causal minimum-phase per-band factorization |
| Spatial RF | `rf_tools` SLR families | Independent equiripple small-tip subpulse |
| Large-tip correction | Two-dimensional SLR and spectral correction options | Bloch validation only; no 2D SLR correction |
| Gradient design | `grad_ss`, bridge control, equal/unequal lobes | Slew-limited excitation lobe and independent minimum-time rewinder |
| VERSE and B1 optimization | VERSE and B1-limited VERSE | Hard B1 constraint; no VERSE optimization |
| Complex band targets | Phase-aware toolbox variants | Relative complex amplitude and phase targets |
| Validation | Toolbox response functions and examples | Native 2D Bloch simulation, Pulseq round trip, hardware and regression tests |
| MATLAB reference vectors | Native result | Not yet available in this project; numerical equivalence is not claimed |

## Validation and interpretation

The GUI displays a two-dimensional relaxation-free Bloch simulation over
position and frequency. It also reports sampled pass-band error, stop-band
flip, duration, peak B1, peak gradient, and peak slew. Tests cover waveform
balance, hardware limits, alias conflicts, phase selection, multiband complex
targets, transition-width sensitivity, flip-angle scaling, and Pulseq
export/import. The profile and schedule preview use the same rotating-frame
RF, off-resonance, and gradient sign convention as the complete event-based
Bloch simulator; complex band phases are mapped to the requested $M_{xy}$
phase under that convention.

The requested ripple values weight the finite-length FIR designs. They are not
silently treated as guarantees when the chosen subpulse count or spatial TBW is
too small. Stored metrics include achieved spatial and spectral pass-/stop-band
errors and explicit `*_specifications_met` flags; the GUI reports when the
sampled specifications are not met. These are discrete design-grid
measurements, not a formal continuous-domain guarantee. Scanner deployment
requires independent sequence, SAR, RF-amplifier, gradient, and safety
validation.

## Pulse library and ordered schedule

Sequence Mode stores spectral-spatial state in two parts:

1. `SpectralSpatialPulseLibrary` contains reusable `SpectralSpatialPulse`
   waveforms with stable string IDs.
2. `SpectralSpatialSchedule` contains ordered
   `SpectralSpatialScheduleEntry` references to those IDs. Each entry also
   stores its display name, target metabolite, enabled state, scaling mode, and
   number of immediately adjacent repetitions.

An entry scales RF in either of two ways:

- **Flip angle** sets the requested flip angle of the strongest designed pass
  band. The RF scale is this value divided by the reference flip angle.
- **RF amplitude scale** directly multiplies the stored complex RF waveform.
  For reference flip angle $\alpha_{\mathrm{ref}}$, the nominal flip angle is

$$
\alpha = s_{\mathrm{RF}}\alpha_{\mathrm{ref}}.
$$

Gradient amplitude is unchanged by flip-angle scaling. It changes only when
the reference spatial thickness is adapted to the sequence slice or slab.
Every expanded step is rechecked for nucleus, B1, gradient amplitude, slew
rate, duration, and scale.

The editor provides **New**, **Duplicate**, **Edit**, and **Delete** for
library pulses. The schedule table provides pulse selection, names, scaling,
repeat count, enabled state, duplication, deletion, and up/down ordering.
**Preview pulse schedule** opens the schedule preview.

Projects containing only `sequence_spectral_spatial_pulse` are migrated in
memory to `legacy-single-pulse` and `legacy-single-entry`. New saves retain the
historical singleton compatibility view and store the complete library and
schedule.

## Frequency references and phantom peaks

The preview calculates every compatible phantom peak from its absolute ppm
position, rather than assuming the first preset is the RF reference:

$$
f_{\mathrm{sequence}} = \gamma B_0
(\delta_{\mathrm{absolute}}-\delta_{\mathrm{sequence}})10^{-6},
$$

$$
f_{\mathrm{carrier}} = f_{\mathrm{sequence}}-f_{\mathrm{RF}}.
$$

Its folded position in one periodic SPSP interval is

$$
f_{\mathrm{folded}} =
\left(f_{\mathrm{carrier}}+\frac{F_s}{2}\right)\bmod F_s-
\frac{F_s}{2},
$$

where $F_s$ is the alias spacing. The UI shows absolute ppm,
sequence-relative Hz, carrier-relative Hz, and folded Hz, and identifies
whether peaks come from the current phantom or presets. It warns about a
missing target, a stored/current reference mismatch, a target outside all pass
bands, and unmet sampled filter specifications.

## FLASH integration and metadata

FLASH advances the schedule at complete-image frame level by default. All
slices and phase-encoding lines in one image use the same pulse and scale. The
FLASH repetition count is the number of frames; cyclic selection maps those
frames deterministically onto the expanded schedule.

An explicit per-TR mode is available for advanced experiments. It mixes
metabolite contrasts across k-space and is not interpreted as separate images.
Both modes write schedule IDs, names, scales, flip angles, durations, frame
selections, and warnings to Pulseq definitions. The schedule index is stored
in the Pulseq `SET` label and recovered with the ADC labels on import.
Simulation results retain the sequence definitions and ADC label values.

Each pulse has its own RF centre, slice rephaser, TE delay, and TR delay. A
pulse that makes the requested TE or TR impossible produces a clear error
naming the affected schedule entry.

## Fast schedule preview

The preview evaluates complete complex RF and gradient waveforms on a reduced
spatial grid at the discrete frequencies of the current phantom, or at clearly
identified presets when no compatible phantom is selected. It includes
schedule order and scaling, coherent complex magnetization, inter-pulse
off-resonance phase, optional T1/T2* relaxation, and the selected ideal/no
spoiling assumption. It provides step navigation and playback, waveforms, an
excitation map, metabolite profiles, a step-by-metabolite $|M_{xy}|$ table,
cumulative $M_z$ and $|M_{xy}|$, and common fixed plot scales.

The preview omits readout and encoding gradients, spatial B0 distributions,
motion, exchange, receive sensitivity, k-space sampling, and reconstruction.
It is a rapid excitation/saturation diagnostic, not a replacement for full
sequence simulation.

## Deliberately unsupported features

- echo-planar or bipolar SPSP trajectories;
- half-pulse and opposite-null variants;
- maximum-phase spectral filters;
- MATLAB Optimization Toolbox-equivalent convex programs;
- large-tip two-dimensional SLR spectral correction;
- VERSE and B1-VERSE optimization;
- automatic carrier-offset selection;
- numerical regression against exported MATLAB reference waveforms.

Requests for these modes raise an explicit error rather than silently falling
back to the flyback implementation.

## Attribution and citation

The GUI states: "Design method based on Larson/Kerr spectral-spatial RF pulse
design." It links to the reference toolbox and does not imply endorsement by
LarsonLab, Stanford University, or the University of California.

For scientific work, cite the application-specific methods and at least the
following design references:

- Kerr, A. B., Larson, P. E. Z., Lustig, M., Cunningham, C. H., Chen, A. P.,
  Vigneron, D. B., and Pauly, J. M. "Multiband spectral-spatial design for
  high-field and hyperpolarized C-13 applications." Proceedings of the 16th
  Annual Meeting of ISMRM, Toronto, 2008, abstract 226.
- Larson, P. E. Z., Kerr, A. B., Chen, A. P., et al. "Multiband excitation
  pulses for hyperpolarized 13C dynamic chemical-shift imaging." *Journal of
  Magnetic Resonance* 194(1), 121-127 (2008).
  [doi:10.1016/j.jmr.2008.06.010](https://doi.org/10.1016/j.jmr.2008.06.010)

See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for license and source
provenance details.
