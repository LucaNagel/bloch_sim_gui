"""Single-shot spiral readout design and its sampling PSF.

:func:`design_spiral` designs the same centre-out Archimedean spiral that
:func:`~.pulseq_builders.make_pulseq_spiral` plays, from FOV, matrix,
revolutions and receiver bandwidth, within the gradient amplitude and slew
limits of a scanner for a given nucleus. When the limits are exceeded the
dwell time is lengthened, as in the sequence builder.

:meth:`SpiralDesign.sampling_psf` grids the samples of a point source with
the Pipe-Menon weights of the spiral reconstruction. Optional T2* decay and
off-resonance during the readout weight the samples with
``exp(-t/T2*) exp(2 pi i df t)``, which shows the blurring typical of long
spiral readouts without a Bloch simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from .psf import evaluate_sample_image, profile_metrics
from .scanner import ScannerParameters

LEGACY_SPIRAL_LIMITS = {
    "max_grad": 32,
    "grad_unit": "mT/m",
    "max_slew": 130,
    "slew_unit": "T/m/s",
    "grad_raster_time": 10e-6,
    "rf_ringdown_time": 30e-6,
    "rf_dead_time": 100e-6,
    "adc_dead_time": 20e-6,
}


@dataclass(frozen=True)
class SpiralPSF:
    """Sampling PSF of a spiral design for a point at the FOV centre.

    ``image`` is sampled at ``offsets_m`` (``(y, x)``, array order) and
    normalized to the peak of the ideal PSF of uniform Cartesian weighting;
    ``profiles`` maps ``"x"``/``"y"`` to ``(offsets, values)`` through the
    peak.
    """

    offsets_m: Tuple[np.ndarray, np.ndarray]
    image: np.ndarray
    profiles: Mapping[str, Tuple[np.ndarray, np.ndarray]]
    fwhm_voxels: Mapping[str, float]
    peak_offset_voxels: Mapping[str, float]
    peak_sidelobe_ratio: float
    peak: float


@dataclass(frozen=True)
class SpiralDesign:
    """A hardware-limited single-shot spiral readout.

    Gradients are in Hz/m on the gradient raster as played by the sequence;
    ``k_cyc_per_m`` holds the in-plane k-space position (x, y) of every ADC
    sample at ``sample_times_s`` after the start of the readout.
    """

    fov_m: Tuple[float, float]
    matrix: Tuple[int, int]
    turns: float
    nucleus: str
    gamma_hz_per_t: float
    requested_bandwidth_hz: float
    dwell_s: float
    gradient_raster_s: float
    max_grad_mtm: float
    max_slew_tms: float
    gradient_hz_per_m: np.ndarray
    k_cyc_per_m: np.ndarray
    sample_times_s: np.ndarray

    @property
    def sampling_bandwidth_hz(self) -> float:
        return 1.0 / self.dwell_s

    @property
    def bandwidth_limited(self) -> bool:
        """True when the hardware limits lowered the requested bandwidth."""
        return self.sampling_bandwidth_hz < self.requested_bandwidth_hz * (1 - 1e-9)

    @property
    def readout_duration_s(self) -> float:
        return self.dwell_s * self.k_cyc_per_m.shape[0]

    @property
    def gradient_mtm(self) -> np.ndarray:
        """Gradient waveforms (raster, 2) in mT/m for the design nucleus."""
        return 1e3 * self.gradient_hz_per_m / self.gamma_hz_per_t

    @property
    def slew_tms(self) -> np.ndarray:
        """Slew rate (raster + 1, 2) in T/m/s including the ramps to zero."""
        gradient = self.gradient_hz_per_m / self.gamma_hz_per_t
        padded = np.vstack((np.zeros((1, 2)), gradient, np.zeros((1, 2))))
        return np.diff(padded, axis=0) / self.gradient_raster_s

    @property
    def peak_gradient_mtm(self) -> float:
        return float(np.max(np.abs(self.gradient_mtm)))

    @property
    def peak_slew_tms(self) -> float:
        return float(np.max(np.abs(self.slew_tms)))

    @property
    def gradient_times_s(self) -> np.ndarray:
        return (np.arange(self.gradient_hz_per_m.shape[0]) + 0.5) * (
            self.gradient_raster_s
        )

    @property
    def turn_spacing(self) -> float:
        """Radial distance between neighbouring turns in units of 1/FOV.

        The Nyquist criterion for an alias-free FOV requires at most 1, i.e.
        ``turns >= matrix / 2``. The spiral speeds up radially at its start
        and end (smoothstep), so the spacing peaks in the middle; this is
        the largest spacing.
        """
        fov = np.asarray(self.fov_m)
        k_max = np.asarray(self.matrix) / (2.0 * fov)
        return float(np.max(k_max * fov) / self.turns)

    @property
    def alias_free_fov_m(self) -> Tuple[float, float]:
        factor = max(1.0, self.turn_spacing)
        return tuple(float(value / factor) for value in self.fov_m)

    def sampling_psf(
        self,
        *,
        t2_star_s: Optional[float] = None,
        off_resonance_hz: float = 0.0,
        field_of_views: float = 1.0,
        oversampling: int = 2,
        profile_oversampling: int = 16,
    ) -> SpiralPSF:
        """PSF of the gridding reconstruction for a centred point source.

        ``field_of_views`` sets the extent of the image in FOVs, so that
        aliasing rings of an undersampled spiral appear when it exceeds 1.
        """
        from .nufft import NUFFTOperator

        k = self.k_cyc_per_m
        operator = NUFFTOperator(k, tuple(self.matrix), tuple(self.fov_m))
        weights = operator.density_compensation()
        signal = np.ones(k.shape[0], dtype=complex)
        if t2_star_s is not None and t2_star_s > 0:
            signal = signal * np.exp(-self.sample_times_s / float(t2_star_s))
        if off_resonance_hz:
            signal = signal * np.exp(
                2j * np.pi * float(off_resonance_hz) * self.sample_times_s
            )
        weighted = weights * signal
        # Array order (y, x) as for images.
        k_yx = k[:, ::-1]
        fov_yx = np.asarray(self.fov_m)[::-1]
        matrix_yx = np.asarray(self.matrix)[::-1]
        voxel_yx = fov_yx / matrix_yx
        scale = 1.0 / float(np.prod(self.matrix))
        offsets = tuple(
            (
                np.arange(int(round(count * field_of_views * oversampling)))
                - int(round(count * field_of_views * oversampling)) // 2
            )
            * (size / oversampling)
            for count, size in zip(matrix_yx, voxel_yx)
        )
        image = scale * evaluate_sample_image(weighted, k_yx, offsets)
        peak_index = np.unravel_index(int(np.argmax(np.abs(image))), image.shape)
        profiles = {}
        widths = {}
        shifts = {}
        for axis, role in enumerate(("y", "x")):
            count = int(matrix_yx[axis])
            line = (
                np.arange(count * profile_oversampling)
                - count * profile_oversampling // 2
            ) * (voxel_yx[axis] / profile_oversampling)
            coordinates = [
                np.atleast_1d(offsets[other][peak_index[other]]) for other in range(2)
            ]
            coordinates[axis] = line
            values = scale * evaluate_sample_image(weighted, k_yx, coordinates)
            values = values.reshape(-1)
            profiles[role] = (line, values)
            width, shift, _ = profile_metrics(line, values)
            widths[role] = width / voxel_yx[axis]
            shifts[role] = shift / voxel_yx[axis]
        magnitude = np.abs(image)
        peak = float(magnitude.max())
        # Side lobes: everything farther than two voxels from the peak.
        distance = np.sqrt(
            sum(
                (
                    (offsets[axis] - offsets[axis][peak_index[axis]]) / voxel_yx[axis]
                ).reshape([-1 if other == axis else 1 for other in range(2)])
                ** 2
                for axis in range(2)
            )
        )
        outside = magnitude[distance > 2.0]
        return SpiralPSF(
            offsets_m=offsets,
            image=image,
            profiles=profiles,
            fwhm_voxels=widths,
            peak_offset_voxels=shifts,
            peak_sidelobe_ratio=float(outside.max() / peak) if outside.size else 0.0,
            peak=peak,
        )

    def to_dict(self) -> dict:
        """Trajectory and waveforms for export (SI units)."""
        return {
            "fov_m": np.asarray(self.fov_m),
            "matrix": np.asarray(self.matrix),
            "turns": self.turns,
            "nucleus": self.nucleus,
            "gamma_hz_per_t": self.gamma_hz_per_t,
            "dwell_s": self.dwell_s,
            "gradient_raster_s": self.gradient_raster_s,
            "gradient_hz_per_m": self.gradient_hz_per_m,
            "gradient_mt_per_m": self.gradient_mtm,
            "k_cyc_per_m": self.k_cyc_per_m,
            "sample_times_s": self.sample_times_s,
        }


def _system(pp, scanner_parameters, gamma_hz_per_t):
    if scanner_parameters is None:
        kwargs = dict(LEGACY_SPIRAL_LIMITS)
    else:
        kwargs = ScannerParameters.from_mapping(scanner_parameters).to_pypulseq_kwargs()
    return pp.Opts(**kwargs, gamma=gamma_hz_per_t)


def design_spiral(
    *,
    fov_m: Sequence[float],
    matrix: Sequence[int],
    sampling_bandwidth_hz: float,
    spiral_turns: Optional[float] = None,
    nucleus: str = "H1",
    scanner_parameters: ScannerParameters | Mapping[str, float] | None = None,
) -> SpiralDesign:
    """Design the spiral readout of :func:`make_pulseq_spiral`.

    The gradient limits apply to the physical gradient of ``nucleus``:
    a 13C spiral needs about four times the gradient of a 1H spiral with
    the same k-space trajectory.
    """
    from .pulseq_builders import NUCLEUS_GAMMA_HZ_PER_T, _pypulseq, _spiral_waveforms

    if nucleus not in NUCLEUS_GAMMA_HZ_PER_T:
        raise ValueError(f"unknown nucleus: {nucleus}")
    fov = tuple(float(value) for value in fov_m)
    size = tuple(int(value) for value in matrix)
    if len(fov) != 2 or len(size) != 2:
        raise ValueError("fov_m and matrix must contain x and y values")
    if min(fov) <= 0 or not np.all(np.isfinite(fov)) or min(size) < 2:
        raise ValueError("the FOV must be positive and the matrix at least 2")
    bandwidth = float(sampling_bandwidth_hz)
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("sampling_bandwidth_hz must be positive and finite")
    turns = max(1.0, min(size) / 2.0) if spiral_turns is None else float(spiral_turns)
    if not np.isfinite(turns) or turns <= 0:
        raise ValueError("spiral_turns must be positive and finite")
    gamma = float(NUCLEUS_GAMMA_HZ_PER_T[nucleus])
    pp = _pypulseq()
    system = _system(pp, scanner_parameters, gamma)
    gx, gy, dwell = _spiral_waveforms(
        matrix=size,
        fov_m=fov,
        requested_dwell_s=1.0 / bandwidth,
        turns=turns,
        system=system,
    )
    raster = float(system.grad_raster_time)
    gradient = np.column_stack((gx, gy))
    # k at the raster edges is exact for the piecewise-constant design; ADC
    # samples sit at the centres of their dwell intervals.
    edges = np.vstack((np.zeros((1, 2)), np.cumsum(gradient, axis=0) * raster))
    edge_times = np.arange(edges.shape[0]) * raster
    samples = size[0] * size[1]
    times = (np.arange(samples) + 0.5) * dwell
    k = np.column_stack(
        [np.interp(times, edge_times, edges[:, axis]) for axis in range(2)]
    )
    return SpiralDesign(
        fov_m=fov,
        matrix=size,
        turns=turns,
        nucleus=nucleus,
        gamma_hz_per_t=gamma,
        requested_bandwidth_hz=bandwidth,
        dwell_s=float(dwell),
        gradient_raster_s=raster,
        max_grad_mtm=float(system.max_grad) / gamma * 1e3,
        max_slew_tms=float(system.max_slew) / gamma,
        gradient_hz_per_m=gradient,
        k_cyc_per_m=k,
        sample_times_s=times,
    )
