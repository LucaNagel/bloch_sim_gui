"""Cartesian acquisition layout, reconstruction, and reference sequence builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .model import ADCEvent, GradientEvent, RFEvent, SequenceProgram


@dataclass(frozen=True)
class CartesianAcquisition:
    """Describe how a chronological ADC stream maps to a 2D Cartesian grid."""

    read_matrix: int
    phase_matrix: int
    fov_m: Tuple[float, float]
    dwell_s: float
    phase_indices: Optional[Tuple[int, ...]] = None
    readout_directions: Optional[Tuple[int, ...]] = None
    kx_offset_cells: float = 0.0
    ky_offset_cells: float = 0.0

    def __post_init__(self) -> None:
        read_matrix = _positive_integer(self.read_matrix, "read_matrix")
        phase_matrix = _positive_integer(self.phase_matrix, "phase_matrix")
        fov = tuple(float(value) for value in self.fov_m)
        if len(fov) != 2 or not np.all(np.isfinite(fov)) or min(fov) <= 0:
            raise ValueError("fov_m must contain two positive finite values")
        dwell = float(self.dwell_s)
        if not np.isfinite(dwell) or dwell <= 0:
            raise ValueError("dwell_s must be positive and finite")
        kx_offset = float(self.kx_offset_cells)
        ky_offset = float(self.ky_offset_cells)
        if not np.isfinite(kx_offset) or not np.isfinite(ky_offset):
            raise ValueError("Cartesian k-space offsets must be finite")

        phase_indices = (
            tuple(range(phase_matrix))
            if self.phase_indices is None
            else tuple(int(value) for value in self.phase_indices)
        )
        if len(phase_indices) != phase_matrix or sorted(phase_indices) != list(
            range(phase_matrix)
        ):
            raise ValueError("phase_indices must be a permutation of phase rows")

        directions = (
            tuple(1 for _ in range(phase_matrix))
            if self.readout_directions is None
            else tuple(int(value) for value in self.readout_directions)
        )
        if len(directions) != phase_matrix or any(
            value not in (-1, 1) for value in directions
        ):
            raise ValueError("readout_directions must contain one +1/-1 per row")

        object.__setattr__(self, "read_matrix", read_matrix)
        object.__setattr__(self, "phase_matrix", phase_matrix)
        object.__setattr__(self, "fov_m", fov)
        object.__setattr__(self, "dwell_s", dwell)
        object.__setattr__(self, "phase_indices", phase_indices)
        object.__setattr__(self, "readout_directions", directions)
        object.__setattr__(self, "kx_offset_cells", kx_offset)
        object.__setattr__(self, "ky_offset_cells", ky_offset)

    @classmethod
    def epi(
        cls,
        read_matrix: int,
        phase_matrix: int,
        fov_m: Tuple[float, float],
        dwell_s: float,
        *,
        phase_indices: Optional[Tuple[int, ...]] = None,
        first_readout_direction: int = 1,
    ) -> "CartesianAcquisition":
        """Create a Cartesian layout with alternating EPI readout directions."""
        if first_readout_direction not in (-1, 1):
            raise ValueError("first_readout_direction must be +1 or -1")
        directions = tuple(
            first_readout_direction * (-1 if line % 2 else 1)
            for line in range(int(phase_matrix))
        )
        return cls(
            read_matrix=read_matrix,
            phase_matrix=phase_matrix,
            fov_m=fov_m,
            dwell_s=dwell_s,
            phase_indices=phase_indices,
            readout_directions=directions,
        )

    @property
    def num_samples(self) -> int:
        return self.read_matrix * self.phase_matrix

    @property
    def sampling_bandwidth_hz(self) -> float:
        """Complex receiver sampling bandwidth (spectral width)."""
        return 1.0 / self.dwell_s

    @property
    def pixel_bandwidth_hz(self) -> float:
        """Nominal receiver bandwidth per readout pixel."""
        return self.sampling_bandwidth_hz / self.read_matrix

    @property
    def kx_cyc_per_m(self) -> np.ndarray:
        return (
            np.arange(self.read_matrix, dtype=float)
            - self.read_matrix // 2
            + self.kx_offset_cells
        ) / self.fov_m[0]

    @property
    def ky_cyc_per_m(self) -> np.ndarray:
        return (
            np.arange(self.phase_matrix, dtype=float)
            - self.phase_matrix // 2
            + self.ky_offset_cells
        ) / self.fov_m[1]

    def reshape_signal(self, signal: np.ndarray) -> np.ndarray:
        """Map chronological ADC data to ``(..., phase, read)`` k-space."""
        values = np.asarray(signal)
        if values.ndim not in (1, 2) or values.shape[-1] != self.num_samples:
            raise ValueError(
                f"signal must end with {self.num_samples} chronological samples"
            )
        raw = values.reshape(values.shape[:-1] + (self.phase_matrix, self.read_matrix))
        grid = np.empty_like(raw)
        for acquired_line, phase_index in enumerate(self.phase_indices):
            line = raw[..., acquired_line, :]
            if self.readout_directions[acquired_line] < 0:
                line = line[..., ::-1]
            grid[..., phase_index, :] = line
        return grid

    def reconstruct(
        self,
        signal: np.ndarray,
        *,
        norm: Optional[str] = None,
        coil_combine: Optional[str] = None,
        voxel_centered: bool = True,
    ) -> np.ndarray:
        """Reconstruct Cartesian data with a centred inverse 2D FFT."""
        kspace = self.reshape_signal(signal)
        if voxel_centered:
            dx = self.fov_m[0] / self.read_matrix
            dy = self.fov_m[1] / self.phase_matrix
            centre_phase = np.exp(
                2j
                * np.pi
                * (
                    self.ky_cyc_per_m[:, None] * dy / 2
                    + self.kx_cyc_per_m[None, :] * dx / 2
                )
            )
            kspace = kspace * centre_phase
        axes = (-2, -1)
        image = np.fft.fftshift(
            np.fft.ifft2(np.fft.ifftshift(kspace, axes=axes), axes=axes, norm=norm),
            axes=axes,
        )
        if coil_combine is None:
            return image
        if image.ndim != 3:
            raise ValueError("coil combination requires signal shape (coil, adc)")
        if coil_combine == "rss":
            return np.sqrt(np.sum(np.abs(image) ** 2, axis=0))
        if coil_combine == "sum":
            return np.sum(image, axis=0)
        raise ValueError("coil_combine must be None, 'rss', or 'sum'")

    def validate_adc_times(self, adc_times_s: np.ndarray) -> None:
        """Validate sample count and within-line dwell spacing."""
        times = np.asarray(adc_times_s, dtype=float)
        if times.shape != (self.num_samples,) or not np.all(np.isfinite(times)):
            raise ValueError("ADC times do not match the Cartesian acquisition")
        lines = times.reshape(self.phase_matrix, self.read_matrix)
        if self.read_matrix > 1 and not np.allclose(
            np.diff(lines, axis=1), self.dwell_s, rtol=0.0, atol=1e-12
        ):
            raise ValueError("ADC dwell spacing does not match the acquisition")

    def validate_gradient_moments(self, moments_cyc_per_m: np.ndarray) -> None:
        """Validate that ADC gradient moments lie on the described 2D grid."""
        moments = np.asarray(moments_cyc_per_m, dtype=float)
        if moments.shape != (self.num_samples, 3):
            raise ValueError("gradient moments must have shape (num_samples, 3)")
        raw = moments.reshape(self.phase_matrix, self.read_matrix, 3)
        for acquired_line, phase_index in enumerate(self.phase_indices):
            expected_x = self.kx_cyc_per_m
            if self.readout_directions[acquired_line] < 0:
                expected_x = expected_x[::-1]
            expected_y = self.ky_cyc_per_m[phase_index]
            x_tolerance = max(1e-9, 1e-3 / self.fov_m[0])
            y_tolerance = max(1e-9, 1e-3 / self.fov_m[1])
            if not np.allclose(
                raw[acquired_line, :, 0], expected_x, rtol=0.0, atol=x_tolerance
            ):
                raise ValueError("readout gradient moments do not match the grid")
            if not np.allclose(
                raw[acquired_line, :, 1], expected_y, rtol=0.0, atol=y_tolerance
            ):
                raise ValueError("phase gradient moments do not match the grid")

    def to_metadata(self) -> dict:
        return {
            "type": "cartesian_2d",
            "read_matrix": self.read_matrix,
            "phase_matrix": self.phase_matrix,
            "fov_m": self.fov_m,
            "dwell_s": self.dwell_s,
            "sampling_bandwidth_hz": self.sampling_bandwidth_hz,
            "pixel_bandwidth_hz": self.pixel_bandwidth_hz,
            "phase_indices": self.phase_indices,
            "readout_directions": self.readout_directions,
            "kx_offset_cells": self.kx_offset_cells,
            "ky_offset_cells": self.ky_offset_cells,
        }


def infer_cartesian_acquisition(
    program: SequenceProgram,
    *,
    compiled=None,
) -> CartesianAcquisition:
    """Infer one regular 2D Cartesian acquisition from a sequence program.

    The conservative inference currently accepts one chronological ADC event
    per phase line, x readout, y phase encoding, and an explicit Pulseq FOV
    definition. Multi-slice, repeated, segmented, or non-Cartesian streams are
    rejected instead of being reshaped ambiguously.
    """
    from .compiler import SequenceCompiler

    adc_events = program.adc_events
    if not adc_events:
        raise ValueError("sequence contains no ADC events")
    read_matrix = adc_events[0].num_samples
    if read_matrix < 2:
        raise ValueError("Cartesian inference requires at least two samples per line")
    if any(event.num_samples != read_matrix for event in adc_events):
        raise ValueError("ADC events do not have a common read matrix")
    dwell_s = adc_events[0].dwell_s
    if any(
        not np.isclose(event.dwell_s, dwell_s, rtol=0.0, atol=1e-15)
        for event in adc_events
    ):
        raise ValueError("ADC events do not have a common dwell time")
    if any(
        current.start_s <= previous.start_s
        for previous, current in zip(adc_events, adc_events[1:])
    ):
        raise ValueError("ADC events are not strictly chronological")

    definitions = dict(program.metadata.get("definitions", {}))
    fov_value = next(
        (value for key, value in definitions.items() if str(key).lower() == "fov"),
        None,
    )
    if fov_value is None:
        raise ValueError("Pulseq sequence has no FOV definition")
    fov = np.asarray(fov_value, dtype=float).reshape(-1)
    if fov.size < 2 or not np.all(np.isfinite(fov[:2])) or np.any(fov[:2] <= 0):
        raise ValueError("Pulseq FOV definition does not contain valid x/y values")
    fov_x, fov_y = (float(fov[0]), float(fov[1]))

    compiled = SequenceCompiler().compile(program) if compiled is None else compiled
    expected_times = np.concatenate([event.sample_times_s for event in adc_events])
    if compiled.adc_times_s.shape != expected_times.shape or not np.allclose(
        compiled.adc_times_s, expected_times, rtol=0.0, atol=1e-12
    ):
        raise ValueError("ADC samples cannot be grouped into chronological lines")

    phase_matrix = len(adc_events)
    moments = np.asarray(compiled.adc_gradient_moment_cyc_per_m, dtype=float)
    if moments.shape != (phase_matrix * read_matrix, 3):
        raise ValueError("compiled ADC gradient moments have an invalid shape")
    raw = moments.reshape(phase_matrix, read_matrix, 3)
    tolerance_x = max(1e-9, 1e-3 / fov_x)
    tolerance_y = max(1e-9, 1e-3 / fov_y)

    delta_x = np.diff(raw[:, :, 0], axis=1)
    mean_delta_x = np.mean(delta_x, axis=1)
    if np.any(np.abs(mean_delta_x) <= tolerance_x):
        raise ValueError("ADC events do not contain an x readout gradient")
    directions = np.where(mean_delta_x > 0, 1, -1)
    expected_delta_x = directions[:, None] / fov_x
    if not np.allclose(delta_x, expected_delta_x, rtol=0.0, atol=tolerance_x):
        raise ValueError("ADC readout samples are not on a regular x grid")
    if not np.allclose(np.diff(raw[:, :, 1], axis=1), 0.0, rtol=0.0, atol=tolerance_y):
        raise ValueError("phase gradient changes during an ADC line")
    z_scale = max(1.0, float(np.max(np.abs(raw[:, :, 2]))))
    if not np.allclose(
        np.diff(raw[:, :, 2], axis=1), 0.0, rtol=0.0, atol=1e-9 * z_scale
    ):
        raise ValueError("slice gradient changes during an ADC line")

    ordered_x = np.stack(
        [
            raw[line, :, 0] if directions[line] > 0 else raw[line, ::-1, 0]
            for line in range(phase_matrix)
        ]
    )
    common_x = np.mean(ordered_x, axis=0)
    if not np.allclose(ordered_x, common_x, rtol=0.0, atol=tolerance_x):
        raise ValueError("EPI readout lines do not share one Cartesian kx grid")

    line_y = np.mean(raw[:, :, 1], axis=1)
    phase_order = np.argsort(line_y)
    sorted_y = line_y[phase_order]
    if phase_matrix > 1 and not np.allclose(
        np.diff(sorted_y), 1.0 / fov_y, rtol=0.0, atol=tolerance_y
    ):
        raise ValueError("ADC lines do not form one regular Cartesian ky grid")
    phase_indices = np.empty(phase_matrix, dtype=int)
    phase_indices[phase_order] = np.arange(phase_matrix)

    base_x = np.arange(read_matrix, dtype=float) - read_matrix // 2
    kx_offset = float(np.mean(common_x * fov_x - base_x))
    if not np.allclose(
        common_x * fov_x,
        base_x + kx_offset,
        rtol=0.0,
        atol=tolerance_x * fov_x,
    ):
        raise ValueError("kx coordinates cannot be represented by one Cartesian grid")
    rounded_kx_offset = round(2.0 * kx_offset) / 2.0
    if abs(kx_offset - rounded_kx_offset) <= tolerance_x * fov_x:
        kx_offset = rounded_kx_offset
    base_y = np.arange(phase_matrix, dtype=float) - phase_matrix // 2
    ky_offset = float(np.mean(sorted_y * fov_y - base_y))
    if not np.allclose(
        sorted_y * fov_y,
        base_y + ky_offset,
        rtol=0.0,
        atol=tolerance_y * fov_y,
    ):
        raise ValueError("ky coordinates cannot be represented by one Cartesian grid")
    rounded_ky_offset = round(2.0 * ky_offset) / 2.0
    if abs(ky_offset - rounded_ky_offset) <= tolerance_y * fov_y:
        ky_offset = rounded_ky_offset

    acquisition = CartesianAcquisition(
        read_matrix=read_matrix,
        phase_matrix=phase_matrix,
        fov_m=(fov_x, fov_y),
        dwell_s=dwell_s,
        phase_indices=tuple(int(value) for value in phase_indices),
        readout_directions=tuple(int(value) for value in directions),
        kx_offset_cells=kx_offset,
        ky_offset_cells=ky_offset,
    )
    acquisition.validate_adc_times(compiled.adc_times_s)
    acquisition.validate_gradient_moments(moments)
    return acquisition


def make_cartesian_epi(
    acquisition: CartesianAcquisition,
    *,
    flip_angle_deg: float = 90.0,
    rf_duration_s: float = 1e-3,
    prephaser_duration_s: float = 1e-3,
    blip_duration_s: float = 100e-6,
    delay_after_prephaser_s: float = 0.0,
    tail_s: float = 0.0,
) -> SequenceProgram:
    """Build a non-slice-selective single-shot Cartesian EPI program."""
    for name, value, allow_zero in (
        ("rf_duration_s", rf_duration_s, False),
        ("prephaser_duration_s", prephaser_duration_s, False),
        ("blip_duration_s", blip_duration_s, False),
        ("delay_after_prephaser_s", delay_after_prephaser_s, True),
        ("tail_s", tail_s, True),
    ):
        if not np.isfinite(value) or value < 0 or (not allow_zero and value == 0):
            raise ValueError(f"{name} has an invalid duration")
    if not np.isfinite(flip_angle_deg):
        raise ValueError("flip_angle_deg must be finite")

    nx = acquisition.read_matrix
    ny = acquisition.phase_matrix
    fov_x, fov_y = acquisition.fov_m
    dwell = acquisition.dwell_s
    readout_duration = nx * dwell
    events = []

    rf_hz = np.deg2rad(flip_angle_deg) / (2 * np.pi * rf_duration_s)
    events.append(RFEvent(0.0, np.asarray([rf_hz]), rf_duration_s))

    first_direction = acquisition.readout_directions[0]
    first_kx = (
        acquisition.kx_cyc_per_m[0]
        if first_direction > 0
        else acquisition.kx_cyc_per_m[-1]
    )
    current_x = first_kx - first_direction * 0.5 / fov_x
    first_phase_index = acquisition.phase_indices[0]
    current_y = acquisition.ky_cyc_per_m[first_phase_index]
    prephaser_start = rf_duration_s
    if current_x != 0:
        events.append(
            GradientEvent(
                "x",
                prephaser_start,
                np.asarray([current_x / prephaser_duration_s]),
                prephaser_duration_s,
            )
        )
    if current_y != 0:
        events.append(
            GradientEvent(
                "y",
                prephaser_start,
                np.asarray([current_y / prephaser_duration_s]),
                prephaser_duration_s,
            )
        )

    time_s = prephaser_start + prephaser_duration_s + delay_after_prephaser_s
    for line in range(ny):
        direction = acquisition.readout_directions[line]
        read_gradient = direction / (fov_x * dwell)
        events.append(
            GradientEvent("x", time_s, np.asarray([read_gradient]), readout_duration)
        )
        events.append(
            ADCEvent(
                start_s=time_s + dwell / 2,
                num_samples=nx,
                dwell_s=dwell,
            )
        )
        current_x += direction * nx / fov_x
        time_s += readout_duration

        if line == ny - 1:
            continue
        next_direction = acquisition.readout_directions[line + 1]
        next_first_kx = (
            acquisition.kx_cyc_per_m[0]
            if next_direction > 0
            else acquisition.kx_cyc_per_m[-1]
        )
        desired_x = next_first_kx - next_direction * 0.5 / fov_x
        next_phase_index = acquisition.phase_indices[line + 1]
        desired_y = acquisition.ky_cyc_per_m[next_phase_index]
        x_area = desired_x - current_x
        y_area = desired_y - current_y
        if x_area != 0:
            events.append(
                GradientEvent(
                    "x",
                    time_s,
                    np.asarray([x_area / blip_duration_s]),
                    blip_duration_s,
                )
            )
        if y_area != 0:
            events.append(
                GradientEvent(
                    "y",
                    time_s,
                    np.asarray([y_area / blip_duration_s]),
                    blip_duration_s,
                )
            )
        current_x = desired_x
        current_y = desired_y
        time_s += blip_duration_s

    return SequenceProgram(
        events=tuple(events),
        duration_s=time_s + tail_s,
        source="internal-cartesian-epi",
        metadata={"acquisition": acquisition.to_metadata()},
    )


def _positive_integer(value: int, name: str) -> int:
    if int(value) != value or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)
