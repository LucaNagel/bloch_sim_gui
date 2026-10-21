"""Extra gradient lobes and slice-axis remapping for Free Mode sequences.

Free Mode sequences put their slice gradient on z. To excite or encode in x
or y, the slice channel can be moved to another axis, and user-defined
trapezoid lobes can be added on any axis on top of any sequence.

Gradients are ``(Nt, 3)`` arrays in G/cm. Sample ``i`` holds its gradient
over ``[time[i], time[i + 1])``, the last sample repeats the previous step,
matching how ``BlochSimulator.simulate`` derives its time steps. Lobes are
averaged over each step, so their gradient moment is exact on any raster.
"""

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np

from .position_sampling import AXIS_NAMES, axis_index
from .units import LEGACY_GAMMA_HZ_PER_GAUSS

G_PER_CM_PER_MT_PER_M = 0.1
SLICE_AXIS = 2


@dataclass(frozen=True)
class GradientLobe:
    """Trapezoid gradient lobe on one axis.

    ``duration_ms`` is the full length including both ramps. A ramp longer
    than half the duration is clipped, which gives a triangle.
    """

    axis: str = "x"
    start_ms: float = 0.0
    duration_ms: float = 1.0
    amplitude_mt_per_m: float = 10.0
    ramp_ms: float = 0.1

    def __post_init__(self):
        object.__setattr__(self, "axis", AXIS_NAMES[axis_index(self.axis)])
        if not np.isfinite(self.start_ms) or self.start_ms < 0.0:
            raise ValueError("Gradient lobe start must be a finite time >= 0 ms.")
        if not np.isfinite(self.duration_ms) or self.duration_ms <= 0.0:
            raise ValueError("Gradient lobe duration must be positive.")
        if not np.isfinite(self.amplitude_mt_per_m):
            raise ValueError("Gradient lobe amplitude must be finite.")
        if not np.isfinite(self.ramp_ms) or self.ramp_ms < 0.0:
            raise ValueError("Gradient lobe ramp must be a finite time >= 0 ms.")

    @property
    def effective_ramp_ms(self) -> float:
        return min(float(self.ramp_ms), 0.5 * float(self.duration_ms))

    @property
    def end_ms(self) -> float:
        return float(self.start_ms) + float(self.duration_ms)

    @property
    def moment_mt_ms_per_m(self) -> float:
        """Gradient area of the lobe in mT·ms/m."""
        return float(self.amplitude_mt_per_m) * (
            float(self.duration_ms) - self.effective_ramp_ms
        )

    def cumulative_area(self, time_s) -> np.ndarray:
        """Lobe area from the lobe start up to ``time_s``, in G/cm·s."""
        t = (np.asarray(time_s, dtype=float) - self.start_ms * 1e-3).clip(min=0.0)
        duration = self.duration_ms * 1e-3
        ramp = self.effective_ramp_ms * 1e-3
        amplitude = self.amplitude_mt_per_m * G_PER_CM_PER_MT_PER_M
        t = np.minimum(t, duration)
        if ramp <= 0.0:
            return amplitude * t
        rise = np.minimum(t, ramp)
        area = 0.5 * rise * rise / ramp
        area += np.clip(t - ramp, 0.0, duration - 2.0 * ramp)
        fall = np.clip(t - (duration - ramp), 0.0, ramp)
        area += fall - 0.5 * fall * fall / ramp
        return amplitude * area

    def to_state(self) -> dict:
        return {
            "axis": self.axis,
            "start_ms": float(self.start_ms),
            "duration_ms": float(self.duration_ms),
            "amplitude_mt_per_m": float(self.amplitude_mt_per_m),
            "ramp_ms": float(self.ramp_ms),
        }

    @classmethod
    def from_state(cls, state: dict) -> "GradientLobe":
        return cls(
            axis=state.get("axis", "x"),
            start_ms=float(state.get("start_ms", 0.0)),
            duration_ms=float(state.get("duration_ms", 1.0)),
            amplitude_mt_per_m=float(state.get("amplitude_mt_per_m", 10.0)),
            ramp_ms=float(state.get("ramp_ms", 0.1)),
        )


def time_steps(time) -> np.ndarray:
    """Per-sample step lengths as used by the simulator (last step repeats)."""
    time = np.asarray(time, dtype=float)
    if time.size < 2:
        return np.zeros(time.size)
    steps = np.diff(time)
    return np.append(steps, steps[-1])


def remap_slice_axis(gradients, axis) -> np.ndarray:
    """Swap the slice (z) channel with ``axis`` so the slice lies along it."""
    gradients = np.array(gradients, dtype=float, copy=True)
    target = axis_index(axis)
    if target != SLICE_AXIS:
        gradients[:, [SLICE_AXIS, target]] = gradients[:, [target, SLICE_AXIS]]
    return gradients


def add_gradient_lobes(
    b1, gradients, time, lobes: Iterable[GradientLobe], dt_s: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Add trapezoid lobes, extending the sequence when a lobe ends later.

    Lobe start times are relative to ``time[0]``. Appended samples carry no RF
    and use ``dt_s`` (or the last raster step) as their spacing.
    """
    lobes = list(lobes)
    b1 = np.asarray(b1)
    gradients = np.array(gradients, dtype=float, copy=True)
    time = np.asarray(time, dtype=float)
    if not lobes or time.size == 0:
        return b1, gradients, time
    steps = time_steps(time)
    step = float(steps[-1]) if time.size > 1 and steps[-1] > 0 else float(dt_s)
    origin = float(time[0])
    end = float(time[-1]) + step - origin
    lobe_end = max(lobe.end_ms for lobe in lobes) * 1e-3
    if lobe_end > end + 1e-12:
        n_extra = int(np.ceil((lobe_end - end) / step - 1e-9))
        extra_time = float(time[-1]) + step * np.arange(1, n_extra + 1)
        time = np.concatenate([time, extra_time])
        b1 = np.concatenate([b1, np.zeros(n_extra, dtype=b1.dtype)])
        gradients = np.concatenate([gradients, np.zeros((n_extra, 3))])
        steps = time_steps(time)
    edges = np.append(time, time[-1] + steps[-1]) - origin
    widths = np.diff(edges)
    for lobe in lobes:
        area = np.diff(lobe.cumulative_area(edges))
        gradients[:, axis_index(lobe.axis)] += np.divide(
            area, widths, out=np.zeros_like(area), where=widths > 0
        )
    return b1, gradients, time


def gradient_moments_mt_ms_per_m(gradients, time) -> np.ndarray:
    """Net gradient area per axis over the whole sequence, in mT·ms/m."""
    gradients = np.asarray(gradients, dtype=float).reshape(-1, 3)
    steps = time_steps(time)
    area_g_s_per_cm = (gradients * steps[:, None]).sum(axis=0)
    return area_g_s_per_cm / G_PER_CM_PER_MT_PER_M * 1e3


def phase_twist_cycles_per_mm(moments_mt_ms_per_m: Sequence[float]) -> np.ndarray:
    """Phase cycles per mm that a gradient moment winds into the spins."""
    moments = np.asarray(moments_mt_ms_per_m, dtype=float)
    area_g_s_per_cm = moments * G_PER_CM_PER_MT_PER_M * 1e-3
    # The Free Mode kernels use the legacy proton gyromagnetic ratio.
    return LEGACY_GAMMA_HZ_PER_GAUSS * area_g_s_per_cm / 10.0
