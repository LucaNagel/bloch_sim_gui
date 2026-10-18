"""Dynamic hyperpolarized phantoms with pyruvate/lactate kinetics."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, replace
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Dict, Optional, Tuple

import numpy as np

from .phantom import Phantom
from .spectral_phantom import ChemicalSpecies
from .units import NUCLEUS_GAMMA_HZ_PER_T, hz_to_ppm, ppm_to_hz
from .dynamic_execution import (
    ACTION_ADC,
    ACTION_CHECKPOINT,
    ACTION_CRUSH,
    SEGMENT_RF,
    build_dynamic_execution_plan,
)
from .kernel_capabilities import DYNAMIC_KERNEL_NAMES


class _BoundedArrayCache:
    """Small byte-bounded LRU for state-independent solver coefficients."""

    def __init__(self, limit_bytes):
        self.limit_bytes = max(0, int(limit_bytes))
        self.current_bytes = 0
        self.values = OrderedDict()

    @staticmethod
    def _nbytes(value):
        if isinstance(value, np.ndarray):
            return int(value.nbytes)
        if isinstance(value, (tuple, list)):
            return sum(_BoundedArrayCache._nbytes(item) for item in value)
        return 0

    def get(self, key):
        try:
            value, byte_count = self.values.pop(key)
        except KeyError:
            return None
        self.values[key] = (value, byte_count)
        return value

    def put(self, key, value):
        byte_count = self._nbytes(value)
        if byte_count > self.limit_bytes:
            return value
        previous = self.values.pop(key, None)
        if previous is not None:
            self.current_bytes -= previous[1]
        while self.values and self.current_bytes + byte_count > self.limit_bytes:
            _, (_, removed_bytes) = self.values.popitem(last=False)
            self.current_bytes -= removed_bytes
        self.values[key] = (value, byte_count)
        self.current_bytes += byte_count
        return value


class _PairwiseArrayAccumulator:
    """Streaming fixed binary tree for deterministic parent-voxel sums."""

    def __init__(self):
        self._levels = []
        self._shape = None
        self._dtype = None

    def add(self, value):
        value = np.asarray(value)
        if self._shape is None:
            self._shape = value.shape
            self._dtype = value.dtype
        elif value.shape != self._shape:
            raise ValueError("pairwise accumulator values must have one shape")
        current = np.array(value, dtype=self._dtype, copy=True, order="C")
        level = 0
        while level < len(self._levels) and self._levels[level] is not None:
            current = self._levels[level] + current
            self._levels[level] = None
            level += 1
        if level == len(self._levels):
            self._levels.append(current)
        else:
            self._levels[level] = current

    def total(self):
        result = None
        for value in reversed(self._levels):
            if value is None:
                continue
            result = np.array(value, copy=True) if result is None else result + value
        if result is None:
            if self._shape is None:
                raise ValueError("cannot finish an empty pairwise accumulator")
            return np.zeros(self._shape, dtype=self._dtype)
        return result


@dataclass
class _DynamicBlockResult:
    """Sparse private result returned by one parent-voxel memory block."""

    active_indices: np.ndarray
    parent_species_signal: np.ndarray
    final_pool_active: np.ndarray
    checkpoint_pool_active: Optional[np.ndarray]
    runtime_stats: Dict
    template_result: object = None


@dataclass(frozen=True)
class KineticRegionDefinition:
    """One box or ellipsoid that overwrites a spatial kPL map."""

    name: str
    kind: str = "ellipsoid"
    center: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    size: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    kpl_s_inv: float = 0.0

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("kinetic region name must not be empty")
        if self.kind not in {"ellipsoid", "box"}:
            raise ValueError("kinetic region kind must be 'ellipsoid' or 'box'")
        if len(self.center) != 3 or not np.all(np.isfinite(self.center)):
            raise ValueError("kinetic region center requires three finite values")
        if len(self.size) != 3 or not np.all(np.isfinite(self.size)):
            raise ValueError("kinetic region size requires three finite values")
        if np.any(np.asarray(self.size) <= 0):
            raise ValueError("kinetic region size must be positive")
        if not np.isfinite(self.kpl_s_inv) or self.kpl_s_inv < 0:
            raise ValueError("kPL must be finite and non-negative")

    def rasterize(self, shape: Tuple[int, int, int]) -> np.ndarray:
        self.validate()
        coordinates = [(np.arange(count, dtype=float) + 0.5) / count for count in shape]
        x, y, z = np.meshgrid(*coordinates, indexing="ij")
        center = np.asarray(self.center, dtype=float)
        half = np.asarray(self.size, dtype=float) / 2.0
        if self.kind == "box":
            return (
                (np.abs(x - center[0]) <= half[0])
                & (np.abs(y - center[1]) <= half[1])
                & (np.abs(z - center[2]) <= half[2])
            )
        return ((x - center[0]) / half[0]) ** 2 + ((y - center[1]) / half[1]) ** 2 + (
            (z - center[2]) / half[2]
        ) ** 2 <= 1.0


@dataclass(frozen=True)
class TimeCurve:
    """Serializable scalar time course with exact piecewise integration.

    ``interpolation`` is either ``"linear"`` or ``"step"``.  ``outside``
    determines whether values before the first and after the last sample are
    zero or whether the nearest endpoint is held.  Times use seconds.
    """

    times_s: Tuple[float, ...]
    values: Tuple[float, ...]
    interpolation: str = "linear"
    outside: str = "zero"

    def __post_init__(self) -> None:
        times = tuple(float(value) for value in self.times_s)
        values = tuple(float(value) for value in self.values)
        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "values", values)
        if not times or len(times) != len(values):
            raise ValueError("time curve requires equally sized, non-empty samples")
        if not np.all(np.isfinite(times)):
            raise ValueError("time curve times must be finite")
        if len(times) > 1 and np.any(np.diff(times) <= 0):
            raise ValueError("time curve times must be strictly increasing")
        if not np.all(np.isfinite(values)):
            raise ValueError("time curve values must be finite")
        if self.interpolation not in {"linear", "step"}:
            raise ValueError("time curve interpolation must be 'linear' or 'step'")
        if self.outside not in {"zero", "hold"}:
            raise ValueError("time curve outside mode must be 'zero' or 'hold'")

    def value_at(self, time_s: float) -> float:
        """Evaluate this curve at one time point."""
        time_s = float(time_s)
        if not np.isfinite(time_s):
            raise ValueError("curve evaluation time must be finite")
        times = np.asarray(self.times_s)
        values = np.asarray(self.values)
        if time_s < times[0]:
            return float(values[0] if self.outside == "hold" else 0.0)
        if time_s > times[-1]:
            return float(values[-1] if self.outside == "hold" else 0.0)
        if self.interpolation == "linear" and times.size > 1:
            return float(np.interp(time_s, times, values))
        index = min(
            int(np.searchsorted(times, time_s, side="right") - 1), times.size - 1
        )
        return float(values[max(0, index)])

    def values_at(self, times_s) -> np.ndarray:
        """Vectorized evaluation preserving the scalar ``value_at`` rules."""
        query = np.asarray(times_s, dtype=float)
        if not np.all(np.isfinite(query)):
            raise ValueError("curve evaluation times must be finite")
        times = np.asarray(self.times_s, dtype=float)
        values = np.asarray(self.values, dtype=float)
        if self.interpolation == "linear" and times.size > 1:
            left = values[0] if self.outside == "hold" else 0.0
            right = values[-1] if self.outside == "hold" else 0.0
            result = np.interp(query.reshape(-1), times, values, left=left, right=right)
            return np.asarray(result).reshape(query.shape)
        indices = np.searchsorted(times, query, side="right") - 1
        clipped = np.clip(indices, 0, times.size - 1)
        result = values[clipped]
        if self.outside == "zero":
            result = np.where((query < times[0]) | (query > times[-1]), 0.0, result)
        return np.asarray(result)

    def interval_values(self, start_s: float, end_s: float) -> Tuple[float, float]:
        """Return endpoint values describing one knot-free interval."""
        if end_s < start_s:
            raise ValueError("time curve interval end must not precede its start")
        midpoint = (float(start_s) + float(end_s)) / 2.0
        if self.outside == "zero" and (
            midpoint < self.times_s[0] or midpoint > self.times_s[-1]
        ):
            return 0.0, 0.0
        if self.interpolation == "step" and end_s > start_s:
            value = self.value_at(midpoint)
            return value, value
        return self.value_at(start_s), self.value_at(end_s)

    def integral(self, start_s: float, end_s: float) -> float:
        """Integrate the piecewise curve exactly over an arbitrary interval."""
        start_s = float(start_s)
        end_s = float(end_s)
        if not np.isfinite(start_s) or not np.isfinite(end_s):
            raise ValueError("time curve integration bounds must be finite")
        if end_s < start_s:
            return -self.integral(end_s, start_s)
        if end_s == start_s:
            return 0.0
        internal = [value for value in self.times_s if start_s < value < end_s]
        boundaries = (start_s, *internal, end_s)
        total = 0.0
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            midpoint = (left + right) / 2.0
            if self.outside == "zero" and (
                midpoint < self.times_s[0] or midpoint > self.times_s[-1]
            ):
                continue
            if self.interpolation == "step":
                total += self.value_at(midpoint) * (right - left)
            else:
                total += (
                    0.5 * (self.value_at(left) + self.value_at(right)) * (right - left)
                )
        return float(total)

    def breakpoints_s(self, duration_s: float) -> Tuple[float, ...]:
        """Return curve knots lying on the requested simulation timeline."""
        duration_s = float(duration_s)
        return tuple(value for value in self.times_s if 0.0 <= value <= duration_s)

    def shifted(self, offset_s: float) -> "TimeCurve":
        """Return a copy whose sample times are translated by ``offset_s``."""
        offset_s = float(offset_s)
        if not np.isfinite(offset_s):
            raise ValueError("time curve offset must be finite")
        return TimeCurve(
            times_s=tuple(value + offset_s for value in self.times_s),
            values=self.values,
            interpolation=self.interpolation,
            outside=self.outside,
        )

    def to_dict(self) -> Dict:
        return {
            "times_s": self.times_s,
            "values": self.values,
            "interpolation": self.interpolation,
            "outside": self.outside,
        }

    @classmethod
    def from_dict(cls, values: Dict) -> "TimeCurve":
        return cls(
            times_s=tuple(values["times_s"]),
            values=tuple(values["values"]),
            interpolation=str(values.get("interpolation", "linear")),
            outside=str(values.get("outside", "zero")),
        )


def _time_curve_interval_values(
    curve: TimeCurve,
    start_s: float,
    end_s: float,
    arrival_delay_s=None,
):
    """Evaluate one source interval with optional per-spin arrival delays."""
    if arrival_delay_s is None:
        return curve.interval_values(start_s, end_s)
    delays = np.asarray(arrival_delay_s, dtype=float)
    if curve.interpolation == "step" and end_s > start_s:
        values = curve.values_at(0.5 * (start_s + end_s) - delays)
        return values, values
    return (
        curve.values_at(float(start_s) - delays),
        curve.values_at(float(end_s) - delays),
    )


@dataclass
class PyruvateInflow:
    """Voxelwise pyruvate delivery driven by a scalar concentration-rate curve.

    New phantoms provide ``polarization_curve`` so concentration influx and its
    polarization remain separate.  A missing polarization curve denotes the
    legacy direct-Mz source for backwards compatibility. An optional
    ``arrival_delay_map_s`` shifts both curves independently in every voxel;
    ``max_step_s`` bounds their piecewise-linear integration grid.
    """

    rate_curve_s_inv: TimeCurve
    delivery_map: np.ndarray
    polarization_curve: Optional[TimeCurve] = None
    arrival_delay_map_s: Optional[np.ndarray] = None
    max_step_s: Optional[float] = None

    def validate(self, shape: Tuple[int, int, int]) -> None:
        values = np.asarray(self.delivery_map, dtype=np.float64)
        if values.shape != shape or not np.all(np.isfinite(values)):
            raise ValueError(
                "pyruvate delivery map must be finite and match phantom shape"
            )
        if np.any(values < 0) or np.any(np.asarray(self.rate_curve_s_inv.values) < 0):
            raise ValueError("pyruvate inflow maps and rates must be non-negative")
        if self.polarization_curve is not None and np.any(
            np.asarray(self.polarization_curve.values) < 0
        ):
            raise ValueError("pyruvate inflow polarization must be non-negative")
        self.delivery_map = values
        if self.arrival_delay_map_s is not None:
            delays = np.asarray(self.arrival_delay_map_s, dtype=np.float64)
            if delays.shape != shape:
                raise ValueError("pyruvate arrival-delay map must match phantom shape")
            support = values > 0
            if not np.all(np.isfinite(delays[support])) or np.any(delays[support] < 0):
                raise ValueError(
                    "pyruvate arrival delays must be finite and non-negative "
                    "inside the delivery support"
                )
            self.arrival_delay_map_s = delays
        if self.max_step_s is not None:
            self.max_step_s = float(self.max_step_s)
            if not np.isfinite(self.max_step_s) or self.max_step_s <= 0:
                raise ValueError("inflow maximum step must be positive and finite")

    @property
    def support_mask(self) -> np.ndarray:
        return np.asarray(self.delivery_map) > 0

    @property
    def has_spatial_delays(self) -> bool:
        return self.arrival_delay_map_s is not None

    def breakpoints_s(
        self, duration_s: float, *, timeline_offset_s: float = 0.0
    ) -> Tuple[float, ...]:
        """Return source boundaries, including a bounded spatial-delay grid."""
        duration_s = float(duration_s)
        timeline_offset_s = float(timeline_offset_s)
        if not self.has_spatial_delays:
            values = list(
                self.rate_curve_s_inv.shifted(timeline_offset_s).breakpoints_s(
                    duration_s
                )
            )
            if self.polarization_curve is not None:
                values.extend(
                    self.polarization_curve.shifted(timeline_offset_s).breakpoints_s(
                        duration_s
                    )
                )
            return tuple(sorted(set(values)))
        support = self.support_mask
        if not np.any(support):
            return ()
        delays = np.asarray(self.arrival_delay_map_s)[support]
        start = (
            self.rate_curve_s_inv.times_s[0] + float(np.min(delays)) + timeline_offset_s
        )
        end = (
            self.rate_curve_s_inv.times_s[-1]
            + float(np.max(delays))
            + timeline_offset_s
        )
        step = self.max_step_s
        if step is None:
            curve_steps = np.diff(np.asarray(self.rate_curve_s_inv.times_s))
            step = (
                float(np.min(curve_steps)) if curve_steps.size else max(duration_s, 1.0)
            )
        first = max(0.0, start)
        last = min(duration_s, end)
        if last < first:
            return ()
        count = max(1, int(np.ceil((last - first) / step)))
        grid = first + np.arange(count + 1, dtype=float) * step
        grid = grid[grid <= last]
        values = [first, last, *grid.tolist()]
        return tuple(sorted(set(float(value) for value in values)))


@dataclass
class DynamicB0:
    """Time-dependent B0 offset in Hz with a voxelwise scale map."""

    offset_curve_hz: TimeCurve
    spatial_scale_map: np.ndarray
    pool_scale: Tuple[float, ...] = (1.0, 1.0)

    def validate(self, shape: Tuple[int, int, int]) -> None:
        values = np.asarray(self.spatial_scale_map, dtype=np.float64)
        if values.shape != shape or not np.all(np.isfinite(values)):
            raise ValueError(
                "dynamic B0 scale map must be finite and match phantom shape"
            )
        scales = tuple(float(value) for value in self.pool_scale)
        if not scales or not np.all(np.isfinite(scales)):
            raise ValueError("dynamic B0 pool scale requires finite values")
        self.spatial_scale_map = values
        self.pool_scale = scales


def rasterize_kpl_regions(
    shape: Tuple[int, int, int],
    regions: Tuple[KineticRegionDefinition, ...],
    default_kpl_s_inv: float = 0.0,
) -> np.ndarray:
    """Rasterize ordered kinetic regions; later regions overwrite earlier ones."""
    if not np.isfinite(default_kpl_s_inv) or default_kpl_s_inv < 0:
        raise ValueError("default kPL must be finite and non-negative")
    result = np.full(shape, float(default_kpl_s_inv), dtype=np.float64)
    for region in regions:
        result[region.rasterize(shape)] = region.kpl_s_inv
    return result


@dataclass
class DynamicSpectralPhantom:
    """Multipool phantom with irreversible precursor/product kinetics.

    Pools zero and one are the default pyruvate/lactate kinetic pair. Additional
    pools normally undergo RF excitation, off-resonance evolution, and T1/T2
    relaxation without exchange. A specialized phantom may expose a
    ``kpa_map_s_inv``; the first additional pool then receives exact
    precursor-to-product conversion (used by the mouse liver Alanine pool).
    """

    shape: Tuple[int, int, int]
    fov: Tuple[float, float, float]
    pools: Tuple[ChemicalSpecies, ...]
    initial_concentration_maps: Dict[str, np.ndarray]
    kpl_map_s_inv: np.ndarray
    initial_spin_density_maps: Optional[Dict[str, np.ndarray]] = None
    equilibrium_polarization: float = 0.0
    b0_map_ppm: Optional[np.ndarray] = None
    b0_map: Optional[np.ndarray] = None
    field_strength: float = 3.0
    nucleus: str = "C13"
    spectral_reference_ppm: float = 0.0
    spectral_window_center_ppm: Optional[float] = None
    spectral_bandwidth_ppm: float = 20.0
    spectral_points: int = 1024
    name: str = "Dynamic pyruvate/lactate phantom"
    kinetic_regions: Tuple[KineticRegionDefinition, ...] = ()
    pyruvate_inflow: Optional[PyruvateInflow] = None
    dynamic_b0: Optional[DynamicB0] = None
    conversion_start_s: float = 0.0
    kinetics_time_offset_s: float = 0.0
    metadata: Dict = field(default_factory=dict)
    coordinate_system: str = "object_xyz"
    affine_ijk_to_xyz_m: Optional[np.ndarray] = None
    positions: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.shape = tuple(int(value) for value in self.shape)
        self.fov = tuple(float(value) for value in self.fov)
        self.pools = tuple(self.pools)
        self.kinetic_regions = tuple(self.kinetic_regions)
        if self.spectral_window_center_ppm is None:
            self.spectral_window_center_ppm = float(self.spectral_reference_ppm)
        if len(self.shape) != 3 or any(value <= 0 for value in self.shape):
            raise ValueError("dynamic phantom shape requires three positive values")
        if (
            len(self.fov) != 3
            or not np.all(np.isfinite(self.fov))
            or min(self.fov) <= 0
        ):
            raise ValueError("dynamic phantom FOV requires three positive values")
        if len(self.pools) < 2:
            raise ValueError("the dynamic model requires at least two pools")
        if len({pool.name for pool in self.pools}) != len(self.pools):
            raise ValueError("dynamic pool names must be unique")
        if self.nucleus not in NUCLEUS_GAMMA_HZ_PER_T:
            raise ValueError(f"unsupported nucleus {self.nucleus!r}")
        if not np.isfinite(self.field_strength) or self.field_strength <= 0:
            raise ValueError("field strength must be positive and finite")
        if not np.isfinite(self.spectral_reference_ppm):
            raise ValueError("spectral reference must be finite")
        if not np.isfinite(self.spectral_window_center_ppm):
            raise ValueError("spectral window center must be finite")
        if (
            not np.isfinite(self.spectral_bandwidth_ppm)
            or self.spectral_bandwidth_ppm <= 0
        ):
            raise ValueError("spectral bandwidth must be positive and finite")
        if (
            int(self.spectral_points) != self.spectral_points
            or self.spectral_points < 2
        ):
            raise ValueError("spectral points must be an integer >= 2")
        if not str(self.coordinate_system).strip():
            raise ValueError("coordinate_system must not be empty")
        if self.affine_ijk_to_xyz_m is None:
            self.affine_ijk_to_xyz_m = Phantom.default_affine(self.shape, self.fov)
        else:
            affine = np.asarray(self.affine_ijk_to_xyz_m, dtype=np.float64)
            if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
                raise ValueError("affine_ijk_to_xyz_m must be a finite 4x4 matrix")
            self.affine_ijk_to_xyz_m = affine
        for pool in self.pools:
            if pool.name not in self.initial_concentration_maps:
                raise ValueError(f"missing initial map for pool {pool.name!r}")
            values = np.asarray(
                self.initial_concentration_maps[pool.name], dtype=np.float64
            )
            if values.shape != self.shape or not np.all(np.isfinite(values)):
                raise ValueError(f"initial map for {pool.name!r} has invalid values")
            if np.any(values < 0):
                raise ValueError("initial pool magnetization must be non-negative")
            self.initial_concentration_maps[pool.name] = values
        if self.initial_spin_density_maps is not None:
            for pool in self.pools:
                if pool.name not in self.initial_spin_density_maps:
                    raise ValueError(f"missing spin-density map for pool {pool.name!r}")
                values = np.asarray(
                    self.initial_spin_density_maps[pool.name], dtype=np.float64
                )
                if values.shape != self.shape or not np.all(np.isfinite(values)):
                    raise ValueError(
                        f"spin-density map for {pool.name!r} has invalid values"
                    )
                if np.any(values < 0):
                    raise ValueError("initial spin density must be non-negative")
                self.initial_spin_density_maps[pool.name] = values
        self.equilibrium_polarization = float(self.equilibrium_polarization)
        if (
            not np.isfinite(self.equilibrium_polarization)
            or self.equilibrium_polarization < 0
        ):
            raise ValueError("equilibrium polarization must be finite and non-negative")
        self.kpl_map_s_inv = np.asarray(self.kpl_map_s_inv, dtype=np.float64)
        if self.kpl_map_s_inv.shape != self.shape:
            raise ValueError("kPL map shape must match dynamic phantom")
        if not np.all(np.isfinite(self.kpl_map_s_inv)) or np.any(
            self.kpl_map_s_inv < 0
        ):
            raise ValueError("kPL map must be finite and non-negative")
        if self.b0_map is not None and self.b0_map_ppm is not None:
            raise ValueError("b0_map and b0_map_ppm cannot be combined")
        for name in ("b0_map", "b0_map_ppm"):
            values = getattr(self, name)
            if values is not None:
                values = np.asarray(values, dtype=np.float64)
                if values.shape != self.shape or not np.all(np.isfinite(values)):
                    raise ValueError(f"{name} must be finite and match phantom shape")
                setattr(self, name, values)
        for region in self.kinetic_regions:
            region.validate()
        if self.pyruvate_inflow is not None:
            self.pyruvate_inflow.validate(self.shape)
        if self.dynamic_b0 is not None:
            self.dynamic_b0.validate(self.shape)
            scales = tuple(self.dynamic_b0.pool_scale)
            if len(scales) == 2 and len(self.pools) > 2:
                scales += (1.0,) * (len(self.pools) - 2)
            if len(scales) != len(self.pools):
                raise ValueError(
                    "dynamic B0 pool scale must contain one value per pool"
                )
            self.dynamic_b0.pool_scale = scales
        self.conversion_start_s = float(self.conversion_start_s)
        if not np.isfinite(self.conversion_start_s):
            raise ValueError("conversion start time must be finite")
        self.kinetics_time_offset_s = float(self.kinetics_time_offset_s)
        if not np.isfinite(self.kinetics_time_offset_s):
            raise ValueError("kinetics time offset must be finite")
        axes = Phantom.coordinate_vectors(self.shape, self.affine_ijk_to_xyz_m)
        x, y, z = np.meshgrid(*axes, indexing="ij")
        self.positions = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

    @property
    def ndim(self) -> int:
        return 3

    @property
    def nvoxels(self) -> int:
        return int(np.prod(self.shape))

    @property
    def voxel_volume_m3(self) -> float:
        return float(np.prod(np.asarray(self.fov) / np.asarray(self.shape)))

    @property
    def n_species(self) -> int:
        return len(self.pools)

    @property
    def species(self):
        return self.pools

    @property
    def concentration_maps(self):
        return self.initial_concentration_maps

    @property
    def mask(self) -> np.ndarray:
        active = self.pd_map > 0
        if self.pyruvate_inflow is not None:
            active = active | self.pyruvate_inflow.support_mask
        return active

    @property
    def n_active(self) -> int:
        return int(np.count_nonzero(self.mask))

    @property
    def pd_map(self) -> np.ndarray:
        maps = self.initial_spin_density_maps or self.initial_concentration_maps
        return sum(maps[pool.name] for pool in self.pools)

    @property
    def initial_spin_density(self) -> np.ndarray:
        """Return pool-resolved spin density, or zeros for legacy excess-Mz data."""
        if self.initial_spin_density_maps is None:
            return np.zeros((self.n_species,) + self.shape, dtype=np.float64)
        return np.stack(
            [self.initial_spin_density_maps[pool.name] for pool in self.pools]
        )

    @property
    def t1_map(self) -> np.ndarray:
        return self._weighted_pool_property("t1")

    @property
    def t2_map(self) -> np.ndarray:
        return self._weighted_pool_property("t2")

    def _weighted_pool_property(self, name: str) -> np.ndarray:
        total = self.pd_map
        result = np.zeros(self.shape, dtype=float)
        maps = self.initial_spin_density_maps or self.initial_concentration_maps
        for pool in self.pools:
            result += maps[pool.name] * float(getattr(pool, name))
        result = np.divide(result, total, out=np.zeros_like(result), where=total > 0)
        if self.pyruvate_inflow is not None:
            input_only = self.pyruvate_inflow.support_mask & (total == 0)
            result[input_only] = float(getattr(self.pools[0], name))
        return result

    def dynamic_breakpoints_s(self, duration_s: float) -> Tuple[float, ...]:
        """Return all phantom-driver knots on the sequence timeline."""
        values = []
        if self.pyruvate_inflow is not None:
            values.extend(
                self.pyruvate_inflow.breakpoints_s(
                    duration_s,
                    timeline_offset_s=-self.kinetics_time_offset_s,
                )
            )
        if self.dynamic_b0 is not None:
            values.extend(self.dynamic_b0.offset_curve_hz.breakpoints_s(duration_s))
        conversion_start_s = self.conversion_start_on_sequence_timeline_s
        if 0.0 <= conversion_start_s <= duration_s:
            values.append(conversion_start_s)
        return tuple(sorted(set(values)))

    @property
    def conversion_start_on_sequence_timeline_s(self) -> float:
        """Return when conversion starts relative to Pulseq sequence time zero."""
        return self.conversion_start_s - self.kinetics_time_offset_s

    @property
    def inflow_curve_on_sequence_timeline(self) -> Optional[TimeCurve]:
        """Return the inflow curve expressed relative to sequence time zero."""
        if self.pyruvate_inflow is None:
            return None
        return self.pyruvate_inflow.rate_curve_s_inv.shifted(
            -self.kinetics_time_offset_s
        )

    @property
    def inflow_polarization_curve_on_sequence_timeline(self) -> Optional[TimeCurve]:
        if (
            self.pyruvate_inflow is None
            or self.pyruvate_inflow.polarization_curve is None
        ):
            return None
        return self.pyruvate_inflow.polarization_curve.shifted(
            -self.kinetics_time_offset_s
        )

    def b0_offset_hz(self, field_strength=None, nucleus=None) -> np.ndarray:
        if self.b0_map_ppm is not None:
            return np.asarray(
                ppm_to_hz(
                    self.b0_map_ppm,
                    self.field_strength if field_strength is None else field_strength,
                    self.nucleus if nucleus is None else nucleus,
                )
            )
        if self.b0_map is not None:
            return self.b0_map
        return np.zeros(self.shape, dtype=float)

    def get_b0_offset_map_hz(self, field_strength=None, nucleus=None) -> np.ndarray:
        """Return the static B0 map in Hz for spectral-viewer compatibility."""
        return self.b0_offset_hz(field_strength, nucleus)

    def get_b0_offset_map_ppm(self, field_strength=None, nucleus=None) -> np.ndarray:
        """Return the static B0 map in ppm for spectral-viewer compatibility."""
        if self.b0_map_ppm is not None:
            return np.asarray(self.b0_map_ppm, dtype=float)
        effective_field = (
            self.field_strength if field_strength is None else float(field_strength)
        )
        effective_nucleus = self.nucleus if nucleus is None else str(nucleus)
        if self.b0_map is not None:
            return np.asarray(
                hz_to_ppm(self.b0_map, effective_field, effective_nucleus),
                dtype=float,
            )
        return np.zeros(self.shape, dtype=float)

    def get_frequency_offset(self, name, field_strength=None, nucleus=None) -> float:
        """Return one pool frequency in Hz at the requested field strength."""
        for pool in self.pools:
            if pool.name == name:
                return pool.get_frequency_offset(
                    (
                        self.field_strength
                        if field_strength is None
                        else float(field_strength)
                    ),
                    self.nucleus if nucleus is None else str(nucleus),
                )
        return 0.0

    def get_frequency_offset_ppm(
        self, name, field_strength=None, nucleus=None
    ) -> float:
        """Return one pool frequency as an offset from the spectral reference."""
        for pool in self.pools:
            if pool.name != name:
                continue
            if pool.frequency_offset_hz is None:
                return float(pool.chemical_shift_ppm)
            effective_field = (
                self.field_strength if field_strength is None else float(field_strength)
            )
            effective_nucleus = self.nucleus if nucleus is None else str(nucleus)
            return float(
                hz_to_ppm(
                    pool.frequency_offset_hz,
                    effective_field,
                    effective_nucleus,
                )
            )
        return 0.0

    @property
    def df_map_ppm(self) -> np.ndarray:
        """Concentration-weighted initial mean frequency map in ppm."""
        total = self.pd_map
        result = self.get_b0_offset_map_ppm()
        for pool in self.pools:
            result = result + self.initial_concentration_maps[pool.name] * (
                self.get_frequency_offset_ppm(pool.name) / np.maximum(total, 1e-15)
            )
        return result

    def spectrum_at(
        self,
        index,
        frequency_hz=None,
        points=None,
        field_strength=None,
        nucleus=None,
    ):
        """Return the initial multipool Lorentzian spectrum at one voxel."""
        if len(index) != self.ndim:
            raise ValueError("index dimensionality must match dynamic phantom")
        effective_field = (
            self.field_strength if field_strength is None else float(field_strength)
        )
        effective_nucleus = self.nucleus if nucleus is None else str(nucleus)
        if points is None:
            points = self.spectral_points
        if frequency_hz is None:
            half_bandwidth_hz = float(
                ppm_to_hz(
                    self.spectral_bandwidth_ppm / 2.0,
                    effective_field,
                    effective_nucleus,
                )
            )
            window_center_hz = float(
                ppm_to_hz(
                    self.spectral_window_center_ppm - self.spectral_reference_ppm,
                    effective_field,
                    effective_nucleus,
                )
            )
            frequency_hz = np.linspace(
                window_center_hz - half_bandwidth_hz,
                window_center_hz + half_bandwidth_hz,
                int(points),
            )
        frequency_hz = np.asarray(frequency_hz, dtype=float)
        spectrum = np.zeros(frequency_hz.shape, dtype=float)
        b0_hz = float(self.b0_offset_hz(effective_field, effective_nucleus)[index])
        for pool in self.pools:
            centre_hz = self.get_frequency_offset(
                pool.name, effective_field, effective_nucleus
            )
            half_width_hz = 1.0 / (2.0 * np.pi * pool.t2_star)
            amplitude = float(self.initial_concentration_maps[pool.name][index])
            spectrum += amplitude / (
                1.0 + ((frequency_hz - (centre_hz + b0_hz)) / half_width_hz) ** 2
            )
        return frequency_hz, spectrum

    def spectrum_at_ppm(
        self,
        index,
        frequency_ppm=None,
        points=None,
        *,
        absolute=True,
        linewidth_field_strength=None,
        nucleus=None,
    ):
        """Return the initial two-pool Lorentzian spectrum on a ppm axis."""
        if len(index) != self.ndim:
            raise ValueError("index dimensionality must match dynamic phantom")
        if points is None:
            points = self.spectral_points
        if frequency_ppm is None:
            centre_ppm = (
                self.spectral_window_center_ppm
                if absolute
                else self.spectral_window_center_ppm - self.spectral_reference_ppm
            )
            half_bandwidth_ppm = self.spectral_bandwidth_ppm / 2.0
            frequency_ppm = np.linspace(
                centre_ppm - half_bandwidth_ppm,
                centre_ppm + half_bandwidth_ppm,
                int(points),
            )
        frequency_ppm = np.asarray(frequency_ppm, dtype=float)
        relative_frequency_ppm = (
            frequency_ppm - self.spectral_reference_ppm if absolute else frequency_ppm
        )
        effective_field = (
            self.field_strength
            if linewidth_field_strength is None
            else float(linewidth_field_strength)
        )
        effective_nucleus = self.nucleus if nucleus is None else str(nucleus)
        b0_ppm = float(
            self.get_b0_offset_map_ppm(effective_field, effective_nucleus)[index]
        )
        spectrum = np.zeros(frequency_ppm.shape, dtype=float)
        for pool in self.pools:
            centre_ppm = self.get_frequency_offset_ppm(
                pool.name, effective_field, effective_nucleus
            )
            fwhm_ppm = abs(
                float(
                    hz_to_ppm(
                        1.0 / (np.pi * pool.t2_star),
                        effective_field,
                        effective_nucleus,
                    )
                )
            )
            half_width_ppm = max(fwhm_ppm / 2.0, np.finfo(float).eps)
            amplitude = float(self.initial_concentration_maps[pool.name][index])
            spectrum += amplitude / (
                1.0
                + ((relative_frequency_ppm - (centre_ppm + b0_ppm)) / half_width_ppm)
                ** 2
            )
        return frequency_ppm, spectrum

    @property
    def effective_df_map(self) -> np.ndarray:
        total = self.pd_map
        result = self.b0_offset_hz() * (total > 0)
        for pool in self.pools:
            result += self.initial_concentration_maps[pool.name] * (
                pool.get_frequency_offset(self.field_strength, self.nucleus)
                / np.maximum(total, 1e-15)
            )
        return result

    @property
    def initial_magnetization(self) -> np.ndarray:
        result = np.zeros((self.n_species,) + self.shape + (3,), dtype=np.float64)
        for index, pool in enumerate(self.pools):
            result[index, ..., 2] = self.initial_concentration_maps[pool.name]
        return result

    def restore_missing_designed_pools(self) -> "DynamicSpectralPhantom":
        """Rebuild legacy dynamic phantoms that silently dropped extra peaks."""
        design_data = self.metadata.get("phantom_design")
        if not isinstance(design_data, dict) or not design_data.get("dynamic_enabled"):
            return self
        designed_names = []
        for shape in design_data.get("shapes", ()):
            for peak in shape.get("peaks", ()):
                name = str(peak.get("name", "")).strip()
                if name and name not in designed_names:
                    designed_names.append(name)
        existing_names = {pool.name for pool in self.pools}
        missing_names = [name for name in designed_names if name not in existing_names]
        if not missing_names:
            return self

        # Import lazily to avoid the module cycle: phantom_design builds this
        # class, while legacy dynamic files retain their original design there.
        from .phantom_design import PhantomDesign

        design = PhantomDesign.from_dict(
            design_data,
            legacy_field_strength_t=self.field_strength,
            legacy_nucleus=self.nucleus,
        )
        rebuilt = design.build()
        if not isinstance(rebuilt, DynamicSpectralPhantom):
            raise ValueError(
                "stored dynamic phantom design did not rebuild dynamically"
            )
        rebuilt.name = self.name
        rebuilt.metadata["restored_missing_dynamic_pools"] = tuple(missing_names)
        return rebuilt

    def to_xarray(self):
        """Return this dynamic phantom as a coordinate-aware xarray Dataset."""
        import xarray as xr

        spatial_dims = ("x", "y", "z")
        pool_names = [pool.name for pool in self.pools]
        coords = {
            dim: (
                dim,
                values,
                {
                    "units": "m",
                    "long_name": f"{dim} coordinate in {self.coordinate_system}",
                },
            )
            for dim, values in zip(
                spatial_dims,
                Phantom.coordinate_vectors(self.shape, self.affine_ijk_to_xyz_m),
            )
        }
        coords["species"] = ("species", pool_names)
        header = {
            "pools": [pool.__dict__ for pool in self.pools],
            "kinetic_regions": [region.__dict__ for region in self.kinetic_regions],
            "pyruvate_inflow": (
                None
                if self.pyruvate_inflow is None
                else {
                    "rate_curve_s_inv": self.pyruvate_inflow.rate_curve_s_inv.to_dict(),
                    "max_step_s": self.pyruvate_inflow.max_step_s,
                    "polarization_curve": (
                        None
                        if self.pyruvate_inflow.polarization_curve is None
                        else self.pyruvate_inflow.polarization_curve.to_dict()
                    ),
                }
            ),
            "dynamic_b0": (
                None
                if self.dynamic_b0 is None
                else {
                    "offset_curve_hz": self.dynamic_b0.offset_curve_hz.to_dict(),
                    "pool_scale": self.dynamic_b0.pool_scale,
                }
            ),
            "conversion_start_s": self.conversion_start_s,
            "kinetics_time_offset_s": self.kinetics_time_offset_s,
            "equilibrium_polarization": self.equilibrium_polarization,
            "metadata": self.metadata,
        }
        data_vars = {
            "initial_concentration": (
                ("species",) + spatial_dims,
                np.stack(
                    [self.initial_concentration_maps[name] for name in pool_names]
                ),
                {"units": "relative"},
            ),
            "kpl_map_s_inv": (
                spatial_dims,
                self.kpl_map_s_inv,
                {"units": "1/s"},
            ),
            "b0_ppm": (
                spatial_dims,
                (
                    self.b0_map_ppm
                    if self.b0_map_ppm is not None
                    else np.zeros(self.shape, dtype=float)
                ),
                {"units": "ppm"},
            ),
            "b0_hz": (
                spatial_dims,
                (
                    self.b0_map
                    if self.b0_map is not None
                    else np.zeros(self.shape, dtype=float)
                ),
                {"units": "Hz"},
            ),
            "species_chemical_shift_ppm": (
                "species",
                [pool.chemical_shift_ppm for pool in self.pools],
                {"units": "ppm"},
            ),
            "species_t1": ("species", [pool.t1 for pool in self.pools]),
            "species_t2": ("species", [pool.t2 for pool in self.pools]),
            "species_t2_star": (
                "species",
                [pool.t2_star for pool in self.pools],
            ),
            "species_frequency_offset_hz": (
                "species",
                [
                    (
                        np.nan
                        if pool.frequency_offset_hz is None
                        else pool.frequency_offset_hz
                    )
                    for pool in self.pools
                ],
                {"units": "Hz"},
            ),
        }
        if self.initial_spin_density_maps is not None:
            data_vars["initial_spin_density"] = (
                ("species",) + spatial_dims,
                np.stack([self.initial_spin_density_maps[name] for name in pool_names]),
                {"units": "relative concentration"},
            )
        if self.pyruvate_inflow is not None:
            data_vars["pyruvate_delivery_map"] = (
                spatial_dims,
                self.pyruvate_inflow.delivery_map,
                {"units": "relative"},
            )
            if self.pyruvate_inflow.arrival_delay_map_s is not None:
                data_vars["pyruvate_arrival_delay_s"] = (
                    spatial_dims,
                    self.pyruvate_inflow.arrival_delay_map_s,
                    {"units": "s"},
                )
        if self.dynamic_b0 is not None:
            data_vars["dynamic_b0_scale_map"] = (
                spatial_dims,
                self.dynamic_b0.spatial_scale_map,
                {"units": "dimensionless"},
            )
        return xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs={
                "format": "blochsimulator-dynamic-spectral-phantom-xarray",
                "version": 5,
                "name": self.name,
                "fov_m": np.asarray(self.fov, dtype=np.float64),
                "field_strength": self.field_strength,
                "nucleus": self.nucleus,
                "spectral_reference_ppm": self.spectral_reference_ppm,
                "spectral_window_center_ppm": self.spectral_window_center_ppm,
                "spectral_bandwidth_ppm": self.spectral_bandwidth_ppm,
                "spectral_points": self.spectral_points,
                "has_b0_map": self.b0_map is not None,
                "has_b0_map_ppm": self.b0_map_ppm is not None,
                "has_pyruvate_inflow": self.pyruvate_inflow is not None,
                "has_dynamic_b0": self.dynamic_b0 is not None,
                "coordinate_system": self.coordinate_system,
                "affine_ijk_to_xyz_m": self.affine_ijk_to_xyz_m.reshape(-1),
                "dynamic_header_json": json.dumps(header, default=str),
            },
        )

    @classmethod
    def from_xarray(cls, dataset) -> "DynamicSpectralPhantom":
        """Create a dynamic spectral phantom from an xarray Dataset."""
        ds = dataset.load()
        if "species" not in ds.sizes or any(dim not in ds.sizes for dim in "xyz"):
            raise ValueError("dynamic phantom dataset requires species, x, y, and z")
        shape = tuple(int(ds.sizes[dim]) for dim in "xyz")
        fov = tuple(float(value) for value in np.asarray(ds.attrs["fov_m"]).ravel())
        affine = np.asarray(ds.attrs["affine_ijk_to_xyz_m"], dtype=float).reshape(4, 4)
        header = json.loads(ds.attrs.get("dynamic_header_json", "{}"))
        pool_metadata = header.get("pools", [])
        pool_names = [str(value) for value in np.asarray(ds.coords["species"])]
        pools = []
        for index, name in enumerate(pool_names):
            if index < len(pool_metadata):
                item = dict(pool_metadata[index])
                item["name"] = name
                pools.append(ChemicalSpecies(**item))
            else:
                frequency_hz = float(ds["species_frequency_offset_hz"][index])
                pools.append(
                    ChemicalSpecies(
                        name=name,
                        chemical_shift_ppm=float(
                            ds["species_chemical_shift_ppm"][index]
                        ),
                        t1=float(ds["species_t1"][index]),
                        t2=float(ds["species_t2"][index]),
                        t2_star=float(ds["species_t2_star"][index]),
                        frequency_offset_hz=(
                            None if np.isnan(frequency_hz) else frequency_hz
                        ),
                    )
                )
        regions = tuple(
            KineticRegionDefinition(**item)
            for item in header.get("kinetic_regions", ())
        )
        inflow_metadata = header.get("pyruvate_inflow")
        pyruvate_inflow = None
        if inflow_metadata is not None and "pyruvate_delivery_map" in ds:
            pyruvate_inflow = PyruvateInflow(
                rate_curve_s_inv=TimeCurve.from_dict(
                    inflow_metadata["rate_curve_s_inv"]
                ),
                delivery_map=np.asarray(ds["pyruvate_delivery_map"]),
                polarization_curve=(
                    None
                    if inflow_metadata.get("polarization_curve") is None
                    else TimeCurve.from_dict(inflow_metadata["polarization_curve"])
                ),
                arrival_delay_map_s=(
                    None
                    if "pyruvate_arrival_delay_s" not in ds
                    else np.asarray(ds["pyruvate_arrival_delay_s"])
                ),
                max_step_s=inflow_metadata.get("max_step_s"),
            )
        dynamic_b0_metadata = header.get("dynamic_b0")
        dynamic_b0 = None
        if dynamic_b0_metadata is not None and "dynamic_b0_scale_map" in ds:
            dynamic_b0 = DynamicB0(
                offset_curve_hz=TimeCurve.from_dict(
                    dynamic_b0_metadata["offset_curve_hz"]
                ),
                spatial_scale_map=np.asarray(ds["dynamic_b0_scale_map"]),
                pool_scale=tuple(dynamic_b0_metadata.get("pool_scale", (1.0, 1.0))),
            )
        phantom = cls(
            shape=shape,
            fov=fov,
            pools=tuple(pools),
            initial_concentration_maps={
                name: np.asarray(ds["initial_concentration"].sel(species=name))
                for name in pool_names
            },
            initial_spin_density_maps=(
                None
                if "initial_spin_density" not in ds
                else {
                    name: np.asarray(ds["initial_spin_density"].sel(species=name))
                    for name in pool_names
                }
            ),
            equilibrium_polarization=float(header.get("equilibrium_polarization", 0.0)),
            kpl_map_s_inv=np.asarray(ds["kpl_map_s_inv"]),
            b0_map=(
                np.asarray(ds["b0_hz"])
                if bool(ds.attrs.get("has_b0_map", False))
                else None
            ),
            b0_map_ppm=(
                np.asarray(ds["b0_ppm"])
                if bool(ds.attrs.get("has_b0_map_ppm", False))
                else None
            ),
            field_strength=float(ds.attrs["field_strength"]),
            nucleus=str(ds.attrs["nucleus"]),
            spectral_reference_ppm=float(ds.attrs.get("spectral_reference_ppm", 0.0)),
            spectral_window_center_ppm=float(
                ds.attrs.get(
                    "spectral_window_center_ppm",
                    ds.attrs.get("spectral_reference_ppm", 0.0),
                )
            ),
            spectral_bandwidth_ppm=float(ds.attrs.get("spectral_bandwidth_ppm", 20.0)),
            spectral_points=int(ds.attrs.get("spectral_points", 1024)),
            name=str(ds.attrs.get("name", "Dynamic pyruvate/lactate phantom")),
            kinetic_regions=regions,
            pyruvate_inflow=pyruvate_inflow,
            dynamic_b0=dynamic_b0,
            conversion_start_s=float(header.get("conversion_start_s", 0.0)),
            kinetics_time_offset_s=float(header.get("kinetics_time_offset_s", 0.0)),
            metadata=dict(header.get("metadata", {})),
            coordinate_system=str(ds.attrs.get("coordinate_system", "object_xyz")),
            affine_ijk_to_xyz_m=affine,
        )
        return phantom.restore_missing_designed_pools()

    def save(self, filename) -> Path:
        path = Path(filename)
        header = {
            "format": "blochsimulator-dynamic-spectral-phantom",
            "version": 5,
            "shape": self.shape,
            "fov": self.fov,
            "field_strength": self.field_strength,
            "nucleus": self.nucleus,
            "spectral_reference_ppm": self.spectral_reference_ppm,
            "spectral_window_center_ppm": self.spectral_window_center_ppm,
            "spectral_bandwidth_ppm": self.spectral_bandwidth_ppm,
            "spectral_points": self.spectral_points,
            "name": self.name,
            "pools": [pool.__dict__ for pool in self.pools],
            "kinetic_regions": [region.__dict__ for region in self.kinetic_regions],
            "pyruvate_inflow": (
                None
                if self.pyruvate_inflow is None
                else {
                    "rate_curve_s_inv": self.pyruvate_inflow.rate_curve_s_inv.to_dict(),
                    "max_step_s": self.pyruvate_inflow.max_step_s,
                    "polarization_curve": (
                        None
                        if self.pyruvate_inflow.polarization_curve is None
                        else self.pyruvate_inflow.polarization_curve.to_dict()
                    ),
                }
            ),
            "dynamic_b0": (
                None
                if self.dynamic_b0 is None
                else {
                    "offset_curve_hz": self.dynamic_b0.offset_curve_hz.to_dict(),
                    "pool_scale": self.dynamic_b0.pool_scale,
                }
            ),
            "conversion_start_s": self.conversion_start_s,
            "kinetics_time_offset_s": self.kinetics_time_offset_s,
            "equilibrium_polarization": self.equilibrium_polarization,
            "metadata": self.metadata,
            "has_b0_map": self.b0_map is not None,
            "has_b0_map_ppm": self.b0_map_ppm is not None,
            "coordinate_system": self.coordinate_system,
            "affine_ijk_to_xyz_m": self.affine_ijk_to_xyz_m.tolist(),
        }
        arrays = {"kpl_map_s_inv": self.kpl_map_s_inv}
        for index, pool in enumerate(self.pools):
            arrays[f"initial_{index}"] = self.initial_concentration_maps[pool.name]
        if self.initial_spin_density_maps is not None:
            for index, pool in enumerate(self.pools):
                arrays[f"spin_density_{index}"] = self.initial_spin_density_maps[
                    pool.name
                ]
        if self.b0_map is not None:
            arrays["b0_map"] = self.b0_map
        if self.b0_map_ppm is not None:
            arrays["b0_map_ppm"] = self.b0_map_ppm
        if self.pyruvate_inflow is not None:
            arrays["pyruvate_delivery_map"] = self.pyruvate_inflow.delivery_map
            if self.pyruvate_inflow.arrival_delay_map_s is not None:
                arrays["pyruvate_arrival_delay_s"] = (
                    self.pyruvate_inflow.arrival_delay_map_s
                )
        if self.dynamic_b0 is not None:
            arrays["dynamic_b0_scale_map"] = self.dynamic_b0.spatial_scale_map
        header_json = json.dumps(header, default=str)
        if path.suffix.lower() == ".npz":
            np.savez_compressed(path, header_json=np.asarray(header_json), **arrays)
        elif path.suffix.lower() == ".nc":
            self.to_xarray().to_netcdf(path)
        elif path.suffix.lower() in {".h5", ".hdf5"}:
            import h5py

            with h5py.File(path, "w") as handle:
                handle.attrs["header_json"] = header_json
                for key, values in arrays.items():
                    handle.create_dataset(key, data=values)
        else:
            raise ValueError("dynamic phantoms require .npz, .h5, or .hdf5")
        return path

    @classmethod
    def load(cls, filename) -> "DynamicSpectralPhantom":
        path = Path(filename)
        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as data:
                if "header_json" not in data:
                    raise ValueError("not a dynamic spectral phantom")
                header = json.loads(str(data["header_json"].item()))
                arrays = {name: np.asarray(data[name]) for name in data.files}
        elif path.suffix.lower() in {".h5", ".hdf5"}:
            import h5py

            with h5py.File(path, "r") as handle:
                if "header_json" not in handle.attrs:
                    raise ValueError("not a dynamic spectral phantom")
                header = json.loads(handle.attrs["header_json"])
                arrays = {name: handle[name][...] for name in handle.keys()}
        elif path.suffix.lower() == ".nc":
            import xarray as xr

            with xr.open_dataset(path) as dataset:
                return cls.from_xarray(dataset)
        else:
            raise ValueError("unsupported dynamic phantom file")
        if header.get("format") != "blochsimulator-dynamic-spectral-phantom":
            raise ValueError("not a dynamic spectral phantom")
        pools = tuple(ChemicalSpecies(**item) for item in header["pools"])
        regions = tuple(
            KineticRegionDefinition(**item)
            for item in header.get("kinetic_regions", ())
        )
        inflow_metadata = header.get("pyruvate_inflow")
        pyruvate_inflow = None
        if inflow_metadata is not None and "pyruvate_delivery_map" in arrays:
            pyruvate_inflow = PyruvateInflow(
                rate_curve_s_inv=TimeCurve.from_dict(
                    inflow_metadata["rate_curve_s_inv"]
                ),
                delivery_map=arrays["pyruvate_delivery_map"],
                polarization_curve=(
                    None
                    if inflow_metadata.get("polarization_curve") is None
                    else TimeCurve.from_dict(inflow_metadata["polarization_curve"])
                ),
                arrival_delay_map_s=arrays.get("pyruvate_arrival_delay_s"),
                max_step_s=inflow_metadata.get("max_step_s"),
            )
        dynamic_b0_metadata = header.get("dynamic_b0")
        dynamic_b0 = None
        if dynamic_b0_metadata is not None and "dynamic_b0_scale_map" in arrays:
            dynamic_b0 = DynamicB0(
                offset_curve_hz=TimeCurve.from_dict(
                    dynamic_b0_metadata["offset_curve_hz"]
                ),
                spatial_scale_map=arrays["dynamic_b0_scale_map"],
                pool_scale=tuple(dynamic_b0_metadata.get("pool_scale", (1.0, 1.0))),
            )
        phantom = cls(
            shape=tuple(header["shape"]),
            fov=tuple(header["fov"]),
            pools=pools,
            initial_concentration_maps={
                pool.name: arrays[f"initial_{index}"]
                for index, pool in enumerate(pools)
            },
            initial_spin_density_maps=(
                None
                if "spin_density_0" not in arrays
                else {
                    pool.name: arrays[f"spin_density_{index}"]
                    for index, pool in enumerate(pools)
                }
            ),
            equilibrium_polarization=float(header.get("equilibrium_polarization", 0.0)),
            kpl_map_s_inv=arrays["kpl_map_s_inv"],
            b0_map=arrays.get("b0_map"),
            b0_map_ppm=arrays.get("b0_map_ppm"),
            field_strength=header["field_strength"],
            nucleus=header["nucleus"],
            spectral_reference_ppm=float(header.get("spectral_reference_ppm", 0.0)),
            spectral_window_center_ppm=float(
                header.get(
                    "spectral_window_center_ppm",
                    header.get("spectral_reference_ppm", 0.0),
                )
            ),
            spectral_bandwidth_ppm=float(header.get("spectral_bandwidth_ppm", 20.0)),
            spectral_points=int(header.get("spectral_points", 1024)),
            name=header["name"],
            kinetic_regions=regions,
            pyruvate_inflow=pyruvate_inflow,
            dynamic_b0=dynamic_b0,
            conversion_start_s=float(header.get("conversion_start_s", 0.0)),
            kinetics_time_offset_s=float(header.get("kinetics_time_offset_s", 0.0)),
            metadata=header.get("metadata", {}),
            coordinate_system=header.get("coordinate_system", "object_xyz"),
            affine_ijk_to_xyz_m=header.get("affine_ijk_to_xyz_m"),
        )
        return phantom.restore_missing_designed_pools()


def _decay_convolution(rate, duration):
    """Return integrals for a linearly varying source under exponential decay."""
    rate = np.asarray(rate, dtype=float)
    x = rate * duration
    small = np.abs(x) < 1e-5
    f0 = np.empty_like(x, dtype=float)
    f1 = np.empty_like(x, dtype=float)
    f0[small] = duration * (
        1.0 - x[small] / 2.0 + x[small] ** 2 / 6.0 - x[small] ** 3 / 24.0
    )
    f1[small] = duration**2 * (
        0.5 - x[small] / 6.0 + x[small] ** 2 / 24.0 - x[small] ** 3 / 120.0
    )
    regular = ~small
    f0[regular] = duration * (-np.expm1(-x[regular])) / x[regular]
    f1[regular] = duration**2 * (x[regular] + np.expm1(-x[regular])) / x[regular] ** 2
    return f0, f1


def _equal_rate_exchange_convolution(rate, duration):
    """Source-to-product convolution limits for equal precursor/product rates."""
    rate = np.asarray(rate, dtype=float)
    x = rate * duration
    small = np.abs(x) < 1e-4
    j0 = np.empty_like(x, dtype=float)
    j1 = np.empty_like(x, dtype=float)
    j0[small] = duration**2 * (
        0.5 - x[small] / 3.0 + x[small] ** 2 / 8.0 - x[small] ** 3 / 30.0
    )
    j1[small] = duration**3 * (
        1.0 / 6.0 - x[small] / 12.0 + x[small] ** 2 / 40.0 - x[small] ** 3 / 180.0
    )
    regular = ~small
    exp_x = np.exp(-x[regular])
    j0[regular] = duration**2 * (1.0 - exp_x * (1.0 + x[regular])) / x[regular] ** 2
    j1[regular] = (
        duration**3
        * (x[regular] * (1.0 + exp_x) - 2.0 * (1.0 - exp_x))
        / x[regular] ** 3
    )
    return j0, j1


def _zero_target_longitudinal_step(
    state,
    kpl,
    r1_p,
    r1_l,
    duration,
    source_start=None,
    source_end=None,
    prepared=None,
    scratch=None,
    *,
    precursor_loss=None,
    passive_conversion=None,
    passive_r1=None,
):
    if duration == 0:
        return
    if scratch is None:
        pyruvate = state[0, :, 2].copy()
        transfer = np.empty_like(pyruvate)
        regular_mode = 0
        decay_delta = None
    else:
        pyruvate, transfer, decay_delta, regular_mode = scratch
        np.copyto(pyruvate, state[0, :, 2])
    lactate = state[1, :, 2]
    if precursor_loss is None:
        precursor_loss = kpl
    else:
        precursor_loss = np.asarray(precursor_loss, dtype=state.dtype)
    with_source = source_start is not None or source_end is not None
    if prepared is None:
        prepared = _prepare_longitudinal_step(
            kpl,
            r1_p,
            r1_l,
            duration,
            with_source=with_source,
            precursor_loss=precursor_loss,
        )
    exp_a, exp_b, difference, regular, source_coefficients = prepared
    if scratch is None:
        transfer[regular] = (
            kpl[regular]
            * pyruvate[regular]
            * (exp_b - exp_a[regular])
            / difference[regular]
        )
        transfer[~regular] = kpl[~regular] * pyruvate[~regular] * duration * exp_b
        pyruvate_next = pyruvate * exp_a
        lactate_next = lactate * exp_b + transfer
    else:
        np.multiply(kpl, pyruvate, out=transfer)
        if regular_mode == 1:
            np.subtract(exp_b, exp_a, out=decay_delta)
            np.multiply(transfer, decay_delta, out=transfer)
            np.divide(transfer, difference, out=transfer)
        elif regular_mode == -1:
            np.multiply(transfer, duration, out=transfer)
            np.multiply(transfer, exp_b, out=transfer)
        else:
            np.subtract(exp_b, exp_a, out=decay_delta)
            transfer[regular] = (
                transfer[regular] * decay_delta[regular] / difference[regular]
            )
            transfer[~regular] = transfer[~regular] * duration * exp_b
        pyruvate_next = state[0, :, 2]
        np.multiply(pyruvate, exp_a, out=pyruvate_next)
        lactate_next = lactate
        np.multiply(lactate_next, exp_b, out=lactate_next)
        np.add(lactate_next, transfer, out=lactate_next)

    if source_start is not None or source_end is not None:
        if source_start is None:
            source_start = source_end
        if source_end is None:
            source_end = source_start
        source_start = np.asarray(source_start, dtype=state.dtype)
        source_end = np.asarray(source_end, dtype=state.dtype)
        slope = (source_end - source_start) / duration
        if source_coefficients is None:
            raise RuntimeError("prepared longitudinal step lacks inflow coefficients")
        f0_a, f1_a, j0, j1 = source_coefficients
        pyruvate_next += source_start * f0_a + slope * f1_a
        lactate_next += kpl * (source_start * j0 + slope * j1)

    if passive_conversion is not None and state.shape[0] > 2:
        conversion_maps = np.asarray(passive_conversion, dtype=state.dtype)
        passive_rates = np.asarray(passive_r1, dtype=state.dtype).reshape(-1)
        if conversion_maps.shape != (state.shape[0] - 2, state.shape[1]):
            raise ValueError("passive conversion maps must match passive pool state")
        if passive_rates.size != state.shape[0] - 2:
            raise ValueError("passive T1 rates must match passive pool state")
        a = np.asarray(r1_p + precursor_loss, dtype=state.dtype)
        for product_index, (conversion, product_r1) in enumerate(
            zip(conversion_maps, passive_rates), start=2
        ):
            exp_product = np.exp(-product_r1 * duration)
            difference_product = a - product_r1
            regular_product = np.abs(difference_product) > 1e-12
            product_transfer = np.empty_like(pyruvate)
            product_transfer[regular_product] = (
                conversion[regular_product]
                * pyruvate[regular_product]
                * (exp_product - exp_a[regular_product])
                / difference_product[regular_product]
            )
            product_transfer[~regular_product] = (
                conversion[~regular_product]
                * pyruvate[~regular_product]
                * duration
                * exp_product
            )
            product_next = state[product_index, :, 2]
            product_next *= exp_product
            product_next += product_transfer
            if with_source:
                f0_a, f1_a = source_coefficients[:2]
                f0_product, f1_product = _decay_convolution(
                    np.full_like(a, product_r1, dtype=float), duration
                )
                j0_product = np.empty_like(a)
                j1_product = np.empty_like(a)
                separated = np.abs(difference_product * duration) > 1e-7
                j0_product[separated] = (
                    f0_product[separated] - f0_a[separated]
                ) / difference_product[separated]
                j1_product[separated] = (
                    f1_product[separated] - f1_a[separated]
                ) / difference_product[separated]
                if np.any(~separated):
                    equal_j0, equal_j1 = _equal_rate_exchange_convolution(
                        0.5 * (a[~separated] + product_r1), duration
                    )
                    j0_product[~separated] = equal_j0
                    j1_product[~separated] = equal_j1
                product_next += conversion * (
                    source_start * j0_product + slope * j1_product
                )

    state[0, :, 2] = pyruvate_next
    state[1, :, 2] = lactate_next


def _longitudinal_step(
    state,
    kpl,
    r1_p,
    r1_l,
    duration,
    source_start=None,
    source_end=None,
    prepared=None,
    scratch=None,
    *,
    concentration_state=None,
    concentration_source_start=None,
    concentration_source_end=None,
    concentration_prepared=None,
    concentration_scratch=None,
    equilibrium_polarization=0.0,
    passive_r1=None,
    passive_conversion=None,
):
    """Advance total Mz, optionally relaxing polarization toward equilibrium.

    With a concentration state, total magnetization is represented as
    ``Mz = equilibrium_polarization * C + excess``.  The existing exact
    zero-target solver advances the excess, while concentration follows the
    same irreversible P→L exchange without T1 decay.
    """
    if passive_conversion is not None:
        # Cached two-pool coefficients do not include precursor loss into the
        # additional products. Recompute the exact coefficients for this
        # uncommon multi-product path.
        prepared = None
        concentration_prepared = None
        scratch = None
        concentration_scratch = None
    if concentration_state is None:
        _zero_target_longitudinal_step(
            state,
            kpl,
            r1_p,
            r1_l,
            duration,
            source_start,
            source_end,
            prepared,
            scratch,
            precursor_loss=(
                None
                if passive_conversion is None
                else kpl + np.sum(passive_conversion, axis=0)
            ),
            passive_conversion=passive_conversion,
            passive_r1=passive_r1,
        )
        if state.shape[0] > 2 and passive_conversion is None:
            rates = np.asarray(passive_r1, dtype=state.dtype).reshape(-1, 1)
            state[2:, :, 2] *= np.exp(-rates * duration)
        return
    equilibrium = float(equilibrium_polarization)
    if equilibrium != 0.0:
        state[:, :, 2] -= equilibrium * concentration_state[:, :, 2]
    concentration_source_start = (
        0.0 if concentration_source_start is None else concentration_source_start
    )
    concentration_source_end = (
        0.0 if concentration_source_end is None else concentration_source_end
    )
    excess_source_start = (
        None
        if source_start is None
        else source_start - equilibrium * concentration_source_start
    )
    excess_source_end = (
        None
        if source_end is None
        else source_end - equilibrium * concentration_source_end
    )
    _zero_target_longitudinal_step(
        state,
        kpl,
        r1_p,
        r1_l,
        duration,
        excess_source_start,
        excess_source_end,
        prepared,
        scratch,
        precursor_loss=(
            None
            if passive_conversion is None
            else kpl + np.sum(passive_conversion, axis=0)
        ),
        passive_conversion=passive_conversion,
        passive_r1=passive_r1,
    )
    if state.shape[0] > 2 and passive_conversion is None:
        rates = np.asarray(passive_r1, dtype=state.dtype).reshape(-1, 1)
        state[2:, :, 2] *= np.exp(-rates * duration)
    _zero_target_longitudinal_step(
        concentration_state,
        kpl,
        0.0,
        0.0,
        duration,
        concentration_source_start,
        concentration_source_end,
        concentration_prepared,
        concentration_scratch,
        precursor_loss=(
            None
            if passive_conversion is None
            else kpl + np.sum(passive_conversion, axis=0)
        ),
        passive_conversion=passive_conversion,
        passive_r1=np.zeros(state.shape[0] - 2, dtype=state.dtype),
    )
    if equilibrium != 0.0:
        state[:, :, 2] += equilibrium * concentration_state[:, :, 2]


def _prepare_longitudinal_step(
    kpl,
    r1_p,
    r1_l,
    duration,
    *,
    with_source,
    precursor_loss=None,
):
    """Precompute state-independent coefficients for one free half-step."""
    a = r1_p + (kpl if precursor_loss is None else precursor_loss)
    b = r1_l
    exp_a = np.exp(-a * duration)
    exp_b = np.exp(-b * duration)
    difference = a - b
    regular = np.abs(difference) > 1e-12
    source_coefficients = None
    if with_source:
        f0_a, f1_a = _decay_convolution(a, duration)
        f0_b, f1_b = _decay_convolution(np.full_like(a, b, dtype=float), duration)
        j0 = np.empty_like(a)
        j1 = np.empty_like(a)
        rate_separated = np.abs(difference * duration) > 1e-7
        j0[rate_separated] = (f0_b[rate_separated] - f0_a[rate_separated]) / difference[
            rate_separated
        ]
        j1[rate_separated] = (f1_b[rate_separated] - f1_a[rate_separated]) / difference[
            rate_separated
        ]
        if np.any(~rate_separated):
            equal_j0, equal_j1 = _equal_rate_exchange_convolution(
                0.5 * (a[~rate_separated] + b), duration
            )
            j0[~rate_separated] = equal_j0
            j1[~rate_separated] = equal_j1
        source_coefficients = (f0_a, f1_a, j0, j1)
    return exp_a, exp_b, difference, regular, source_coefficients


def _prepare_multi_pool_longitudinal_step(
    conversion_rates,
    r1,
    duration,
    *,
    with_source,
):
    """Prepare exact precursor-to-products coefficients for one half-step."""
    conversion_rates = np.asarray(conversion_rates, dtype=np.float64)
    r1 = np.asarray(r1, dtype=np.float64)
    if conversion_rates.ndim != 2 or conversion_rates.shape[0] != r1.size - 1:
        raise ValueError("conversion rates must contain one row per product pool")
    precursor_rate = r1[0] + np.sum(conversion_rates, axis=0)
    product_rates = r1[1:]
    exp_precursor = np.exp(-precursor_rate * duration)
    exp_products = np.exp(-product_rates * duration)
    difference = precursor_rate[None, :] - product_rates[:, None]
    regular = np.abs(difference) > 1e-12
    transfer_factor = np.empty_like(difference)
    for product in range(product_rates.size):
        transfer_factor[product, regular[product]] = (
            exp_products[product] - exp_precursor[regular[product]]
        ) / difference[product, regular[product]]
        transfer_factor[product, ~regular[product]] = duration * exp_products[product]
    source_coefficients = None
    if with_source:
        f0_precursor, f1_precursor = _decay_convolution(precursor_rate, duration)
        j0 = np.empty_like(difference)
        j1 = np.empty_like(difference)
        for product, product_rate in enumerate(product_rates):
            product_rate_values = np.full_like(
                precursor_rate, product_rate, dtype=np.float64
            )
            f0_product, f1_product = _decay_convolution(product_rate_values, duration)
            separated = np.abs(difference[product] * duration) > 1e-7
            j0[product, separated] = (
                f0_product[separated] - f0_precursor[separated]
            ) / difference[product, separated]
            j1[product, separated] = (
                f1_product[separated] - f1_precursor[separated]
            ) / difference[product, separated]
            if np.any(~separated):
                equal_j0, equal_j1 = _equal_rate_exchange_convolution(
                    0.5
                    * (precursor_rate[~separated] + product_rate_values[~separated]),
                    duration,
                )
                j0[product, ~separated] = equal_j0
                j1[product, ~separated] = equal_j1
        source_coefficients = (f0_precursor, f1_precursor, j0, j1)
    return (
        exp_precursor,
        exp_products,
        transfer_factor,
        source_coefficients,
    )


def kinetic_preroll_start_s(
    inflow_curve: Optional[TimeCurve],
    conversion_start_s: float,
    kinetics_time_offset_s: float = 0.0,
    arrival_delay_s=None,
) -> float:
    """Return the earliest sequence-relative free-kinetics pre-roll time."""
    conversion_start_s = float(conversion_start_s)
    kinetics_time_offset_s = float(kinetics_time_offset_s)
    if not np.isfinite(conversion_start_s):
        raise ValueError("conversion start time must be finite")
    if not np.isfinite(kinetics_time_offset_s):
        raise ValueError("kinetics time offset must be finite")
    candidates = [0.0, conversion_start_s - kinetics_time_offset_s]
    if inflow_curve is not None:
        earliest_delay = 0.0
        if arrival_delay_s is not None:
            delays = np.asarray(arrival_delay_s, dtype=float)
            finite = delays[np.isfinite(delays)]
            if finite.size:
                earliest_delay = float(np.min(finite))
        candidates.append(
            float(inflow_curve.times_s[0]) + earliest_delay - kinetics_time_offset_s
        )
    return min(candidates)


def _advance_longitudinal_kinetics(
    state,
    kpl,
    r1,
    start_s,
    end_s,
    *,
    inflow_curve: Optional[TimeCurve],
    inflow_delivery,
    conversion_start_s,
    inflow_polarization_curve: Optional[TimeCurve] = None,
    concentration_state=None,
    equilibrium_polarization=0.0,
    inflow_arrival_delay_s=None,
    inflow_max_step_s=None,
    passive_conversion=None,
):
    if end_s < start_s:
        raise ValueError("kinetics interval end must not precede its start")
    internal_knots = []
    if inflow_curve is not None:
        internal_knots.extend(
            knot for knot in inflow_curve.times_s if start_s < knot < end_s
        )
    if inflow_polarization_curve is not None:
        internal_knots.extend(
            knot for knot in inflow_polarization_curve.times_s if start_s < knot < end_s
        )
    if start_s < conversion_start_s < end_s:
        internal_knots.append(float(conversion_start_s))
    if inflow_arrival_delay_s is not None and inflow_max_step_s is not None:
        step_count = int(np.ceil((end_s - start_s) / inflow_max_step_s))
        internal_knots.extend(
            start_s + index * inflow_max_step_s for index in range(1, step_count)
        )
    boundaries = (float(start_s), *sorted(set(internal_knots)), float(end_s))
    zero_kpl = np.zeros_like(kpl)
    zero_passive_conversion = (
        None if passive_conversion is None else np.zeros_like(passive_conversion)
    )
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end == start:
            continue
        interval_kpl = kpl if (start + end) / 2.0 >= conversion_start_s else zero_kpl
        interval_passive_conversion = (
            passive_conversion
            if (start + end) / 2.0 >= conversion_start_s
            else zero_passive_conversion
        )
        if inflow_curve is None:
            source_start = source_end = None
            concentration_source_start = concentration_source_end = None
        else:
            start_value, end_value = _time_curve_interval_values(
                inflow_curve,
                start,
                end,
                inflow_arrival_delay_s,
            )
            if inflow_polarization_curve is None:
                source_start = inflow_delivery * start_value
                source_end = inflow_delivery * end_value
                concentration_source_start = concentration_source_end = None
            else:
                polarization_start, polarization_end = _time_curve_interval_values(
                    inflow_polarization_curve,
                    start,
                    end,
                    inflow_arrival_delay_s,
                )
                concentration_source_start = inflow_delivery * start_value
                concentration_source_end = inflow_delivery * end_value
                source_start = concentration_source_start * polarization_start
                source_end = concentration_source_end * polarization_end
        _longitudinal_step(
            state,
            interval_kpl,
            r1[0],
            r1[1],
            end - start,
            source_start,
            source_end,
            concentration_state=concentration_state,
            concentration_source_start=concentration_source_start,
            concentration_source_end=concentration_source_end,
            equilibrium_polarization=equilibrium_polarization,
            passive_r1=r1[2:],
            passive_conversion=interval_passive_conversion,
        )


def simulate_two_pool_kinetics(
    times_s,
    initial_mz,
    t1_s,
    kpl_s_inv,
    inflow_curve: Optional[TimeCurve] = None,
    conversion_start_s: float = 0.0,
    initial_time_s: float = 0.0,
    kinetics_time_offset_s: float = 0.0,
    initial_concentration=None,
    inflow_polarization_curve: Optional[TimeCurve] = None,
    equilibrium_polarization: float = 0.0,
    return_concentration: bool = False,
):
    """Evaluate free pyruvate/lactate kinetics at sequence-relative times.

    This is the longitudinal part of the dynamic sequence solver without RF or
    gradients. Inflow is integrated piecewise-exactly using the same solver as
    :func:`simulate_dynamic_sequence`. The returned array has shape
    ``(2, n_times)`` in pyruvate/lactate order. Inflow and conversion times are
    on the shared kinetics timeline; ``kinetics_time_offset_s`` selects its time
    at sequence ``t=0``.
    """
    times = np.asarray(times_s, dtype=float)
    initial = np.asarray(initial_mz, dtype=float)
    relaxation = np.asarray(t1_s, dtype=float)
    kpl = float(kpl_s_inv)
    if times.ndim != 1 or not times.size:
        raise ValueError("kinetics preview times must be a non-empty 1D array")
    if not np.all(np.isfinite(times)):
        raise ValueError("kinetics preview times must be finite")
    if np.any(np.diff(times) < 0):
        raise ValueError("kinetics preview times must be sorted")
    if initial.shape != (2,) or not np.all(np.isfinite(initial)):
        raise ValueError("initial pyruvate/lactate Mz must contain two finite values")
    if relaxation.shape != (2,) or not np.all(np.isfinite(relaxation)):
        raise ValueError("pyruvate/lactate T1 must contain two finite values")
    if np.any(relaxation <= 0):
        raise ValueError("pyruvate/lactate T1 must be positive")
    if not np.isfinite(kpl) or kpl < 0:
        raise ValueError("kPL must be finite and non-negative")
    conversion_start_s = float(conversion_start_s)
    initial_time_s = float(initial_time_s)
    kinetics_time_offset_s = float(kinetics_time_offset_s)
    equilibrium_polarization = float(equilibrium_polarization)
    if not np.isfinite(conversion_start_s):
        raise ValueError("conversion start time must be finite")
    if not np.isfinite(kinetics_time_offset_s):
        raise ValueError("kinetics time offset must be finite")
    if not np.isfinite(equilibrium_polarization) or equilibrium_polarization < 0:
        raise ValueError("equilibrium polarization must be finite and non-negative")
    if not np.isfinite(initial_time_s) or initial_time_s > times[0]:
        raise ValueError(
            "initial kinetics time must be finite and not exceed the first sample"
        )

    sequence_inflow_curve = (
        None if inflow_curve is None else inflow_curve.shifted(-kinetics_time_offset_s)
    )
    sequence_inflow_polarization_curve = (
        None
        if inflow_polarization_curve is None
        else inflow_polarization_curve.shifted(-kinetics_time_offset_s)
    )
    sequence_conversion_start_s = conversion_start_s - kinetics_time_offset_s
    state = np.zeros((2, 1, 3), dtype=float)
    state[:, 0, 2] = initial
    concentration_state = None
    if initial_concentration is not None:
        initial_concentration = np.asarray(initial_concentration, dtype=float)
        if initial_concentration.shape != (2,) or np.any(initial_concentration < 0):
            raise ValueError(
                "initial concentration must contain two non-negative values"
            )
        concentration_state = np.zeros_like(state)
        concentration_state[:, 0, 2] = initial_concentration
    result = np.empty((2, times.size), dtype=float)
    concentration_result = (
        np.empty_like(result) if concentration_state is not None else None
    )
    kpl_array = np.asarray([kpl], dtype=float)
    r1 = 1.0 / relaxation
    inflow_delivery = np.ones(1, dtype=float)
    current_time = initial_time_s
    for index, target_time in enumerate(times):
        _advance_longitudinal_kinetics(
            state,
            kpl_array,
            r1,
            current_time,
            float(target_time),
            inflow_curve=sequence_inflow_curve,
            inflow_delivery=inflow_delivery,
            conversion_start_s=sequence_conversion_start_s,
            inflow_polarization_curve=sequence_inflow_polarization_curve,
            concentration_state=concentration_state,
            equilibrium_polarization=equilibrium_polarization,
        )
        result[:, index] = state[:, 0, 2]
        if concentration_result is not None:
            concentration_result[:, index] = concentration_state[:, 0, 2]
        current_time = float(target_time)
    if return_concentration:
        if concentration_result is None:
            concentration_result = np.zeros_like(result)
        return result, concentration_result
    return result


def _free_step(
    state,
    phase_cycles,
    t2,
    kpl,
    r1,
    duration,
    source_start=None,
    source_end=None,
    transverse_factors=None,
    longitudinal_prepared=None,
    transverse_state=None,
    longitudinal_scratch=None,
    concentration_state=None,
    concentration_source_start=None,
    concentration_source_end=None,
    concentration_longitudinal_prepared=None,
    concentration_longitudinal_scratch=None,
    equilibrium_polarization=0.0,
    passive_conversion=None,
):
    if duration == 0:
        return
    for pool in range(state.shape[0]):
        transverse = (
            state[pool, :, 0] + 1j * state[pool, :, 1]
            if transverse_state is None
            else transverse_state[pool]
        )
        factor = (
            np.exp(-duration / t2[pool] - 2j * np.pi * phase_cycles[pool])
            if transverse_factors is None
            else transverse_factors[pool]
        )
        transverse *= factor
        if transverse_state is None:
            state[pool, :, 0] = transverse.real
            state[pool, :, 1] = transverse.imag
    _longitudinal_step(
        state,
        kpl,
        r1[0],
        r1[1],
        duration,
        source_start=source_start,
        source_end=source_end,
        prepared=longitudinal_prepared,
        scratch=longitudinal_scratch,
        concentration_state=concentration_state,
        concentration_source_start=concentration_source_start,
        concentration_source_end=concentration_source_end,
        concentration_prepared=concentration_longitudinal_prepared,
        concentration_scratch=concentration_longitudinal_scratch,
        equilibrium_polarization=equilibrium_polarization,
        passive_r1=r1[2:],
        passive_conversion=passive_conversion,
    )


def _prepare_transverse_factors(phase_cycles, t2, duration):
    """Evaluate the state-independent transverse factors once."""
    return tuple(
        np.exp(-duration / t2[pool] - 2j * np.pi * phase_cycles[pool])
        for pool in range(len(t2))
    )


def _prepare_rf_rotation(rf_hz, duration):
    """Return the already-rounded scalar coefficients for one RF rotation."""
    nx = -2 * np.pi * rf_hz.real * duration
    ny = 2 * np.pi * rf_hz.imag * duration
    angle = float(np.hypot(nx, ny))
    if angle == 0:
        return None
    axis = np.asarray([nx / angle, ny / angle, 0.0])
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return (
        float(axis[0]),
        float(axis[1]),
        float(cosine),
        float(sine),
        float(1.0 - cosine),
    )


_SPIN_MAJOR_SLICE_INTERVALS = 16384
_SPIN_MAJOR_SLICE_ADC_VALUES = 4096
_SPIN_MAJOR_WORK_BLOCKS = 64
_SPIN_MAJOR_RF_CACHE_ENTRIES = 65536
# Conservative working-set terms of one slice: per-thread coefficient slots
# and tile state, interval tables, and per-block ADC accumulators. Each pool
# beyond two adds coefficient slot fields and tile state; delayed inflow adds
# the interval times.
_SPIN_MAJOR_THREAD_BYTES = 480 * 1024
_SPIN_MAJOR_POOL_THREAD_BYTES = 160 * 1024
_SPIN_MAJOR_INTERVAL_BYTES = 256
_SPIN_MAJOR_SPATIAL_INTERVAL_BYTES = 32
_SPIN_MAJOR_MAX_POOLS = 6
# Repeated RF segments run as cached per-spin affine propagators. Entries are
# planned per block, kept across slices, and evicted in LRU order.
_SPIN_MAJOR_RF_PROPAGATOR_ENTRIES = 8
_SPIN_MAJOR_RF_PROPAGATOR_MIN_OCCURRENCES = 16
_SPIN_MAJOR_RF_PROPAGATOR_SOURCE_TOLERANCE = 1e-12
# Segments whose RF differs only by a constant phase share one entry, applied
# as R_z(t) A R_z(-t). The phase-rotated RF must match the occurrence's own RF
# to this many float64 epsilons of the pulse amplitude.
_SPIN_MAJOR_RF_PROPAGATOR_PHASE_EPSILONS = 64


def _spin_major_requested(value):
    """Resolve the private spin-major switch; the environment is the default."""
    if value is not None:
        return bool(value)
    setting = os.environ.get("BLOCHSIM_DYNAMIC_SPIN_MAJOR", "").strip().lower()
    return setting not in {"0", "false", "off", "no"}


def _spin_major_rf_cache_requested(value):
    """Resolve the private RF propagator cache switch; on by default."""
    if value is not None:
        return bool(value)
    setting = os.environ.get("BLOCHSIM_DYNAMIC_SPIN_MAJOR_RF_CACHE", "").strip().lower()
    return setting not in {"0", "false", "off", "no"}


def _spin_major_kernel():
    """Return the spin-major executor and its tile width, or None."""
    try:
        from .dynamic_spin_major_cy import API_VERSION, TILE_SPINS, run_spin_major_chunk
    except (ImportError, OSError):
        return None
    if API_VERSION != 4:
        return None
    return run_spin_major_chunk, int(TILE_SPINS)


def _spin_major_fallback_reason(
    *,
    requested,
    available,
    phantom,
    real_dtype,
    checkpoint_dtype,
    spatial_tx,
    execution_plan,
):
    """Return why the spin-major executor cannot run a plan, or None.

    The executor covers float64 plans of two to six pools with per-spin B0 and
    conversion-rate maps, including inflow with per-spin arrival delays.
    Everything else keeps the existing native paths.
    """
    if not requested:
        return "disabled by request"
    if not available:
        return "extension is unavailable"
    if phantom.n_species > _SPIN_MAJOR_MAX_POOLS:
        return "more than six pools are unsupported"
    if np.dtype(real_dtype) != np.dtype(np.float64) or np.dtype(
        checkpoint_dtype
    ) != np.dtype(np.float64):
        return "requires float64 state and checkpoints"
    if phantom.dynamic_b0 is not None:
        return "dynamic B0 is unsupported"
    if spatial_tx:
        try:
            from .dynamic_spin_major_cy import SPATIAL_TX_SUPPORTED
        except ImportError:
            return "the installed extension does not support spatial Tx sensitivity"
        if not SPATIAL_TX_SUPPORTED:
            return "the installed extension does not support spatial Tx sensitivity"
    if execution_plan.n_intervals == 0:
        return "the plan has no intervals"
    if not np.all(execution_plan.dt_s > 0.0):
        return "the plan has empty intervals"
    adc_actions = execution_plan.action_kinds == ACTION_ADC
    if np.any(execution_plan.action_state_indices[adc_actions] == 0):
        return "ADC at sequence time zero"
    return None


def _spin_major_slice_limits(
    memory_budget_bytes, n_rx_coils, n_threads, n_pools=2, spatial_inflow=False
):
    """Return (interval, ADC) limits per slice, or None if the budget is too small.

    Slice sizes do not change results: state stays float64 between slices and
    every ADC value is reduced from the same fixed work blocks.
    """
    interval_limit = _SPIN_MAJOR_SLICE_INTERVALS
    adc_limit = max(1, _SPIN_MAJOR_SLICE_ADC_VALUES // n_rx_coils)
    if memory_budget_bytes is None:
        return interval_limit, adc_limit
    extra_pools = max(0, int(n_pools) - 2)
    work_bytes = int(memory_budget_bytes) // 8 - n_threads * (
        _SPIN_MAJOR_THREAD_BYTES
        + extra_pools * _SPIN_MAJOR_POOL_THREAD_BYTES
        + n_rx_coils * 2 * 64 * 8
    )
    adc_value_bytes = (
        16 * max(2, int(n_pools)) * n_rx_coils * (_SPIN_MAJOR_WORK_BLOCKS + 1)
    )
    interval_bytes = _SPIN_MAJOR_INTERVAL_BYTES + (
        _SPIN_MAJOR_SPATIAL_INTERVAL_BYTES if spatial_inflow else 0
    )
    interval_limit = min(interval_limit, work_bytes // 2 // interval_bytes)
    adc_limit = min(adc_limit, work_bytes // 2 // adc_value_bytes)
    if interval_limit < 1 or adc_limit < 1:
        return None
    return interval_limit, adc_limit


def _spin_major_slice_end(
    first_interval,
    interval_count,
    adc_states,
    adc_limit,
    interval_limit,
    segment_first=None,
    segment_end=None,
):
    """Return a slice end that bounds both interval and ADC counts.

    Cached RF segments are never split: the end moves before a segment it
    would cut, or past it when the slice starts there. Segments have no inner
    actions, so the interval limit can be exceeded by one segment length and
    the ADC limit only by ADC values at one state, which no slice splits.
    """
    end_interval = min(interval_count, first_interval + interval_limit)
    adc_index = int(np.searchsorted(adc_states, first_interval + 1, side="left"))
    if adc_index + adc_limit < adc_states.size:
        # The interval ending at state s is s - 1; stop before it.
        limit_state = int(adc_states[adc_index + adc_limit])
        end_interval = min(end_interval, max(first_interval + 1, limit_state - 1))
    if segment_first is not None and segment_first.size:
        index = int(np.searchsorted(segment_first, end_interval, side="right")) - 1
        if index >= 0 and segment_first[index] < end_interval < segment_end[index]:
            if segment_first[index] > first_interval:
                end_interval = int(segment_first[index])
            else:
                end_interval = int(segment_end[index])
    return end_interval


@dataclass(frozen=True)
class _SpinMajorRfPropagatorPlan:
    """Cached RF segment occurrences of one block in plan interval order.

    Occurrence ``k`` replaces intervals ``[first[k], end[k])`` by propagator
    entry ``entry[k]`` with source weights ``sources[k]``. A nonnegative
    ``basis[k]`` builds that entry first from the source basis rows starting
    at that row of ``basis_rows``. ``phase[k]`` holds (cos t, sin t) of the
    rotation R_z(t) A R_z(-t) that maps the entry's build to occurrence ``k``.
    """

    first: np.ndarray
    end: np.ndarray
    entry: np.ndarray
    basis: np.ndarray
    sources: np.ndarray
    phase: np.ndarray
    basis_rows: np.ndarray
    capacity: int
    keys: int
    builds: int
    rotated: int


def _spin_major_source_samples(execution_plan, index, *, source_mode, equilibrium):
    """Return excess-magnetization and concentration source samples."""
    rate = execution_plan.source_rate_samples[index]
    polarization = execution_plan.source_polarization_samples
    polarization = np.ones_like(rate) if polarization is None else polarization[index]
    magnetization = rate if source_mode == 1 else rate * polarization
    concentration = rate if source_mode == 0 else np.zeros_like(rate)
    return magnetization - equilibrium * concentration, concentration


def _spin_major_source_basis(phase, half):
    """Return quadratic source shapes of one RF segment, or None.

    ``phase`` holds (mid - start, end - mid) per interval. The Lagrange nodes
    are the segment start, the midpoint of its middle interval, and its end,
    so the weights of a source are its samples there. Returns the shapes at
    the (start, mid, mid, end) sample times, shape (3, m, 4), and basis rows
    (m, 3, 4) built from them like the stepwise source rows.
    """
    count = phase.shape[0]
    start = np.zeros(count, dtype=np.float64)
    start[1:] = np.cumsum(phase[:, 0] + phase[:, 1])[:-1]
    middle = start + phase[:, 0]
    end = middle + phase[:, 1]
    nodes = (0.0, float(middle[count // 2]), float(end[-1]))
    if not 0.0 < nodes[1] < nodes[2]:
        return None
    times = np.stack((start, middle, middle, end), axis=1)
    shapes = np.empty((3, count, 4), dtype=np.float64)
    for node in range(3):
        others = [nodes[other] for other in range(3) if other != node]
        shapes[node] = (
            (times - others[0])
            * (times - others[1])
            / ((nodes[node] - others[0]) * (nodes[node] - others[1]))
        )
    rows = np.empty((count, 3, 4), dtype=np.float64)
    for half_index in range(2):
        first = 2 * half_index
        rows[:, :, first] = shapes[:, :, first].T
        rows[:, :, first + 1] = (
            (shapes[:, :, first + 1] - shapes[:, :, first]) / half
        ).T
    return shapes, rows


def _spin_major_source_weights(samples, shapes):
    """Return node weights and whether the shapes reproduce the samples."""
    count = samples.shape[1]
    weights = np.stack(
        (samples[:, 0, 0], samples[:, count // 2, 1], samples[:, -1, 3]), axis=1
    )
    residual = np.abs(np.einsum("oj,jmk->omk", weights, shapes) - samples)
    scale = np.abs(samples).max(axis=(1, 2))
    return weights, residual.max(axis=(1, 2)) <= (
        _SPIN_MAJOR_RF_PROPAGATOR_SOURCE_TOLERANCE * scale
    )


def _spin_major_rf_lru(occurrence_keys, capacity):
    """Assign LRU propagator entries; return entries and build flags."""
    resident = OrderedDict()
    entries = np.empty(len(occurrence_keys), dtype=np.int32)
    builds = np.zeros(len(occurrence_keys), dtype=bool)
    for position, key in enumerate(occurrence_keys):
        entry = resident.get(key)
        if entry is None:
            entry = (
                len(resident)
                if len(resident) < capacity
                else resident.popitem(last=False)[1]
            )
            resident[key] = entry
            builds[position] = True
        else:
            resident.move_to_end(key)
        entries[position] = entry
    return entries, builds


def _spin_major_rf_phasor(rf, reference):
    """Return the unit phasor u with rf == u * reference, or None.

    The match must hold to _SPIN_MAJOR_RF_PROPAGATOR_PHASE_EPSILONS float64
    epsilons of the reference amplitude in every interval, so a propagator
    built for ``reference`` and rotated by the phase of u reproduces ``rf`` to
    rounding.
    """
    scale = float(np.abs(reference).max())
    if scale == 0.0:
        return 1.0 + 0.0j if not np.any(rf) else None
    overlap = complex(np.sum(rf * np.conj(reference)))
    magnitude = abs(overlap)
    if magnitude == 0.0:
        return None
    phasor = overlap / magnitude
    tolerance = (
        _SPIN_MAJOR_RF_PROPAGATOR_PHASE_EPSILONS * np.finfo(np.float64).eps * scale
    )
    if float(np.abs(rf - phasor * reference).max()) > tolerance:
        return None
    return phasor


def _spin_major_rf_segment_keys(execution_plan, native_rf_hz, firsts, ends):
    """Return phase-free segment keys.

    Segments share a key when every per-interval input of the stepwise code
    but the RF is bitwise equal and their RF differs only by a constant phase,
    checked against the first segment of the key.
    """
    dt = execution_plan.dt_s
    half = execution_plan.interval_half_s
    gradient = execution_plan.gradient_hz_per_m
    conversion = execution_plan.conversion_active
    representatives = {}
    segment_key = np.empty(firsts.size, dtype=np.int64)
    keys = 0
    for index, (first, end) in enumerate(zip(firsts.tolist(), ends.tolist())):
        rf = native_rf_hz[first:end]
        energy = float(np.sum(rf.real * rf.real + rf.imag * rf.imag))
        # The energy bucket only narrows the phase search; a near-boundary
        # rounding split costs a cache hit, never accuracy.
        group = b"".join(
            (
                np.float64(float(f"{energy:.9e}")).tobytes(),
                dt[first:end].tobytes(),
                half[first:end].tobytes(),
                gradient[first:end].tobytes(),
                conversion[first:end].tobytes(),
            )
        )
        candidates = representatives.setdefault(group, [])
        for key, reference in candidates:
            if _spin_major_rf_phasor(rf, reference) is not None:
                segment_key[index] = key
                break
        else:
            candidates.append((keys, rf))
            segment_key[index] = keys
            keys += 1
    return segment_key


def _spin_major_rf_propagator_plan(
    execution_plan,
    native_rf_hz,
    *,
    source_mode,
    equilibrium,
    track_concentration,
    columns,
    entry_bytes,
    budget_bytes,
    spatial_sources=False,
):
    """Plan cached RF segment propagators of one block.

    Returns the plan or None, and the fallback reason. Segments are keyed by
    every per-interval input of the stepwise code up to a constant RF phase,
    so equal keys give propagators per spin that are exact z rotations of
    each other wherever they are built. Occurrences whose global source
    samples do not follow the quadratic source basis, or whose RF phase does
    not match the entry's build to rounding, run stepwise. With spatial
    sources the executor evaluates the source weights per spin.
    """
    rf_segments = execution_plan.segment_kinds == SEGMENT_RF
    firsts = execution_plan.segment_start_intervals[rf_segments]
    ends = execution_plan.segment_end_intervals[rf_segments]
    if firsts.size == 0:
        return None, "no repeated RF segments"
    action_states = execution_plan.action_state_indices
    eligible = np.searchsorted(action_states, ends, side="left") == np.searchsorted(
        action_states, firsts + 1, side="left"
    )

    dt = execution_plan.dt_s
    half = execution_plan.interval_half_s
    segment_key = _spin_major_rf_segment_keys(
        execution_plan, native_rf_hz, firsts, ends
    )

    sources = np.zeros((firsts.size, 6), dtype=np.float64)
    cached = np.zeros(firsts.size, dtype=bool)
    key_rows = {}
    order = np.argsort(segment_key, kind="stable")
    boundaries = np.flatnonzero(np.diff(segment_key[order])) + 1
    for members in np.split(order, boundaries):
        members = members[eligible[members]]
        # One build advances every propagator column through the segment, so
        # a key needs more occurrences than columns to pay off.
        minimum = max(_SPIN_MAJOR_RF_PROPAGATOR_MIN_OCCURRENCES, columns + 1)
        if members.size < minimum:
            continue
        first = int(firsts[members[0]])
        end = int(ends[members[0]])
        phase = np.ascontiguousarray(half[first:end])
        basis = _spin_major_source_basis(phase, dt[first:end] / 2.0)
        if basis is None:
            continue
        shapes, rows = basis
        if source_mode != 2 and not spatial_sources:
            intervals = firsts[members][:, None] + np.arange(end - first)
            excess, concentration = _spin_major_source_samples(
                execution_plan,
                intervals,
                source_mode=source_mode,
                equilibrium=equilibrium,
            )
            excess_weights, fits = _spin_major_source_weights(excess, shapes)
            sources[members, 0:3] = excess_weights
            if source_mode == 0 and track_concentration:
                concentration_weights, concentration_fits = _spin_major_source_weights(
                    concentration, shapes
                )
                sources[members, 3:6] = concentration_weights
                fits &= concentration_fits
            members = members[fits]
            if members.size < minimum:
                continue
        cached[members] = True
        key_rows[int(segment_key[members[0]])] = rows
    positions = np.flatnonzero(cached)
    if positions.size == 0:
        return None, "no repeated RF segments"

    occurrence_keys = segment_key[positions].tolist()
    limit = min(_SPIN_MAJOR_RF_PROPAGATOR_ENTRIES, len(key_rows))
    entries, builds = _spin_major_rf_lru(occurrence_keys, limit)
    capacity = limit
    # The smallest capacity with the same number of builds saves memory.
    for candidate in range(1, limit):
        candidate_entries, candidate_builds = _spin_major_rf_lru(
            occurrence_keys, candidate
        )
        if np.count_nonzero(candidate_builds) == np.count_nonzero(builds):
            capacity, entries, builds = candidate, candidate_entries, candidate_builds
            break
    if budget_bytes is not None and capacity * entry_bytes > budget_bytes:
        capacity = int(budget_bytes) // entry_bytes
        if capacity < 1:
            return None, "memory budget is too small"
        entries, builds = _spin_major_rf_lru(occurrence_keys, capacity)

    # Each application rotates the build of its entry by the phase between
    # their RF. Hits whose RF does not match that build to rounding run
    # stepwise; the entry keeps its build either way.
    phase = np.empty((positions.size, 2), dtype=np.float64)
    phase[:, 0] = 1.0
    phase[:, 1] = 0.0
    keep = np.ones(positions.size, dtype=bool)
    builder = {}
    for position, segment in enumerate(positions.tolist()):
        rf = native_rf_hz[firsts[segment] : ends[segment]]
        entry = int(entries[position])
        if builds[position]:
            builder[entry] = rf
            continue
        phasor = _spin_major_rf_phasor(rf, builder[entry])
        if phasor is None:
            keep[position] = False
        elif phasor != 1.0:
            phase[position] = (phasor.real, -phasor.imag)
    positions = positions[keep]
    entries = entries[keep]
    builds = builds[keep]
    phase = phase[keep]

    # A build advances every propagator column through the segment; it must
    # cost less than the stepwise work the cached occurrences replace.
    lengths = ends[positions] - firsts[positions]
    build_steps = columns * int(lengths[builds].sum())
    if build_steps + positions.size >= int(lengths.sum()):
        return None, "RF segments repeat too rarely for the cache capacity"

    row_offsets = {}
    offset = 0
    for key, rows in key_rows.items():
        row_offsets[key] = offset
        offset += rows.shape[0]
    basis = np.full(positions.size, -1, dtype=np.int64)
    basis[builds] = [row_offsets[key] for key in segment_key[positions][builds]]
    rotated = int(np.count_nonzero((phase[:, 0] != 1.0) | (phase[:, 1] != 0.0)))
    return (
        _SpinMajorRfPropagatorPlan(
            first=np.ascontiguousarray(firsts[positions], dtype=np.int64),
            end=np.ascontiguousarray(ends[positions], dtype=np.int64),
            entry=np.ascontiguousarray(entries),
            basis=basis,
            sources=np.ascontiguousarray(sources[positions]),
            phase=np.ascontiguousarray(phase),
            basis_rows=np.ascontiguousarray(np.concatenate(list(key_rows.values()))),
            capacity=capacity,
            keys=len(key_rows),
            builds=int(np.count_nonzero(builds)),
            rotated=rotated,
        ),
        None,
    )


def _spin_major_rf_slice(plan, propagators, first_interval, end_interval):
    """Return the cached-segment arguments of one plan slice."""
    low, high = np.searchsorted(plan.first, (first_interval, end_interval))
    return {
        "segment_first": np.ascontiguousarray(
            plan.first[low:high] - first_interval, dtype=np.int32
        ),
        "segment_end": np.ascontiguousarray(
            plan.end[low:high] - first_interval, dtype=np.int32
        ),
        "segment_entry": plan.entry[low:high],
        "segment_basis": plan.basis[low:high],
        "segment_sources": plan.sources[low:high],
        "segment_phase": plan.phase[low:high],
        "basis_rows": plan.basis_rows,
        "propagators": propagators,
    }


def _spin_major_plan_slice(
    execution_plan,
    native_rf_hz,
    first_interval,
    end_interval,
    *,
    source_mode,
    equilibrium,
    rf_cache,
    spatial_inflow=False,
):
    """Build the compact interval tables of one spin-major plan slice.

    Free-evolution coefficients are keyed by (half step, conversion active)
    and evaluated per spin inside the executor. RF rotations and source rates
    do not depend on the spin, so they are prepared here once per value.
    Delayed inflow is evaluated per spin from the interval times instead.
    """
    interval_slice = slice(first_interval, end_interval)
    interval_count = end_interval - first_interval
    dt = execution_plan.dt_s[interval_slice]
    half = dt / 2.0
    group_keys = np.column_stack(
        (half, execution_plan.conversion_active[interval_slice].astype(np.float64))
    )
    unique_groups, interval_group = np.unique(group_keys, axis=0, return_inverse=True)

    rf_values = native_rf_hz[interval_slice]
    interval_rf = np.full(interval_count, -1, dtype=np.int32)
    rf_rows = []
    rf_input_rows = []
    rf_mask = rf_values != 0.0
    if np.any(rf_mask):
        rf_keys = np.column_stack(
            (rf_values.real[rf_mask], rf_values.imag[rf_mask], dt[rf_mask])
        )
        unique_rf, rf_inverse = np.unique(rf_keys, axis=0, return_inverse=True)
        row_by_unique = np.empty(unique_rf.shape[0], dtype=np.int32)
        for index, (real, imag, duration) in enumerate(unique_rf):
            key = (float(real), float(imag), float(duration))
            if key not in rf_cache:
                if len(rf_cache) >= _SPIN_MAJOR_RF_CACHE_ENTRIES:
                    rf_cache.clear()
                rf_cache[key] = _prepare_rf_rotation(complex(key[0], key[1]), key[2])
            prepared = rf_cache[key]
            if prepared is None:
                row_by_unique[index] = -1
            else:
                row_by_unique[index] = len(rf_rows)
                rf_rows.append(prepared)
                rf_input_rows.append((real, imag, duration))
        interval_rf[rf_mask] = row_by_unique[rf_inverse.reshape(-1)]
    rf_table = np.asarray(rf_rows, dtype=np.float64).reshape(-1, 5)

    gradients = execution_plan.gradient_hz_per_m[interval_slice]
    gradient_mask = np.any(gradients != 0.0, axis=1)
    interval_gradient = np.full(interval_count, -1, dtype=np.int32)
    interval_gradient[gradient_mask] = np.arange(
        np.count_nonzero(gradient_mask), dtype=np.int32
    )
    gradient_table = np.ascontiguousarray(gradients[gradient_mask], dtype=np.float64)

    interval_source = np.full(interval_count, -1, dtype=np.int32)
    source_table = np.zeros((0, 8), dtype=np.float64)
    if source_mode != 2 and not spatial_inflow:
        excess, concentration = _spin_major_source_samples(
            execution_plan,
            interval_slice,
            source_mode=source_mode,
            equilibrium=equilibrium,
        )
        rows = np.empty((interval_count, 8), dtype=np.float64)
        for half_index in range(2):
            start = 2 * half_index
            column = 4 * half_index
            rows[:, column] = excess[:, start]
            rows[:, column + 1] = (excess[:, start + 1] - excess[:, start]) / half
            rows[:, column + 2] = concentration[:, start]
            rows[:, column + 3] = (
                concentration[:, start + 1] - concentration[:, start]
            ) / half
        source_mask = np.any(rows != 0.0, axis=1)
        interval_source[source_mask] = np.arange(
            np.count_nonzero(source_mask), dtype=np.int32
        )
        source_table = np.ascontiguousarray(rows[source_mask])

    phase = np.ascontiguousarray(
        execution_plan.interval_half_s[interval_slice], dtype=np.float64
    )

    action_offsets = np.searchsorted(
        execution_plan.action_state_indices,
        np.arange(first_interval + 1, end_interval + 2, dtype=np.int64),
        side="left",
    ).astype(np.int64)
    action_slice = slice(int(action_offsets[0]), int(action_offsets[-1]))
    adc_payloads = execution_plan.action_payload_indices[action_slice][
        execution_plan.action_kinds[action_slice] == ACTION_ADC
    ]
    adc_first = int(adc_payloads.min()) if adc_payloads.size else 0
    adc_count = int(adc_payloads.max()) - adc_first + 1 if adc_payloads.size else 0
    spatial_tables = {}
    if spatial_inflow:
        spatial_tables["interval_time_s"] = np.ascontiguousarray(
            np.column_stack(
                (
                    execution_plan.interval_start_s[interval_slice],
                    execution_plan.interval_mid_s[interval_slice],
                    execution_plan.interval_end_s[interval_slice],
                )
            ),
            dtype=np.float64,
        )
    return {
        **spatial_tables,
        "group_half_s": np.ascontiguousarray(unique_groups[:, 0]),
        "group_conversion": np.ascontiguousarray(
            unique_groups[:, 1] != 0.0, dtype=np.int8
        ),
        "interval_group": np.ascontiguousarray(
            interval_group.reshape(-1), dtype=np.int32
        ),
        "interval_rf": interval_rf,
        "interval_gradient": interval_gradient,
        "interval_source": interval_source,
        "interval_phase_s": phase,
        "rf_table": rf_table,
        "rf_inputs": np.asarray(rf_input_rows, dtype=np.float64).reshape(-1, 3),
        "gradient_table": gradient_table,
        "source_table": source_table,
        "action_offsets": action_offsets,
        "adc_first_payload": adc_first,
        "adc_count": adc_count,
    }


def _classify_tx_sensitivity(tx_sensitivity):
    """Classify an already validated Tx vector using exact equality."""
    tx_sensitivity = np.asarray(tx_sensitivity, dtype=np.complex128)
    if tx_sensitivity.ndim != 1 or tx_sensitivity.size == 0:
        raise ValueError("Tx sensitivity must be a non-empty 1D array")
    first = complex(tx_sensitivity[0])
    if first == (1.0 + 0.0j) and np.all(tx_sensitivity == (1.0 + 0.0j)):
        return "unity", 1.0 + 0.0j
    if np.all(tx_sensitivity == first):
        return "constant", first
    return "spatial", None


def _prepare_spatial_rf_rotation(rf_hz, tx_sensitivity, duration):
    """Prepare the established spatial-B1 plane-basis coefficients in Float64."""
    effective_rf = complex(rf_hz) * np.asarray(tx_sensitivity, dtype=np.complex128)
    nx = -2.0 * np.pi * effective_rf.real * float(duration)
    ny = 2.0 * np.pi * effective_rf.imag * float(duration)
    angle = np.hypot(nx, ny)
    nonzero = angle > 0.0
    safe_angle = np.where(nonzero, angle, 1.0)
    return tuple(
        np.ascontiguousarray(value, dtype=np.float64)
        for value in (
            np.where(nonzero, nx / safe_angle, 1.0),
            np.where(nonzero, ny / safe_angle, 0.0),
            np.cos(angle),
            np.sin(angle),
        )
    )


def _rf_rotate(state, rf_hz, duration):
    prepared = _prepare_rf_rotation(rf_hz, duration)
    if prepared is None:
        return
    axis_x, axis_y, cosine, sine, one_minus_cosine = prepared
    axis = np.asarray([axis_x, axis_y, 0.0])
    for pool in range(state.shape[0]):
        vectors = state[pool]
        cross = np.cross(np.broadcast_to(axis, vectors.shape), vectors)
        projection = vectors @ axis
        state[pool] = (
            vectors * cosine
            + cross * sine
            + projection[:, None] * axis * one_minus_cosine
        )


def _rf_rotate_spatial(state, rf_hz, tx_sensitivity, duration):
    """Apply one RF rotation with a complex per-voxel transmit profile."""
    axis_x, axis_y, cosine, sine = _prepare_spatial_rf_rotation(
        rf_hz, tx_sensitivity, duration
    )
    for pool in range(state.shape[0]):
        vectors = state[pool]
        value_x = vectors[:, 0].copy()
        value_y = vectors[:, 1].copy()
        value_z = vectors[:, 2].copy()
        parallel = value_x * axis_x + value_y * axis_y
        perpendicular = -value_x * axis_y + value_y * axis_x
        rotated_perpendicular = perpendicular * cosine - value_z * sine
        vectors[:, 0] = parallel * axis_x - rotated_perpendicular * axis_y
        vectors[:, 1] = parallel * axis_y + rotated_perpendicular * axis_x
        vectors[:, 2] = perpendicular * sine + value_z * cosine


def _gradient_frequency_hz(positions_m, gradient_hz_per_m):
    """Evaluate rowwise position/gradient products in a chunk-stable order."""
    values = positions_m[:, 0] * gradient_hz_per_m[0]
    values = values + positions_m[:, 1] * gradient_hz_per_m[1]
    values = values + positions_m[:, 2] * gradient_hz_per_m[2]
    return values


def _simulate_dynamic_sequence_block(
    program,
    phantom: DynamicSpectralPhantom,
    *,
    checkpoints_s=(),
    field_strength_t=None,
    nucleus=None,
    sequence_reference_ppm=None,
    progress_callback=None,
    preview_callback=None,
    cancel_callback=None,
    status_callback=None,
    simulation_timestep_s=1e-6,
    signal_weighting="voxel",
    sequence_kernel="optimized",
    use_parallel=True,
    num_threads=1,
    memory_budget_bytes=None,
    spin_sampling=None,
    spoiler_mode="ideal",
    checkpoint_dtype=None,
    _compiled=None,
    _execution_plan=None,
    _active_indices=None,
    _parent_resolved_signal=False,
    _block_result_mode=None,
    _sequence_compile_s=0.0,
    _execution_plan_build_s=0.0,
    _tx_classification=None,
    _constant_tx_sensitivity=None,
    _native_spin_major=None,
    _native_spin_major_rf_cache=None,
    _spin_major_rf_propagator_budget_bytes=None,
    **_ignored,
):
    """Run the complete sequence on a regional dynamic spectral phantom.

    Ideal spoiling always uses one spin per voxel; subvoxel sampling is active
    only for gradient-waveform spoiling.
    """
    from .sequence import (
        AcquisitionDimensions,
        SequenceCompiler,
        SequenceSimulationResult,
        physical_b1_field_arrays,
        physical_sequence_waveforms,
    )
    from .sequence.acquisition import (
        CartesianAcquisitionFrames,
        infer_cartesian_acquisition,
        infer_cartesian_acquisition_frames,
        infer_cartesian_acquisition_volumes,
        infer_spectroscopic_acquisition,
        infer_spiral_acquisition,
    )
    from .sequence.spin_sampling import (
        coerce_spin_sampling,
        phantom_voxel_basis_m,
    )

    spoiler_mode = str(spoiler_mode).strip().lower()
    if spoiler_mode not in {"ideal", "gradient"}:
        raise ValueError("spoiler_mode must be 'ideal' or 'gradient'")
    sampling = coerce_spin_sampling(
        spin_sampling if spoiler_mode == "gradient" else None
    )
    sampling.validate_phantom_dimensions(phantom.ndim)

    field = (
        phantom.field_strength if field_strength_t is None else float(field_strength_t)
    )
    effective_nucleus = phantom.nucleus if nucleus is None else str(nucleus)
    effective_reference_ppm = (
        phantom.spectral_reference_ppm
        if sequence_reference_ppm is None
        else float(sequence_reference_ppm)
    )
    if not np.isfinite(effective_reference_ppm):
        raise ValueError("sequence_reference_ppm must be finite")
    inflow_curve = phantom.inflow_curve_on_sequence_timeline
    inflow_polarization_curve = phantom.inflow_polarization_curve_on_sequence_timeline
    spatial_inflow_active = bool(
        phantom.pyruvate_inflow is not None
        and phantom.pyruvate_inflow.has_spatial_delays
    )
    conversion_start_s = phantom.conversion_start_on_sequence_timeline_s
    if sequence_kernel is None:
        sequence_kernel = "optimized"
    if sequence_kernel not in DYNAMIC_KERNEL_NAMES:
        raise ValueError(
            "sequence_kernel must be 'optimized', 'reference', 'native_serial', "
            "or 'native_parallel'"
        )
    real_dtype = np.dtype(np.float64)
    checkpoint_dtype = np.dtype(
        real_dtype if checkpoint_dtype is None else checkpoint_dtype
    )
    if checkpoint_dtype not in {
        np.dtype(np.float16),
        np.dtype(np.float32),
        np.dtype(np.float64),
    }:
        raise ValueError("checkpoint_dtype must be float16, float32, or float64")
    complex_dtype = np.dtype(np.complex128)
    real_type = real_dtype.type
    requested_sequence_kernel = sequence_kernel
    native_fallback_reason = None
    native_longitudinal_fallback_reason = None
    native_longitudinal_step = None
    native_longitudinal_block = None
    native_concentration_inflow_step = None
    native_rf_concentration_block = None
    native_rf_rotation_block = None
    native_spatial_rf_rotation_block = None
    native_spatial_rf_prepare = None
    native_transverse_prepare = None
    native_plan_driver = None
    native_plan_inflow_driver = None
    native_plan_multi_pool_driver = None
    if sequence_kernel in {"native_serial", "native_parallel"}:
        try:
            from .dynamic_bloch_cy import (
                apply_longitudinal_block_no_inflow,
                apply_longitudinal_step_no_inflow,
                apply_longitudinal_step_with_concentration_inflow,
                apply_dynamic_rf_block_with_concentration_inflow,
                apply_rf_rotation_transverse_block,
                apply_spatial_rf_rotation_transverse_block,
                prepare_spatial_rf_rotation_coefficients,
                prepare_transverse_factors_coefficients,
                run_dynamic_plan_block_no_inflow,
                run_dynamic_plan_block_multi_pool,
            )

            # RF rotation is independent of inflow, concentration tracking,
            # conversion timing, and dynamic B0. Keep that native fast path
            # available even when longitudinal evolution must stay in NumPy.
            native_rf_rotation_block = apply_rf_rotation_transverse_block
            native_spatial_rf_rotation_block = (
                apply_spatial_rf_rotation_transverse_block
            )
            native_spatial_rf_prepare = prepare_spatial_rf_rotation_coefficients
            native_transverse_prepare = prepare_transverse_factors_coefficients
            native_plan_driver = run_dynamic_plan_block_no_inflow
            native_plan_multi_pool_driver = run_dynamic_plan_block_multi_pool
            native_plan_inflow_driver = apply_dynamic_rf_block_with_concentration_inflow
            coupled_concentration_inflow = bool(
                phantom.initial_spin_density_maps is not None
                and inflow_curve is not None
                and inflow_polarization_curve is not None
            )
            if coupled_concentration_inflow and phantom.n_species == 2:
                native_concentration_inflow_step = (
                    apply_longitudinal_step_with_concentration_inflow
                )
                if not spatial_inflow_active:
                    native_rf_concentration_block = (
                        apply_dynamic_rf_block_with_concentration_inflow
                    )
            unsupported_longitudinal_drivers = []
            if inflow_curve is not None and not coupled_concentration_inflow:
                unsupported_longitudinal_drivers.append("pyruvate inflow")
            if conversion_start_s > 0.0 and not coupled_concentration_inflow:
                unsupported_longitudinal_drivers.append("delayed conversion")
            if (
                phantom.initial_spin_density_maps is not None
                and not coupled_concentration_inflow
            ):
                unsupported_longitudinal_drivers.append("concentration tracking")
            if unsupported_longitudinal_drivers:
                native_longitudinal_fallback_reason = (
                    "native longitudinal evolution does not support "
                    + ", ".join(unsupported_longitudinal_drivers)
                )
            elif not coupled_concentration_inflow:
                native_longitudinal_step = apply_longitudinal_step_no_inflow
                native_longitudinal_block = apply_longitudinal_block_no_inflow
        except ImportError:
            native_fallback_reason = "strict native extension is unavailable"
            sequence_kernel = "optimized"
    native_spin_major_requested = _spin_major_requested(_native_spin_major)
    native_spin_major_rf_cache_requested = _spin_major_rf_cache_requested(
        _native_spin_major_rf_cache
    )
    if (
        _spin_major_rf_propagator_budget_bytes is None
        and memory_budget_bytes is not None
    ):
        _spin_major_rf_propagator_budget_bytes = int(memory_budget_bytes) // 8
    native_spin_major_chunk = None
    native_spin_major_tile_spins = 0
    if native_spin_major_requested and sequence_kernel in {
        "native_serial",
        "native_parallel",
    }:
        spin_major_kernel = _spin_major_kernel()
        if spin_major_kernel is not None:
            native_spin_major_chunk, native_spin_major_tile_spins = spin_major_kernel
    if _compiled is None:
        compile_started = perf_counter()
        compiled = SequenceCompiler().compile(
            program,
            checkpoints_s=checkpoints_s,
            extra_boundaries_s=phantom.dynamic_breakpoints_s(program.duration_s),
            simulation_timestep_s=simulation_timestep_s,
            status_callback=status_callback,
        )
        sequence_compile_s = perf_counter() - compile_started
    else:
        compiled = _compiled
        sequence_compile_s = float(_sequence_compile_s)
    if _execution_plan is None:
        plan_started = perf_counter()
        execution_plan = build_dynamic_execution_plan(compiled, phantom)
        execution_plan_build_s = perf_counter() - plan_started
    else:
        execution_plan = _execution_plan
        execution_plan_build_s = float(_execution_plan_build_s)
    if _active_indices is None:
        active = np.flatnonzero(phantom.mask.ravel())
    else:
        active = np.ascontiguousarray(_active_indices, dtype=np.int64)
        if (
            active.ndim != 1
            or np.any(active < 0)
            or np.any(active >= phantom.nvoxels)
            or np.any(np.diff(active) <= 0)
        ):
            raise ValueError("active block indices must be strictly increasing")
    if active.size == 0:
        raise ValueError(
            "dynamic phantom has neither initial magnetization nor inflow support"
        )
    parent_active_count = int(active.size)
    spins_per_voxel = sampling.spins_per_voxel
    n_simulated_spins = parent_active_count * spins_per_voxel
    subvoxel_offsets_m, subvoxel_weights = sampling.offsets_m(
        phantom_voxel_basis_m(phantom)
    )
    spin_signal_weights = np.tile(
        np.asarray(subvoxel_weights, dtype=real_dtype), parent_active_count
    )
    if sampling.enabled and memory_budget_bytes is not None:
        # Dynamic state is retained for the complete active object rather than
        # streamed in voxel chunks. Include all pools, sparse checkpoints and
        # the principal coefficient/position arrays in a conservative estimate.
        checkpoint_count = int(compiled.checkpoint_times_s.size)
        estimated_bytes = n_simulated_spins * (
            160 + checkpoint_count * phantom.n_species * 3 * checkpoint_dtype.itemsize
        )
        if phantom.initial_spin_density_maps is not None:
            estimated_bytes += (
                n_simulated_spins * phantom.n_species * 3 * real_dtype.itemsize
            )
        if estimated_bytes > int(memory_budget_bytes):
            raise MemoryError(
                "Memory limit exceeded: dynamic subvoxel simulation needs "
                f"approximately {estimated_bytes / 1024**2:.2f} MiB for "
                f"{n_simulated_spins:,} spins, above the current safe budget. "
                "Reduce X/Y/Z subvoxel spin counts or phantom resolution."
            )
    state = (
        np.asarray(phantom.initial_magnetization, dtype=real_dtype)
        .reshape(phantom.n_species, phantom.nvoxels, 3)[:, active]
        .copy()
    )
    if sampling.enabled:
        state = np.repeat(state, spins_per_voxel, axis=1)
    concentration_state = None
    if phantom.initial_spin_density_maps is not None:
        initial_spin_density = np.asarray(
            phantom.initial_spin_density, dtype=real_dtype
        ).reshape(phantom.n_species, phantom.nvoxels)[:, active]
        if sampling.enabled:
            initial_spin_density = np.repeat(
                initial_spin_density, spins_per_voxel, axis=1
            )
        concentration_state = np.zeros_like(state)
        concentration_state[:, :, 2] = initial_spin_density
    positions = np.asarray(phantom.positions[active], dtype=np.float64)
    if sampling.enabled:
        positions = np.repeat(positions, spins_per_voxel, axis=0) + np.tile(
            subvoxel_offsets_m, (parent_active_count, 1)
        )
    tx_map = getattr(phantom, "tx_sensitivity_map", None)
    if tx_map is None:
        tx_sensitivity = np.ones(parent_active_count, dtype=np.complex128)
    else:
        tx_map = np.asarray(tx_map, dtype=np.complex128)
        if tx_map.shape != phantom.shape or not np.all(np.isfinite(tx_map)):
            raise ValueError(
                "dynamic phantom Tx sensitivity must be finite and match its shape"
            )
        tx_sensitivity = tx_map.ravel()[active]
    if sampling.enabled:
        tx_sensitivity = np.repeat(tx_sensitivity, spins_per_voxel)
    if _tx_classification is None:
        tx_classification, constant_tx_sensitivity = _classify_tx_sensitivity(
            tx_sensitivity
        )
    else:
        tx_classification = str(_tx_classification)
        constant_tx_sensitivity = _constant_tx_sensitivity
    spatial_tx_active = tx_classification == "spatial"
    nonunity_tx_active = tx_classification != "unity"
    if (
        sequence_kernel in {"native_serial", "native_parallel"}
        and native_longitudinal_step is None
        and native_concentration_inflow_step is None
        and spatial_tx_active
        and native_spatial_rf_rotation_block is None
    ):
        native_fallback_reason = (
            "spatial transmit sensitivity and dynamic longitudinal drivers "
            "require the optimized kernel"
        )
        sequence_kernel = "optimized"
        native_rf_rotation_block = None
        native_spatial_rf_rotation_block = None
    rx_maps = getattr(phantom, "rx_sensitivity_maps", None)
    if rx_maps is None:
        rx_sensitivities = np.ones((1, parent_active_count), dtype=np.complex128)
    else:
        rx_maps = np.asarray(rx_maps, dtype=np.complex128)
        if (
            rx_maps.ndim != 4
            or rx_maps.shape[0] < 1
            or rx_maps.shape[1:] != phantom.shape
            or not np.all(np.isfinite(rx_maps))
        ):
            raise ValueError(
                "dynamic phantom Rx sensitivities must have finite shape "
                "(coil, *phantom.shape)"
            )
        rx_sensitivities = rx_maps.reshape(rx_maps.shape[0], -1)[:, active]
    if sampling.enabled:
        rx_sensitivities = np.repeat(rx_sensitivities, spins_per_voxel, axis=1)
    n_rx_coils = int(rx_sensitivities.shape[0])
    unity_single_rx = n_rx_coils == 1 and np.all(rx_sensitivities == (1.0 + 0.0j))
    coefficient_kpl = np.asarray(
        phantom.kpl_map_s_inv.ravel()[active], dtype=np.float64
    )
    if sampling.enabled:
        coefficient_kpl = np.repeat(coefficient_kpl, spins_per_voxel)
    kpl = np.asarray(coefficient_kpl, dtype=real_dtype)
    passive_conversion = None
    if phantom.n_species > 2 and hasattr(phantom, "kpa_map_s_inv"):
        kpa_map = np.asarray(phantom.kpa_map_s_inv, dtype=np.float64)
        if kpa_map.shape != phantom.shape or not np.all(np.isfinite(kpa_map)):
            raise ValueError("kPA map must be finite and match dynamic phantom")
        if np.any(kpa_map < 0):
            raise ValueError("kPA map must be non-negative")
        coefficient_passive_conversion = np.zeros(
            (phantom.n_species - 2, parent_active_count), dtype=np.float64
        )
        coefficient_passive_conversion[0] = kpa_map.ravel()[active]
        if sampling.enabled:
            coefficient_passive_conversion = np.repeat(
                coefficient_passive_conversion, spins_per_voxel, axis=1
            )
        passive_conversion = np.asarray(
            coefficient_passive_conversion, dtype=real_dtype
        )
    multi_pool_conversion = None
    if phantom.n_species > 2:
        multi_pool_conversion = np.zeros(
            (phantom.n_species - 1, n_simulated_spins), dtype=real_dtype
        )
        multi_pool_conversion[0] = kpl
        if passive_conversion is not None:
            multi_pool_conversion[1:] = passive_conversion
    b0 = np.asarray(
        phantom.b0_offset_hz(field, effective_nucleus).ravel()[active],
        dtype=np.float64,
    )
    if sampling.enabled:
        b0 = np.repeat(b0, spins_per_voxel)
    pool_offsets = np.asarray(
        [pool.get_frequency_offset(field, effective_nucleus) for pool in phantom.pools],
        dtype=np.float64,
    )
    pool_offsets += float(
        ppm_to_hz(
            phantom.spectral_reference_ppm - effective_reference_ppm,
            field,
            effective_nucleus,
        )
    )
    t2 = np.asarray([pool.t2 for pool in phantom.pools], dtype=np.float64)[:, None]

    def prepare_transverse_factors(phase_cycles, duration_s):
        if not _parent_resolved_signal or native_transverse_prepare is None:
            return _prepare_transverse_factors(phase_cycles, t2, duration_s)
        table = native_transverse_prepare(
            np.ascontiguousarray(phase_cycles, dtype=np.float64),
            np.ascontiguousarray(t2[:, 0], dtype=np.float64),
            float(duration_s),
        )
        return tuple(
            np.ascontiguousarray(table[pool], dtype=np.complex128)
            for pool in range(table.shape[0])
        )

    coefficient_r1 = np.asarray(
        [1.0 / pool.t1 for pool in phantom.pools], dtype=np.float64
    )
    r1 = np.asarray(coefficient_r1, dtype=real_dtype)
    inflow_delivery = None
    inflow_arrival_delay = None
    if phantom.pyruvate_inflow is not None:
        inflow_delivery = np.asarray(
            phantom.pyruvate_inflow.delivery_map.ravel()[active], dtype=real_dtype
        )
        if phantom.pyruvate_inflow.arrival_delay_map_s is not None:
            inflow_arrival_delay = np.asarray(
                phantom.pyruvate_inflow.arrival_delay_map_s.ravel()[active],
                dtype=real_dtype,
            )
        if sampling.enabled:
            inflow_delivery = np.repeat(inflow_delivery, spins_per_voxel)
            if inflow_arrival_delay is not None:
                inflow_arrival_delay = np.repeat(inflow_arrival_delay, spins_per_voxel)
    preroll_start_s = kinetic_preroll_start_s(
        (
            None
            if phantom.pyruvate_inflow is None
            else phantom.pyruvate_inflow.rate_curve_s_inv
        ),
        phantom.conversion_start_s,
        phantom.kinetics_time_offset_s,
        arrival_delay_s=inflow_arrival_delay,
    )
    if preroll_start_s < 0.0:
        _advance_longitudinal_kinetics(
            state,
            kpl,
            r1,
            preroll_start_s,
            0.0,
            inflow_curve=inflow_curve,
            inflow_delivery=inflow_delivery,
            conversion_start_s=conversion_start_s,
            inflow_polarization_curve=inflow_polarization_curve,
            concentration_state=concentration_state,
            equilibrium_polarization=phantom.equilibrium_polarization,
            inflow_arrival_delay_s=inflow_arrival_delay,
            inflow_max_step_s=(
                None
                if phantom.pyruvate_inflow is None
                else phantom.pyruvate_inflow.max_step_s
            ),
            passive_conversion=passive_conversion,
        )
        if status_callback is not None:
            status_callback(
                f"Applied free kinetic pre-roll from {preroll_start_s:.6g} s "
                "to sequence time zero."
            )
    transverse_state = (
        np.asarray(state[:, :, 0] + 1j * state[:, :, 1], dtype=complex_dtype)
        if sequence_kernel in {"optimized", "native_serial", "native_parallel"}
        else None
    )
    dynamic_b0_scale = None
    dynamic_b0_pool_scale = None
    if phantom.dynamic_b0 is not None:
        dynamic_b0_scale = np.asarray(
            phantom.dynamic_b0.spatial_scale_map.ravel()[active], dtype=np.float64
        )
        if sampling.enabled:
            dynamic_b0_scale = np.repeat(dynamic_b0_scale, spins_per_voxel)
        dynamic_b0_pool_scale = np.asarray(
            phantom.dynamic_b0.pool_scale, dtype=np.float64
        )[:, None]
    native_spin_major_fallback_reason = None
    if sequence_kernel in {"native_serial", "native_parallel"}:
        native_spin_major_fallback_reason = _spin_major_fallback_reason(
            requested=native_spin_major_requested,
            available=native_spin_major_chunk is not None,
            phantom=phantom,
            real_dtype=real_dtype,
            checkpoint_dtype=checkpoint_dtype,
            spatial_tx=spatial_tx_active,
            execution_plan=execution_plan,
        )
    native_spin_major_limits = None
    if (
        sequence_kernel in {"native_serial", "native_parallel"}
        and native_spin_major_fallback_reason is None
    ):
        native_spin_major_limits = _spin_major_slice_limits(
            memory_budget_bytes,
            n_rx_coils,
            max(1, int(num_threads)) if use_parallel else 1,
            phantom.n_species,
            spatial_inflow_active,
        )
        if native_spin_major_limits is None:
            native_spin_major_fallback_reason = "memory budget is too small"
    native_spin_major_enabled = bool(
        sequence_kernel in {"native_serial", "native_parallel"}
        and native_spin_major_fallback_reason is None
    )
    # Spin-major ADC sums are reduced over the whole block in a fixed order,
    # so a parent-resolved block only needs one signal row.
    signal_parent_rows = 1 if native_spin_major_enabled else parent_active_count
    if _parent_resolved_signal:
        species_signal_shape = (
            (signal_parent_rows, phantom.n_species, compiled.adc_times_s.size)
            if n_rx_coils == 1
            else (
                signal_parent_rows,
                phantom.n_species,
                n_rx_coils,
                compiled.adc_times_s.size,
            )
        )
    else:
        species_signal_shape = (
            (phantom.n_species, compiled.adc_times_s.size)
            if n_rx_coils == 1
            else (phantom.n_species, n_rx_coils, compiled.adc_times_s.size)
        )
    species_signal = np.zeros(species_signal_shape, dtype=complex_dtype)
    if signal_weighting not in {"voxel", "voxel_volume"}:
        raise ValueError("signal_weighting must be 'voxel' or 'voxel_volume'")
    signal_scale = real_type(
        phantom.voxel_volume_m3 if signal_weighting == "voxel_volume" else 1.0
    )
    if _parent_resolved_signal:
        parent_signal_weights = spin_signal_weights.reshape(
            parent_active_count, spins_per_voxel
        )
        parent_rx_sensitivities = rx_sensitivities.reshape(
            n_rx_coils, parent_active_count, spins_per_voxel
        )
    checkpoint_states = np.zeros(
        (
            compiled.checkpoint_times_s.size,
            phantom.n_species,
            n_simulated_spins,
            3,
        ),
        dtype=checkpoint_dtype,
    )
    gradient_hz_per_m = execution_plan.gradient_hz_per_m
    rf_hz = execution_plan.rf_hz
    native_rf_hz = (
        rf_hz * constant_tx_sensitivity if tx_classification == "constant" else rf_hz
    )
    adc_demodulation = compiled.adc_demodulation
    action_cursor = 0
    transverse_crush_state_set = set(
        int(value)
        for value in (
            compiled.transverse_crush_state_indices if spoiler_mode == "ideal" else ()
        )
    )

    def sync_transverse_to_state():
        if transverse_state is not None:
            state[:, :, 0] = transverse_state.real
            state[:, :, 1] = transverse_state.imag

    def combined_signal_snapshot():
        return (
            species_signal.sum(axis=(0, 1))
            if _parent_resolved_signal
            else species_signal.sum(axis=0)
        )

    def process_boundary_actions(state_index):
        """Apply one plan boundary in crusher, ADC, checkpoint order."""
        nonlocal action_cursor
        while (
            action_cursor < execution_plan.n_actions
            and execution_plan.action_state_indices[action_cursor] == state_index
        ):
            action_kind = int(execution_plan.action_kinds[action_cursor])
            payload = int(execution_plan.action_payload_indices[action_cursor])
            if action_kind == ACTION_CRUSH:
                if spoiler_mode == "ideal":
                    state[:, :, 0:2] = 0.0
                    if transverse_state is not None:
                        transverse_state[:] = 0.0
                action_cursor += 1
                continue
            if action_kind == ACTION_ADC:
                demodulation = adc_demodulation[payload]
                for pool in range(phantom.n_species):
                    transverse = (
                        state[pool, :, 0] + 1j * state[pool, :, 1]
                        if transverse_state is None
                        else transverse_state[pool]
                    )
                    if _parent_resolved_signal:
                        parent_transverse = transverse.reshape(
                            parent_active_count, spins_per_voxel
                        )
                        if n_rx_coils == 1:
                            if unity_single_rx and not sampling.enabled:
                                received = parent_transverse[:, 0]
                            elif unity_single_rx:
                                received = np.sum(
                                    parent_transverse * parent_signal_weights,
                                    axis=1,
                                )
                            else:
                                received = np.sum(
                                    parent_transverse
                                    * parent_rx_sensitivities[0]
                                    * parent_signal_weights,
                                    axis=1,
                                )
                            species_signal[:, pool, payload] = (
                                received * signal_scale * demodulation
                            )
                        else:
                            received = np.sum(
                                parent_rx_sensitivities
                                * parent_transverse[None, :, :]
                                * parent_signal_weights[None, :, :],
                                axis=2,
                            )
                            species_signal[:, pool, :, payload] = (
                                received.T * signal_scale * demodulation
                            )
                        continue
                    if n_rx_coils == 1:
                        received = (
                            np.sum(transverse)
                            if unity_single_rx and not sampling.enabled
                            else (
                                np.sum(transverse * spin_signal_weights)
                                if unity_single_rx
                                else np.sum(
                                    transverse
                                    * rx_sensitivities[0]
                                    * spin_signal_weights
                                )
                            )
                        )
                        species_signal[pool, payload] = (
                            received * signal_scale * demodulation
                        )
                    else:
                        species_signal[pool, :, payload] = (
                            np.sum(
                                rx_sensitivities
                                * transverse[None, :]
                                * spin_signal_weights[None, :],
                                axis=1,
                            )
                            * signal_scale
                            * demodulation
                        )
                action_cursor += 1
                continue
            if action_kind == ACTION_CHECKPOINT:
                sync_transverse_to_state()
                checkpoint_states[payload] = state
                action_cursor += 1
                continue
            raise RuntimeError(f"unknown dynamic plan action kind {action_kind}")

    process_boundary_actions(0)
    interval_count = execution_plan.n_intervals
    progress_stride = max(1, interval_count // 100)
    coefficient_cache = (
        _BoundedArrayCache(16 * 1024**2)
        if sequence_kernel in {"optimized", "native_serial", "native_parallel"}
        else None
    )
    transverse_cache = (
        _BoundedArrayCache(48 * 1024**2)
        if sequence_kernel in {"optimized", "native_serial", "native_parallel"}
        else None
    )
    longitudinal_regular = (
        np.abs(coefficient_r1[0] + coefficient_kpl - coefficient_r1[1]) > 1e-12
    )
    regular_mode = (
        1
        if np.all(longitudinal_regular)
        else -1 if not np.any(longitudinal_regular) else 0
    )
    longitudinal_scratch = (
        (
            np.empty_like(kpl),
            np.empty_like(kpl),
            np.empty_like(kpl),
            regular_mode,
        )
        if sequence_kernel in {"optimized", "native_serial", "native_parallel"}
        else None
    )
    coefficient_zero_kpl = np.zeros_like(coefficient_kpl)
    zero_kpl = np.zeros_like(kpl)
    zero_passive_conversion = (
        None if passive_conversion is None else np.zeros_like(passive_conversion)
    )
    inactive_longitudinal_regular = (
        np.abs(coefficient_r1[0] - coefficient_r1[1]) > 1e-12
    )
    inactive_regular_mode = 1 if inactive_longitudinal_regular else -1
    inactive_longitudinal_scratch = (
        (
            np.empty_like(kpl),
            np.empty_like(kpl),
            np.empty_like(kpl),
            inactive_regular_mode,
        )
        if sequence_kernel == "optimized" or native_longitudinal_step is None
        else None
    )
    concentration_longitudinal_scratch = None
    inactive_concentration_longitudinal_scratch = None
    if concentration_state is not None and sequence_kernel in {
        "optimized",
        "native_serial",
        "native_parallel",
    }:
        concentration_regular = np.abs(coefficient_kpl) > 1e-12
        concentration_regular_mode = (
            1
            if np.all(concentration_regular)
            else -1 if not np.any(concentration_regular) else 0
        )
        concentration_longitudinal_scratch = (
            np.empty_like(kpl),
            np.empty_like(kpl),
            np.empty_like(kpl),
            concentration_regular_mode,
        )
        inactive_concentration_longitudinal_scratch = (
            np.empty_like(kpl),
            np.empty_like(kpl),
            np.empty_like(kpl),
            -1,
        )
    native_parallel_threshold = 1024
    requested_native_threads = max(1, int(num_threads)) if use_parallel else 1
    native_threads = (
        requested_native_threads
        if n_simulated_spins >= native_parallel_threshold
        else 1
    )
    native_rf_threads = native_threads if sequence_kernel == "native_parallel" else 1
    native_longitudinal_threads = (
        native_threads if sequence_kernel == "native_parallel" else 1
    )
    native_block_interval_limit = 256
    native_block_table_limit_bytes = 8 * 1024**2
    native_full_plan_table_limit_bytes = 64 * 1024**2
    if memory_budget_bytes is not None:
        native_block_table_limit_bytes = min(
            native_block_table_limit_bytes,
            max(0, int(memory_budget_bytes) // 16),
        )
        native_full_plan_table_limit_bytes = min(
            native_full_plan_table_limit_bytes,
            max(0, int(memory_budget_bytes) // 8),
        )
    native_block_enabled = native_block_table_limit_bytes >= n_simulated_spins * 8
    if not native_block_enabled:
        native_threads = 1
    native_rf_available = bool(
        native_spatial_rf_rotation_block is not None
        if spatial_tx_active
        else native_rf_rotation_block is not None
    )
    native_full_plan_source_model = bool(
        native_plan_inflow_driver is not None
        and (concentration_state is not None or inflow_curve is not None)
    )
    native_full_plan_multi_pool = bool(
        phantom.n_species > 2 and native_plan_multi_pool_driver is not None
    )
    if inflow_curve is None:
        native_full_plan_source_mode = 2
        native_full_plan_model = "tracked-concentration"
    elif inflow_polarization_curve is None:
        native_full_plan_source_mode = 1
        native_full_plan_model = "direct-magnetization-inflow"
    elif concentration_state is None:
        native_full_plan_source_mode = 3
        native_full_plan_model = "polarized-magnetization-inflow"
    else:
        native_full_plan_source_mode = 0
        native_full_plan_model = "coupled-concentration-inflow"
    native_full_plan_no_inflow = bool(
        phantom.n_species == 2
        and native_plan_driver is not None
        and inflow_curve is None
        and concentration_state is None
    )
    if native_full_plan_multi_pool:
        native_full_plan_bytes_per_spin = (
            96
            + 64 * phantom.n_species
            + (64 if native_full_plan_source_model else 0)
            + (64 if spatial_inflow_active else 0)
            + (40 if spatial_tx_active else 0)
        )
    else:
        native_full_plan_bytes_per_spin = (
            (248 if spatial_tx_active else 208) + (64 if spatial_inflow_active else 0)
            if native_full_plan_source_model
            else (136 if spatial_tx_active else 96)
        )
    native_full_plan_enabled = bool(
        sequence_kernel in {"native_serial", "native_parallel"}
        and (
            native_full_plan_multi_pool
            or native_full_plan_no_inflow
            or (
                phantom.n_species == 2
                and native_full_plan_source_model
                and passive_conversion is None
            )
        )
        and checkpoint_dtype == np.dtype(np.float64)
        and native_full_plan_table_limit_bytes
        >= n_simulated_spins * native_full_plan_bytes_per_spin
        and not np.any(
            execution_plan.action_state_indices[
                execution_plan.action_kinds == ACTION_ADC
            ]
            == 0
        )
    )
    if (
        phantom.n_species > 2
        and sequence_kernel in {"native_serial", "native_parallel"}
        and not native_full_plan_enabled
    ):
        native_fallback_reason = (
            "the multi-pool native plan is unavailable for this precision, "
            "memory budget, or initial-time ADC contract"
        )
        sequence_kernel = "optimized"
        native_rf_available = False
        native_rf_rotation_block = None
        native_spatial_rf_rotation_block = None
        native_spatial_rf_prepare = None
        native_transverse_prepare = None
        native_longitudinal_step = None
        native_longitudinal_block = None
        native_concentration_inflow_step = None
        native_rf_concentration_block = None
    if status_callback is not None and native_rf_available:
        if (
            native_longitudinal_fallback_reason is not None
            and not native_full_plan_enabled
        ):
            status_callback(
                "Using the hybrid dynamic kernel: strict native RF rotation "
                "with optimized longitudinal kinetics ("
                f"{native_longitudinal_fallback_reason})."
            )
        status_callback(
            f"Using the strict native {tx_classification} Tx RF voxel-block "
            f"kernel with "
            f"{native_rf_threads} thread(s) for {n_simulated_spins:,} spins "
            f"in {parent_active_count:,} active voxels."
        )
    if status_callback is not None and native_concentration_inflow_step is not None:
        status_callback(
            "Using fused native pyruvate/lactate concentration and inflow "
            f"kinetics with {native_longitudinal_threads} thread(s)."
        )
    native_preapplied_end = 0
    checkpoint_state_set = set(
        int(value) for value in compiled.checkpoint_state_indices
    )
    native_rf_fused_spin_limit = 131072
    native_rf_fused_interval_limit = 512
    native_rf_fused_min_intervals = 8
    native_rf_fused_cache_limit_bytes = 192 * 1024**2
    if memory_budget_bytes is not None:
        native_rf_fused_cache_limit_bytes = min(
            native_rf_fused_cache_limit_bytes,
            max(0, int(memory_budget_bytes) // 8),
        )
    native_spatial_rf_table_limit_bytes = 64 * 1024**2
    if memory_budget_bytes is not None:
        native_spatial_rf_table_limit_bytes = min(
            native_spatial_rf_table_limit_bytes,
            max(0, int(memory_budget_bytes) // 16),
        )
    native_spatial_rf_max_groups = (
        native_spatial_rf_table_limit_bytes
        // max(1, n_simulated_spins * 5 * np.dtype(np.float64).itemsize)
        if spatial_tx_active
        else native_rf_fused_interval_limit
    )
    native_rf_fused_fallback_reason = None
    if spatial_inflow_active:
        native_rf_fused_fallback_reason = (
            "persistent RF blocks do not yet support voxelwise inflow arrival delays"
        )
    elif native_rf_concentration_block is None:
        native_rf_fused_fallback_reason = (
            "persistent RF blocks require coupled concentration/polarized inflow"
        )
    elif phantom.dynamic_b0 is not None:
        native_rf_fused_fallback_reason = (
            "persistent RF blocks do not yet support dynamic B0"
        )
    elif tx_classification == "constant" and constant_tx_sensitivity == (0.0 + 0.0j):
        native_rf_fused_fallback_reason = (
            "zero transmit sensitivity does not need persistent RF rotation"
        )
    elif n_simulated_spins > native_rf_fused_spin_limit:
        native_rf_fused_fallback_reason = (
            f"{n_simulated_spins:,} spins exceed the persistent RF block limit "
            f"of {native_rf_fused_spin_limit:,}; full-sequence voxel chunking "
            "is required"
        )
    elif native_rf_fused_cache_limit_bytes < n_simulated_spins * 64:
        native_rf_fused_fallback_reason = (
            "the simulation memory budget is too small for a persistent RF plan"
        )
    elif spatial_tx_active and native_spatial_rf_max_groups < 1:
        native_rf_fused_fallback_reason = (
            "the simulation memory budget is too small for one spatial RF "
            "coefficient group"
        )
    native_rf_fused_enabled = native_rf_fused_fallback_reason is None
    native_rf_fused_plan_cache = (
        _BoundedArrayCache(native_rf_fused_cache_limit_bytes)
        if native_rf_fused_enabled
        else None
    )
    spatial_rf_cache_limit_bytes = 64 * 1024**2
    if memory_budget_bytes is not None:
        spatial_rf_cache_limit_bytes = min(
            spatial_rf_cache_limit_bytes,
            max(0, int(memory_budget_bytes) // 16),
        )
    spatial_rf_cache = (
        _BoundedArrayCache(spatial_rf_cache_limit_bytes)
        if spatial_tx_active and native_spatial_rf_rotation_block is not None
        else None
    )
    spatial_rf_cache_hits = 0
    spatial_rf_cache_misses = 0

    def spatial_rf_coefficients(rf_value, duration_s):
        nonlocal spatial_rf_cache_hits, spatial_rf_cache_misses
        key = (complex(rf_value), float(duration_s))
        cached = spatial_rf_cache.get(key)
        if cached is not None:
            spatial_rf_cache_hits += 1
            return cached
        spatial_rf_cache_misses += 1
        if native_spatial_rf_prepare is None:
            prepared = _prepare_spatial_rf_rotation(
                rf_value, tx_sensitivity, duration_s
            )
        else:
            prepared = native_spatial_rf_prepare(
                float(complex(rf_value).real),
                float(complex(rf_value).imag),
                np.ascontiguousarray(tx_sensitivity, dtype=np.complex128),
                float(duration_s),
            )
        return spatial_rf_cache.put(key, prepared)

    native_rf_fused_preapplied_end = 0
    native_rf_fused_blocks = 0
    native_rf_fused_intervals = 0
    protected_state_set = set(checkpoint_state_set)
    protected_state_set.update(int(value) for value in compiled.adc_state_indices)
    protected_state_set.update(transverse_crush_state_set)

    def _native_rf_fused_block_end(first_interval):
        if not native_rf_fused_enabled or native_rf_hz[first_interval] == 0.0:
            return first_interval
        end = first_interval
        spatial_rf_group_keys = set()
        hard_end = min(interval_count, first_interval + native_rf_fused_interval_limit)
        while end < hard_end and native_rf_hz[end] != 0.0:
            if end > first_interval and end in protected_state_set:
                break
            if not execution_plan.conversion_active[end]:
                break
            if spatial_tx_active:
                rf_group_key = (
                    complex(native_rf_hz[end]),
                    float(compiled.dt_s[end]),
                )
                if (
                    rf_group_key not in spatial_rf_group_keys
                    and len(spatial_rf_group_keys) >= native_spatial_rf_max_groups
                ):
                    break
                spatial_rf_group_keys.add(rf_group_key)
            end += 1
            if end in protected_state_set:
                break
        if end - first_interval < native_rf_fused_min_intervals:
            return first_interval
        return end

    def _native_rf_fused_plan(first_interval, end_interval):
        factor_keys = []
        longitudinal_keys = []
        rf_values = []
        for block_interval in range(first_interval, end_interval):
            dt_value = float(compiled.dt_s[block_interval])
            first_phase, second_phase = execution_plan.interval_half_s[block_interval]
            half_duration = dt_value / 2.0
            gradient_key = tuple(
                float(value) for value in gradient_hz_per_m[block_interval]
            )
            first_key = (
                half_duration,
                float(first_phase),
                gradient_key,
                None,
            )
            second_key = (
                half_duration,
                float(second_phase),
                gradient_key,
                None,
            )
            factor_keys.append((first_key, second_key))
            longitudinal_keys.append((half_duration, True))
            rf_values.append(complex(native_rf_hz[block_interval]))
        plan_key = (
            tuple(factor_keys),
            tuple(longitudinal_keys),
            tuple(rf_values),
            tx_classification,
        )
        cached = native_rf_fused_plan_cache.get(plan_key)
        if cached is not None:
            return cached

        static_frequency_by_gradient = {}
        factor_group_by_key = {}
        factor_values = []

        def factor_group(key):
            group = factor_group_by_key.get(key)
            if group is not None:
                return group
            factors = transverse_cache.get(key)
            if factors is None:
                half_duration, phase_duration, gradient_key, _ = key
                static_frequencies = static_frequency_by_gradient.get(gradient_key)
                if static_frequencies is None:
                    gradient = np.asarray(gradient_key, dtype=np.float64)
                    gradient_frequency = _gradient_frequency_hz(positions, gradient)
                    static_frequencies = (
                        b0[None, :]
                        + pool_offsets[:, None]
                        + gradient_frequency[None, :]
                    )
                    static_frequency_by_gradient[gradient_key] = static_frequencies
                phase = static_frequencies * phase_duration
                factors = prepare_transverse_factors(phase, half_duration)
                transverse_cache.put(key, factors)
            group = len(factor_values)
            factor_group_by_key[key] = group
            factor_values.append(factors)
            return group

        first_factor_groups = []
        second_factor_groups = []
        for first_key, second_key in factor_keys:
            first_factor_groups.append(factor_group(first_key))
            second_factor_groups.append(factor_group(second_key))
        transverse_factor_table = np.ascontiguousarray(
            np.stack(
                [np.stack(factors, axis=0) for factors in factor_values],
                axis=0,
            ),
            dtype=np.complex128,
        )

        longitudinal_group_by_key = {}
        longitudinal_values = []
        concentration_values = []
        duration_by_group = []
        regular_mode_by_group = []
        concentration_regular_mode_by_group = []
        longitudinal_groups = []
        for longitudinal_key in longitudinal_keys:
            group = longitudinal_group_by_key.get(longitudinal_key)
            if group is None:
                group = len(longitudinal_values)
                longitudinal_group_by_key[longitudinal_key] = group
                half_duration, _ = longitudinal_key
                prepared = coefficient_cache.get(longitudinal_key)
                if prepared is None:
                    prepared = _prepare_longitudinal_step(
                        coefficient_kpl,
                        coefficient_r1[0],
                        coefficient_r1[1],
                        half_duration,
                        with_source=True,
                    )
                    coefficient_cache.put(longitudinal_key, prepared)
                concentration_key = (
                    "concentration",
                    half_duration,
                    True,
                )
                concentration_prepared = coefficient_cache.get(concentration_key)
                if concentration_prepared is None:
                    concentration_prepared = _prepare_longitudinal_step(
                        coefficient_kpl,
                        0.0,
                        0.0,
                        half_duration,
                        with_source=True,
                    )
                    coefficient_cache.put(concentration_key, concentration_prepared)
                longitudinal_values.append(prepared)
                concentration_values.append(concentration_prepared)
                duration_by_group.append(half_duration)
                regular_mode_by_group.append(regular_mode)
                concentration_regular_mode_by_group.append(
                    concentration_longitudinal_scratch[3]
                )
            longitudinal_groups.append(group)

        def stacked(values, index, dtype=np.float64):
            return np.ascontiguousarray(
                np.stack([value[index] for value in values], axis=0),
                dtype=dtype,
            )

        def stacked_source(values, index):
            return np.ascontiguousarray(
                np.stack([value[4][index] for value in values], axis=0),
                dtype=np.float64,
            )

        rf_group_by_key = {}
        rf_groups = []
        rf_prepared = []
        for index, value in zip(range(first_interval, end_interval), rf_values):
            rf_key = (value, float(compiled.dt_s[index]))
            group = rf_group_by_key.get(rf_key)
            if group is None:
                group = len(rf_prepared)
                rf_group_by_key[rf_key] = group
                if spatial_tx_active:
                    prepared = spatial_rf_coefficients(value, rf_key[1])
                else:
                    scalar = _prepare_rf_rotation(value, rf_key[1])
                    prepared = tuple(
                        np.asarray([coefficient], dtype=np.float64)
                        for coefficient in scalar
                    )
                rf_prepared.append(prepared)
            rf_groups.append(group)
        rf_coefficient_count = 4 if spatial_tx_active else 5
        rf_tables = [
            np.ascontiguousarray(
                np.stack([prepared[coefficient] for prepared in rf_prepared], axis=0),
                dtype=np.float64,
            )
            for coefficient in range(rf_coefficient_count)
        ]
        if spatial_tx_active:
            rf_tables.append(np.zeros_like(rf_tables[0]))
        plan = (
            transverse_factor_table,
            stacked(longitudinal_values, 0),
            np.ascontiguousarray(
                [value[1] for value in longitudinal_values], dtype=np.float64
            ),
            stacked(longitudinal_values, 2),
            stacked(longitudinal_values, 3, dtype=np.bool_),
            stacked_source(longitudinal_values, 0),
            stacked_source(longitudinal_values, 1),
            stacked_source(longitudinal_values, 2),
            stacked_source(longitudinal_values, 3),
            stacked(concentration_values, 0),
            np.ascontiguousarray(
                [value[1] for value in concentration_values], dtype=np.float64
            ),
            stacked(concentration_values, 2),
            stacked(concentration_values, 3, dtype=np.bool_),
            stacked_source(concentration_values, 0),
            stacked_source(concentration_values, 1),
            stacked_source(concentration_values, 2),
            stacked_source(concentration_values, 3),
            np.ascontiguousarray(duration_by_group, dtype=np.float64),
            np.ascontiguousarray(regular_mode_by_group, dtype=np.int32),
            np.ascontiguousarray(concentration_regular_mode_by_group, dtype=np.int32),
            np.ascontiguousarray(longitudinal_groups, dtype=np.int32),
            np.ascontiguousarray(first_factor_groups, dtype=np.int32),
            np.ascontiguousarray(second_factor_groups, dtype=np.int32),
            np.ascontiguousarray(rf_groups, dtype=np.int32),
            *rf_tables,
        )
        return native_rf_fused_plan_cache.put(plan_key, plan)

    def preapply_native_rf_fused_block(first_interval):
        nonlocal native_rf_fused_preapplied_end
        nonlocal native_rf_fused_blocks, native_rf_fused_intervals
        end_interval = _native_rf_fused_block_end(first_interval)
        if end_interval == first_interval:
            native_rf_fused_preapplied_end = first_interval
            return
        plan = _native_rf_fused_plan(first_interval, end_interval)
        inflow_values = [[] for _ in range(4)]
        polarization_values = [[] for _ in range(4)]
        interval_left = (
            0.0
            if first_interval == 0
            else float(compiled.interval_end_s[first_interval - 1])
        )
        for block_interval in range(first_interval, end_interval):
            dt_value = float(compiled.dt_s[block_interval])
            interval_middle = interval_left + dt_value / 2.0
            interval_right = float(compiled.interval_end_s[block_interval])
            first_start, first_end, second_start, second_end = (
                execution_plan.source_rate_samples[block_interval]
            )
            (
                polarization_first_start,
                polarization_first_end,
                polarization_second_start,
                polarization_second_end,
            ) = execution_plan.source_polarization_samples[block_interval]
            for target, value in zip(
                inflow_values,
                (first_start, first_end, second_start, second_end),
            ):
                target.append(value)
            for target, value in zip(
                polarization_values,
                (
                    polarization_first_start,
                    polarization_first_end,
                    polarization_second_start,
                    polarization_second_end,
                ),
            ):
                target.append(value)
            interval_left = interval_right
        native_rf_concentration_block(
            state,
            transverse_state,
            concentration_state,
            kpl,
            inflow_delivery,
            *plan,
            *[
                np.ascontiguousarray(values, dtype=np.float64)
                for values in (*inflow_values, *polarization_values)
            ],
            phantom.equilibrium_polarization,
            native_longitudinal_threads,
        )
        native_rf_fused_preapplied_end = end_interval
        native_rf_fused_blocks += 1
        native_rf_fused_intervals += end_interval - first_interval

    if status_callback is not None and native_rf_fused_enabled:
        status_callback(
            "Persistent native RF waveform blocks are available as a coupled "
            "dynamic fallback."
        )
    elif (
        status_callback is not None
        and native_rf_concentration_block is not None
        and native_rf_fused_fallback_reason is not None
    ):
        status_callback(
            "Persistent native RF waveform blocks are unavailable: "
            f"{native_rf_fused_fallback_reason}."
        )

    def preapply_native_longitudinal_block(first_interval):
        """Advance Mz through one RF-free, checkpoint-free bounded block."""
        nonlocal native_preapplied_end
        max_groups = max(
            1,
            native_block_table_limit_bytes // max(1, n_simulated_spins * 8),
        )
        end = first_interval
        half_durations = []
        seen_durations = set()
        hard_end = min(interval_count, first_interval + native_block_interval_limit)
        while end < hard_end:
            if rf_hz[end] != 0.0:
                break
            half_duration = float(compiled.dt_s[end] / 2.0)
            if half_duration not in seen_durations:
                if len(seen_durations) >= max_groups:
                    break
                seen_durations.add(half_duration)
            half_durations.append(half_duration)
            end += 1
            if end in checkpoint_state_set:
                break
        if end == first_interval:
            native_preapplied_end = first_interval
            return

        group_by_duration = {}
        prepared_by_group = []
        duration_by_group = []
        step_groups = []
        for half_duration in half_durations:
            group = group_by_duration.get(half_duration)
            if group is None:
                group = len(prepared_by_group)
                group_by_duration[half_duration] = group
                prepared = coefficient_cache.get(half_duration)
                if prepared is None:
                    prepared = _prepare_longitudinal_step(
                        kpl,
                        r1[0],
                        r1[1],
                        half_duration,
                        with_source=False,
                    )
                    coefficient_cache.put(half_duration, prepared)
                prepared_by_group.append(prepared)
                duration_by_group.append(half_duration)
            step_groups.extend((group, group))
        exp_a_by_group = np.ascontiguousarray(
            np.stack([prepared[0] for prepared in prepared_by_group]),
            dtype=np.float64,
        )
        exp_b_by_group = np.ascontiguousarray(
            [prepared[1] for prepared in prepared_by_group], dtype=np.float64
        )
        native_longitudinal_block(
            state,
            kpl,
            exp_a_by_group,
            exp_b_by_group,
            prepared_by_group[0][2],
            prepared_by_group[0][3],
            np.ascontiguousarray(step_groups, dtype=np.int32),
            np.ascontiguousarray(duration_by_group, dtype=np.float64),
            regular_mode,
            native_threads,
        )
        native_preapplied_end = end

    native_full_plan_blocks = 0
    native_full_plan_intervals = 0

    def execute_native_full_plan():
        nonlocal action_cursor
        nonlocal native_rf_fused_preapplied_end
        nonlocal native_full_plan_blocks, native_full_plan_intervals
        if not native_full_plan_enabled or interval_count == 0:
            return False

        # The kernels accumulate per parent voxel so threads never share an
        # output element. Only the ADC samples of the current time block are
        # held per parent; a whole-acquisition parent array can exceed RAM.
        if _parent_resolved_signal:
            resolved_signal = (
                species_signal[:, :, None, :] if n_rx_coils == 1 else species_signal
            )
        else:
            reduced_signal = (
                species_signal[:, None, :] if n_rx_coils == 1 else species_signal
            )
        all_adc_demodulation = np.ascontiguousarray(
            adc_demodulation, dtype=np.complex128
        )
        last_preview_step = -1
        plan_concentration_state = concentration_state
        if (
            native_full_plan_source_model or native_full_plan_multi_pool
        ) and plan_concentration_state is None:
            plan_concentration_state = np.zeros_like(state)
        plan_inflow_delivery = inflow_delivery
        if (
            native_full_plan_source_model or native_full_plan_multi_pool
        ) and plan_inflow_delivery is None:
            plan_inflow_delivery = np.zeros(n_simulated_spins, dtype=np.float64)
        plan_equilibrium = (
            float(phantom.equilibrium_polarization)
            if concentration_state is not None
            else 0.0
        )
        worst_group_bytes = n_simulated_spins * native_full_plan_bytes_per_spin
        time_block_intervals = max(
            1,
            min(
                128,
                native_full_plan_table_limit_bytes // max(1, worst_group_bytes),
            ),
        )

        # Each block also holds a per-parent window over its ADC samples, so
        # dense acquisitions shorten the block to keep that window bounded.
        adc_state_indices = execution_plan.action_state_indices[
            np.asarray(execution_plan.action_kinds) == ACTION_ADC
        ]
        adc_window_sample_bytes = (
            parent_active_count * phantom.n_species * n_rx_coils * 16
        )
        max_block_adc_samples = max(
            1, native_full_plan_table_limit_bytes // max(1, adc_window_sample_bytes)
        )
        plan_blocks = []
        first_interval = 0
        while first_interval < interval_count:
            end_interval = min(interval_count, first_interval + time_block_intervals)
            adc_start = int(
                np.searchsorted(adc_state_indices, first_interval, side="right")
            )
            limit_index = adc_start + max_block_adc_samples
            if limit_index < adc_state_indices.size:
                end_interval = max(
                    first_interval + 1,
                    min(end_interval, int(adc_state_indices[limit_index]) - 1),
                )
            plan_blocks.append((first_interval, end_interval))
            first_interval = end_interval

        for first_interval, end_interval in plan_blocks:
            if cancel_callback is not None and cancel_callback():
                raise RuntimeError("sequence simulation cancelled")
            factor_group_by_key = {}
            factor_values = []
            longitudinal_group_by_key = {}
            longitudinal_values = []
            concentration_values = []
            longitudinal_kpl_values = []
            multi_longitudinal_values = []
            multi_concentration_values = []
            multi_conversion_values = []
            longitudinal_durations = []
            longitudinal_modes = []
            concentration_modes = []
            rf_group_by_key = {}
            rf_values = []
            first_factor_groups = []
            second_factor_groups = []
            longitudinal_groups = []
            rf_groups = []
            static_frequency_by_gradient = {}

            def transverse_factor_group(
                half_duration,
                phase_duration,
                gradient_key,
                dynamic_integral,
            ):
                factor_key = (
                    half_duration,
                    phase_duration,
                    gradient_key,
                    dynamic_integral,
                )
                factor_group = factor_group_by_key.get(factor_key)
                if factor_group is not None:
                    return factor_group
                factor_group = len(factor_values)
                factor_group_by_key[factor_key] = factor_group
                prepared_factors = transverse_cache.get(factor_key)
                if prepared_factors is None:
                    static_frequencies = static_frequency_by_gradient.get(gradient_key)
                    if static_frequencies is None:
                        gradient = np.asarray(gradient_key, dtype=np.float64)
                        gradient_frequency = _gradient_frequency_hz(positions, gradient)
                        static_frequencies = (
                            b0[None, :]
                            + pool_offsets[:, None]
                            + gradient_frequency[None, :]
                        )
                        static_frequency_by_gradient[gradient_key] = static_frequencies
                    phase = static_frequencies * phase_duration
                    if dynamic_integral is not None:
                        phase = phase + (
                            dynamic_b0_pool_scale
                            * dynamic_b0_scale[None, :]
                            * dynamic_integral
                        )
                    prepared_factors = prepare_transverse_factors(phase, half_duration)
                    transverse_cache.put(factor_key, prepared_factors)
                factor_values.append(prepared_factors)
                return factor_group

            for block_interval in range(first_interval, end_interval):
                half_duration = float(execution_plan.dt_s[block_interval]) / 2.0
                first_phase_duration, second_phase_duration = (
                    float(value)
                    for value in execution_plan.interval_half_s[block_interval]
                )
                gradient_key = tuple(
                    float(value) for value in gradient_hz_per_m[block_interval]
                )
                if phantom.dynamic_b0 is None:
                    first_dynamic_integral = second_dynamic_integral = None
                else:
                    first_dynamic_integral = float(
                        execution_plan.dynamic_b0_first_half_integral_hz_s[
                            block_interval
                        ]
                    )
                    second_dynamic_integral = float(
                        execution_plan.dynamic_b0_second_half_integral_hz_s[
                            block_interval
                        ]
                    )
                first_factor_groups.append(
                    transverse_factor_group(
                        half_duration,
                        first_phase_duration,
                        gradient_key,
                        first_dynamic_integral,
                    )
                )
                second_factor_groups.append(
                    transverse_factor_group(
                        half_duration,
                        second_phase_duration,
                        gradient_key,
                        second_dynamic_integral,
                    )
                )

                conversion_active = bool(
                    execution_plan.conversion_active[block_interval]
                )
                longitudinal_key = (half_duration, conversion_active)
                longitudinal_group = longitudinal_group_by_key.get(longitudinal_key)
                if longitudinal_group is None:
                    longitudinal_group = (
                        len(multi_longitudinal_values)
                        if native_full_plan_multi_pool
                        else len(longitudinal_values)
                    )
                    longitudinal_group_by_key[longitudinal_key] = longitudinal_group
                    longitudinal_durations.append(half_duration)
                    if native_full_plan_multi_pool:
                        interval_conversion = (
                            multi_pool_conversion
                            if conversion_active
                            else np.zeros_like(multi_pool_conversion)
                        )
                        prepared_key = (
                            "full-plan-multi-magnetization",
                            half_duration,
                            conversion_active,
                        )
                        prepared = coefficient_cache.get(prepared_key)
                        if prepared is None:
                            prepared = _prepare_multi_pool_longitudinal_step(
                                interval_conversion,
                                r1,
                                half_duration,
                                with_source=True,
                            )
                            coefficient_cache.put(prepared_key, prepared)
                        concentration_key = (
                            "full-plan-multi-concentration",
                            half_duration,
                            conversion_active,
                        )
                        concentration_prepared = coefficient_cache.get(
                            concentration_key
                        )
                        if concentration_prepared is None:
                            concentration_prepared = (
                                _prepare_multi_pool_longitudinal_step(
                                    interval_conversion,
                                    np.zeros_like(r1),
                                    half_duration,
                                    with_source=True,
                                )
                            )
                            coefficient_cache.put(
                                concentration_key, concentration_prepared
                            )
                        multi_conversion_values.append(interval_conversion)
                        multi_longitudinal_values.append(prepared)
                        multi_concentration_values.append(concentration_prepared)
                    else:
                        interval_kpl = kpl if conversion_active else zero_kpl
                        prepared_key = (
                            "full-plan-magnetization",
                            half_duration,
                            conversion_active,
                            native_full_plan_source_model,
                        )
                        prepared = coefficient_cache.get(prepared_key)
                        if prepared is None:
                            prepared = _prepare_longitudinal_step(
                                interval_kpl,
                                r1[0],
                                r1[1],
                                half_duration,
                                with_source=native_full_plan_source_model,
                            )
                            coefficient_cache.put(prepared_key, prepared)
                        regular_values = prepared[3]
                        longitudinal_values.append(prepared)
                        if native_full_plan_source_model:
                            concentration_key = (
                                "full-plan-concentration",
                                half_duration,
                                conversion_active,
                            )
                            concentration_prepared = coefficient_cache.get(
                                concentration_key
                            )
                            if concentration_prepared is None:
                                concentration_prepared = _prepare_longitudinal_step(
                                    interval_kpl,
                                    0.0,
                                    0.0,
                                    half_duration,
                                    with_source=True,
                                )
                                coefficient_cache.put(
                                    concentration_key, concentration_prepared
                                )
                            concentration_values.append(concentration_prepared)
                            concentration_regular_values = concentration_prepared[3]
                            concentration_modes.append(
                                1
                                if np.all(concentration_regular_values)
                                else (
                                    -1
                                    if not np.any(concentration_regular_values)
                                    else 0
                                )
                            )
                        longitudinal_kpl_values.append(interval_kpl)
                        longitudinal_modes.append(
                            1
                            if np.all(regular_values)
                            else -1 if not np.any(regular_values) else 0
                        )
                longitudinal_groups.append(longitudinal_group)

                rf_value = native_rf_hz[block_interval]
                if rf_value == 0.0:
                    rf_groups.append(-1)
                else:
                    rf_key_value = (
                        complex(
                            rf_hz[block_interval] if spatial_tx_active else rf_value
                        ),
                        float(execution_plan.dt_s[block_interval]),
                    )
                    rf_group = rf_group_by_key.get(rf_key_value)
                    if rf_group is None:
                        rf_group = len(rf_values)
                        rf_group_by_key[rf_key_value] = rf_group
                        if spatial_tx_active:
                            prepared_rf = spatial_rf_coefficients(*rf_key_value)
                        else:
                            scalar_rf = _prepare_rf_rotation(*rf_key_value)
                            prepared_rf = tuple(
                                np.asarray([coefficient], dtype=np.float64)
                                for coefficient in scalar_rf
                            )
                        rf_values.append(prepared_rf)
                    rf_groups.append(rf_group)

            transverse_factor_table = np.ascontiguousarray(
                np.stack(
                    [np.stack(value, axis=0) for value in factor_values],
                    axis=0,
                ),
                dtype=np.complex128,
            )

            def stack_longitudinal(index, dtype=np.float64):
                return np.ascontiguousarray(
                    np.stack([value[index] for value in longitudinal_values]),
                    dtype=dtype,
                )

            def stack_prepared(values, index, dtype=np.float64):
                return np.ascontiguousarray(
                    np.stack([value[index] for value in values]),
                    dtype=dtype,
                )

            def stack_source(values, index):
                return np.ascontiguousarray(
                    np.stack([value[4][index] for value in values]),
                    dtype=np.float64,
                )

            def stack_multi(values, index):
                return np.ascontiguousarray(
                    np.stack([value[index] for value in values]),
                    dtype=np.float64,
                )

            def stack_multi_source(values, index):
                return np.ascontiguousarray(
                    np.stack([value[3][index] for value in values]),
                    dtype=np.float64,
                )

            if rf_values:
                coefficient_count = 4 if spatial_tx_active else 5
                rf_tables = [
                    np.ascontiguousarray(
                        np.stack([value[index] for value in rf_values], axis=0),
                        dtype=np.float64,
                    )
                    for index in range(coefficient_count)
                ]
                if spatial_tx_active:
                    rf_tables.append(np.zeros_like(rf_tables[0]))
            else:
                rf_tables = [np.zeros((1, 1), dtype=np.float64) for _ in range(5)]

            action_offsets = [0]
            action_kinds = []
            action_payloads = []
            for block_interval in range(first_interval, end_interval):
                state_index = block_interval + 1
                first_action = int(
                    np.searchsorted(
                        execution_plan.action_state_indices,
                        state_index,
                        side="left",
                    )
                )
                last_action = int(
                    np.searchsorted(
                        execution_plan.action_state_indices,
                        state_index,
                        side="right",
                    )
                )
                action_kinds.extend(
                    execution_plan.action_kinds[first_action:last_action]
                )
                action_payloads.extend(
                    execution_plan.action_payload_indices[first_action:last_action]
                )
                action_offsets.append(len(action_kinds))

            # ADC samples are time ordered, so one block covers a contiguous
            # sample range. Kernels index the window with block-local payloads.
            block_action_kinds = np.asarray(action_kinds, dtype=np.int8)
            block_action_payloads = np.array(action_payloads, dtype=np.int64)
            block_adc = block_action_kinds == ACTION_ADC
            if np.any(block_adc):
                adc_first = int(block_action_payloads[block_adc].min())
                adc_end = int(block_action_payloads[block_adc].max()) + 1
                block_action_payloads[block_adc] -= adc_first
            else:
                adc_first = adc_end = 0
            parent_signal = np.zeros(
                (
                    parent_active_count,
                    phantom.n_species,
                    n_rx_coils,
                    max(1, adc_end - adc_first),
                ),
                dtype=np.complex128,
            )
            kpl_table = (
                None
                if native_full_plan_multi_pool
                else np.ascontiguousarray(
                    np.stack(longitudinal_kpl_values), dtype=np.float64
                )
            )
            action_arrays = (
                np.ascontiguousarray(action_offsets, dtype=np.int32),
                block_action_kinds,
                block_action_payloads,
                np.ascontiguousarray(
                    all_adc_demodulation[adc_first : max(adc_first + 1, adc_end)]
                ),
                np.ascontiguousarray(rx_sensitivities, dtype=np.complex128),
                np.ascontiguousarray(spin_signal_weights, dtype=np.float64),
                parent_signal,
                checkpoint_states,
                spins_per_voxel,
                float(signal_scale),
                spoiler_mode == "ideal",
            )
            worker_threads = (
                native_threads if sequence_kernel == "native_parallel" else 1
            )
            if native_full_plan_source_model or native_full_plan_multi_pool:
                spatial_source_table = None
                if spatial_inflow_active:
                    spatial_source_table = np.empty(
                        (
                            8,
                            end_interval - first_interval,
                            n_simulated_spins,
                        ),
                        dtype=np.float64,
                    )
                    for local_interval, block_interval in enumerate(
                        range(first_interval, end_interval)
                    ):
                        start_time = float(
                            execution_plan.interval_start_s[block_interval]
                        )
                        middle_time = float(
                            execution_plan.interval_mid_s[block_interval]
                        )
                        end_time = float(execution_plan.interval_end_s[block_interval])
                        spatial_source_table[0:2, local_interval] = (
                            _time_curve_interval_values(
                                inflow_curve,
                                start_time,
                                middle_time,
                                inflow_arrival_delay,
                            )
                        )
                        spatial_source_table[2:4, local_interval] = (
                            _time_curve_interval_values(
                                inflow_curve,
                                middle_time,
                                end_time,
                                inflow_arrival_delay,
                            )
                        )
                        if inflow_polarization_curve is None:
                            spatial_source_table[4:8, local_interval] = 1.0
                        else:
                            spatial_source_table[4:6, local_interval] = (
                                _time_curve_interval_values(
                                    inflow_polarization_curve,
                                    start_time,
                                    middle_time,
                                    inflow_arrival_delay,
                                )
                            )
                            spatial_source_table[6:8, local_interval] = (
                                _time_curve_interval_values(
                                    inflow_polarization_curve,
                                    middle_time,
                                    end_time,
                                    inflow_arrival_delay,
                                )
                            )
                    empty_source = np.zeros(
                        end_interval - first_interval, dtype=np.float64
                    )
                    source_tables = (empty_source,) * 8
                else:
                    block_interval_count = end_interval - first_interval
                    if execution_plan.source_rate_samples is None:
                        source_samples = np.zeros(
                            (block_interval_count, 4), dtype=np.float64
                        )
                    else:
                        source_samples = execution_plan.source_rate_samples[
                            first_interval:end_interval
                        ]
                    if execution_plan.source_polarization_samples is None:
                        polarization_samples = np.ones(
                            (block_interval_count, 4), dtype=np.float64
                        )
                    else:
                        polarization_samples = (
                            execution_plan.source_polarization_samples[
                                first_interval:end_interval
                            ]
                        )
                    source_tables = tuple(
                        np.ascontiguousarray(samples[:, index], dtype=np.float64)
                        for samples in (source_samples, polarization_samples)
                        for index in range(4)
                    )
                if native_full_plan_multi_pool:
                    native_plan_multi_pool_driver(
                        state,
                        transverse_state,
                        plan_concentration_state,
                        transverse_factor_table,
                        np.ascontiguousarray(
                            np.stack(multi_conversion_values), dtype=np.float64
                        ),
                        stack_multi(multi_longitudinal_values, 0),
                        stack_multi(multi_longitudinal_values, 1),
                        stack_multi(multi_longitudinal_values, 2),
                        stack_multi_source(multi_longitudinal_values, 0),
                        stack_multi_source(multi_longitudinal_values, 1),
                        stack_multi_source(multi_longitudinal_values, 2),
                        stack_multi_source(multi_longitudinal_values, 3),
                        stack_multi(multi_concentration_values, 0),
                        stack_multi(multi_concentration_values, 1),
                        stack_multi(multi_concentration_values, 2),
                        stack_multi_source(multi_concentration_values, 0),
                        stack_multi_source(multi_concentration_values, 1),
                        stack_multi_source(multi_concentration_values, 2),
                        stack_multi_source(multi_concentration_values, 3),
                        np.ascontiguousarray(longitudinal_durations, dtype=np.float64),
                        np.ascontiguousarray(longitudinal_groups, dtype=np.int32),
                        np.ascontiguousarray(first_factor_groups, dtype=np.int32),
                        np.ascontiguousarray(second_factor_groups, dtype=np.int32),
                        np.ascontiguousarray(rf_groups, dtype=np.int32),
                        *rf_tables,
                        *source_tables,
                        plan_inflow_delivery,
                        plan_equilibrium,
                        native_full_plan_source_mode,
                        *action_arrays,
                        worker_threads,
                        spatial_source_table,
                    )
                else:
                    native_plan_inflow_driver(
                        state,
                        transverse_state,
                        plan_concentration_state,
                        kpl,
                        plan_inflow_delivery,
                        transverse_factor_table,
                        stack_longitudinal(0),
                        np.ascontiguousarray(
                            [value[1] for value in longitudinal_values],
                            dtype=np.float64,
                        ),
                        stack_longitudinal(2),
                        stack_longitudinal(3, dtype=np.bool_),
                        stack_source(longitudinal_values, 0),
                        stack_source(longitudinal_values, 1),
                        stack_source(longitudinal_values, 2),
                        stack_source(longitudinal_values, 3),
                        stack_prepared(concentration_values, 0),
                        np.ascontiguousarray(
                            [value[1] for value in concentration_values],
                            dtype=np.float64,
                        ),
                        stack_prepared(concentration_values, 2),
                        stack_prepared(concentration_values, 3, dtype=np.bool_),
                        stack_source(concentration_values, 0),
                        stack_source(concentration_values, 1),
                        stack_source(concentration_values, 2),
                        stack_source(concentration_values, 3),
                        np.ascontiguousarray(longitudinal_durations, dtype=np.float64),
                        np.ascontiguousarray(longitudinal_modes, dtype=np.int32),
                        np.ascontiguousarray(concentration_modes, dtype=np.int32),
                        np.ascontiguousarray(longitudinal_groups, dtype=np.int32),
                        np.ascontiguousarray(first_factor_groups, dtype=np.int32),
                        np.ascontiguousarray(second_factor_groups, dtype=np.int32),
                        np.ascontiguousarray(rf_groups, dtype=np.int32),
                        *rf_tables,
                        *source_tables,
                        plan_equilibrium,
                        worker_threads,
                        *action_arrays,
                        kpl_table,
                        spatial_source_table,
                        native_full_plan_source_mode,
                    )
            else:
                native_plan_driver(
                    state,
                    transverse_state,
                    transverse_factor_table,
                    kpl_table,
                    stack_longitudinal(0),
                    np.ascontiguousarray(
                        [value[1] for value in longitudinal_values],
                        dtype=np.float64,
                    ),
                    stack_longitudinal(2),
                    stack_longitudinal(3, dtype=np.bool_),
                    np.ascontiguousarray(longitudinal_durations, dtype=np.float64),
                    np.ascontiguousarray(longitudinal_modes, dtype=np.int32),
                    np.ascontiguousarray(longitudinal_groups, dtype=np.int32),
                    np.ascontiguousarray(first_factor_groups, dtype=np.int32),
                    np.ascontiguousarray(second_factor_groups, dtype=np.int32),
                    np.ascontiguousarray(rf_groups, dtype=np.int32),
                    *rf_tables,
                    *action_arrays,
                    worker_threads,
                )
            if adc_end > adc_first:
                window = parent_signal[..., : adc_end - adc_first]
                if _parent_resolved_signal:
                    resolved_signal[..., adc_first:adc_end] += window
                else:
                    reduced_signal[..., adc_first:adc_end] += window.sum(axis=0)
            native_full_plan_blocks += 1
            native_full_plan_intervals += end_interval - first_interval
            if progress_callback is not None:
                progress_callback(end_interval, interval_count)
            preview_step = end_interval // progress_stride
            if preview_callback is not None and (
                preview_step > last_preview_step or end_interval == interval_count
            ):
                last_preview_step = preview_step
                preview_callback(
                    end_interval / interval_count, combined_signal_snapshot()
                )

        action_cursor = execution_plan.n_actions
        native_rf_fused_preapplied_end = interval_count
        return True

    native_spin_major_chunks = 0
    native_spin_major_intervals = 0
    native_spin_major_threads = 0
    native_spin_major_table_s = 0.0
    native_spin_major_kernel_s = 0.0
    rf_propagator_metadata = {
        "native_spin_major_rf_propagator_enabled": False,
        "native_spin_major_rf_propagator_fallback_reason": None,
        "native_spin_major_rf_propagator_keys": 0,
    }
    rf_propagator_stats = {
        "native_spin_major_rf_propagator_builds": 0,
        "native_spin_major_rf_propagator_applications": 0,
        "native_spin_major_rf_propagator_stepwise_segments": 0,
        "native_spin_major_rf_propagator_rotated_applications": 0,
        "native_spin_major_rf_propagator_stepwise_tiles": 0,
        "native_spin_major_rf_propagator_bytes": 0,
        "native_spin_major_rf_propagator_build_s": 0.0,
    }

    def plan_rf_propagators(source_mode, equilibrium):
        """Plan cached RF segments and allocate their propagator entries."""
        rf_segments = int(np.count_nonzero(execution_plan.segment_kinds == SEGMENT_RF))
        rf_propagator_stats["native_spin_major_rf_propagator_stepwise_segments"] = (
            rf_segments
        )
        if not native_spin_major_rf_cache_requested:
            rf_propagator_metadata[
                "native_spin_major_rf_propagator_fallback_reason"
            ] = "disabled by request"
            return None, None
        from .dynamic_spin_major_cy import propagator_columns, propagator_values

        track_concentration = concentration_state is not None
        values = propagator_values(
            phantom.n_species, track_concentration, source_mode != 2, source_mode == 0
        )
        tiles = -(-n_simulated_spins // native_spin_major_tile_spins)
        entry_shape = (tiles, values, native_spin_major_tile_spins)
        plan, reason = _spin_major_rf_propagator_plan(
            execution_plan,
            native_rf_hz,
            source_mode=source_mode,
            equilibrium=equilibrium,
            track_concentration=track_concentration,
            columns=propagator_columns(
                track_concentration, source_mode != 2, source_mode == 0
            ),
            entry_bytes=int(np.prod(entry_shape)) * 8,
            budget_bytes=_spin_major_rf_propagator_budget_bytes,
            spatial_sources=spatial_inflow_active,
        )
        if plan is None:
            rf_propagator_metadata[
                "native_spin_major_rf_propagator_fallback_reason"
            ] = reason
            return None, None
        propagators = np.empty((plan.capacity,) + entry_shape, dtype=np.float64)
        rf_propagator_metadata.update(
            {
                "native_spin_major_rf_propagator_enabled": True,
                "native_spin_major_rf_propagator_keys": plan.keys,
            }
        )
        rf_propagator_stats.update(
            {
                "native_spin_major_rf_propagator_builds": plan.builds,
                "native_spin_major_rf_propagator_applications": int(plan.first.size),
                "native_spin_major_rf_propagator_stepwise_segments": (
                    rf_segments - int(plan.first.size)
                ),
                "native_spin_major_rf_propagator_rotated_applications": plan.rotated,
                "native_spin_major_rf_propagator_bytes": int(propagators.nbytes),
            }
        )
        return plan, propagators

    def execute_native_spin_major():
        nonlocal action_cursor
        nonlocal native_rf_fused_preapplied_end
        nonlocal native_full_plan_blocks, native_full_plan_intervals
        nonlocal native_spin_major_chunks, native_spin_major_intervals
        nonlocal native_spin_major_threads
        nonlocal native_spin_major_table_s, native_spin_major_kernel_s
        if not native_spin_major_enabled:
            return False

        source_mode = native_full_plan_source_mode
        equilibrium = (
            float(phantom.equilibrium_polarization)
            if concentration_state is not None
            else 0.0
        )
        delivery = None
        if source_mode != 2:
            delivery = (
                np.zeros(n_simulated_spins, dtype=np.float64)
                if inflow_delivery is None
                else np.ascontiguousarray(inflow_delivery, dtype=np.float64)
            )
        signal_target = species_signal[0] if _parent_resolved_signal else species_signal
        adc_states = execution_plan.action_state_indices[
            execution_plan.action_kinds == ACTION_ADC
        ]
        interval_limit, adc_limit = native_spin_major_limits
        worker_threads = native_threads if sequence_kernel == "native_parallel" else 1
        native_spin_major_threads = worker_threads
        conversion = (
            coefficient_kpl[None]
            if multi_pool_conversion is None
            else multi_pool_conversion
        )
        spatial_arguments = {}
        if spatial_inflow_active and source_mode != 2:

            def curve_argument(curve):
                if curve is None:
                    return None
                return (
                    np.ascontiguousarray(curve.times_s, dtype=np.float64),
                    np.ascontiguousarray(curve.values, dtype=np.float64),
                    curve.interpolation == "step",
                    curve.outside == "hold",
                )

            spatial_arguments = dict(
                arrival_delay_s=np.ascontiguousarray(
                    inflow_arrival_delay, dtype=np.float64
                ),
                spatial_mode=source_mode,
                rate_curve=curve_argument(inflow_curve),
                polarization_curve=curve_argument(inflow_polarization_curve),
            )
        spin_arguments = dict(
            tx_sensitivity=(
                np.ascontiguousarray(tx_sensitivity, dtype=np.complex128)
                if spatial_tx_active
                else None
            ),
            positions_m=np.ascontiguousarray(positions, dtype=np.float64),
            b0_hz=np.ascontiguousarray(b0, dtype=np.float64),
            conversion=np.ascontiguousarray(conversion, dtype=np.float64),
            delivery=delivery,
            rx_sensitivities=np.ascontiguousarray(
                rx_sensitivities, dtype=np.complex128
            ),
            signal_weights=np.ascontiguousarray(spin_signal_weights, dtype=np.float64),
            transverse_state=transverse_state,
            state=state,
            concentration_state=concentration_state,
            checkpoint_states=checkpoint_states,
            action_kinds=execution_plan.action_kinds,
            action_payloads=execution_plan.action_payload_indices,
            r1=np.ascontiguousarray(coefficient_r1, dtype=np.float64),
            t2_s=np.ascontiguousarray(t2[:, 0], dtype=np.float64),
            pool_offset_hz=np.ascontiguousarray(pool_offsets, dtype=np.float64),
            equilibrium=equilibrium,
            ideal_crush=spoiler_mode == "ideal",
            concentration_sources=source_mode == 0,
            num_threads=worker_threads,
            num_blocks=_SPIN_MAJOR_WORK_BLOCKS,
            **spatial_arguments,
        )
        rf_cache = {}
        table_started = perf_counter()
        rf_plan, propagators = plan_rf_propagators(source_mode, equilibrium)
        native_spin_major_table_s += perf_counter() - table_started
        build_seconds = 0.0
        stepwise_tiles = 0
        first_interval = 0
        while first_interval < interval_count:
            if cancel_callback is not None and cancel_callback():
                raise RuntimeError("sequence simulation cancelled")
            table_started = perf_counter()
            end_interval = _spin_major_slice_end(
                first_interval,
                interval_count,
                adc_states,
                adc_limit,
                interval_limit,
                *(() if rf_plan is None else (rf_plan.first, rf_plan.end)),
            )
            segments = (
                {}
                if rf_plan is None
                else _spin_major_rf_slice(
                    rf_plan, propagators, first_interval, end_interval
                )
            )
            tables = _spin_major_plan_slice(
                execution_plan,
                native_rf_hz,
                first_interval,
                end_interval,
                source_mode=source_mode,
                equilibrium=equilibrium,
                rf_cache=rf_cache,
                spatial_inflow=bool(spatial_arguments),
            )
            adc_first = tables.pop("adc_first_payload")
            adc_count = tables.pop("adc_count")
            adc_signal = np.zeros(
                (phantom.n_species, n_rx_coils, adc_count), dtype=np.complex128
            )
            kernel_started = perf_counter()
            native_spin_major_table_s += kernel_started - table_started
            chunk_build_seconds, chunk_stepwise_tiles = native_spin_major_chunk(
                **tables,
                **spin_arguments,
                **segments,
                adc_first_payload=adc_first,
                adc_signal=adc_signal,
            )
            build_seconds += chunk_build_seconds
            stepwise_tiles += chunk_stepwise_tiles
            native_spin_major_kernel_s += perf_counter() - kernel_started
            if adc_count:
                payloads = slice(adc_first, adc_first + adc_count)
                received = adc_signal * signal_scale * adc_demodulation[payloads]
                if n_rx_coils == 1:
                    signal_target[:, payloads] = received[:, 0, :]
                else:
                    signal_target[:, :, payloads] = received
            native_spin_major_chunks += 1
            native_spin_major_intervals += end_interval - first_interval
            first_interval = end_interval
            if progress_callback is not None:
                progress_callback(end_interval, interval_count)
            if preview_callback is not None:
                preview_callback(
                    end_interval / interval_count, combined_signal_snapshot()
                )
        # The executor sums build time over threads; report wall-clock share.
        rf_propagator_stats["native_spin_major_rf_propagator_build_s"] = (
            build_seconds / worker_threads
        )
        rf_propagator_stats["native_spin_major_rf_propagator_stepwise_tiles"] = (
            stepwise_tiles
        )
        native_full_plan_blocks += native_spin_major_chunks
        native_full_plan_intervals += native_spin_major_intervals
        action_cursor = execution_plan.n_actions
        native_rf_fused_preapplied_end = interval_count
        return True

    native_spin_major_executed = execute_native_spin_major()
    native_full_plan_executed = native_spin_major_executed or execute_native_full_plan()
    if status_callback is not None and native_spin_major_executed:
        status_callback(
            "Executed dynamic free evolution, RF, ADC, crushers, and "
            f"checkpoints in {native_spin_major_chunks:,} bounded native plan "
            f"slice(s), spin-major with {native_spin_major_threads} thread(s)."
        )
    elif status_callback is not None and native_full_plan_executed:
        status_callback(
            "Executed dynamic free evolution, RF, ADC, crushers, and "
            f"checkpoints in {native_full_plan_blocks:,} bounded native plan "
            "block(s)."
        )

    interval_start = 0.0
    for interval in range(interval_count):
        if (
            not native_full_plan_executed
            and cancel_callback is not None
            and cancel_callback()
        ):
            raise RuntimeError("sequence simulation cancelled")
        if (
            native_rf_fused_enabled
            and interval >= native_rf_fused_preapplied_end
            and native_rf_hz[interval] != 0.0
        ):
            preapply_native_rf_fused_block(interval)
        if interval < native_rf_fused_preapplied_end:
            interval_end = float(compiled.interval_end_s[interval])
            process_boundary_actions(interval + 1)
            interval_start = interval_end
            if native_full_plan_executed:
                continue
            if progress_callback is not None and (
                interval % progress_stride == 0 or interval + 1 == interval_count
            ):
                progress_callback(interval + 1, interval_count)
            if preview_callback is not None and (
                interval % progress_stride == 0 or interval + 1 == interval_count
            ):
                preview_callback(
                    (interval + 1) / interval_count,
                    combined_signal_snapshot(),
                )
            continue
        dt = execution_plan.dt_s[interval]
        interval_end = float(execution_plan.interval_end_s[interval])
        interval_mid = float(execution_plan.interval_mid_s[interval])
        gradient = gradient_hz_per_m[interval]
        conversion_active = bool(execution_plan.conversion_active[interval])
        interval_kpl = kpl if conversion_active else zero_kpl
        interval_passive_conversion = (
            passive_conversion if conversion_active else zero_passive_conversion
        )
        coefficient_interval_kpl = (
            coefficient_kpl if conversion_active else coefficient_zero_kpl
        )
        interval_longitudinal_scratch = (
            longitudinal_scratch if conversion_active else inactive_longitudinal_scratch
        )
        if (
            sequence_kernel == "native_parallel"
            and native_longitudinal_block is not None
            and native_block_enabled
            and interval >= native_preapplied_end
            and rf_hz[interval] == 0.0
        ):
            preapply_native_longitudinal_block(interval)
        longitudinal_preapplied = (
            sequence_kernel == "native_parallel"
            and native_longitudinal_block is not None
            and interval < native_preapplied_end
            and rf_hz[interval] == 0.0
        )
        if sequence_kernel == "reference":
            gradient_frequency = _gradient_frequency_hz(positions, gradient)
            static_frequencies = (
                b0[None, :] + pool_offsets[:, None] + gradient_frequency[None, :]
            )

            def phase_cycles(start_s, end_s, duration_s):
                phase = static_frequencies * duration_s
                if phantom.dynamic_b0 is not None:
                    dynamic_integral = phantom.dynamic_b0.offset_curve_hz.integral(
                        start_s, end_s
                    )
                    phase = phase + (
                        dynamic_b0_pool_scale
                        * dynamic_b0_scale[None, :]
                        * dynamic_integral
                    )
                return phase

            def source_values(start_s, end_s):
                if inflow_curve is None:
                    return None, None, None, None
                start_value, end_value = _time_curve_interval_values(
                    inflow_curve,
                    start_s,
                    end_s,
                    inflow_arrival_delay,
                )
                if inflow_polarization_curve is None:
                    return (
                        inflow_delivery * start_value,
                        inflow_delivery * end_value,
                        None,
                        None,
                    )
                polarization_start, polarization_end = _time_curve_interval_values(
                    inflow_polarization_curve,
                    start_s,
                    end_s,
                    inflow_arrival_delay,
                )
                concentration_start = inflow_delivery * start_value
                concentration_end = inflow_delivery * end_value
                return (
                    concentration_start * polarization_start,
                    concentration_end * polarization_end,
                    concentration_start,
                    concentration_end,
                )

            source_start, source_mid, concentration_start, concentration_mid = (
                source_values(interval_start, interval_mid)
            )
            _free_step(
                state,
                phase_cycles(
                    interval_start,
                    interval_mid,
                    execution_plan.interval_half_s[interval, 0],
                ),
                t2,
                interval_kpl,
                r1,
                dt / 2.0,
                source_start,
                source_mid,
                concentration_state=concentration_state,
                concentration_source_start=concentration_start,
                concentration_source_end=concentration_mid,
                equilibrium_polarization=phantom.equilibrium_polarization,
                passive_conversion=interval_passive_conversion,
            )
            if nonunity_tx_active and rf_hz[interval] != 0.0:
                _rf_rotate_spatial(state, rf_hz[interval], tx_sensitivity, dt)
            else:
                _rf_rotate(state, rf_hz[interval], dt)
            source_mid, source_end, concentration_mid, concentration_end = (
                source_values(interval_mid, interval_end)
            )
            _free_step(
                state,
                phase_cycles(
                    interval_mid,
                    interval_end,
                    execution_plan.interval_half_s[interval, 1],
                ),
                t2,
                interval_kpl,
                r1,
                dt / 2.0,
                source_mid,
                source_end,
                concentration_state=concentration_state,
                concentration_source_start=concentration_mid,
                concentration_source_end=concentration_end,
                equilibrium_polarization=phantom.equilibrium_polarization,
                passive_conversion=interval_passive_conversion,
            )
        else:
            half_duration = real_type(dt / 2.0)
            coefficient_half_duration = dt / 2.0
            longitudinal_key = (
                float(coefficient_half_duration),
                conversion_active,
            )
            longitudinal_prepared = coefficient_cache.get(longitudinal_key)
            if longitudinal_prepared is None:
                longitudinal_prepared = _prepare_longitudinal_step(
                    coefficient_interval_kpl,
                    coefficient_r1[0],
                    coefficient_r1[1],
                    coefficient_half_duration,
                    with_source=phantom.pyruvate_inflow is not None,
                )
                coefficient_cache.put(longitudinal_key, longitudinal_prepared)
            concentration_longitudinal_prepared = None
            concentration_interval_scratch = None
            if concentration_state is not None:
                concentration_key = (
                    "concentration",
                    float(coefficient_half_duration),
                    conversion_active,
                )
                concentration_longitudinal_prepared = coefficient_cache.get(
                    concentration_key
                )
                if concentration_longitudinal_prepared is None:
                    concentration_longitudinal_prepared = _prepare_longitudinal_step(
                        coefficient_interval_kpl,
                        0.0,
                        0.0,
                        coefficient_half_duration,
                        with_source=True,
                    )
                    coefficient_cache.put(
                        concentration_key, concentration_longitudinal_prepared
                    )
                concentration_interval_scratch = (
                    concentration_longitudinal_scratch
                    if conversion_active
                    else inactive_concentration_longitudinal_scratch
                )
            first_phase_duration, second_phase_duration = (
                float(value) for value in execution_plan.interval_half_s[interval]
            )
            if phantom.dynamic_b0 is None:
                first_dynamic_integral = second_dynamic_integral = None
            else:
                first_dynamic_integral = real_type(
                    execution_plan.dynamic_b0_first_half_integral_hz_s[interval]
                )
                second_dynamic_integral = real_type(
                    execution_plan.dynamic_b0_second_half_integral_hz_s[interval]
                )
            gradient_key = tuple(float(value) for value in gradient)
            first_transverse_key = (
                float(half_duration),
                float(first_phase_duration),
                gradient_key,
                first_dynamic_integral,
            )
            second_transverse_key = (
                float(half_duration),
                float(second_phase_duration),
                gradient_key,
                second_dynamic_integral,
            )
            first_transverse_factors = transverse_cache.get(first_transverse_key)
            second_transverse_factors = transverse_cache.get(second_transverse_key)
            static_frequencies = None
            if first_transverse_factors is None or second_transverse_factors is None:
                gradient_frequency = _gradient_frequency_hz(positions, gradient)
                static_frequencies = (
                    b0[None, :] + pool_offsets[:, None] + gradient_frequency[None, :]
                )
            if first_transverse_factors is None:
                first_phase = static_frequencies * first_phase_duration
                if first_dynamic_integral is not None:
                    first_phase = first_phase + (
                        dynamic_b0_pool_scale
                        * dynamic_b0_scale[None, :]
                        * first_dynamic_integral
                    )
                first_transverse_factors = prepare_transverse_factors(
                    first_phase, half_duration
                )
                transverse_cache.put(first_transverse_key, first_transverse_factors)
            if first_transverse_key == second_transverse_key:
                second_transverse_factors = first_transverse_factors
            elif second_transverse_factors is None:
                second_phase = static_frequencies * second_phase_duration
                if second_dynamic_integral is not None:
                    second_phase = second_phase + (
                        dynamic_b0_pool_scale
                        * dynamic_b0_scale[None, :]
                        * second_dynamic_integral
                    )
                second_transverse_factors = prepare_transverse_factors(
                    second_phase, half_duration
                )
                transverse_cache.put(second_transverse_key, second_transverse_factors)
            first_phase = second_phase = None

            if inflow_curve is None:
                source_start = source_mid = source_end = None
                concentration_start = concentration_mid = concentration_end = None
            else:
                if execution_plan.source_rate_samples is None:
                    start_value, mid_value = _time_curve_interval_values(
                        inflow_curve,
                        interval_start,
                        interval_mid,
                        inflow_arrival_delay,
                    )
                else:
                    start_value, mid_value = execution_plan.source_rate_samples[
                        interval, 0:2
                    ]
                start_value = np.asarray(start_value, dtype=real_dtype)
                mid_value = np.asarray(mid_value, dtype=real_dtype)
                if inflow_polarization_curve is None:
                    source_start = inflow_delivery * start_value
                    source_mid = inflow_delivery * mid_value
                    concentration_start = concentration_mid = None
                else:
                    if execution_plan.source_polarization_samples is None:
                        polarization_start, polarization_mid = (
                            _time_curve_interval_values(
                                inflow_polarization_curve,
                                interval_start,
                                interval_mid,
                                inflow_arrival_delay,
                            )
                        )
                    else:
                        polarization_start, polarization_mid = (
                            execution_plan.source_polarization_samples[interval, 0:2]
                        )
                    polarization_start = np.asarray(
                        polarization_start, dtype=real_dtype
                    )
                    polarization_mid = np.asarray(polarization_mid, dtype=real_dtype)
                    concentration_start = inflow_delivery * start_value
                    concentration_mid = inflow_delivery * mid_value
                    source_start = concentration_start * polarization_start
                    source_mid = concentration_mid * polarization_mid
            if native_concentration_inflow_step is not None:
                for pool in range(phantom.n_species):
                    transverse_state[pool] *= first_transverse_factors[pool]
                native_concentration_inflow_step(
                    state,
                    concentration_state,
                    interval_kpl,
                    longitudinal_prepared[0],
                    longitudinal_prepared[1],
                    longitudinal_prepared[2],
                    longitudinal_prepared[3],
                    longitudinal_prepared[4][0],
                    longitudinal_prepared[4][1],
                    longitudinal_prepared[4][2],
                    longitudinal_prepared[4][3],
                    concentration_longitudinal_prepared[0],
                    concentration_longitudinal_prepared[1],
                    concentration_longitudinal_prepared[2],
                    concentration_longitudinal_prepared[3],
                    concentration_longitudinal_prepared[4][0],
                    concentration_longitudinal_prepared[4][1],
                    concentration_longitudinal_prepared[4][2],
                    concentration_longitudinal_prepared[4][3],
                    source_start,
                    source_mid,
                    concentration_start,
                    concentration_mid,
                    half_duration,
                    phantom.equilibrium_polarization,
                    interval_longitudinal_scratch[3],
                    concentration_interval_scratch[3],
                    native_longitudinal_threads,
                )
            elif native_longitudinal_step is None:
                _free_step(
                    state,
                    first_phase,
                    t2,
                    interval_kpl,
                    r1,
                    half_duration,
                    source_start,
                    source_mid,
                    transverse_factors=first_transverse_factors,
                    longitudinal_prepared=longitudinal_prepared,
                    transverse_state=transverse_state,
                    longitudinal_scratch=interval_longitudinal_scratch,
                    concentration_state=concentration_state,
                    concentration_source_start=concentration_start,
                    concentration_source_end=concentration_mid,
                    concentration_longitudinal_prepared=(
                        concentration_longitudinal_prepared
                    ),
                    concentration_longitudinal_scratch=(concentration_interval_scratch),
                    equilibrium_polarization=phantom.equilibrium_polarization,
                    passive_conversion=interval_passive_conversion,
                )
            else:
                for pool in range(phantom.n_species):
                    transverse_state[pool] *= first_transverse_factors[pool]
                if not longitudinal_preapplied:
                    native_longitudinal_step(
                        state,
                        kpl,
                        longitudinal_prepared[0],
                        longitudinal_prepared[1],
                        longitudinal_prepared[2],
                        longitudinal_prepared[3],
                        half_duration,
                        regular_mode,
                    )
            if rf_hz[interval] != 0.0:
                if spatial_tx_active and native_spatial_rf_rotation_block is not None:
                    spatial_prepared = spatial_rf_coefficients(rf_hz[interval], dt)
                    native_spatial_rf_rotation_block(
                        state,
                        transverse_state,
                        spatial_prepared[0],
                        spatial_prepared[1],
                        spatial_prepared[2],
                        spatial_prepared[3],
                        native_rf_threads,
                    )
                elif nonunity_tx_active and (
                    sequence_kernel == "optimized" or spatial_tx_active
                ):
                    sync_transverse_to_state()
                    _rf_rotate_spatial(
                        state,
                        rf_hz[interval],
                        tx_sensitivity,
                        dt,
                    )
                    transverse_state[:] = state[:, :, 0] + 1j * state[:, :, 1]
                elif native_rf_rotation_block is None:
                    sync_transverse_to_state()
                    _rf_rotate(state, native_rf_hz[interval], dt)
                    transverse_state[:] = state[:, :, 0] + 1j * state[:, :, 1]
                else:
                    rf_prepared = _prepare_rf_rotation(native_rf_hz[interval], dt)
                    if rf_prepared is not None:
                        native_rf_rotation_block(
                            state,
                            transverse_state,
                            rf_prepared[0],
                            rf_prepared[1],
                            rf_prepared[2],
                            rf_prepared[3],
                            rf_prepared[4],
                            native_rf_threads,
                        )
            if inflow_curve is not None:
                if execution_plan.source_rate_samples is None:
                    mid_value, end_value = _time_curve_interval_values(
                        inflow_curve,
                        interval_mid,
                        interval_end,
                        inflow_arrival_delay,
                    )
                else:
                    mid_value, end_value = execution_plan.source_rate_samples[
                        interval, 2:4
                    ]
                mid_value = np.asarray(mid_value, dtype=real_dtype)
                end_value = np.asarray(end_value, dtype=real_dtype)
                if inflow_polarization_curve is None:
                    source_mid = inflow_delivery * mid_value
                    source_end = inflow_delivery * end_value
                    concentration_mid = concentration_end = None
                else:
                    if execution_plan.source_polarization_samples is None:
                        polarization_mid, polarization_end = (
                            _time_curve_interval_values(
                                inflow_polarization_curve,
                                interval_mid,
                                interval_end,
                                inflow_arrival_delay,
                            )
                        )
                    else:
                        polarization_mid, polarization_end = (
                            execution_plan.source_polarization_samples[interval, 2:4]
                        )
                    polarization_mid = np.asarray(polarization_mid, dtype=real_dtype)
                    polarization_end = np.asarray(polarization_end, dtype=real_dtype)
                    concentration_mid = inflow_delivery * mid_value
                    concentration_end = inflow_delivery * end_value
                    source_mid = concentration_mid * polarization_mid
                    source_end = concentration_end * polarization_end
            if native_concentration_inflow_step is not None:
                for pool in range(phantom.n_species):
                    transverse_state[pool] *= second_transverse_factors[pool]
                native_concentration_inflow_step(
                    state,
                    concentration_state,
                    interval_kpl,
                    longitudinal_prepared[0],
                    longitudinal_prepared[1],
                    longitudinal_prepared[2],
                    longitudinal_prepared[3],
                    longitudinal_prepared[4][0],
                    longitudinal_prepared[4][1],
                    longitudinal_prepared[4][2],
                    longitudinal_prepared[4][3],
                    concentration_longitudinal_prepared[0],
                    concentration_longitudinal_prepared[1],
                    concentration_longitudinal_prepared[2],
                    concentration_longitudinal_prepared[3],
                    concentration_longitudinal_prepared[4][0],
                    concentration_longitudinal_prepared[4][1],
                    concentration_longitudinal_prepared[4][2],
                    concentration_longitudinal_prepared[4][3],
                    source_mid,
                    source_end,
                    concentration_mid,
                    concentration_end,
                    half_duration,
                    phantom.equilibrium_polarization,
                    interval_longitudinal_scratch[3],
                    concentration_interval_scratch[3],
                    native_longitudinal_threads,
                )
            elif native_longitudinal_step is None:
                _free_step(
                    state,
                    second_phase,
                    t2,
                    interval_kpl,
                    r1,
                    half_duration,
                    source_mid,
                    source_end,
                    transverse_factors=second_transverse_factors,
                    longitudinal_prepared=longitudinal_prepared,
                    transverse_state=transverse_state,
                    longitudinal_scratch=interval_longitudinal_scratch,
                    concentration_state=concentration_state,
                    concentration_source_start=concentration_mid,
                    concentration_source_end=concentration_end,
                    concentration_longitudinal_prepared=(
                        concentration_longitudinal_prepared
                    ),
                    concentration_longitudinal_scratch=(concentration_interval_scratch),
                    equilibrium_polarization=phantom.equilibrium_polarization,
                    passive_conversion=interval_passive_conversion,
                )
            else:
                for pool in range(phantom.n_species):
                    transverse_state[pool] *= second_transverse_factors[pool]
                if not longitudinal_preapplied:
                    native_longitudinal_step(
                        state,
                        kpl,
                        longitudinal_prepared[0],
                        longitudinal_prepared[1],
                        longitudinal_prepared[2],
                        longitudinal_prepared[3],
                        half_duration,
                        regular_mode,
                    )
        process_boundary_actions(interval + 1)
        interval_start = interval_end
        if progress_callback is not None and (
            interval % progress_stride == 0 or interval + 1 == interval_count
        ):
            progress_callback(interval + 1, interval_count)
        if preview_callback is not None and (
            interval % progress_stride == 0 or interval + 1 == interval_count
        ):
            preview_callback(
                (interval + 1) / interval_count,
                combined_signal_snapshot(),
            )

    sync_transverse_to_state()
    if sampling.enabled:
        state = np.sum(
            state.reshape(phantom.n_species, parent_active_count, spins_per_voxel, 3)
            * subvoxel_weights[None, None, :, None],
            axis=2,
        )
        if checkpoint_states.size:
            checkpoint_states = np.sum(
                checkpoint_states.reshape(
                    compiled.checkpoint_times_s.size,
                    phantom.n_species,
                    parent_active_count,
                    spins_per_voxel,
                    3,
                )
                * subvoxel_weights[None, None, None, :, None],
                axis=3,
            )
    block_runtime_stats = {
        "native_full_plan_blocks": native_full_plan_blocks,
        "native_full_plan_intervals": native_full_plan_intervals,
        "native_spin_major_chunks": native_spin_major_chunks,
        "native_spin_major_intervals": native_spin_major_intervals,
        "native_spin_major_parent_blocks": int(native_spin_major_executed),
        "native_rf_fused_blocks": native_rf_fused_blocks,
        "native_rf_fused_intervals": native_rf_fused_intervals,
        "native_spatial_rf_cache_hits": spatial_rf_cache_hits,
        "native_spatial_rf_cache_misses": spatial_rf_cache_misses,
        "native_spatial_rf_cache_bytes": (
            spatial_rf_cache.current_bytes if spatial_rf_cache is not None else 0
        ),
        "native_rf_fused_cache_bytes": (
            native_rf_fused_plan_cache.current_bytes
            if native_rf_fused_plan_cache is not None
            else 0
        ),
        **rf_propagator_stats,
    }
    if _block_result_mode == "minimal":
        return _DynamicBlockResult(
            active_indices=active,
            parent_species_signal=species_signal,
            final_pool_active=state,
            checkpoint_pool_active=(
                checkpoint_states if checkpoint_states.size else None
            ),
            runtime_stats=block_runtime_stats,
        )
    result_species_signal = (
        species_signal.sum(axis=0) if _parent_resolved_signal else species_signal
    )
    final_pool = np.zeros((phantom.n_species, phantom.nvoxels, 3), dtype=real_dtype)
    final_pool[:, active] = state
    final_pool = final_pool.reshape((phantom.n_species,) + phantom.shape + (3,))
    checkpoint_pool = None
    if checkpoint_states.size:
        checkpoint_pool = np.zeros(
            (
                compiled.checkpoint_times_s.size,
                phantom.n_species,
                phantom.nvoxels,
                3,
            ),
            dtype=checkpoint_dtype,
        )
        checkpoint_pool[:, :, active] = checkpoint_states
        checkpoint_pool = checkpoint_pool.reshape(
            (compiled.checkpoint_times_s.size, phantom.n_species) + phantom.shape + (3,)
        )
    dimensions = AcquisitionDimensions.from_program(program)
    spectroscopic_metadata = program.metadata.get("spectroscopic_acquisition")
    if spectroscopic_metadata is None:
        try:
            spectroscopic_metadata = infer_spectroscopic_acquisition(
                program, compiled=compiled
            ).to_metadata()
        except ValueError:
            spectroscopic_metadata = None
    spiral_metadata = program.metadata.get("spiral_acquisition")
    if spiral_metadata is None:
        try:
            spiral_metadata = infer_spiral_acquisition(
                program, compiled=compiled
            ).to_metadata()
        except ValueError:
            spiral_metadata = None
    cartesian_metadata = program.metadata.get("cartesian_acquisition")
    if cartesian_metadata is None:
        program_acquisition = program.metadata.get("acquisition")
        if (
            isinstance(program_acquisition, dict)
            and program_acquisition.get("type") == "cartesian_2d"
        ):
            cartesian_metadata = program_acquisition
    cartesian_frame_metadata = program.metadata.get("cartesian_acquisition_frames")
    cartesian_volume_metadata = program.metadata.get("cartesian_acquisition_volumes")
    if (
        spectroscopic_metadata is None
        and spiral_metadata is None
        and cartesian_metadata is None
    ):
        try:
            cartesian_metadata = infer_cartesian_acquisition(
                program, compiled=compiled
            ).to_metadata()
        except ValueError:
            if cartesian_frame_metadata is None:
                try:
                    cartesian_frame_metadata = infer_cartesian_acquisition_frames(
                        program, compiled=compiled
                    ).to_metadata()
                except ValueError:
                    cartesian_frame_metadata = None
    if (
        spectroscopic_metadata is None
        and cartesian_metadata is None
        and cartesian_frame_metadata is not None
        and cartesian_volume_metadata is None
    ):
        try:
            cartesian_volume_metadata = infer_cartesian_acquisition_volumes(
                program,
                compiled=compiled,
                frames=CartesianAcquisitionFrames.from_metadata(
                    cartesian_frame_metadata
                ),
            ).to_metadata()
        except ValueError:
            cartesian_volume_metadata = None
    sequence_waveforms = physical_sequence_waveforms(program, effective_nucleus)
    physical_field_maps = physical_b1_field_arrays(phantom, sequence_waveforms)
    result = SequenceSimulationResult(
        signal=result_species_signal.sum(axis=0),
        adc_times_s=compiled.adc_times_s,
        final_magnetization=final_pool.sum(axis=0),
        checkpoint_magnetization=(
            None if checkpoint_pool is None else checkpoint_pool.sum(axis=1)
        ),
        checkpoint_times_s=compiled.checkpoint_times_s,
        adc_gradient_moment_cyc_per_m=compiled.adc_gradient_moment_cyc_per_m,
        metadata={
            "dynamic_phantom": True,
            "phantom_metadata": dict(phantom.metadata),
            "phantom_family": phantom.metadata.get("phantom_family"),
            "transport_solver": phantom.metadata.get("transport_solver"),
            "breathing_model": phantom.metadata.get("breathing_model"),
            "ph_model": phantom.metadata.get("ph_model"),
            "additional_product_conversion": bool(
                passive_conversion is not None
                and np.any(np.asarray(passive_conversion) > 0)
            ),
            "pool_names": tuple(pool.name for pool in phantom.pools),
            "acquisition_dimensions": dimensions.to_metadata(),
            "spectroscopic_acquisition": spectroscopic_metadata,
            "cartesian_acquisition": cartesian_metadata,
            "cartesian_acquisition_frames": cartesian_frame_metadata,
            "cartesian_acquisition_volumes": cartesian_volume_metadata,
            "spiral_acquisition": spiral_metadata,
            "sequence_definitions": dict(program.metadata.get("definitions", {})),
            "pool_frequency_offsets_hz": tuple(float(value) for value in pool_offsets),
            "field_strength_t": field,
            "nucleus": effective_nucleus,
            "physical_waveform_nucleus": effective_nucleus,
            "physical_rf_unit": "G",
            "physical_gradient_unit": "T/m",
            "spectral_reference_ppm": phantom.spectral_reference_ppm,
            "sequence_reference_ppm": effective_reference_ppm,
            "spectral_window_center_ppm": phantom.spectral_window_center_ppm,
            "spectral_bandwidth_ppm": phantom.spectral_bandwidth_ppm,
            "spectral_points": phantom.spectral_points,
            "signal_weighting": signal_weighting,
            "spoiler_mode": spoiler_mode,
            "n_simulated_spins": n_simulated_spins,
            "spin_sampling": sampling.to_metadata(),
            "subvoxel_spin_counts_xyz": sampling.counts_xyz,
            "subvoxel_spins_per_voxel": spins_per_voxel,
            "tx_sensitivity": tx_classification,
            "constant_tx_sensitivity": (
                {
                    "real": float(constant_tx_sensitivity.real),
                    "imag": float(constant_tx_sensitivity.imag),
                }
                if tx_classification in {"unity", "constant"}
                else None
            ),
            "n_rx_coils": n_rx_coils,
            "simulation_timestep_s": float(simulation_timestep_s),
            "compiled_interval_count": interval_count,
            "execution_plan_segment_count": execution_plan.n_segments,
            "execution_plan_action_count": execution_plan.n_actions,
            "execution_plan_bytes": execution_plan.nbytes,
            "execution_plan_fingerprint": execution_plan.fingerprint,
            "sequence_compile_s": sequence_compile_s,
            "execution_plan_build_s": execution_plan_build_s,
            "state_dtype": real_dtype.name,
            "signal_dtype": complex_dtype.name,
            "sequence_kernel": sequence_kernel,
            "requested_sequence_kernel": requested_sequence_kernel,
            "numerical_contract": (
                "float64-close"
                if sequence_kernel in {"native_serial", "native_parallel"}
                and (
                    native_full_plan_executed
                    or native_rf_fused_enabled
                    or nonunity_tx_active
                )
                else "bitwise"
            ),
            "native_fallback_reason": native_fallback_reason,
            "native_longitudinal_fallback_reason": (
                None
                if native_full_plan_executed
                else native_longitudinal_fallback_reason
            ),
            "native_hybrid": bool(
                sequence_kernel in {"native_serial", "native_parallel"}
                and not native_full_plan_executed
                and native_rf_available
                and native_longitudinal_step is None
                and native_concentration_inflow_step is None
            ),
            "native_longitudinal_step_enabled": bool(
                native_longitudinal_step is not None
                or native_concentration_inflow_step is not None
            ),
            "native_concentration_inflow_step_enabled": (
                native_concentration_inflow_step is not None
            ),
            "native_longitudinal_block_enabled": bool(
                sequence_kernel == "native_parallel"
                and native_longitudinal_block is not None
                and native_block_enabled
            ),
            "native_parallel_threads": (
                native_threads if sequence_kernel == "native_parallel" else 1
            ),
            "effective_threads": (
                native_threads if sequence_kernel == "native_parallel" else 1
            ),
            "native_rf_threads": (
                native_rf_threads
                if sequence_kernel in {"native_serial", "native_parallel"}
                and native_rf_available
                else 1
            ),
            "native_longitudinal_threads": (
                native_longitudinal_threads
                if sequence_kernel in {"native_serial", "native_parallel"}
                and native_concentration_inflow_step is not None
                else 1
            ),
            "native_rf_block_enabled": (native_rf_available),
            "native_spatial_rf_block_enabled": bool(
                spatial_tx_active
                and native_spatial_rf_rotation_block is not None
                and sequence_kernel in {"native_serial", "native_parallel"}
            ),
            "native_rf_fused_block_enabled": native_rf_fused_enabled,
            "native_rf_fused_fallback_reason": native_rf_fused_fallback_reason,
            "native_rf_fused_blocks": native_rf_fused_blocks,
            "native_rf_fused_intervals": native_rf_fused_intervals,
            "native_full_plan_enabled": native_full_plan_executed,
            "native_full_plan_model": (
                (
                    "multi-pool-" + native_full_plan_model
                    if native_full_plan_source_model
                    else "multi-pool-no-inflow"
                )
                if native_full_plan_executed and native_full_plan_multi_pool
                else (
                    native_full_plan_model
                    if native_full_plan_executed and native_full_plan_source_model
                    else "no-inflow" if native_full_plan_executed else None
                )
            ),
            "native_full_plan_blocks": native_full_plan_blocks,
            "native_full_plan_intervals": native_full_plan_intervals,
            "native_spin_major_enabled": native_spin_major_executed,
            "native_spin_major_fallback_reason": native_spin_major_fallback_reason,
            "native_spin_major_chunks": native_spin_major_chunks,
            "native_spin_major_intervals": native_spin_major_intervals,
            "native_spin_major_threads": native_spin_major_threads,
            "native_spin_major_tile_spins": native_spin_major_tile_spins,
            "native_spin_major_table_s": native_spin_major_table_s,
            "native_spin_major_kernel_s": native_spin_major_kernel_s,
            **rf_propagator_metadata,
            **rf_propagator_stats,
            "native_rf_fused_spin_limit": native_rf_fused_spin_limit,
            "native_rf_fused_interval_limit": native_rf_fused_interval_limit,
            "native_rf_fused_cache_limit_bytes": (native_rf_fused_cache_limit_bytes),
            "native_rf_fused_cache_bytes": (
                native_rf_fused_plan_cache.current_bytes
                if native_rf_fused_plan_cache is not None
                else 0
            ),
            "native_spatial_rf_table_limit_bytes": (
                native_spatial_rf_table_limit_bytes
            ),
            "native_spatial_rf_max_groups_per_block": int(native_spatial_rf_max_groups),
            "native_spatial_rf_cache_limit_bytes": spatial_rf_cache_limit_bytes,
            "native_spatial_rf_cache_bytes": (
                spatial_rf_cache.current_bytes if spatial_rf_cache is not None else 0
            ),
            "native_spatial_rf_cache_hits": spatial_rf_cache_hits,
            "native_spatial_rf_cache_misses": spatial_rf_cache_misses,
            "native_rf_fused_numerical_contract": (
                "float64-close" if native_rf_fused_enabled else None
            ),
            "native_parallel_threshold": native_parallel_threshold,
            "native_block_interval_limit": native_block_interval_limit,
            "native_block_table_limit_bytes": native_block_table_limit_bytes,
            "native_full_plan_table_limit_bytes": (native_full_plan_table_limit_bytes),
            "native_parallel_memory_limited": (
                sequence_kernel == "native_parallel"
                and native_longitudinal_block is not None
                and not native_block_enabled
            ),
            "pyruvate_inflow": phantom.pyruvate_inflow is not None,
            "spatial_inflow_arrival_delays": spatial_inflow_active,
            "inflow_max_step_s": (
                None
                if phantom.pyruvate_inflow is None
                else phantom.pyruvate_inflow.max_step_s
            ),
            "dynamic_b0": phantom.dynamic_b0 is not None,
            "conversion_start_s": phantom.conversion_start_s,
            "kinetics_time_offset_s": phantom.kinetics_time_offset_s,
            "sequence_conversion_start_s": conversion_start_s,
            "kinetic_preroll_start_s": preroll_start_s,
            "ideal_spoiling_applied": bool(
                spoiler_mode == "ideal" and compiled.transverse_crush_times_s.size
            ),
            "ideal_spoiler_end_times_s": (
                compiled.transverse_crush_times_s.tolist()
                if spoiler_mode == "ideal"
                else []
            ),
            "declared_ideal_spoiler_end_times_s": (
                compiled.transverse_crush_times_s.tolist()
            ),
            "pyruvate_inflow_curve": (
                None
                if phantom.pyruvate_inflow is None
                else phantom.pyruvate_inflow.rate_curve_s_inv.to_dict()
            ),
            "pyruvate_inflow_polarization_curve": (
                None
                if phantom.pyruvate_inflow is None
                or phantom.pyruvate_inflow.polarization_curve is None
                else phantom.pyruvate_inflow.polarization_curve.to_dict()
            ),
            "equilibrium_polarization": phantom.equilibrium_polarization,
            "dynamic_b0_curve": (
                None
                if phantom.dynamic_b0 is None
                else phantom.dynamic_b0.offset_curve_hz.to_dict()
            ),
        },
        pool_names=tuple(pool.name for pool in phantom.pools),
        species_signal=result_species_signal,
        final_pool_magnetization=final_pool,
        checkpoint_pool_magnetization=checkpoint_pool,
        sequence_waveforms=sequence_waveforms,
        physical_field_maps=physical_field_maps,
    )
    if _block_result_mode == "template":
        return _DynamicBlockResult(
            active_indices=active,
            parent_species_signal=species_signal,
            final_pool_active=state,
            checkpoint_pool_active=(
                checkpoint_states if checkpoint_states.size else None
            ),
            runtime_stats=block_runtime_stats,
            template_result=result,
        )
    return result


def simulate_dynamic_sequence(
    program,
    phantom: DynamicSpectralPhantom,
    *,
    checkpoints_s=(),
    field_strength_t=None,
    nucleus=None,
    sequence_reference_ppm=None,
    progress_callback=None,
    preview_callback=None,
    cancel_callback=None,
    status_callback=None,
    simulation_timestep_s=1e-6,
    signal_weighting="voxel",
    sequence_kernel="optimized",
    use_parallel=True,
    num_threads=1,
    memory_budget_bytes=None,
    spin_sampling=None,
    spoiler_mode="ideal",
    checkpoint_dtype=None,
    parent_voxel_chunk_size=None,
    **_ignored,
):
    """Run a dynamic sequence, streaming large objects by parent voxel.

    Small objects retain the established single-block path. Large objects and
    explicitly requested chunks use one shared execution plan, keep all
    subspins of a parent voxel together, and reduce parent-voxel signals with a
    fixed streaming binary tree.
    """
    from .sequence import SequenceCompiler
    from .sequence.spin_sampling import coerce_spin_sampling

    spoiler_mode = str(spoiler_mode).strip().lower()
    if spoiler_mode not in {"ideal", "gradient"}:
        raise ValueError("spoiler_mode must be 'ideal' or 'gradient'")
    sampling = coerce_spin_sampling(
        spin_sampling if spoiler_mode == "gradient" else None
    )
    sampling.validate_phantom_dimensions(phantom.ndim)
    active = np.flatnonzero(phantom.mask.ravel())
    if active.size == 0:
        raise ValueError(
            "dynamic phantom has neither initial magnetization nor inflow support"
        )
    if parent_voxel_chunk_size is not None:
        if (
            isinstance(parent_voxel_chunk_size, bool)
            or int(parent_voxel_chunk_size) != parent_voxel_chunk_size
            or int(parent_voxel_chunk_size) < 1
        ):
            raise ValueError("parent_voxel_chunk_size must be a positive integer")
        parent_voxel_chunk_size = int(parent_voxel_chunk_size)

    compile_started = perf_counter()
    compiled = SequenceCompiler().compile(
        program,
        checkpoints_s=checkpoints_s,
        extra_boundaries_s=phantom.dynamic_breakpoints_s(program.duration_s),
        simulation_timestep_s=simulation_timestep_s,
        status_callback=status_callback,
    )
    sequence_compile_s = perf_counter() - compile_started
    plan_started = perf_counter()
    execution_plan = build_dynamic_execution_plan(compiled, phantom)
    execution_plan_build_s = perf_counter() - plan_started

    real_dtype = np.dtype(np.float64)
    complex_dtype = np.dtype(np.complex128)
    effective_checkpoint_dtype = np.dtype(
        real_dtype if checkpoint_dtype is None else checkpoint_dtype
    )
    if effective_checkpoint_dtype not in {
        np.dtype(np.float16),
        np.dtype(np.float32),
        np.dtype(np.float64),
    }:
        raise ValueError("checkpoint_dtype must be float16, float32, or float64")

    parent_count = int(active.size)
    spins_per_parent = sampling.spins_per_voxel
    total_spins = parent_count * spins_per_parent
    rx_maps = getattr(phantom, "rx_sensitivity_maps", None)
    n_rx_coils = 1 if rx_maps is None else int(np.asarray(rx_maps).shape[0])
    checkpoint_count = int(compiled.checkpoint_times_s.size)
    adc_count = int(compiled.adc_times_s.size)
    tx_map = getattr(phantom, "tx_sensitivity_map", None)
    if tx_map is None:
        full_tx_sensitivity = np.ones(parent_count, dtype=np.complex128)
    else:
        tx_map = np.asarray(tx_map, dtype=np.complex128)
        if tx_map.shape != phantom.shape or not np.all(np.isfinite(tx_map)):
            raise ValueError(
                "dynamic phantom Tx sensitivity must be finite and match its shape"
            )
        full_tx_sensitivity = tx_map.ravel()[active]
    global_tx_classification, global_constant_tx = _classify_tx_sensitivity(
        full_tx_sensitivity
    )
    # Every block sees the same plan, phantom, and Tx classification, so a
    # spin-major prediction here holds for all of them. Spin-major blocks keep
    # one ADC row per block instead of one per parent voxel.
    spin_major_planned = False
    if sequence_kernel in {"native_serial", "native_parallel"}:
        spin_major_planned = (
            _spin_major_fallback_reason(
                requested=_spin_major_requested(_ignored.get("_native_spin_major")),
                available=_spin_major_kernel() is not None,
                phantom=phantom,
                real_dtype=real_dtype,
                checkpoint_dtype=effective_checkpoint_dtype,
                spatial_tx=global_tx_classification == "spatial",
                execution_plan=execution_plan,
            )
            is None
            and _spin_major_slice_limits(
                memory_budget_bytes,
                n_rx_coils,
                max(1, int(num_threads)) if use_parallel else 1,
                phantom.n_species,
                bool(
                    phantom.pyruvate_inflow is not None
                    and phantom.pyruvate_inflow.has_spatial_delays
                ),
            )
            is not None
        )
    adc_row_bytes = phantom.n_species * n_rx_coils * adc_count * complex_dtype.itemsize

    # Public outputs remain global. Everything proportional to the expanded
    # subspin count belongs to the current parent-voxel memory block.
    global_output_bytes = (
        phantom.n_species * phantom.nvoxels * 3 * real_dtype.itemsize
        + phantom.nvoxels * 3 * real_dtype.itemsize
        + checkpoint_count
        * (phantom.n_species + 1)
        * phantom.nvoxels
        * 3
        * effective_checkpoint_dtype.itemsize
        + (phantom.n_species + 1) * n_rx_coils * adc_count * complex_dtype.itemsize
    )
    per_spin_work_bytes = (
        phantom.n_species * 3 * real_dtype.itemsize
        + phantom.n_species * complex_dtype.itemsize
        + 3 * np.dtype(np.float64).itemsize
        + 16 * np.dtype(np.float64).itemsize
    )
    if phantom.initial_spin_density_maps is not None:
        per_spin_work_bytes += phantom.n_species * 3 * real_dtype.itemsize
    per_parent_work_bytes = (
        spins_per_parent * per_spin_work_bytes
        + checkpoint_count
        * phantom.n_species
        * spins_per_parent
        * 3
        * effective_checkpoint_dtype.itemsize
        + (0 if spin_major_planned else adc_row_bytes)
    )
    # The reserve covers bounded coefficient/RF caches and allocator overhead.
    reserve_bytes = 64 * 1024**2
    per_block_work_bytes = adc_row_bytes if spin_major_planned else 0
    memory_parent_limit = parent_count
    enforce_memory_plan = bool(
        sampling.enabled or parent_voxel_chunk_size is not None or total_spins > 131072
    )
    if memory_budget_bytes is not None and enforce_memory_plan:
        memory_budget_bytes = int(memory_budget_bytes)
        if memory_budget_bytes <= 0:
            raise MemoryError("Memory limit must leave room for simulation outputs")
        reserve_bytes = min(reserve_bytes, max(1024**2, memory_budget_bytes // 10))
        available_work_bytes = (
            memory_budget_bytes
            - global_output_bytes
            - reserve_bytes
            - per_block_work_bytes
        )
        memory_parent_limit = available_work_bytes // max(1, per_parent_work_bytes)
        if memory_parent_limit < 1:
            required = (
                global_output_bytes
                + reserve_bytes
                + per_block_work_bytes
                + per_parent_work_bytes
            )
            raise MemoryError(
                "Memory limit exceeded before dynamic state allocation: one "
                "parent voxel plus requested outputs needs approximately "
                f"{required / 1024**2:.2f} MiB, above the current budget of "
                f"{memory_budget_bytes / 1024**2:.2f} MiB. Reduce checkpoints, "
                "ADC samples, receive coils, or subvoxel spin counts."
            )

    requested_chunk = (
        parent_count
        if parent_voxel_chunk_size is None
        else min(parent_count, parent_voxel_chunk_size)
    )
    native_expanded_spin_limit = 131072
    native_parent_limit = max(1, native_expanded_spin_limit // max(1, spins_per_parent))
    chunk_parent_count = max(
        1,
        min(
            parent_count,
            requested_chunk,
            int(memory_parent_limit),
            native_parent_limit,
        ),
    )
    use_chunking = bool(
        parent_voxel_chunk_size is not None
        or total_spins > native_expanded_spin_limit
        or chunk_parent_count < parent_count
    )
    # Cached RF propagators use what the planned blocks and spin-major slice
    # tables leave of the budget; they never shrink the blocks.
    spin_major_rf_budget_bytes = _ignored.pop(
        "_spin_major_rf_propagator_budget_bytes", None
    )
    if spin_major_rf_budget_bytes is None and memory_budget_bytes is not None:
        spin_major_rf_budget_bytes = max(
            0,
            int(memory_budget_bytes)
            - global_output_bytes
            - reserve_bytes
            - per_block_work_bytes
            - chunk_parent_count * per_parent_work_bytes
            - int(memory_budget_bytes) // 8,
        )
    common_arguments = dict(
        checkpoints_s=checkpoints_s,
        field_strength_t=field_strength_t,
        nucleus=nucleus,
        sequence_reference_ppm=sequence_reference_ppm,
        cancel_callback=cancel_callback,
        simulation_timestep_s=simulation_timestep_s,
        signal_weighting=signal_weighting,
        sequence_kernel=sequence_kernel,
        use_parallel=use_parallel,
        num_threads=num_threads,
        memory_budget_bytes=memory_budget_bytes,
        spin_sampling=spin_sampling,
        spoiler_mode=spoiler_mode,
        checkpoint_dtype=checkpoint_dtype,
        _compiled=compiled,
        _execution_plan=execution_plan,
        _sequence_compile_s=sequence_compile_s,
        _execution_plan_build_s=execution_plan_build_s,
        _tx_classification=global_tx_classification,
        _constant_tx_sensitivity=global_constant_tx,
        _spin_major_rf_propagator_budget_bytes=spin_major_rf_budget_bytes,
        **_ignored,
    )
    if not use_chunking:
        return _simulate_dynamic_sequence_block(
            program,
            phantom,
            progress_callback=progress_callback,
            preview_callback=preview_callback,
            status_callback=status_callback,
            **common_arguments,
        )

    block_count = (parent_count + chunk_parent_count - 1) // chunk_parent_count
    if status_callback is not None:
        status_callback(
            "Streaming the dynamic object in "
            f"{block_count:,} parent-voxel block(s) of at most "
            f"{chunk_parent_count:,}; preview signals contain only completed "
            "object blocks."
        )

    accumulator = _PairwiseArrayAccumulator()
    final_pool_flat = np.zeros(
        (phantom.n_species, phantom.nvoxels, 3), dtype=real_dtype
    )
    checkpoint_pool_flat = (
        np.zeros(
            (checkpoint_count, phantom.n_species, phantom.nvoxels, 3),
            dtype=effective_checkpoint_dtype,
        )
        if checkpoint_count
        else None
    )
    template_result = None
    aggregate_stats = {
        "native_full_plan_blocks": 0,
        "native_full_plan_intervals": 0,
        "native_spin_major_chunks": 0,
        "native_spin_major_intervals": 0,
        "native_spin_major_parent_blocks": 0,
        "native_rf_fused_blocks": 0,
        "native_rf_fused_intervals": 0,
        "native_spatial_rf_cache_hits": 0,
        "native_spatial_rf_cache_misses": 0,
        "native_spatial_rf_cache_bytes": 0,
        "native_rf_fused_cache_bytes": 0,
        "native_spin_major_rf_propagator_builds": 0,
        "native_spin_major_rf_propagator_applications": 0,
        "native_spin_major_rf_propagator_stepwise_segments": 0,
        "native_spin_major_rf_propagator_rotated_applications": 0,
        "native_spin_major_rf_propagator_stepwise_tiles": 0,
        "native_spin_major_rf_propagator_bytes": 0,
        "native_spin_major_rf_propagator_build_s": 0.0,
    }
    completed_parents = 0
    for block_index, first_parent in enumerate(
        range(0, parent_count, chunk_parent_count)
    ):
        block_active = active[first_parent : first_parent + chunk_parent_count]
        block = _simulate_dynamic_sequence_block(
            program,
            phantom,
            progress_callback=None,
            preview_callback=None,
            status_callback=status_callback if block_index == 0 else None,
            _active_indices=block_active,
            _parent_resolved_signal=True,
            _block_result_mode="template" if block_index == 0 else "minimal",
            **common_arguments,
        )
        if block.template_result is not None:
            template_result = block.template_result
        for parent_signal in block.parent_species_signal:
            accumulator.add(parent_signal)
        final_pool_flat[:, block.active_indices] = block.final_pool_active
        if checkpoint_pool_flat is not None:
            checkpoint_pool_flat[:, :, block.active_indices] = (
                block.checkpoint_pool_active
            )
        for key, value in block.runtime_stats.items():
            if key.endswith("_bytes"):
                aggregate_stats[key] = max(aggregate_stats[key], int(value))
            elif key.endswith("_s"):
                aggregate_stats[key] += float(value)
            else:
                aggregate_stats[key] += int(value)
        completed_parents += int(block.active_indices.size)
        if progress_callback is not None:
            progress_callback(completed_parents, parent_count)
        if preview_callback is not None:
            preview_callback(
                completed_parents / parent_count,
                accumulator.total().sum(axis=0),
            )

    if template_result is None:
        raise RuntimeError("dynamic parent-voxel execution produced no result")
    species_signal = accumulator.total()
    final_pool = final_pool_flat.reshape((phantom.n_species,) + phantom.shape + (3,))
    checkpoint_pool = (
        None
        if checkpoint_pool_flat is None
        else checkpoint_pool_flat.reshape(
            (checkpoint_count, phantom.n_species) + phantom.shape + (3,)
        )
    )
    metadata = dict(template_result.metadata)
    metadata.update(aggregate_stats)
    metadata.update(
        {
            "n_simulated_spins": total_spins,
            "sequence_compile_s": sequence_compile_s,
            "execution_plan_build_s": execution_plan_build_s,
            "dynamic_parent_chunking": True,
            "dynamic_parent_chunk_count": block_count,
            "dynamic_parent_chunk_size": chunk_parent_count,
            "dynamic_requested_parent_chunk_size": parent_voxel_chunk_size,
            "dynamic_reduction_group_parent_voxels": (
                chunk_parent_count
                if aggregate_stats["native_spin_major_parent_blocks"] == block_count
                else 1
            ),
            "dynamic_signal_reduction": (
                "fixed-block-streaming-pairwise"
                if aggregate_stats["native_spin_major_parent_blocks"]
                else "fixed-parent-streaming-pairwise"
            ),
            "dynamic_global_output_bytes_estimate": int(global_output_bytes),
            "dynamic_parent_work_bytes_estimate": int(per_parent_work_bytes),
            "dynamic_block_work_bytes_estimate": int(per_block_work_bytes),
            "dynamic_memory_reserve_bytes": int(reserve_bytes),
            "numerical_contract": "float64-close",
        }
    )
    return replace(
        template_result,
        signal=species_signal.sum(axis=0),
        final_magnetization=final_pool.sum(axis=0),
        checkpoint_magnetization=(
            None if checkpoint_pool is None else checkpoint_pool.sum(axis=1)
        ),
        metadata=metadata,
        species_signal=species_signal,
        final_pool_magnetization=final_pool,
        checkpoint_pool_magnetization=checkpoint_pool,
    )
