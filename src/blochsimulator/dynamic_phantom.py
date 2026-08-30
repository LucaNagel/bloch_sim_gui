"""Dynamic two-pool hyperpolarized pyruvate/lactate phantoms."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .phantom import Phantom
from .spectral_phantom import ChemicalSpecies
from .units import NUCLEUS_GAMMA_HZ_PER_T, hz_to_ppm, ppm_to_hz


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


@dataclass
class PyruvateInflow:
    """Voxelwise pyruvate delivery driven by a scalar concentration-rate curve.

    New phantoms provide ``polarization_curve`` so concentration influx and its
    polarization remain separate.  A missing polarization curve denotes the
    legacy direct-Mz source for backwards compatibility.
    """

    rate_curve_s_inv: TimeCurve
    delivery_map: np.ndarray
    polarization_curve: Optional[TimeCurve] = None

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

    @property
    def support_mask(self) -> np.ndarray:
        return np.asarray(self.delivery_map) > 0


@dataclass
class DynamicB0:
    """Time-dependent B0 offset in Hz with a voxelwise scale map."""

    offset_curve_hz: TimeCurve
    spatial_scale_map: np.ndarray
    pool_scale: Tuple[float, float] = (1.0, 1.0)

    def validate(self, shape: Tuple[int, int, int]) -> None:
        values = np.asarray(self.spatial_scale_map, dtype=np.float64)
        if values.shape != shape or not np.all(np.isfinite(values)):
            raise ValueError(
                "dynamic B0 scale map must be finite and match phantom shape"
            )
        scales = tuple(float(value) for value in self.pool_scale)
        if len(scales) != 2 or not np.all(np.isfinite(scales)):
            raise ValueError("dynamic B0 pool scale requires two finite values")
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
    """Two-pool hyperpolarized phantom with a voxelwise irreversible kPL map."""

    shape: Tuple[int, int, int]
    fov: Tuple[float, float, float]
    pools: Tuple[ChemicalSpecies, ChemicalSpecies]
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
        if len(self.pools) != 2:
            raise ValueError("the initial dynamic model requires exactly two pools")
        if len({pool.name for pool in self.pools}) != 2:
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
        return 2

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
            return np.zeros((2,) + self.shape, dtype=np.float64)
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
                self.inflow_curve_on_sequence_timeline.breakpoints_s(duration_s)
            )
            if self.inflow_polarization_curve_on_sequence_timeline is not None:
                values.extend(
                    self.inflow_polarization_curve_on_sequence_timeline.breakpoints_s(
                        duration_s
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
        """Return the initial two-pool Lorentzian spectrum at one voxel."""
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
        result = np.zeros((2,) + self.shape + (3,), dtype=np.float64)
        for index, pool in enumerate(self.pools):
            result[index, ..., 2] = self.initial_concentration_maps[pool.name]
        return result

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
                "version": 3,
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
        return cls(
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

    def save(self, filename) -> Path:
        path = Path(filename)
        header = {
            "format": "blochsimulator-dynamic-spectral-phantom",
            "version": 3,
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
        arrays = {
            "kpl_map_s_inv": self.kpl_map_s_inv,
            "initial_0": self.initial_concentration_maps[self.pools[0].name],
            "initial_1": self.initial_concentration_maps[self.pools[1].name],
        }
        if self.initial_spin_density_maps is not None:
            arrays["spin_density_0"] = self.initial_spin_density_maps[
                self.pools[0].name
            ]
            arrays["spin_density_1"] = self.initial_spin_density_maps[
                self.pools[1].name
            ]
        if self.b0_map is not None:
            arrays["b0_map"] = self.b0_map
        if self.b0_map_ppm is not None:
            arrays["b0_map_ppm"] = self.b0_map_ppm
        if self.pyruvate_inflow is not None:
            arrays["pyruvate_delivery_map"] = self.pyruvate_inflow.delivery_map
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
        return cls(
            shape=tuple(header["shape"]),
            fov=tuple(header["fov"]),
            pools=pools,
            initial_concentration_maps={
                pools[0].name: arrays["initial_0"],
                pools[1].name: arrays["initial_1"],
            },
            initial_spin_density_maps=(
                None
                if "spin_density_0" not in arrays
                else {
                    pools[0].name: arrays["spin_density_0"],
                    pools[1].name: arrays["spin_density_1"],
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
    with_source = source_start is not None or source_end is not None
    if prepared is None:
        prepared = _prepare_longitudinal_step(
            kpl,
            r1_p,
            r1_l,
            duration,
            with_source=with_source,
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
):
    """Advance total Mz, optionally relaxing polarization toward equilibrium.

    With a concentration state, total magnetization is represented as
    ``Mz = equilibrium_polarization * C + excess``.  The existing exact
    zero-target solver advances the excess, while concentration follows the
    same irreversible P→L exchange without T1 decay.
    """
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
        )
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
    )
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
):
    """Precompute state-independent coefficients for one free half-step."""
    a = r1_p + kpl
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


def _prepare_longitudinal_step_for_dtype(
    kpl,
    r1_p,
    r1_l,
    duration,
    *,
    with_source,
    dtype,
):
    """Prepare in float64, then round once to the requested state dtype."""
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.float64):
        return _prepare_longitudinal_step(
            kpl,
            r1_p,
            r1_l,
            duration,
            with_source=with_source,
        )
    prepared = _prepare_longitudinal_step(
        np.asarray(kpl, dtype=np.float64),
        float(r1_p),
        float(r1_l),
        float(duration),
        with_source=with_source,
    )
    exp_a, exp_b, difference, regular, source_coefficients = prepared
    if source_coefficients is not None:
        source_coefficients = tuple(
            np.asarray(value, dtype=dtype) for value in source_coefficients
        )
    return (
        np.asarray(exp_a, dtype=dtype),
        dtype.type(exp_b),
        np.asarray(difference, dtype=dtype),
        regular,
        source_coefficients,
    )


def kinetic_preroll_start_s(
    inflow_curve: Optional[TimeCurve],
    conversion_start_s: float,
    kinetics_time_offset_s: float = 0.0,
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
        candidates.append(float(inflow_curve.times_s[0]) - kinetics_time_offset_s)
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
    boundaries = (float(start_s), *sorted(set(internal_knots)), float(end_s))
    zero_kpl = np.zeros_like(kpl)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end == start:
            continue
        interval_kpl = kpl if (start + end) / 2.0 >= conversion_start_s else zero_kpl
        if inflow_curve is None:
            source_start = source_end = None
            concentration_source_start = concentration_source_end = None
        else:
            start_value, end_value = inflow_curve.interval_values(start, end)
            if inflow_polarization_curve is None:
                source_start = inflow_delivery * start_value
                source_end = inflow_delivery * end_value
                concentration_source_start = concentration_source_end = None
            else:
                polarization_start, polarization_end = (
                    inflow_polarization_curve.interval_values(start, end)
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
):
    if duration == 0:
        return
    for pool in range(2):
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
    )


def _prepare_transverse_factors(phase_cycles, t2, duration):
    """Evaluate the two state-independent transverse factors once."""
    return tuple(
        np.exp(-duration / t2[pool] - 2j * np.pi * phase_cycles[pool])
        for pool in range(2)
    )


def _prepare_transverse_factors_for_dtype(phase_cycles, t2, duration, dtype):
    """Prepare complex factors in float64 and round once for float32 runs."""
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.float64):
        return _prepare_transverse_factors(phase_cycles, t2, duration)
    prepared = _prepare_transverse_factors(
        np.asarray(phase_cycles, dtype=np.float64),
        np.asarray(t2, dtype=np.float64),
        float(duration),
    )
    return tuple(np.asarray(value, dtype=np.complex64) for value in prepared)


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


def _prepare_rf_rotation_for_dtype(rf_hz, duration, dtype):
    """Prepare RF scalars in float64 and round once for float32 execution."""
    prepared = _prepare_rf_rotation(complex(rf_hz), float(duration))
    dtype = np.dtype(dtype)
    if prepared is None or dtype == np.dtype(np.float64):
        return prepared
    return tuple(dtype.type(value) for value in prepared)


def _rf_rotate_float32(state, prepared):
    """Apply one xy-axis RF rotation in a stable float32 plane basis."""
    if prepared is None:
        return
    axis_x, axis_y, cosine, sine, one_minus_cosine = prepared
    for pool in range(2):
        vectors = state[pool]
        value_x = vectors[:, 0]
        value_y = vectors[:, 1]
        value_z = vectors[:, 2]
        parallel = value_x * axis_x + value_y * axis_y
        perpendicular = -value_x * axis_y + value_y * axis_x
        rotated_perpendicular = perpendicular * cosine - value_z * sine
        rotated_z = perpendicular * sine + value_z * cosine
        vectors[:, 0] = parallel * axis_x - rotated_perpendicular * axis_y
        vectors[:, 1] = parallel * axis_y + rotated_perpendicular * axis_x
        vectors[:, 2] = rotated_z


def _rf_rotate(state, rf_hz, duration):
    prepared = _prepare_rf_rotation(rf_hz, duration)
    if prepared is None:
        return
    axis_x, axis_y, cosine, sine, one_minus_cosine = prepared
    axis = np.asarray([axis_x, axis_y, 0.0])
    for pool in range(2):
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
    effective_rf = complex(rf_hz) * np.asarray(tx_sensitivity)
    nx = -2.0 * np.pi * effective_rf.real * float(duration)
    ny = 2.0 * np.pi * effective_rf.imag * float(duration)
    angle = np.hypot(nx, ny)
    nonzero = angle > 0.0
    safe_angle = np.where(nonzero, angle, 1.0)
    axis_x = np.where(nonzero, nx / safe_angle, 1.0).astype(state.dtype, copy=False)
    axis_y = np.where(nonzero, ny / safe_angle, 0.0).astype(state.dtype, copy=False)
    cosine = np.cos(angle).astype(state.dtype, copy=False)
    sine = np.sin(angle).astype(state.dtype, copy=False)
    for pool in range(2):
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
    simulation_precision="float64",
    use_parallel=True,
    num_threads=1,
    memory_budget_bytes=None,
    spin_sampling=None,
    spoiler_mode="ideal",
    checkpoint_dtype=None,
    **_ignored,
):
    """Run the complete sequence on a regional two-pool dynamic phantom.

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
    conversion_start_s = phantom.conversion_start_on_sequence_timeline_s
    if sequence_kernel is None:
        sequence_kernel = "optimized"
    if sequence_kernel not in {
        "optimized",
        "reference",
        "native_serial",
        "native_parallel",
    }:
        raise ValueError(
            "sequence_kernel must be 'optimized', 'reference', 'native_serial', "
            "or 'native_parallel'"
        )
    if simulation_precision not in {"float64", "float32"}:
        raise ValueError("simulation_precision must be 'float64' or 'float32'")
    if simulation_precision == "float32" and sequence_kernel != "optimized":
        raise ValueError(
            "float32 precision currently requires sequence_kernel='optimized'"
        )
    real_dtype = np.dtype(simulation_precision)
    checkpoint_dtype = np.dtype(
        real_dtype if checkpoint_dtype is None else checkpoint_dtype
    )
    if checkpoint_dtype not in {
        np.dtype(np.float16),
        np.dtype(np.float32),
        np.dtype(np.float64),
    }:
        raise ValueError("checkpoint_dtype must be float16, float32, or float64")
    complex_dtype = np.dtype(
        np.complex64 if real_dtype == np.dtype(np.float32) else np.complex128
    )
    real_type = real_dtype.type
    requested_sequence_kernel = sequence_kernel
    native_fallback_reason = None
    native_longitudinal_fallback_reason = None
    native_longitudinal_step = None
    native_longitudinal_block = None
    native_concentration_inflow_step = None
    native_rf_concentration_block = None
    native_rf_rotation_block = None
    if sequence_kernel in {"native_serial", "native_parallel"}:
        try:
            from .dynamic_bloch_cy import (
                apply_longitudinal_block_no_inflow,
                apply_longitudinal_step_no_inflow,
                apply_longitudinal_step_with_concentration_inflow,
                apply_dynamic_rf_block_with_concentration_inflow,
                apply_rf_rotation_transverse_block,
            )

            # RF rotation is independent of inflow, concentration tracking,
            # conversion timing, and dynamic B0. Keep that native fast path
            # available even when longitudinal evolution must stay in NumPy.
            native_rf_rotation_block = apply_rf_rotation_transverse_block
            coupled_concentration_inflow = bool(
                phantom.initial_spin_density_maps is not None
                and inflow_curve is not None
                and inflow_polarization_curve is not None
            )
            if coupled_concentration_inflow:
                native_concentration_inflow_step = (
                    apply_longitudinal_step_with_concentration_inflow
                )
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
    compiled = SequenceCompiler().compile(
        program,
        checkpoints_s=checkpoints_s,
        extra_boundaries_s=phantom.dynamic_breakpoints_s(program.duration_s),
        simulation_timestep_s=simulation_timestep_s,
        status_callback=status_callback,
    )
    active = np.flatnonzero(phantom.mask.ravel())
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
        # streamed in voxel chunks. Include both pools, sparse checkpoints and
        # the principal coefficient/position arrays in a conservative estimate.
        checkpoint_count = int(compiled.checkpoint_times_s.size)
        estimated_bytes = n_simulated_spins * (
            160 + checkpoint_count * 2 * 3 * checkpoint_dtype.itemsize
        )
        if phantom.initial_spin_density_maps is not None:
            estimated_bytes += n_simulated_spins * 2 * 3 * real_dtype.itemsize
        if estimated_bytes > int(memory_budget_bytes):
            raise MemoryError(
                "Memory limit exceeded: dynamic subvoxel simulation needs "
                f"approximately {estimated_bytes / 1024**2:.2f} MiB for "
                f"{n_simulated_spins:,} spins, above the current safe budget. "
                "Reduce X/Y/Z subvoxel spin counts or phantom resolution."
            )
    state = (
        np.asarray(phantom.initial_magnetization, dtype=real_dtype)
        .reshape(2, phantom.nvoxels, 3)[:, active]
        .copy()
    )
    if sampling.enabled:
        state = np.repeat(state, spins_per_voxel, axis=1)
    concentration_state = None
    if phantom.initial_spin_density_maps is not None:
        initial_spin_density = np.asarray(
            phantom.initial_spin_density, dtype=real_dtype
        ).reshape(2, phantom.nvoxels)[:, active]
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
    spatial_tx_active = not np.all(tx_sensitivity == (1.0 + 0.0j))
    if (
        sequence_kernel in {"native_serial", "native_parallel"}
        and native_longitudinal_step is None
        and native_concentration_inflow_step is None
        and spatial_tx_active
    ):
        # Spatial Tx currently needs the NumPy RF implementation. If dynamic
        # longitudinal drivers also rule out the native Mz primitives, no
        # native work remains and the actual kernel is the optimized fallback.
        native_fallback_reason = (
            "spatial transmit sensitivity and dynamic longitudinal drivers "
            "require the optimized kernel"
        )
        sequence_kernel = "optimized"
        native_rf_rotation_block = None
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
    coefficient_r1 = np.asarray(
        [1.0 / pool.t1 for pool in phantom.pools], dtype=np.float64
    )
    r1 = np.asarray(coefficient_r1, dtype=real_dtype)
    inflow_delivery = None
    if phantom.pyruvate_inflow is not None:
        inflow_delivery = np.asarray(
            phantom.pyruvate_inflow.delivery_map.ravel()[active], dtype=real_dtype
        )
        if sampling.enabled:
            inflow_delivery = np.repeat(inflow_delivery, spins_per_voxel)
    preroll_start_s = kinetic_preroll_start_s(
        (
            None
            if phantom.pyruvate_inflow is None
            else phantom.pyruvate_inflow.rate_curve_s_inv
        ),
        phantom.conversion_start_s,
        phantom.kinetics_time_offset_s,
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
    species_signal_shape = (
        (2, compiled.adc_times_s.size)
        if n_rx_coils == 1
        else (2, n_rx_coils, compiled.adc_times_s.size)
    )
    species_signal = np.zeros(species_signal_shape, dtype=complex_dtype)
    if signal_weighting not in {"voxel", "voxel_volume"}:
        raise ValueError("signal_weighting must be 'voxel' or 'voxel_volume'")
    signal_scale = real_type(
        phantom.voxel_volume_m3 if signal_weighting == "voxel_volume" else 1.0
    )
    checkpoint_states = np.zeros(
        (compiled.checkpoint_times_s.size, 2, n_simulated_spins, 3),
        dtype=checkpoint_dtype,
    )
    gradient_hz_per_m = compiled.gradient_hz_per_m
    rf_hz = compiled.rf_hz
    adc_demodulation = (
        compiled.adc_demodulation
        if complex_dtype == np.dtype(np.complex128)
        else np.asarray(compiled.adc_demodulation, dtype=complex_dtype)
    )
    adc_cursor = 0
    checkpoint_cursor = 0
    transverse_crush_state_set = set(
        int(value)
        for value in (
            compiled.transverse_crush_state_indices if spoiler_mode == "ideal" else ()
        )
    )

    def crush_transverse_if_requested(state_index):
        if state_index not in transverse_crush_state_set:
            return
        state[:, :, 0:2] = 0.0
        if transverse_state is not None:
            transverse_state[:] = 0.0

    def sync_transverse_to_state():
        if transverse_state is not None:
            state[:, :, 0] = transverse_state.real
            state[:, :, 1] = transverse_state.imag

    def observe(state_index):
        nonlocal adc_cursor, checkpoint_cursor
        while (
            adc_cursor < compiled.adc_state_indices.size
            and compiled.adc_state_indices[adc_cursor] == state_index
        ):
            demodulation = adc_demodulation[adc_cursor]
            for pool in range(2):
                transverse = (
                    state[pool, :, 0] + 1j * state[pool, :, 1]
                    if transverse_state is None
                    else transverse_state[pool]
                )
                if n_rx_coils == 1:
                    received = (
                        np.sum(transverse)
                        if unity_single_rx and not sampling.enabled
                        else (
                            np.sum(transverse * spin_signal_weights)
                            if unity_single_rx
                            else np.sum(
                                transverse * rx_sensitivities[0] * spin_signal_weights
                            )
                        )
                    )
                    species_signal[pool, adc_cursor] = (
                        received * signal_scale * demodulation
                    )
                else:
                    species_signal[pool, :, adc_cursor] = (
                        np.sum(
                            rx_sensitivities
                            * transverse[None, :]
                            * spin_signal_weights[None, :],
                            axis=1,
                        )
                        * signal_scale
                        * demodulation
                    )
            adc_cursor += 1
        while (
            checkpoint_cursor < compiled.checkpoint_state_indices.size
            and compiled.checkpoint_state_indices[checkpoint_cursor] == state_index
        ):
            sync_transverse_to_state()
            checkpoint_states[checkpoint_cursor] = state
            checkpoint_cursor += 1

    crush_transverse_if_requested(0)
    observe(0)
    interval_count = compiled.n_intervals
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
    if memory_budget_bytes is not None:
        native_block_table_limit_bytes = min(
            native_block_table_limit_bytes,
            max(0, int(memory_budget_bytes) // 16),
        )
    native_block_enabled = native_block_table_limit_bytes >= n_simulated_spins * 8
    if not native_block_enabled:
        native_threads = 1
    if (
        status_callback is not None
        and native_rf_rotation_block is not None
        and not spatial_tx_active
    ):
        if native_longitudinal_fallback_reason is not None:
            status_callback(
                "Using the hybrid dynamic kernel: strict native RF rotation "
                "with optimized longitudinal kinetics ("
                f"{native_longitudinal_fallback_reason})."
            )
        status_callback(
            f"Using the strict native RF voxel-block kernel with "
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
    native_rf_fused_fallback_reason = None
    if native_rf_concentration_block is None:
        native_rf_fused_fallback_reason = (
            "persistent RF blocks require coupled concentration/polarized inflow"
        )
    elif phantom.dynamic_b0 is not None:
        native_rf_fused_fallback_reason = (
            "persistent RF blocks do not yet support dynamic B0"
        )
    elif spatial_tx_active:
        native_rf_fused_fallback_reason = (
            "persistent RF blocks require uniform transmit sensitivity"
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
    native_rf_fused_enabled = native_rf_fused_fallback_reason is None
    native_rf_fused_plan_cache = (
        _BoundedArrayCache(native_rf_fused_cache_limit_bytes)
        if native_rf_fused_enabled
        else None
    )
    native_rf_fused_preapplied_end = 0
    native_rf_fused_blocks = 0
    native_rf_fused_intervals = 0
    protected_state_set = set(checkpoint_state_set)
    protected_state_set.update(int(value) for value in compiled.adc_state_indices)
    protected_state_set.update(transverse_crush_state_set)

    def _native_rf_fused_block_end(first_interval):
        if not native_rf_fused_enabled or rf_hz[first_interval] == 0.0:
            return first_interval
        end = first_interval
        hard_end = min(interval_count, first_interval + native_rf_fused_interval_limit)
        while end < hard_end and rf_hz[end] != 0.0:
            if end > first_interval and end in protected_state_set:
                break
            interval_left = 0.0 if end == 0 else float(compiled.interval_end_s[end - 1])
            interval_right = float(compiled.interval_end_s[end])
            if (interval_left + interval_right) / 2.0 < conversion_start_s:
                break
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
        interval_left = (
            0.0
            if first_interval == 0
            else float(compiled.interval_end_s[first_interval - 1])
        )
        for block_interval in range(first_interval, end_interval):
            dt_value = float(compiled.dt_s[block_interval])
            interval_right = float(compiled.interval_end_s[block_interval])
            interval_middle = interval_left + dt_value / 2.0
            half_duration = dt_value / 2.0
            gradient_key = tuple(
                float(value) for value in gradient_hz_per_m[block_interval]
            )
            first_key = (
                half_duration,
                float(interval_middle - interval_left),
                gradient_key,
                None,
            )
            second_key = (
                half_duration,
                float(interval_right - interval_middle),
                gradient_key,
                None,
            )
            factor_keys.append((first_key, second_key))
            longitudinal_keys.append((half_duration, True))
            rf_values.append(complex(rf_hz[block_interval]))
            interval_left = interval_right
        plan_key = (
            tuple(factor_keys),
            tuple(longitudinal_keys),
            tuple(rf_values),
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
                    gradient_frequency = positions @ gradient
                    static_frequencies = (
                        b0[None, :]
                        + pool_offsets[:, None]
                        + gradient_frequency[None, :]
                    )
                    static_frequency_by_gradient[gradient_key] = static_frequencies
                phase = static_frequencies * phase_duration
                factors = _prepare_transverse_factors_for_dtype(
                    phase, t2, half_duration, real_dtype
                )
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
                    prepared = _prepare_longitudinal_step_for_dtype(
                        coefficient_kpl,
                        coefficient_r1[0],
                        coefficient_r1[1],
                        half_duration,
                        with_source=True,
                        dtype=real_dtype,
                    )
                    coefficient_cache.put(longitudinal_key, prepared)
                concentration_key = (
                    "concentration",
                    half_duration,
                    True,
                )
                concentration_prepared = coefficient_cache.get(concentration_key)
                if concentration_prepared is None:
                    concentration_prepared = _prepare_longitudinal_step_for_dtype(
                        coefficient_kpl,
                        0.0,
                        0.0,
                        half_duration,
                        with_source=True,
                        dtype=real_dtype,
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

        rf_prepared = [
            _prepare_rf_rotation(value, float(compiled.dt_s[index]))
            for index, value in zip(range(first_interval, end_interval), rf_values)
        ]
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
            np.ascontiguousarray([value[0] for value in rf_prepared], dtype=np.float64),
            np.ascontiguousarray([value[1] for value in rf_prepared], dtype=np.float64),
            np.ascontiguousarray([value[2] for value in rf_prepared], dtype=np.float64),
            np.ascontiguousarray([value[3] for value in rf_prepared], dtype=np.float64),
            np.ascontiguousarray([value[4] for value in rf_prepared], dtype=np.float64),
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
            first_start, first_end = inflow_curve.interval_values(
                interval_left, interval_middle
            )
            second_start, second_end = inflow_curve.interval_values(
                interval_middle, interval_right
            )
            polarization_first_start, polarization_first_end = (
                inflow_polarization_curve.interval_values(
                    interval_left, interval_middle
                )
            )
            polarization_second_start, polarization_second_end = (
                inflow_polarization_curve.interval_values(
                    interval_middle, interval_right
                )
            )
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
            "Using persistent native RF waveform blocks for coupled dynamic "
            "evolution."
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
                    prepared = _prepare_longitudinal_step_for_dtype(
                        kpl,
                        r1[0],
                        r1[1],
                        half_duration,
                        with_source=False,
                        dtype=real_dtype,
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

    interval_start = 0.0
    for interval in range(interval_count):
        if cancel_callback is not None and cancel_callback():
            raise RuntimeError("sequence simulation cancelled")
        if (
            native_rf_fused_enabled
            and interval >= native_rf_fused_preapplied_end
            and rf_hz[interval] != 0.0
        ):
            preapply_native_rf_fused_block(interval)
        if interval < native_rf_fused_preapplied_end:
            interval_end = float(compiled.interval_end_s[interval])
            crush_transverse_if_requested(interval + 1)
            observe(interval + 1)
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
                    species_signal.sum(axis=0),
                )
            continue
        dt = compiled.dt_s[interval]
        interval_end = float(compiled.interval_end_s[interval])
        interval_mid = interval_start + dt / 2.0
        gradient = gradient_hz_per_m[interval]
        conversion_active = interval_mid >= conversion_start_s
        interval_kpl = kpl if conversion_active else zero_kpl
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
            gradient_frequency = positions @ gradient
            static_frequencies = (
                b0[None, :] + pool_offsets[:, None] + gradient_frequency[None, :]
            )

            def phase_cycles(start_s, end_s):
                phase = static_frequencies * (end_s - start_s)
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
                start_value, end_value = inflow_curve.interval_values(start_s, end_s)
                if inflow_polarization_curve is None:
                    return (
                        inflow_delivery * start_value,
                        inflow_delivery * end_value,
                        None,
                        None,
                    )
                polarization_start, polarization_end = (
                    inflow_polarization_curve.interval_values(start_s, end_s)
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
                phase_cycles(interval_start, interval_mid),
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
            )
            if spatial_tx_active and rf_hz[interval] != 0.0:
                _rf_rotate_spatial(state, rf_hz[interval], tx_sensitivity, dt)
            else:
                _rf_rotate(state, rf_hz[interval], dt)
            source_mid, source_end, concentration_mid, concentration_end = (
                source_values(interval_mid, interval_end)
            )
            _free_step(
                state,
                phase_cycles(interval_mid, interval_end),
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
                longitudinal_prepared = _prepare_longitudinal_step_for_dtype(
                    coefficient_interval_kpl,
                    coefficient_r1[0],
                    coefficient_r1[1],
                    coefficient_half_duration,
                    with_source=phantom.pyruvate_inflow is not None,
                    dtype=real_dtype,
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
                    concentration_longitudinal_prepared = (
                        _prepare_longitudinal_step_for_dtype(
                            coefficient_interval_kpl,
                            0.0,
                            0.0,
                            coefficient_half_duration,
                            with_source=True,
                            dtype=real_dtype,
                        )
                    )
                    coefficient_cache.put(
                        concentration_key, concentration_longitudinal_prepared
                    )
                concentration_interval_scratch = (
                    concentration_longitudinal_scratch
                    if conversion_active
                    else inactive_concentration_longitudinal_scratch
                )
            first_phase_duration = interval_mid - interval_start
            second_phase_duration = interval_end - interval_mid
            if phantom.dynamic_b0 is None:
                first_dynamic_integral = second_dynamic_integral = None
            else:
                first_dynamic_integral = real_type(
                    phantom.dynamic_b0.offset_curve_hz.integral(
                        interval_start, interval_mid
                    )
                )
                second_dynamic_integral = real_type(
                    phantom.dynamic_b0.offset_curve_hz.integral(
                        interval_mid, interval_end
                    )
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
                gradient_frequency = positions @ gradient
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
                first_transverse_factors = _prepare_transverse_factors_for_dtype(
                    first_phase, t2, half_duration, real_dtype
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
                second_transverse_factors = _prepare_transverse_factors_for_dtype(
                    second_phase, t2, half_duration, real_dtype
                )
                transverse_cache.put(second_transverse_key, second_transverse_factors)
            first_phase = second_phase = None

            if inflow_curve is None:
                source_start = source_mid = source_end = None
                concentration_start = concentration_mid = concentration_end = None
            else:
                start_value, mid_value = inflow_curve.interval_values(
                    interval_start, interval_mid
                )
                if inflow_polarization_curve is None:
                    source_start = inflow_delivery * real_type(start_value)
                    source_mid = inflow_delivery * real_type(mid_value)
                    concentration_start = concentration_mid = None
                else:
                    polarization_start, polarization_mid = (
                        inflow_polarization_curve.interval_values(
                            interval_start, interval_mid
                        )
                    )
                    concentration_start = inflow_delivery * real_type(start_value)
                    concentration_mid = inflow_delivery * real_type(mid_value)
                    source_start = concentration_start * real_type(polarization_start)
                    source_mid = concentration_mid * real_type(polarization_mid)
            if native_concentration_inflow_step is not None:
                for pool in range(2):
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
                )
            else:
                for pool in range(2):
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
                if spatial_tx_active:
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
                    if real_dtype == np.dtype(np.float32):
                        _rf_rotate_float32(
                            state,
                            _prepare_rf_rotation_for_dtype(
                                rf_hz[interval], dt, real_dtype
                            ),
                        )
                    else:
                        _rf_rotate(state, rf_hz[interval], dt)
                    transverse_state[:] = state[:, :, 0] + 1j * state[:, :, 1]
                else:
                    rf_prepared = _prepare_rf_rotation(rf_hz[interval], dt)
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
                mid_value, end_value = inflow_curve.interval_values(
                    interval_mid, interval_end
                )
                if inflow_polarization_curve is None:
                    source_mid = inflow_delivery * real_type(mid_value)
                    source_end = inflow_delivery * real_type(end_value)
                    concentration_mid = concentration_end = None
                else:
                    polarization_mid, polarization_end = (
                        inflow_polarization_curve.interval_values(
                            interval_mid, interval_end
                        )
                    )
                    concentration_mid = inflow_delivery * real_type(mid_value)
                    concentration_end = inflow_delivery * real_type(end_value)
                    source_mid = concentration_mid * real_type(polarization_mid)
                    source_end = concentration_end * real_type(polarization_end)
            if native_concentration_inflow_step is not None:
                for pool in range(2):
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
                )
            else:
                for pool in range(2):
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
        crush_transverse_if_requested(interval + 1)
        observe(interval + 1)
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
                species_signal.sum(axis=0),
            )

    sync_transverse_to_state()
    if sampling.enabled:
        state = np.einsum(
            "s,uvsd->uvd",
            subvoxel_weights,
            state.reshape(2, parent_active_count, spins_per_voxel, 3),
            optimize=True,
        )
    final_pool = np.zeros((2, phantom.nvoxels, 3), dtype=real_dtype)
    final_pool[:, active] = state
    final_pool = final_pool.reshape((2,) + phantom.shape + (3,))
    checkpoint_pool = None
    if checkpoint_states.size:
        if sampling.enabled:
            checkpoint_states = np.einsum(
                "s,cuvsd->cuvd",
                subvoxel_weights,
                checkpoint_states.reshape(
                    compiled.checkpoint_times_s.size,
                    2,
                    parent_active_count,
                    spins_per_voxel,
                    3,
                ),
                optimize=True,
            )
        checkpoint_pool = np.zeros(
            (compiled.checkpoint_times_s.size, 2, phantom.nvoxels, 3),
            dtype=checkpoint_dtype,
        )
        checkpoint_pool[:, :, active] = checkpoint_states
        checkpoint_pool = checkpoint_pool.reshape(
            (compiled.checkpoint_times_s.size, 2) + phantom.shape + (3,)
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
    return SequenceSimulationResult(
        signal=species_signal.sum(axis=0),
        adc_times_s=compiled.adc_times_s,
        final_magnetization=final_pool.sum(axis=0),
        checkpoint_magnetization=(
            None if checkpoint_pool is None else checkpoint_pool.sum(axis=1)
        ),
        checkpoint_times_s=compiled.checkpoint_times_s,
        adc_gradient_moment_cyc_per_m=compiled.adc_gradient_moment_cyc_per_m,
        metadata={
            "dynamic_phantom": True,
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
            "tx_sensitivity": "spatial" if spatial_tx_active else "uniform",
            "n_rx_coils": n_rx_coils,
            "simulation_precision": simulation_precision,
            "simulation_timestep_s": float(simulation_timestep_s),
            "compiled_interval_count": interval_count,
            "state_dtype": real_dtype.name,
            "signal_dtype": complex_dtype.name,
            "coefficient_precompute_dtype": "float64",
            "sequence_kernel": sequence_kernel,
            "requested_sequence_kernel": requested_sequence_kernel,
            "native_fallback_reason": native_fallback_reason,
            "native_longitudinal_fallback_reason": (
                native_longitudinal_fallback_reason
            ),
            "native_hybrid": bool(
                sequence_kernel in {"native_serial", "native_parallel"}
                and native_rf_rotation_block is not None
                and not spatial_tx_active
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
            "native_rf_threads": (
                native_rf_threads
                if sequence_kernel in {"native_serial", "native_parallel"}
                and not spatial_tx_active
                else 1
            ),
            "native_longitudinal_threads": (
                native_longitudinal_threads
                if sequence_kernel in {"native_serial", "native_parallel"}
                and native_concentration_inflow_step is not None
                else 1
            ),
            "native_rf_block_enabled": (
                native_rf_rotation_block is not None and not spatial_tx_active
            ),
            "native_rf_fused_block_enabled": native_rf_fused_enabled,
            "native_rf_fused_fallback_reason": native_rf_fused_fallback_reason,
            "native_rf_fused_blocks": native_rf_fused_blocks,
            "native_rf_fused_intervals": native_rf_fused_intervals,
            "native_rf_fused_spin_limit": native_rf_fused_spin_limit,
            "native_rf_fused_interval_limit": native_rf_fused_interval_limit,
            "native_rf_fused_cache_limit_bytes": (native_rf_fused_cache_limit_bytes),
            "native_rf_fused_cache_bytes": (
                native_rf_fused_plan_cache.current_bytes
                if native_rf_fused_plan_cache is not None
                else 0
            ),
            "native_rf_fused_numerical_contract": (
                "float64-close" if native_rf_fused_enabled else None
            ),
            "native_parallel_threshold": native_parallel_threshold,
            "native_block_interval_limit": native_block_interval_limit,
            "native_block_table_limit_bytes": native_block_table_limit_bytes,
            "native_parallel_memory_limited": (
                sequence_kernel == "native_parallel"
                and native_longitudinal_block is not None
                and not native_block_enabled
            ),
            "pyruvate_inflow": phantom.pyruvate_inflow is not None,
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
        species_signal=species_signal,
        final_pool_magnetization=final_pool,
        checkpoint_pool_magnetization=checkpoint_pool,
        sequence_waveforms=sequence_waveforms,
        physical_field_maps=physical_field_maps,
    )
