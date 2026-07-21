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
from .units import NUCLEUS_GAMMA_HZ_PER_T, ppm_to_hz


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
        if not np.all(np.isfinite(times)) or np.any(np.asarray(times) < 0):
            raise ValueError("time curve times must be finite and non-negative")
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
    """Voxelwise longitudinal pyruvate source driven by a scalar rate curve.

    The source used by the solver is ``delivery_map * rate_curve_s_inv(t)`` and
    therefore has units of relative hyperpolarized magnetization per second.
    """

    rate_curve_s_inv: TimeCurve
    delivery_map: np.ndarray

    def validate(self, shape: Tuple[int, int, int]) -> None:
        values = np.asarray(self.delivery_map, dtype=np.float64)
        if values.shape != shape or not np.all(np.isfinite(values)):
            raise ValueError(
                "pyruvate delivery map must be finite and match phantom shape"
            )
        if np.any(values < 0) or np.any(np.asarray(self.rate_curve_s_inv.values) < 0):
            raise ValueError("pyruvate inflow maps and rates must be non-negative")
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
    b0_map_ppm: Optional[np.ndarray] = None
    b0_map: Optional[np.ndarray] = None
    field_strength: float = 3.0
    nucleus: str = "C13"
    spectral_reference_ppm: float = 0.0
    spectral_bandwidth_ppm: float = 20.0
    spectral_points: int = 1024
    name: str = "Dynamic pyruvate/lactate phantom"
    kinetic_regions: Tuple[KineticRegionDefinition, ...] = ()
    pyruvate_inflow: Optional[PyruvateInflow] = None
    dynamic_b0: Optional[DynamicB0] = None
    metadata: Dict = field(default_factory=dict)
    coordinate_system: str = "object_xyz"
    affine_ijk_to_xyz_m: Optional[np.ndarray] = None
    positions: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.shape = tuple(int(value) for value in self.shape)
        self.fov = tuple(float(value) for value in self.fov)
        self.pools = tuple(self.pools)
        self.kinetic_regions = tuple(self.kinetic_regions)
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
        return sum(self.initial_concentration_maps[pool.name] for pool in self.pools)

    @property
    def t1_map(self) -> np.ndarray:
        return self._weighted_pool_property("t1")

    @property
    def t2_map(self) -> np.ndarray:
        return self._weighted_pool_property("t2")

    def _weighted_pool_property(self, name: str) -> np.ndarray:
        total = self.pd_map
        result = np.zeros(self.shape, dtype=float)
        for pool in self.pools:
            result += self.initial_concentration_maps[pool.name] * float(
                getattr(pool, name)
            )
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
                self.pyruvate_inflow.rate_curve_s_inv.breakpoints_s(duration_s)
            )
        if self.dynamic_b0 is not None:
            values.extend(self.dynamic_b0.offset_curve_hz.breakpoints_s(duration_s))
        return tuple(sorted(set(values)))

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
                    "rate_curve_s_inv": self.pyruvate_inflow.rate_curve_s_inv.to_dict()
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
                "version": 2,
                "name": self.name,
                "fov_m": np.asarray(self.fov, dtype=np.float64),
                "field_strength": self.field_strength,
                "nucleus": self.nucleus,
                "spectral_reference_ppm": self.spectral_reference_ppm,
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
            spectral_bandwidth_ppm=float(ds.attrs.get("spectral_bandwidth_ppm", 20.0)),
            spectral_points=int(ds.attrs.get("spectral_points", 1024)),
            name=str(ds.attrs.get("name", "Dynamic pyruvate/lactate phantom")),
            kinetic_regions=regions,
            pyruvate_inflow=pyruvate_inflow,
            dynamic_b0=dynamic_b0,
            metadata=dict(header.get("metadata", {})),
            coordinate_system=str(ds.attrs.get("coordinate_system", "object_xyz")),
            affine_ijk_to_xyz_m=affine,
        )

    def save(self, filename) -> Path:
        path = Path(filename)
        header = {
            "format": "blochsimulator-dynamic-spectral-phantom",
            "version": 2,
            "shape": self.shape,
            "fov": self.fov,
            "field_strength": self.field_strength,
            "nucleus": self.nucleus,
            "spectral_reference_ppm": self.spectral_reference_ppm,
            "spectral_bandwidth_ppm": self.spectral_bandwidth_ppm,
            "spectral_points": self.spectral_points,
            "name": self.name,
            "pools": [pool.__dict__ for pool in self.pools],
            "kinetic_regions": [region.__dict__ for region in self.kinetic_regions],
            "pyruvate_inflow": (
                None
                if self.pyruvate_inflow is None
                else {
                    "rate_curve_s_inv": self.pyruvate_inflow.rate_curve_s_inv.to_dict()
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
            kpl_map_s_inv=arrays["kpl_map_s_inv"],
            b0_map=arrays.get("b0_map"),
            b0_map_ppm=arrays.get("b0_map_ppm"),
            field_strength=header["field_strength"],
            nucleus=header["nucleus"],
            spectral_reference_ppm=float(header.get("spectral_reference_ppm", 0.0)),
            spectral_bandwidth_ppm=float(header.get("spectral_bandwidth_ppm", 20.0)),
            spectral_points=int(header.get("spectral_points", 1024)),
            name=header["name"],
            kinetic_regions=regions,
            pyruvate_inflow=pyruvate_inflow,
            dynamic_b0=dynamic_b0,
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


def _longitudinal_step(
    state,
    kpl,
    r1_p,
    r1_l,
    duration,
    source_start=None,
    source_end=None,
    prepared=None,
):
    if duration == 0:
        return
    pyruvate = state[0, :, 2].copy()
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
    transfer = np.empty_like(pyruvate)
    transfer[regular] = (
        kpl[regular]
        * pyruvate[regular]
        * (exp_b - exp_a[regular])
        / difference[regular]
    )
    transfer[~regular] = kpl[~regular] * pyruvate[~regular] * duration * exp_b
    pyruvate_next = pyruvate * exp_a
    lactate_next = lactate * exp_b + transfer

    if source_start is not None or source_end is not None:
        if source_start is None:
            source_start = source_end
        if source_end is None:
            source_end = source_start
        source_start = np.asarray(source_start, dtype=float)
        source_end = np.asarray(source_end, dtype=float)
        slope = (source_end - source_start) / duration
        if source_coefficients is None:
            raise RuntimeError("prepared longitudinal step lacks inflow coefficients")
        f0_a, f1_a, j0, j1 = source_coefficients
        pyruvate_next += source_start * f0_a + slope * f1_a
        lactate_next += kpl * (source_start * j0 + slope * j1)

    state[0, :, 2] = pyruvate_next
    state[1, :, 2] = lactate_next


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


def simulate_two_pool_kinetics(
    times_s,
    initial_mz,
    t1_s,
    kpl_s_inv,
    inflow_curve: Optional[TimeCurve] = None,
):
    """Evaluate free pyruvate/lactate kinetics at requested absolute times.

    This is the longitudinal part of the dynamic sequence solver without RF or
    gradients. Inflow is integrated piecewise-exactly using the same solver as
    :func:`simulate_dynamic_sequence`. The returned array has shape
    ``(2, n_times)`` in pyruvate/lactate order.
    """
    times = np.asarray(times_s, dtype=float)
    initial = np.asarray(initial_mz, dtype=float)
    relaxation = np.asarray(t1_s, dtype=float)
    kpl = float(kpl_s_inv)
    if times.ndim != 1 or not times.size:
        raise ValueError("kinetics preview times must be a non-empty 1D array")
    if not np.all(np.isfinite(times)) or np.any(times < 0):
        raise ValueError("kinetics preview times must be finite and non-negative")
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

    state = np.zeros((2, 1, 3), dtype=float)
    state[:, 0, 2] = initial
    result = np.empty((2, times.size), dtype=float)
    kpl_array = np.asarray([kpl], dtype=float)
    r1 = 1.0 / relaxation
    current_time = 0.0
    for index, target_time in enumerate(times):
        internal_knots = (
            ()
            if inflow_curve is None
            else tuple(
                knot
                for knot in inflow_curve.times_s
                if current_time < knot < target_time
            )
        )
        boundaries = (current_time, *internal_knots, float(target_time))
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if end == start:
                continue
            if inflow_curve is None:
                source_start = source_end = None
            else:
                start_value, end_value = inflow_curve.interval_values(start, end)
                source_start = np.asarray([start_value], dtype=float)
                source_end = np.asarray([end_value], dtype=float)
            _longitudinal_step(
                state,
                kpl_array,
                r1[0],
                r1[1],
                end - start,
                source_start,
                source_end,
            )
        result[:, index] = state[:, 0, 2]
        current_time = float(target_time)
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
):
    if duration == 0:
        return
    for pool in range(2):
        transverse = state[pool, :, 0] + 1j * state[pool, :, 1]
        factor = (
            np.exp(-duration / t2[pool] - 2j * np.pi * phase_cycles[pool])
            if transverse_factors is None
            else transverse_factors[pool]
        )
        transverse *= factor
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
    )


def _prepare_transverse_factors(phase_cycles, t2, duration):
    """Evaluate the two state-independent transverse factors once."""
    return tuple(
        np.exp(-duration / t2[pool] - 2j * np.pi * phase_cycles[pool])
        for pool in range(2)
    )


def _rf_rotate(state, rf_hz, duration):
    nx = -2 * np.pi * rf_hz.real * duration
    ny = 2 * np.pi * rf_hz.imag * duration
    angle = float(np.hypot(nx, ny))
    if angle == 0:
        return
    axis = np.asarray([nx / angle, ny / angle, 0.0])
    cosine = np.cos(angle)
    sine = np.sin(angle)
    for pool in range(2):
        vectors = state[pool]
        cross = np.cross(np.broadcast_to(axis, vectors.shape), vectors)
        projection = vectors @ axis
        state[pool] = (
            vectors * cosine
            + cross * sine
            + projection[:, None] * axis * (1.0 - cosine)
        )


def simulate_dynamic_sequence(
    program,
    phantom: DynamicSpectralPhantom,
    *,
    checkpoints_s=(),
    field_strength_t=None,
    nucleus=None,
    progress_callback=None,
    preview_callback=None,
    cancel_callback=None,
    status_callback=None,
    simulation_timestep_s=1e-6,
    signal_weighting="voxel",
    sequence_kernel="optimized",
    **_ignored,
):
    """Run the complete sequence on a regional two-pool dynamic phantom."""
    from .sequence import (
        AcquisitionDimensions,
        SequenceCompiler,
        SequenceSimulationResult,
    )

    field = (
        phantom.field_strength if field_strength_t is None else float(field_strength_t)
    )
    effective_nucleus = phantom.nucleus if nucleus is None else str(nucleus)
    if sequence_kernel is None:
        sequence_kernel = "optimized"
    if sequence_kernel not in {"optimized", "reference"}:
        raise ValueError("sequence_kernel must be 'optimized' or 'reference'")
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
    state = phantom.initial_magnetization.reshape(2, phantom.nvoxels, 3)[
        :, active
    ].copy()
    positions = phantom.positions[active]
    kpl = phantom.kpl_map_s_inv.ravel()[active]
    b0 = phantom.b0_offset_hz(field, effective_nucleus).ravel()[active]
    pool_offsets = np.asarray(
        [pool.get_frequency_offset(field, effective_nucleus) for pool in phantom.pools]
    )
    t2 = np.asarray([pool.t2 for pool in phantom.pools], dtype=float)[:, None]
    r1 = np.asarray([1.0 / pool.t1 for pool in phantom.pools], dtype=float)
    inflow_delivery = None
    if phantom.pyruvate_inflow is not None:
        inflow_delivery = phantom.pyruvate_inflow.delivery_map.ravel()[active]
    dynamic_b0_scale = None
    dynamic_b0_pool_scale = None
    if phantom.dynamic_b0 is not None:
        dynamic_b0_scale = phantom.dynamic_b0.spatial_scale_map.ravel()[active]
        dynamic_b0_pool_scale = np.asarray(phantom.dynamic_b0.pool_scale, dtype=float)[
            :, None
        ]
    species_signal = np.zeros((2, compiled.adc_times_s.size), dtype=np.complex128)
    if signal_weighting not in {"voxel", "voxel_volume"}:
        raise ValueError("signal_weighting must be 'voxel' or 'voxel_volume'")
    signal_scale = (
        phantom.voxel_volume_m3 if signal_weighting == "voxel_volume" else 1.0
    )
    checkpoint_states = np.zeros(
        (compiled.checkpoint_times_s.size, 2, active.size, 3), dtype=np.float64
    )
    adc_cursor = 0
    checkpoint_cursor = 0

    def observe(state_index):
        nonlocal adc_cursor, checkpoint_cursor
        while (
            adc_cursor < compiled.adc_state_indices.size
            and compiled.adc_state_indices[adc_cursor] == state_index
        ):
            demodulation = compiled.adc_demodulation[adc_cursor]
            for pool in range(2):
                species_signal[pool, adc_cursor] = (
                    np.sum(state[pool, :, 0] + 1j * state[pool, :, 1])
                    * signal_scale
                    * demodulation
                )
            adc_cursor += 1
        while (
            checkpoint_cursor < compiled.checkpoint_state_indices.size
            and compiled.checkpoint_state_indices[checkpoint_cursor] == state_index
        ):
            checkpoint_states[checkpoint_cursor] = state
            checkpoint_cursor += 1

    observe(0)
    interval_count = compiled.n_intervals
    progress_stride = max(1, interval_count // 100)
    coefficient_cache = (
        _BoundedArrayCache(16 * 1024**2) if sequence_kernel == "optimized" else None
    )
    transverse_cache = (
        _BoundedArrayCache(48 * 1024**2)
        if sequence_kernel == "optimized" and phantom.dynamic_b0 is None
        else None
    )
    interval_start = 0.0
    for interval in range(interval_count):
        if cancel_callback is not None and cancel_callback():
            raise RuntimeError("sequence simulation cancelled")
        dt = compiled.dt_s[interval]
        interval_end = float(compiled.interval_end_s[interval])
        interval_mid = interval_start + dt / 2.0
        gradient = compiled.gradient_hz_per_m[interval]
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
                if phantom.pyruvate_inflow is None:
                    return None, None
                start_value, end_value = (
                    phantom.pyruvate_inflow.rate_curve_s_inv.interval_values(
                        start_s, end_s
                    )
                )
                return inflow_delivery * start_value, inflow_delivery * end_value

            source_start, source_mid = source_values(interval_start, interval_mid)
            _free_step(
                state,
                phase_cycles(interval_start, interval_mid),
                t2,
                kpl,
                r1,
                dt / 2.0,
                source_start,
                source_mid,
            )
            _rf_rotate(state, compiled.rf_hz[interval], dt)
            source_mid, source_end = source_values(interval_mid, interval_end)
            _free_step(
                state,
                phase_cycles(interval_mid, interval_end),
                t2,
                kpl,
                r1,
                dt / 2.0,
                source_mid,
                source_end,
            )
        else:
            half_duration = dt / 2.0
            longitudinal_key = float(half_duration)
            longitudinal_prepared = coefficient_cache.get(longitudinal_key)
            if longitudinal_prepared is None:
                longitudinal_prepared = _prepare_longitudinal_step(
                    kpl,
                    r1[0],
                    r1[1],
                    half_duration,
                    with_source=phantom.pyruvate_inflow is not None,
                )
                coefficient_cache.put(longitudinal_key, longitudinal_prepared)
            first_phase_duration = interval_mid - interval_start
            second_phase_duration = interval_end - interval_mid
            if phantom.dynamic_b0 is None:
                gradient_key = tuple(float(value) for value in gradient)
                first_transverse_key = (
                    float(half_duration),
                    float(first_phase_duration),
                    gradient_key,
                )
                second_transverse_key = (
                    float(half_duration),
                    float(second_phase_duration),
                    gradient_key,
                )
                first_transverse_factors = transverse_cache.get(first_transverse_key)
                second_transverse_factors = transverse_cache.get(second_transverse_key)
                static_frequencies = None
                if (
                    first_transverse_factors is None
                    or second_transverse_factors is None
                ):
                    gradient_frequency = positions @ gradient
                    static_frequencies = (
                        b0[None, :]
                        + pool_offsets[:, None]
                        + gradient_frequency[None, :]
                    )
                if first_transverse_factors is None:
                    first_phase = static_frequencies * first_phase_duration
                    first_transverse_factors = _prepare_transverse_factors(
                        first_phase, t2, half_duration
                    )
                    transverse_cache.put(first_transverse_key, first_transverse_factors)
                if first_transverse_key == second_transverse_key:
                    second_transverse_factors = first_transverse_factors
                elif second_transverse_factors is None:
                    second_phase = static_frequencies * second_phase_duration
                    second_transverse_factors = _prepare_transverse_factors(
                        second_phase, t2, half_duration
                    )
                    transverse_cache.put(
                        second_transverse_key, second_transverse_factors
                    )
                first_phase = second_phase = None
            else:
                gradient_frequency = positions @ gradient
                static_frequencies = (
                    b0[None, :] + pool_offsets[:, None] + gradient_frequency[None, :]
                )
                first_phase = static_frequencies * first_phase_duration
                first_dynamic_integral = phantom.dynamic_b0.offset_curve_hz.integral(
                    interval_start, interval_mid
                )
                first_phase = first_phase + (
                    dynamic_b0_pool_scale
                    * dynamic_b0_scale[None, :]
                    * first_dynamic_integral
                )
                second_phase = static_frequencies * second_phase_duration
                second_dynamic_integral = phantom.dynamic_b0.offset_curve_hz.integral(
                    interval_mid, interval_end
                )
                second_phase = second_phase + (
                    dynamic_b0_pool_scale
                    * dynamic_b0_scale[None, :]
                    * second_dynamic_integral
                )
                first_transverse_factors = None
                second_transverse_factors = None

            if phantom.pyruvate_inflow is None:
                source_start = source_mid = source_end = None
            else:
                start_value, mid_value = (
                    phantom.pyruvate_inflow.rate_curve_s_inv.interval_values(
                        interval_start, interval_mid
                    )
                )
                source_start = inflow_delivery * start_value
                source_mid = inflow_delivery * mid_value
            _free_step(
                state,
                first_phase,
                t2,
                kpl,
                r1,
                half_duration,
                source_start,
                source_mid,
                transverse_factors=first_transverse_factors,
                longitudinal_prepared=longitudinal_prepared,
            )
            if compiled.rf_hz[interval] != 0.0:
                _rf_rotate(state, compiled.rf_hz[interval], dt)
            if phantom.pyruvate_inflow is not None:
                mid_value, end_value = (
                    phantom.pyruvate_inflow.rate_curve_s_inv.interval_values(
                        interval_mid, interval_end
                    )
                )
                source_mid = inflow_delivery * mid_value
                source_end = inflow_delivery * end_value
            _free_step(
                state,
                second_phase,
                t2,
                kpl,
                r1,
                half_duration,
                source_mid,
                source_end,
                transverse_factors=second_transverse_factors,
                longitudinal_prepared=longitudinal_prepared,
            )
        observe(interval + 1)
        interval_start = interval_end
        if progress_callback is not None and (
            interval % progress_stride == 0 or interval + 1 == interval_count
        ):
            progress_callback(interval + 1, interval_count)
        if preview_callback is not None and interval + 1 == interval_count:
            preview_callback(1.0, species_signal.sum(axis=0))

    final_pool = np.zeros((2, phantom.nvoxels, 3), dtype=np.float64)
    final_pool[:, active] = state
    final_pool = final_pool.reshape((2,) + phantom.shape + (3,))
    checkpoint_pool = None
    if checkpoint_states.size:
        checkpoint_pool = np.zeros(
            (compiled.checkpoint_times_s.size, 2, phantom.nvoxels, 3),
            dtype=np.float64,
        )
        checkpoint_pool[:, :, active] = checkpoint_states
        checkpoint_pool = checkpoint_pool.reshape(
            (compiled.checkpoint_times_s.size, 2) + phantom.shape + (3,)
        )
    dimensions = AcquisitionDimensions.from_program(program)
    spectroscopic_metadata = program.metadata.get("spectroscopic_acquisition")
    if spectroscopic_metadata is None:
        try:
            from .sequence.acquisition import infer_spectroscopic_acquisition

            spectroscopic_metadata = infer_spectroscopic_acquisition(
                program, compiled=compiled
            ).to_metadata()
        except ValueError:
            spectroscopic_metadata = None
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
            "field_strength_t": field,
            "nucleus": effective_nucleus,
            "spectral_reference_ppm": phantom.spectral_reference_ppm,
            "spectral_bandwidth_ppm": phantom.spectral_bandwidth_ppm,
            "spectral_points": phantom.spectral_points,
            "signal_weighting": signal_weighting,
            "sequence_kernel": sequence_kernel,
            "pyruvate_inflow": phantom.pyruvate_inflow is not None,
            "dynamic_b0": phantom.dynamic_b0 is not None,
            "pyruvate_inflow_curve": (
                None
                if phantom.pyruvate_inflow is None
                else phantom.pyruvate_inflow.rate_curve_s_inv.to_dict()
            ),
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
    )
