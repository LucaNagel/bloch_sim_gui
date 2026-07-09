"""Dynamic two-pool hyperpolarized pyruvate/lactate phantoms."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .phantom import Phantom
from .spectral_phantom import ChemicalSpecies
from .units import NUCLEUS_GAMMA_HZ_PER_T, ppm_to_hz


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
        return self.pd_map > 0

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
        return np.divide(result, total, out=np.zeros_like(result), where=total > 0)

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
            "metadata": self.metadata,
        }
        return xr.Dataset(
            data_vars={
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
            },
            coords=coords,
            attrs={
                "format": "blochsimulator-dynamic-spectral-phantom-xarray",
                "version": 1,
                "name": self.name,
                "fov_m": np.asarray(self.fov, dtype=np.float64),
                "field_strength": self.field_strength,
                "nucleus": self.nucleus,
                "spectral_reference_ppm": self.spectral_reference_ppm,
                "spectral_bandwidth_ppm": self.spectral_bandwidth_ppm,
                "spectral_points": self.spectral_points,
                "has_b0_map": self.b0_map is not None,
                "has_b0_map_ppm": self.b0_map_ppm is not None,
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
            metadata=dict(header.get("metadata", {})),
            coordinate_system=str(ds.attrs.get("coordinate_system", "object_xyz")),
            affine_ijk_to_xyz_m=affine,
        )

    def save(self, filename) -> Path:
        path = Path(filename)
        header = {
            "format": "blochsimulator-dynamic-spectral-phantom",
            "version": 1,
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
            metadata=header.get("metadata", {}),
            coordinate_system=header.get("coordinate_system", "object_xyz"),
            affine_ijk_to_xyz_m=header.get("affine_ijk_to_xyz_m"),
        )


def _longitudinal_step(state, kpl, r1_p, r1_l, duration):
    if duration == 0:
        return
    pyruvate = state[0, :, 2].copy()
    lactate = state[1, :, 2]
    a = r1_p + kpl
    b = r1_l
    exp_a = np.exp(-a * duration)
    exp_b = np.exp(-b * duration)
    difference = a - b
    transfer = np.empty_like(pyruvate)
    regular = np.abs(difference) > 1e-12
    transfer[regular] = (
        kpl[regular]
        * pyruvate[regular]
        * (exp_b - exp_a[regular])
        / difference[regular]
    )
    transfer[~regular] = kpl[~regular] * pyruvate[~regular] * duration * exp_b
    state[0, :, 2] = pyruvate * exp_a
    state[1, :, 2] = lactate * exp_b + transfer


def _free_step(state, frequencies, t2, kpl, r1, duration):
    if duration == 0:
        return
    for pool in range(2):
        transverse = state[pool, :, 0] + 1j * state[pool, :, 1]
        transverse *= np.exp(
            (-1.0 / t2[pool] - 2j * np.pi * frequencies[pool]) * duration
        )
        state[pool, :, 0] = transverse.real
        state[pool, :, 1] = transverse.imag
    _longitudinal_step(state, kpl, r1[0], r1[1], duration)


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
    compiled = SequenceCompiler().compile(
        program,
        checkpoints_s=checkpoints_s,
        simulation_timestep_s=simulation_timestep_s,
        status_callback=status_callback,
    )
    active = np.flatnonzero(phantom.mask.ravel())
    if active.size == 0:
        raise ValueError("dynamic phantom has no initial hyperpolarized magnetization")
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
    for interval in range(interval_count):
        if cancel_callback is not None and cancel_callback():
            raise RuntimeError("sequence simulation cancelled")
        dt = compiled.dt_s[interval]
        gradient_frequency = positions @ compiled.gradient_hz_per_m[interval]
        frequencies = b0[None, :] + pool_offsets[:, None] + gradient_frequency[None, :]
        _free_step(state, frequencies, t2, kpl, r1, dt / 2.0)
        _rf_rotate(state, compiled.rf_hz[interval], dt)
        _free_step(state, frequencies, t2, kpl, r1, dt / 2.0)
        observe(interval + 1)
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
        },
        pool_names=tuple(pool.name for pool in phantom.pools),
        species_signal=species_signal,
        final_pool_magnetization=final_pool,
        checkpoint_pool_magnetization=checkpoint_pool,
    )
