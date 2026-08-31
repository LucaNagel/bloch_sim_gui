"""Serializable shape-based spectral phantom designs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .spectral_phantom import ChemicalSpecies, SpectralPhantom
from .dynamic_phantom import (
    DynamicB0,
    DynamicSpectralPhantom,
    KineticRegionDefinition,
    PyruvateInflow,
    TimeCurve,
)
from .units import NUCLEUS_GAMMA_HZ_PER_T, hz_to_ppm


@dataclass
class SpectralPeakDefinition:
    """One Lorentzian peak assigned to a spatial shape."""

    name: str = "Water"
    amplitude: float = 1.0
    frequency_ppm: float = 0.0
    t2_star_s: float = 0.050
    frequency_hz: float = None
    t1_s: Optional[float] = None
    initial_polarization: Optional[float] = None

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("peak name must not be empty")
        if not np.isfinite(self.amplitude) or self.amplitude < 0:
            raise ValueError("peak amplitude must be finite and non-negative")
        if not np.isfinite(self.frequency_ppm):
            raise ValueError("peak frequency must be finite")
        if self.frequency_hz is not None and not np.isfinite(self.frequency_hz):
            raise ValueError("legacy peak frequency must be finite")
        if not np.isfinite(self.t2_star_s) or self.t2_star_s <= 0:
            raise ValueError("peak T2* must be positive and finite")
        if self.t1_s is not None and (not np.isfinite(self.t1_s) or self.t1_s <= 0):
            raise ValueError("peak T1 must be positive and finite")
        if self.initial_polarization is not None and (
            not np.isfinite(self.initial_polarization) or self.initial_polarization < 0
        ):
            raise ValueError(
                "peak initial polarization must be finite and non-negative"
            )

    def effective_t1_s(self, fallback_t1_s: float) -> float:
        """Return this metabolite's T1 or the enclosing shape default."""
        return float(fallback_t1_s if self.t1_s is None else self.t1_s)

    def effective_initial_polarization(self, fallback: float) -> float:
        """Return this metabolite's polarization or the enclosing shape default."""
        return float(
            fallback if self.initial_polarization is None else self.initial_polarization
        )


@dataclass
class ShapeDefinition:
    """Rotatable primitive in normalized phantom coordinates ``[0, 1]``.

    ``rotation_deg`` contains Euler rotations about X, Y, and Z.  The rotations
    are applied in that order in physical space.  A cylinder's unrotated local
    axis is Z; ``size`` specifies its X/Y diameters and Z length.
    """

    name: str
    kind: str = "cylinder"
    center: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    size: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    t1_s: float = 1.0
    initial_mz: float = 1.0
    b0_ppm: float = 0.0
    peaks: List[SpectralPeakDefinition] = field(
        default_factory=lambda: [SpectralPeakDefinition()]
    )
    b0_hz: float = None
    rotation_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    kpl_s_inv: Optional[float] = None

    def validate(self) -> None:
        if self.kind not in {"ellipsoid", "box", "cylinder"}:
            raise ValueError("shape kind must be 'ellipsoid', 'box', or 'cylinder'")
        if not self.name.strip():
            raise ValueError("shape name must not be empty")
        if len(self.center) != 3 or not np.all(np.isfinite(self.center)):
            raise ValueError("shape center must contain three finite values")
        if len(self.size) != 3 or not np.all(np.isfinite(self.size)):
            raise ValueError("shape size must contain three finite values")
        if np.any(np.asarray(self.size) <= 0):
            raise ValueError("shape size must be positive")
        if len(self.rotation_deg) != 3 or not np.all(np.isfinite(self.rotation_deg)):
            raise ValueError("shape rotation must contain three finite angles")
        if not np.isfinite(self.t1_s) or self.t1_s <= 0:
            raise ValueError("shape T1 must be positive and finite")
        if not np.isfinite(self.initial_mz) or self.initial_mz < 0:
            raise ValueError("shape initial Mz must be finite and non-negative")
        if not np.isfinite(self.b0_ppm):
            raise ValueError("shape B0 offset must be finite")
        if self.b0_hz is not None and not np.isfinite(self.b0_hz):
            raise ValueError("legacy shape B0 offset must be finite")
        if self.kpl_s_inv is not None and (
            not np.isfinite(self.kpl_s_inv) or self.kpl_s_inv < 0
        ):
            raise ValueError("shape kPL must be finite and non-negative")
        if not self.peaks:
            raise ValueError("each shape requires at least one spectral peak")
        for peak in self.peaks:
            peak.validate()

    def effective_kpl_s_inv(self, fallback: float) -> float:
        """Return this shape's kPL or a legacy design-wide fallback."""
        return float(fallback if self.kpl_s_inv is None else self.kpl_s_inv)


@dataclass
class PhantomDesign:
    """Editable geometry which can be rasterized to a :class:`SpectralPhantom`."""

    name: str = "Designed spectral phantom"
    shape: Tuple[int, int, int] = (128, 128, 128)
    fov_m: Tuple[float, float, float] = (0.22, 0.22, 0.22)
    field_strength_t: float = 3.0
    nucleus: Optional[str] = None
    spectral_reference_ppm: float = 0.0
    spectral_window_center_ppm: Optional[float] = None
    spectral_bandwidth_ppm: float = 20.0
    spectral_points: int = 1024
    supersampling_enabled: bool = True
    supersampling_factor: int = 4
    shapes: List[ShapeDefinition] = field(default_factory=list)
    b0_inhomogeneity_mode: str = "none"
    b0_inhomogeneity_ppm: float = 0.0
    dynamic_enabled: bool = False
    pyruvate_peak_name: str = "Pyruvate"
    lactate_peak_name: str = "Lactate"
    default_kpl_s_inv: float = 0.0
    kinetic_regions: List[KineticRegionDefinition] = field(default_factory=list)
    pyruvate_inflow_curve: Optional[TimeCurve] = None
    pyruvate_inflow_polarization_curve: Optional[TimeCurve] = None
    conversion_start_s: float = 0.0
    kinetics_time_offset_s: float = 0.0
    dynamic_b0_curve: Optional[TimeCurve] = None

    def validate(self) -> None:
        if len(self.shape) != 3 or any(
            int(value) != value or value <= 0 for value in self.shape
        ):
            raise ValueError("design matrix must contain three positive integers")
        if len(self.fov_m) != 3 or not np.all(np.isfinite(self.fov_m)):
            raise ValueError("design FOV must contain three finite values")
        if np.any(np.asarray(self.fov_m) <= 0):
            raise ValueError("design FOV must be positive")
        if not np.isfinite(self.field_strength_t) or self.field_strength_t <= 0:
            raise ValueError("design field strength must be positive and finite")
        if self.nucleus is not None and self.nucleus not in NUCLEUS_GAMMA_HZ_PER_T:
            raise ValueError(f"unsupported nucleus {self.nucleus!r}")
        if not np.isfinite(self.spectral_reference_ppm):
            raise ValueError("spectral reference must be finite")
        if self.spectral_window_center_ppm is not None and not np.isfinite(
            self.spectral_window_center_ppm
        ):
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
        if (
            int(self.supersampling_factor) != self.supersampling_factor
            or self.supersampling_factor < 2
        ):
            raise ValueError("supersampling factor must be an integer >= 2")
        if self.b0_inhomogeneity_mode not in {
            "none",
            "linear_x",
            "linear_y",
            "linear_z",
            "radial_xy",
            "radial_xyz",
        }:
            raise ValueError("unsupported B0 inhomogeneity mode")

        if not np.isfinite(self.b0_inhomogeneity_ppm):
            raise ValueError("B0 inhomogeneity amplitude must be finite")
        if not np.isfinite(self.default_kpl_s_inv) or self.default_kpl_s_inv < 0:
            raise ValueError("default kPL must be finite and non-negative")
        if not np.isfinite(self.conversion_start_s):
            raise ValueError("conversion start time must be finite")
        if not np.isfinite(self.kinetics_time_offset_s):
            raise ValueError("kinetics time offset must be finite")
        if self.pyruvate_inflow_curve is not None and np.any(
            np.asarray(self.pyruvate_inflow_curve.values) < 0
        ):
            raise ValueError("pyruvate inflow rates must be non-negative")
        if self.pyruvate_inflow_polarization_curve is not None and np.any(
            np.asarray(self.pyruvate_inflow_polarization_curve.values) < 0
        ):
            raise ValueError("pyruvate inflow polarization must be non-negative")
        if (
            self.pyruvate_inflow_polarization_curve is not None
            and self.pyruvate_inflow_curve is None
        ):
            raise ValueError("pyruvate inflow polarization requires an inflow rate")
        if not self.pyruvate_peak_name.strip() or not self.lactate_peak_name.strip():
            raise ValueError("dynamic pool peak names must not be empty")
        if self.pyruvate_peak_name == self.lactate_peak_name:
            raise ValueError("pyruvate and lactate peak names must differ")
        region_names = [region.name for region in self.kinetic_regions]
        if len(region_names) != len(set(region_names)):
            raise ValueError("kinetic region names must be unique")
        for region in self.kinetic_regions:
            region.validate()
        if not self.shapes:
            raise ValueError("design requires at least one shape")
        names = [shape.name for shape in self.shapes]
        if len(set(names)) != len(names):
            raise ValueError("shape names must be unique")
        for item in self.shapes:
            item.validate()

    @property
    def effective_spectral_window_center_ppm(self) -> float:
        """Absolute center of the spectral sampling/preview window."""
        if self.spectral_window_center_ppm is None:
            return float(self.spectral_reference_ppm)
        return float(self.spectral_window_center_ppm)

    def rasterize_mask(self, item: ShapeDefinition) -> np.ndarray:
        """Rasterize a shape, optionally returning subvoxel volume fractions.

        With supersampling disabled this preserves the original boolean
        voxel-centre rasterization.  When enabled, every output voxel is sampled
        on a regular ``factor ** 3`` subvoxel grid and the samples are averaged.
        The high-resolution grid is evaluated one offset at a time so a large
        temporary volume is never allocated.
        """
        item.validate()
        if not self.supersampling_enabled:
            offsets = ((0.5, 0.5, 0.5),)
        else:
            factor = int(self.supersampling_factor)
            subvoxel_centres = (np.arange(factor, dtype=float) + 0.5) / factor
            offsets = (
                (offset_x, offset_y, offset_z)
                for offset_x in subvoxel_centres
                for offset_y in subvoxel_centres
                for offset_z in subvoxel_centres
            )

        centre = np.asarray(item.center, dtype=float)
        fov = np.asarray(self.fov_m, dtype=float)
        angles = np.deg2rad(np.asarray(item.rotation_deg, dtype=float))
        cx, cy, cz = np.cos(angles)
        sx, sy, sz = np.sin(angles)
        rotation_x = np.asarray(((1, 0, 0), (0, cx, -sx), (0, sx, cx)))
        rotation_y = np.asarray(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)))
        rotation_z = np.asarray(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)))
        # R maps local coordinates to world coordinates.  Multiplication by R
        # on row-vector world coordinates therefore applies the inverse R.T.
        rotation = rotation_z @ rotation_y @ rotation_x
        half_size = np.asarray(item.size, dtype=float) * fov / 2.0
        coverage = np.zeros(self.shape, dtype=float)
        for offset in offsets:
            axes = [
                ((np.arange(count, dtype=float) + offset[axis]) / count - centre[axis])
                * fov[axis]
                for axis, count in enumerate(self.shape)
            ]
            world_x = axes[0][:, None, None]
            world_y = axes[1][None, :, None]
            world_z = axes[2][None, None, :]
            local = [
                world_x * rotation[0, axis]
                + world_y * rotation[1, axis]
                + world_z * rotation[2, axis]
                for axis in range(3)
            ]
            if item.kind == "box":
                inside = (
                    (np.abs(local[0]) <= half_size[0])
                    & (np.abs(local[1]) <= half_size[1])
                    & (np.abs(local[2]) <= half_size[2])
                )
            else:
                radial = (local[0] / half_size[0]) ** 2 + (local[1] / half_size[1]) ** 2
                if item.kind == "cylinder":
                    inside = (radial <= 1.0) & (np.abs(local[2]) <= half_size[2])
                else:
                    inside = radial + (local[2] / half_size[2]) ** 2 <= 1.0
            coverage += inside

        if not self.supersampling_enabled:
            return coverage.astype(bool)
        return coverage / float(int(self.supersampling_factor) ** 3)

    def rasterize_b0_inhomogeneity(self) -> np.ndarray:
        """Return an analytic 2D/3D B0 map with edge amplitude in ppm."""
        mode = self.b0_inhomogeneity_mode
        amplitude = float(self.b0_inhomogeneity_ppm)
        if mode == "none" or amplitude == 0.0:
            return np.zeros(self.shape, dtype=float)
        axes = [
            2.0 * (np.arange(count, dtype=float) + 0.5) / count - 1.0
            for count in self.shape
        ]
        x, y, z = np.meshgrid(*axes, indexing="ij")
        if mode == "linear_x":
            normalized = x
        elif mode == "linear_y":
            normalized = y
        elif mode == "linear_z":
            normalized = z
        elif mode == "radial_xy":
            normalized = 2.0 * np.sqrt((x * x + y * y) / 2.0) - 1.0
        else:
            normalized = 2.0 * np.sqrt((x * x + y * y + z * z) / 3.0) - 1.0
        return amplitude * normalized

    def rasterize_kpl(self) -> np.ndarray:
        """Rasterize per-shape kPL values, followed by optional overrides.

        Designs written before shapes carried kPL retain their design-wide
        fallback. Later shapes win in overlaps, matching the other shape
        properties in the designer, and explicit kinetic regions remain the
        final, highest-priority override.
        """
        result = np.full(self.shape, float(self.default_kpl_s_inv), dtype=float)
        for item in self.shapes:
            if item.kpl_s_inv is None:
                continue
            support = np.asarray(self.rasterize_mask(item)) > 0.0
            result[support] = float(item.kpl_s_inv)
        for region in self.kinetic_regions:
            result[region.rasterize(self.shape)] = region.kpl_s_inv
        return result

    def build(self):
        """Build independent spectral components from all shapes and peaks."""
        self.validate()
        effective_nucleus = self.nucleus or ("C13" if self.dynamic_enabled else "H1")
        species = []
        concentration_maps: Dict[str, np.ndarray] = {}
        initial_mz_maps: Dict[str, np.ndarray] = {}
        uses_legacy_b0 = any(item.b0_hz is not None for item in self.shapes)
        if uses_legacy_b0 and any(item.b0_hz is None for item in self.shapes):
            raise ValueError("cannot mix ppm and legacy Hz B0 shape definitions")
        if uses_legacy_b0 and self.b0_inhomogeneity_mode != "none":
            raise ValueError("cannot add a ppm B0 inhomogeneity to legacy Hz B0 data")
        b0_map = np.zeros(self.shape, dtype=float)
        for item in self.shapes:
            region = self.rasterize_mask(item)
            b0_value = item.b0_hz if uses_legacy_b0 else item.b0_ppm
            if region.dtype == np.bool_:
                b0_map[region] = b0_value
            else:
                # Preserve the documented later-shape overwrite order while
                # blending partially occupied boundary voxels by volume.
                b0_map = b0_map * (1.0 - region) + float(b0_value) * region
            for peak in item.peaks:
                component_name = f"{item.name}: {peak.name}"
                species.append(
                    ChemicalSpecies(
                        name=component_name,
                        chemical_shift_ppm=peak.frequency_ppm,
                        t1=peak.effective_t1_s(item.t1_s),
                        t2=peak.t2_star_s,
                        t2_star=peak.t2_star_s,
                        frequency_offset_hz=peak.frequency_hz,
                    )
                )
                concentration_maps[component_name] = (
                    region.astype(float) * peak.amplitude
                )
                region_float = region.astype(float)
                initial_polarization = peak.effective_initial_polarization(
                    item.initial_mz
                )
                initial_mz_maps[component_name] = 1.0 + region_float * (
                    initial_polarization - 1.0
                )
        if not uses_legacy_b0:
            b0_map += self.rasterize_b0_inhomogeneity()
        if self.dynamic_enabled:
            if uses_legacy_b0:
                raise ValueError("dynamic designs require ppm-based B0 maps")
            target_names = (self.pyruvate_peak_name, self.lactate_peak_name)
            maps = {name: np.zeros(self.shape, dtype=float) for name in target_names}
            spin_density_maps = {
                name: np.zeros(self.shape, dtype=float) for name in target_names
            }
            definitions = {name: [] for name in target_names}
            for item in self.shapes:
                region = self.rasterize_mask(item)
                for peak in item.peaks:
                    if peak.name in maps:
                        spin_density_maps[peak.name] += (
                            region.astype(float) * peak.amplitude
                        )
                        maps[peak.name] += (
                            region.astype(float)
                            * peak.amplitude
                            * peak.effective_initial_polarization(item.initial_mz)
                        )
                        definitions[peak.name].append((item, peak))
            missing = [name for name in target_names if not definitions[name]]
            if missing:
                raise ValueError(
                    "dynamic design is missing peak definition(s): "
                    + ", ".join(missing)
                )
            pools = []
            for name in target_names:
                first_shape, first_peak = definitions[name][0]
                first_t1_s = first_peak.effective_t1_s(first_shape.t1_s)
                for item, peak in definitions[name][1:]:
                    if (
                        not np.isclose(peak.effective_t1_s(item.t1_s), first_t1_s)
                        or not np.isclose(peak.t2_star_s, first_peak.t2_star_s)
                        or not np.isclose(peak.frequency_ppm, first_peak.frequency_ppm)
                    ):
                        raise ValueError(
                            f"all {name!r} peaks require identical T1, T2*, and frequency"
                        )
                pools.append(
                    ChemicalSpecies(
                        name=name,
                        chemical_shift_ppm=first_peak.frequency_ppm,
                        t1=first_t1_s,
                        t2=first_peak.t2_star_s,
                        t2_star=first_peak.t2_star_s,
                    )
                )
            delivery_map = np.zeros(self.shape, dtype=float)
            for item in self.shapes:
                if any(peak.name == self.pyruvate_peak_name for peak in item.peaks):
                    delivery_map += self.rasterize_mask(item).astype(float)
            return DynamicSpectralPhantom(
                shape=tuple(int(value) for value in self.shape),
                fov=tuple(float(value) for value in self.fov_m),
                pools=tuple(pools),
                initial_concentration_maps=maps,
                initial_spin_density_maps=spin_density_maps,
                equilibrium_polarization=1.0,
                kpl_map_s_inv=self.rasterize_kpl(),
                b0_map_ppm=b0_map,
                field_strength=float(self.field_strength_t),
                nucleus=effective_nucleus,
                spectral_reference_ppm=float(self.spectral_reference_ppm),
                spectral_window_center_ppm=(self.effective_spectral_window_center_ppm),
                spectral_bandwidth_ppm=float(self.spectral_bandwidth_ppm),
                spectral_points=int(self.spectral_points),
                name=self.name,
                kinetic_regions=tuple(self.kinetic_regions),
                pyruvate_inflow=(
                    None
                    if self.pyruvate_inflow_curve is None
                    else PyruvateInflow(
                        rate_curve_s_inv=self.pyruvate_inflow_curve,
                        delivery_map=delivery_map,
                        polarization_curve=self.pyruvate_inflow_polarization_curve,
                    )
                ),
                conversion_start_s=float(self.conversion_start_s),
                kinetics_time_offset_s=float(self.kinetics_time_offset_s),
                dynamic_b0=(
                    None
                    if self.dynamic_b0_curve is None
                    else DynamicB0(
                        offset_curve_hz=self.dynamic_b0_curve,
                        spatial_scale_map=np.ones(self.shape, dtype=float),
                    )
                ),
                metadata={"phantom_design": self.to_dict()},
            )
        return SpectralPhantom(
            shape=tuple(int(value) for value in self.shape),
            fov=tuple(float(value) for value in self.fov_m),
            species=species,
            concentration_maps=concentration_maps,
            initial_mz_maps=initial_mz_maps,
            b0_map=b0_map if uses_legacy_b0 else None,
            b0_map_ppm=None if uses_legacy_b0 else b0_map,
            field_strength=float(self.field_strength_t),
            nucleus=effective_nucleus,
            spectral_reference_ppm=float(self.spectral_reference_ppm),
            spectral_window_center_ppm=self.effective_spectral_window_center_ppm,
            spectral_bandwidth_ppm=float(self.spectral_bandwidth_ppm),
            spectral_points=int(self.spectral_points),
            name=self.name,
            metadata={"phantom_design": self.to_dict()},
        )

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        data: Dict,
        *,
        legacy_field_strength_t: float = 3.0,
        legacy_nucleus: str = "H1",
    ) -> "PhantomDesign":
        shapes = []
        for item in data.get("shapes", []):
            peaks = []
            for peak in item.get("peaks", []):
                amplitude = float(peak.get("amplitude", 1.0))
                initial_polarization = peak.get("initial_polarization")
                if (
                    bool(data.get("dynamic_enabled", False))
                    and "initial_polarization" not in peak
                ):
                    # Older dynamic designs called amplitude an initial pool
                    # weight and multiplied it by the shape-wide HP Mz scale.
                    # Re-express the same product using the now-separate terms.
                    initial_polarization = (
                        float(item.get("initial_mz", 1.0)) * amplitude
                    )
                    amplitude = 1.0
                frequency_ppm = peak.get("frequency_ppm")
                legacy_frequency_hz = peak.get("frequency_hz")
                if legacy_frequency_hz is not None:
                    if frequency_ppm not in (None, 0, 0.0):
                        raise ValueError(
                            "peak metadata cannot combine ppm and legacy Hz frequency"
                        )
                    frequency_ppm = float(
                        hz_to_ppm(
                            legacy_frequency_hz,
                            legacy_field_strength_t,
                            legacy_nucleus,
                        )
                    )
                elif frequency_ppm is None:
                    frequency_ppm = 0.0
                peaks.append(
                    SpectralPeakDefinition(
                        name=str(peak.get("name", "Water")),
                        amplitude=amplitude,
                        frequency_ppm=float(frequency_ppm),
                        t2_star_s=float(peak.get("t2_star_s", 0.050)),
                        t1_s=(
                            None if peak.get("t1_s") is None else float(peak["t1_s"])
                        ),
                        initial_polarization=(
                            None
                            if initial_polarization is None
                            else float(initial_polarization)
                        ),
                    )
                )
            b0_ppm = item.get("b0_ppm")
            legacy_b0_hz = item.get("b0_hz")
            if legacy_b0_hz is not None:
                if b0_ppm not in (None, 0, 0.0):
                    raise ValueError(
                        "shape metadata cannot combine ppm and legacy Hz B0"
                    )
                b0_ppm = float(
                    hz_to_ppm(
                        legacy_b0_hz,
                        legacy_field_strength_t,
                        legacy_nucleus,
                    )
                )
            elif b0_ppm is None:
                b0_ppm = 0.0
            shapes.append(
                ShapeDefinition(
                    name=item["name"],
                    kind=item.get("kind", "ellipsoid"),
                    center=tuple(item.get("center", (0.5, 0.5, 0.5))),
                    size=tuple(item.get("size", (0.5, 0.5, 0.5))),
                    t1_s=float(item.get("t1_s", 1.0)),
                    initial_mz=float(item.get("initial_mz", 1.0)),
                    b0_ppm=float(b0_ppm),
                    peaks=peaks or [SpectralPeakDefinition()],
                    rotation_deg=tuple(item.get("rotation_deg", (0.0, 0.0, 0.0))),
                    kpl_s_inv=(
                        None
                        if item.get("kpl_s_inv") is None
                        else float(item["kpl_s_inv"])
                    ),
                )
            )
        return cls(
            name=str(data.get("name", "Designed spectral phantom")),
            shape=tuple(int(value) for value in data.get("shape", (128, 128, 128))),
            fov_m=tuple(
                float(value) for value in data.get("fov_m", (0.22, 0.22, 0.22))
            ),
            field_strength_t=float(
                data.get("field_strength_t", legacy_field_strength_t)
            ),
            nucleus=(
                None
                if data.get("nucleus") is None and "nucleus" in data
                else str(data.get("nucleus", legacy_nucleus))
            ),
            spectral_reference_ppm=float(data.get("spectral_reference_ppm", 0.0)),
            spectral_window_center_ppm=(
                None
                if data.get("spectral_window_center_ppm") is None
                else float(data["spectral_window_center_ppm"])
            ),
            spectral_bandwidth_ppm=float(data.get("spectral_bandwidth_ppm", 20.0)),
            spectral_points=int(data.get("spectral_points", 1024)),
            # Preserve the voxel-centre rasterization used by designs saved
            # before supersampling metadata existed. New designs use the
            # dataclass default above and therefore start with it enabled.
            supersampling_enabled=bool(data.get("supersampling_enabled", False)),
            supersampling_factor=int(data.get("supersampling_factor", 4)),
            shapes=shapes,
            b0_inhomogeneity_mode=str(data.get("b0_inhomogeneity_mode", "none")),
            b0_inhomogeneity_ppm=float(data.get("b0_inhomogeneity_ppm", 0.0)),
            dynamic_enabled=bool(data.get("dynamic_enabled", False)),
            pyruvate_peak_name=str(data.get("pyruvate_peak_name", "Pyruvate")),
            lactate_peak_name=str(data.get("lactate_peak_name", "Lactate")),
            default_kpl_s_inv=float(data.get("default_kpl_s_inv", 0.0)),
            kinetic_regions=[
                KineticRegionDefinition(
                    name=item["name"],
                    kind=item.get("kind", "ellipsoid"),
                    center=tuple(item.get("center", (0.5, 0.5, 0.5))),
                    size=tuple(item.get("size", (0.5, 0.5, 0.5))),
                    kpl_s_inv=float(item.get("kpl_s_inv", 0.0)),
                )
                for item in data.get("kinetic_regions", [])
            ],
            pyruvate_inflow_curve=(
                None
                if data.get("pyruvate_inflow_curve") is None
                else TimeCurve.from_dict(data["pyruvate_inflow_curve"])
            ),
            pyruvate_inflow_polarization_curve=(
                None
                if data.get("pyruvate_inflow_polarization_curve") is None
                else TimeCurve.from_dict(data["pyruvate_inflow_polarization_curve"])
            ),
            conversion_start_s=float(data.get("conversion_start_s", 0.0)),
            kinetics_time_offset_s=float(data.get("kinetics_time_offset_s", 0.0)),
            dynamic_b0_curve=(
                None
                if data.get("dynamic_b0_curve") is None
                else TimeCurve.from_dict(data["dynamic_b0_curve"])
            ),
        )

    @classmethod
    def from_phantom(cls, phantom: SpectralPhantom) -> "PhantomDesign":
        data = phantom.metadata.get("phantom_design")
        if data is None:
            raise ValueError(
                "spectral phantom does not contain editable shape metadata"
            )
        return cls.from_dict(
            data,
            legacy_field_strength_t=phantom.field_strength,
            legacy_nucleus=phantom.nucleus,
        )
