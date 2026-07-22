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
    rasterize_kpl_regions,
)
from .units import hz_to_ppm


@dataclass
class SpectralPeakDefinition:
    """One Lorentzian peak assigned to a spatial shape."""

    name: str = "Water"
    amplitude: float = 1.0
    frequency_ppm: float = 0.0
    t2_star_s: float = 0.050
    frequency_hz: float = None
    t1_s: Optional[float] = None

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

    def effective_t1_s(self, fallback_t1_s: float) -> float:
        """Return this metabolite's T1 or the enclosing shape default."""
        return float(fallback_t1_s if self.t1_s is None else self.t1_s)


@dataclass
class ShapeDefinition:
    """Ellipsoid or box in normalized phantom coordinates ``[0, 1]``."""

    name: str
    kind: str = "ellipsoid"
    center: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    size: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    t1_s: float = 1.0
    initial_mz: float = 1.0
    b0_ppm: float = 0.0
    peaks: List[SpectralPeakDefinition] = field(
        default_factory=lambda: [SpectralPeakDefinition()]
    )
    b0_hz: float = None

    def validate(self) -> None:
        if self.kind not in {"ellipsoid", "box"}:
            raise ValueError("shape kind must be 'ellipsoid' or 'box'")
        if not self.name.strip():
            raise ValueError("shape name must not be empty")
        if len(self.center) != 3 or not np.all(np.isfinite(self.center)):
            raise ValueError("shape center must contain three finite values")
        if len(self.size) != 3 or not np.all(np.isfinite(self.size)):
            raise ValueError("shape size must contain three finite values")
        if np.any(np.asarray(self.size) <= 0):
            raise ValueError("shape size must be positive")
        if not np.isfinite(self.t1_s) or self.t1_s <= 0:
            raise ValueError("shape T1 must be positive and finite")
        if not np.isfinite(self.initial_mz) or self.initial_mz < 0:
            raise ValueError("shape initial Mz must be finite and non-negative")
        if not np.isfinite(self.b0_ppm):
            raise ValueError("shape B0 offset must be finite")
        if self.b0_hz is not None and not np.isfinite(self.b0_hz):
            raise ValueError("legacy shape B0 offset must be finite")
        if not self.peaks:
            raise ValueError("each shape requires at least one spectral peak")
        for peak in self.peaks:
            peak.validate()


@dataclass
class PhantomDesign:
    """Editable geometry which can be rasterized to a :class:`SpectralPhantom`."""

    name: str = "Designed spectral phantom"
    shape: Tuple[int, int, int] = (128, 128, 128)
    fov_m: Tuple[float, float, float] = (0.22, 0.22, 0.22)
    spectral_reference_ppm: float = 0.0
    spectral_bandwidth_ppm: float = 20.0
    spectral_points: int = 1024
    shapes: List[ShapeDefinition] = field(default_factory=list)
    b0_inhomogeneity_mode: str = "none"
    b0_inhomogeneity_ppm: float = 0.0
    dynamic_enabled: bool = False
    pyruvate_peak_name: str = "Pyruvate"
    lactate_peak_name: str = "Lactate"
    default_kpl_s_inv: float = 0.0
    kinetic_regions: List[KineticRegionDefinition] = field(default_factory=list)
    pyruvate_inflow_curve: Optional[TimeCurve] = None
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
        if self.pyruvate_inflow_curve is not None and np.any(
            np.asarray(self.pyruvate_inflow_curve.values) < 0
        ):
            raise ValueError("pyruvate inflow rates must be non-negative")
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

    def rasterize_mask(self, item: ShapeDefinition) -> np.ndarray:
        """Rasterize one normalized shape using voxel-centre coordinates."""
        coords = [(np.arange(count, dtype=float) + 0.5) / count for count in self.shape]
        x, y, z = np.meshgrid(*coords, indexing="ij")
        centre = np.asarray(item.center, dtype=float)
        half_size = np.asarray(item.size, dtype=float) / 2.0
        if item.kind == "box":
            return (
                (np.abs(x - centre[0]) <= half_size[0])
                & (np.abs(y - centre[1]) <= half_size[1])
                & (np.abs(z - centre[2]) <= half_size[2])
            )
        return ((x - centre[0]) / half_size[0]) ** 2 + (
            (y - centre[1]) / half_size[1]
        ) ** 2 + ((z - centre[2]) / half_size[2]) ** 2 <= 1.0

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

    def build(self):
        """Build independent spectral components from all shapes and peaks."""
        self.validate()
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
            b0_map[region] = item.b0_hz if uses_legacy_b0 else item.b0_ppm
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
                initial_mz_maps[component_name] = region.astype(
                    float
                ) * item.initial_mz + (~region).astype(float)
        if not uses_legacy_b0:
            b0_map += self.rasterize_b0_inhomogeneity()
        if self.dynamic_enabled:
            if uses_legacy_b0:
                raise ValueError("dynamic designs require ppm-based B0 maps")
            target_names = (self.pyruvate_peak_name, self.lactate_peak_name)
            maps = {name: np.zeros(self.shape, dtype=float) for name in target_names}
            definitions = {name: [] for name in target_names}
            for item in self.shapes:
                region = self.rasterize_mask(item)
                for peak in item.peaks:
                    if peak.name in maps:
                        maps[peak.name] += (
                            region.astype(float) * peak.amplitude * item.initial_mz
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
                kpl_map_s_inv=rasterize_kpl_regions(
                    self.shape,
                    tuple(self.kinetic_regions),
                    self.default_kpl_s_inv,
                ),
                b0_map_ppm=b0_map,
                spectral_reference_ppm=float(self.spectral_reference_ppm),
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
                    )
                ),
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
            spectral_reference_ppm=float(self.spectral_reference_ppm),
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
                        amplitude=float(peak.get("amplitude", 1.0)),
                        frequency_ppm=float(frequency_ppm),
                        t2_star_s=float(peak.get("t2_star_s", 0.050)),
                        t1_s=(
                            None if peak.get("t1_s") is None else float(peak["t1_s"])
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
                )
            )
        return cls(
            name=str(data.get("name", "Designed spectral phantom")),
            shape=tuple(int(value) for value in data.get("shape", (128, 128, 128))),
            fov_m=tuple(
                float(value) for value in data.get("fov_m", (0.22, 0.22, 0.22))
            ),
            spectral_reference_ppm=float(data.get("spectral_reference_ppm", 0.0)),
            spectral_bandwidth_ppm=float(data.get("spectral_bandwidth_ppm", 20.0)),
            spectral_points=int(data.get("spectral_points", 1024)),
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
