"""Serializable shape-based spectral phantom designs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .spectral_phantom import ChemicalSpecies, SpectralPhantom
from .units import hz_to_ppm


@dataclass
class SpectralPeakDefinition:
    """One Lorentzian peak assigned to a spatial shape."""

    name: str = "Water"
    amplitude: float = 1.0
    frequency_ppm: float = 0.0
    t2_star_s: float = 0.050
    frequency_hz: float = None

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


@dataclass
class ShapeDefinition:
    """Ellipsoid or box in normalized phantom coordinates ``[0, 1]``."""

    name: str
    kind: str = "ellipsoid"
    center: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    size: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    t1_s: float = 1.0
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
    shapes: List[ShapeDefinition] = field(default_factory=list)

    def validate(self) -> None:
        if len(self.shape) != 3 or any(
            int(value) != value or value <= 0 for value in self.shape
        ):
            raise ValueError("design matrix must contain three positive integers")
        if len(self.fov_m) != 3 or not np.all(np.isfinite(self.fov_m)):
            raise ValueError("design FOV must contain three finite values")
        if np.any(np.asarray(self.fov_m) <= 0):
            raise ValueError("design FOV must be positive")
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

    def build(self) -> SpectralPhantom:
        """Build independent spectral components from all shapes and peaks."""
        self.validate()
        species = []
        concentration_maps: Dict[str, np.ndarray] = {}
        uses_legacy_b0 = any(item.b0_hz is not None for item in self.shapes)
        if uses_legacy_b0 and any(item.b0_hz is None for item in self.shapes):
            raise ValueError("cannot mix ppm and legacy Hz B0 shape definitions")
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
                        t1=item.t1_s,
                        t2=peak.t2_star_s,
                        t2_star=peak.t2_star_s,
                        frequency_offset_hz=peak.frequency_hz,
                    )
                )
                concentration_maps[component_name] = (
                    region.astype(float) * peak.amplitude
                )
        return SpectralPhantom(
            shape=tuple(int(value) for value in self.shape),
            fov=tuple(float(value) for value in self.fov_m),
            species=species,
            concentration_maps=concentration_maps,
            b0_map=b0_map if uses_legacy_b0 else None,
            b0_map_ppm=None if uses_legacy_b0 else b0_map,
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
            shapes=shapes,
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
