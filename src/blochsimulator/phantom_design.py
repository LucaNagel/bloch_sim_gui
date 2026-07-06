"""Serializable shape-based spectral phantom designs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .spectral_phantom import ChemicalSpecies, SpectralPhantom


@dataclass
class SpectralPeakDefinition:
    """One Lorentzian peak assigned to a spatial shape."""

    name: str = "Water"
    amplitude: float = 1.0
    frequency_hz: float = 0.0
    t2_star_s: float = 0.050

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("peak name must not be empty")
        if not np.isfinite(self.amplitude) or self.amplitude < 0:
            raise ValueError("peak amplitude must be finite and non-negative")
        if not np.isfinite(self.frequency_hz):
            raise ValueError("peak frequency must be finite")
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
    b0_hz: float = 0.0
    peaks: List[SpectralPeakDefinition] = field(
        default_factory=lambda: [SpectralPeakDefinition()]
    )

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
        if not np.isfinite(self.b0_hz):
            raise ValueError("shape B0 offset must be finite")
        if not self.peaks:
            raise ValueError("each shape requires at least one spectral peak")
        for peak in self.peaks:
            peak.validate()


@dataclass
class PhantomDesign:
    """Editable geometry which can be rasterized to a :class:`SpectralPhantom`."""

    name: str = "Designed spectral phantom"
    shape: Tuple[int, int, int] = (32, 32, 16)
    fov_m: Tuple[float, float, float] = (0.22, 0.22, 0.006)
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
        b0_map = np.zeros(self.shape, dtype=float)
        for item in self.shapes:
            region = self.rasterize_mask(item)
            b0_map[region] = item.b0_hz
            for peak in item.peaks:
                component_name = f"{item.name}: {peak.name}"
                species.append(
                    ChemicalSpecies(
                        name=component_name,
                        chemical_shift_ppm=0.0,
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
            b0_map=b0_map,
            name=self.name,
            metadata={"phantom_design": self.to_dict()},
        )

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "PhantomDesign":
        shapes = []
        for item in data.get("shapes", []):
            peaks = [SpectralPeakDefinition(**peak) for peak in item.get("peaks", [])]
            shapes.append(
                ShapeDefinition(
                    name=item["name"],
                    kind=item.get("kind", "ellipsoid"),
                    center=tuple(item.get("center", (0.5, 0.5, 0.5))),
                    size=tuple(item.get("size", (0.5, 0.5, 0.5))),
                    t1_s=float(item.get("t1_s", 1.0)),
                    b0_hz=float(item.get("b0_hz", 0.0)),
                    peaks=peaks or [SpectralPeakDefinition()],
                )
            )
        return cls(
            name=str(data.get("name", "Designed spectral phantom")),
            shape=tuple(int(value) for value in data.get("shape", (32, 32, 16))),
            fov_m=tuple(
                float(value) for value in data.get("fov_m", (0.22, 0.22, 0.006))
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
        return cls.from_dict(data)
