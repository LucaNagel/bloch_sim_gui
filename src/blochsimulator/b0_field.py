"""Static B0 inhomogeneity applied to an existing phantom.

The B0 workspace edits one field description for whichever phantom is
current: an analytic spatial pattern, a constant offset per region and an
optional uniform drift.  Regions are the shapes of a Phantom Designer phantom
or the anatomy labels of an anatomical phantom such as the renal mouse.

The generated field is applied on top of the phantom's own B0 map.  The
previously applied settings are stored with the phantom (in its design for
designer phantoms, otherwise in ``metadata["b0_field"]``), so re-applying
first removes the old field and then adds the new one; nothing else in the
map is lost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .dynamic_phantom import DynamicB0, DynamicSpectralPhantom, TimeCurve
from .phantom_design import PhantomDesign, analytic_b0_pattern
from .units import hz_to_ppm, ppm_to_hz

B0_PATTERN_MODES = (
    ("none", "None"),
    ("linear_x", "Linear X"),
    ("linear_y", "Linear Y"),
    ("linear_z", "Linear Z (3D)"),
    ("radial_xy", "Radial XY"),
    ("radial_xyz", "Radial XYZ (3D)"),
)
B0_METADATA_KEY = "b0_field"


@dataclass
class B0FieldSettings:
    """Spatial pattern, per-region offsets and drift of a B0 field."""

    mode: str = "none"
    strength_ppm: float = 0.0
    center: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    region_offsets_ppm: Dict[str, float] = field(default_factory=dict)
    drift_curve_hz: Optional[TimeCurve] = None

    def __post_init__(self) -> None:
        if self.mode not in {key for key, _label in B0_PATTERN_MODES}:
            raise ValueError(f"unsupported B0 pattern {self.mode!r}")
        self.strength_ppm = float(self.strength_ppm)
        self.center = tuple(float(value) for value in self.center)
        if len(self.center) != 3 or not np.all(np.isfinite(self.center)):
            raise ValueError("B0 pattern centre requires three finite values")
        if not np.isfinite(self.strength_ppm):
            raise ValueError("B0 pattern strength must be finite")
        self.region_offsets_ppm = {
            str(name): float(value) for name, value in self.region_offsets_ppm.items()
        }
        if not np.all(np.isfinite(list(self.region_offsets_ppm.values()))):
            raise ValueError("B0 region offsets must be finite")

    @property
    def has_pattern(self) -> bool:
        return self.mode != "none" and self.strength_ppm != 0.0

    def is_empty(self) -> bool:
        """True when the settings generate no field at all."""
        return (
            not self.has_pattern
            and not any(self.region_offsets_ppm.values())
            and self.drift_curve_hz is None
        )

    def to_dict(self) -> Dict:
        return {
            "mode": self.mode,
            "strength_ppm": self.strength_ppm,
            "center": list(self.center),
            "region_offsets_ppm": dict(self.region_offsets_ppm),
            "drift_curve_hz": (
                None if self.drift_curve_hz is None else self.drift_curve_hz.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "B0FieldSettings":
        drift = data.get("drift_curve_hz")
        return cls(
            mode=str(data.get("mode", "none")),
            strength_ppm=float(data.get("strength_ppm", 0.0)),
            center=tuple(data.get("center", (0.5, 0.5, 0.5))),
            region_offsets_ppm=dict(data.get("region_offsets_ppm", {})),
            drift_curve_hz=None if drift is None else TimeCurve.from_dict(drift),
        )


def _unique_names(names) -> List[str]:
    seen: Dict[str, int] = {}
    result = []
    for name in names:
        name = str(name)
        count = seen.get(name, 0) + 1
        seen[name] = count
        result.append(name if count == 1 else f"{name} ({count})")
    return result


class PhantomB0Field:
    """B0 field bookkeeping for one phantom.

    ``region_kind`` is ``"shape"`` for designer phantoms, ``"tissue"`` for
    phantoms with anatomy labels and ``None`` when the phantom has no regions.
    """

    def __init__(self, phantom):
        self.phantom = phantom
        self.design: Optional[PhantomDesign] = None
        self.legacy_design = False
        design_data = getattr(phantom, "metadata", {}).get("phantom_design")
        if design_data is not None:
            try:
                design = PhantomDesign.from_phantom(phantom)
            except (AttributeError, KeyError, TypeError, ValueError):
                design = None
            if design is not None and int(np.prod(design.simulation_shape)) == int(
                np.prod(phantom.shape)
            ):
                self.legacy_design = any(
                    item.b0_hz is not None for item in design.shapes
                )
                self.design = design
        self._weights = None
        labels = getattr(phantom, "anatomy_labels", None)
        if self.design is not None:
            self.region_kind = "shape"
            self.region_names = _unique_names(item.name for item in self.design.shapes)
        elif labels is not None:
            self.region_kind = "tissue"
            names = getattr(phantom, "tissue_label_names", {})
            names = names() if callable(names) else dict(names)
            present = np.unique(np.asarray(labels, dtype=int))
            self._tissue_labels = [
                int(label)
                for label in present
                if int(label) != 0 and int(label) in names
            ]
            self.region_names = [names[label] for label in self._tissue_labels]
        else:
            self.region_kind = None
            self.region_names = []

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(int(count) for count in self.phantom.shape)

    @property
    def supports_drift(self) -> bool:
        return isinstance(self.phantom, DynamicSpectralPhantom)

    @property
    def editable(self) -> bool:
        return not self.legacy_design

    # -- field strength / units ---------------------------------------
    def _field_and_nucleus(self):
        phantom = self.phantom
        field_strength = getattr(phantom, "field_strength", None)
        nucleus = getattr(phantom, "nucleus", None)
        metadata = getattr(phantom, "metadata", {})
        if field_strength is None:
            field_strength = metadata.get(
                "field_strength_t", metadata.get("field_strength", 3.0)
            )
        if nucleus is None:
            nucleus = metadata.get("nucleus", "H1")
        return float(field_strength), str(nucleus)

    def current_map_ppm(self) -> np.ndarray:
        phantom = self.phantom
        if hasattr(phantom, "get_b0_offset_map_ppm"):
            return np.asarray(phantom.get_b0_offset_map_ppm(), dtype=float).reshape(
                self.shape
            )
        b0_hz = getattr(phantom, "b0_map", None)
        if b0_hz is None:
            return np.zeros(self.shape, dtype=float)
        return np.asarray(
            hz_to_ppm(np.asarray(b0_hz, dtype=float), *self._field_and_nucleus()),
            dtype=float,
        ).reshape(self.shape)

    def _write_map_ppm(self, b0_map_ppm: np.ndarray) -> None:
        phantom = self.phantom
        b0_map_ppm = np.asarray(b0_map_ppm, dtype=float).reshape(self.shape)
        if hasattr(phantom, "b0_map_ppm"):
            phantom.b0_map = None
            phantom.b0_map_ppm = b0_map_ppm
        else:
            phantom.b0_map = np.asarray(
                ppm_to_hz(b0_map_ppm, *self._field_and_nucleus()), dtype=float
            )

    # -- stored settings -----------------------------------------------
    def stored_settings(self) -> Optional[B0FieldSettings]:
        """Settings already applied to the phantom, None when it has none."""
        if self.design is not None:
            design = self.design
            settings = B0FieldSettings(
                mode=design.b0_inhomogeneity_mode,
                strength_ppm=design.b0_inhomogeneity_ppm,
                center=tuple(design.b0_inhomogeneity_center),
                region_offsets_ppm={
                    name: (0.0 if item.b0_hz is not None else item.b0_ppm)
                    for name, item in zip(self.region_names, design.shapes)
                },
                drift_curve_hz=design.dynamic_b0_curve,
            )
            return None if settings.is_empty() else settings
        data = getattr(self.phantom, "metadata", {}).get(B0_METADATA_KEY)
        settings = None
        if data is not None:
            try:
                settings = B0FieldSettings.from_dict(data)
            except (KeyError, TypeError, ValueError):
                settings = None
        dynamic_b0 = getattr(self.phantom, "dynamic_b0", None)
        if dynamic_b0 is not None:
            settings = settings or B0FieldSettings()
            settings.drift_curve_hz = dynamic_b0.offset_curve_hz
        return settings

    # -- field generation ----------------------------------------------
    def _region_weights(self):
        if self._weights is None:
            if self.region_kind == "shape":
                design = self.design
                self._weights = [
                    np.asarray(design.rasterize_mask(item)).reshape(self.shape)
                    for item in design.shapes
                ]
            elif self.region_kind == "tissue":
                labels = np.asarray(self.phantom.anatomy_labels, dtype=int).reshape(
                    self.shape
                )
                self._weights = [labels == label for label in self._tissue_labels]
            else:
                self._weights = []
        return self._weights

    def region_mask(self, name: str) -> Optional[np.ndarray]:
        if name not in self.region_names:
            return None
        weight = self._region_weights()[self.region_names.index(name)]
        return np.asarray(weight) > 0.0

    def offset_map(self, settings: B0FieldSettings) -> np.ndarray:
        result = np.zeros(self.shape, dtype=float)
        for name, weight in zip(self.region_names, self._region_weights()):
            value = float(settings.region_offsets_ppm.get(name, 0.0))
            if weight.dtype == np.bool_:
                result[weight] = value
            else:
                # Later regions overwrite earlier ones; partially covered
                # boundary voxels are blended by volume, as in the designer.
                result = result * (1.0 - weight) + value * weight
        return result

    def pattern_map(self, settings: B0FieldSettings) -> np.ndarray:
        if not settings.has_pattern:
            return np.zeros(self.shape, dtype=float)
        if self.design is not None:
            design = self.design
            axes = design._normalized_axes()
            full_shape = design.shape
        else:
            padded = self.shape + (1,) * (3 - len(self.shape))
            full_shape = padded
            axes = None
        pattern = analytic_b0_pattern(
            settings.mode, settings.strength_ppm, settings.center, full_shape, axes
        )
        return pattern.reshape(self.shape)

    def generated_map(self, settings: Optional[B0FieldSettings]) -> np.ndarray:
        if settings is None:
            return np.zeros(self.shape, dtype=float)
        return self.offset_map(settings) + self.pattern_map(settings)

    def base_map_ppm(self) -> np.ndarray:
        """The phantom's own B0 map without the applied workspace field."""
        return self.current_map_ppm() - self.generated_map(self.stored_settings())

    def preview_map(self, settings: B0FieldSettings) -> np.ndarray:
        return self.base_map_ppm() + self.generated_map(settings)

    # -- application -----------------------------------------------------
    def apply(self, settings: B0FieldSettings) -> np.ndarray:
        """Replace the previously applied field by ``settings``."""
        if not self.editable:
            raise ValueError(
                "this phantom stores B0 offsets in Hz (legacy design); re-create "
                "its shapes in the Phantom Designer to edit B0 here"
            )
        if settings.drift_curve_hz is not None and not self.supports_drift:
            raise ValueError(
                "a B0 drift needs a time-resolved phantom; enable the kinetics "
                "model in the Phantom Designer or use a dynamic phantom"
            )
        new_map = self.base_map_ppm() + self.generated_map(settings)
        self._write_map_ppm(new_map)
        if self.supports_drift:
            self._apply_drift(settings.drift_curve_hz)
        if self.design is not None:
            self._store_in_design(settings)
        else:
            self.phantom.metadata[B0_METADATA_KEY] = {
                **settings.to_dict(),
                # The drift lives in the phantom's dynamic B0 model.
                "drift_curve_hz": None,
            }
        return new_map

    def _apply_drift(self, curve: Optional[TimeCurve]) -> None:
        phantom = self.phantom
        if curve is None:
            phantom.dynamic_b0 = None
            return
        existing = phantom.dynamic_b0
        if existing is not None:
            existing.offset_curve_hz = curve
            return
        dynamic_b0 = DynamicB0(
            offset_curve_hz=curve,
            spatial_scale_map=np.ones(self.shape, dtype=float),
            pool_scale=(1.0,) * len(phantom.pools),
        )
        dynamic_b0.validate(self.shape)
        phantom.dynamic_b0 = dynamic_b0

    def _store_in_design(self, settings: B0FieldSettings) -> None:
        design = self.design
        design.b0_inhomogeneity_mode = settings.mode
        design.b0_inhomogeneity_ppm = settings.strength_ppm
        design.b0_inhomogeneity_center = tuple(settings.center)
        for name, item in zip(self.region_names, design.shapes):
            item.b0_ppm = float(settings.region_offsets_ppm.get(name, 0.0))
        if self.supports_drift:
            design.dynamic_b0_curve = settings.drift_curve_hz
        data = self.phantom.metadata.get("phantom_design")
        if isinstance(data, dict):
            data["b0_inhomogeneity_mode"] = design.b0_inhomogeneity_mode
            data["b0_inhomogeneity_ppm"] = design.b0_inhomogeneity_ppm
            data["b0_inhomogeneity_center"] = list(design.b0_inhomogeneity_center)
            if self.supports_drift:
                data["dynamic_b0_curve"] = (
                    None
                    if design.dynamic_b0_curve is None
                    else design.dynamic_b0_curve.to_dict()
                )
            for shape_data, item in zip(data.get("shapes", []), design.shapes):
                shape_data["b0_ppm"] = item.b0_ppm
        else:
            self.phantom.metadata["phantom_design"] = design.to_dict()
