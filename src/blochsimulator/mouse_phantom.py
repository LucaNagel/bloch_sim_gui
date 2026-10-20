"""Atlas-based mouse phantom configuration and procedural reference anatomy.

The phantom combines a compact editable anatomy, a directed major-vessel
scaffold, dose-conserving delayed inflow, organ perfusion previews, renal pH,
and optional respiratory displacement.  The same delayed inflow drives both
the sequence solver and the animated authoring preview.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from .dynamic_phantom import DynamicSpectralPhantom, PyruvateInflow, TimeCurve
from .spectral_phantom import ChemicalSpecies


MOUSE_TISSUE_LABELS = {
    0: "Background",
    1: "Soft tissue",
    2: "Left kidney cortex",
    3: "Left kidney medulla",
    4: "Right kidney cortex",
    5: "Right kidney medulla",
    6: "Liver",
    7: "Lung",
    8: "Heart",
    9: "Arterial blood",
    10: "Venous blood",
    11: "Brain",
    12: "Urinary bladder",
    13: "Left ureter",
    14: "Right ureter",
}

RENAL_LABELS = {"left": (2, 3), "right": (4, 5)}
URETER_LABELS = {"left": 13, "right": 14}
BLADDER_LABEL = 12
URINARY_LUMEN_LABELS = (BLADDER_LABEL, *URETER_LABELS.values())


INJECTION_COMPOUNDS = {
    "pyruvate": "[1-13C]Pyruvate",
    "lactate": "[1-13C]Lactate",
    "z_ompd": "[1,5-13C2]Z-OMPD",
    "contrast_agent": "Generic contrast agent",
}


# Bounds of the procedural body surface in atlas coordinates. ``mouse_size_m``
# refers to this visible bounding box, rather than to the wider abstract atlas
# coordinate cube.
MOUSE_ATLAS_BOUNDS = np.asarray(
    ((-0.62, 0.62), (-0.52, 0.52), (-1.15, 1.10)), dtype=float
)


@dataclass(frozen=True)
class VascularNode:
    """One named vascular junction in normalized mouse coordinates."""

    name: str
    position: Tuple[float, float, float]
    kind: str = "junction"

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("vascular node name must not be empty")
        values = np.asarray(self.position, dtype=float)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            raise ValueError("vascular node position requires three finite values")
        if np.any(np.abs(values) > 1.5):
            raise ValueError("vascular node position is outside the mouse template")


@dataclass(frozen=True)
class VascularEdge:
    """Directed major-vessel segment with a piecewise-linear centerline."""

    name: str
    source: str
    destination: str
    centerline: Tuple[Tuple[float, float, float], ...]
    radius_m: float
    relative_flow: float
    kind: str = "arterial"

    def validate(self, node_names: Iterable[str]) -> None:
        known = set(node_names)
        if not self.name.strip():
            raise ValueError("vascular edge name must not be empty")
        if self.source not in known or self.destination not in known:
            raise ValueError("vascular edge references an unknown node")
        points = np.asarray(self.centerline, dtype=float)
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 3:
            raise ValueError("vascular centerline requires at least two 3D points")
        if not np.all(np.isfinite(points)):
            raise ValueError("vascular centerline must be finite")
        if not np.isfinite(self.radius_m) or self.radius_m <= 0:
            raise ValueError("vascular radius must be positive and finite")
        if not np.isfinite(self.relative_flow) or self.relative_flow <= 0:
            raise ValueError("vascular relative flow must be positive and finite")
        if self.kind not in {"arterial", "venous"}:
            raise ValueError("vascular edge kind must be arterial or venous")


@dataclass(frozen=True)
class VascularGraph:
    """Small directed graph of the explicitly resolved major vessels."""

    nodes: Tuple[VascularNode, ...]
    edges: Tuple[VascularEdge, ...]

    def validate(self) -> None:
        names = [node.name for node in self.nodes]
        if not names or len(names) != len(set(names)):
            raise ValueError("vascular graph requires uniquely named nodes")
        for node in self.nodes:
            node.validate()
        edge_names = [edge.name for edge in self.edges]
        if not edge_names or len(edge_names) != len(set(edge_names)):
            raise ValueError("vascular graph requires uniquely named edges")
        for edge in self.edges:
            edge.validate(names)

    def to_dict(self) -> Dict:
        return {
            "nodes": [asdict(node) for node in self.nodes],
            "edges": [asdict(edge) for edge in self.edges],
        }

    @classmethod
    def from_dict(cls, values: Dict) -> "VascularGraph":
        graph = cls(
            nodes=tuple(VascularNode(**item) for item in values["nodes"]),
            edges=tuple(
                VascularEdge(
                    **{
                        **item,
                        "centerline": tuple(
                            tuple(point) for point in item["centerline"]
                        ),
                    }
                )
                for item in values["edges"]
            ),
        )
        graph.validate()
        return graph


@dataclass
class MousePerfusionConfig:
    """Serializable authoring configuration for the reference mouse phantom."""

    name: str = "Mouse renal perfusion phantom"
    shape: Tuple[int, int, int] = (48, 32, 96)
    fov_m: Tuple[float, float, float] = (0.045, 0.032, 0.105)
    mouse_size_m: Tuple[float, float, float] = (0.045, 0.032, 0.105)
    mouse_offset_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    field_strength_t: float = 7.0
    injection_compound: str = "pyruvate"
    injection_amount_umol: float = 0.25
    injection_start_s: float = 0.0
    injection_duration_s: float = 2.0
    circulation_delay_s: float = 0.6
    perfusion_timestep_s: float = 0.02
    injection_polarization: float = 10000.0
    bolus_alpha: float = 2.0
    bolus_beta: float = 5.0
    background_perfusion_weight: float = 0.15
    renal_cortex_perfusion_weight: float = 4.0
    renal_medulla_perfusion_weight: float = 2.0
    liver_perfusion_weight: float = 2.8
    brain_perfusion_weight: float = 2.4
    vessel_perfusion_weight: float = 8.0
    tissue_clearance_s_inv: float = 0.18
    renal_excretion_s_inv: float = 0.12
    renal_tubular_transit_s: float = 45.0
    ureter_transit_s: float = 15.0
    left_cortex_kpl_s_inv: float = 0.08
    left_medulla_kpl_s_inv: float = 0.04
    right_cortex_kpl_s_inv: float = 0.08
    right_medulla_kpl_s_inv: float = 0.04
    liver_kpa_s_inv: float = 0.05
    left_cortex_ph: float = 7.20
    left_medulla_ph: float = 6.90
    right_cortex_ph: float = 7.20
    right_medulla_ph: float = 6.90
    background_ph: float = 7.20
    breathing_enabled: bool = False
    breathing_rate_hz: float = 1.2
    breathing_amplitude_mm: float = 1.0
    breathing_phase_rad: float = 0.0
    pyruvate_t1_s: float = 25.0
    lactate_t1_s: float = 25.0
    pyruvate_t2_s: float = 0.300
    lactate_t2_s: float = 0.300
    spectral_reference_ppm: float = 171.076
    spectral_bandwidth_ppm: float = 18.0
    spectral_points: int = 257
    metadata: Dict = field(default_factory=dict)

    def validate(self) -> None:
        self.shape = tuple(int(value) for value in self.shape)
        self.fov_m = tuple(float(value) for value in self.fov_m)
        self.mouse_size_m = tuple(float(value) for value in self.mouse_size_m)
        self.mouse_offset_m = tuple(float(value) for value in self.mouse_offset_m)
        if not self.name.strip():
            raise ValueError("mouse phantom name must not be empty")
        self.injection_compound = str(self.injection_compound).strip().lower()
        if self.injection_compound not in INJECTION_COMPOUNDS:
            raise ValueError(
                "injection compound must be one of "
                + ", ".join(sorted(INJECTION_COMPOUNDS))
            )
        if len(self.shape) != 3 or any(value < 8 for value in self.shape):
            raise ValueError("mouse phantom matrix requires three values >= 8")
        if (
            len(self.fov_m) != 3
            or not np.all(np.isfinite(self.fov_m))
            or np.any(np.asarray(self.fov_m) <= 0)
        ):
            raise ValueError("mouse phantom FOV requires three positive values")
        if (
            len(self.mouse_size_m) != 3
            or not np.all(np.isfinite(self.mouse_size_m))
            or np.any(np.asarray(self.mouse_size_m) <= 0)
        ):
            raise ValueError("mouse size requires three positive values")
        if len(self.mouse_offset_m) != 3 or not np.all(
            np.isfinite(self.mouse_offset_m)
        ):
            raise ValueError("mouse position requires three finite values")
        positive = {
            "field strength": self.field_strength_t,
            "injection duration": self.injection_duration_s,
            "perfusion timestep": self.perfusion_timestep_s,
            "injection polarization": self.injection_polarization,
            "bolus alpha": self.bolus_alpha,
            "bolus beta": self.bolus_beta,
            "breathing rate": self.breathing_rate_hz,
            "pyruvate T1": self.pyruvate_t1_s,
            "lactate T1": self.lactate_t1_s,
            "pyruvate T2": self.pyruvate_t2_s,
            "lactate T2": self.lactate_t2_s,
            "spectral bandwidth": self.spectral_bandwidth_ppm,
            "tissue clearance": self.tissue_clearance_s_inv,
        }
        for label, value in positive.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{label} must be positive and finite")
        if (
            not np.isfinite(self.injection_amount_umol)
            or self.injection_amount_umol < 0
        ):
            raise ValueError("injection amount must be finite and non-negative")
        finite = (
            self.injection_start_s,
            self.circulation_delay_s,
            self.breathing_amplitude_mm,
            self.breathing_phase_rad,
            self.spectral_reference_ppm,
        )
        if not np.all(np.isfinite(finite)) or self.circulation_delay_s < 0:
            raise ValueError("mouse timing and spectral values must be finite")
        excretion = (
            self.renal_excretion_s_inv,
            self.renal_tubular_transit_s,
            self.ureter_transit_s,
        )
        if not np.all(np.isfinite(excretion)) or np.any(np.asarray(excretion) < 0):
            raise ValueError(
                "renal excretion rate and urinary transit times must be finite "
                "and non-negative"
            )
        weights = (
            self.background_perfusion_weight,
            self.renal_cortex_perfusion_weight,
            self.renal_medulla_perfusion_weight,
            self.liver_perfusion_weight,
            self.brain_perfusion_weight,
            self.vessel_perfusion_weight,
        )
        if not np.all(np.isfinite(weights)) or np.any(np.asarray(weights) < 0):
            raise ValueError("perfusion weights must be finite and non-negative")
        if not any(value > 0 for value in weights):
            raise ValueError("at least one perfusion weight must be positive")
        rates = (
            self.left_cortex_kpl_s_inv,
            self.left_medulla_kpl_s_inv,
            self.right_cortex_kpl_s_inv,
            self.right_medulla_kpl_s_inv,
            self.liver_kpa_s_inv,
        )
        if not np.all(np.isfinite(rates)) or np.any(np.asarray(rates) < 0):
            raise ValueError("renal kPL values must be finite and non-negative")
        ph_values = (
            self.left_cortex_ph,
            self.left_medulla_ph,
            self.right_cortex_ph,
            self.right_medulla_ph,
            self.background_ph,
        )
        if not np.all(np.isfinite(ph_values)) or np.any(
            (np.asarray(ph_values) < 0) | (np.asarray(ph_values) > 14)
        ):
            raise ValueError("pH values must lie between 0 and 14")
        if (
            int(self.spectral_points) != self.spectral_points
            or self.spectral_points < 2
        ):
            raise ValueError("spectral points must be an integer >= 2")

    def to_dict(self) -> Dict:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict) -> "MousePerfusionConfig":
        restored = dict(values)
        # Older saved phantoms used the FOV as the implicit [-1, 1] atlas
        # extent. Convert that extent to the approximate visible body bounds
        # used by the current, independently editable mouse size.
        # Phantoms saved before renal excretion existed keep their kinetics.
        restored.setdefault("renal_excretion_s_inv", 0.0)
        if "mouse_size_m" not in restored:
            legacy_fov = np.asarray(restored.get("fov_m", cls.fov_m), dtype=float)
            restored["mouse_size_m"] = tuple(
                legacy_fov * 0.5 * np.ptp(MOUSE_ATLAS_BOUNDS, axis=1)
            )
        config = cls(**restored)
        config.shape = tuple(config.shape)
        config.fov_m = tuple(config.fov_m)
        config.mouse_size_m = tuple(config.mouse_size_m)
        config.mouse_offset_m = tuple(config.mouse_offset_m)
        return config

    def build(self) -> "MousePerfusionPhantom":
        self.validate()
        return MousePerfusionPhantom.from_config(self)


def reference_mouse_vascular_graph() -> VascularGraph:
    """Return the deterministic major-vessel scaffold of the reference mouse."""

    nodes = (
        VascularNode("tail_vein", (0.10, 0.02, -0.92), "injection"),
        VascularNode("right_heart", (0.07, 0.02, 0.42), "heart"),
        VascularNode("lungs", (0.00, 0.00, 0.52), "lung_bed"),
        VascularNode("left_heart", (-0.07, 0.02, 0.42), "heart"),
        VascularNode("renal_aorta", (-0.055, 0.00, -0.08), "bifurcation"),
        VascularNode("left_kidney", (-0.36, -0.03, -0.10), "organ_bed"),
        VascularNode("right_kidney", (0.36, -0.03, -0.10), "organ_bed"),
        VascularNode("renal_vena_cava", (0.055, 0.00, -0.08), "junction"),
        VascularNode("aortic_arch", (-0.06, 0.01, 0.48), "bifurcation"),
        VascularNode("brain", (0.00, 0.00, 0.76), "organ_bed"),
        VascularNode("liver", (0.10, 0.08, 0.18), "organ_bed"),
    )
    edges = (
        VascularEdge(
            "tail_venous_return",
            "tail_vein",
            "right_heart",
            ((0.10, 0.02, -0.92), (0.07, 0.02, 0.42)),
            0.00055,
            1.0,
            "venous",
        ),
        VascularEdge(
            "pulmonary_artery",
            "right_heart",
            "lungs",
            ((0.07, 0.02, 0.42), (0.00, 0.00, 0.52)),
            0.00065,
            1.0,
            "venous",
        ),
        VascularEdge(
            "pulmonary_vein",
            "lungs",
            "left_heart",
            ((0.00, 0.00, 0.52), (-0.07, 0.02, 0.42)),
            0.00065,
            1.0,
            "arterial",
        ),
        VascularEdge(
            "descending_aorta",
            "left_heart",
            "renal_aorta",
            ((-0.07, 0.02, 0.42), (-0.055, 0.00, -0.08)),
            0.00060,
            1.0,
            "arterial",
        ),
        VascularEdge(
            "aortic_arch",
            "left_heart",
            "aortic_arch",
            ((-0.07, 0.02, 0.42), (-0.06, 0.01, 0.48)),
            0.00055,
            1.0,
            "arterial",
        ),
        VascularEdge(
            "carotid_artery",
            "aortic_arch",
            "brain",
            ((-0.06, 0.01, 0.48), (-0.03, 0.0, 0.62), (0.0, 0.0, 0.76)),
            0.00028,
            0.16,
            "arterial",
        ),
        VascularEdge(
            "cerebral_venous_return",
            "brain",
            "right_heart",
            ((0.0, 0.0, 0.76), (0.08, 0.01, 0.58), (0.07, 0.02, 0.42)),
            0.00030,
            0.16,
            "venous",
        ),
        VascularEdge(
            "hepatic_artery",
            "renal_aorta",
            "liver",
            ((-0.055, 0.00, -0.08), (-0.06, 0.01, 0.30), (0.10, 0.08, 0.18)),
            0.00032,
            0.25,
            "arterial",
        ),
        VascularEdge(
            "hepatic_vein",
            "liver",
            "renal_vena_cava",
            ((0.10, 0.08, 0.18), (0.055, 0.00, -0.08)),
            0.00038,
            0.25,
            "venous",
        ),
        VascularEdge(
            "left_renal_artery",
            "renal_aorta",
            "left_kidney",
            ((-0.055, 0.00, -0.08), (-0.36, -0.03, -0.10)),
            0.00035,
            0.5,
            "arterial",
        ),
        VascularEdge(
            "right_renal_artery",
            "renal_aorta",
            "right_kidney",
            ((-0.055, 0.00, -0.08), (0.36, -0.03, -0.10)),
            0.00035,
            0.5,
            "arterial",
        ),
        VascularEdge(
            "left_renal_vein",
            "left_kidney",
            "renal_vena_cava",
            ((-0.36, -0.03, -0.10), (0.055, 0.00, -0.08)),
            0.00040,
            0.5,
            "venous",
        ),
        VascularEdge(
            "right_renal_vein",
            "right_kidney",
            "renal_vena_cava",
            ((0.36, -0.03, -0.10), (0.055, 0.00, -0.08)),
            0.00040,
            0.5,
            "venous",
        ),
        VascularEdge(
            "vena_cava",
            "renal_vena_cava",
            "right_heart",
            ((0.055, 0.00, -0.08), (0.07, 0.02, 0.42)),
            0.00065,
            1.0,
            "venous",
        ),
    )
    graph = VascularGraph(nodes=nodes, edges=edges)
    graph.validate()
    return graph


def _normalized_axes(shape: Tuple[int, int, int]):
    return tuple(
        (np.arange(count, dtype=float) + 0.5) / count * 2.0 - 1.0 for count in shape
    )


def _anatomy_axes(
    shape: Tuple[int, int, int],
    fov_m: Tuple[float, float, float],
    mouse_size_m: Tuple[float, float, float],
    mouse_offset_m: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    grid_center_m: Tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    """Return voxel centres in atlas coordinates for a physical FOV."""

    normalized_fov_axes = _normalized_axes(shape)
    atlas_centers = np.mean(MOUSE_ATLAS_BOUNDS, axis=1)
    atlas_spans = np.ptp(MOUSE_ATLAS_BOUNDS, axis=1)
    return tuple(
        atlas_center
        + (float(grid_center) + axis * 0.5 * float(fov_extent) - float(mouse_offset))
        * float(atlas_span)
        / float(mouse_extent)
        for axis, fov_extent, mouse_extent, mouse_offset, grid_center, atlas_center, atlas_span in zip(
            normalized_fov_axes,
            fov_m,
            mouse_size_m,
            mouse_offset_m,
            grid_center_m,
            atlas_centers,
            atlas_spans,
        )
    )


def _ellipsoid(x, y, z, center, radii):
    return ((x - center[0]) / radii[0]) ** 2 + ((y - center[1]) / radii[1]) ** 2 + (
        (z - center[2]) / radii[2]
    ) ** 2 <= 1.0


_BLADDER_CENTER = (0.0, 0.06, -0.68)
_BLADDER_RADII = (0.12, 0.12, 0.10)
_URETER_RADIUS_M = 0.00025
_URETER_CENTERLINES = {
    "left": ((-0.22, -0.03, -0.14), (-0.16, -0.01, -0.40), (-0.07, 0.04, -0.60)),
    "right": ((0.22, -0.03, -0.14), (0.16, -0.01, -0.40), (0.07, 0.04, -0.60)),
}


def _tube(x, y, z, centerline, radius_m, atlas_to_m):
    """Return voxels within ``radius_m`` of an atlas-space polyline."""

    points = np.asarray(centerline, dtype=float) * atlas_to_m
    coordinates = np.stack((x, y, z), axis=-1) * atlas_to_m
    inside = np.zeros(x.shape, dtype=bool)
    for start, end in zip(points[:-1], points[1:]):
        direction = end - start
        parameter = np.clip(
            np.sum((coordinates - start) * direction, axis=-1)
            / float(direction @ direction),
            0.0,
            1.0,
        )
        closest = start + parameter[..., None] * direction
        inside |= np.sum((coordinates - closest) ** 2, axis=-1) <= radius_m**2
    return inside


def _rasterize_vascular_graph(
    shape: Tuple[int, int, int],
    fov_m: Tuple[float, float, float],
    graph: VascularGraph,
    mouse_size_m: Optional[Tuple[float, float, float]] = None,
    mouse_offset_m: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    grid_center_m: Tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    axes = _normalized_axes(shape)
    normalized = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    physical = normalized * (np.asarray(fov_m, dtype=float) / 2.0) + np.asarray(
        grid_center_m, dtype=float
    )
    voxel_spacing_m = np.asarray(fov_m, dtype=float) / np.asarray(shape, dtype=float)
    resolved_axes = np.asarray(shape, dtype=int) > 1
    minimum_resolved_radius_m = 0.6 * float(
        np.max(voxel_spacing_m[resolved_axes])
        if np.any(resolved_axes)
        else np.min(voxel_spacing_m)
    )
    arterial = np.zeros(shape, dtype=bool)
    venous = np.zeros(shape, dtype=bool)
    transit_fraction = np.full(shape, np.inf, dtype=float)
    edge_transit = {
        "tail_venous_return": (0.00, 0.32),
        "pulmonary_artery": (0.32, 0.46),
        "pulmonary_vein": (0.46, 0.58),
        "descending_aorta": (0.58, 0.78),
        "aortic_arch": (0.58, 0.68),
        "carotid_artery": (0.68, 0.92),
        "hepatic_artery": (0.72, 0.92),
        "left_renal_artery": (0.78, 0.96),
        "right_renal_artery": (0.78, 0.96),
        "cerebral_venous_return": (1.05, 1.28),
        "hepatic_vein": (1.05, 1.25),
        "left_renal_vein": (1.05, 1.22),
        "right_renal_vein": (1.05, 1.22),
        "vena_cava": (1.22, 1.38),
    }
    visible_mouse_size_m = np.asarray(
        fov_m if mouse_size_m is None else mouse_size_m, dtype=float
    )
    atlas_centers = np.mean(MOUSE_ATLAS_BOUNDS, axis=1)
    atlas_spans = np.ptp(MOUSE_ATLAS_BOUNDS, axis=1)
    for edge in graph.edges:
        points = (
            np.asarray(edge.centerline, dtype=float) - atlas_centers
        ) / atlas_spans * visible_mouse_size_m + np.asarray(mouse_offset_m, dtype=float)
        target = arterial if edge.kind == "arterial" else venous
        resolved_radius_m = max(edge.radius_m, minimum_resolved_radius_m)
        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        total_length = float(np.sum(segment_lengths))
        cumulative_length = 0.0
        transit_start, transit_end = edge_transit.get(edge.name, (0.7, 1.0))
        for segment_index, (start, end) in enumerate(zip(points[:-1], points[1:])):
            direction = end - start
            length_squared = float(direction @ direction)
            if length_squared == 0.0:
                continue
            relative = physical - start
            parameter = np.clip(
                np.sum(relative * direction, axis=-1) / length_squared, 0.0, 1.0
            )
            closest = start + parameter[..., None] * direction
            distance_squared = np.sum((physical - closest) ** 2, axis=-1)
            inside = distance_squared <= resolved_radius_m**2
            target |= inside
            if total_length > 0.0:
                path_fraction = (
                    cumulative_length + parameter * segment_lengths[segment_index]
                ) / total_length
                candidate = transit_start + path_fraction * (
                    transit_end - transit_start
                )
                transit_fraction[inside] = np.minimum(
                    transit_fraction[inside], candidate[inside]
                )
            cumulative_length += segment_lengths[segment_index]
    return arterial, venous, transit_fraction


def build_reference_mouse_anatomy(
    shape: Tuple[int, int, int],
    fov_m: Tuple[float, float, float],
    graph: Optional[VascularGraph] = None,
    *,
    mouse_size_m: Optional[Tuple[float, float, float]] = None,
    mouse_offset_m: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    grid_center_m: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    return_vascular_transit: bool = False,
):
    """Create a deterministic labelled mouse template and large-vessel masks."""

    graph = reference_mouse_vascular_graph() if graph is None else graph
    atlas_extent_m = fov_m if mouse_size_m is None else mouse_size_m
    x_axis, y_axis, z_axis = _anatomy_axes(
        shape,
        fov_m,
        atlas_extent_m,
        mouse_offset_m,
        grid_center_m,
    )
    x, y, z = np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")
    labels = np.zeros(shape, dtype=np.uint8)

    torso = _ellipsoid(x, y, z, (0.0, 0.0, -0.10), (0.62, 0.52, 0.78))
    tail = _ellipsoid(x, y, z, (0.10, 0.02, -0.87), (0.12, 0.10, 0.28))
    head = _ellipsoid(x, y, z, (0.0, 0.0, 0.72), (0.43, 0.39, 0.30))
    snout = _ellipsoid(x, y, z, (0.0, -0.06, 0.94), (0.24, 0.25, 0.16))
    body = torso | tail | head | snout
    labels[body] = 1

    liver = _ellipsoid(x, y, z, (0.10, 0.08, 0.18), (0.42, 0.34, 0.22)) & body
    lungs = (
        _ellipsoid(x, y, z, (-0.18, 0.0, 0.46), (0.20, 0.24, 0.22))
        | _ellipsoid(x, y, z, (0.18, 0.0, 0.46), (0.20, 0.24, 0.22))
    ) & body
    heart = _ellipsoid(x, y, z, (0.0, 0.04, 0.35), (0.16, 0.18, 0.18)) & body
    brain = _ellipsoid(x, y, z, (0.0, 0.0, 0.74), (0.30, 0.27, 0.18)) & head
    labels[liver] = 6
    labels[lungs] = 7
    labels[heart] = 8
    labels[brain] = 11

    left_kidney = _ellipsoid(x, y, z, (-0.36, -0.03, -0.10), (0.19, 0.16, 0.22)) & body
    right_kidney = _ellipsoid(x, y, z, (0.36, -0.03, -0.10), (0.19, 0.16, 0.22)) & body
    left_medulla = (
        _ellipsoid(x, y, z, (-0.36, -0.03, -0.10), (0.105, 0.09, 0.14)) & left_kidney
    )
    right_medulla = (
        _ellipsoid(x, y, z, (0.36, -0.03, -0.10), (0.105, 0.09, 0.14)) & right_kidney
    )
    labels[left_kidney] = 2
    labels[left_medulla] = 3
    labels[right_kidney] = 4
    labels[right_medulla] = 5

    kidneys = left_kidney | right_kidney
    bladder = _ellipsoid(x, y, z, _BLADDER_CENTER, _BLADDER_RADII) & body
    labels[bladder] = BLADDER_LABEL
    atlas_to_m = np.asarray(atlas_extent_m, dtype=float) / np.ptp(
        MOUSE_ATLAS_BOUNDS, axis=1
    )
    voxel_spacing_m = np.asarray(fov_m, dtype=float) / np.asarray(shape, dtype=float)
    resolved = np.asarray(shape, dtype=int) > 1
    ureter_radius_m = max(
        _URETER_RADIUS_M,
        0.6
        * float(
            np.max(voxel_spacing_m[resolved])
            if np.any(resolved)
            else np.min(voxel_spacing_m)
        ),
    )
    for side, centerline in _URETER_CENTERLINES.items():
        ureter = (
            _tube(x, y, z, centerline, ureter_radius_m, atlas_to_m)
            & body
            & ~kidneys
            & ~bladder
        )
        labels[ureter] = URETER_LABELS[side]

    arterial, venous, vascular_transit = _rasterize_vascular_graph(
        shape,
        fov_m,
        graph,
        atlas_extent_m,
        mouse_offset_m,
        grid_center_m,
    )
    labels[arterial & body] = 9
    labels[venous & body] = 10
    if return_vascular_transit:
        return labels, arterial & body, venous & body, vascular_transit
    return labels, arterial & body, venous & body


def _dose_rate_curve(config: MousePerfusionConfig, *, delay_s: float) -> TimeCurve:
    sample_count = 65
    normalized_time = np.linspace(0.0, 1.0, sample_count)
    profile = np.power(normalized_time, config.bolus_alpha) * np.power(
        1.0 - normalized_time, config.bolus_beta
    )
    times = (
        config.injection_start_s
        + float(delay_s)
        + normalized_time * config.injection_duration_s
    )
    area = float(np.sum(0.5 * (profile[:-1] + profile[1:]) * np.diff(times)))
    if area <= 0.0:
        raise ValueError("bolus profile has zero integral")
    amount_mol = config.injection_amount_umol * 1e-6
    rates_mol_s = amount_mol * profile / area
    return TimeCurve(
        tuple(float(value) for value in times),
        tuple(float(value) for value in rates_mol_s),
        interpolation="linear",
        outside="zero",
    )


def _compound_pools(config: MousePerfusionConfig) -> Tuple[ChemicalSpecies, ...]:
    """Return the spectral pools associated with the selected injection.

    Pool zero is always the injected source because this is the convention of
    :class:`DynamicSpectralPhantom`.  Non-pyruvate presets deliberately have
    zero conversion and use their second resonance only as a spectral
    reference component for now.
    """

    if config.injection_compound == "pyruvate":
        return (
            ChemicalSpecies(
                "Pyruvate",
                0.0,
                config.pyruvate_t1_s,
                config.pyruvate_t2_s,
                t2_star=config.pyruvate_t2_s,
            ),
            ChemicalSpecies(
                "Lactate",
                183.35 - config.spectral_reference_ppm,
                config.lactate_t1_s,
                config.lactate_t2_s,
                t2_star=config.lactate_t2_s,
            ),
            ChemicalSpecies(
                "Alanine",
                176.5 - config.spectral_reference_ppm,
                config.lactate_t1_s,
                config.lactate_t2_s,
                t2_star=config.lactate_t2_s,
            ),
        )
    if config.injection_compound == "lactate":
        return (
            ChemicalSpecies(
                "Lactate",
                0.0,
                config.lactate_t1_s,
                config.lactate_t2_s,
                t2_star=config.lactate_t2_s,
            ),
            ChemicalSpecies(
                "Spectral reference",
                -12.274,
                config.lactate_t1_s,
                config.lactate_t2_s,
                t2_star=config.lactate_t2_s,
            ),
        )
    if config.injection_compound == "z_ompd":
        return (
            ChemicalSpecies(
                "Z-OMPD C5",
                3.0,
                config.pyruvate_t1_s,
                config.pyruvate_t2_s,
                t2_star=config.pyruvate_t2_s,
            ),
            ChemicalSpecies(
                "Z-OMPD C1 reference",
                0.4,
                config.pyruvate_t1_s,
                config.pyruvate_t2_s,
                t2_star=config.pyruvate_t2_s,
            ),
        )
    return (
        ChemicalSpecies(
            "Contrast agent",
            0.0,
            config.pyruvate_t1_s,
            config.pyruvate_t2_s,
            t2_star=config.pyruvate_t2_s,
        ),
        ChemicalSpecies(
            "Spectral reference",
            0.5 * config.spectral_bandwidth_ppm,
            config.pyruvate_t1_s,
            config.pyruvate_t2_s,
            t2_star=config.pyruvate_t2_s,
        ),
    )


def _cumulative_trapezoid(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values, dtype=float)
    result[1:] = np.cumsum(0.5 * (values[1:] + values[:-1]) * np.diff(times))
    return result


def _urinary_transport(
    times: np.ndarray,
    cumulative_excreted: np.ndarray,
    tubular_transit_s: float,
    ureter_transit_s: float,
    quadrature_points: int = 32,
):
    """Split cumulatively excreted amounts into tubules, ureter and bladder.

    Tubular transit times are uniformly distributed between 0.5 and 1.5 times
    the configured mean; ureteral transport is a plug-flow delay.  The three
    returned amounts always sum to ``cumulative_excreted``.
    """

    def delayed(values, delay_s):
        return np.interp(
            times - delay_s, times, values, left=0.0, right=float(values[-1])
        )

    fractions = 0.5 + (np.arange(quadrature_points) + 0.5) / quadrature_points
    left_tubules = np.mean(
        [
            delayed(cumulative_excreted, tubular_transit_s * fraction)
            for fraction in fractions
        ],
        axis=0,
    )
    bladder = delayed(left_tubules, ureter_transit_s)
    return (
        cumulative_excreted - left_tubules,
        left_tubules - bladder,
        bladder,
    )


class MousePerfusionPhantom(DynamicSpectralPhantom):
    """Reference mouse anatomy compiled to the existing dynamic MRI model.

    The sequence solver consumes a dose-conserving delivery map with voxelwise
    arrival delays. This produces a moving, dispersing source without claiming
    to be the later graph-based advection/exchange solver.
    """

    @classmethod
    def from_config(cls, config: MousePerfusionConfig) -> "MousePerfusionPhantom":
        config.validate()
        graph = reference_mouse_vascular_graph()
        labels, arterial, venous, vascular_transit = build_reference_mouse_anatomy(
            config.shape,
            config.fov_m,
            graph,
            mouse_size_m=config.mouse_size_m,
            mouse_offset_m=config.mouse_offset_m,
            return_vascular_transit=True,
        )
        if not np.any(labels > 0):
            raise ValueError("mouse position and FOV do not overlap")

        kpl = np.zeros(config.shape, dtype=np.float64)
        kpl[labels == 2] = config.left_cortex_kpl_s_inv
        kpl[labels == 3] = config.left_medulla_kpl_s_inv
        kpl[labels == 4] = config.right_cortex_kpl_s_inv
        kpl[labels == 5] = config.right_medulla_kpl_s_inv
        if config.injection_compound != "pyruvate":
            kpl.fill(0.0)

        kpa = np.zeros(config.shape, dtype=np.float64)
        if config.injection_compound == "pyruvate":
            kpa[labels == 6] = config.liver_kpa_s_inv

        ph_map = np.full(config.shape, config.background_ph, dtype=np.float64)
        ph_map[labels == 2] = config.left_cortex_ph
        ph_map[labels == 3] = config.left_medulla_ph
        ph_map[labels == 4] = config.right_cortex_ph
        ph_map[labels == 5] = config.right_medulla_ph

        perfusion_weight = np.zeros(config.shape, dtype=np.float64)
        body = labels > 0
        perfusion_weight[body] = config.background_perfusion_weight
        perfusion_weight[np.isin(labels, (2, 4))] = config.renal_cortex_perfusion_weight
        perfusion_weight[np.isin(labels, (3, 5))] = (
            config.renal_medulla_perfusion_weight
        )
        perfusion_weight[labels == 6] = config.liver_perfusion_weight
        perfusion_weight[labels == 11] = config.brain_perfusion_weight
        perfusion_weight[arterial | venous] = config.vessel_perfusion_weight
        # Urine in the ureters and bladder is reached only through renal
        # excretion, never directly by the vascular bolus.
        perfusion_weight[np.isin(labels, URINARY_LUMEN_LABELS)] = 0.0
        voxel_volume_m3 = float(np.prod(np.asarray(config.fov_m) / config.shape))
        spatial_integral = float(np.sum(perfusion_weight) * voxel_volume_m3)
        if spatial_integral <= 0.0:
            raise ValueError("mouse delivery map has zero volume integral")
        # Integral(delivery_map dV) = 1. Multiplication by the mol/s curve
        # therefore yields mol/m^3/s and conserves the configured injected dose.
        delivery_map = perfusion_weight / spatial_integral

        relative_z = _anatomy_axes(
            config.shape,
            config.fov_m,
            config.mouse_size_m,
            config.mouse_offset_m,
        )[2][None, None, :]
        arrival_time_map = np.full(config.shape, np.nan, dtype=np.float64)
        arrival_variation = 0.08 * (relative_z + 0.25)
        arrival_time_map[body] = (
            config.injection_start_s
            + config.circulation_delay_s
            + np.broadcast_to(arrival_variation, config.shape)[body]
        )
        arrival_time_map[np.isin(labels, (2, 3, 4, 5))] += 0.08
        arrival_time_map[labels == 6] += 0.04
        arrival_time_map[labels == 11] += 0.12
        vascular_support = np.isfinite(vascular_transit) & body
        arrival_time_map[vascular_support] = (
            config.injection_start_s
            + config.circulation_delay_s * vascular_transit[vascular_support]
        )
        arrival_time_map[body] = np.maximum(
            arrival_time_map[body], config.injection_start_s
        )

        pools = _compound_pools(config)
        zeros = np.zeros(config.shape, dtype=np.float64)
        rate_curve = _dose_rate_curve(config, delay_s=0.0)
        polarization_curve = TimeCurve(
            (rate_curve.times_s[0], rate_curve.times_s[-1]),
            (config.injection_polarization, config.injection_polarization),
            interpolation="linear",
            outside="hold",
        )
        metadata = dict(config.metadata)
        metadata.update(
            {
                "phantom_family": "mouse_perfusion",
                "mouse_perfusion_config": config.to_dict(),
                "mouse_tissue_labels": {
                    str(key): value for key, value in MOUSE_TISSUE_LABELS.items()
                },
                "vascular_graph": graph.to_dict(),
                "concentration_units": "mol/m^3 (numerically mM)",
                "injection_amount_mol": config.injection_amount_umol * 1e-6,
                "injection_compound": config.injection_compound,
                "injection_compound_label": INJECTION_COMPOUNDS[
                    config.injection_compound
                ],
                "transport_solver": "voxelwise_arrival_delay",
                "transport_preview": "matches_sequence_source_timing",
                "perfusion_preview_model": "delayed_inflow_with_exponential_clearance",
                "renal_excretion_model": (
                    "preview_only_tubular_ureter_bladder_transfer"
                ),
                "breathing_model": (
                    "preview_only" if config.breathing_enabled else "disabled"
                ),
                "ph_model": (
                    "z_ompd_ground_truth_context"
                    if config.injection_compound == "z_ompd"
                    else "ground_truth_map_only"
                ),
                "alanine_conversion_model": (
                    "liver_kpa_dynamic_third_pool"
                    if config.injection_compound == "pyruvate"
                    else "disabled"
                ),
            }
        )
        initial_maps = {pool.name: zeros.copy() for pool in pools}
        phantom = cls(
            shape=config.shape,
            fov=config.fov_m,
            pools=pools,
            initial_concentration_maps={
                name: values.copy() for name, values in initial_maps.items()
            },
            initial_spin_density_maps={
                name: values.copy() for name, values in initial_maps.items()
            },
            equilibrium_polarization=1.0,
            kpl_map_s_inv=kpl,
            b0_map_ppm=np.zeros(config.shape, dtype=np.float64),
            field_strength=config.field_strength_t,
            nucleus="C13",
            spectral_reference_ppm=config.spectral_reference_ppm,
            spectral_window_center_ppm=(0.5 * (config.spectral_reference_ppm + 183.35)),
            spectral_bandwidth_ppm=config.spectral_bandwidth_ppm,
            spectral_points=config.spectral_points,
            name=config.name,
            pyruvate_inflow=PyruvateInflow(
                rate_curve_s_inv=rate_curve,
                delivery_map=delivery_map,
                polarization_curve=polarization_curve,
                arrival_delay_map_s=(arrival_time_map - config.injection_start_s),
                max_step_s=config.perfusion_timestep_s,
            ),
            conversion_start_s=config.injection_start_s,
            metadata=metadata,
        )
        phantom.config = config
        phantom.anatomy_labels = labels
        phantom.ph_map = ph_map
        phantom.kpa_map_s_inv = kpa
        phantom.arterial_mask = arterial
        phantom.venous_mask = venous
        phantom.perfusion_arrival_time_s = arrival_time_map
        phantom.vascular_graph = graph
        phantom.vascular_transit_fraction = vascular_transit
        phantom.preview_bolus_time_s = (
            config.injection_start_s - 0.1 * config.injection_duration_s
        )
        return phantom

    @classmethod
    def from_dynamic_phantom(
        cls, phantom: DynamicSpectralPhantom
    ) -> "MousePerfusionPhantom":
        config_values = phantom.metadata.get("mouse_perfusion_config")
        if config_values is None:
            raise ValueError("dynamic phantom has no mouse perfusion configuration")
        config = MousePerfusionConfig.from_dict(config_values)
        template = cls.from_config(config)
        result = cls.__new__(cls)
        result.__dict__.update(phantom.__dict__)
        for name in (
            "config",
            "anatomy_labels",
            "ph_map",
            "kpa_map_s_inv",
            "arterial_mask",
            "venous_mask",
            "perfusion_arrival_time_s",
            "vascular_graph",
            "vascular_transit_fraction",
            "preview_bolus_time_s",
        ):
            setattr(result, name, getattr(template, name))
        return result

    @classmethod
    def load(cls, filename) -> "MousePerfusionPhantom":
        return cls.from_dynamic_phantom(DynamicSpectralPhantom.load(filename))

    @property
    def tissue_label_names(self) -> Dict[int, str]:
        return dict(MOUSE_TISSUE_LABELS)

    def bolus_rate_map(self, time_s: float) -> np.ndarray:
        """Return the spatially delayed preview source in mol/m^3/s.

        This uses the same curve, delay map, and delivery map as the sequence
        solver and is therefore suitable for UI previews of the injected source.
        """

        base_curve = _dose_rate_curve(self.config, delay_s=0.0)
        relative_delay = self.perfusion_arrival_time_s - self.config.injection_start_s
        result = np.zeros(self.shape, dtype=np.float64)
        active = np.isfinite(relative_delay) & (
            np.asarray(self.pyruvate_inflow.delivery_map) > 0
        )
        flat_delays = relative_delay[active]
        result[active] = (
            np.asarray(
                [base_curve.value_at(float(time_s) - value) for value in flat_delays]
            )
            * np.asarray(self.pyruvate_inflow.delivery_map)[active]
        )
        return result

    @property
    def injection_compound_label(self) -> str:
        return INJECTION_COMPOUNDS[self.config.injection_compound]

    def preview_time_bounds_s(self) -> Tuple[float, float]:
        """Return a useful time interval containing arrival and wash-out."""

        finite = self.perfusion_arrival_time_s[
            np.isfinite(self.perfusion_arrival_time_s)
        ]
        earliest = (
            self.config.injection_start_s if finite.size == 0 else float(np.min(finite))
        )
        latest = (
            self.config.injection_start_s + self.config.injection_duration_s
            if finite.size == 0
            else float(np.max(finite)) + self.config.injection_duration_s
        )
        end = latest + 4.0 / self.config.tissue_clearance_s_inv
        if self._renal_excretion_active:
            # Show the excreted bolus until the bladder has nearly plateaued.
            end = max(
                end,
                latest
                + 1.5 * self.config.renal_tubular_transit_s
                + self.config.ureter_transit_s
                + 4.0 / self._renal_loss_s_inv,
            )
        return (earliest - 0.15 * self.config.injection_duration_s, end)

    @property
    def _renal_excretion_active(self) -> bool:
        return self.config.renal_excretion_s_inv > 0.0 and bool(
            np.any(np.isin(self.anatomy_labels, (2, 3, 4, 5)))
        )

    @property
    def _renal_loss_s_inv(self) -> float:
        return self.config.tissue_clearance_s_inv + self.config.renal_excretion_s_inv

    def _local_loss_map_s_inv(self) -> np.ndarray:
        loss = np.full(self.shape, self.config.tissue_clearance_s_inv, dtype=float)
        loss[np.isin(self.anatomy_labels, (2, 3, 4, 5))] = self._renal_loss_s_inv
        return loss

    def _response_lookup(self, loss_s_inv: float):
        """Return the source convolved with ``exp(-loss t)`` on a fine grid."""

        cache = self.__dict__.setdefault("_perfusion_response_lookups", {})
        key = float(loss_s_inv)
        if key in cache:
            return cache[key]
        curve = self.pyruvate_inflow.rate_curve_s_inv
        start = float(curve.times_s[0])
        end = float(curve.times_s[-1]) + 6.0 / key
        step = min(
            self.config.perfusion_timestep_s,
            self.config.injection_duration_s / 128.0,
        )
        count = max(2, int(np.ceil((end - start) / step)) + 1)
        lookup_t = np.linspace(start, end, count)
        rate = np.interp(
            lookup_t,
            np.asarray(curve.times_s, dtype=float),
            np.asarray(curve.values, dtype=float),
            left=0.0,
            right=0.0,
        )
        response = np.zeros_like(lookup_t)
        for index in range(1, lookup_t.size):
            dt = lookup_t[index] - lookup_t[index - 1]
            decay = np.exp(-key * dt)
            response[index] = (
                response[index - 1] * decay
                + 0.5 * (rate[index - 1] * decay + rate[index]) * dt
            )
        cache[key] = (lookup_t, response)
        return cache[key]

    def _urinary_grid_s(self) -> np.ndarray:
        curve = self.pyruvate_inflow.rate_curve_s_inv
        start = float(curve.times_s[0])
        end = self.preview_time_bounds_s()[1] + 2.0 / self._renal_loss_s_inv
        step = min(0.1, self.config.injection_duration_s / 16.0)
        count = min(20001, max(2, int(np.ceil((end - start) / step)) + 1))
        return np.linspace(start, end, count)

    def _renal_groups(self, side: str):
        """Group one kidney's voxels by identical source delay and kinetics."""

        delays = self.pyruvate_inflow.arrival_delay_map_s
        delivery = np.asarray(self.pyruvate_inflow.delivery_map)
        mask = (
            np.isin(self.anatomy_labels, RENAL_LABELS[side])
            & np.isfinite(delays)
            & (delivery > 0)
        )
        keys = np.column_stack(
            (
                np.round(delays[mask], 9),
                self.kpl_map_s_inv[mask] if len(self.pools) > 1 else 0 * delays[mask],
                self.kpa_map_s_inv[mask],
            )
        )
        if keys.shape[0] == 0:
            return keys, np.zeros(0)
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        weights = np.bincount(inverse.ravel(), weights=delivery[mask])
        return unique, weights

    def _urinary_lookup(self):
        """Return excreted injected compound in tubules, ureters and bladder.

        Kidney tissue loses tracer at ``renal_excretion_s_inv`` in addition to
        tissue clearance. The excreted amount spends a uniformly distributed
        tubular transit time (0.5–1.5 × the configured mean) in the medulla,
        then a plug-flow ureter transit, and finally accumulates in the
        bladder. Amounts are in mol and conserve the excreted dose.
        """

        cached = self.__dict__.get("_urinary_lookup_cache")
        if cached is not None:
            return cached
        times = self._urinary_grid_s()
        lookup_t, response = self._response_lookup(self._renal_loss_s_inv)
        voxel_volume = self.voxel_volume_m3
        result = {"times_s": times, "bladder": np.zeros_like(times)}
        for side in RENAL_LABELS:
            groups, weights = self._renal_groups(side)
            flux = np.zeros_like(times)
            for delay, weight in zip(groups[:, 0], weights):
                flux += weight * np.interp(
                    times - delay, lookup_t, response, left=0.0, right=0.0
                )
            flux *= self.config.renal_excretion_s_inv * voxel_volume
            tubular, ureter, bladder = _urinary_transport(
                times,
                _cumulative_trapezoid(flux, times),
                self.config.renal_tubular_transit_s,
                self.config.ureter_transit_s,
            )
            result[f"{side}_tubular"] = tubular
            result[f"{side}_ureter"] = ureter
            result["bladder"] += bladder
        self._urinary_lookup_cache = result
        return result

    def _urinary_compartments(self):
        """Return ``(name, mask)`` pairs that display excreted amounts."""

        labels = self.anatomy_labels
        compartments = []
        for side, (cortex, medulla) in RENAL_LABELS.items():
            tubules = labels == medulla
            if not np.any(tubules):
                tubules = labels == cortex
            compartments.append((f"{side}_tubular", tubules))
            compartments.append((f"{side}_ureter", labels == URETER_LABELS[side]))
        compartments.append(("bladder", labels == BLADDER_LABEL))
        return compartments

    def _add_urinary_concentration(self, result, amounts_at):
        for name, mask in self._urinary_compartments():
            count = int(np.count_nonzero(mask))
            if count:
                result[mask] += amounts_at(name) / (count * self.voxel_volume_m3)

    def injected_concentration_map(self, time_s: float) -> np.ndarray:
        """Return locally present injected compound in mol/m^3 (numerically mM).

        The result convolves the exact configured source curve with a local
        mono-exponential clearance and, in the kidneys, renal excretion into
        the tubules, ureters and bladder.  It is an authoring/visualization
        model, not an additional advection compartment in the Bloch solver.
        """

        delays = self.perfusion_arrival_time_s - self.config.injection_start_s
        delivery = np.asarray(self.pyruvate_inflow.delivery_map)
        active = np.isfinite(delays) & (delivery > 0)
        result = np.zeros(self.shape, dtype=np.float64)
        if not np.any(active):
            return result
        loss = self._local_loss_map_s_inv()
        for value in np.unique(loss[active]):
            selected = active & (loss == value)
            lookup_t, response = self._response_lookup(value)
            result[selected] = (
                np.interp(
                    float(time_s) - delays[selected],
                    lookup_t,
                    response,
                    left=0.0,
                    right=0.0,
                )
                * delivery[selected]
            )
        if self._renal_excretion_active:
            urinary = self._urinary_lookup()
            self._add_urinary_concentration(
                result,
                lambda name: float(
                    np.interp(
                        float(time_s),
                        urinary["times_s"],
                        urinary[name],
                        left=0.0,
                        right=float(urinary[name][-1]),
                    )
                ),
            )
        return result

    def organ_bolus_curves(self, times_s) -> Dict[str, np.ndarray]:
        """Return mean injected concentration curves for major organ regions."""

        times = np.asarray(times_s, dtype=float)
        regions = {
            "Brain": self.anatomy_labels == 11,
            "Liver": self.anatomy_labels == 6,
            "Kidneys": np.isin(self.anatomy_labels, (2, 3, 4, 5)),
        }
        if np.any(self.anatomy_labels == BLADDER_LABEL):
            regions["Bladder"] = self.anatomy_labels == BLADDER_LABEL
        curves = {name: np.zeros(times.shape, dtype=float) for name in regions}
        for index, time_s in np.ndenumerate(times):
            frame = self.injected_concentration_map(float(time_s))
            for name, mask in regions.items():
                if np.any(mask):
                    curves[name][index] = float(np.mean(frame[mask]))
        return curves

    def preview_pool_curves_at(self, index, times_s) -> Dict[str, np.ndarray]:
        """Return local perfusion/conversion preview curves for one voxel.

        This compact authoring preview applies the configured delayed source,
        tissue clearance, renal kPL, liver kPA and renal excretion into the
        urinary tract. It intentionally does not include RF pulses or readout
        effects from a sequence simulation.
        """

        index = tuple(int(value) for value in index)
        if len(index) != 3 or any(
            value < 0 or value >= extent for value, extent in zip(index, self.shape)
        ):
            raise IndexError("mouse preview voxel is outside the phantom")
        times = np.asarray(times_s, dtype=float)
        if times.ndim != 1 or np.any(~np.isfinite(times)):
            raise ValueError("preview times must be a finite one-dimensional array")
        if times.size > 1 and np.any(np.diff(times) < 0):
            raise ValueError("preview times must be monotonically increasing")
        names = [pool.name for pool in self.pools]
        curves = {name: np.zeros(times.shape, dtype=np.float64) for name in names}
        if times.size == 0 or self.anatomy_labels[index] == 0:
            return curves

        delay = float(self.pyruvate_inflow.arrival_delay_map_s[index])
        delivery = float(self.pyruvate_inflow.delivery_map[index])
        if np.isfinite(delay) and delivery > 0.0:
            local = self._pool_kinetics(
                times,
                np.asarray([delay]),
                np.asarray([delivery]),
                np.asarray([self.kpl_map_s_inv[index] if len(self.pools) > 1 else 0]),
                np.asarray([self.kpa_map_s_inv[index] if len(self.pools) > 2 else 0]),
                float(self._local_loss_map_s_inv()[index]),
            )
            for name in names:
                curves[name] += local[name][0]

        if self._renal_excretion_active:
            compartment = next(
                (
                    (name, mask)
                    for name, mask in self._urinary_compartments()
                    if mask[index]
                ),
                None,
            )
            if compartment is not None:
                name, mask = compartment
                volume = np.count_nonzero(mask) * self.voxel_volume_m3
                pools = self._urinary_pool_lookup()
                for pool in names:
                    curves[pool] += (
                        np.interp(
                            times,
                            pools["times_s"],
                            pools[name][pool],
                            left=0.0,
                            right=float(pools[name][pool][-1]),
                        )
                        / volume
                    )
        return curves

    def _pool_kinetics(self, times, delays, deliveries, kpl, kpa, loss_s_inv):
        """Integrate delayed source, conversion and loss for voxel groups."""

        source_curve = self.pyruvate_inflow.rate_curve_s_inv
        source = deliveries[:, None] * source_curve.values_at(
            times[None, :] - delays[:, None]
        ).reshape(delays.size, times.size)
        names = [pool.name for pool in self.pools]
        curves = {name: np.zeros(source.shape, dtype=np.float64) for name in names}
        precursor = curves[names[0]]
        lactate = curves.get("Lactate")
        alanine = curves.get("Alanine")
        for sample in range(1, times.size):
            dt = float(times[sample] - times[sample - 1])
            precursor_decay = np.exp(-(loss_s_inv + kpl + kpa) * dt)
            precursor[:, sample] = (
                precursor[:, sample - 1] * precursor_decay
                + 0.5
                * (source[:, sample - 1] * precursor_decay + source[:, sample])
                * dt
            )
            product_decay = np.exp(-loss_s_inv * dt)
            for product, rate in ((lactate, kpl), (alanine, kpa)):
                if product is not None:
                    product[:, sample] = (
                        product[:, sample - 1] * product_decay
                        + 0.5
                        * rate
                        * (
                            precursor[:, sample - 1] * product_decay
                            + precursor[:, sample]
                        )
                        * dt
                    )
        return curves

    def _urinary_pool_lookup(self):
        """Return excreted amounts in mol per pool and urinary compartment."""

        cached = self.__dict__.get("_urinary_pool_lookup_cache")
        if cached is not None:
            return cached
        times = self._urinary_grid_s()
        names = [pool.name for pool in self.pools]
        result = {
            "times_s": times,
            "bladder": {name: np.zeros_like(times) for name in names},
        }
        scale = self.config.renal_excretion_s_inv * self.voxel_volume_m3
        for side in RENAL_LABELS:
            groups, weights = self._renal_groups(side)
            result[f"{side}_tubular"] = {}
            result[f"{side}_ureter"] = {}
            kinetics = (
                self._pool_kinetics(
                    times,
                    groups[:, 0],
                    weights,
                    groups[:, 1],
                    groups[:, 2],
                    self._renal_loss_s_inv,
                )
                if weights.size
                else None
            )
            for name in names:
                flux = (
                    np.zeros_like(times)
                    if kinetics is None
                    else scale * np.sum(kinetics[name], axis=0)
                )
                tubular, ureter, bladder = _urinary_transport(
                    times,
                    _cumulative_trapezoid(flux, times),
                    self.config.renal_tubular_transit_s,
                    self.config.ureter_transit_s,
                )
                result[f"{side}_tubular"][name] = tubular
                result[f"{side}_ureter"][name] = ureter
                result["bladder"][name] += bladder
        self._urinary_pool_lookup_cache = result
        return result

    def displacement_map_m(self, time_s: float) -> np.ndarray:
        """Return the configured smooth breathing preview displacement field."""

        result = np.zeros(self.shape + (3,), dtype=np.float64)
        if not self.config.breathing_enabled:
            return result
        phase = (
            2.0 * np.pi * self.config.breathing_rate_hz * float(time_s)
            + self.config.breathing_phase_rad
        )
        z = _anatomy_axes(
            self.shape,
            self.config.fov_m,
            self.config.mouse_size_m,
            self.config.mouse_offset_m,
        )[2][None, None, :]
        envelope = np.exp(-(((z - 0.08) / 0.55) ** 2))
        envelope = np.broadcast_to(envelope, self.shape) * (self.anatomy_labels > 0)
        amplitude_m = self.config.breathing_amplitude_mm * 1e-3
        result[..., 2] = amplitude_m * np.sin(phase) * envelope
        return result


def is_mouse_perfusion_phantom(phantom) -> bool:
    return isinstance(phantom, MousePerfusionPhantom) or bool(
        getattr(phantom, "metadata", {}).get("phantom_family") == "mouse_perfusion"
    )
