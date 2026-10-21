"""Scalar phantom authoring parameters that a parameter loop can vary.

Phantoms built by the mouse perfusion designer or the spectral phantom
designer keep their authoring configuration in ``phantom.metadata``. A loop
changes one scalar of that configuration and rebuilds the phantom, so derived
maps (perfusion curves, kinetics, relaxation) stay consistent with the value.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Dict, List, Tuple

import numpy as np


MOUSE_PREFIX = "mouse."
DESIGN_PREFIX = "design."


@dataclass(frozen=True)
class PhantomLoopParameter:
    """One scalar phantom parameter with its current value and editing range."""

    key: str
    label: str
    unit: str
    value: float
    minimum: float = 0.0
    maximum: float = 1e6
    decimals: int = 4
    integer: bool = False


# Label, unit, minimum, maximum and decimals for each loopable mouse field.
_MOUSE_FIELDS: Dict[str, Tuple[str, str, float, float, int]] = {
    "injection_concentration_mm": ("Injection concentration", "mM", 0, 1e4, 3),
    "injection_volume_ul": ("Injection volume", "µl", 0, 1e5, 3),
    "injection_polarization": ("Injection polarization", "× thermal", 0, 1e9, 3),
    "injection_start_s": ("Injection start", "s", -1e4, 1e4, 4),
    "injection_duration_s": ("Injection duration", "s", 0, 1e4, 4),
    "circulation_delay_s": ("Circulation delay", "s", 0, 1e4, 4),
    "perfusion_timestep_s": ("Inflow time step", "s", 0, 100, 5),
    "bolus_alpha": ("Bolus shape alpha", "", 0, 1e3, 4),
    "bolus_beta": ("Bolus shape beta", "", 0, 1e3, 4),
    "background_perfusion_weight": ("Background perfusion weight", "", 0, 1e3, 4),
    "renal_cortex_perfusion_weight": ("Renal cortex perfusion weight", "", 0, 1e3, 4),
    "renal_medulla_perfusion_weight": (
        "Renal medulla perfusion weight",
        "",
        0,
        1e3,
        4,
    ),
    "liver_perfusion_weight": ("Liver perfusion weight", "", 0, 1e3, 4),
    "heart_perfusion_weight": ("Heart perfusion weight", "", 0, 1e3, 4),
    "brain_perfusion_weight": ("Brain perfusion weight", "", 0, 1e3, 4),
    "vessel_perfusion_weight": ("Vessel perfusion weight", "", 0, 1e3, 4),
    "tissue_clearance_s_inv": ("Tissue clearance", "1/s", 0, 100, 5),
    "renal_excretion_s_inv": ("Renal excretion", "1/s", 0, 100, 5),
    "renal_tubular_transit_s": ("Tubular transit", "s", 0, 1e4, 3),
    "ureter_transit_s": ("Ureter transit", "s", 0, 1e4, 3),
    "left_cortex_kpl_s_inv": ("Left cortex kPL", "1/s", 0, 100, 5),
    "left_medulla_kpl_s_inv": ("Left medulla kPL", "1/s", 0, 100, 5),
    "right_cortex_kpl_s_inv": ("Right cortex kPL", "1/s", 0, 100, 5),
    "right_medulla_kpl_s_inv": ("Right medulla kPL", "1/s", 0, 100, 5),
    "liver_kpa_s_inv": ("Liver kPA", "1/s", 0, 100, 5),
    "heart_kpl_s_inv": ("Heart kPL", "1/s", 0, 100, 5),
    "heart_kpa_s_inv": ("Heart kPA", "1/s", 0, 100, 5),
    "brain_kpl_s_inv": ("Brain kPL", "1/s", 0, 100, 5),
    "brain_kpa_s_inv": ("Brain kPA", "1/s", 0, 100, 5),
    "left_cortex_ph": ("Left cortex pH", "", 0, 14, 3),
    "left_medulla_ph": ("Left medulla pH", "", 0, 14, 3),
    "right_cortex_ph": ("Right cortex pH", "", 0, 14, 3),
    "right_medulla_ph": ("Right medulla pH", "", 0, 14, 3),
    "background_ph": ("Background pH", "", 0, 14, 3),
    "blood_ph": ("Blood pH", "", 0, 14, 3),
    "urine_ph": ("Urine pH", "", 0, 14, 3),
    "breathing_rate_hz": ("Breathing rate", "Hz", 0, 100, 4),
    "breathing_amplitude_mm": ("Breathing displacement", "mm", 0, 100, 3),
    "breathing_phase_rad": ("Breathing phase at t=0", "rad", -100, 100, 4),
    "injected_t1_s": ("Injected compound T1", "s", 0, 1e4, 4),
    "injected_t2_s": ("Injected compound T2", "s", 0, 1e4, 5),
    "injected_t2_star_s": ("Injected compound T2*", "s", 0, 1e4, 5),
    "metabolite_t1_s": ("Metabolite T1", "s", 0, 1e4, 4),
    "metabolite_t2_s": ("Metabolite T2", "s", 0, 1e4, 5),
    "metabolite_t2_star_s": ("Metabolite T2*", "s", 0, 1e4, 5),
}

_DESIGN_FIELDS: Dict[str, Tuple[str, str, float, float, int]] = {
    "b0_inhomogeneity_ppm": ("B0 inhomogeneity amplitude", "ppm", -1e4, 1e4, 4),
}
_DYNAMIC_DESIGN_FIELDS: Dict[str, Tuple[str, str, float, float, int]] = {
    "default_kpl_s_inv": ("Default kPL", "1/s", 0, 100, 5),
    "conversion_start_s": ("Conversion start", "s", -1e4, 1e4, 4),
    "kinetics_time_offset_s": ("Kinetics time offset", "s", -1e4, 1e4, 4),
}
_SHAPE_FIELDS: Dict[str, Tuple[str, str, float, float, int]] = {
    "t1_s": ("T1", "s", 0, 1e4, 4),
    "initial_mz": ("Initial Mz", "", 0, 1e9, 4),
    "b0_ppm": ("B0 offset", "ppm", -1e4, 1e4, 4),
}
_PEAK_FIELDS: Dict[str, Tuple[str, str, float, float, int]] = {
    "amplitude": ("amplitude", "", 0, 1e9, 4),
    "frequency_ppm": ("frequency", "ppm", -1e4, 1e4, 4),
    "t2_star_s": ("T2*", "s", 0, 1e4, 5),
    "t1_s": ("T1", "s", 0, 1e4, 4),
    "initial_polarization": ("initial polarization", "", 0, 1e9, 4),
}


def phantom_loop_parameters(phantom) -> List[PhantomLoopParameter]:
    """Return the loopable authoring parameters stored with ``phantom``."""
    if phantom is None:
        return []
    metadata = getattr(phantom, "metadata", None) or {}
    if isinstance(metadata.get("mouse_perfusion_config"), dict):
        return _mouse_parameters(_mouse_config(phantom))
    if metadata.get("phantom_design") is not None:
        return _design_parameters(_design(phantom))
    return []


def phantom_with_parameter(phantom, key: str, value: float):
    """Rebuild ``phantom`` from its authoring configuration with one change."""
    key = str(key)
    if key.startswith(MOUSE_PREFIX):
        name = key[len(MOUSE_PREFIX) :]
        if name not in _MOUSE_FIELDS:
            raise KeyError(f"unknown mouse phantom parameter {name!r}")
        config = replace(_mouse_config(phantom), **{name: float(value)})
        rebuilt = config.build()
    elif key.startswith(DESIGN_PREFIX):
        design = _design(phantom)
        _set_design_value(design, key[len(DESIGN_PREFIX) :], float(value))
        rebuilt = design.build()
    else:
        raise KeyError(f"unknown phantom loop parameter {key!r}")
    # The Phantom tab stores the chosen sequence reference with the phantom.
    reference = getattr(phantom, "metadata", {}).get("sequence_reference_ppm")
    if reference is not None:
        rebuilt.metadata["sequence_reference_ppm"] = reference
    return rebuilt


def _mouse_config(phantom):
    from .mouse_phantom import MousePerfusionConfig

    return MousePerfusionConfig.from_dict(phantom.metadata["mouse_perfusion_config"])


def _mouse_parameters(config) -> List[PhantomLoopParameter]:
    names = {item.name for item in fields(config)}
    return [
        PhantomLoopParameter(
            key=f"{MOUSE_PREFIX}{name}",
            label=label,
            unit=unit,
            value=float(getattr(config, name)),
            minimum=float(minimum),
            maximum=float(maximum),
            decimals=int(decimals),
        )
        for name, (label, unit, minimum, maximum, decimals) in _MOUSE_FIELDS.items()
        if name in names
    ]


def _design(phantom):
    from .phantom_design import PhantomDesign

    return PhantomDesign.from_phantom(phantom)


def _parameter(key, spec, value):
    label, unit, minimum, maximum, decimals = spec
    return PhantomLoopParameter(
        key=f"{DESIGN_PREFIX}{key}",
        label=label,
        unit=unit,
        value=float(value),
        minimum=float(minimum),
        maximum=float(maximum),
        decimals=int(decimals),
    )


def _design_parameters(design) -> List[PhantomLoopParameter]:
    parameters = [
        _parameter(name, spec, getattr(design, name))
        for name, spec in _DESIGN_FIELDS.items()
    ]
    if design.dynamic_enabled:
        parameters.extend(
            _parameter(name, spec, getattr(design, name))
            for name, spec in _DYNAMIC_DESIGN_FIELDS.items()
        )
    for shape_index, shape in enumerate(design.shapes):
        for name, (label, *rest) in _SHAPE_FIELDS.items():
            parameters.append(
                _parameter(
                    f"shape[{shape_index}].{name}",
                    (f"{shape.name}: {label}", *rest),
                    getattr(shape, name),
                )
            )
        if design.dynamic_enabled:
            parameters.append(
                _parameter(
                    f"shape[{shape_index}].kpl_s_inv",
                    (f"{shape.name}: kPL", "1/s", 0, 100, 5),
                    shape.effective_kpl_s_inv(design.default_kpl_s_inv),
                )
            )
        for peak_index, peak in enumerate(shape.peaks):
            for name, (label, *rest) in _PEAK_FIELDS.items():
                value = getattr(peak, name)
                if name == "t1_s":
                    value = peak.effective_t1_s(shape.t1_s)
                elif name == "initial_polarization":
                    value = peak.effective_initial_polarization(shape.initial_mz)
                parameters.append(
                    _parameter(
                        f"shape[{shape_index}].peak[{peak_index}].{name}",
                        (f"{shape.name}: {peak.name} {label}", *rest),
                        value,
                    )
                )
    return parameters


def _set_design_value(design, path: str, value: float) -> None:
    target = design
    parts = path.split(".")
    for part in parts[:-1]:
        name, _, index = part.partition("[")
        if not index.endswith("]") or name not in {"shape", "peak"}:
            raise KeyError(f"unknown phantom design parameter {path!r}")
        items = design.shapes if name == "shape" else target.peaks
        target = items[int(index[:-1])]
    name = parts[-1]
    if not hasattr(target, name) or isinstance(getattr(target, name), (tuple, list)):
        raise KeyError(f"unknown phantom design parameter {path!r}")
    if not np.isfinite(value):
        raise ValueError(f"{path} must be finite")
    setattr(target, name, float(value))
