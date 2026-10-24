"""Keep authored absolute polarization when the workspace changes B0 or nucleus."""

from dataclasses import replace

from .units import thermal_polarization


def rebase_phantom_polarization(phantom, *, field_strength_t, nucleus):
    """Rebase thermal-relative states of a temperature-aware authored phantom.

    Concentrations, geometry, relaxation and dose are unchanged, except that
    mouse phantoms are rebuilt when they gain or carry 1H tissue water. Older objects
    without an explicit temperature keep their historical relative convention.
    The caller continues to apply the new field/nucleus to the phantom itself.
    """
    temperature = (phantom.metadata or {}).get("temperature_k")
    if temperature is None:
        return
    old_pth = thermal_polarization(phantom.nucleus, phantom.field_strength, temperature)
    new_pth = thermal_polarization(nucleus, field_strength_t, temperature)
    factor = old_pth / new_pth
    if factor == 1.0:
        return
    if hasattr(phantom, "initial_mz_maps"):
        phantom.initial_mz_maps = {
            name: values * factor for name, values in phantom.initial_mz_maps.items()
        }
    else:
        phantom.initial_concentration_maps = {
            name: values * factor
            for name, values in phantom.initial_concentration_maps.items()
        }
        inflow = phantom.pyruvate_inflow
        if inflow is not None:
            if inflow.polarization_curve is None:
                # A legacy direct-Mz source carries magnetization in the rate.
                curve = inflow.rate_curve_s_inv
                inflow.rate_curve_s_inv = replace(
                    curve, values=tuple(value * factor for value in curve.values)
                )
            else:
                curve = inflow.polarization_curve
                inflow.polarization_curve = replace(
                    curve, values=tuple(value * factor for value in curve.values)
                )
    design = phantom.metadata.get("phantom_design")
    if isinstance(design, dict):
        for shape in design.get("shapes", ()):
            shape["initial_mz"] *= factor
            for peak in shape.get("peaks", ()):
                if peak.get("initial_polarization") is not None:
                    peak["initial_polarization"] *= factor
        curve = design.get("pyruvate_inflow_polarization_curve")
        if curve is None:
            curve = design.get("pyruvate_inflow_curve")
        if curve is not None:
            curve["values"] = tuple(value * factor for value in curve["values"])
    mouse = phantom.metadata.get("mouse_perfusion_config")
    if isinstance(mouse, dict):
        mouse["injection_polarization"] *= factor
        mouse["field_strength_t"] = float(field_strength_t)
        mouse["nucleus"] = str(nucleus)
    config = getattr(phantom, "config", None)
    if config is not None:
        config.injection_polarization *= factor
        config.field_strength_t = float(field_strength_t)
        config.nucleus = str(nucleus)
    _rebuild_mouse_for_nucleus(phantom)


def _rebuild_mouse_for_nucleus(phantom):
    """Rebuild a mouse phantom in place when it shows tissue water.

    A nucleus change swaps 13C compound and 1H water pools. Thermal water
    magnetization must not follow the relative rebase above, so a water
    phantom is always rebuilt from its configuration.
    """
    from .mouse_phantom import (
        MousePerfusionConfig,
        MousePerfusionPhantom,
        _has_tissue_water,
        injection_compounds,
        uses_tissue_water,
    )

    mouse = phantom.metadata.get("mouse_perfusion_config")
    if not isinstance(mouse, dict):
        return
    water = uses_tissue_water(mouse["nucleus"])
    if not water and not _has_tissue_water(phantom):
        return
    if mouse["injection_compound"] not in injection_compounds(mouse["nucleus"]):
        # A 13C compound is invisible to 1H, so water keeps no injection
        # rather than silently starting DCE; 13C cannot image the
        # injection-free water mouse and falls back to its first compound.
        mouse["injection_compound"] = (
            "none" if water else next(iter(injection_compounds(mouse["nucleus"])))
        )
    rebuilt = MousePerfusionPhantom.from_config(MousePerfusionConfig.from_dict(mouse))
    phantom.__class__ = MousePerfusionPhantom
    phantom.__dict__.clear()
    phantom.__dict__.update(rebuilt.__dict__)
