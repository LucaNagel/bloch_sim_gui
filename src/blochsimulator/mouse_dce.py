"""Gadolinium DCE kinetics of the 1H mouse perfusion phantom.

A three-compartment whole-body model (central plasma, systemic plasma,
interstitial extracellular fluid, plus renal excretion by glomerular
filtration) provides the arterial input function. Every tissue label follows
the extended Tofts model driven by its delayed arterial plasma; blood voxels
carry the whole-blood concentration and the urinary tract receives the
filtered agent through the tubular/ureter/bladder transport of the 13C
excretion model. Fast water exchange turns concentrations into relaxation
rate increments ``r1 * C`` and ``r2 * C``.

Units: amounts in µmol, volumes in ml, concentrations in mM (µmol/ml).
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.linalg import expm

from .dynamic_phantom import ContrastRelaxation

# Relaxivities in plasma at 7 T and 37 °C, in 1/(mM s).
CONTRAST_AGENTS = {
    "gd_dota": ("Gd-DOTA (gadoterate)", 3.0, 4.0),
    "gadobutrol": ("Gadobutrol", 4.5, 5.5),
    "custom": ("Custom relaxivity", None, None),
}

# Extended Tofts parameters per tissue: (Ktrans in 1/min, ve, vp).
DEFAULT_TOFTS_PARAMETERS = {
    "soft_tissue": (0.05, 0.10, 0.02),
    "kidney_cortex": (0.50, 0.20, 0.20),
    "kidney_medulla": (0.30, 0.25, 0.10),
    "liver": (1.00, 0.25, 0.12),
    "lung": (0.10, 0.20, 0.15),
    "heart": (2.00, 0.25, 0.10),
    # Intact blood-brain barrier: intravascular only.
    "brain": (0.0, 0.0, 0.03),
}
TOFTS_TISSUE_LABELS = {
    "soft_tissue": (1,),
    "kidney_cortex": (2, 4),
    "kidney_medulla": (3, 5),
    "liver": (6,),
    "lung": (7,),
    "heart": (8,),
    "brain": (11,),
}
TOFTS_TISSUE_NAMES = {
    "soft_tissue": "Soft tissue / muscle",
    "kidney_cortex": "Kidney cortex",
    "kidney_medulla": "Kidney medulla",
    "liver": "Liver",
    "lung": "Lung",
    "heart": "Heart",
    "brain": "Brain",
}

# Whole-body volumes per gram body weight: blood 72 µl/g, interstitial
# extracellular fluid 200 µl/g. A quarter of the plasma is the central
# (cardiopulmonary) compartment that shapes the first-pass peak.
BLOOD_VOLUME_ML_PER_G = 0.072
INTERSTITIAL_VOLUME_ML_PER_G = 0.20
CENTRAL_PLASMA_FRACTION = 0.25
KINETICS_STEP_S = 0.1

ARTERIAL_COMPARTMENT = 8
VENOUS_COMPARTMENT = 9
# Compartment k holds anatomy label k + 1 (labels 1..14).
N_COMPARTMENTS = 14


def tofts_parameters(values) -> Dict[str, Tuple[float, float, float]]:
    """Return validated Tofts parameters, filling missing tissues."""
    result = dict(DEFAULT_TOFTS_PARAMETERS)
    for key, entry in dict(values or {}).items():
        if key not in DEFAULT_TOFTS_PARAMETERS:
            raise ValueError(f"unknown Tofts tissue {key!r}")
        ktrans, ve, vp = (float(value) for value in entry)
        result[key] = (ktrans, ve, vp)
    for key, (ktrans, ve, vp) in result.items():
        if not np.all(np.isfinite((ktrans, ve, vp))) or min(ktrans, ve, vp) < 0:
            raise ValueError(f"{key} Tofts parameters must be finite and >= 0")
        if ve + vp > 1.0:
            raise ValueError(f"{key} ve + vp must not exceed 1")
        if ktrans > 0 and ve <= 0:
            raise ValueError(f"{key} needs ve > 0 when Ktrans > 0")
    return result


def _first_order_response(times, inputs, rates):
    """Integrate ``y' = -k y + u`` exactly for piecewise-linear ``u``.

    ``inputs`` has shape (times, columns) and ``rates`` shape (columns,).
    """
    result = np.zeros_like(inputs)
    rates = np.asarray(rates, dtype=float)
    for index, step in enumerate(np.diff(times), start=1):
        x = rates * step
        decay = np.exp(-x)
        safe = np.where(rates > 0, rates, 1.0)
        f0 = np.where(rates > 0, -np.expm1(-x) / safe, step)
        f1 = np.where(rates > 0, (x + np.expm1(-x)) / safe**2, 0.5 * step**2)
        slope = (inputs[index] - inputs[index - 1]) / step
        result[index] = result[index - 1] * decay + inputs[index - 1] * f0 + slope * f1
    return result


@dataclass
class MouseDCEKinetics:
    """Ground-truth contrast agent kinetics of one mouse phantom."""

    times_s: np.ndarray
    # Whole-body amounts (µmol): central, systemic, interstitial, excreted.
    body_amounts_umol: np.ndarray
    arterial_plasma_mm: np.ndarray
    venous_plasma_mm: np.ndarray
    urinary_amounts_umol: Dict[str, np.ndarray]
    compartment_map: np.ndarray
    compartment_names: Tuple[str, ...]
    concentration_mm: np.ndarray
    r1_mm_s_inv: float
    r2_mm_s_inv: float
    dose_umol: float

    def concentrations_at(self, time_s) -> np.ndarray:
        """Return compartment concentrations (times, compartments) in mM."""
        times = np.atleast_1d(np.asarray(time_s, dtype=float))
        return np.stack(
            [
                np.interp(times, self.times_s, self.concentration_mm[:, column])
                for column in range(self.concentration_mm.shape[1])
            ],
            axis=1,
        )

    def concentration_map(self, time_s: float) -> np.ndarray:
        values = self.concentrations_at(float(time_s))[0]
        result = np.zeros(self.compartment_map.shape, dtype=np.float64)
        inside = self.compartment_map >= 0
        result[inside] = values[self.compartment_map[inside]]
        return result

    def relaxation(self, step_s: float) -> ContrastRelaxation:
        return ContrastRelaxation(
            compartment_map=self.compartment_map,
            times_s=self.times_s,
            delta_r1_s_inv=self.r1_mm_s_inv * self.concentration_mm,
            delta_r2_s_inv=self.r2_mm_s_inv * self.concentration_mm,
            step_s=step_s,
            compartment_names=self.compartment_names,
        )


def _body_kinetics(config, times, dose_rate_curve):
    """Solve the whole-body plasma/ECF/excretion model on ``times``."""
    hematocrit = config.hematocrit
    plasma_ml = config.body_mass_g * BLOOD_VOLUME_ML_PER_G * (1.0 - hematocrit)
    central_ml = CENTRAL_PLASMA_FRACTION * plasma_ml
    systemic_ml = plasma_ml - central_ml
    interstitial_ml = config.body_mass_g * INTERSTITIAL_VOLUME_ML_PER_G
    plasma_flow = config.cardiac_output_ml_min / 60.0 * (1.0 - hematocrit)
    gfr = config.gfr_ml_min / 60.0
    ps = config.capillary_ps_ml_min / 60.0
    # Amount states [central, systemic, interstitial, excreted]; every column
    # sums to zero, so the propagator conserves the total amount.
    qc, qs, pe = (
        plasma_flow / central_ml,
        plasma_flow / systemic_ml,
        ps / interstitial_ml,
    )
    ps_s = ps / systemic_ml
    gc = gfr / central_ml
    matrix = np.array(
        [
            [-qc - gc, qs, 0.0, 0.0],
            [qc, -qs - ps_s, pe, 0.0],
            [0.0, ps_s, -pe, 0.0],
            [gc, 0.0, 0.0, 0.0],
        ]
    )
    step = float(times[1] - times[0])
    augmented = np.zeros((5, 5))
    augmented[:4, :4] = matrix
    augmented[0, 4] = 1.0
    propagator = expm(augmented * step)
    transition, infusion = propagator[:4, :4], propagator[:4, 4]

    # Exact injected amount per step of the piecewise-linear rate curve.
    knots = np.asarray(dose_rate_curve.times_s, dtype=float)
    rates = np.asarray(dose_rate_curve.values, dtype=float) * 1e6
    fine = np.union1d(knots, times)
    fine_rates = np.interp(fine, knots, rates, left=0.0, right=0.0)
    cumulative = np.concatenate(
        ([0.0], np.cumsum(0.5 * (fine_rates[1:] + fine_rates[:-1]) * np.diff(fine)))
    )
    injected = np.diff(np.interp(times, fine, cumulative))

    amounts = np.zeros((times.size, 4))
    for index in range(1, times.size):
        amounts[index] = (
            transition @ amounts[index - 1] + infusion * injected[index - 1] / step
        )
    return amounts, central_ml, systemic_ml, gfr


def build_mouse_dce_kinetics(
    config, labels, arterial, venous, arrival_time_s, voxel_volume_m3, dose_rate_curve
) -> MouseDCEKinetics:
    """Compile ground-truth DCE kinetics for the labelled mouse anatomy."""
    from .mouse_phantom import (
        MOUSE_TISSUE_LABELS,
        RENAL_LABELS,
        URETER_LABELS,
        BLADDER_LABEL,
        _urinary_transport,
    )

    start = float(config.injection_start_s)
    count = int(np.ceil(config.contrast_horizon_s / KINETICS_STEP_S)) + 1
    times = start + KINETICS_STEP_S * np.arange(count)
    amounts, central_ml, systemic_ml, gfr = _body_kinetics(
        config, times, dose_rate_curve
    )
    arterial_plasma = amounts[:, 0] / central_ml
    venous_plasma = amounts[:, 1] / systemic_ml
    blood_fraction = 1.0 - config.hematocrit

    compartment_map = labels.astype(np.int32) - 1
    body = labels > 0
    compartment_map[venous & body] = VENOUS_COMPARTMENT
    compartment_map[arterial & body] = ARTERIAL_COMPARTMENT
    names = tuple(MOUSE_TISSUE_LABELS[label] for label in range(1, 15))

    def delay_of(mask):
        values = arrival_time_s[mask & np.isfinite(arrival_time_s)]
        return float(np.mean(values)) - start if values.size else 0.0

    def delayed(values, delay_s):
        return np.interp(times - delay_s, times, values, left=0.0)

    concentration = np.zeros((count, N_COMPARTMENTS))
    parameters = tofts_parameters(config.dce_tofts_parameters)
    tissue_columns, plasma_inputs, k_values, tofts_rows = [], [], [], []
    for key, tissue_labels in TOFTS_TISSUE_LABELS.items():
        ktrans, ve, vp = parameters[key]
        ktrans /= 60.0
        for label in tissue_labels:
            mask = compartment_map == label - 1
            tissue_columns.append(label - 1)
            plasma_inputs.append(delayed(arterial_plasma, delay_of(mask)))
            k_values.append(ktrans / ve if ktrans > 0 else 0.0)
            tofts_rows.append((ktrans, vp))
    plasma_inputs = np.stack(plasma_inputs, axis=1)
    response = _first_order_response(times, plasma_inputs, k_values)
    tofts = np.asarray(tofts_rows)
    concentration[:, tissue_columns] = (
        tofts[:, 1] * plasma_inputs + tofts[:, 0] * response
    )
    concentration[:, ARTERIAL_COMPARTMENT] = blood_fraction * delayed(
        arterial_plasma, delay_of(compartment_map == ARTERIAL_COMPARTMENT)
    )
    concentration[:, VENOUS_COMPARTMENT] = blood_fraction * delayed(
        venous_plasma, delay_of(compartment_map == VENOUS_COMPARTMENT)
    )

    # Glomerular filtration splits evenly between both kidneys.
    voxel_ml = voxel_volume_m3 * 1e6
    filtered = 0.5 * gfr * arterial_plasma
    urinary = {"bladder": np.zeros(count)}
    for side, (cortex, medulla) in RENAL_LABELS.items():
        transport = _urinary_transport(
            times, filtered, config.renal_tubular_transit_s, config.ureter_transit_s
        )
        urinary[f"{side}_tubular"] = transport["tubular"]
        urinary[f"{side}_ureter"] = transport["ureter"]
        urinary["bladder"] += transport["bladder"]
        tubules = medulla if np.any(labels == medulla) else cortex
        for name, label in (
            (f"{side}_tubular", tubules),
            (f"{side}_ureter", URETER_LABELS[side]),
        ):
            voxels = int(np.count_nonzero(compartment_map == label - 1))
            if voxels:
                concentration[:, label - 1] += urinary[name] / (voxels * voxel_ml)
    voxels = int(np.count_nonzero(compartment_map == BLADDER_LABEL - 1))
    if voxels:
        concentration[:, BLADDER_LABEL - 1] += urinary["bladder"] / (voxels * voxel_ml)

    return MouseDCEKinetics(
        times_s=times,
        body_amounts_umol=amounts,
        arterial_plasma_mm=arterial_plasma,
        venous_plasma_mm=venous_plasma,
        urinary_amounts_umol=urinary,
        compartment_map=compartment_map,
        compartment_names=names,
        concentration_mm=np.maximum(concentration, 0.0),
        r1_mm_s_inv=float(config.contrast_r1_mm_s_inv),
        r2_mm_s_inv=float(config.contrast_r2_mm_s_inv),
        dose_umol=float(config.injection_amount_umol),
    )
