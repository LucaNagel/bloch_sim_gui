"""Single-channel receiver noise from a relative SNR or a phantom calibration.

Noise standard deviations always describe one quadrature (real or imaginary),
not the magnitude or complex RMS. Relative SNR refers to the ideal 90-degree
FID of the whole simulated object. With ``signal_weighting="voxel_volume"`` and
concentrations in mM (= mol/m^3), simulated signal is in mol times
polarization relative to thermal equilibrium at the calibration's target
temperature.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Optional

import numpy as np

from ..units import thermal_polarization


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _single_receiver(phantom):
    maps = getattr(phantom, "rx_sensitivity_maps", None)
    if maps is not None and np.asarray(maps).shape[0] != 1:
        raise ValueError("Receiver noise currently supports one receive channel")
    return 1.0 if maps is None else np.asarray(maps)[0]


def _curve_product_integral(first, second):
    """Integrate the product of two piecewise curves over their sample range."""
    knots = sorted({*first.times_s, *(() if second is None else second.times_s)})
    if len(knots) < 2:
        return 0.0
    total = 0.0
    for left, right in zip(knots[:-1], knots[1:]):
        f_left, f_right = first.interval_values(left, right)
        g_left, g_right = (
            (1.0, 1.0) if second is None else second.interval_values(left, right)
        )
        # Exact for the product of two linear (or constant) segments.
        total += (
            (right - left)
            * (
                2 * f_left * g_left
                + f_left * g_right
                + f_right * g_left
                + 2 * f_right * g_right
            )
            / 6
        )
    return float(total)


def ideal_amplitude_map(phantom, signal_weighting="voxel_volume"):
    """Complex ideal 90-degree amplitude of every phantom voxel.

    Its sum is the whole-object FID of :func:`ideal_fid_amplitude`; a pyruvate
    inflow adds its delivered dose where the delivery map places it.
    """
    if signal_weighting not in {"voxel", "voxel_volume"}:
        raise ValueError("signal_weighting must be 'voxel' or 'voxel_volume'")
    if signal_weighting == "voxel_volume" and len(phantom.shape) != 3:
        raise ValueError("Physical signal weighting requires a 3D phantom with volume")
    receive = np.broadcast_to(_single_receiver(phantom), phantom.shape)
    if hasattr(phantom, "initial_spin_density_maps"):
        magnetization = np.asarray(sum(phantom.initial_concentration_maps.values()))
    elif hasattr(phantom, "concentration_maps"):
        magnetization = sum(
            density * phantom.initial_mz_maps[name]
            for name, density in phantom.concentration_maps.items()
        )
    else:
        magnetization = phantom.pd_map * np.linalg.norm(phantom.m0_map, axis=-1)
    amplitude = np.where(phantom.mask, np.asarray(magnetization), 0.0) * receive
    inflow = getattr(phantom, "pyruvate_inflow", None)
    if inflow is not None:
        delivery = np.asarray(inflow.delivery_map)
        dose = _curve_product_integral(
            inflow.rate_curve_s_inv, inflow.polarization_curve
        )
        amplitude = amplitude + dose * delivery * receive
    scale = phantom.voxel_volume_m3 if signal_weighting == "voxel_volume" else 1.0
    return amplitude * scale


def ideal_fid_amplitude(phantom, signal_weighting="voxel_volume"):
    """Ideal 90-degree FID amplitude of all initial and delivered magnetization.

    Static Mz (or |M0|) and spectral Mz are normalized polarization, while
    dynamic concentration maps already contain concentration times
    polarization. A pyruvate inflow adds its whole delivered dose (rate times
    polarization integrated over the curves) as if it were present at once.
    Tx inhomogeneity, off-resonance, relaxation and metabolism do not enter
    this ideal instantaneous reference; it uses the run's signal weighting.
    """
    amplitude = float(abs(np.sum(ideal_amplitude_map(phantom, signal_weighting))))
    if not np.isfinite(amplitude) or amplitude <= 0:
        raise ValueError(
            "Relative SNR needs a phantom with initial or delivered magnetization"
        )
    return amplitude


@dataclass(frozen=True)
class ReceiverCalibration:
    """Scanner sensitivity from one FID of a phantom with known spin content.

    ``measured_amplitude`` is the FID amplitude at t=0 after a
    ``flip_angle_deg`` excitation and ``measured_noise_std`` the per-quadrature
    noise std (e.g. from a 0-degree scan), both in the same scanner units and
    from scans with the same gain, ``averages`` and ADC ``dwell_s``. The phantom
    holds ``concentration_mm`` (mM) in ``volume_ml`` (ml); ``polarization`` is
    the absolute reference polarization, or None for thermal equilibrium at
    ``temperature_k``.

    Simulated signal is read as mol times polarization relative to thermal at
    ``target_temperature_k``; ``noise_factor`` multiplies the target noise (e.g.
    extra coil loading in vivo).
    """

    measured_amplitude: float
    measured_noise_std: float
    concentration_mm: float
    volume_ml: float
    nucleus: str
    field_strength_t: float
    dwell_s: float
    averages: int = 1
    flip_angle_deg: float = 90.0
    temperature_k: float = 293.15
    nuclei_per_molecule: int = 1
    enrichment: float = 1.0
    polarization: Optional[float] = None
    target_temperature_k: float = 310.15
    noise_factor: float = 1.0

    def __post_init__(self):
        for name, label in (
            ("measured_amplitude", "Measured FID amplitude"),
            ("measured_noise_std", "Measured noise std"),
            ("concentration_mm", "Reference concentration"),
            ("volume_ml", "Reference volume"),
            ("field_strength_t", "Field strength"),
            ("dwell_s", "Reference dwell"),
            ("temperature_k", "Reference temperature"),
            ("target_temperature_k", "Target temperature"),
            ("noise_factor", "Noise factor"),
        ):
            _positive(getattr(self, name), label)
        for name, label in (
            ("averages", "Averages"),
            ("nuclei_per_molecule", "Nuclei per molecule"),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        if not 0 < self.flip_angle_deg < 180:
            raise ValueError("Reference flip angle must be between 0 and 180 degrees")
        if not 0 < self.enrichment <= 1:
            raise ValueError("Enrichment must be in (0, 1]")
        if self.polarization is not None and not 0 < self.polarization <= 1:
            raise ValueError("Reference polarization must be in (0, 1]")
        if not self.nucleus:
            raise ValueError("Reference nucleus must not be empty")
        # Validates the nucleus/field/temperature for both polarizations.
        self.reference_polarization
        self.target_thermal_polarization

    @property
    def reference_moles(self):
        """Observed nuclei in the reference phantom, in mol (mM * ml * 1e-6)."""
        return (
            self.concentration_mm
            * self.volume_ml
            * 1e-6
            * self.nuclei_per_molecule
            * self.enrichment
        )

    @property
    def reference_polarization(self):
        if self.polarization is not None:
            return float(self.polarization)
        return thermal_polarization(
            self.nucleus, self.field_strength_t, self.temperature_k
        )

    @property
    def target_thermal_polarization(self):
        return thermal_polarization(
            self.nucleus, self.field_strength_t, self.target_temperature_k
        )

    @property
    def measured_snr(self):
        return self.measured_amplitude / self.measured_noise_std

    @property
    def single_scan_snr(self):
        return self.measured_snr / np.sqrt(self.averages)

    @property
    def sensitivity_per_mol(self):
        """Single-scan FID SNR per mol of fully polarized nuclei at ``dwell_s``."""
        transverse = np.sin(np.deg2rad(self.flip_angle_deg))
        return self.single_scan_snr / (
            self.reference_moles * self.reference_polarization * transverse
        )

    def noise_std(self, dwell_s):
        """Per-quadrature std in simulation units for one scan at ``dwell_s``.

        Noise power is proportional to the receiver bandwidth 1/dwell.
        """
        bandwidth_scale = np.sqrt(self.dwell_s / _positive(dwell_s, "ADC dwell"))
        return (
            bandwidth_scale
            * self.noise_factor
            / (self.sensitivity_per_mol * self.target_thermal_polarization)
        )

    def check_context(self, *, nucleus, field_strength_t):
        if nucleus != self.nucleus or not np.isclose(
            field_strength_t, self.field_strength_t, rtol=1e-9, atol=0
        ):
            raise ValueError(
                "The receiver calibration requires the same nucleus and field "
                f"strength ({self.nucleus}, {self.field_strength_t:g} T)"
            )

    def to_metadata(self):
        return asdict(self)

    def derived_metadata(self):
        return {
            "reference_moles": self.reference_moles,
            "reference_polarization": self.reference_polarization,
            "target_thermal_polarization": self.target_thermal_polarization,
            "single_scan_snr": self.single_scan_snr,
            "sensitivity_per_mol": self.sensitivity_per_mol,
        }


def receiver_acquisition_metadata(
    program, *, nucleus, field_strength_t, phantom=None, signal_weighting=None
):
    """Scanner context of a run; ADC events are in signal order.

    With a phantom and its signal weighting, the ideal FID amplitude that
    relative SNR refers to is stored as well (None if it is not defined).
    """
    events = program.adc_events
    metadata = {
        "nucleus": nucleus,
        "field_strength_t": float(field_strength_t),
        "adc_dwell_s": [float(event.dwell_s) for event in events],
        "adc_samples": [int(event.num_samples) for event in events],
    }
    if phantom is not None:
        temperature = (getattr(phantom, "metadata", {}) or {}).get("temperature_k")
        if temperature is not None:
            metadata["temperature_k"] = _positive(temperature, "Phantom temperature")
        try:
            amplitude = ideal_fid_amplitude(phantom, signal_weighting or "voxel")
        except ValueError:
            amplitude = None
        metadata["ideal_fid_amplitude"] = amplitude
    return metadata


@dataclass(frozen=True)
class ReceiverNoise:
    """Receiver noise settings.

    ``"relative"`` sets the single-scan SNR of the object's ideal 90-degree FID
    at ``reference_dwell_s``; ``"calibrated"`` uses a measured phantom FID. Both
    scale the noise std with sqrt(bandwidth) for each ADC event.
    """

    mode: str = "off"
    snr: float = 100.0
    seed: int = 0
    reference_dwell_s: float = 1e-4
    calibration: Optional[ReceiverCalibration] = None

    def __post_init__(self):
        if self.mode not in {"off", "relative", "calibrated"}:
            raise ValueError("Unknown receiver noise mode")
        _positive(self.snr, "Relative SNR")
        _positive(self.reference_dwell_s, "Reference dwell")
        if isinstance(self.seed, bool) or int(self.seed) != self.seed or self.seed < 0:
            raise ValueError("Noise seed must be a non-negative integer")
        if self.mode == "calibrated" and self.calibration is None:
            raise ValueError(
                "Set the receiver calibration before running or applying noise"
            )

    def noise_std_for_dwell(self, dwell_s, ideal_fid_amplitude=None):
        """Per-quadrature std for one scan at ``dwell_s`` in simulation units."""
        if self.mode == "off":
            return 0.0
        if self.mode == "calibrated":
            return self.calibration.noise_std(dwell_s)
        if ideal_fid_amplitude is None:
            raise ValueError(
                "Relative SNR needs the simulated object's ideal FID amplitude"
            )
        bandwidth_scale = np.sqrt(
            self.reference_dwell_s / _positive(dwell_s, "ADC dwell")
        )
        return bandwidth_scale * float(ideal_fid_amplitude) / self.snr

    def for_temperature(self, temperature_k):
        """Use a phantom's explicit thermal reference, retaining the measured calibration.

        Legacy objects without a temperature keep the configured target value.
        """
        if self.mode != "calibrated" or temperature_k is None:
            return self
        return replace(
            self,
            calibration=replace(
                self.calibration,
                target_temperature_k=_positive(temperature_k, "Phantom temperature"),
            ),
        )


def receiver_noise_samples(shape, seed, sigma):
    """Draw per-quadrature Gaussian noise; ``sigma`` may vary along the last axis."""
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    return sigma * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def _sample_std(settings, context, count):
    """Noise std for every ADC sample and for every ADC event."""
    label = (
        "Calibrated receiver noise" if settings.mode == "calibrated" else "Relative SNR"
    )
    if not context:
        raise ValueError(
            f"{label} needs the run's nucleus, field strength and ADC dwell"
        )
    settings = settings.for_temperature(context.get("temperature_k"))
    amplitude = None
    if settings.mode == "calibrated":
        settings.calibration.check_context(
            nucleus=context["nucleus"], field_strength_t=context["field_strength_t"]
        )
    else:
        if "ideal_fid_amplitude" not in context:
            raise ValueError(
                "Rerun the simulation: this result lacks the ideal FID amplitude "
                "that relative SNR refers to"
            )
        amplitude = context["ideal_fid_amplitude"]
        if amplitude is None:
            raise ValueError(
                "Relative SNR needs a phantom with initial or delivered magnetization"
            )
    dwells = list(context["adc_dwell_s"])
    samples = list(context.get("adc_samples") or [])
    if not dwells:
        raise ValueError(f"{label} needs ADC samples")
    event_std = [
        float(settings.noise_std_for_dwell(dwell, amplitude)) for dwell in dwells
    ]
    if not samples:
        if not np.allclose(event_std, event_std[0], rtol=1e-12, atol=0):
            raise ValueError("ADC sample counts are needed for mixed dwells")
        return np.full(count, event_std[0]), event_std
    if len(samples) != len(dwells) or sum(samples) != count:
        raise ValueError("ADC events do not match the received signal length")
    return np.repeat(event_std, samples), event_std


def apply_receiver_noise(result, settings: ReceiverNoise):
    """Apply one realization to the summed ADC stream, always from clean data.

    Magnetization and species-resolved ground truth stay unchanged. Calling this
    function again never adds a second noise realization to the previous one.
    Noise scales with sqrt(bandwidth) per ADC event.
    """
    clean = result.signal_noiseless
    if clean is None:
        clean = result.signal
    clean = np.asarray(clean)
    if settings.mode == "off" and result.signal_noiseless is None:
        return result
    if clean.ndim not in (1, 2) or (clean.ndim == 2 and clean.shape[0] != 1):
        raise ValueError("Receiver noise currently supports one receive channel")
    context = result.metadata.get("receiver_acquisition")
    settings = settings.for_temperature((context or {}).get("temperature_k"))
    calibration = settings.calibration if settings.mode == "calibrated" else None
    if calibration is not None:
        if result.metadata.get("signal_weighting") != "voxel_volume":
            raise ValueError(
                "Calibrated receiver noise requires physical voxel-volume "
                "signal weighting"
            )
    if settings.mode == "off":
        sigma, event_std = 0.0, [0.0] * len((context or {}).get("adc_dwell_s", ()))
    else:
        sigma, event_std = _sample_std(settings, context, clean.shape[-1])
    signal = clean + receiver_noise_samples(clean.shape, settings.seed, sigma)
    metadata = dict(result.metadata)
    metadata["receiver_noise"] = {
        "mode": settings.mode,
        "noise_std_per_quadrature": event_std,
        "noise_std_scope": "per_adc_event",
        "seed": int(settings.seed),
        "generator": "PCG64",
        "relative_snr": (
            None
            if settings.mode != "relative"
            else {
                "snr": settings.snr,
                "reference_dwell_s": settings.reference_dwell_s,
                "ideal_fid_amplitude": context.get("ideal_fid_amplitude"),
            }
        ),
        "calibration": None if calibration is None else calibration.to_metadata(),
        "calibration_derived": (
            None if calibration is None else calibration.derived_metadata()
        ),
        "signal_unit": (
            None
            if calibration is None
            else "mol x polarization relative to thermal at target_temperature_k"
        ),
        "species_signal": "noiseless_ground_truth",
    }
    return replace(result, signal=signal, signal_noiseless=clean, metadata=metadata)
