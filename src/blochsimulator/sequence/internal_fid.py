"""Configurable, gradient-free excitation and FID repetitions."""

from __future__ import annotations

import numpy as np

from .model import ADCEvent, RFEvent, SequenceProgram
from .rf_pulses import design_rf_envelope, scale_rf_envelope_to_flip


def make_internal_fid(
    *,
    flip_angle_deg=90.0,
    rf_phase_deg=0.0,
    rf_pulse_type="block",
    rf_duration_s=1e-3,
    rf_raster_s=1e-6,
    rf_time_bandwidth_product=4.0,
    rf_apodization=0.5,
    rf_slr_sharpness=1,
    rf_frequency_offset_hz=0.0,
    rf_custom_waveform_hz=None,
    rf_custom_raster_s=None,
    rf_custom_flip_angle_deg=None,
    rf_custom_name=None,
    num_samples=256,
    sampling_bandwidth_hz=10_000.0,
    adc_delay_s=0.0,
    receiver_frequency_offset_hz=0.0,
    repetitions=1,
    repetition_time_s=0.0,
):
    """Build a continuous Bloch sequence with one excitation/ADC per repetition.

    ADC samples lie at dwell centres. TR is RF-start to RF-start; zero selects
    RF duration + ADC delay + acquisition duration. The sequence ends after
    the final acquisition, without an additional recovery interval or spoiling.
    The RF carrier phase is restarted relative to each pulse's own start.
    """
    for name, value in (("num_samples", num_samples), ("repetitions", repetitions)):
        if not np.isfinite(value) or int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite(sampling_bandwidth_hz) or sampling_bandwidth_hz <= 0:
        raise ValueError("sampling_bandwidth_hz must be positive and finite")
    for name, value in (
        ("adc_delay_s", adc_delay_s),
        ("repetition_time_s", repetition_time_s),
    ):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be non-negative and finite")
    envelope, duration, tbw, pulse_type = design_rf_envelope(
        pulse_type=rf_pulse_type,
        duration_s=rf_duration_s,
        raster_s=rf_raster_s,
        time_bandwidth_product=rf_time_bandwidth_product,
        apodization=rf_apodization,
        slr_sharpness=rf_slr_sharpness,
        custom_waveform=rf_custom_waveform_hz,
        custom_raster_s=rf_custom_raster_s,
        flip_angle_deg=flip_angle_deg,
    )
    samples = scale_rf_envelope_to_flip(
        envelope,
        flip_angle_deg=flip_angle_deg,
        raster_s=rf_raster_s,
        reference_flip_angle_deg=(
            rf_custom_flip_angle_deg if pulse_type == "designer" else None
        ),
    )
    # A constant block needs just one event cell, as in the original FID.
    raster = rf_raster_s
    if pulse_type == "block":
        samples = samples[:1]
        raster = duration
    dwell = 1.0 / sampling_bandwidth_hz
    acquisition_end = duration + adc_delay_s + int(num_samples) * dwell
    tr = float(repetition_time_s) or acquisition_end
    if tr < acquisition_end - 1e-12:
        raise ValueError(
            f"TR must be at least {acquisition_end * 1000:.6g} ms "
            "to fit the RF pulse, ADC delay and complete FID"
        )
    tr = max(tr, acquisition_end)
    events = []
    for index in range(int(repetitions)):
        start = index * tr
        events.extend(
            (
                RFEvent(
                    start,
                    samples,
                    raster,
                    frequency_offset_hz=rf_frequency_offset_hz,
                    phase_offset_rad=np.deg2rad(rf_phase_deg),
                ),
                ADCEvent(
                    start + duration + adc_delay_s + dwell / 2,
                    int(num_samples),
                    dwell,
                    frequency_offset_hz=receiver_frequency_offset_hz,
                ),
            )
        )
    return SequenceProgram(
        events=tuple(events),
        duration_s=(int(repetitions) - 1) * tr + acquisition_end,
        source="internal-fid",
        metadata={
            "definitions": {
                "Name": "Internal FID",
                "RFPulseType": pulse_type,
                "FlipAngleDeg": float(flip_angle_deg),
                "RFDuration": duration,
                "RFTimeBandwidthProduct": tbw,
                "Repetitions": int(repetitions),
                "RepetitionTime": tr,
                "FIDPoints": int(num_samples),
                "SamplingBandwidthHz": float(sampling_bandwidth_hz),
                "ADCDelay": float(adc_delay_s),
            }
        },
    )
