"""SNR budget from the receiver setting to the expected image-voxel SNR.

The chain is: the receiver setting fixes the noise std at the reference
bandwidth; the simulated object's ideal 90-degree FID sets the best possible
whole-object SNR; every ADC event rescales the noise with its bandwidth; image
voxels hold their share of the object. After a run, the clean signal gives the
matched-filter SNR of the whole acquisition. For a unitary encoding (fully
sampled Cartesian or EPSI) the squared voxel SNRs over all voxels and frames
add up to exactly this value (Parseval); if every frame keeps the object's
spatial distribution, voxel ``v`` therefore reaches
``SNR_total * a_v / sqrt(sum(a**2))``. Spiral, radial or undersampled
reconstructions are not unitary, so the value is an upper bound there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .receiver import ReceiverNoise, _sample_std, ideal_amplitude_map

KELVIN_AT_0_CELSIUS = 273.15
_PHYSICAL_UNIT = "mol × P/P_th"


@dataclass(frozen=True)
class BudgetRow:
    label: str
    value: str = ""
    note: str = ""
    number: Optional[float] = None


@dataclass(frozen=True)
class BudgetSection:
    title: str
    rows: Tuple[BudgetRow, ...]
    status: str = ""


@dataclass(frozen=True)
class VoxelShares:
    """How the ideal object amplitude spreads over image voxels."""

    voxels: int
    effective_voxels: float
    brightest_amplitude: float
    brightest_share: float
    sum_rule_share: float
    outside_fov_fraction: float


@dataclass(frozen=True)
class AcquisitionSNR:
    """SNR of a run's noiseless signal under one receiver setting."""

    samples: int
    peak: float
    peak_time_s: Optional[float]
    total: float
    species: Tuple[Tuple[str, float], ...] = ()


def grid_voxel_amplitudes(amplitude_map, positions_m, matrix, fov_m, basis_xyz=None):
    """Sum complex phantom amplitudes into the image voxels of an encoding grid.

    Voxel ``n`` of an axis with ``N`` cells covers ``[(n - N/2) d, (n - N/2 + 1) d)``
    with ``d = fov / N`` (the voxel-centred Cartesian reconstruction), measured
    along the columns of ``basis_xyz`` (read, phase, partition). Phantom voxels
    are binned by their centres; axes beyond ``len(matrix)`` (the slice of a 2D
    grid) are summed. Returns the binned complex values and the summed
    magnitude outside the FOV, which would alias.
    """
    matrix = tuple(int(value) for value in matrix)
    dimensions = len(matrix)
    if not 1 <= dimensions <= 3 or min(matrix) < 1:
        raise ValueError("Encoding matrix must have one to three positive sizes")
    fov = np.asarray(fov_m, dtype=float)[:dimensions]
    if fov.size != dimensions or not np.all(fov > 0):
        raise ValueError("Encoding FOV must have one positive size per matrix axis")
    basis = np.eye(3) if basis_xyz is None else np.asarray(basis_xyz, dtype=float)
    values = np.asarray(amplitude_map).ravel()
    positions = np.asarray(positions_m, dtype=float).reshape(-1, 3)
    if positions.shape[0] != values.size:
        raise ValueError("Phantom positions do not match its amplitude map")
    logical = positions @ basis[:, :dimensions]
    sizes = np.asarray(matrix)
    index = np.floor(logical / (fov / sizes) + sizes / 2).astype(np.int64)
    inside = np.all((index >= 0) & (index < sizes), axis=1)
    cells = np.zeros(int(np.prod(sizes)), dtype=complex)
    np.add.at(cells, np.ravel_multi_index(index[inside].T, matrix), values[inside])
    outside = float(np.sum(np.abs(values[~inside])))
    return cells.reshape(matrix), outside


def voxel_shares(cells, object_amplitude, outside=0.0):
    """Brightest-voxel share of the whole-object FID and of the sum rule."""
    magnitude = np.abs(np.asarray(cells)).ravel()
    energy = float(np.sqrt(np.sum(magnitude**2)))
    if energy <= 0 or object_amplitude <= 0:
        return None
    brightest = float(np.max(magnitude))
    total = float(np.sum(magnitude))
    return VoxelShares(
        voxels=int(np.count_nonzero(magnitude > brightest * 1e-9)),
        effective_voxels=total**2 / energy**2,
        brightest_amplitude=brightest,
        brightest_share=brightest / float(object_amplitude),
        sum_rule_share=brightest / energy,
        outside_fov_fraction=float(outside) / (total + float(outside)),
    )


def acquisition_snr(result, noise: ReceiverNoise):
    """SNR of a run's noiseless signal with the noise std of ``noise``.

    ``peak`` is the strongest single sample, ``total`` the matched-filter SNR
    sqrt(sum |s_i|^2 / sigma_i^2) of all samples, also for each species.
    """
    if noise.mode == "off":
        raise ValueError("Receiver noise is off")
    clean = result.signal_noiseless
    clean = np.asarray(result.signal if clean is None else clean)
    if clean.ndim == 2 and clean.shape[0] == 1:
        clean = clean[0]
    if clean.ndim != 1:
        raise ValueError("Receiver noise currently supports one receive channel")
    if (
        noise.mode == "calibrated"
        and result.metadata.get("signal_weighting") != "voxel_volume"
    ):
        raise ValueError(
            "Rerun with physical voxel-volume signal weighting for calibrated SNR"
        )
    sigma, _ = _sample_std(
        noise, result.metadata.get("receiver_acquisition"), clean.shape[-1]
    )
    ratio = np.abs(clean) / sigma
    peak = int(np.argmax(ratio)) if ratio.size else 0
    times = getattr(result, "adc_times_s", None)
    peak_time = (
        float(np.asarray(times)[peak])
        if times is not None and len(times) == ratio.size and ratio.size
        else None
    )
    species = []
    names = tuple(getattr(result, "pool_names", ()) or ())
    species_signal = getattr(result, "species_signal", None)
    if species_signal is not None and len(names) > 1:
        values = np.asarray(species_signal).reshape(len(names), -1)
        if values.shape[1] == clean.size:
            species = [
                (str(name), float(np.sqrt(np.sum((np.abs(row) / sigma) ** 2))))
                for name, row in zip(names, values)
            ]
    return AcquisitionSNR(
        samples=int(clean.size),
        peak=float(ratio[peak]) if ratio.size else 0.0,
        peak_time_s=peak_time,
        total=float(np.sqrt(np.sum(ratio**2))),
        species=tuple(species),
    )


def _number(value, digits=4):
    return f"{value:.{digits}g}"


def _bandwidth(dwell_s):
    bandwidth = 1.0 / dwell_s
    if bandwidth >= 1e3:
        return f"{bandwidth / 1e3:.4g} kHz"
    return f"{bandwidth:.4g} Hz"


def format_amount(mol):
    """Amount of substance with a prefix that keeps the number readable."""
    for factor, unit in ((1.0, "mol"), (1e-3, "mmol"), (1e-6, "µmol")):
        if abs(mol) >= factor:
            return f"{mol / factor:.4g} {unit}"
    return f"{mol * 1e9:.4g} nmol"


def _celsius(kelvin):
    return f"{kelvin - KELVIN_AT_0_CELSIUS:.4g} °C"


def _message(title, text, status=""):
    return BudgetSection(title, (BudgetRow(text),), status)


def _reference_model_note(calibration):
    """Short description of the reference scan model behind the efficiency."""
    scan = calibration.reference_scan or {}
    parts = [f"sin {calibration.flip_angle_deg:g}°"]
    if scan.get("t1_s") and scan.get("dummy_scans", 0) + scan.get("averages", 1) > 1:
        parts.append(
            f"T1 {scan['t1_s']:g} s at TR {scan['repetition_time_s']:g} s, "
            f"{scan.get('dummy_scans', 0)} dummy scan(s)"
        )
    if scan.get("t2star_s"):
        parts.append(
            f"T2* {scan['t2star_s'] * 1e3:g} ms after "
            f"{scan.get('amplitude_delay_s', 0.0) * 1e3:g} ms"
        )
    if scan.get("coil_model"):
        parts.append(scan["coil_model"])
    return " × ".join(parts)


def _calibration_section(calibration, preview):
    transverse = calibration.reference_signal_fraction
    polarization = calibration.reference_polarization
    if calibration.polarization is None:
        polarization_row = BudgetRow(
            "Polarization",
            f"{polarization * 1e6:.4g} ppm",
            f"thermal at {_celsius(calibration.temperature_k)}, "
            f"{calibration.field_strength_t:g} T",
            polarization,
        )
    else:
        polarization_row = BudgetRow(
            "Polarization", f"{polarization * 100:.4g} %", "entered", polarization
        )
    moles = calibration.reference_moles
    enrichment = calibration.enrichment * 100
    return BudgetSection(
        "Reference phantom",
        (
            BudgetRow(
                "Observed nuclei",
                format_amount(moles),
                f"{calibration.concentration_mm:g} mM × {calibration.volume_ml:g} ml"
                f" × {calibration.nuclei_per_molecule} label(s) × {enrichment:g} %",
                moles,
            ),
            polarization_row,
            BudgetRow(
                "Fully polarized equivalent",
                format_amount(moles * polarization),
                "nuclei × polarization",
                moles * polarization,
            ),
            (
                BudgetRow(
                    "Transverse fraction",
                    _number(transverse, 3),
                    f"sin {calibration.flip_angle_deg:g}°",
                    transverse,
                )
                if calibration.signal_efficiency is None
                else BudgetRow(
                    "Reference signal fraction",
                    _number(transverse, 4),
                    _reference_model_note(calibration),
                    transverse,
                )
            ),
            BudgetRow(
                "Measured SNR",
                _number(calibration.measured_snr),
                f"{calibration.measured_amplitude:g} / "
                f"{calibration.measured_noise_std:g}, "
                f"{calibration.averages} average(s)",
                calibration.measured_snr,
            ),
            BudgetRow(
                "Single-scan SNR",
                _number(calibration.single_scan_snr),
                "measured / √averages",
                calibration.single_scan_snr,
            ),
            BudgetRow(
                "Sensitivity",
                f"{calibration.sensitivity_per_mol * 1e-6:.4g} per µmol",
                "single-scan FID SNR per µmol fully polarized nuclei",
                calibration.sensitivity_per_mol,
            ),
            BudgetRow(
                "Reference bandwidth",
                _bandwidth(calibration.dwell_s),
                f"dwell {calibration.dwell_s * 1e6:.4g} µs",
                1.0 / calibration.dwell_s,
            ),
        ),
        "preview — not set" if preview else "set",
    )


def _readout_rows(program, noise, amplitude, ceiling):
    events = tuple(getattr(program, "adc_events", ()) or ())
    if not events:
        return (BudgetRow("Build a sequence with ADC events to see its readout."),)
    groups = {}
    for event in events:
        key = float(f"{float(event.dwell_s):.12g}")
        count, samples = groups.get(key, (0, 0))
        groups[key] = (count + 1, samples + int(event.num_samples))
    rows = []
    for dwell, (count, samples) in sorted(groups.items()):
        scale = np.sqrt(noise.reference_dwell_s / dwell)
        if noise.mode == "relative" and amplitude is None:
            sigma_text = f"σ × {scale:.3g}"
        else:
            sigma = noise.noise_std_for_dwell(dwell, amplitude)
            sigma_text = f"σ = {sigma:.3g}"
        snr = None if ceiling is None else ceiling / scale
        rows.append(
            BudgetRow(
                f"Whole object at {_bandwidth(dwell)}",
                "—" if snr is None else _number(snr),
                f"SNR per sample; {count} ADC event(s), {samples} samples, "
                f"{sigma_text}",
                snr,
            )
        )
    return tuple(rows)


def _voxel_rows(grid, shares, sigma_reference):
    matrix, fov_m, _ = grid
    sizes = " × ".join(str(int(value)) for value in matrix)
    widths = " × ".join(
        f"{width * 1e3:.3g}" for width in np.asarray(fov_m)[: len(matrix)] / matrix
    )
    roles = "read × phase × partition" if len(matrix) == 3 else "read × phase"
    rows = [
        BudgetRow("Grid", sizes, roles),
        BudgetRow(
            "Voxel size",
            f"{widths} mm",
            (
                "2D grid: each voxel spans the whole excited slab"
                if len(matrix) == 2
                else ""
            ),
        ),
    ]
    if shares is None:
        rows.append(BudgetRow("No object signal falls inside the encoding FOV."))
        return tuple(rows)
    rows.extend(
        (
            BudgetRow(
                "Voxels with signal",
                str(shares.voxels),
                f"equivalent to {shares.effective_voxels:.4g} equally bright voxels",
                shares.voxels,
            ),
            BudgetRow(
                "Brightest voxel share",
                f"{shares.brightest_share * 100:.3g} %",
                "of the whole-object FID amplitude",
                shares.brightest_share,
            ),
            BudgetRow(
                "Brightest voxel, one sample",
                _number(shares.brightest_amplitude / sigma_reference),
                "SNR at the reference bandwidth; reconstruction adds the other "
                "samples (see last run)",
                shares.brightest_amplitude / sigma_reference,
            ),
        )
    )
    if shares.outside_fov_fraction > 1e-3:
        rows.append(
            BudgetRow(
                "Outside the FOV",
                f"{shares.outside_fov_fraction * 100:.3g} %",
                "of the object amplitude; aliases into the image",
                shares.outside_fov_fraction,
            )
        )
    return tuple(rows)


def _last_run_rows(result, noise, shares):
    if result is None:
        return (BudgetRow("Run the simulation to measure its acquisition SNR."),)
    try:
        measured = acquisition_snr(result, noise)
    except (TypeError, ValueError) as exc:
        return (BudgetRow(str(exc)),)
    rows = [
        BudgetRow(
            "Strongest sample",
            _number(measured.peak),
            (
                ""
                if measured.peak_time_s is None
                else f"at {measured.peak_time_s * 1e3:.4g} ms"
            ),
            measured.peak,
        ),
        BudgetRow(
            "All samples combined",
            _number(measured.total),
            f"√Σ|s|²/σ² over {measured.samples} samples (matched filter)",
            measured.total,
        ),
    ]
    rows.extend(
        BudgetRow(name, _number(value), "noiseless species signal, all samples", value)
        for name, value in measured.species
    )
    if shares is not None:
        value = measured.total * shares.sum_rule_share
        rows.append(
            BudgetRow(
                "Brightest voxel, all samples",
                _number(value),
                "sum rule over all frames: exact for fully sampled Cartesian / "
                "EPSI if the object keeps its distribution, upper bound otherwise",
                value,
            )
        )
    return tuple(rows)


def snr_budget(
    noise: Optional[ReceiverNoise],
    *,
    calibration_preview=False,
    phantom=None,
    signal_weighting="voxel",
    program=None,
    grid=None,
    result=None,
    nucleus=None,
    field_strength_t=None,
):
    """Build the budget sections for the current receiver setting.

    ``calibration_preview`` marks a calibration built from entered values that
    is not set yet. ``grid`` is ``(matrix, fov_m, EncodingFrame)`` of the
    sequence's spatial encoding, or None.
    """
    if noise is None or noise.mode == "off":
        return (
            _message(
                "Receiver noise is off",
                "Choose Relative SNR or Calibrated from phantom FID to see the "
                "budget from the receiver setting to the voxel SNR.",
            ),
        )
    noise = noise.for_temperature(
        (getattr(phantom, "metadata", {}) or {}).get("temperature_k")
    )
    sections = []
    calibration = noise.calibration
    if noise.mode == "relative":
        sections.append(
            BudgetSection(
                "Receiver setting",
                (
                    BudgetRow(
                        "Relative SNR",
                        _number(noise.snr),
                        "ideal 90° FID of the whole object at t = 0, one scan",
                        noise.snr,
                    ),
                    BudgetRow(
                        "Reference bandwidth",
                        _bandwidth(noise.reference_dwell_s),
                        f"dwell {noise.reference_dwell_s * 1e6:.4g} µs",
                        1.0 / noise.reference_dwell_s,
                    ),
                ),
            )
        )
    else:
        sections.append(_calibration_section(calibration, calibration_preview))
        target = calibration.target_thermal_polarization
        sections.append(
            BudgetSection(
                "Simulation units",
                (
                    BudgetRow(
                        "Signal unit",
                        _PHYSICAL_UNIT,
                        f"P_th = {target * 1e6:.4g} ppm, thermal at "
                        f"{_celsius(calibration.target_temperature_k)}",
                        target,
                    ),
                    BudgetRow(
                        "Noise factor",
                        f"{calibration.noise_factor:g}",
                        "extra noise of the simulated object, e.g. coil loading",
                        calibration.noise_factor,
                    ),
                    BudgetRow(
                        "Noise σ at reference bandwidth",
                        f"{calibration.noise_std(calibration.dwell_s):.4g}",
                        "per quadrature, one scan",
                        calibration.noise_std(calibration.dwell_s),
                    ),
                ),
            )
        )

    amplitude = None
    amplitude_map = None
    object_rows = None
    if phantom is None:
        object_rows = (BudgetRow("Choose or design a phantom to see its SNR."),)
    else:
        weighting = "voxel_volume" if noise.mode == "calibrated" else signal_weighting
        try:
            if noise.mode == "calibrated":
                calibration.check_context(
                    nucleus=nucleus, field_strength_t=field_strength_t
                )
            amplitude_map = ideal_amplitude_map(phantom, weighting)
            amplitude = float(abs(np.sum(amplitude_map)))
            if not np.isfinite(amplitude) or amplitude <= 0:
                raise ValueError(
                    "The phantom has no initial or delivered magnetization"
                )
        except (TypeError, ValueError) as exc:
            amplitude = amplitude_map = None
            object_rows = (BudgetRow(str(exc)),)
    ceiling = noise.snr if noise.mode == "relative" else None
    sigma_reference = None
    if amplitude is not None:
        sigma_reference = noise.noise_std_for_dwell(noise.reference_dwell_s, amplitude)
        ceiling = amplitude / sigma_reference
        unit = f" {_PHYSICAL_UNIT}" if noise.mode == "calibrated" else ""
        rows = [
            BudgetRow(
                "Ideal FID amplitude",
                f"{amplitude:.4g}{unit}",
                "initial plus delivered magnetization, ideal 90°",
                amplitude,
            )
        ]
        if noise.mode == "calibrated":
            target = calibration.target_thermal_polarization
            rows.append(
                BudgetRow(
                    "Fully polarized equivalent",
                    format_amount(amplitude * target),
                    "moles × polarization of the object",
                    amplitude * target,
                )
            )
        else:
            rows.append(
                BudgetRow(
                    "Noise σ at reference bandwidth",
                    f"{sigma_reference:.4g}",
                    "per quadrature, one scan: amplitude / relative SNR",
                    sigma_reference,
                )
            )
        rows.append(
            BudgetRow(
                "Whole-object FID SNR",
                _number(ceiling),
                "ideal 90° at the reference bandwidth, one scan: the ceiling",
                ceiling,
            )
        )
        object_rows = tuple(rows)
    sections.append(BudgetSection("Simulated object", object_rows))

    if program is None:
        readout = (BudgetRow("Build a sequence to see its readout bandwidths."),)
    else:
        readout = _readout_rows(program, noise, amplitude, ceiling)
    sections.append(BudgetSection("Sequence readout", readout))

    shares = None
    if grid is not None and amplitude_map is not None:
        matrix, fov_m, frame = grid
        try:
            cells, outside = grid_voxel_amplitudes(
                amplitude_map,
                phantom.positions,
                matrix,
                fov_m,
                None if frame is None else frame.matrix,
            )
            shares = voxel_shares(cells, amplitude, outside)
            voxels = _voxel_rows(grid, shares, sigma_reference)
        except (TypeError, ValueError) as exc:
            voxels = (BudgetRow(str(exc)),)
        sections.append(BudgetSection("Image voxels", voxels))
    elif grid is None and program is not None:
        sections.append(
            _message(
                "Image voxels",
                "This sequence declares no spatial encoding grid; voxel SNR is "
                "not available.",
            )
        )
    sections.append(BudgetSection("Last run", _last_run_rows(result, noise, shares)))
    return tuple(sections)
