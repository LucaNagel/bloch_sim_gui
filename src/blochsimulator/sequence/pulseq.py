"""Optional Pulseq 1.5.0 importer backed by public PyPulseq APIs."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import warnings

import numpy as np

from .model import ADCEvent, GradientEvent, RFEvent, SequenceProgram


class PulseqImportError(ValueError):
    """Raised when a Pulseq file cannot be represented safely."""


class UnsupportedPulseqVersionError(PulseqImportError):
    """Raised for Pulseq formats newer than the supported 1.5.0 format."""


def load_pulseq(path, strict: bool = True) -> SequenceProgram:
    """Load a Pulseq text file up to format version 1.5.0.

    Parameters
    ----------
    path : str or Path
        Pulseq ``.seq`` file.
    strict : bool
        Reject unsupported soft delays and malformed event data. Labels are
        retained as metadata because they do not alter Bloch physics.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    declared_version = _read_text_version(file_path)
    if declared_version is None and strict:
        raise PulseqImportError("Pulseq file has no readable [VERSION] section")
    if declared_version is not None and declared_version > (1, 5, 0):
        version_text = ".".join(str(value) for value in declared_version)
        raise UnsupportedPulseqVersionError(
            f"Pulseq {version_text} is not supported; maximum supported format is 1.5.0"
        )

    try:
        import pypulseq as pp
    except ImportError as exc:
        raise ImportError(
            "Pulseq import requires the optional dependency: "
            "pip install 'blochsimulator[pulseq]'"
        ) from exc

    sequence = pp.Sequence()
    try:
        sequence.read(str(file_path), detect_rf_use=False, remove_duplicates=True)
    except Exception as exc:
        raise PulseqImportError(
            f"could not read Pulseq file {file_path}: {exc}"
        ) from exc

    parsed_version = (
        int(sequence.version_major),
        int(sequence.version_minor),
        int(sequence.version_revision),
    )
    if parsed_version > (1, 5, 0):
        raise UnsupportedPulseqVersionError(
            f"Pulseq {parsed_version[0]}.{parsed_version[1]}.{parsed_version[2]} "
            "is not supported; maximum supported format is 1.5.0"
        )

    events = []
    labels = []
    triggers = []
    import_warnings: List[str] = []
    block_start = 0.0
    gradient_raster = float(sequence.system.grad_raster_time)
    rf_raster = float(sequence.system.rf_raster_time)
    gamma_b0 = float(sequence.system.gamma * sequence.system.B0)

    for block_index in sequence.block_events.keys():
        block = sequence.get_block(block_index)
        block_duration = float(block.block_duration)
        if getattr(block, "soft_delay", None) is not None:
            message = f"block {block_index} contains a Pulseq soft delay"
            if strict:
                raise PulseqImportError(
                    message + "; soft delays require Pulseq 1.5.1 support"
                )
            warnings.warn(message + "; ignoring it", RuntimeWarning, stacklevel=2)
            import_warnings.append(message)

        label = getattr(block, "label", None)
        if label:
            labels.append({"block": int(block_index), "label": label})
        trigger = getattr(block, "trigger", None)
        if trigger:
            triggers.append(
                {
                    "block": int(block_index),
                    "trigger": {
                        int(key): vars(value).copy() for key, value in trigger.items()
                    },
                }
            )
            message = f"block {block_index} trigger retained as metadata and ignored"
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            import_warnings.append(message)

        known_block_fields = {
            "block_duration",
            "rf",
            "gx",
            "gy",
            "gz",
            "adc",
            "label",
            "soft_delay",
            "trigger",
        }
        unknown_fields = {
            key
            for key, value in vars(block).items()
            if key not in known_block_fields and value is not None
        }
        if unknown_fields:
            message = (
                f"block {block_index} contains unsupported extensions: "
                + ", ".join(sorted(unknown_fields))
            )
            if strict:
                raise PulseqImportError(message)
            warnings.warn(message + "; ignoring them", RuntimeWarning, stacklevel=2)
            import_warnings.append(message)

        rf = getattr(block, "rf", None)
        if rf is not None:
            signal = np.asarray(rf.signal, dtype=np.complex128)
            times = np.asarray(rf.t, dtype=float)
            if signal.ndim != 1 or signal.size == 0 or times.shape != signal.shape:
                raise PulseqImportError(f"block {block_index} has malformed RF samples")
            regular_centres = (
                signal.size > 1
                and np.allclose(np.diff(times), rf_raster, rtol=1e-8, atol=1e-15)
                and np.isclose(times[0], rf_raster / 2, atol=1e-15)
            )
            if regular_centres:
                rf_samples = signal
            else:
                rf_samples = _rasterize_linear(times, signal, rf_raster)
            full_frequency = float(rf.freq_offset + rf.freq_ppm * 1e-6 * gamma_b0)
            full_phase = float(rf.phase_offset + rf.phase_ppm * 1e-6 * gamma_b0)
            events.append(
                RFEvent(
                    start_s=block_start + float(rf.delay),
                    samples_hz=rf_samples,
                    raster_s=rf_raster,
                    frequency_offset_hz=full_frequency,
                    phase_offset_rad=(
                        full_phase + 2 * np.pi * full_frequency * rf_raster / 2
                    ),
                )
            )

        for axis_name in ("gx", "gy", "gz"):
            gradient = getattr(block, axis_name, None)
            if gradient is None:
                continue
            samples = _gradient_samples(gradient, gradient_raster, block_index)
            events.append(
                GradientEvent(
                    axis=axis_name[-1],
                    start_s=block_start + float(gradient.delay),
                    samples_hz_per_m=samples,
                    raster_s=gradient_raster,
                )
            )

        adc = getattr(block, "adc", None)
        if adc is not None:
            full_frequency = float(adc.freq_offset + adc.freq_ppm * 1e-6 * gamma_b0)
            full_phase = float(adc.phase_offset + adc.phase_ppm * 1e-6 * gamma_b0)
            # PyPulseq defines ADC samples at dwell centres.
            first_sample = block_start + float(adc.delay) + float(adc.dwell) / 2
            events.append(
                ADCEvent(
                    start_s=first_sample,
                    num_samples=int(adc.num_samples),
                    dwell_s=float(adc.dwell),
                    frequency_offset_hz=full_frequency,
                    phase_offset_rad=full_phase,
                )
            )

        block_start += block_duration

    duration = float(sequence.duration()[0])
    if not np.isclose(block_start, duration, rtol=0.0, atol=1e-12):
        raise PulseqImportError(
            f"Pulseq block duration sum {block_start} differs from sequence duration {duration}"
        )
    version = parsed_version if declared_version is None else declared_version
    return SequenceProgram(
        events=tuple(events),
        duration_s=duration,
        source=str(file_path),
        version=".".join(str(value) for value in version),
        metadata={
            "format": "pulseq",
            "definitions": dict(sequence.definitions),
            "labels": labels,
            "triggers": triggers,
            "warnings": import_warnings,
            "blocks": len(sequence.block_events),
        },
    )


def _gradient_samples(gradient, raster: float, block_index: int) -> np.ndarray:
    if gradient.type == "trap":
        rise = float(gradient.rise_time)
        flat = float(gradient.flat_time)
        fall = float(gradient.fall_time)
        amplitude = float(gradient.amplitude)
        times = [0.0]
        amplitudes = [0.0 if rise > 0 else amplitude]
        if rise > 0:
            times.append(rise)
            amplitudes.append(amplitude)
        if flat > 0:
            times.append(rise + flat)
            amplitudes.append(amplitude)
        if fall > 0:
            times.append(rise + flat + fall)
            amplitudes.append(0.0)
        times = np.asarray(times)
        amplitudes = np.asarray(amplitudes)
    elif gradient.type == "grad":
        waveform = np.asarray(gradient.waveform, dtype=float)
        times_raw = np.asarray(gradient.tt, dtype=float)
        if waveform.ndim != 1 or times_raw.shape != waveform.shape:
            raise PulseqImportError(
                f"block {block_index} has malformed gradient samples"
            )
        raster_positions = times_raw / raster + 0.5
        regular = np.allclose(
            raster_positions,
            np.arange(1, len(raster_positions) + 1),
            rtol=0.0,
            atol=1e-8,
        )
        if regular:
            times = np.concatenate(([0.0], times_raw, [times_raw[-1] + raster / 2]))
            amplitudes = np.concatenate(
                ([float(gradient.first)], waveform, [float(gradient.last)])
            )
        else:
            times = times_raw
            amplitudes = waveform
    else:
        raise PulseqImportError(
            f"block {block_index} has unsupported gradient type {gradient.type!r}"
        )
    return _rasterize_linear(times, amplitudes, raster)


def _rasterize_linear(
    times: np.ndarray, amplitudes: np.ndarray, raster: float
) -> np.ndarray:
    if times.size < 2 or np.any(np.diff(times) <= 0):
        raise PulseqImportError("gradient time points must be strictly increasing")
    duration = float(times[-1])
    sample_count = int(round(duration / raster))
    if sample_count <= 0 or not np.isclose(sample_count * raster, duration, atol=1e-12):
        raise PulseqImportError(
            "gradient duration is not aligned to the gradient raster"
        )
    result = np.empty(sample_count, dtype=np.result_type(amplitudes.dtype, float))
    for index in range(sample_count):
        start = index * raster
        end = start + raster
        internal = times[(times > start) & (times < end)]
        points = np.concatenate(([start], internal, [end]))
        values = np.interp(points, times, amplitudes)
        result[index] = np.trapezoid(values, points) / raster
    return result


def _read_text_version(path: Path):
    major = minor = revision = None
    in_version = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith("["):
                if in_version:
                    break
                in_version = line.upper() == "[VERSION]"
                continue
            if not in_version:
                continue
            fields = line.split()
            if len(fields) != 2:
                continue
            key, value = fields[0].lower(), fields[1]
            if key == "major":
                major = int(value)
            elif key == "minor":
                minor = int(value)
            elif key == "revision":
                revision = int(value)
    if None in (major, minor, revision):
        return None
    return major, minor, revision
