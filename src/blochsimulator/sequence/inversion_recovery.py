"""Standalone multi-slice inversion-recovery (T1) sequence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .pulseq_builders import (
    _normalized_choice,
    _normalized_slice_order,
    make_pulseq_epi,
    make_pulseq_flash,
)


INVERSION_RECOVERY_READOUTS = ("epi", "gre", "turboflash")
_READOUT_ALIASES = {
    "epi": "epi",
    "gre": "gre",
    "flash": "gre",
    "line": "gre",
    "gre line": "gre",
    "turboflash": "turboflash",
    "turbo flash": "turboflash",
    "mprage": "turboflash",
    "snapshot flash": "turboflash",
}
_DEFAULT_FLIP_ANGLE_DEG = {"epi": 90.0, "gre": 90.0, "turboflash": 10.0}
# Builder-specific arguments the wrapper sets itself.
_RESERVED_PARAMETERS = (
    "inversion_times_s",
    "inversion_mode",
    "inversion_interval",
    "inversion_repetition_time_s",
    "acquisition_interval_s",
    "repetition_time_s",
    "repetitions",
    "slice_order",
    "flip_angle_deg",
    "timing_only",
)


@dataclass(frozen=True)
class InversionRecoveryTimingPlan:
    """Shortest feasible inversion-recovery timing.

    ``minimum_repetition_time_s`` is the shortest inversion-to-inversion
    time and ``minimum_readout_repetition_time_s`` the shortest line TR of
    a gradient-echo readout (``None`` for EPI).
    """

    readout: str
    sampling_bandwidth_hz: float
    dwell_s: float
    minimum_echo_time_s: float
    minimum_readout_repetition_time_s: float | None
    minimum_repetition_time_s: float
    minimum_inversion_time_s: float


def normalized_inversion_recovery_readout(readout) -> str:
    """Return the canonical inversion-recovery readout name."""
    return _normalized_choice(readout, _READOUT_ALIASES, "readout")


def default_inversion_recovery_repetitions(
    inversion_times_s: float | Sequence[float],
    *,
    n_slices: int = 1,
    slice_order: str = "sequential",
) -> int:
    """Return the repetitions that sample every TI once.

    A cyclic slice order repeats the TI series once per slice so that every
    slice is acquired at every position of the slice train.
    """
    count = int(np.atleast_1d(np.asarray(inversion_times_s, dtype=float)).size)
    if _normalized_slice_order(slice_order) == "cyclic":
        count *= max(1, int(n_slices))
    return count


def make_pulseq_inversion_recovery(
    *,
    readout: str = "epi",
    inversion_times_s: float | Sequence[float] = (0.1, 0.4, 1.0, 2.5),
    inversion_mode: str = "global",
    slice_order: str = "sequential",
    repetition_time_s: float | None = 5.0,
    readout_repetition_time_s: float | None = None,
    flip_angle_deg: float | None = None,
    repetitions: int | None = None,
    timing_only: bool = False,
    **readout_parameters,
):
    """Build a multi-slice inversion-recovery sequence for T1 mapping.

    ``readout`` selects the acquisition after each inversion:

    * ``'epi'``: one single-shot EPI image per slice. With the global
      ``inversion_mode`` one non-selective 180° is followed by
      TI - 90° slice 1 - 90° slice 2 ... 90° slice N, so slice k sees TI plus
      k excitation-and-readout durations.
    * ``'gre'``: classic IR with one gradient-echo line per slice and
      inversion; the inversion repeats for every phase-encoding line.
    * ``'turboflash'``: the whole low-flip FLASH image follows one inversion
      (MP-RAGE-like); TI targets the k-space centre line.

    ``inversion_mode='slice_selective'`` inverts each slice right before its
    own excitations instead, so every slice sees the requested TI.
    ``repetition_time_s`` is the inversion-to-inversion time (``None`` for
    the shortest) and ``readout_repetition_time_s`` the gradient-echo line TR
    (``None`` for the shortest). ``repetitions`` defaults to one complete TI
    series, or one per slice with the ``'cyclic'`` slice order. Every other
    keyword is passed to :func:`make_pulseq_epi` or :func:`make_pulseq_flash`.

    The definitions record ``SliceInversionTimes`` for every repetition and
    slice; :func:`slice_inversion_times` reshapes them for
    :func:`fit_inversion_recovery_t1`.
    """
    readout = normalized_inversion_recovery_readout(readout)
    reserved = sorted(set(readout_parameters) & set(_RESERVED_PARAMETERS))
    if reserved:
        raise TypeError(
            "make_pulseq_inversion_recovery sets these arguments itself: "
            + ", ".join(reserved)
        )
    if inversion_times_s is None:
        raise ValueError("inversion_times_s must contain at least one value")
    if flip_angle_deg is None:
        flip_angle_deg = _DEFAULT_FLIP_ANGLE_DEG[readout]
    if repetitions is None:
        repetitions = default_inversion_recovery_repetitions(
            inversion_times_s,
            n_slices=readout_parameters.get("n_slices", 1),
            slice_order=slice_order,
        )
    parameters = dict(
        readout_parameters,
        inversion_times_s=inversion_times_s,
        inversion_mode=inversion_mode,
        slice_order=slice_order,
        flip_angle_deg=float(flip_angle_deg),
        repetitions=int(repetitions),
    )

    if readout == "epi":
        if timing_only:
            plan = make_pulseq_epi(
                **parameters, repetition_time_s=repetition_time_s, timing_only=True
            )
            return InversionRecoveryTimingPlan(
                readout=readout,
                sampling_bandwidth_hz=plan.sampling_bandwidth_hz,
                dwell_s=plan.dwell_s,
                minimum_echo_time_s=plan.minimum_echo_time_s,
                minimum_readout_repetition_time_s=None,
                minimum_repetition_time_s=plan.minimum_repetition_time_s,
                minimum_inversion_time_s=float(plan.minimum_inversion_time_s),
            )
        sequence = make_pulseq_epi(**parameters, repetition_time_s=repetition_time_s)
    else:
        if readout == "gre":
            parameters.update(
                inversion_interval="line",
                inversion_repetition_time_s=repetition_time_s,
            )
        else:
            parameters.update(
                inversion_interval="image", acquisition_interval_s=repetition_time_s
            )
        line_tr = readout_repetition_time_s
        if line_tr is None or timing_only:
            # Timing-only mode accepts a too-short line TR and reports the
            # shortest one; the inversion timing then uses that TR.
            plan = make_pulseq_flash(
                **parameters,
                repetition_time_s=1e-6 if line_tr is None else float(line_tr),
                timing_only=True,
            )
            if timing_only:
                return InversionRecoveryTimingPlan(
                    readout=readout,
                    sampling_bandwidth_hz=plan.sampling_bandwidth_hz,
                    dwell_s=plan.dwell_s,
                    minimum_echo_time_s=plan.minimum_echo_time_s,
                    minimum_readout_repetition_time_s=plan.minimum_repetition_time_s,
                    minimum_repetition_time_s=float(
                        plan.minimum_inversion_repetition_time_s
                        if readout == "gre"
                        else plan.minimum_acquisition_interval_s
                    ),
                    minimum_inversion_time_s=float(plan.minimum_inversion_time_s),
                )
            line_tr = plan.minimum_repetition_time_s
        sequence = make_pulseq_flash(**parameters, repetition_time_s=float(line_tr))

    sequence.set_definition("Name", f"ir_{readout}_2d")
    sequence.set_definition("SequenceFamily", "inversion_recovery")
    sequence.set_definition("InversionReadout", readout)
    return sequence
