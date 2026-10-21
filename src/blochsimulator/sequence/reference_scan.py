"""Signal model of the phantom FID scan behind a receiver calibration.

A measured calibration FID is rarely the ideal fully relaxed 90-degree
amplitude at t=0 in a homogeneous coil. :func:`reference_signal_efficiency`
predicts the measured amplitude relative to that ideal, so the calibration
can be read back as the scanner sensitivity per mol of polarized nuclei.

The model assumes spoiled repetitions (no transverse coherence left after a
TR, i.e. TR much longer than T2*), a fully relaxed sample before the first
dummy scan and instantaneous excitation. With a coil model the effective flip
angle follows |B1+|, while the signal of every sample point carries the
transmit phase conjugate and the complex B1- weight, as in the simulator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np


_GEOMETRIES = ("sphere", "cylinder")


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _optional_positive(value, name):
    return None if value is None else _positive(value, name)


@dataclass(frozen=True)
class ReferenceScan:
    """Acquisition, relaxation and placement of the calibration phantom.

    ``t1_s`` None means full recovery between repetitions and ``t2star_s``
    None means no decay before the point at ``amplitude_delay_s`` after the
    excitation whose amplitude was measured. ``dummy_scans`` precede the
    ``averages`` acquired repetitions at ``repetition_time_s``.
    ``equilibrium_fraction`` is the Mz that T1 recovers to relative to the
    reference polarization (1 for a thermal reference).

    The sample is a sphere or a cylinder along z of ``volume_ml`` centred at
    ``position_m`` in object space; a cylinder needs ``diameter_m``.
    """

    volume_ml: float
    flip_angle_deg: float = 90.0
    averages: int = 1
    dummy_scans: int = 0
    repetition_time_s: Optional[float] = None
    t1_s: Optional[float] = None
    t2star_s: Optional[float] = None
    amplitude_delay_s: float = 0.0
    equilibrium_fraction: float = 1.0
    geometry: str = "sphere"
    diameter_m: Optional[float] = None
    position_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self):
        _positive(self.volume_ml, "Reference volume")
        if not 0 < self.flip_angle_deg < 180:
            raise ValueError("Reference flip angle must be between 0 and 180 degrees")
        for name, label, minimum in (
            ("averages", "Averages", 1),
            ("dummy_scans", "Dummy scans", 0),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < minimum:
                raise ValueError(f"{label} must be an integer of at least {minimum}")
        _optional_positive(self.t1_s, "Reference T1")
        _optional_positive(self.t2star_s, "Reference T2*")
        _optional_positive(self.repetition_time_s, "Reference TR")
        if self.t1_s is not None and self.repetitions > 1:
            if self.repetition_time_s is None:
                raise ValueError("Enter the reference TR to model T1 saturation")
        delay = float(self.amplitude_delay_s)
        if not np.isfinite(delay) or delay < 0:
            raise ValueError("Amplitude delay must be non-negative and finite")
        fraction = float(self.equilibrium_fraction)
        if not np.isfinite(fraction) or fraction < 0:
            raise ValueError("Equilibrium fraction must be non-negative and finite")
        if self.geometry not in _GEOMETRIES:
            raise ValueError("Reference geometry must be 'sphere' or 'cylinder'")
        if self.geometry == "cylinder":
            _positive(
                0.0 if self.diameter_m is None else self.diameter_m,
                "Cylinder diameter",
            )
        position = np.asarray(self.position_m, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("Reference position must contain three finite values")
        object.__setattr__(self, "position_m", tuple(float(v) for v in position))

    @property
    def repetitions(self):
        return int(self.dummy_scans) + int(self.averages)

    @property
    def length_m(self):
        """Cylinder length that holds the volume, or None for a sphere."""
        if self.geometry != "cylinder":
            return None
        radius = 0.5 * self.diameter_m
        return self.volume_ml * 1e-6 / (np.pi * radius**2)

    def sample_points(self, points_per_axis=24):
        """Equal-volume sample points inside the reference volume, in metres."""
        count = int(points_per_axis)
        if count < 1:
            raise ValueError("points_per_axis must be positive")
        volume = self.volume_ml * 1e-6
        if self.geometry == "sphere":
            radius = (3 * volume / (4 * np.pi)) ** (1 / 3)
            half = np.array([radius, radius, radius])
        else:
            radius = 0.5 * self.diameter_m
            half = np.array([radius, radius, 0.5 * self.length_m])
        # Cell centres of a grid over the bounding box; ``count`` cells across
        # the largest extent, at least one cell along every axis.
        cell = 2 * half.max() / count
        axes = []
        for extent in half:
            cells = max(1, int(np.ceil(2 * extent / cell - 1e-9)))
            axes.append(((np.arange(cells) + 0.5) / cells - 0.5) * 2 * extent)
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        if self.geometry == "sphere":
            inside = np.sum(grid**2, axis=1) <= radius**2
        else:
            inside = grid[:, 0] ** 2 + grid[:, 1] ** 2 <= radius**2
        points = grid[inside]
        if points.size == 0:
            points = np.zeros((1, 3))
        return points + np.asarray(self.position_m)

    def mean_longitudinal(self, flip_rad):
        """Mean Mz before the acquired repetitions, relative to the start.

        ``flip_rad`` may be an array of effective flip angles. Spoiled
        repetitions follow M(n+1) = E1 cos(flip) M(n) + (1 - E1) M_eq.
        """
        flip = np.asarray(flip_rad, dtype=float)
        if self.t1_s is None or self.repetitions == 1:
            return np.ones_like(flip)
        e1 = np.exp(-self.repetition_time_s / self.t1_s)
        q = e1 * np.cos(flip)
        steady = np.divide(
            (1 - e1) * self.equilibrium_fraction,
            1 - q,
            out=np.ones_like(q),
            where=np.abs(1 - q) > 1e-15,
        )
        averages = int(self.averages)
        # Mean of q**n over the acquired repetitions n = D .. D + NA - 1.
        geometric = np.divide(
            1 - q**averages,
            averages * (1 - q),
            out=np.ones_like(q),
            where=np.abs(1 - q) > 1e-15,
        )
        transient = q ** int(self.dummy_scans) * geometric
        return steady + (1 - steady) * transient

    @property
    def decay(self):
        if self.t2star_s is None:
            return 1.0
        return float(np.exp(-self.amplitude_delay_s / self.t2star_s))

    def to_metadata(self):
        values = asdict(self)
        # JSON round trips keep a list, so stored and fresh inputs compare equal.
        values["position_m"] = list(self.position_m)
        return values


@dataclass(frozen=True)
class ReferenceEfficiency:
    """Measured FID amplitude per ideal 90-degree amplitude of the same spins.

    ``homogeneous`` is the value without a coil model; ``coverage`` the volume
    fraction of the reference inside both B1 fields.
    """

    efficiency: float
    homogeneous: float
    coverage: float
    sample_points: int
    coil_model: Optional[str] = None


def _field_values(field, points, kind):
    if field is None:
        return np.ones(points.shape[0], dtype=np.complex128)
    values = field.sample_world(points)
    if values.shape[0] != 1:
        raise ValueError(
            f"The reference scan supports one {kind} channel; the coil model has "
            f"{values.shape[0]}"
        )
    return values[0]


def reference_signal_efficiency(
    scan: ReferenceScan, tx_field=None, rx_field=None, *, points_per_axis=24
):
    """Predict the reference FID amplitude relative to the ideal reference.

    The ideal reference is the fully relaxed 90-degree amplitude at t=0 in a
    coil with unit B1+ and B1-. A homogeneous fully relaxed scan at t=0 gives
    sin(flip). ``tx_field`` and ``rx_field`` are optional
    :class:`~blochsimulator.b1_fields.B1Field` maps; the same fields must be
    applied to the simulated object, because noise is calibrated per unit
    B1- weight.
    """
    nominal = np.deg2rad(scan.flip_angle_deg)
    homogeneous = float(np.sin(nominal) * scan.mean_longitudinal(nominal) * scan.decay)
    if tx_field is None and rx_field is None:
        return ReferenceEfficiency(homogeneous, homogeneous, 1.0, 1, None)
    points = scan.sample_points(points_per_axis)
    tx = _field_values(tx_field, points, "transmit")
    rx = _field_values(rx_field, points, "receive")
    flip = nominal * np.abs(tx)
    signal = (
        np.sin(flip) * scan.mean_longitudinal(flip) * np.exp(-1j * np.angle(tx)) * rx
    )
    efficiency = float(abs(np.mean(signal)) * scan.decay)
    eps = np.finfo(float).eps
    coverage = float(np.mean((np.abs(tx) > eps) & (np.abs(rx) > eps)))
    names = [
        f"{label} {field.name}"
        for label, field in (("Tx", tx_field), ("Rx", rx_field))
        if field is not None
    ]
    return ReferenceEfficiency(
        efficiency, homogeneous, coverage, int(points.shape[0]), ", ".join(names)
    )


def coil_dependent(calibration, rel_tol=1e-9):
    """Whether a calibration's efficiency differs from its homogeneous model.

    Such a calibration only holds for an object carrying the same B1 maps.
    """
    metadata = calibration.reference_scan
    if not metadata or not metadata.get("coil_model"):
        return False
    scan = ReferenceScan(
        **{key: value for key, value in metadata.items() if key != "coil_model"}
    )
    homogeneous = reference_signal_efficiency(scan).efficiency
    return not np.isclose(calibration.signal_efficiency, homogeneous, rtol=rel_tol)
