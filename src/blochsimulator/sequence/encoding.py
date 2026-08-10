"""Logical MRI encoding coordinates mapped to physical scanner axes.

Sequence builders describe gradients in the role order ``read``, ``phase``,
``partition``.  :class:`EncodingFrame` maps those logical coordinates to the
physical scanner ``x``, ``y``, ``z`` coordinate system used by Pulseq and the
Bloch kernel.  Keeping this mapping explicit prevents acquisition and display
code from silently assuming that read/phase/partition always mean x/y/z.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


SCANNER_AXES = ("x", "y", "z")
ENCODING_ROLES = ("read", "phase", "partition")
_AXIS_VECTORS = {
    "x": np.asarray((1.0, 0.0, 0.0)),
    "y": np.asarray((0.0, 1.0, 0.0)),
    "z": np.asarray((0.0, 0.0, 1.0)),
}


def numeric_definition_array(value, name: str) -> np.ndarray:
    """Normalize numeric Pulseq definitions stored as arrays or array strings."""
    if isinstance(value, str):
        text = value.strip()
        if text.lower().startswith("array(") and text.endswith(")"):
            text = text[6:-1].strip()
        text = text.strip("[]()")
        tokens = text.replace(",", " ").replace(";", " ").split()
        try:
            result = np.asarray([float(token) for token in tokens], dtype=float)
        except ValueError as exc:
            raise ValueError(f"{name} must contain numeric values") from exc
    else:
        try:
            result = np.asarray(value, dtype=float).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain numeric values") from exc
    return result.reshape(-1)


def _axis_vector(code: str, name: str) -> np.ndarray:
    normalized = str(code).strip().lower()
    sign = 1.0
    if normalized.startswith(("+", "-")):
        sign = -1.0 if normalized[0] == "-" else 1.0
        normalized = normalized[1:]
    if normalized not in _AXIS_VECTORS:
        raise ValueError(f"{name} must be one of +x, -x, +y, -y, +z, -z")
    return sign * _AXIS_VECTORS[normalized]


def _axis_code(vector: np.ndarray) -> str:
    values = np.asarray(vector, dtype=float)
    nonzero = np.flatnonzero(np.abs(values) > 1e-9)
    if nonzero.size != 1 or not np.isclose(abs(values[nonzero[0]]), 1.0):
        raise ValueError("encoding direction is not aligned with one scanner axis")
    axis = SCANNER_AXES[int(nonzero[0])]
    sign = "+" if values[nonzero[0]] > 0 else "-"
    return sign + axis


@dataclass(frozen=True)
class EncodingFrame:
    """Right-handed orthonormal encoding basis expressed in scanner xyz.

    Columns of ``basis_xyz`` are respectively the read, phase, and partition
    unit vectors.  The full matrix representation deliberately supports future
    oblique acquisitions, while the current Pulseq builders use signed
    axis-aligned frames so each logical gradient remains one physical event.
    """

    basis_xyz: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    def __post_init__(self) -> None:
        basis = np.asarray(self.basis_xyz, dtype=float)
        if basis.shape != (3, 3) or not np.all(np.isfinite(basis)):
            raise ValueError("encoding basis must be a finite 3x3 matrix")
        if not np.allclose(basis.T @ basis, np.eye(3), rtol=0.0, atol=1e-9):
            raise ValueError("encoding basis directions must be orthonormal")
        determinant = float(np.linalg.det(basis))
        if not np.isclose(determinant, 1.0, rtol=0.0, atol=1e-9):
            raise ValueError("encoding basis must be right-handed")
        object.__setattr__(
            self,
            "basis_xyz",
            tuple(tuple(float(value) for value in row) for row in basis),
        )

    @classmethod
    def identity(cls) -> "EncodingFrame":
        return cls()

    @classmethod
    def from_axis_codes(cls, axes: Sequence[str]) -> "EncodingFrame":
        """Create a signed axis-aligned frame in read/phase/partition order."""
        values = tuple(axes)
        if len(values) != 3:
            raise ValueError("encoding_axes must contain read, phase, and partition")
        directions = [
            _axis_vector(value, f"{role} axis")
            for role, value in zip(ENCODING_ROLES, values)
        ]
        return cls(tuple(tuple(row) for row in np.column_stack(directions)))

    @classmethod
    def from_read_phase_axes(
        cls,
        read_axis: str,
        phase_axis: str,
    ) -> "EncodingFrame":
        """Create a right-handed frame and derive partition as read × phase."""
        read = _axis_vector(read_axis, "read axis")
        phase = _axis_vector(phase_axis, "phase axis")
        if not np.isclose(float(read @ phase), 0.0, atol=1e-12):
            raise ValueError("read and phase axes must be different")
        partition = np.cross(read, phase)
        return cls(
            tuple(tuple(row) for row in np.column_stack((read, phase, partition)))
        )

    @classmethod
    def from_metadata(cls, metadata: Mapping) -> "EncodingFrame":
        if not isinstance(metadata, Mapping):
            raise TypeError("encoding frame metadata must be a mapping")
        kind = str(metadata.get("type", "encoding_frame"))
        if kind != "encoding_frame":
            raise ValueError(f"unsupported encoding frame type {kind!r}")
        return cls(
            tuple(
                tuple(float(value) for value in row)
                for row in metadata.get("basis_xyz", np.eye(3))
            )
        )

    @classmethod
    def from_definitions(cls, definitions: Mapping | None) -> "EncodingFrame":
        """Restore generated-sequence orientation, defaulting to legacy xyz."""
        normalized = {
            str(key).lower(): value for key, value in dict(definitions or {}).items()
        }
        basis_value = normalized.get("encodingbasisxyz")
        if basis_value is not None:
            basis = numeric_definition_array(basis_value, "EncodingBasisXYZ")
            if basis.size != 9:
                raise ValueError("EncodingBasisXYZ must contain nine values")
            return cls(tuple(tuple(row) for row in basis.reshape(3, 3)))

        vector_keys = (
            "readoutdirectionxyz",
            "phaseencodingdirectionxyz",
            "partitionencodingdirectionxyz",
        )
        if all(key in normalized for key in vector_keys):
            directions = []
            for key in vector_keys:
                vector = numeric_definition_array(normalized[key], key)
                if vector.size != 3:
                    raise ValueError(f"{key} must contain three values")
                directions.append(vector)
            return cls(tuple(tuple(row) for row in np.column_stack(directions)))

        axis_keys = ("readoutaxis", "phaseencodingaxis", "partitionencodingaxis")
        if all(key in normalized for key in axis_keys):
            return cls.from_axis_codes(tuple(str(normalized[key]) for key in axis_keys))
        return cls.identity()

    @property
    def matrix(self) -> np.ndarray:
        value = np.asarray(self.basis_xyz, dtype=float)
        value.setflags(write=False)
        return value

    @property
    def is_axis_aligned(self) -> bool:
        absolute = np.abs(self.matrix)
        return bool(
            np.allclose(absolute.sum(axis=0), 1.0, atol=1e-9)
            and np.allclose(absolute.sum(axis=1), 1.0, atol=1e-9)
            and np.allclose(absolute, np.rint(absolute), atol=1e-9)
        )

    @property
    def axis_codes(self) -> tuple[str, str, str]:
        return tuple(_axis_code(self.matrix[:, index]) for index in range(3))

    def direction(self, role: str) -> np.ndarray:
        normalized = str(role).strip().lower()
        try:
            index = ENCODING_ROLES.index(normalized)
        except ValueError as exc:
            raise ValueError(
                f"encoding role must be one of {', '.join(ENCODING_ROLES)}"
            ) from exc
        result = np.asarray(self.matrix[:, index], dtype=float)
        result.setflags(write=False)
        return result

    def axis_and_sign(self, role: str) -> tuple[str, int]:
        code = _axis_code(self.direction(role))
        return code[-1], -1 if code.startswith("-") else 1

    def dimension_name(self, role: str) -> str:
        axis, _ = self.axis_and_sign(role)
        return f"{str(role).strip().lower()}_{axis}"

    def encoding_to_scanner(self, values) -> np.ndarray:
        """Map vectors ending in (read, phase, partition) to scanner xyz."""
        array = np.asarray(values, dtype=float)
        if array.shape[-1] != 3:
            raise ValueError("encoding vectors must end with three components")
        return array @ self.matrix.T

    def scanner_to_encoding(self, values) -> np.ndarray:
        """Project scanner xyz vectors onto read, phase, and partition."""
        array = np.asarray(values, dtype=float)
        if array.shape[-1] != 3:
            raise ValueError("scanner vectors must end with three components")
        return array @ self.matrix

    def required_encoding_extents(self, scanner_extents: Sequence[float]) -> np.ndarray:
        """Bounding-box extent required along each encoding direction."""
        extents = np.asarray(tuple(scanner_extents), dtype=float)
        if extents.shape != (3,) or not np.all(np.isfinite(extents)):
            raise ValueError("scanner extents must contain three finite values")
        if np.any(extents < 0):
            raise ValueError("scanner extents must be non-negative")
        return np.abs(self.matrix).T @ extents

    def to_metadata(self) -> dict:
        return {
            "type": "encoding_frame",
            "basis_xyz": self.basis_xyz,
            "axis_codes": self.axis_codes if self.is_axis_aligned else (),
        }

    def pulseq_definitions(self) -> dict:
        """Return JSON/Pulseq-compatible orientation definitions."""
        basis = self.matrix
        definitions = {
            "EncodingBasisXYZ": [float(value) for value in basis.reshape(-1)],
            "ReadoutDirectionXYZ": [float(value) for value in basis[:, 0]],
            "PhaseEncodingDirectionXYZ": [float(value) for value in basis[:, 1]],
            "PartitionEncodingDirectionXYZ": [float(value) for value in basis[:, 2]],
            "EncodingCoordinateSystem": "read-phase-partition",
        }
        if self.is_axis_aligned:
            read, phase, partition = self.axis_codes
            definitions.update(
                {
                    "ReadoutAxis": read,
                    "PhaseEncodingAxis": phase,
                    "PartitionEncodingAxis": partition,
                }
            )
        return definitions


def resolve_encoding_frame(
    encoding_axes: Sequence[str] | EncodingFrame | None,
) -> EncodingFrame:
    if encoding_axes is None:
        return EncodingFrame.identity()
    if isinstance(encoding_axes, EncodingFrame):
        return encoding_axes
    return EncodingFrame.from_axis_codes(encoding_axes)


def make_role_trapezoid(pp, frame: EncodingFrame, role: str, **kwargs):
    """Create one axis-aligned Pulseq trapezoid from logical role values."""
    axis, sign = frame.axis_and_sign(role)
    transformed = dict(kwargs)
    for key in ("area", "flat_area", "amplitude"):
        if key in transformed:
            transformed[key] = sign * transformed[key]
    return pp.make_trapezoid(axis, **transformed)


def logical_gradient_area(event, frame: EncodingFrame, role: str) -> float:
    """Return a physical Pulseq event's signed area in logical coordinates."""
    axis, sign = frame.axis_and_sign(role)
    if str(event.channel) != axis:
        raise ValueError(f"gradient event is on {event.channel}, expected {axis}")
    return float(event.area) * sign


def set_pulseq_encoding_definitions(
    sequence,
    frame: EncodingFrame,
    *,
    fov_m: Sequence[float] | None = None,
    matrix: Sequence[int] | None = None,
) -> None:
    for name, value in frame.pulseq_definitions().items():
        sequence.set_definition(name, value)
    if fov_m is not None:
        sequence.set_definition("EncodingFOV", [float(value) for value in fov_m])
    if matrix is not None:
        sequence.set_definition("EncodingMatrixSize", [int(value) for value in matrix])
