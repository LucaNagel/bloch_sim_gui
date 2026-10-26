"""Probe-oriented sequence simulation result containers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np

PROBE_FILE_FORMAT = "blochsimulator-spin-probe"


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


@dataclass(frozen=True)
class SequenceProbeResult:
    """Time-resolved magnetization for explicit position/frequency probes.

    ``magnetization`` has shape ``(time, position, frequency, component)``.
    It is intentionally separate from :class:`SequenceSimulationResult`, which
    represents ADC-centric imaging output.
    """

    time_s: np.ndarray
    positions_m: np.ndarray
    frequency_offsets_hz: np.ndarray
    magnetization: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def mx(self) -> np.ndarray:
        return self.magnetization[..., 0]

    @property
    def my(self) -> np.ndarray:
        return self.magnetization[..., 1]

    @property
    def mz(self) -> np.ndarray:
        return self.magnetization[..., 2]

    @property
    def mxy(self) -> np.ndarray:
        return self.mx + 1j * self.my

    @property
    def coherent_mxy(self) -> np.ndarray:
        """Complex ensemble mean over positions, retaining dephasing effects."""
        return np.mean(self.mxy, axis=1)

    @property
    def coherent_mxy_magnitude(self) -> np.ndarray:
        """Magnitude of the complex position ensemble, not mean spin magnitude."""
        return np.abs(self.coherent_mxy)

    def to_xarray(self):
        """Return a coordinate-aware xarray representation."""
        import xarray as xr

        return xr.Dataset(
            data_vars={
                "magnetization": (
                    ("time", "position", "frequency", "component"),
                    self.magnetization,
                ),
                "mx": (("time", "position", "frequency"), self.mx),
                "my": (("time", "position", "frequency"), self.my),
                "mz": (("time", "position", "frequency"), self.mz),
                "mxy_magnitude": (
                    ("time", "position", "frequency"),
                    np.abs(self.mxy),
                ),
                "coherent_mxy_real": (
                    ("time", "frequency"),
                    np.real(self.coherent_mxy),
                ),
                "coherent_mxy_imag": (
                    ("time", "frequency"),
                    np.imag(self.coherent_mxy),
                ),
                "coherent_mxy_magnitude": (
                    ("time", "frequency"),
                    self.coherent_mxy_magnitude,
                ),
            },
            coords={
                "time": ("time", self.time_s, {"units": "s"}),
                "position": np.arange(self.positions_m.shape[0]),
                "frequency": (
                    "frequency",
                    self.frequency_offsets_hz,
                    {"units": "Hz"},
                ),
                "component": ["mx", "my", "mz"],
                "x": ("position", self.positions_m[:, 0], {"units": "m"}),
                "y": ("position", self.positions_m[:, 1], {"units": "m"}),
                "z": ("position", self.positions_m[:, 2], {"units": "m"}),
            },
            attrs=dict(self.metadata),
        )

    def save(self, filename) -> Path:
        """Save the probe as xarray NetCDF (``.nc``) or NumPy archive (``.npz``).

        The complete metadata, including a parameter-loop description, is
        stored as JSON so :meth:`load` restores it unchanged.
        """
        path = Path(filename)
        suffix = path.suffix.lower()
        metadata_json = json.dumps(self.metadata, default=_json_default)
        if suffix == ".npz":
            np.savez_compressed(
                path,
                time_s=self.time_s,
                positions_m=self.positions_m,
                frequency_offsets_hz=self.frequency_offsets_hz,
                magnetization=self.magnetization,
                metadata_json=np.asarray(metadata_json),
                format=np.asarray(PROBE_FILE_FORMAT),
            )
            return path
        if suffix != ".nc":
            raise ValueError("spin probes are saved as .nc or .npz")
        dataset = self.to_xarray()
        # NetCDF attributes only hold scalars and flat arrays.
        attrs = {
            key: value
            for key, value in self.metadata.items()
            if isinstance(value, (str, int, float, bool))
        }
        attrs["metadata_json"] = metadata_json
        attrs["format"] = PROBE_FILE_FORMAT
        dataset.attrs = attrs
        dataset.to_netcdf(path)
        return path

    @classmethod
    def load(cls, filename) -> "SequenceProbeResult":
        """Load a probe written by :meth:`save`."""
        path = Path(filename)
        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as data:
                if "format" not in data or str(data["format"]) != PROBE_FILE_FORMAT:
                    raise ValueError(f"{path.name} is not a spin-probe export")
                return cls(
                    time_s=np.asarray(data["time_s"]),
                    positions_m=np.asarray(data["positions_m"]),
                    frequency_offsets_hz=np.asarray(data["frequency_offsets_hz"]),
                    magnetization=np.asarray(data["magnetization"]),
                    metadata=json.loads(str(data["metadata_json"])),
                )
        import xarray as xr

        with xr.open_dataset(path) as dataset:
            if dataset.attrs.get("format") != PROBE_FILE_FORMAT:
                raise ValueError(f"{path.name} is not a spin-probe export")
            positions = np.stack(
                [dataset[axis].values for axis in ("x", "y", "z")], axis=-1
            )
            return cls(
                time_s=np.asarray(dataset["time"].values, dtype=float),
                positions_m=np.asarray(positions, dtype=float),
                frequency_offsets_hz=np.asarray(
                    dataset["frequency"].values, dtype=float
                ),
                magnetization=np.asarray(
                    dataset["magnetization"]
                    .transpose("time", "position", "frequency", "component")
                    .values
                ),
                metadata=json.loads(dataset.attrs.get("metadata_json", "{}")),
            )
