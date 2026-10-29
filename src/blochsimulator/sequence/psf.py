"""Point-spread functions and slice profiles of simulated sequences.

The PSF of a sequence is the reconstructed image of a point object. Bloch
signals are linear in proton density and isochromats do not interact, so one
isochromat at ``r0`` simulated through the real sequence yields the
shift-variant PSF at ``r0``. It includes relaxation during the readout,
variable flip angles, hyperpolarized depletion along the k-space order,
off-resonance, and the approach to steady state. ``T2*`` is modelled with a
Lorentzian frequency ensemble per source.

Several sources share one simulation: each one is received by its own
virtual coil with unit sensitivity, which keeps the source signals apart.

For Cartesian frames and volumes the reconstruction is the continuous
inverse DFT over the acquired k-space grid. At the voxel centres it equals
the sequence's centred-FFT reconstruction; in between it samples the same
band-limited image finely. 2D spiral frames are reconstructed by exact
density-compensated gridding, ``sum_j w_j s_j exp(2 pi i k_j . x)``, with the
Pipe-Menon weights of the NUFFT reconstruction.

Gradient errors (a per-sample deviation ``dk`` of the actual from the
nominal trajectory, see :func:`.kspace_trajectory.gradient_error`) multiply
a point source's signal by ``exp(-2 pi i dk . r0)``; this is exact for a
point. The image is then reconstructed on the nominal trajectory
(uncorrected) or on the actual one (trajectory-corrected, 2D only).
All lengths are in metres, frequencies in Hz, and times in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from .acquisition import AcquisitionDimensions, CartesianAcquisitionVolumes
from .encoding import EncodingFrame, numeric_definition_array

# Each virtual coil accumulates every spin of its run, so run cost grows with
# the square of the source count; batches keep long sweeps linear.
MAX_SOURCES_PER_RUN = 64
# Bound the PSF image of a 3D volume to about 4 M complex points (64 MB).
MAX_PSF_IMAGE_POINTS = 1 << 22
# Bound non-uniform evaluations (image points times samples) to about 1 G
# complex products, a few seconds.
MAX_SAMPLE_IMAGE_PRODUCTS = 1 << 30
MAX_SLICE_PROFILE_POSITIONS = 801
# The T2* ensemble rephases at 12 / R2' (see PointSource.frequency_ensemble).
T2_STAR_REVIVAL_RATE_PRODUCT = 12.0


@dataclass(frozen=True)
class PointSource:
    """Isochromat point object in scanner xyz.

    ``t2_star_s`` adds reversible dephasing ``R2' = 1/T2* - 1/T2`` through a
    Lorentzian ensemble of ``t2_star_spins`` isochromats whose mean signal
    decays as ``exp(-R2' |t|)`` and refocuses in spin echoes; see
    :meth:`frequency_ensemble` for its accuracy.
    ``equilibrium_mz`` is the Mz that T1 recovery approaches in units of the
    initial magnetization; use ``1 / enhancement`` for hyperpolarized spins.
    """

    position_m: tuple = (0.0, 0.0, 0.0)
    t1_s: float = 1e6
    t2_s: float = 1e6
    t2_star_s: Optional[float] = None
    off_resonance_hz: float = 0.0
    equilibrium_mz: float = 1.0
    t2_star_spins: int = 49

    def __post_init__(self) -> None:
        position = tuple(float(value) for value in np.ravel(self.position_m))
        if len(position) != 3 or not np.all(np.isfinite(position)):
            raise ValueError("position_m must contain three finite values")
        for name in ("t1_s", "t2_s"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        if self.t2_star_s is not None:
            t2_star = float(self.t2_star_s)
            if not np.isfinite(t2_star) or t2_star <= 0:
                raise ValueError("t2_star_s must be positive and finite")
            object.__setattr__(self, "t2_star_s", t2_star)
        for name in ("off_resonance_hz", "equilibrium_mz"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        spins = int(self.t2_star_spins)
        if spins != self.t2_star_spins or spins < 1:
            raise ValueError("t2_star_spins must be a positive integer")
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "t2_star_spins", spins)

    @property
    def reversible_rate_per_s(self) -> float:
        """Reversible dephasing rate R2' in 1/s; zero without T2*."""
        if self.t2_star_s is None:
            return 0.0
        return max(0.0, 1.0 / self.t2_star_s - 1.0 / self.t2_s)

    def frequency_ensemble(self) -> tuple[np.ndarray, np.ndarray]:
        """Return isochromat frequencies in Hz and their weights.

        The frequencies are spaced by ``R2' / 12`` (in Hz for R2' in 1/s) around
        ``off_resonance_hz``; each weight is the Lorentzian probability of
        its frequency bin. The far tails outside the outermost bins are left
        out: those spins would dephase within ``0.3 / R2'``, so dropping them
        keeps the mean signal within about 1 % of ``exp(-R2' |t|)`` for
        ``0.3 / R2' <= |t| <= 5 / R2'`` with the default 49 spins. Spin-echo
        peaks reach the sum of the weights (about 0.96) instead of 1, and the
        equispaced ensemble rephases spuriously at ``|t| = 12 / R2'``.
        """
        rate = self.reversible_rate_per_s
        if rate == 0.0 or self.t2_star_spins == 1:
            return np.asarray([self.off_resonance_hz]), np.ones(1)
        count = self.t2_star_spins
        spacing_hz = rate / T2_STAR_REVIVAL_RATE_PRODUCT
        offsets_hz = (np.arange(count, dtype=float) - (count - 1) / 2.0) * spacing_hz
        edges_hz = np.append(
            offsets_hz - spacing_hz / 2.0, offsets_hz[-1] + spacing_hz / 2.0
        )
        half_width_hz = rate / (2.0 * np.pi)
        weights = np.diff(np.arctan(edges_hz / half_width_hz)) / np.pi
        return self.off_resonance_hz + offsets_hz, weights

    def moved_to(self, position_m) -> "PointSource":
        return replace(self, position_m=tuple(np.ravel(position_m)))


def simulate_point_sources(
    program,
    sources: Sequence[PointSource],
    *,
    simulator=None,
    simulation_timestep_s=None,
    sequence_kernel=None,
    spoiler_mode: str = "ideal",
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
):
    """Simulate point sources through ``program`` with separate signals.

    Returns the :class:`SequenceSimulationResult` of the last run with
    ``signal`` replaced by all source signals, shape ``(source, adc)``.
    A point does not dephase across a gradient, so ``spoiler_mode='ideal'``
    is the meaningful default; ``'gradient'`` keeps only gradient phases.
    """
    from ..phantom import Phantom
    from ..simulator import BlochSimulator

    sources = tuple(sources)
    if not sources:
        raise ValueError("at least one point source is required")
    if not all(isinstance(source, PointSource) for source in sources):
        raise TypeError("sources must be PointSource instances")
    if simulator is None:
        simulator = BlochSimulator()

    signals = []
    result = None
    if progress_callback is not None:
        progress_callback(0, len(sources))
    for start in range(0, len(sources), MAX_SOURCES_PER_RUN):
        batch = sources[start : start + MAX_SOURCES_PER_RUN]
        ensembles = [source.frequency_ensemble() for source in batch]
        counts = np.asarray([frequencies.size for frequencies, _ in ensembles])
        n_spins = int(counts.sum())
        owner = np.repeat(np.arange(len(batch)), counts)
        positions = np.repeat(
            np.asarray([source.position_m for source in batch]), counts, axis=0
        )
        receive = np.zeros((len(batch), n_spins), dtype=np.complex128)
        receive[owner, np.arange(n_spins)] = 1.0
        phantom = Phantom(
            shape=(n_spins,),
            fov=(1.0,),
            t1_map=np.repeat([source.t1_s for source in batch], counts),
            t2_map=np.repeat([source.t2_s for source in batch], counts),
            pd_map=np.concatenate([weights for _, weights in ensembles]),
            chemical_shift_map=np.concatenate(
                [frequencies for frequencies, _ in ensembles]
            ),
            rx_sensitivity_maps=receive,
            mask=np.ones(n_spins, dtype=bool),
            name="PSF point sources",
            metadata={"psf_point_sources": True, "n_point_sources": len(batch)},
        )
        # Point sources are not constrained to an axis-aligned grid.
        phantom.positions = positions.copy()
        phantom.x = positions[:, 0].copy()
        phantom.y = positions[:, 1].copy()
        phantom.z = positions[:, 2].copy()
        result = simulator.simulate_sequence(
            program,
            phantom,
            simulation_timestep_s=simulation_timestep_s,
            sequence_kernel=sequence_kernel,
            spoiler_mode=spoiler_mode,
            equilibrium_mz=np.repeat(
                [source.equilibrium_mz for source in batch], counts
            ),
            cancel_callback=cancel_callback,
        )
        signals.append(np.asarray(result.signal).reshape(len(batch), -1))
        if progress_callback is not None:
            progress_callback(start + len(batch), len(sources))
    return replace(result, signal=np.concatenate(signals, axis=0))


# Encoding roles index the columns of a layout's EncodingFrame; non-Cartesian
# layouts use the identity frame with scanner axis roles.
ROLE_INDEX = {"read": 0, "phase": 1, "partition": 2, "x": 0, "y": 1, "z": 2}


def evaluate_sample_image(samples, k_cyc_per_m, offsets_m) -> np.ndarray:
    """Evaluate ``sum_j s_j exp(2 pi i k_j . x)`` on a separable 2D grid.

    ``samples`` has shape ``(..., J)`` and ``k_cyc_per_m`` shape ``(J, 2)``
    with one column per entry of ``offsets_m``. The sum is not normalized.
    """
    values = np.asarray(samples, dtype=np.complex128)
    k = np.asarray(k_cyc_per_m, dtype=float)
    if k.ndim != 2 or k.shape[1] != 2 or len(offsets_m) != 2:
        raise ValueError("sample images are evaluated on 2D grids")
    first, second = (
        np.exp(2j * np.pi * np.outer(np.atleast_1d(x), k[:, axis]))
        for axis, x in enumerate(offsets_m)
    )
    return (values[..., None, :] * first) @ second.T


def _gradient_error_array(gradient_error_cyc_per_m, num_samples) -> np.ndarray:
    delta = np.asarray(gradient_error_cyc_per_m, dtype=float)
    if delta.shape != (num_samples, 3):
        raise ValueError("gradient_error_cyc_per_m must have shape (num_samples, 3)")
    # Samples before the first excitation carry no k-space position.
    return np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass(frozen=True)
class CartesianPSFLayout:
    """One reconstructable Cartesian 2D frame or 3D volume of a result.

    ``roles`` and ``k_cyc_per_m`` follow the k-space array order, i.e.
    ``(phase, read)`` or ``(partition, phase, read)``. ``center_m`` is the
    logical read/phase/partition FOV centre the reconstruction is placed on.
    ``sample_k`` replaces the nominal grid by per-cell k vectors (array-order
    components, shape ``matrix + (dims,)``) for trajectory-corrected
    reconstructions of 2D frames.
    """

    kind = "cartesian"

    label: str
    roles: tuple
    k_cyc_per_m: tuple
    fov_m: tuple
    matrix: tuple
    encoding_frame: EncodingFrame
    center_m: tuple
    slice_index: int
    _layout: object = field(repr=False, compare=False)
    _index: int = 0
    sample_k: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    @property
    def voxel_size_m(self) -> tuple:
        return tuple(fov / count for fov, count in zip(self.fov_m, self.matrix))

    def _k_vectors(self) -> np.ndarray:
        if self.sample_k is not None:
            return self.sample_k
        grids = np.meshgrid(*self.k_cyc_per_m, indexing="ij")
        return np.stack(grids, axis=-1)

    def samples(self, result) -> np.ndarray:
        return self.kspace(result)

    def encoding_phase(self, offset_m) -> np.ndarray:
        """``exp(-2 pi i k . x0)`` of a source at array-order ``offset_m``."""
        return np.exp(-2j * np.pi * (self._k_vectors() @ np.asarray(offset_m)))

    def centre_index(self) -> tuple:
        return tuple(int(np.argmin(np.abs(k))) for k in self.k_cyc_per_m)

    def evaluate(self, values, offsets_m) -> np.ndarray:
        """Image of k-space ``values`` at array-order offset vectors."""
        if self.sample_k is None:
            return evaluate_cartesian_image(values, self.k_cyc_per_m, offsets_m)
        dims = len(self.matrix)
        values = np.asarray(values)
        flat = values.reshape(values.shape[: values.ndim - dims] + (-1,))
        return evaluate_sample_image(
            flat / flat.shape[-1], self.sample_k.reshape(-1, dims), offsets_m
        )

    def with_gradient_error(self, result, gradient_error_cyc_per_m):
        """Layout reconstructed on the actual trajectory ``k + dk`` (2D only)."""
        if len(self.matrix) != 2:
            raise ValueError("trajectory correction supports 2D frames only")
        count = np.asarray(result.signal).shape[-1]
        delta = _gradient_error_array(gradient_error_cyc_per_m, count)
        # The k-space layout reorders ADC samples; an index stream reveals
        # which sample fills each cell (0 marks cells without a sample).
        stream = np.arange(1, count + 1, dtype=np.complex128)[None, :]
        cells = np.rint(np.abs(self.kspace(replace(result, signal=stream))[0]))
        cells = cells.astype(int)
        if cells.min() < 0 or cells.max() > count:
            raise ValueError("unable to map ADC samples onto the k-space grid")
        logical = self.encoding_frame.scanner_to_encoding(delta)
        logical = logical[:, [ROLE_INDEX[role] for role in self.roles]]
        logical = np.concatenate((np.zeros((1, len(self.roles))), logical))
        nominal = np.stack(np.meshgrid(*self.k_cyc_per_m, indexing="ij"), -1)
        return replace(self, sample_k=nominal + logical[cells])

    def kspace(self, result) -> np.ndarray:
        """Return ``(source, [partition], phase, read)`` k-space of a result."""
        if np.asarray(result.signal).ndim != 2:
            raise ValueError("point-source results must have signal (source, adc)")
        if hasattr(self._layout, "to_cartesian_kspace"):
            return self._layout.to_cartesian_kspace(result, self._index)
        return self._layout.reshape_signal(result.signal)

    def source_offset_m(self, position_m) -> np.ndarray:
        """Source position relative to the FOV centre in array-order roles."""
        logical = self.encoding_frame.scanner_to_encoding(
            np.asarray(position_m, dtype=float)
        ) - np.asarray(self.center_m, dtype=float)
        order = [ROLE_INDEX[role] for role in self.roles]
        return logical[order]


def cartesian_psf_layouts(result) -> tuple:
    """Return every Cartesian frame or volume a PSF can be computed for."""
    volumes = result.cartesian_acquisition_volumes
    slice_axis = CartesianAcquisitionVolumes.OUTER_AXIS_NAMES.index("slice")
    if volumes is not None:
        return tuple(
            CartesianPSFLayout(
                label=volumes.volume_label(index),
                roles=("partition", "phase", "read"),
                k_cyc_per_m=(
                    volumes.kz_cyc_per_m,
                    volumes.ky_cyc_per_m,
                    volumes.kx_cyc_per_m,
                ),
                fov_m=(volumes.fov_m[2], volumes.fov_m[1], volumes.fov_m[0]),
                matrix=(
                    volumes.partition_matrix,
                    volumes.phase_matrix,
                    volumes.read_matrix,
                ),
                encoding_frame=volumes.encoding_frame,
                center_m=(*volumes.center_m, 0.0),
                slice_index=int(volumes.volume_indices[index][slice_axis]),
                _layout=volumes,
                _index=index,
            )
            for index in range(volumes.num_volumes)
        )

    def planar(acquisition, label, slice_index, layout, index):
        return CartesianPSFLayout(
            label=label,
            roles=("phase", "read"),
            k_cyc_per_m=(acquisition.ky_cyc_per_m, acquisition.kx_cyc_per_m),
            fov_m=(acquisition.fov_m[1], acquisition.fov_m[0]),
            matrix=(acquisition.phase_matrix, acquisition.read_matrix),
            encoding_frame=acquisition.encoding_frame,
            center_m=(*acquisition.center_m, 0.0),
            slice_index=slice_index,
            _layout=layout,
            _index=index,
        )

    frames = result.cartesian_acquisition_frames
    if frames is not None:
        frame_slice_axis = AcquisitionDimensions.AXIS_NAMES.index("slice")
        return tuple(
            planar(
                acquisition,
                frames.frame_label(index),
                int(frames.frame_indices[index][frame_slice_axis]),
                frames,
                index,
            )
            for index, acquisition in enumerate(frames.acquisitions)
        )
    acquisition = result.cartesian_acquisition
    if acquisition is not None:
        return (planar(acquisition, "frame=0", 0, acquisition, 0),)
    raise ValueError(
        "PSF estimation supports Cartesian 2D frames and 3D volumes; this "
        "sequence has no Cartesian acquisition layout"
    )


@dataclass(frozen=True)
class SpiralPSFLayout:
    """One 2D spiral frame of a result, reconstructed by gridding.

    ``roles`` are the scanner axes in image array order ``(y, x)``, like the
    ``(phase_y, read_x)`` spiral reconstructions. ``sample_k`` holds the
    per-sample k vectors in that order; ``center_m`` is the scanner-xyz
    position of the reconstruction origin.
    """

    kind = "spiral"

    label: str
    roles: tuple
    fov_m: tuple
    matrix: tuple
    encoding_frame: EncodingFrame
    center_m: tuple
    slice_index: int
    conjugated: bool
    sample_k: np.ndarray = field(repr=False, compare=False)
    _trajectory: np.ndarray = field(repr=False, compare=False)
    _spiral: object = field(repr=False, compare=False)
    _index: int = 0

    @property
    def voxel_size_m(self) -> tuple:
        return tuple(fov / count for fov, count in zip(self.fov_m, self.matrix))

    @property
    def k_cyc_per_m(self) -> tuple:
        """Per-sample k components in array order."""
        return tuple(self.sample_k[:, axis] for axis in range(len(self.roles)))

    @property
    def weights(self) -> np.ndarray:
        """Density compensation of the frame, one per Cartesian cell."""
        return self._spiral.nufft_operator(self._trajectory).density_compensation()

    def samples(self, result) -> np.ndarray:
        """Return ``(source, sample)`` data of the frame."""
        if np.asarray(result.signal).ndim != 2:
            raise ValueError("point-source results must have signal (source, adc)")
        values = self._spiral._frame_values(result.signal, self._index)
        if self._spiral.offset_in_reconstruction and any(self.center_m):
            values = values * np.exp(
                2j * np.pi * (self._trajectory @ np.asarray(self.center_m))
            )
        return values

    def encoding_phase(self, offset_m) -> np.ndarray:
        return np.exp(-2j * np.pi * (self.sample_k @ np.asarray(offset_m)))

    def centre_index(self) -> tuple:
        return (int(np.argmin(np.linalg.norm(self.sample_k, axis=1))),)

    def evaluate(self, values, offsets_m) -> np.ndarray:
        weighted = np.asarray(values) * self.weights / float(np.prod(self.matrix))
        return evaluate_sample_image(weighted, self.sample_k, offsets_m)

    def source_offset_m(self, position_m) -> np.ndarray:
        relative = np.asarray(position_m, dtype=float) - np.asarray(self.center_m)
        return relative[[ROLE_INDEX[role] for role in self.roles]]

    def with_gradient_error(self, result, gradient_error_cyc_per_m):
        """Layout reconstructed on the actual trajectory ``k + dk``."""
        count = np.asarray(result.signal).shape[-1]
        delta = _gradient_error_array(gradient_error_cyc_per_m, count)
        trajectory = (
            self._trajectory + delta[list(self._spiral.sample_indices[self._index])]
        )
        return replace(
            self,
            sample_k=trajectory[:, [ROLE_INDEX[role] for role in self.roles]],
            _trajectory=trajectory,
        )


def spiral_psf_layouts(result) -> tuple:
    """Return every 2D spiral frame of a result a PSF can be computed for."""
    spiral = result.spiral_acquisition
    if spiral is None:
        raise ValueError("this result has no spiral acquisition layout")
    moments = result.adc_gradient_moment_cyc_per_m
    if moments is None:
        raise ValueError("spiral PSFs require ADC gradient moments")
    roles = ("y", "x")
    slice_axis = AcquisitionDimensions.AXIS_NAMES.index("slice")
    origin = spiral.fov_center_xyz_m if spiral.offset_in_reconstruction else (0.0,) * 3
    layouts = []
    for index in range(spiral.num_frames):
        trajectory = spiral.trajectory(moments, index)
        layouts.append(
            SpiralPSFLayout(
                label=spiral.frame_label(index),
                roles=roles,
                fov_m=(spiral.fov_m[1], spiral.fov_m[0]),
                matrix=(spiral.matrix[1], spiral.matrix[0]),
                encoding_frame=EncodingFrame.identity(),
                center_m=tuple(float(value) for value in origin),
                slice_index=int(spiral.frame_indices[index][slice_axis]),
                conjugated=bool(spiral.conjugated_frames[index]),
                sample_k=trajectory[:, [ROLE_INDEX[role] for role in roles]],
                _trajectory=trajectory,
                _spiral=spiral,
                _index=index,
            )
        )
    return tuple(layouts)


def psf_layouts(result) -> tuple:
    """Return the Cartesian or spiral frames a PSF can be computed for."""
    if result.spiral_acquisition is not None:
        return spiral_psf_layouts(result)
    try:
        return cartesian_psf_layouts(result)
    except ValueError:
        raise ValueError(
            "PSF estimation supports Cartesian 2D frames and 3D volumes and "
            "2D spiral frames; this sequence has no supported acquisition layout"
        ) from None


def evaluate_cartesian_image(kspace, k_cyc_per_m, offsets_m) -> np.ndarray:
    """Evaluate the inverse DFT of trailing k-space axes at given offsets.

    ``offsets_m`` holds one coordinate vector per k axis, relative to the FOV
    centre. The result is normalized like an unnormalized ``ifftn``, so a
    flat unit k-space yields a unit peak.
    """
    values = np.asarray(kspace, dtype=np.complex128)
    n_axes = len(k_cyc_per_m)
    lead = values.ndim - n_axes
    for axis, (k, x) in enumerate(zip(k_cyc_per_m, offsets_m)):
        kernel = np.exp(
            2j * np.pi * np.outer(np.atleast_1d(x), np.asarray(k, dtype=float))
        )
        values = np.moveaxis(
            np.tensordot(values, kernel, axes=([lead + axis], [1])),
            -1,
            lead + axis,
        )
    return values / float(np.prod([np.size(k) for k in k_cyc_per_m]))


@dataclass(frozen=True)
class AxisPSFMetrics:
    """1D PSF characteristics along one encoding role through the peak."""

    role: str
    voxel_size_m: float
    fwhm_m: float
    ideal_fwhm_m: float
    peak_offset_m: float
    peak_sidelobe_ratio: float
    ideal_peak_sidelobe_ratio: float

    @property
    def fwhm_voxels(self) -> float:
        return self.fwhm_m / self.voxel_size_m

    @property
    def broadening(self) -> float:
        """FWHM relative to the PSF of uniform k-space weighting."""
        return self.fwhm_m / self.ideal_fwhm_m

    @property
    def peak_offset_voxels(self) -> float:
        return self.peak_offset_m / self.voxel_size_m


@dataclass(frozen=True)
class PSFResult:
    """PSF of one frame for one point source.

    ``psf`` and ``ideal_psf`` are sampled at ``offsets_m``, the displacement
    from the source along ``roles`` (array order) covering one FOV period.
    The ideal PSF uses flat k-space weighting equal to the measured
    modulation at the k-space centre on the nominal trajectory, without
    gradient errors. ``modulation`` is the sampled k-space with the source's
    encoding phase removed, i.e. the transfer function. For Cartesian
    layouts (``kind == "cartesian"``) it is a grid over the axes
    ``k_cyc_per_m``; for spiral frames it holds one value per sample and
    ``k_cyc_per_m`` the per-sample k components of the reconstruction.
    ``gradient_error`` is ``"none"``, ``"uncorrected"`` or ``"corrected"``.
    """

    source: PointSource
    label: str
    frame: int
    frame_labels: tuple
    roles: tuple
    offsets_m: tuple
    psf: np.ndarray
    ideal_psf: np.ndarray
    profiles: Mapping
    k_cyc_per_m: tuple
    modulation: np.ndarray
    axis_metrics: Mapping
    peak_efficiency: float
    voxel_energy_fraction: float
    ideal_voxel_energy_fraction: float
    voxel_size_m: tuple
    encoding_frame: EncodingFrame
    kind: str = "cartesian"
    gradient_error: str = "none"


def _wrap(values, period):
    return (np.asarray(values) + period / 2.0) % period - period / 2.0


def _periodic_offsets(count, step):
    return (np.arange(count, dtype=float) - count // 2) * step


def _half_maximum_crossing(x, magnitude, peak, level, direction):
    index = peak
    while 0 <= index + direction < magnitude.size:
        following = index + direction
        if magnitude[following] <= level:
            span = magnitude[index] - magnitude[following]
            fraction = 0.0 if span == 0 else (magnitude[index] - level) / span
            return x[index] + fraction * (x[following] - x[index])
        index = following
    return np.nan


def _main_lobe_bounds(magnitude, peak):
    left = peak
    while left > 0 and magnitude[left - 1] <= magnitude[left]:
        left -= 1
    right = peak
    while right < magnitude.size - 1 and magnitude[right + 1] <= magnitude[right]:
        right += 1
    return left, right


def profile_metrics(offsets_m, values) -> tuple[float, float, float]:
    """Return FWHM, interpolated peak offset, and peak side-lobe ratio."""
    x = np.asarray(offsets_m, dtype=float)
    magnitude = np.abs(np.asarray(values))
    peak = int(np.argmax(magnitude))
    peak_value = float(magnitude[peak])
    if peak_value == 0.0:
        return np.nan, np.nan, np.nan
    level = 0.5 * peak_value
    width = _half_maximum_crossing(
        x, magnitude, peak, level, 1
    ) - _half_maximum_crossing(x, magnitude, peak, level, -1)
    peak_offset = float(x[peak])
    if 0 < peak < x.size - 1:
        left, centre, right = magnitude[peak - 1 : peak + 2]
        curvature = left - 2.0 * centre + right
        if curvature < 0:
            peak_offset += 0.5 * (left - right) / curvature * (x[peak + 1] - x[peak])
    left, right = _main_lobe_bounds(magnitude, peak)
    outside = np.concatenate((magnitude[:left], magnitude[right + 1 :]))
    sidelobe = float(outside.max() / peak_value) if outside.size else 0.0
    return float(width), peak_offset, sidelobe


def _voxel_energy_fraction(image, offsets_m, voxel_size_m):
    energy = np.abs(image) ** 2
    total = float(energy.sum())
    if total == 0.0:
        return np.nan
    inside = np.ones(energy.shape, dtype=bool)
    for axis, (x, size) in enumerate(zip(offsets_m, voxel_size_m)):
        shape = [1] * energy.ndim
        shape[axis] = -1
        inside &= (np.abs(x) <= size / 2.0 + 1e-12 * size).reshape(shape)
    return float(energy[inside].sum() / total)


def psf_from_result(
    result,
    source: PointSource,
    *,
    frame: int = 0,
    source_index: int = 0,
    oversampling: int = 4,
    profile_oversampling: int = 32,
    gradient_error_cyc_per_m=None,
    trajectory_correction: bool = False,
) -> PSFResult:
    """Reconstruct the PSF of one simulated point source and one frame.

    ``gradient_error_cyc_per_m`` is the deviation of the actual from the
    nominal k-space position for every ADC sample, shape ``(adc, 3)`` in
    scanner xyz. It is imposed on the source's signal; with
    ``trajectory_correction`` the reconstruction uses the actual trajectory.
    """
    layouts = psf_layouts(result)
    frame = int(frame)
    if not 0 <= frame < len(layouts):
        raise ValueError(f"frame must lie in [0, {len(layouts) - 1}]")
    if int(oversampling) < 1 or int(profile_oversampling) < 1:
        raise ValueError("oversampling factors must be positive integers")
    nominal = layouts[frame]
    layout = nominal
    error_mode = "none"
    if gradient_error_cyc_per_m is not None:
        signal = np.asarray(result.signal)
        delta = _gradient_error_array(gradient_error_cyc_per_m, signal.shape[-1])
        # Exact for a point: s = s_nominal exp(-2 pi i dk . r0), r0 in scanner xyz.
        error_phase = np.exp(-2j * np.pi * (delta @ np.asarray(source.position_m)))
        result = replace(result, signal=signal * error_phase)
        error_mode = "uncorrected"
        if trajectory_correction:
            layout = nominal.with_gradient_error(result, delta)
            error_mode = "corrected"
    data = layout.samples(result)[int(source_index)]
    source_offset = layout.source_offset_m(source.position_m)

    encoding_phase = layout.encoding_phase(source_offset)
    modulation = data / encoding_phase
    reference = modulation[nominal.centre_index()]
    if reference == 0:
        reference = modulation.reshape(-1)[np.argmax(np.abs(modulation))]
    ideal_data = reference * nominal.encoding_phase(source_offset)

    factor = int(oversampling)
    dims = len(layout.matrix)
    image_points = float(np.prod(layout.matrix))
    if dims == 3:
        budget = MAX_PSF_IMAGE_POINTS / image_points
        factor = max(1, min(factor, int(np.floor(budget ** (1.0 / 3.0)))))
    if layout.kind != "cartesian" or layout.sample_k is not None:
        budget = MAX_SAMPLE_IMAGE_PRODUCTS / (image_points * data.size)
        factor = max(1, min(factor, int(np.floor(np.sqrt(budget)))))
    offsets = tuple(
        _periodic_offsets(count * factor, size / factor)
        for count, size in zip(layout.matrix, layout.voxel_size_m)
    )
    absolute = tuple(x + x0 for x, x0 in zip(offsets, source_offset))
    conjugate = np.conj if getattr(layout, "conjugated", False) else np.asarray
    psf = conjugate(layout.evaluate(data, absolute))
    ideal_psf = conjugate(nominal.evaluate(ideal_data, absolute))
    ideal_peak = float(np.abs(ideal_psf).max())

    peak_index = np.unravel_index(int(np.argmax(np.abs(psf))), psf.shape)
    ideal_peak_index = np.unravel_index(
        int(np.argmax(np.abs(ideal_psf))), ideal_psf.shape
    )
    profiles = {}
    metrics = {}
    for axis, role in enumerate(layout.roles):
        count = layout.matrix[axis]
        size = layout.voxel_size_m[axis]
        line = _periodic_offsets(
            count * int(profile_oversampling), size / int(profile_oversampling)
        )

        def through(peak_at, reconstruction, values):
            coordinates = [
                np.atleast_1d(offsets[other][peak_at[other]] + source_offset[other])
                for other in range(dims)
            ]
            coordinates[axis] = line + source_offset[axis]
            return conjugate(reconstruction.evaluate(values, coordinates)).reshape(-1)

        values = through(peak_index, layout, data)
        ideal_values = through(ideal_peak_index, nominal, ideal_data)
        profiles[role] = (line, values, ideal_values)
        width, peak_offset, sidelobe = profile_metrics(line, values)
        ideal_width, _, ideal_sidelobe = profile_metrics(line, ideal_values)
        metrics[role] = AxisPSFMetrics(
            role=role,
            voxel_size_m=size,
            fwhm_m=width,
            ideal_fwhm_m=ideal_width,
            peak_offset_m=float(_wrap(peak_offset, layout.fov_m[axis])),
            peak_sidelobe_ratio=sidelobe,
            ideal_peak_sidelobe_ratio=ideal_sidelobe,
        )

    return PSFResult(
        source=source,
        label=layout.label,
        frame=frame,
        frame_labels=tuple(item.label for item in layouts),
        roles=layout.roles,
        offsets_m=offsets,
        psf=psf,
        ideal_psf=ideal_psf,
        profiles=profiles,
        k_cyc_per_m=tuple(np.asarray(k, dtype=float) for k in layout.k_cyc_per_m),
        modulation=modulation,
        axis_metrics=metrics,
        peak_efficiency=(
            float(np.abs(psf).max() / ideal_peak) if ideal_peak > 0 else np.nan
        ),
        voxel_energy_fraction=_voxel_energy_fraction(psf, offsets, layout.voxel_size_m),
        ideal_voxel_energy_fraction=_voxel_energy_fraction(
            ideal_psf, offsets, layout.voxel_size_m
        ),
        voxel_size_m=layout.voxel_size_m,
        encoding_frame=layout.encoding_frame,
        kind=layout.kind,
        gradient_error=error_mode,
    )


def estimate_psf(
    program,
    source: PointSource = PointSource(),
    *,
    frame: int = 0,
    oversampling: int = 4,
    profile_oversampling: int = 32,
    imperfections=None,
    trajectory_correction: bool = False,
    **simulation_options,
) -> PSFResult:
    """Simulate one point source through ``program`` and return its PSF.

    ``imperfections`` (:class:`.GradientImperfections` or a mapping) adds
    the resulting gradient errors to the PSF, see :func:`psf_from_result`.
    ``simulation_options`` are passed to :func:`simulate_point_sources`.
    """
    result = simulate_point_sources(program, (source,), **simulation_options)
    error = None
    if imperfections is not None:
        from .kspace_trajectory import gradient_error

        error = gradient_error(program, imperfections)
    return psf_from_result(
        result,
        source,
        frame=frame,
        oversampling=oversampling,
        profile_oversampling=profile_oversampling,
        gradient_error_cyc_per_m=error,
        trajectory_correction=trajectory_correction,
    )


def _definitions(program) -> dict:
    return {
        str(key).lower(): value
        for key, value in dict(program.metadata.get("definitions", {})).items()
    }


def _definition_values(definitions, key) -> np.ndarray:
    value = definitions.get(key.lower())
    if value is None:
        return np.zeros(0)
    return numeric_definition_array(value, key)


def slice_geometry(program) -> tuple[Optional[float], np.ndarray]:
    """Return nominal slice thickness and slice centres along the normal."""
    definitions = _definitions(program)
    thickness = _definition_values(definitions, "SliceThickness")
    thickness_m = float(thickness[0]) if thickness.size == 1 else None
    positions = _definition_values(definitions, "SlicePositions")
    if positions.size == 0:
        positions = _definition_values(definitions, "SliceOffset")[:1]
    if positions.size == 0:
        positions = np.zeros(1)
    return thickness_m, positions


@dataclass(frozen=True)
class SliceProfileMetrics:
    """Through-slice response of one frame; lengths along the slice normal."""

    nominal_thickness_m: Optional[float]
    nominal_centre_m: float
    fwhm_m: float
    centre_m: float
    out_of_slice_fraction: float

    @property
    def thickness_ratio(self) -> float:
        if not self.nominal_thickness_m:
            return np.nan
        return self.fwhm_m / self.nominal_thickness_m


@dataclass(frozen=True)
class SliceProfileResult:
    """Slice profiles of a 2D sequence, one row per reconstructed frame.

    ``profiles[frame, position]`` is the complex image value at the source's
    in-plane position when the source sits at ``positions_m`` along
    ``normal_xyz``. Rows of other slices show slice cross-talk.
    """

    source: PointSource
    normal_xyz: np.ndarray
    positions_m: np.ndarray
    profiles: np.ndarray
    frame_labels: tuple
    slice_indices: tuple
    metrics: tuple


def _slice_metrics(positions, values, thickness, centre):
    magnitude = np.abs(values)
    width, peak_position, _ = profile_metrics(positions, values)
    left = _half_maximum_crossing(
        positions, magnitude, int(np.argmax(magnitude)), 0.5 * magnitude.max(), -1
    )
    centre_of_lobe = left + 0.5 * width if np.isfinite(width) else peak_position
    total = float(np.trapezoid(magnitude, positions))
    fraction = np.nan
    if thickness and total > 0:
        inside = np.abs(positions - centre) <= thickness / 2.0
        if np.count_nonzero(inside) > 1:
            fraction = 1.0 - float(
                np.trapezoid(magnitude[inside], positions[inside]) / total
            )
    return SliceProfileMetrics(
        nominal_thickness_m=thickness,
        nominal_centre_m=float(centre),
        fwhm_m=float(width),
        centre_m=float(centre_of_lobe),
        out_of_slice_fraction=fraction,
    )


def estimate_slice_profile(
    program,
    source: PointSource = PointSource(),
    *,
    positions_m: Optional[Sequence[float]] = None,
    margin_thicknesses: float = 1.5,
    points_per_thickness: int = 24,
    **simulation_options,
) -> SliceProfileResult:
    """Sweep a point source along the slice normal of a 2D sequence.

    ``positions_m`` are absolute positions along the normal (the partition
    direction); the source keeps its in-plane position. By default the sweep
    covers the slice package plus ``margin_thicknesses`` slice thicknesses on
    either side. ``simulation_options`` go to :func:`simulate_point_sources`.
    """
    frame_basis = EncodingFrame.from_definitions(
        program.metadata.get("definitions", {})
    )
    normal = np.asarray(frame_basis.direction("partition"), dtype=float)
    thickness, centres = slice_geometry(program)
    if positions_m is None:
        if thickness is None:
            raise ValueError(
                "the sequence declares no SliceThickness; pass positions_m"
            )
        low = float(centres.min()) - margin_thicknesses * thickness
        high = float(centres.max()) + margin_thicknesses * thickness
        count = int(np.ceil((high - low) / thickness * points_per_thickness)) + 1
        positions = np.linspace(low, high, min(count, MAX_SLICE_PROFILE_POSITIONS))
    else:
        positions = np.asarray(positions_m, dtype=float).reshape(-1)
        if positions.size == 0 or not np.all(np.isfinite(positions)):
            raise ValueError("positions_m must contain finite positions")
    in_plane = np.asarray(source.position_m) - normal * float(
        normal @ np.asarray(source.position_m)
    )
    sources = tuple(source.moved_to(in_plane + normal * value) for value in positions)
    result = simulate_point_sources(program, sources, **simulation_options)
    layouts = psf_layouts(result)
    if len(layouts[0].roles) == 3:
        raise ValueError(
            "3D acquisitions encode the partition direction; use estimate_psf"
        )

    rows = []
    for layout in layouts:
        kspace = layout.samples(result)
        offsets = np.asarray(
            [layout.source_offset_m(item.position_m) for item in sources]
        )
        rows.append(
            [
                layout.evaluate(
                    kspace[index], [np.atleast_1d(value) for value in offsets[index]]
                ).item()
                for index in range(len(sources))
            ]
        )
    profiles = np.asarray(rows, dtype=np.complex128)
    metrics = tuple(
        _slice_metrics(
            positions,
            profiles[index],
            thickness,
            centres[layout.slice_index] if layout.slice_index < centres.size else 0.0,
        )
        for index, layout in enumerate(layouts)
    )
    return SliceProfileResult(
        source=source,
        normal_xyz=normal,
        positions_m=positions,
        profiles=profiles,
        frame_labels=tuple(layout.label for layout in layouts),
        slice_indices=tuple(layout.slice_index for layout in layouts),
        metrics=metrics,
    )
