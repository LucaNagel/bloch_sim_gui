"""Private adapter that runs static sequences on the spin-major executor.

``BlochSimulator.simulate_sequence`` and ``simulate_spectral_sequence`` call
:func:`run_static_components` for the ``"production"`` kernel (the default).
The static components become pools of a dynamic phantom without kinetics,
which reuses the dynamic compile, the execution plan and the float64
spin-major executor.

Mapping of static component ``c`` onto pool ``c``:

* a single component keeps unit spin density on its active voxels and folds
  its signal weight (proton density, optionally times the voxel volume) into
  the receive sensitivity;
* several components (spectral species) carry their weights as spin
  densities, normalised per pass to at most one so small checkpoint dtypes
  stay in range; the normalisation is folded into the receive sensitivity;
* Mz relaxes toward ``equilibrium * density``; per-voxel T1/T2 become
  relaxation groups whose most frequent row is the pool constant;
* B0 is shared, the component's constant chemical shift is the pool offset;
* more than six components run in passes of at most six pools.

Results are returned per component in per-unit magnetization over that
component's active voxels, aligned with ``get_active_properties()``.
Progress counts plan intervals summed over passes, not voxels. Anything the
executor cannot run raises :class:`StaticSpinMajorFallback`; callers warn
with ``KernelFallbackWarning`` and run the ``"legacy"`` kernel instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .dynamic_phantom import (
    _SPIN_MAJOR_MAX_POOLS,
    DynamicSpectralPhantom,
    _run_superposed_passes,
    _SpinMajorRequired,
)
from .spectral_phantom import ChemicalSpecies
from .units import NUCLEUS_GAMMA_HZ_PER_T

# Each group adds one precomputed relaxation entry per plan group in C.
MAX_RELAXATION_GROUPS = 256

PROGRESS_UNIT = "intervals"


class StaticSpinMajorFallback(RuntimeError):
    """The static run cannot use the spin-major executor; ``reason`` says why."""

    def __init__(self, reason):
        super().__init__(str(reason))
        self.reason = str(reason)


@dataclass
class StaticComponentOutput:
    """Per-component output in per-unit magnetization over active voxels."""

    signal: np.ndarray  # (n_rx, n_adc)
    final_active: np.ndarray  # (n_active, 3)
    checkpoints_active: np.ndarray  # (n_checkpoints, n_active, 3)


class _StaticPoolPhantom(DynamicSpectralPhantom):
    """Dynamic phantom view of static components; needs the spin-major path."""

    _MIN_POOLS = 1
    requires_spin_major = True
    relaxation_group_map = None
    relaxation_groups = None
    spin_equilibrium_map = None
    _initial_state = None

    @property
    def initial_magnetization(self) -> np.ndarray:
        return self._initial_state


def _relaxation_groups(rows: np.ndarray, active: np.ndarray):
    """Return (pool constants, group map, groups) for (nvoxels, npools, 2) rows.

    Rows hold (R1, T2) per pool. The most frequent active row becomes the
    pool constants; every other distinct row is one group.
    """
    nvoxels, npools, _ = rows.shape
    flat = np.ascontiguousarray(rows[active].reshape(-1, npools * 2))
    unique, inverse, counts = np.unique(
        flat, axis=0, return_inverse=True, return_counts=True
    )
    inverse = np.asarray(inverse).reshape(-1)
    if unique.shape[0] - 1 > MAX_RELAXATION_GROUPS:
        raise StaticSpinMajorFallback(
            f"{unique.shape[0]:,} distinct T1/T2 combinations exceed "
            f"{MAX_RELAXATION_GROUPS} relaxation groups"
        )
    base = int(np.argmax(counts))
    constants = unique[base].reshape(npools, 2)
    if unique.shape[0] == 1:
        return constants, None, None
    others = np.flatnonzero(np.arange(unique.shape[0]) != base)
    group_of_unique = np.full(unique.shape[0], -1, dtype=np.int32)
    group_of_unique[others] = np.arange(others.size, dtype=np.int32)
    group_map = np.full(nvoxels, -1, dtype=np.int32)
    group_map[active] = group_of_unique[inverse]
    groups = np.ascontiguousarray(unique[others].reshape(-1, npools, 2))
    return constants, group_map, groups


def _most_frequent_row(values: np.ndarray) -> np.ndarray:
    unique, counts = np.unique(values, axis=0, return_counts=True)
    return unique[int(np.argmax(counts))]


@dataclass
class _Component:
    name: str
    mask: np.ndarray  # (nvoxels,) bool
    indices: np.ndarray  # active flat indices
    density: np.ndarray  # (nvoxels,) >= 0, zero outside the mask
    m0: np.ndarray  # (nvoxels, 3) per unit, zero outside the mask
    rows: np.ndarray  # (nvoxels, 2) R1 and T2, valid everywhere
    offset_hz: float


def _component(name, flat, *, single, weight_scale) -> _Component:
    mask = np.asarray(flat["mask"], dtype=bool)
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError("phantom has no active voxels")
    t1 = np.asarray(flat["t1"], dtype=np.float64)
    t2 = np.asarray(flat["t2"], dtype=np.float64)
    pd = np.asarray(flat["pd"], dtype=np.float64)
    m0 = np.asarray(flat["m0"], dtype=np.float64)
    for label, values in (
        ("T1", t1),
        ("T2", t2),
        ("proton density", pd),
        ("initial magnetization", m0),
    ):
        if not np.all(np.isfinite(values[mask])):
            raise ValueError(f"active phantom {label} contains NaN or infinite values")
    if np.any(t1[mask] <= 0) or np.any(t2[mask] <= 0):
        raise ValueError("active phantom voxels require T1 > 0 and T2 > 0")
    if np.any(pd[mask] < 0):
        raise ValueError("active phantom proton density must not be negative")
    rows = np.empty((mask.size, 2), dtype=np.float64)
    rows[:, 0] = 1.0 / np.where(mask, t1, 1.0)
    rows[:, 1] = np.where(mask, t2, 1.0)
    # Voxels outside the component carry no magnetization; give them the
    # component's most frequent relaxation so they add no groups.
    rows[~mask] = _most_frequent_row(rows[mask])
    if single:
        density = mask.astype(np.float64)
        offset_hz = 0.0
    else:
        density = np.where(mask, pd * weight_scale, 0.0)
        shift = np.asarray(flat["chemical_shift"], dtype=np.float64)[mask]
        if not np.all(shift == shift[0]):
            raise StaticSpinMajorFallback(
                f"component {name!r} has a spatially varying chemical shift"
            )
        offset_hz = float(shift[0])
    return _Component(
        name=str(name),
        mask=mask,
        indices=indices,
        density=density,
        m0=np.where(mask[:, None], m0, 0.0),
        rows=rows,
        offset_hz=offset_hz,
    )


def _pass_phantom(
    components: Sequence[_Component],
    *,
    shape3,
    fov3,
    affine,
    positions,
    active,
    b0_hz,
    tx,
    rx,
    equilibrium,
    field_strength,
    nucleus,
):
    """Build the dynamic phantom of one pass and its density normalisation."""
    scale = max(float(np.max(item.density)) for item in components)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    density = np.stack([item.density / scale for item in components])
    rows = np.stack([item.rows for item in components], axis=1)
    constants, group_map, groups = _relaxation_groups(rows, active)
    species = tuple(
        ChemicalSpecies(
            item.name,
            0.0,
            1.0 / constants[index, 0],
            constants[index, 1],
            frequency_offset_hz=item.offset_hz,
        )
        for index, item in enumerate(components)
    )
    maps = {
        item.name: density[index].reshape(shape3)
        for index, item in enumerate(components)
    }
    equilibrium_polarization = 1.0
    if equilibrium is not None:
        values = equilibrium[active]
        if np.any(values < 0.0):
            raise StaticSpinMajorFallback("negative equilibrium Mz is unsupported")
        if np.all(values == values[0]):
            equilibrium_polarization = float(values[0])
            equilibrium = None
    phantom = _StaticPoolPhantom(
        shape=shape3,
        fov=fov3,
        pools=species,
        initial_concentration_maps={key: value.copy() for key, value in maps.items()},
        kpl_map_s_inv=np.zeros(shape3),
        initial_spin_density_maps=maps,
        equilibrium_polarization=equilibrium_polarization,
        b0_map=b0_hz.reshape(shape3),
        field_strength=field_strength,
        nucleus=nucleus,
        name="static spin-major view",
        affine_ijk_to_xyz_m=affine,
    )
    phantom.positions = positions
    if tx is not None:
        phantom.tx_sensitivity_map = tx.reshape(shape3)
    phantom.rx_sensitivity_maps = (rx * scale).reshape((rx.shape[0],) + shape3)
    state = density[:, :, None] * np.stack([item.m0 for item in components])
    phantom._initial_state = state.reshape((len(components),) + shape3 + (3,))
    phantom.relaxation_group_map = group_map
    phantom.relaxation_groups = groups
    if equilibrium is not None:
        phantom.spin_equilibrium_map = np.ascontiguousarray(equilibrium)
    n_groups = 0 if groups is None else int(groups.shape[0])
    return phantom, density, n_groups


def _phantom_field(phantom) -> Tuple[float, str]:
    metadata = getattr(phantom, "metadata", None) or {}
    nucleus = str(metadata.get("nucleus") or "H1")
    if nucleus not in NUCLEUS_GAMMA_HZ_PER_T:
        nucleus = "H1"
    try:
        field_strength = float(metadata.get("field_strength_t") or 3.0)
    except (TypeError, ValueError):
        field_strength = 3.0
    if not np.isfinite(field_strength) or field_strength <= 0:
        field_strength = 3.0
    return field_strength, nucleus


def run_static_components(
    program,
    components: Sequence[Tuple[str, object]],
    *,
    signal_weighting="voxel",
    equilibrium_mz=None,
    kernel="production",
    use_parallel=True,
    num_threads=1,
    memory_budget_bytes=None,
    checkpoints_s=(),
    simulation_timestep_s=None,
    spin_sampling=None,
    spoiler_mode="ideal",
    checkpoint_dtype=np.float64,
    progress_callback=None,
    preview_callback=None,
    cancel_callback=None,
    status_callback=None,
):
    """Run static ``(name, Phantom)`` components on the spin-major executor.

    All components share the grid, positions, B0 and coil maps (as the
    components of a spectral phantom do). One component is a plain static
    object; several are independent spectral species. Returns
    ``(outputs, metadata)`` with one :class:`StaticComponentOutput` per
    component. Raises :class:`StaticSpinMajorFallback` before any output
    exists if the executor cannot run the object.
    """
    if kernel == "strict_reference":
        raise StaticSpinMajorFallback(
            "the strict reference kernel has no spin-major executor"
        )
    first = components[0][1]
    single = len(components) == 1
    if first.has_explicit_positions and not single:
        raise StaticSpinMajorFallback("explicit spin positions with several pools")
    weight_scale = 1.0
    if signal_weighting == "voxel_volume":
        weight_scale = float(first.voxel_volume_m3)
    flats = [phantom.get_flat_properties() for _, phantom in components]
    reference = flats[0]
    for flat in flats[1:]:
        for key in ("positions", "tx_sensitivity", "rx_sensitivities", "b0"):
            if not np.array_equal(flat[key], reference[key]):
                raise StaticSpinMajorFallback(f"components differ in {key}")
    items = [
        _component(name, flat, single=single, weight_scale=weight_scale)
        for (name, _), flat in zip(components, flats)
    ]
    union = np.any([item.mask for item in items], axis=0)
    union_indices = np.flatnonzero(union)
    for label, values in (
        ("Tx sensitivity", reference["tx_sensitivity"][union]),
        ("Rx sensitivities", reference["rx_sensitivities"][:, union]),
        ("positions", reference["positions"][union]),
        ("off-resonance", reference["df"][union]),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"active phantom {label} contains NaN or infinite values")

    shape = tuple(int(value) for value in first.shape)
    if len(shape) > 3:
        raise StaticSpinMajorFallback("phantoms with more than three axes")
    shape3 = shape + (1,) * (3 - len(shape))
    fov = [float(value) for value in np.ravel(first.fov)][: len(shape)]
    fov3 = tuple(
        value if np.isfinite(value) and value > 0 else 1.0
        for value in fov + [1.0] * (3 - len(fov))
    )
    affine = np.array(first.affine_ijk_to_xyz_m, dtype=np.float64, copy=True)
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        affine = np.eye(4)
    # Zero the columns beyond the static dimensionality, as the static
    # voxel basis does, so subvoxel offsets agree.
    affine[:3, len(shape) : 3] = 0.0
    positions = np.ascontiguousarray(reference["positions"], dtype=np.float64)
    if positions.shape != (int(np.prod(shape3)), 3):
        raise StaticSpinMajorFallback("phantom positions are not one per voxel")
    b0_hz = np.asarray(reference["df" if single else "b0"], dtype=np.float64)
    tx = np.asarray(reference["tx_sensitivity"], dtype=np.complex128)
    if np.all(tx[union] == 1.0):
        tx = None
    rx = np.asarray(reference["rx_sensitivities"], dtype=np.complex128)
    if single:
        rx = rx * (np.asarray(reference["pd"], dtype=np.float64) * weight_scale)
    equilibrium = None
    if equilibrium_mz is not None:
        equilibrium = np.ascontiguousarray(
            np.broadcast_to(
                np.asarray(equilibrium_mz, dtype=np.float64), first.shape
            ).reshape(-1)
        )
        if np.all(equilibrium[union] == 1.0):
            equilibrium = None
    field_strength, nucleus = _phantom_field(first)

    passes = [
        items[start : start + _SPIN_MAJOR_MAX_POOLS]
        for start in range(0, len(items), _SPIN_MAJOR_MAX_POOLS)
    ]
    built = [
        _pass_phantom(
            group,
            shape3=shape3,
            fov3=fov3,
            affine=affine,
            positions=positions,
            active=union,
            b0_hz=b0_hz,
            tx=tx,
            rx=rx,
            equilibrium=equilibrium,
            field_strength=field_strength,
            nucleus=nucleus,
        )
        for group in passes
    ]
    n_rx = int(rx.shape[0])
    pass_preview = None
    if preview_callback is not None:

        def pass_preview(fraction, signal):
            values = np.asarray(signal)
            if n_rx == 1 and values.ndim == 2:
                values = values[0]
            preview_callback(float(fraction), np.array(values, copy=True))

    kwargs = dict(
        checkpoints_s=checkpoints_s,
        progress_callback=progress_callback,
        preview_callback=pass_preview,
        cancel_callback=cancel_callback,
        status_callback=status_callback,
        simulation_timestep_s=simulation_timestep_s,
        signal_weighting="voxel",
        sequence_kernel="native_parallel" if use_parallel else "native_serial",
        use_parallel=use_parallel,
        num_threads=num_threads,
        memory_budget_bytes=memory_budget_bytes,
        spin_sampling=spin_sampling,
        spoiler_mode=spoiler_mode,
        checkpoint_dtype=checkpoint_dtype,
        _voxel_indices=union_indices,
    )
    try:
        runs = _run_superposed_passes(
            program, [phantom for phantom, _, _ in built], kwargs
        )
    except _SpinMajorRequired as exc:
        raise StaticSpinMajorFallback(exc.reason) from exc

    checkpoint_dtype = np.dtype(checkpoint_dtype)
    outputs: List[StaticComponentOutput] = []
    for group, (_, density, _), run in zip(passes, built, runs):
        species_signal = np.asarray(run.species_signal)
        if species_signal.ndim == 2:
            species_signal = species_signal[:, None, :]
        final_pool = np.asarray(run.final_pool_magnetization, dtype=np.float64)
        checkpoint_pool = run.checkpoint_pool_magnetization
        for index, item in enumerate(group):
            rows = np.searchsorted(union_indices, item.indices)
            unit = density[index][item.indices][:, None]
            final = np.divide(
                final_pool[index][rows],
                unit,
                out=np.zeros((rows.size, 3)),
                where=unit > 0.0,
            )
            if checkpoint_pool is None:
                checkpoints = np.zeros((0, rows.size, 3), dtype=checkpoint_dtype)
            else:
                values = np.asarray(checkpoint_pool[:, index], dtype=np.float64)
                checkpoints = np.divide(
                    values[:, rows],
                    unit[None],
                    out=np.zeros((values.shape[0], rows.size, 3)),
                    where=unit[None] > 0.0,
                ).astype(checkpoint_dtype, copy=False)
            outputs.append(
                StaticComponentOutput(
                    signal=np.array(species_signal[index], copy=True),
                    final_active=final,
                    checkpoints_active=checkpoints,
                )
            )

    run_metadata = runs[0].metadata
    metadata = {
        "static_spin_major": True,
        "static_spin_major_passes": len(runs),
        "static_spin_major_relaxation_groups": [n for _, _, n in built],
        "static_spin_major_intervals": int(
            run_metadata.get("native_spin_major_intervals", 0)
        ),
        "native_sequence_kernel": "spin_major",
        "numerical_contract": "float64-close",
        "progress_unit": PROGRESS_UNIT,
    }
    for key in (
        "native_spin_major_enabled",
        "native_spin_major_threads",
        "native_spin_major_chunks",
        "native_spin_major_kernel_s",
        "sequence_compile_s",
        "execution_plan_build_s",
    ):
        if key in run_metadata:
            metadata[key] = run_metadata[key]
    return outputs, metadata
