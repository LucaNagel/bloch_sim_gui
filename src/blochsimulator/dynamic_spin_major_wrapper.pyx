"""Python entry point for the spin-major two-pool dynamic executor.

The executor applies a bounded slice of a dynamic execution plan in strict
float64. Callers build the compact interval tables in NumPy; this module only
checks shapes, dtypes, and contiguity before releasing the GIL. Repeated RF
segments can optionally be replaced by cached per-spin affine propagators.
"""

import numpy as np
cimport numpy as cnp
from libc.stdint cimport int8_t, int32_t, int64_t

cnp.import_array()


cdef extern from "dynamic_spin_major.h":
    int DSM_TILE

    ctypedef struct dsm_chunk:
        int64_t nspins
        int32_t ncoils
        int32_t n_threads
        int32_t n_blocks
        int32_t track_concentration
        int32_t has_sources
        int32_t concentration_sources
        int32_t ideal_crush
        double r1_pyruvate
        double r1_lactate
        double t2_s[2]
        double pool_offset_hz[2]
        double equilibrium
        const double *positions_m
        const double *b0_hz
        const double *kpl
        const double *delivery
        const double *rx
        const double *weights
        const double *tx
        double *transverse
        double *state
        double *concentration
        double *checkpoints
        int64_t n_checkpoints
        int32_t n_groups
        const double *group_half_s
        const int8_t *group_conversion
        int32_t n_intervals
        const int32_t *interval_group
        const int32_t *interval_rf
        const int32_t *interval_gradient
        const int32_t *interval_source
        const double *interval_phase_s
        int32_t n_rf
        int32_t n_gradient
        int32_t n_source
        const double *rf_table
        const double *rf_inputs
        const double *gradient_table
        const double *source_table
        int64_t n_actions
        const int64_t *action_offsets
        const int8_t *action_kinds
        const int64_t *action_payloads
        int64_t adc_first_payload
        int64_t adc_payload_count
        double *adc_signal
        int32_t n_segments
        const int32_t *segment_first
        const int32_t *segment_end
        const int32_t *segment_entry
        const int64_t *segment_basis
        const double *segment_sources
        int64_t n_basis_rows
        const double *basis_rows
        int32_t n_propagators
        double *propagators
        double *build_seconds

    int dsm_run_chunk(const dsm_chunk *chunk) nogil
    int32_t dsm_propagator_values(
        int32_t track_concentration, int32_t has_sources,
        int32_t concentration_sources)


TILE_SPINS = DSM_TILE
API_VERSION = 2
SPATIAL_TX_SUPPORTED = True

cdef double _empty_storage[8]


cdef void *_data(object array, object dtype, tuple shape, str name) except NULL:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {np.dtype(dtype)}")
    if not array.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if len(shape) != array.ndim:
        raise ValueError(f"{name} must have {len(shape)} dimensions")
    for axis, expected in enumerate(shape):
        if expected is not None and array.shape[axis] != expected:
            raise ValueError(
                f"{name} has shape {array.shape}, expected {shape}")
    if array.size == 0:
        return <void *>_empty_storage
    return cnp.PyArray_DATA(<cnp.ndarray>array)


cdef void *_writable(object array, object dtype, tuple shape, str name) except NULL:
    cdef void *pointer = _data(array, dtype, shape, name)
    if not array.flags.writeable:
        raise ValueError(f"{name} must be writable")
    return pointer


def propagator_values(bint track_concentration, bint has_sources,
                      bint concentration_sources):
    """Return the doubles per spin of one cached RF propagator entry."""
    return int(dsm_propagator_values(
        track_concentration, has_sources, concentration_sources))


def run_spin_major_chunk(
        *,
        positions_m,
        b0_hz,
        kpl,
        delivery,
        rx_sensitivities,
        signal_weights,
        transverse_state,
        state,
        concentration_state,
        checkpoint_states,
        group_half_s,
        group_conversion,
        interval_group,
        interval_rf,
        interval_gradient,
        interval_source,
        interval_phase_s,
        rf_table,
        gradient_table,
        source_table,
        action_offsets,
        action_kinds,
        action_payloads,
        long long adc_first_payload,
        adc_signal,
        double r1_pyruvate,
        double r1_lactate,
        t2_s,
        pool_offset_hz,
        double equilibrium,
        bint ideal_crush,
        bint concentration_sources,
        int num_threads,
        int num_blocks,
        segment_first=None,
        segment_end=None,
        segment_entry=None,
        segment_basis=None,
        segment_sources=None,
        basis_rows=None,
        propagators=None,
        tx_sensitivity=None,
        rf_inputs=None):
    """Advance ``state`` through one plan slice and add unscaled ADC sums.

    ``adc_signal`` has shape ``(2, ncoils, payload_count)`` and receives
    ``sum(transverse * rx * weight)`` per pool for payloads
    ``adc_first_payload + index``. Signal scaling and demodulation stay with
    the caller.

    The optional segment arrays replace intervals ``[first, end)`` by the
    propagator entry ``propagators[entry]`` with shape
    ``(n_tiles, values, TILE_SPINS)``. Segments with ``basis >= 0`` build their
    entry from ``basis_rows[basis:basis + end - first]`` first. Returns the
    build time summed over threads in seconds.
    """
    cdef dsm_chunk chunk
    cdef int status
    cdef double build_seconds = 0.0
    cdef Py_ssize_t n_segments = 0
    cdef Py_ssize_t n_tiles
    cdef Py_ssize_t nspins = kpl.shape[0]
    cdef Py_ssize_t ncoils = rx_sensitivities.shape[0]
    cdef Py_ssize_t n_intervals = interval_group.shape[0]
    cdef Py_ssize_t n_groups = group_half_s.shape[0]
    cdef Py_ssize_t n_checkpoints = checkpoint_states.shape[0]

    if num_threads < 1 or num_blocks < 1:
        raise ValueError("thread and block counts must be positive")
    if (n_intervals >= 2**31 or n_groups >= 2**31 or
            rf_table.shape[0] >= 2**31 or gradient_table.shape[0] >= 2**31 or
            source_table.shape[0] >= 2**31):
        raise ValueError("plan slice is too large for 32-bit table indices")
    if t2_s.shape != (2,) or pool_offset_hz.shape != (2,):
        raise ValueError("t2_s and pool_offset_hz must contain two pools")

    chunk.nspins = nspins
    chunk.ncoils = <int32_t>ncoils
    chunk.n_threads = num_threads
    chunk.n_blocks = num_blocks
    chunk.track_concentration = concentration_state is not None
    chunk.has_sources = delivery is not None
    chunk.concentration_sources = concentration_sources
    chunk.ideal_crush = ideal_crush
    chunk.r1_pyruvate = r1_pyruvate
    chunk.r1_lactate = r1_lactate
    chunk.t2_s[0] = float(t2_s[0])
    chunk.t2_s[1] = float(t2_s[1])
    chunk.pool_offset_hz[0] = float(pool_offset_hz[0])
    chunk.pool_offset_hz[1] = float(pool_offset_hz[1])
    chunk.equilibrium = equilibrium

    chunk.positions_m = <const double *>_data(
        positions_m, np.float64, (nspins, 3), "positions_m")
    chunk.b0_hz = <const double *>_data(b0_hz, np.float64, (nspins,), "b0_hz")
    chunk.kpl = <const double *>_data(kpl, np.float64, (nspins,), "kpl")
    chunk.delivery = (
        NULL if delivery is None else
        <const double *>_data(delivery, np.float64, (nspins,), "delivery"))
    chunk.rx = <const double *>_data(
        rx_sensitivities, np.complex128, (ncoils, nspins), "rx_sensitivities")
    chunk.weights = <const double *>_data(
        signal_weights, np.float64, (nspins,), "signal_weights")
    chunk.tx = (
        NULL if tx_sensitivity is None else
        <const double *>_data(tx_sensitivity, np.complex128, (nspins,), "tx_sensitivity"))
    chunk.rf_inputs = (
        NULL if tx_sensitivity is None else
        <const double *>_data(rf_inputs, np.float64, (rf_table.shape[0], 3), "rf_inputs"))
    chunk.transverse = <double *>_writable(
        transverse_state, np.complex128, (2, nspins), "transverse_state")
    chunk.state = <double *>_writable(state, np.float64, (2, nspins, 3), "state")
    chunk.concentration = (
        NULL if concentration_state is None else
        <double *>_writable(
            concentration_state, np.float64, (2, nspins, 3),
            "concentration_state"))
    if n_checkpoints:
        chunk.checkpoints = <double *>_writable(
            checkpoint_states, np.float64, (n_checkpoints, 2, nspins, 3),
            "checkpoint_states")
    else:
        chunk.checkpoints = NULL
    chunk.n_checkpoints = n_checkpoints

    chunk.n_groups = <int32_t>n_groups
    chunk.group_half_s = <const double *>_data(
        group_half_s, np.float64, (n_groups,), "group_half_s")
    chunk.group_conversion = <const int8_t *>_data(
        group_conversion, np.int8, (n_groups,), "group_conversion")

    chunk.n_intervals = <int32_t>n_intervals
    chunk.interval_group = <const int32_t *>_data(
        interval_group, np.int32, (n_intervals,), "interval_group")
    chunk.interval_rf = <const int32_t *>_data(
        interval_rf, np.int32, (n_intervals,), "interval_rf")
    chunk.interval_gradient = <const int32_t *>_data(
        interval_gradient, np.int32, (n_intervals,), "interval_gradient")
    chunk.interval_source = <const int32_t *>_data(
        interval_source, np.int32, (n_intervals,), "interval_source")
    chunk.interval_phase_s = <const double *>_data(
        interval_phase_s, np.float64, (n_intervals, 2), "interval_phase_s")
    chunk.n_rf = <int32_t>rf_table.shape[0]
    chunk.n_gradient = <int32_t>gradient_table.shape[0]
    chunk.n_source = <int32_t>source_table.shape[0]
    chunk.rf_table = <const double *>_data(
        rf_table, np.float64, (None, 5), "rf_table")
    chunk.gradient_table = <const double *>_data(
        gradient_table, np.float64, (None, 3), "gradient_table")
    chunk.source_table = <const double *>_data(
        source_table, np.float64, (None, 8), "source_table")

    chunk.n_actions = action_kinds.shape[0]
    chunk.action_offsets = <const int64_t *>_data(
        action_offsets, np.int64, (n_intervals + 1,), "action_offsets")
    chunk.action_kinds = <const int8_t *>_data(
        action_kinds, np.int8, (None,), "action_kinds")
    chunk.action_payloads = <const int64_t *>_data(
        action_payloads, np.int64, (chunk.n_actions,), "action_payloads")

    chunk.adc_first_payload = adc_first_payload
    chunk.adc_payload_count = adc_signal.shape[2]
    chunk.adc_signal = <double *>_writable(
        adc_signal, np.complex128, (2, ncoils, None), "adc_signal")

    chunk.n_segments = 0
    chunk.segment_first = NULL
    chunk.segment_end = NULL
    chunk.segment_entry = NULL
    chunk.segment_basis = NULL
    chunk.segment_sources = NULL
    chunk.n_basis_rows = 0
    chunk.basis_rows = NULL
    chunk.n_propagators = 0
    chunk.propagators = NULL
    chunk.build_seconds = &build_seconds
    if segment_first is not None and segment_first.shape[0] > 0:
        n_segments = segment_first.shape[0]
        if n_segments >= 2**31:
            raise ValueError("too many propagated segments")
        if propagators is None or basis_rows is None:
            raise ValueError("propagated segments need propagators and basis rows")
        n_tiles = (nspins + DSM_TILE - 1) // DSM_TILE
        chunk.n_segments = <int32_t>n_segments
        chunk.segment_first = <const int32_t *>_data(
            segment_first, np.int32, (n_segments,), "segment_first")
        chunk.segment_end = <const int32_t *>_data(
            segment_end, np.int32, (n_segments,), "segment_end")
        chunk.segment_entry = <const int32_t *>_data(
            segment_entry, np.int32, (n_segments,), "segment_entry")
        chunk.segment_basis = <const int64_t *>_data(
            segment_basis, np.int64, (n_segments,), "segment_basis")
        chunk.segment_sources = <const double *>_data(
            segment_sources, np.float64, (n_segments, 6), "segment_sources")
        chunk.n_basis_rows = basis_rows.shape[0]
        chunk.basis_rows = <const double *>_data(
            basis_rows, np.float64, (None, 3, 4), "basis_rows")
        if propagators.shape[0] >= 2**31:
            raise ValueError("too many propagator entries")
        chunk.n_propagators = <int32_t>propagators.shape[0]
        chunk.propagators = <double *>_writable(
            propagators, np.float64,
            (None, n_tiles,
             dsm_propagator_values(
                 chunk.track_concentration, chunk.has_sources,
                 chunk.concentration_sources),
             DSM_TILE),
            "propagators")

    with nogil:
        status = dsm_run_chunk(&chunk)
    if status == -2:
        raise MemoryError("spin-major dynamic executor could not allocate buffers")
    if status != 0:
        raise ValueError("spin-major dynamic executor rejected its plan slice")
    return build_seconds
