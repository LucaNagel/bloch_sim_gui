"""Strict floating-point primitives for the dynamic two-pool pilot kernel.

The state-independent longitudinal and RF coefficients are deliberately
computed by NumPy in :mod:`blochsimulator.dynamic_phantom`. This module only
applies those already-rounded values in the same operation order as the
optimized Python implementation. Complex free evolution stays in NumPy; the
RF block reads and writes its complex Mx/My storage as two unchanged doubles.
"""

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel cimport prange
from libc.math cimport cos, exp, hypot, sin


ctypedef np.float64_t DTYPE_t
ctypedef np.npy_bool BOOL_t


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def prepare_spatial_rf_rotation_coefficients(
        double rf_real,
        double rf_imag,
        np.ndarray[np.complex128_t, ndim=1] tx_sensitivity,
        double duration):
    """Prepare spatial-B1 coefficients with chunk-stable scalar libm calls."""
    cdef Py_ssize_t nvoxels = tx_sensitivity.shape[0]
    cdef Py_ssize_t voxel
    cdef double tx_real
    cdef double tx_imag
    cdef double effective_real
    cdef double effective_imag
    cdef double nx
    cdef double ny
    cdef double angle
    cdef double *tx_data
    cdef np.ndarray[DTYPE_t, ndim=1] axis_x = np.empty(nvoxels, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] axis_y = np.empty(nvoxels, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] cosine = np.empty(nvoxels, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] sine = np.empty(nvoxels, dtype=np.float64)

    if not tx_sensitivity.flags.c_contiguous:
        raise ValueError("tx_sensitivity must be C-contiguous")
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    tx_data = <double *>tx_sensitivity.data
    for voxel in range(nvoxels):
        tx_real = tx_data[2 * voxel]
        tx_imag = tx_data[2 * voxel + 1]
        effective_real = rf_real * tx_real - rf_imag * tx_imag
        effective_imag = rf_real * tx_imag + rf_imag * tx_real
        nx = -6.283185307179586476925286766559 * effective_real * duration
        ny = 6.283185307179586476925286766559 * effective_imag * duration
        angle = hypot(nx, ny)
        if angle == 0.0:
            axis_x[voxel] = 1.0
            axis_y[voxel] = 0.0
        else:
            axis_x[voxel] = nx / angle
            axis_y[voxel] = ny / angle
        cosine[voxel] = cos(angle)
        sine[voxel] = sin(angle)
    return axis_x, axis_y, cosine, sine


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def prepare_transverse_factors_coefficients(
        np.ndarray[DTYPE_t, ndim=2] phase_cycles,
        np.ndarray[DTYPE_t, ndim=1] t2_s,
        double duration):
    """Prepare complex free-evolution factors in a chunk-stable order."""
    cdef Py_ssize_t npools = phase_cycles.shape[0]
    cdef Py_ssize_t nvoxels = phase_cycles.shape[1]
    cdef Py_ssize_t pool
    cdef Py_ssize_t voxel
    cdef double decay
    cdef double angle
    cdef double *phase_data
    cdef double *factor_data
    cdef np.ndarray[np.complex128_t, ndim=2] factors = np.empty(
        (npools, nvoxels), dtype=np.complex128)

    if t2_s.shape[0] != npools:
        raise ValueError("t2_s must contain one value per pool")
    if not phase_cycles.flags.c_contiguous or not t2_s.flags.c_contiguous:
        raise ValueError("transverse coefficient inputs must be C-contiguous")
    phase_data = <double *>phase_cycles.data
    factor_data = <double *>factors.data
    for pool in range(npools):
        decay = exp(-duration / t2_s[pool])
        for voxel in range(nvoxels):
            angle = -6.283185307179586476925286766559 * (
                phase_data[pool * nvoxels + voxel])
            factor_data[2 * (pool * nvoxels + voxel)] = decay * cos(angle)
            factor_data[2 * (pool * nvoxels + voxel) + 1] = decay * sin(angle)
    return factors


cdef inline int _advance_coupled_longitudinal_voxel(
        double *pyruvate,
        double *lactate,
        double *concentration_pyruvate,
        double *concentration_lactate,
        double kpl,
        double exp_a,
        double exp_b,
        double difference,
        bint regular,
        double f0_a,
        double f1_a,
        double j0,
        double j1,
        double concentration_exp_a,
        double concentration_exp_b,
        double concentration_difference,
        bint concentration_regular,
        double concentration_f0_a,
        double concentration_f1_a,
        double concentration_j0,
        double concentration_j1,
        double source_start,
        double source_end,
        double concentration_source_start,
        double concentration_source_end,
        double duration,
        double equilibrium,
        int regular_mode,
        int concentration_regular_mode) except -1 nogil:
    cdef double transfer
    cdef double decay_delta
    cdef double source_slope
    cdef double source_delta
    cdef double source_addition

    if equilibrium != 0.0:
        pyruvate[0] = pyruvate[0] - equilibrium * concentration_pyruvate[0]
        lactate[0] = lactate[0] - equilibrium * concentration_lactate[0]

    transfer = kpl * pyruvate[0]
    if regular_mode == 1:
        decay_delta = exp_b - exp_a
        transfer = transfer * decay_delta
        transfer = transfer / difference
    elif regular_mode == -1:
        transfer = transfer * duration
        transfer = transfer * exp_b
    elif regular:
        decay_delta = exp_b - exp_a
        transfer = transfer * decay_delta
        transfer = transfer / difference
    else:
        transfer = transfer * duration
        transfer = transfer * exp_b
    pyruvate[0] = pyruvate[0] * exp_a
    lactate[0] = lactate[0] * exp_b
    lactate[0] = lactate[0] + transfer

    source_start = source_start - equilibrium * concentration_source_start
    source_end = source_end - equilibrium * concentration_source_end
    source_delta = source_end - source_start
    source_slope = source_delta / duration
    source_addition = source_start * f0_a
    source_addition = source_addition + source_slope * f1_a
    pyruvate[0] = pyruvate[0] + source_addition
    source_addition = source_start * j0
    source_addition = source_addition + source_slope * j1
    source_addition = kpl * source_addition
    lactate[0] = lactate[0] + source_addition

    transfer = kpl * concentration_pyruvate[0]
    if concentration_regular_mode == 1:
        decay_delta = concentration_exp_b - concentration_exp_a
        transfer = transfer * decay_delta
        transfer = transfer / concentration_difference
    elif concentration_regular_mode == -1:
        transfer = transfer * duration
        transfer = transfer * concentration_exp_b
    elif concentration_regular:
        decay_delta = concentration_exp_b - concentration_exp_a
        transfer = transfer * decay_delta
        transfer = transfer / concentration_difference
    else:
        transfer = transfer * duration
        transfer = transfer * concentration_exp_b
    concentration_pyruvate[0] = (
        concentration_pyruvate[0] * concentration_exp_a)
    concentration_lactate[0] = (
        concentration_lactate[0] * concentration_exp_b)
    concentration_lactate[0] = concentration_lactate[0] + transfer

    source_delta = concentration_source_end - concentration_source_start
    source_slope = source_delta / duration
    source_addition = concentration_source_start * concentration_f0_a
    source_addition = source_addition + source_slope * concentration_f1_a
    concentration_pyruvate[0] = concentration_pyruvate[0] + source_addition
    source_addition = concentration_source_start * concentration_j0
    source_addition = source_addition + source_slope * concentration_j1
    source_addition = kpl * source_addition
    concentration_lactate[0] = concentration_lactate[0] + source_addition

    if equilibrium != 0.0:
        pyruvate[0] = pyruvate[0] + equilibrium * concentration_pyruvate[0]
        lactate[0] = lactate[0] + equilibrium * concentration_lactate[0]
    return 0


cdef inline int _rotate_rf_voxel(
        double *value_x,
        double *value_y,
        double *value_z,
        double axis_x,
        double axis_y,
        double cosine,
        double sine,
        double one_minus_cosine) except -1 nogil:
    cdef double vx = value_x[0]
    cdef double vy = value_y[0]
    cdef double vz = value_z[0]
    cdef double cross_x = axis_y * vz - 0.0 * vy
    cdef double cross_y = 0.0 * vx - axis_x * vz
    cdef double cross_z = axis_x * vy - axis_y * vx
    cdef double projection = vx * axis_x
    projection = projection + vy * axis_y
    projection = projection + vz * 0.0
    value_x[0] = vx * cosine
    value_x[0] = value_x[0] + cross_x * sine
    value_x[0] = value_x[0] + (projection * axis_x) * one_minus_cosine
    value_y[0] = vy * cosine
    value_y[0] = value_y[0] + cross_y * sine
    value_y[0] = value_y[0] + (projection * axis_y) * one_minus_cosine
    value_z[0] = vz * cosine
    value_z[0] = value_z[0] + cross_z * sine
    value_z[0] = value_z[0] + (projection * 0.0) * one_minus_cosine
    return 0


cdef inline int _rotate_rf_voxel_plane(
        double *value_x,
        double *value_y,
        double *value_z,
        double axis_x,
        double axis_y,
        double cosine,
        double sine) except -1 nogil:
    """Match the established spatial-B1 plane-basis operation order."""
    cdef double vx = value_x[0]
    cdef double vy = value_y[0]
    cdef double vz = value_z[0]
    cdef double parallel = vx * axis_x + vy * axis_y
    cdef double perpendicular = -vx * axis_y + vy * axis_x
    cdef double rotated_perpendicular = perpendicular * cosine - vz * sine

    value_x[0] = parallel * axis_x - rotated_perpendicular * axis_y
    value_y[0] = parallel * axis_y + rotated_perpendicular * axis_x
    value_z[0] = perpendicular * sine + vz * cosine
    return 0


cdef inline int _advance_no_inflow_voxel(
        double *pyruvate,
        double *lactate,
        double kpl,
        double exp_a,
        double exp_b,
        double difference,
        bint regular,
        double duration,
        int regular_mode) except -1 nogil:
    cdef double transfer = kpl * pyruvate[0]
    cdef double decay_delta
    if regular_mode == 1:
        decay_delta = exp_b - exp_a
        transfer = transfer * decay_delta
        transfer = transfer / difference
    elif regular_mode == -1:
        transfer = transfer * duration
        transfer = transfer * exp_b
    elif regular:
        decay_delta = exp_b - exp_a
        transfer = transfer * decay_delta
        transfer = transfer / difference
    else:
        transfer = transfer * duration
        transfer = transfer * exp_b
    pyruvate[0] = pyruvate[0] * exp_a
    lactate[0] = lactate[0] * exp_b
    lactate[0] = lactate[0] + transfer
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def run_dynamic_plan_block_no_inflow(
        np.ndarray[DTYPE_t, ndim=3] state,
        np.ndarray[np.complex128_t, ndim=2] transverse_state,
        np.ndarray[np.complex128_t, ndim=3] transverse_factors,
        np.ndarray[DTYPE_t, ndim=2] kpl_by_group,
        np.ndarray[DTYPE_t, ndim=2] exp_a_by_group,
        np.ndarray[DTYPE_t, ndim=1] exp_b_by_group,
        np.ndarray[DTYPE_t, ndim=2] difference_by_group,
        np.ndarray[BOOL_t, ndim=2] regular_by_group,
        np.ndarray[DTYPE_t, ndim=1] duration_by_group,
        np.ndarray[np.int32_t, ndim=1] regular_mode_by_group,
        np.ndarray[np.int32_t, ndim=1] longitudinal_groups,
        np.ndarray[np.int32_t, ndim=1] first_factor_groups,
        np.ndarray[np.int32_t, ndim=1] second_factor_groups,
        np.ndarray[np.int32_t, ndim=1] rf_groups,
        np.ndarray[DTYPE_t, ndim=2] rf_axis_x,
        np.ndarray[DTYPE_t, ndim=2] rf_axis_y,
        np.ndarray[DTYPE_t, ndim=2] rf_cosine,
        np.ndarray[DTYPE_t, ndim=2] rf_sine,
        np.ndarray[DTYPE_t, ndim=2] rf_one_minus_cosine,
        np.ndarray[np.int32_t, ndim=1] action_offsets,
        np.ndarray[np.int8_t, ndim=1] action_kinds,
        np.ndarray[np.int64_t, ndim=1] action_payloads,
        np.ndarray[np.complex128_t, ndim=1] adc_demodulation,
        np.ndarray[np.complex128_t, ndim=2] rx_sensitivities,
        np.ndarray[DTYPE_t, ndim=1] signal_weights,
        np.ndarray[np.complex128_t, ndim=4] parent_species_signal,
        np.ndarray[DTYPE_t, ndim=4] checkpoint_states,
        int spins_per_parent,
        double signal_scale,
        bint ideal_crushers,
        int num_threads):
    """Execute complete two-pool plan intervals and observations natively."""
    cdef Py_ssize_t nspins = state.shape[1]
    cdef Py_ssize_t nsteps = longitudinal_groups.shape[0]
    cdef Py_ssize_t nparents = parent_species_signal.shape[0]
    cdef Py_ssize_t ncoils = rx_sensitivities.shape[0]
    cdef Py_ssize_t nadc = parent_species_signal.shape[3]
    cdef Py_ssize_t parent
    cdef Py_ssize_t local_spin
    cdef Py_ssize_t spin
    cdef Py_ssize_t step
    cdef Py_ssize_t pool
    cdef Py_ssize_t coil
    cdef Py_ssize_t action
    cdef Py_ssize_t group
    cdef Py_ssize_t factor_group
    cdef Py_ssize_t rf_group
    cdef Py_ssize_t width
    cdef Py_ssize_t coefficient_index
    cdef Py_ssize_t factor_offset
    cdef Py_ssize_t signal_offset
    cdef Py_ssize_t checkpoint_offset
    cdef long payload
    cdef signed char action_kind
    cdef double pyruvate_x
    cdef double pyruvate_y
    cdef double pyruvate_z
    cdef double lactate_x
    cdef double lactate_y
    cdef double lactate_z
    cdef double old_x
    cdef double factor_real
    cdef double factor_imag
    cdef double transverse_real
    cdef double transverse_imag
    cdef double rx_real
    cdef double rx_imag
    cdef double demod_real
    cdef double demod_imag
    cdef double received_real
    cdef double received_imag
    cdef double weighted_real
    cdef double weighted_imag
    cdef double *state_data = <double *>state.data
    cdef double *transverse_data = <double *>transverse_state.data
    cdef double *factor_data = <double *>transverse_factors.data
    cdef double *rx_data = <double *>rx_sensitivities.data
    cdef double *demod_data = <double *>adc_demodulation.data
    cdef double *signal_data = <double *>parent_species_signal.data
    cdef double *checkpoint_data = <double *>checkpoint_states.data

    if state.shape[0] != 2 or state.shape[2] != 3:
        raise ValueError("state must have shape (2, nspins, 3)")
    if transverse_state.shape[0] != 2 or transverse_state.shape[1] != nspins:
        raise ValueError("transverse_state must have shape (2, nspins)")
    if spins_per_parent < 1 or nparents * spins_per_parent != nspins:
        raise ValueError("parent/spin dimensions are inconsistent")
    if parent_species_signal.shape[1] != 2 or parent_species_signal.shape[2] != ncoils:
        raise ValueError("parent signal shape is inconsistent")
    if rx_sensitivities.shape[1] != nspins or signal_weights.shape[0] != nspins:
        raise ValueError("receive arrays must contain every spin")
    if action_offsets.shape[0] != nsteps + 1:
        raise ValueError("action offsets must bracket every plan step")
    if num_threads < 1:
        raise ValueError("num_threads must be positive")
    if not state.flags.c_contiguous or not transverse_state.flags.c_contiguous:
        raise ValueError("state arrays must be C-contiguous")
    if not transverse_factors.flags.c_contiguous:
        raise ValueError("transverse factors must be C-contiguous")
    width = rf_axis_x.shape[1]
    if width != 1 and width != nspins:
        raise ValueError("RF coefficient width must be one or nspins")

    for parent in prange(
            nparents, nogil=True, schedule="static", num_threads=num_threads):
        for local_spin in range(spins_per_parent):
            spin = parent * spins_per_parent + local_spin
            pyruvate_x = transverse_data[2 * spin]
            pyruvate_y = transverse_data[2 * spin + 1]
            lactate_x = transverse_data[2 * (nspins + spin)]
            lactate_y = transverse_data[2 * (nspins + spin) + 1]
            pyruvate_z = state_data[3 * spin + 2]
            lactate_z = state_data[3 * (nspins + spin) + 2]

            for step in range(nsteps):
                factor_group = first_factor_groups[step]
                factor_offset = 2 * ((factor_group * 2) * nspins + spin)
                factor_real = factor_data[factor_offset]
                factor_imag = factor_data[factor_offset + 1]
                old_x = pyruvate_x
                pyruvate_x = old_x * factor_real - pyruvate_y * factor_imag
                pyruvate_y = old_x * factor_imag + pyruvate_y * factor_real
                factor_offset = 2 * ((factor_group * 2 + 1) * nspins + spin)
                factor_real = factor_data[factor_offset]
                factor_imag = factor_data[factor_offset + 1]
                old_x = lactate_x
                lactate_x = old_x * factor_real - lactate_y * factor_imag
                lactate_y = old_x * factor_imag + lactate_y * factor_real

                group = longitudinal_groups[step]
                _advance_no_inflow_voxel(
                    &pyruvate_z, &lactate_z,
                    kpl_by_group[group, spin], exp_a_by_group[group, spin],
                    exp_b_by_group[group], difference_by_group[group, spin],
                    regular_by_group[group, spin], duration_by_group[group],
                    regular_mode_by_group[group])

                rf_group = rf_groups[step]
                if rf_group >= 0:
                    coefficient_index = 0 if width == 1 else spin
                    if width == 1:
                        _rotate_rf_voxel(
                            &pyruvate_x, &pyruvate_y, &pyruvate_z,
                            rf_axis_x[rf_group, coefficient_index],
                            rf_axis_y[rf_group, coefficient_index],
                            rf_cosine[rf_group, coefficient_index],
                            rf_sine[rf_group, coefficient_index],
                            rf_one_minus_cosine[rf_group, coefficient_index])
                        _rotate_rf_voxel(
                            &lactate_x, &lactate_y, &lactate_z,
                            rf_axis_x[rf_group, coefficient_index],
                            rf_axis_y[rf_group, coefficient_index],
                            rf_cosine[rf_group, coefficient_index],
                            rf_sine[rf_group, coefficient_index],
                            rf_one_minus_cosine[rf_group, coefficient_index])
                    else:
                        _rotate_rf_voxel_plane(
                            &pyruvate_x, &pyruvate_y, &pyruvate_z,
                            rf_axis_x[rf_group, coefficient_index],
                            rf_axis_y[rf_group, coefficient_index],
                            rf_cosine[rf_group, coefficient_index],
                            rf_sine[rf_group, coefficient_index])
                        _rotate_rf_voxel_plane(
                            &lactate_x, &lactate_y, &lactate_z,
                            rf_axis_x[rf_group, coefficient_index],
                            rf_axis_y[rf_group, coefficient_index],
                            rf_cosine[rf_group, coefficient_index],
                            rf_sine[rf_group, coefficient_index])

                factor_group = second_factor_groups[step]
                factor_offset = 2 * ((factor_group * 2) * nspins + spin)
                factor_real = factor_data[factor_offset]
                factor_imag = factor_data[factor_offset + 1]
                old_x = pyruvate_x
                pyruvate_x = old_x * factor_real - pyruvate_y * factor_imag
                pyruvate_y = old_x * factor_imag + pyruvate_y * factor_real
                factor_offset = 2 * ((factor_group * 2 + 1) * nspins + spin)
                factor_real = factor_data[factor_offset]
                factor_imag = factor_data[factor_offset + 1]
                old_x = lactate_x
                lactate_x = old_x * factor_real - lactate_y * factor_imag
                lactate_y = old_x * factor_imag + lactate_y * factor_real
                _advance_no_inflow_voxel(
                    &pyruvate_z, &lactate_z,
                    kpl_by_group[group, spin], exp_a_by_group[group, spin],
                    exp_b_by_group[group], difference_by_group[group, spin],
                    regular_by_group[group, spin], duration_by_group[group],
                    regular_mode_by_group[group])

                for action in range(action_offsets[step], action_offsets[step + 1]):
                    action_kind = action_kinds[action]
                    payload = action_payloads[action]
                    if action_kind == 1 and ideal_crushers:
                        pyruvate_x = 0.0
                        pyruvate_y = 0.0
                        lactate_x = 0.0
                        lactate_y = 0.0
                    elif action_kind == 2:
                        demod_real = demod_data[2 * payload]
                        demod_imag = demod_data[2 * payload + 1]
                        for coil in range(ncoils):
                            rx_real = rx_data[2 * (coil * nspins + spin)]
                            rx_imag = rx_data[2 * (coil * nspins + spin) + 1]
                            transverse_real = (
                                pyruvate_x * rx_real - pyruvate_y * rx_imag)
                            transverse_imag = (
                                pyruvate_x * rx_imag + pyruvate_y * rx_real)
                            weighted_real = (
                                transverse_real * signal_weights[spin] * signal_scale)
                            weighted_imag = (
                                transverse_imag * signal_weights[spin] * signal_scale)
                            received_real = (
                                weighted_real * demod_real - weighted_imag * demod_imag)
                            received_imag = (
                                weighted_real * demod_imag + weighted_imag * demod_real)
                            signal_offset = 2 * (
                                ((parent * 2) * ncoils + coil) * nadc + payload)
                            signal_data[signal_offset] = (
                                signal_data[signal_offset] + received_real)
                            signal_data[signal_offset + 1] = (
                                signal_data[signal_offset + 1] + received_imag)

                            transverse_real = (
                                lactate_x * rx_real - lactate_y * rx_imag)
                            transverse_imag = (
                                lactate_x * rx_imag + lactate_y * rx_real)
                            weighted_real = (
                                transverse_real * signal_weights[spin] * signal_scale)
                            weighted_imag = (
                                transverse_imag * signal_weights[spin] * signal_scale)
                            received_real = (
                                weighted_real * demod_real - weighted_imag * demod_imag)
                            received_imag = (
                                weighted_real * demod_imag + weighted_imag * demod_real)
                            signal_offset = 2 * (
                                ((parent * 2 + 1) * ncoils + coil) * nadc + payload)
                            signal_data[signal_offset] = (
                                signal_data[signal_offset] + received_real)
                            signal_data[signal_offset + 1] = (
                                signal_data[signal_offset + 1] + received_imag)
                    elif action_kind == 3:
                        checkpoint_offset = 3 * (
                            (payload * 2) * nspins + spin)
                        checkpoint_data[checkpoint_offset] = pyruvate_x
                        checkpoint_data[checkpoint_offset + 1] = pyruvate_y
                        checkpoint_data[checkpoint_offset + 2] = pyruvate_z
                        checkpoint_offset = 3 * (
                            (payload * 2 + 1) * nspins + spin)
                        checkpoint_data[checkpoint_offset] = lactate_x
                        checkpoint_data[checkpoint_offset + 1] = lactate_y
                        checkpoint_data[checkpoint_offset + 2] = lactate_z

            transverse_data[2 * spin] = pyruvate_x
            transverse_data[2 * spin + 1] = pyruvate_y
            transverse_data[2 * (nspins + spin)] = lactate_x
            transverse_data[2 * (nspins + spin) + 1] = lactate_y
            state_data[3 * spin + 2] = pyruvate_z
            state_data[3 * (nspins + spin) + 2] = lactate_z


cdef inline int _advance_multi_pool_voxel(
        double *state_data,
        double *concentration_data,
        Py_ssize_t npools,
        Py_ssize_t nspins,
        Py_ssize_t spin,
        Py_ssize_t group,
        double *conversion_data,
        double *exp_precursor_data,
        double *exp_products_data,
        double *transfer_factor_data,
        double *f0_precursor_data,
        double *f1_precursor_data,
        double *j0_data,
        double *j1_data,
        double *concentration_exp_precursor_data,
        double *concentration_exp_products_data,
        double *concentration_transfer_factor_data,
        double *concentration_f0_precursor_data,
        double *concentration_f1_precursor_data,
        double *concentration_j0_data,
        double *concentration_j1_data,
        double duration,
        double source_start,
        double source_end,
        double concentration_source_start,
        double concentration_source_end,
        double equilibrium) noexcept nogil:
    cdef Py_ssize_t nproducts = npools - 1
    cdef Py_ssize_t pool
    cdef Py_ssize_t product
    cdef Py_ssize_t state_offset
    cdef Py_ssize_t coefficient_offset
    cdef double precursor
    cdef double concentration_precursor
    cdef double precursor_source_start
    cdef double precursor_source_end
    cdef double source_slope
    cdef double concentration_source_slope
    cdef double rate
    cdef double transfer

    if equilibrium != 0.0:
        for pool in range(npools):
            state_offset = 3 * (pool * nspins + spin) + 2
            state_data[state_offset] = state_data[state_offset] - (
                equilibrium * concentration_data[state_offset])

    state_offset = 3 * spin + 2
    precursor = state_data[state_offset]
    concentration_precursor = concentration_data[state_offset]
    precursor_source_start = source_start - (
        equilibrium * concentration_source_start)
    precursor_source_end = source_end - (
        equilibrium * concentration_source_end)
    source_slope = (precursor_source_end - precursor_source_start) / duration
    concentration_source_slope = (
        concentration_source_end - concentration_source_start) / duration

    state_data[state_offset] = (
        precursor * exp_precursor_data[group * nspins + spin]
        + precursor_source_start * f0_precursor_data[group * nspins + spin]
        + source_slope * f1_precursor_data[group * nspins + spin])
    concentration_data[state_offset] = (
        concentration_precursor
        * concentration_exp_precursor_data[group * nspins + spin]
        + concentration_source_start
        * concentration_f0_precursor_data[group * nspins + spin]
        + concentration_source_slope
        * concentration_f1_precursor_data[group * nspins + spin])

    for product in range(nproducts):
        pool = product + 1
        coefficient_offset = (
            (group * nproducts + product) * nspins + spin)
        rate = conversion_data[coefficient_offset]
        state_offset = 3 * (pool * nspins + spin) + 2
        transfer = rate * precursor
        transfer = transfer * transfer_factor_data[coefficient_offset]
        state_data[state_offset] = (
            state_data[state_offset]
            * exp_products_data[group * nproducts + product]
            + transfer
            + rate
            * (
                precursor_source_start * j0_data[coefficient_offset]
                + source_slope * j1_data[coefficient_offset]
            )
        )
        transfer = rate * concentration_precursor
        transfer = (
            transfer * concentration_transfer_factor_data[coefficient_offset])
        concentration_data[state_offset] = (
            concentration_data[state_offset]
            * concentration_exp_products_data[group * nproducts + product]
            + transfer
            + rate
            * (
                concentration_source_start
                * concentration_j0_data[coefficient_offset]
                + concentration_source_slope
                * concentration_j1_data[coefficient_offset]
            )
        )

    if equilibrium != 0.0:
        for pool in range(npools):
            state_offset = 3 * (pool * nspins + spin) + 2
            state_data[state_offset] = state_data[state_offset] + (
                equilibrium * concentration_data[state_offset])
    return 0


cdef inline void _scale_multi_pool_voxel(
        double *state_data,
        double *concentration_data,
        double *transverse_data,
        Py_ssize_t npools,
        Py_ssize_t nspins,
        Py_ssize_t spin,
        double factor) noexcept nogil:
    cdef Py_ssize_t pool
    cdef Py_ssize_t offset
    for pool in range(npools):
        offset = 2 * (pool * nspins + spin)
        transverse_data[offset] = transverse_data[offset] * factor
        transverse_data[offset + 1] = transverse_data[offset + 1] * factor
        offset = 3 * (pool * nspins + spin) + 2
        state_data[offset] = state_data[offset] * factor
        concentration_data[offset] = concentration_data[offset] * factor


cdef inline void _add_washout_gains(
        double *state_data,
        double *concentration_data,
        Py_ssize_t npools,
        Py_ssize_t nspins,
        Py_ssize_t spin,
        int *compartment_data,
        int *row_data,
        double *table_data,
        Py_ssize_t compartments,
        Py_ssize_t step,
        Py_ssize_t half) noexcept nogil:
    cdef Py_ssize_t pool
    cdef Py_ssize_t offset
    cdef Py_ssize_t gain
    cdef int compartment = compartment_data[spin]
    cdef int row = row_data[step]
    if compartment < 0 or row < 0:
        return
    gain = 2 * npools * (compartment + compartments * (half + 2 * row))
    for pool in range(npools):
        offset = 3 * (pool * nspins + spin) + 2
        state_data[offset] = state_data[offset] + table_data[gain + 2 * pool]
        concentration_data[offset] = (
            concentration_data[offset] + table_data[gain + 2 * pool + 1])


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def run_dynamic_plan_block_multi_pool(
        np.ndarray[DTYPE_t, ndim=3] state,
        np.ndarray[np.complex128_t, ndim=2] transverse_state,
        np.ndarray[DTYPE_t, ndim=3] concentration_state,
        np.ndarray[np.complex128_t, ndim=3] transverse_factors,
        np.ndarray[DTYPE_t, ndim=3] conversion_rates_by_group,
        np.ndarray[DTYPE_t, ndim=2] exp_precursor_by_group,
        np.ndarray[DTYPE_t, ndim=2] exp_products_by_group,
        np.ndarray[DTYPE_t, ndim=3] transfer_factor_by_group,
        np.ndarray[DTYPE_t, ndim=2] f0_precursor_by_group,
        np.ndarray[DTYPE_t, ndim=2] f1_precursor_by_group,
        np.ndarray[DTYPE_t, ndim=3] j0_by_group,
        np.ndarray[DTYPE_t, ndim=3] j1_by_group,
        np.ndarray[DTYPE_t, ndim=2] concentration_exp_precursor_by_group,
        np.ndarray[DTYPE_t, ndim=2] concentration_exp_products_by_group,
        np.ndarray[DTYPE_t, ndim=3] concentration_transfer_factor_by_group,
        np.ndarray[DTYPE_t, ndim=2] concentration_f0_precursor_by_group,
        np.ndarray[DTYPE_t, ndim=2] concentration_f1_precursor_by_group,
        np.ndarray[DTYPE_t, ndim=3] concentration_j0_by_group,
        np.ndarray[DTYPE_t, ndim=3] concentration_j1_by_group,
        np.ndarray[DTYPE_t, ndim=1] duration_by_group,
        np.ndarray[np.int32_t, ndim=1] longitudinal_groups,
        np.ndarray[np.int32_t, ndim=1] first_factor_groups,
        np.ndarray[np.int32_t, ndim=1] second_factor_groups,
        np.ndarray[np.int32_t, ndim=1] rf_groups,
        np.ndarray[DTYPE_t, ndim=2] rf_axis_x,
        np.ndarray[DTYPE_t, ndim=2] rf_axis_y,
        np.ndarray[DTYPE_t, ndim=2] rf_cosine,
        np.ndarray[DTYPE_t, ndim=2] rf_sine,
        np.ndarray[DTYPE_t, ndim=2] rf_one_minus_cosine,
        np.ndarray[DTYPE_t, ndim=1] inflow_first_start,
        np.ndarray[DTYPE_t, ndim=1] inflow_first_end,
        np.ndarray[DTYPE_t, ndim=1] inflow_second_start,
        np.ndarray[DTYPE_t, ndim=1] inflow_second_end,
        np.ndarray[DTYPE_t, ndim=1] polarization_first_start,
        np.ndarray[DTYPE_t, ndim=1] polarization_first_end,
        np.ndarray[DTYPE_t, ndim=1] polarization_second_start,
        np.ndarray[DTYPE_t, ndim=1] polarization_second_end,
        np.ndarray[DTYPE_t, ndim=1] inflow_delivery,
        double equilibrium,
        int source_mode,
        np.ndarray[np.int32_t, ndim=1] action_offsets,
        np.ndarray[np.int8_t, ndim=1] action_kinds,
        np.ndarray[np.int64_t, ndim=1] action_payloads,
        np.ndarray[np.complex128_t, ndim=1] adc_demodulation,
        np.ndarray[np.complex128_t, ndim=2] rx_sensitivities,
        np.ndarray[DTYPE_t, ndim=1] signal_weights,
        np.ndarray[np.complex128_t, ndim=4] parent_species_signal,
        np.ndarray[DTYPE_t, ndim=4] checkpoint_states,
        int spins_per_parent,
        double signal_scale,
        bint ideal_crushers,
        int num_threads,
        np.ndarray[DTYPE_t, ndim=3] spatial_source_values=None,
        np.ndarray[DTYPE_t, ndim=1] washout_loss=None,
        np.ndarray[np.int32_t, ndim=1] washout_compartment=None,
        np.ndarray[np.int32_t, ndim=1] washout_rows=None,
        np.ndarray[DTYPE_t, ndim=5] washout_table=None):
    """Execute a bounded multi-pool precursor-to-products plan block.

    Voxel washout scales every pool's x, y, z and concentration by
    ``exp(-washout_loss * duration / 2)`` before and after each free half
    step. Spins with ``washout_compartment >= 0`` receive the gains
    ``washout_table[washout_rows[step], half, compartment, pool]``
    (magnetization, concentration) after each longitudinal half step.
    """
    cdef Py_ssize_t npools = state.shape[0]
    cdef Py_ssize_t nproducts = npools - 1
    cdef Py_ssize_t nspins = state.shape[1]
    cdef Py_ssize_t nsteps = longitudinal_groups.shape[0]
    cdef Py_ssize_t nparents = parent_species_signal.shape[0]
    cdef Py_ssize_t ncoils = rx_sensitivities.shape[0]
    cdef Py_ssize_t nadc = parent_species_signal.shape[3]
    cdef Py_ssize_t parent
    cdef Py_ssize_t local_spin
    cdef Py_ssize_t spin
    cdef Py_ssize_t step
    cdef Py_ssize_t pool
    cdef Py_ssize_t coil
    cdef Py_ssize_t action
    cdef Py_ssize_t group
    cdef Py_ssize_t factor_group
    cdef Py_ssize_t rf_group
    cdef Py_ssize_t width
    cdef Py_ssize_t coefficient_index
    cdef Py_ssize_t transverse_offset
    cdef Py_ssize_t factor_offset
    cdef Py_ssize_t state_offset
    cdef Py_ssize_t signal_offset
    cdef Py_ssize_t checkpoint_offset
    cdef long payload
    cdef signed char action_kind
    cdef double old_x
    cdef double factor_real
    cdef double factor_imag
    cdef double value_x
    cdef double value_y
    cdef double value_z
    cdef double source_start
    cdef double source_end
    cdef double concentration_source_start
    cdef double concentration_source_end
    cdef double rx_real
    cdef double rx_imag
    cdef double demod_real
    cdef double demod_imag
    cdef double transverse_real
    cdef double transverse_imag
    cdef double weighted_real
    cdef double weighted_imag
    cdef double received_real
    cdef double received_imag
    cdef double washout_factor
    cdef Py_ssize_t washout_offset
    cdef bint has_washout_loss = washout_loss is not None
    cdef bint has_washout_sources = washout_compartment is not None
    cdef double *washout_loss_data = NULL
    cdef int *washout_compartment_data = NULL
    cdef int *washout_row_data = NULL
    cdef double *washout_table_data = NULL
    cdef Py_ssize_t washout_compartments = 0
    cdef bint has_spatial_sources = spatial_source_values is not None
    cdef double *state_data = <double *>state.data
    cdef double *concentration_data = <double *>concentration_state.data
    cdef double *transverse_data = <double *>transverse_state.data
    cdef double *factor_data = <double *>transverse_factors.data
    cdef double *conversion_data = <double *>conversion_rates_by_group.data
    cdef double *exp_precursor_data = <double *>exp_precursor_by_group.data
    cdef double *exp_products_data = <double *>exp_products_by_group.data
    cdef double *transfer_factor_data = <double *>transfer_factor_by_group.data
    cdef double *f0_precursor_data = <double *>f0_precursor_by_group.data
    cdef double *f1_precursor_data = <double *>f1_precursor_by_group.data
    cdef double *j0_data = <double *>j0_by_group.data
    cdef double *j1_data = <double *>j1_by_group.data
    cdef double *concentration_exp_precursor_data = (
        <double *>concentration_exp_precursor_by_group.data)
    cdef double *concentration_exp_products_data = (
        <double *>concentration_exp_products_by_group.data)
    cdef double *concentration_transfer_factor_data = (
        <double *>concentration_transfer_factor_by_group.data)
    cdef double *concentration_f0_precursor_data = (
        <double *>concentration_f0_precursor_by_group.data)
    cdef double *concentration_f1_precursor_data = (
        <double *>concentration_f1_precursor_by_group.data)
    cdef double *concentration_j0_data = <double *>concentration_j0_by_group.data
    cdef double *concentration_j1_data = <double *>concentration_j1_by_group.data
    cdef double *rx_data = <double *>rx_sensitivities.data
    cdef double *demod_data = <double *>adc_demodulation.data
    cdef double *signal_data = <double *>parent_species_signal.data
    cdef double *checkpoint_data = <double *>checkpoint_states.data

    if npools < 3 or state.shape[2] != 3:
        raise ValueError("multi-pool state must have shape (>=3, nspins, 3)")
    if (transverse_state.shape[0] != npools or
            transverse_state.shape[1] != nspins):
        raise ValueError("transverse state must match multi-pool state")
    if (concentration_state.shape[0] != npools or
            concentration_state.shape[1] != nspins or
            concentration_state.shape[2] != 3):
        raise ValueError("concentration state must match multi-pool state")
    if (conversion_rates_by_group.shape[1] != nproducts or
            conversion_rates_by_group.shape[2] != nspins):
        raise ValueError("conversion tables must contain every product and spin")
    if spins_per_parent < 1 or nparents * spins_per_parent != nspins:
        raise ValueError("parent/spin dimensions are inconsistent")
    if (parent_species_signal.shape[1] != npools or
            parent_species_signal.shape[2] != ncoils):
        raise ValueError("parent signal shape is inconsistent")
    if action_offsets.shape[0] != nsteps + 1:
        raise ValueError("action offsets must bracket every plan step")
    if source_mode < 0 or source_mode > 3:
        raise ValueError("source_mode must be between zero and three")
    if num_threads < 1:
        raise ValueError("num_threads must be positive")
    width = rf_axis_x.shape[1]
    if width != 1 and width != nspins:
        raise ValueError("RF coefficient width must be one or nspins")
    if has_washout_loss:
        if washout_loss.shape[0] != nspins:
            raise ValueError("washout loss must contain every spin")
        washout_loss_data = <double *>washout_loss.data
    if has_washout_sources:
        if washout_rows is None or washout_table is None:
            raise ValueError("washout sources need rows and gains")
        if (washout_compartment.shape[0] != nspins or
                washout_rows.shape[0] != nsteps or
                washout_table.shape[1] != 2 or
                washout_table.shape[3] != npools or
                washout_table.shape[4] != 2):
            raise ValueError("washout source tables are inconsistent")
        washout_compartments = washout_table.shape[2]
        for step in range(nsteps):
            if (washout_rows[step] < -1 or
                    washout_rows[step] >= washout_table.shape[0]):
                raise ValueError("washout rows must index the gain table")
        for spin in range(nspins):
            if (washout_compartment[spin] < -1 or
                    washout_compartment[spin] >= washout_compartments):
                raise ValueError("washout compartments must index the gains")
        washout_compartment_data = <int *>washout_compartment.data
        washout_row_data = <int *>washout_rows.data
        washout_table_data = <double *>washout_table.data

    for parent in prange(
            nparents, nogil=True, schedule="static", num_threads=num_threads):
      for local_spin in range(spins_per_parent):
        spin = parent * spins_per_parent + local_spin
        for step in range(nsteps):
            group = longitudinal_groups[step]
            washout_factor = 1.0
            if has_washout_loss:
                washout_factor = exp(
                    -washout_loss_data[spin] * (duration_by_group[group] / 2.0))
                _scale_multi_pool_voxel(
                    state_data, concentration_data, transverse_data, npools,
                    nspins, spin, washout_factor)
            factor_group = first_factor_groups[step]
            for pool in range(npools):
                transverse_offset = 2 * (pool * nspins + spin)
                factor_offset = 2 * (
                    (factor_group * npools + pool) * nspins + spin)
                factor_real = factor_data[factor_offset]
                factor_imag = factor_data[factor_offset + 1]
                old_x = transverse_data[transverse_offset]
                transverse_data[transverse_offset] = (
                    old_x * factor_real
                    - transverse_data[transverse_offset + 1] * factor_imag)
                transverse_data[transverse_offset + 1] = (
                    old_x * factor_imag
                    + transverse_data[transverse_offset + 1] * factor_real)

            if source_mode == 2:
                concentration_source_start = 0.0
                concentration_source_end = 0.0
                source_start = 0.0
                source_end = 0.0
            elif source_mode == 1:
                if has_spatial_sources:
                    source_start = (
                        inflow_delivery[spin]
                        * spatial_source_values[0, step, spin])
                    source_end = (
                        inflow_delivery[spin]
                        * spatial_source_values[1, step, spin])
                else:
                    source_start = (
                        inflow_delivery[spin] * inflow_first_start[step])
                    source_end = (
                        inflow_delivery[spin] * inflow_first_end[step])
                concentration_source_start = 0.0
                concentration_source_end = 0.0
            else:
                if has_spatial_sources:
                    concentration_source_start = (
                        inflow_delivery[spin]
                        * spatial_source_values[0, step, spin])
                    concentration_source_end = (
                        inflow_delivery[spin]
                        * spatial_source_values[1, step, spin])
                    source_start = (
                        concentration_source_start
                        * spatial_source_values[4, step, spin])
                    source_end = (
                        concentration_source_end
                        * spatial_source_values[5, step, spin])
                else:
                    concentration_source_start = (
                        inflow_delivery[spin] * inflow_first_start[step])
                    concentration_source_end = (
                        inflow_delivery[spin] * inflow_first_end[step])
                    source_start = (
                        concentration_source_start
                        * polarization_first_start[step])
                    source_end = (
                        concentration_source_end
                        * polarization_first_end[step])
                if source_mode == 3:
                    concentration_source_start = 0.0
                    concentration_source_end = 0.0
            _advance_multi_pool_voxel(
                state_data, concentration_data, npools, nspins, spin, group,
                conversion_data, exp_precursor_data, exp_products_data,
                transfer_factor_data, f0_precursor_data, f1_precursor_data,
                j0_data, j1_data, concentration_exp_precursor_data,
                concentration_exp_products_data,
                concentration_transfer_factor_data,
                concentration_f0_precursor_data,
                concentration_f1_precursor_data,
                concentration_j0_data, concentration_j1_data,
                duration_by_group[group], source_start, source_end,
                concentration_source_start, concentration_source_end,
                equilibrium)
            if has_washout_sources:
                _add_washout_gains(
                    state_data, concentration_data, npools, nspins, spin,
                    washout_compartment_data, washout_row_data,
                    washout_table_data, washout_compartments, step, 0)
            if has_washout_loss:
                _scale_multi_pool_voxel(
                    state_data, concentration_data, transverse_data, npools,
                    nspins, spin, washout_factor * washout_factor)

            rf_group = rf_groups[step]
            if rf_group >= 0:
                coefficient_index = 0 if width == 1 else spin
                for pool in range(npools):
                    transverse_offset = 2 * (pool * nspins + spin)
                    state_offset = 3 * (pool * nspins + spin) + 2
                    if width == 1:
                        _rotate_rf_voxel(
                            &transverse_data[transverse_offset],
                            &transverse_data[transverse_offset + 1],
                            &state_data[state_offset],
                            rf_axis_x[rf_group, coefficient_index],
                            rf_axis_y[rf_group, coefficient_index],
                            rf_cosine[rf_group, coefficient_index],
                            rf_sine[rf_group, coefficient_index],
                            rf_one_minus_cosine[rf_group, coefficient_index])
                    else:
                        _rotate_rf_voxel_plane(
                            &transverse_data[transverse_offset],
                            &transverse_data[transverse_offset + 1],
                            &state_data[state_offset],
                            rf_axis_x[rf_group, coefficient_index],
                            rf_axis_y[rf_group, coefficient_index],
                            rf_cosine[rf_group, coefficient_index],
                            rf_sine[rf_group, coefficient_index])

            factor_group = second_factor_groups[step]
            for pool in range(npools):
                transverse_offset = 2 * (pool * nspins + spin)
                factor_offset = 2 * (
                    (factor_group * npools + pool) * nspins + spin)
                factor_real = factor_data[factor_offset]
                factor_imag = factor_data[factor_offset + 1]
                old_x = transverse_data[transverse_offset]
                transverse_data[transverse_offset] = (
                    old_x * factor_real
                    - transverse_data[transverse_offset + 1] * factor_imag)
                transverse_data[transverse_offset + 1] = (
                    old_x * factor_imag
                    + transverse_data[transverse_offset + 1] * factor_real)

            if source_mode == 2:
                concentration_source_start = 0.0
                concentration_source_end = 0.0
                source_start = 0.0
                source_end = 0.0
            elif source_mode == 1:
                if has_spatial_sources:
                    source_start = (
                        inflow_delivery[spin]
                        * spatial_source_values[2, step, spin])
                    source_end = (
                        inflow_delivery[spin]
                        * spatial_source_values[3, step, spin])
                else:
                    source_start = (
                        inflow_delivery[spin] * inflow_second_start[step])
                    source_end = (
                        inflow_delivery[spin] * inflow_second_end[step])
                concentration_source_start = 0.0
                concentration_source_end = 0.0
            else:
                if has_spatial_sources:
                    concentration_source_start = (
                        inflow_delivery[spin]
                        * spatial_source_values[2, step, spin])
                    concentration_source_end = (
                        inflow_delivery[spin]
                        * spatial_source_values[3, step, spin])
                    source_start = (
                        concentration_source_start
                        * spatial_source_values[6, step, spin])
                    source_end = (
                        concentration_source_end
                        * spatial_source_values[7, step, spin])
                else:
                    concentration_source_start = (
                        inflow_delivery[spin] * inflow_second_start[step])
                    concentration_source_end = (
                        inflow_delivery[spin] * inflow_second_end[step])
                    source_start = (
                        concentration_source_start
                        * polarization_second_start[step])
                    source_end = (
                        concentration_source_end
                        * polarization_second_end[step])
                if source_mode == 3:
                    concentration_source_start = 0.0
                    concentration_source_end = 0.0
            _advance_multi_pool_voxel(
                state_data, concentration_data, npools, nspins, spin, group,
                conversion_data, exp_precursor_data, exp_products_data,
                transfer_factor_data, f0_precursor_data, f1_precursor_data,
                j0_data, j1_data, concentration_exp_precursor_data,
                concentration_exp_products_data,
                concentration_transfer_factor_data,
                concentration_f0_precursor_data,
                concentration_f1_precursor_data,
                concentration_j0_data, concentration_j1_data,
                duration_by_group[group], source_start, source_end,
                concentration_source_start, concentration_source_end,
                equilibrium)
            if has_washout_sources:
                _add_washout_gains(
                    state_data, concentration_data, npools, nspins, spin,
                    washout_compartment_data, washout_row_data,
                    washout_table_data, washout_compartments, step, 1)
            if has_washout_loss:
                _scale_multi_pool_voxel(
                    state_data, concentration_data, transverse_data, npools,
                    nspins, spin, washout_factor)

            for action in range(action_offsets[step], action_offsets[step + 1]):
                action_kind = action_kinds[action]
                payload = action_payloads[action]
                if action_kind == 1 and ideal_crushers:
                    for pool in range(npools):
                        transverse_offset = 2 * (pool * nspins + spin)
                        transverse_data[transverse_offset] = 0.0
                        transverse_data[transverse_offset + 1] = 0.0
                elif action_kind == 2:
                    demod_real = demod_data[2 * payload]
                    demod_imag = demod_data[2 * payload + 1]
                    for pool in range(npools):
                        transverse_offset = 2 * (pool * nspins + spin)
                        value_x = transverse_data[transverse_offset]
                        value_y = transverse_data[transverse_offset + 1]
                        for coil in range(ncoils):
                            rx_real = rx_data[2 * (coil * nspins + spin)]
                            rx_imag = rx_data[2 * (coil * nspins + spin) + 1]
                            transverse_real = value_x * rx_real - value_y * rx_imag
                            transverse_imag = value_x * rx_imag + value_y * rx_real
                            weighted_real = (
                                transverse_real * signal_weights[spin] * signal_scale)
                            weighted_imag = (
                                transverse_imag * signal_weights[spin] * signal_scale)
                            received_real = (
                                weighted_real * demod_real
                                - weighted_imag * demod_imag)
                            received_imag = (
                                weighted_real * demod_imag
                                + weighted_imag * demod_real)
                            signal_offset = 2 * (
                                ((parent * npools + pool) * ncoils + coil)
                                * nadc + payload)
                            signal_data[signal_offset] = (
                                signal_data[signal_offset] + received_real)
                            signal_data[signal_offset + 1] = (
                                signal_data[signal_offset + 1] + received_imag)
                elif action_kind == 3:
                    for pool in range(npools):
                        transverse_offset = 2 * (pool * nspins + spin)
                        checkpoint_offset = 3 * (
                            (payload * npools + pool) * nspins + spin)
                        checkpoint_data[checkpoint_offset] = (
                            transverse_data[transverse_offset])
                        checkpoint_data[checkpoint_offset + 1] = (
                            transverse_data[transverse_offset + 1])
                        checkpoint_data[checkpoint_offset + 2] = state_data[
                            3 * (pool * nspins + spin) + 2]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def apply_rf_rotation_transverse_block(
        np.ndarray[DTYPE_t, ndim=3] state,
        np.ndarray[np.complex128_t, ndim=2] transverse_state,
        double axis_x,
        double axis_y,
        double cosine,
        double sine,
        double one_minus_cosine,
        int num_threads):
    """Rotate one RF interval across the complete two-pool voxel block.

    The optimized dynamic driver stores Mx/My in a compact complex array. This
    primitive reads and writes that representation directly, avoiding two full
    state synchronizations and all temporary ``numpy.cross`` arrays.
    """
    cdef Py_ssize_t nvoxels = state.shape[1]
    cdef Py_ssize_t item
    cdef Py_ssize_t state_offset
    cdef Py_ssize_t transverse_offset
    cdef double vx
    cdef double vy
    cdef double vz
    cdef double cross_x
    cdef double cross_y
    cdef double cross_z
    cdef double projection
    cdef double value_x
    cdef double value_y
    cdef double value_z
    cdef double *state_data
    cdef double *transverse_data

    if state.shape[0] != 2 or state.shape[2] != 3:
        raise ValueError("state must have shape (2, nvoxels, 3)")
    if transverse_state.shape[0] != 2 or transverse_state.shape[1] != nvoxels:
        raise ValueError("transverse_state must have shape (2, nvoxels)")
    if not state.flags.c_contiguous or not transverse_state.flags.c_contiguous:
        raise ValueError("state arrays must be C-contiguous")
    if num_threads < 1:
        raise ValueError("num_threads must be positive")

    state_data = <double *>state.data
    transverse_data = <double *>transverse_state.data
    for item in prange(
            2 * nvoxels, nogil=True, schedule="static", num_threads=num_threads):
        state_offset = 3 * item + 2
        transverse_offset = 2 * item
        vx = transverse_data[transverse_offset]
        vy = transverse_data[transverse_offset + 1]
        vz = state_data[state_offset]

        cross_x = axis_y * vz - 0.0 * vy
        cross_y = 0.0 * vx - axis_x * vz
        cross_z = axis_x * vy - axis_y * vx
        projection = vx * axis_x
        projection = projection + vy * axis_y
        projection = projection + vz * 0.0

        value_x = vx * cosine
        value_x = value_x + cross_x * sine
        value_x = value_x + (projection * axis_x) * one_minus_cosine

        value_y = vy * cosine
        value_y = value_y + cross_y * sine
        value_y = value_y + (projection * axis_y) * one_minus_cosine

        value_z = vz * cosine
        value_z = value_z + cross_z * sine
        value_z = value_z + (projection * 0.0) * one_minus_cosine

        transverse_data[transverse_offset] = value_x
        transverse_data[transverse_offset + 1] = value_y
        state_data[state_offset] = value_z


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def apply_spatial_rf_rotation_transverse_block(
        np.ndarray[DTYPE_t, ndim=3] state,
        np.ndarray[np.complex128_t, ndim=2] transverse_state,
        np.ndarray[DTYPE_t, ndim=1] axis_x,
        np.ndarray[DTYPE_t, ndim=1] axis_y,
        np.ndarray[DTYPE_t, ndim=1] cosine,
        np.ndarray[DTYPE_t, ndim=1] sine,
        int num_threads):
    """Rotate one RF interval using precomputed per-spin Float64 B1 data."""
    cdef Py_ssize_t nvoxels = state.shape[1]
    cdef Py_ssize_t item
    cdef Py_ssize_t voxel
    cdef Py_ssize_t state_offset
    cdef Py_ssize_t transverse_offset
    cdef double *state_data
    cdef double *transverse_data

    if state.shape[0] != 2 or state.shape[2] != 3:
        raise ValueError("state must have shape (2, nvoxels, 3)")
    if transverse_state.shape[0] != 2 or transverse_state.shape[1] != nvoxels:
        raise ValueError("transverse_state must have shape (2, nvoxels)")
    if (axis_x.shape[0] != nvoxels or axis_y.shape[0] != nvoxels or
            cosine.shape[0] != nvoxels or sine.shape[0] != nvoxels):
        raise ValueError("spatial RF coefficient lengths must match nvoxels")
    if not state.flags.c_contiguous or not transverse_state.flags.c_contiguous:
        raise ValueError("state arrays must be C-contiguous")
    if (not axis_x.flags.c_contiguous or not axis_y.flags.c_contiguous or
            not cosine.flags.c_contiguous or not sine.flags.c_contiguous):
        raise ValueError("spatial RF coefficients must be C-contiguous")
    if num_threads < 1:
        raise ValueError("num_threads must be positive")

    state_data = <double *>state.data
    transverse_data = <double *>transverse_state.data
    for item in prange(
            2 * nvoxels, nogil=True, schedule="static", num_threads=num_threads):
        voxel = item % nvoxels
        state_offset = 3 * item + 2
        transverse_offset = 2 * item
        _rotate_rf_voxel_plane(
            &transverse_data[transverse_offset],
            &transverse_data[transverse_offset + 1],
            &state_data[state_offset],
            axis_x[voxel], axis_y[voxel], cosine[voxel], sine[voxel])


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def apply_longitudinal_step_no_inflow(
        np.ndarray[DTYPE_t, ndim=3] state,
        np.ndarray[DTYPE_t, ndim=1] kpl,
        np.ndarray[DTYPE_t, ndim=1] exp_a,
        double exp_b,
        np.ndarray[DTYPE_t, ndim=1] difference,
        np.ndarray[BOOL_t, ndim=1] regular,
        double duration,
        int regular_mode):
    """Apply one longitudinal half-step without evaluating exponentials."""
    cdef Py_ssize_t nvoxels = kpl.shape[0]
    cdef Py_ssize_t voxel
    cdef Py_ssize_t state_offset
    cdef double pyruvate
    cdef double transfer
    cdef double decay_delta
    cdef double *state_data

    if state.shape[0] != 2 or state.shape[1] != nvoxels or state.shape[2] != 3:
        raise ValueError("state must have shape (2, nvoxels, 3)")
    if exp_a.shape[0] != nvoxels or difference.shape[0] != nvoxels:
        raise ValueError("longitudinal coefficient lengths must match nvoxels")
    if regular.shape[0] != nvoxels:
        raise ValueError("regular mask length must match nvoxels")
    if not state.flags.c_contiguous:
        raise ValueError("state must be C-contiguous")
    if not kpl.flags.c_contiguous or not exp_a.flags.c_contiguous:
        raise ValueError("longitudinal inputs must be C-contiguous")
    if not difference.flags.c_contiguous or not regular.flags.c_contiguous:
        raise ValueError("longitudinal inputs must be C-contiguous")
    if regular_mode < -1 or regular_mode > 1:
        raise ValueError("regular_mode must be -1, 0, or 1")

    state_data = <double *>state.data

    # Match the optimized NumPy scratch path operation-for-operation:
    # multiply kPL, subtract decay factors, multiply/divide the transfer,
    # multiply both pool states, and finally add the transfer to lactate.
    for voxel in range(nvoxels):
        state_offset = 3 * voxel + 2
        pyruvate = state_data[state_offset]
        transfer = kpl[voxel] * pyruvate
        if regular_mode == 1:
            decay_delta = exp_b - exp_a[voxel]
            transfer = transfer * decay_delta
            transfer = transfer / difference[voxel]
        elif regular_mode == -1:
            transfer = transfer * duration
            transfer = transfer * exp_b
        elif regular[voxel]:
            decay_delta = exp_b - exp_a[voxel]
            transfer = transfer * decay_delta
            transfer = transfer / difference[voxel]
        else:
            transfer = transfer * duration
            transfer = transfer * exp_b
        state_data[state_offset] = pyruvate * exp_a[voxel]
        state_offset = 3 * (nvoxels + voxel) + 2
        state_data[state_offset] = state_data[state_offset] * exp_b
        state_data[state_offset] = state_data[state_offset] + transfer


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def apply_longitudinal_step_with_concentration_inflow(
        np.ndarray[DTYPE_t, ndim=3] state,
        np.ndarray[DTYPE_t, ndim=3] concentration_state,
        np.ndarray[DTYPE_t, ndim=1] kpl,
        np.ndarray[DTYPE_t, ndim=1] exp_a,
        double exp_b,
        np.ndarray[DTYPE_t, ndim=1] difference,
        np.ndarray[BOOL_t, ndim=1] regular,
        np.ndarray[DTYPE_t, ndim=1] f0_a,
        np.ndarray[DTYPE_t, ndim=1] f1_a,
        np.ndarray[DTYPE_t, ndim=1] j0,
        np.ndarray[DTYPE_t, ndim=1] j1,
        np.ndarray[DTYPE_t, ndim=1] concentration_exp_a,
        double concentration_exp_b,
        np.ndarray[DTYPE_t, ndim=1] concentration_difference,
        np.ndarray[BOOL_t, ndim=1] concentration_regular,
        np.ndarray[DTYPE_t, ndim=1] concentration_f0_a,
        np.ndarray[DTYPE_t, ndim=1] concentration_f1_a,
        np.ndarray[DTYPE_t, ndim=1] concentration_j0,
        np.ndarray[DTYPE_t, ndim=1] concentration_j1,
        np.ndarray[DTYPE_t, ndim=1] source_start,
        np.ndarray[DTYPE_t, ndim=1] source_end,
        np.ndarray[DTYPE_t, ndim=1] concentration_source_start,
        np.ndarray[DTYPE_t, ndim=1] concentration_source_end,
        double duration,
        double equilibrium,
        int regular_mode,
        int concentration_regular_mode,
        int num_threads):
    """Advance magnetization and concentration in one voxel-parallel pass.

    All exponential and convolution coefficients are prepared by NumPy.  The
    arithmetic order mirrors ``_longitudinal_step`` and its scratch-backed
    zero-target solver while avoiding four complete strided state passes and
    the associated interval-sized temporaries.
    """
    cdef Py_ssize_t nvoxels = kpl.shape[0]
    cdef Py_ssize_t voxel
    cdef Py_ssize_t pyruvate_offset
    cdef Py_ssize_t lactate_offset
    cdef double pyruvate
    cdef double lactate
    cdef double concentration_pyruvate
    cdef double concentration_lactate
    cdef double transfer
    cdef double decay_delta
    cdef double source_value_start
    cdef double source_value_end
    cdef double source_slope
    cdef double source_delta
    cdef double source_addition
    cdef double *state_data
    cdef double *concentration_data

    if state.shape[0] != 2 or state.shape[1] != nvoxels or state.shape[2] != 3:
        raise ValueError("state must have shape (2, nvoxels, 3)")
    if (concentration_state.shape[0] != 2 or
            concentration_state.shape[1] != nvoxels or
            concentration_state.shape[2] != 3):
        raise ValueError(
            "concentration_state must have shape (2, nvoxels, 3)")
    if not state.flags.c_contiguous or not concentration_state.flags.c_contiguous:
        raise ValueError("state arrays must be C-contiguous")
    if (exp_a.shape[0] != nvoxels or difference.shape[0] != nvoxels or
            regular.shape[0] != nvoxels):
        raise ValueError("magnetization coefficient lengths must match nvoxels")
    if (concentration_exp_a.shape[0] != nvoxels or
            concentration_difference.shape[0] != nvoxels or
            concentration_regular.shape[0] != nvoxels):
        raise ValueError("concentration coefficient lengths must match nvoxels")
    if (f0_a.shape[0] != nvoxels or f1_a.shape[0] != nvoxels or
            j0.shape[0] != nvoxels or j1.shape[0] != nvoxels or
            concentration_f0_a.shape[0] != nvoxels or
            concentration_f1_a.shape[0] != nvoxels or
            concentration_j0.shape[0] != nvoxels or
            concentration_j1.shape[0] != nvoxels):
        raise ValueError("source coefficient lengths must match nvoxels")
    if (source_start.shape[0] != nvoxels or source_end.shape[0] != nvoxels or
            concentration_source_start.shape[0] != nvoxels or
            concentration_source_end.shape[0] != nvoxels):
        raise ValueError("source lengths must match nvoxels")
    if regular_mode < -1 or regular_mode > 1:
        raise ValueError("regular_mode must be -1, 0, or 1")
    if concentration_regular_mode < -1 or concentration_regular_mode > 1:
        raise ValueError("concentration_regular_mode must be -1, 0, or 1")
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    if num_threads < 1:
        raise ValueError("num_threads must be positive")

    state_data = <double *>state.data
    concentration_data = <double *>concentration_state.data
    for voxel in prange(
            nvoxels, nogil=True, schedule="static", num_threads=num_threads):
        pyruvate_offset = 3 * voxel + 2
        lactate_offset = 3 * (nvoxels + voxel) + 2
        pyruvate = state_data[pyruvate_offset]
        lactate = state_data[lactate_offset]
        concentration_pyruvate = concentration_data[pyruvate_offset]
        concentration_lactate = concentration_data[lactate_offset]

        if equilibrium != 0.0:
            pyruvate = pyruvate - equilibrium * concentration_pyruvate
            lactate = lactate - equilibrium * concentration_lactate

        transfer = kpl[voxel] * pyruvate
        if regular_mode == 1:
            decay_delta = exp_b - exp_a[voxel]
            transfer = transfer * decay_delta
            transfer = transfer / difference[voxel]
        elif regular_mode == -1:
            transfer = transfer * duration
            transfer = transfer * exp_b
        elif regular[voxel]:
            decay_delta = exp_b - exp_a[voxel]
            transfer = transfer * decay_delta
            transfer = transfer / difference[voxel]
        else:
            transfer = transfer * duration
            transfer = transfer * exp_b
        pyruvate = pyruvate * exp_a[voxel]
        lactate = lactate * exp_b
        lactate = lactate + transfer

        source_value_start = source_start[voxel]
        source_value_start = source_value_start - (
            equilibrium * concentration_source_start[voxel])
        source_value_end = source_end[voxel]
        source_value_end = source_value_end - (
            equilibrium * concentration_source_end[voxel])
        source_delta = source_value_end - source_value_start
        source_slope = source_delta / duration
        source_addition = source_value_start * f0_a[voxel]
        source_addition = source_addition + source_slope * f1_a[voxel]
        pyruvate = pyruvate + source_addition
        source_addition = source_value_start * j0[voxel]
        source_addition = source_addition + source_slope * j1[voxel]
        source_addition = kpl[voxel] * source_addition
        lactate = lactate + source_addition

        transfer = kpl[voxel] * concentration_pyruvate
        if concentration_regular_mode == 1:
            decay_delta = concentration_exp_b - concentration_exp_a[voxel]
            transfer = transfer * decay_delta
            transfer = transfer / concentration_difference[voxel]
        elif concentration_regular_mode == -1:
            transfer = transfer * duration
            transfer = transfer * concentration_exp_b
        elif concentration_regular[voxel]:
            decay_delta = concentration_exp_b - concentration_exp_a[voxel]
            transfer = transfer * decay_delta
            transfer = transfer / concentration_difference[voxel]
        else:
            transfer = transfer * duration
            transfer = transfer * concentration_exp_b
        concentration_pyruvate = (
            concentration_pyruvate * concentration_exp_a[voxel])
        concentration_lactate = (
            concentration_lactate * concentration_exp_b)
        concentration_lactate = concentration_lactate + transfer

        source_value_start = concentration_source_start[voxel]
        source_value_end = concentration_source_end[voxel]
        source_delta = source_value_end - source_value_start
        source_slope = source_delta / duration
        source_addition = source_value_start * concentration_f0_a[voxel]
        source_addition = (
            source_addition + source_slope * concentration_f1_a[voxel])
        concentration_pyruvate = concentration_pyruvate + source_addition
        source_addition = source_value_start * concentration_j0[voxel]
        source_addition = source_addition + source_slope * concentration_j1[voxel]
        source_addition = kpl[voxel] * source_addition
        concentration_lactate = concentration_lactate + source_addition

        if equilibrium != 0.0:
            pyruvate = pyruvate + equilibrium * concentration_pyruvate
            lactate = lactate + equilibrium * concentration_lactate

        state_data[pyruvate_offset] = pyruvate
        state_data[lactate_offset] = lactate
        concentration_data[pyruvate_offset] = concentration_pyruvate
        concentration_data[lactate_offset] = concentration_lactate


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def apply_dynamic_rf_block_with_concentration_inflow(
        np.ndarray[DTYPE_t, ndim=3] state,
        np.ndarray[np.complex128_t, ndim=2] transverse_state,
        np.ndarray[DTYPE_t, ndim=3] concentration_state,
        np.ndarray[DTYPE_t, ndim=1] kpl,
        np.ndarray[DTYPE_t, ndim=1] inflow_delivery,
        np.ndarray[np.complex128_t, ndim=3] transverse_factors,
        np.ndarray[DTYPE_t, ndim=2] exp_a_by_group,
        np.ndarray[DTYPE_t, ndim=1] exp_b_by_group,
        np.ndarray[DTYPE_t, ndim=2] difference_by_group,
        np.ndarray[BOOL_t, ndim=2] regular_by_group,
        np.ndarray[DTYPE_t, ndim=2] f0_a_by_group,
        np.ndarray[DTYPE_t, ndim=2] f1_a_by_group,
        np.ndarray[DTYPE_t, ndim=2] j0_by_group,
        np.ndarray[DTYPE_t, ndim=2] j1_by_group,
        np.ndarray[DTYPE_t, ndim=2] concentration_exp_a_by_group,
        np.ndarray[DTYPE_t, ndim=1] concentration_exp_b_by_group,
        np.ndarray[DTYPE_t, ndim=2] concentration_difference_by_group,
        np.ndarray[BOOL_t, ndim=2] concentration_regular_by_group,
        np.ndarray[DTYPE_t, ndim=2] concentration_f0_a_by_group,
        np.ndarray[DTYPE_t, ndim=2] concentration_f1_a_by_group,
        np.ndarray[DTYPE_t, ndim=2] concentration_j0_by_group,
        np.ndarray[DTYPE_t, ndim=2] concentration_j1_by_group,
        np.ndarray[DTYPE_t, ndim=1] duration_by_group,
        np.ndarray[np.int32_t, ndim=1] regular_mode_by_group,
        np.ndarray[np.int32_t, ndim=1] concentration_regular_mode_by_group,
        np.ndarray[np.int32_t, ndim=1] longitudinal_groups,
        np.ndarray[np.int32_t, ndim=1] first_factor_groups,
        np.ndarray[np.int32_t, ndim=1] second_factor_groups,
        np.ndarray[np.int32_t, ndim=1] rf_groups,
        np.ndarray[DTYPE_t, ndim=2] rf_axis_x,
        np.ndarray[DTYPE_t, ndim=2] rf_axis_y,
        np.ndarray[DTYPE_t, ndim=2] rf_cosine,
        np.ndarray[DTYPE_t, ndim=2] rf_sine,
        np.ndarray[DTYPE_t, ndim=2] rf_one_minus_cosine,
        np.ndarray[DTYPE_t, ndim=1] inflow_first_start,
        np.ndarray[DTYPE_t, ndim=1] inflow_first_end,
        np.ndarray[DTYPE_t, ndim=1] inflow_second_start,
        np.ndarray[DTYPE_t, ndim=1] inflow_second_end,
        np.ndarray[DTYPE_t, ndim=1] polarization_first_start,
        np.ndarray[DTYPE_t, ndim=1] polarization_first_end,
        np.ndarray[DTYPE_t, ndim=1] polarization_second_start,
        np.ndarray[DTYPE_t, ndim=1] polarization_second_end,
        double equilibrium,
        int num_threads,
        np.ndarray[np.int32_t, ndim=1] action_offsets=None,
        np.ndarray[np.int8_t, ndim=1] action_kinds=None,
        np.ndarray[np.int64_t, ndim=1] action_payloads=None,
        np.ndarray[np.complex128_t, ndim=1] adc_demodulation=None,
        np.ndarray[np.complex128_t, ndim=2] rx_sensitivities=None,
        np.ndarray[DTYPE_t, ndim=1] signal_weights=None,
        np.ndarray[np.complex128_t, ndim=4] parent_species_signal=None,
        np.ndarray[DTYPE_t, ndim=4] checkpoint_states=None,
        int spins_per_parent=1,
        double signal_scale=1.0,
        bint ideal_crushers=False,
        np.ndarray[DTYPE_t, ndim=2] kpl_by_group=None,
        np.ndarray[DTYPE_t, ndim=3] spatial_source_values=None,
        int source_mode=0):
    """Execute two-pool source intervals in one persistent parallel region.

    Without action arrays this retains the established RF-waveform-block API.
    With action arrays it also performs deterministic parent-resolved ADC,
    crushers, and checkpoints for a bounded full-plan block. ``source_mode``
    selects coupled concentration/polarization sources (0), direct
    magnetization sources (1), no sources (2), or polarized magnetization
    sources without concentration delivery (3).
    """
    cdef Py_ssize_t nvoxels = kpl.shape[0]
    cdef Py_ssize_t nsteps = longitudinal_groups.shape[0]
    cdef Py_ssize_t nparents
    cdef Py_ssize_t ncoils
    cdef Py_ssize_t nadc
    cdef Py_ssize_t parent
    cdef Py_ssize_t local_spin
    cdef Py_ssize_t voxel
    cdef Py_ssize_t step
    cdef Py_ssize_t pool
    cdef Py_ssize_t coil
    cdef Py_ssize_t action
    cdef Py_ssize_t group
    cdef Py_ssize_t factor_group
    cdef Py_ssize_t rf_group
    cdef Py_ssize_t pyruvate_offset
    cdef Py_ssize_t lactate_offset
    cdef Py_ssize_t transverse_offset
    cdef Py_ssize_t factor_offset
    cdef Py_ssize_t signal_offset
    cdef Py_ssize_t checkpoint_offset
    cdef long payload
    cdef signed char action_kind
    cdef double pyruvate
    cdef double lactate
    cdef double concentration_pyruvate
    cdef double concentration_lactate
    cdef double pyruvate_x
    cdef double pyruvate_y
    cdef double lactate_x
    cdef double lactate_y
    cdef double old_x
    cdef double factor_real
    cdef double factor_imag
    cdef double concentration_source_start
    cdef double concentration_source_end
    cdef double source_start
    cdef double source_end
    cdef double interval_kpl
    cdef double transverse_real
    cdef double transverse_imag
    cdef double rx_real
    cdef double rx_imag
    cdef double demod_real
    cdef double demod_imag
    cdef double received_real
    cdef double received_imag
    cdef double weighted_real
    cdef double weighted_imag
    cdef double *state_data
    cdef double *concentration_data
    cdef double *transverse_data
    cdef double *factor_data
    cdef double *rx_data = NULL
    cdef double *demod_data = NULL
    cdef double *signal_data = NULL
    cdef double *checkpoint_data = NULL
    cdef bint spatial_rf
    cdef bint has_actions = action_offsets is not None
    cdef bint has_kpl_groups = kpl_by_group is not None
    cdef bint has_spatial_sources = spatial_source_values is not None

    if state.shape[0] != 2 or state.shape[1] != nvoxels or state.shape[2] != 3:
        raise ValueError("state must have shape (2, nvoxels, 3)")
    if transverse_state.shape[0] != 2 or transverse_state.shape[1] != nvoxels:
        raise ValueError("transverse_state must have shape (2, nvoxels)")
    if (concentration_state.shape[0] != 2 or
            concentration_state.shape[1] != nvoxels or
            concentration_state.shape[2] != 3):
        raise ValueError(
            "concentration_state must have shape (2, nvoxels, 3)")
    if inflow_delivery.shape[0] != nvoxels:
        raise ValueError("inflow_delivery length must match nvoxels")
    if (first_factor_groups.shape[0] != nsteps or
            second_factor_groups.shape[0] != nsteps or
            rf_groups.shape[0] != nsteps):
        raise ValueError("RF block arrays must have equal step counts")
    if (rf_axis_y.shape[0] != rf_axis_x.shape[0] or
            rf_cosine.shape[0] != rf_axis_x.shape[0] or
            rf_sine.shape[0] != rf_axis_x.shape[0] or
            rf_one_minus_cosine.shape[0] != rf_axis_x.shape[0]):
        raise ValueError("RF coefficient tables must have equal group counts")
    if (rf_axis_x.shape[1] != 1 and rf_axis_x.shape[1] != nvoxels):
        raise ValueError("RF coefficient width must be one or nvoxels")
    if (rf_axis_y.shape[1] != rf_axis_x.shape[1] or
            rf_cosine.shape[1] != rf_axis_x.shape[1] or
            rf_sine.shape[1] != rf_axis_x.shape[1] or
            rf_one_minus_cosine.shape[1] != rf_axis_x.shape[1]):
        raise ValueError("RF coefficient tables must have equal widths")
    if (inflow_first_start.shape[0] != nsteps or
            inflow_first_end.shape[0] != nsteps or
            inflow_second_start.shape[0] != nsteps or
            inflow_second_end.shape[0] != nsteps or
            polarization_first_start.shape[0] != nsteps or
            polarization_first_end.shape[0] != nsteps or
            polarization_second_start.shape[0] != nsteps or
            polarization_second_end.shape[0] != nsteps):
        raise ValueError("source block arrays must have equal step counts")
    if has_spatial_sources and (
            spatial_source_values.shape[0] != 8 or
            spatial_source_values.shape[1] != nsteps or
            spatial_source_values.shape[2] != nvoxels):
        raise ValueError(
            "spatial source values must have shape (8, nsteps, nvoxels)")
    if num_threads < 1:
        raise ValueError("num_threads must be positive")
    if source_mode < 0 or source_mode > 3:
        raise ValueError("source_mode must be between zero and three")
    if not state.flags.c_contiguous or not concentration_state.flags.c_contiguous:
        raise ValueError("state arrays must be C-contiguous")
    if not transverse_state.flags.c_contiguous or not transverse_factors.flags.c_contiguous:
        raise ValueError("transverse arrays must be C-contiguous")
    if has_actions:
        if (action_kinds is None or action_payloads is None or
                adc_demodulation is None or rx_sensitivities is None or
                signal_weights is None or parent_species_signal is None or
                checkpoint_states is None):
            raise ValueError("complete plan actions require observation arrays")
        if action_offsets.shape[0] != nsteps + 1:
            raise ValueError("action offsets must bracket every plan step")
        if spins_per_parent < 1 or nvoxels % spins_per_parent != 0:
            raise ValueError("parent/spin dimensions are inconsistent")
        nparents = nvoxels // spins_per_parent
        ncoils = rx_sensitivities.shape[0]
        nadc = parent_species_signal.shape[3]
        if parent_species_signal.shape[0] != nparents:
            raise ValueError("parent signal shape is inconsistent")
        if (parent_species_signal.shape[1] != 2 or
                parent_species_signal.shape[2] != ncoils):
            raise ValueError("parent signal shape is inconsistent")
        if (rx_sensitivities.shape[1] != nvoxels or
                signal_weights.shape[0] != nvoxels):
            raise ValueError("receive arrays must contain every spin")
        if (kpl_by_group is not None and
                kpl_by_group.shape[1] != nvoxels):
            raise ValueError("kpl_by_group must contain every spin")
        rx_data = <double *>rx_sensitivities.data
        demod_data = <double *>adc_demodulation.data
        signal_data = <double *>parent_species_signal.data
        checkpoint_data = <double *>checkpoint_states.data
    else:
        nparents = nvoxels
        spins_per_parent = 1
        ncoils = 0
        nadc = 0

    state_data = <double *>state.data
    concentration_data = <double *>concentration_state.data
    transverse_data = <double *>transverse_state.data
    factor_data = <double *>transverse_factors.data
    spatial_rf = rf_axis_x.shape[1] != 1

    for parent in prange(
            nparents, nogil=True, schedule="static", num_threads=num_threads):
      for local_spin in range(spins_per_parent):
        voxel = parent * spins_per_parent + local_spin
        pyruvate_offset = 3 * voxel + 2
        lactate_offset = 3 * (nvoxels + voxel) + 2
        transverse_offset = 2 * voxel
        pyruvate = state_data[pyruvate_offset]
        lactate = state_data[lactate_offset]
        concentration_pyruvate = concentration_data[pyruvate_offset]
        concentration_lactate = concentration_data[lactate_offset]
        pyruvate_x = transverse_data[transverse_offset]
        pyruvate_y = transverse_data[transverse_offset + 1]
        transverse_offset = 2 * (nvoxels + voxel)
        lactate_x = transverse_data[transverse_offset]
        lactate_y = transverse_data[transverse_offset + 1]

        for step in range(nsteps):
            group = longitudinal_groups[step]
            interval_kpl = (
                kpl[voxel]
                if not has_kpl_groups
                else kpl_by_group[group, voxel]
            )
            factor_group = first_factor_groups[step]
            factor_offset = 2 * ((factor_group * 2) * nvoxels + voxel)
            factor_real = factor_data[factor_offset]
            factor_imag = factor_data[factor_offset + 1]
            old_x = pyruvate_x
            pyruvate_x = old_x * factor_real - pyruvate_y * factor_imag
            pyruvate_y = old_x * factor_imag + pyruvate_y * factor_real
            factor_offset = 2 * ((factor_group * 2 + 1) * nvoxels + voxel)
            factor_real = factor_data[factor_offset]
            factor_imag = factor_data[factor_offset + 1]
            old_x = lactate_x
            lactate_x = old_x * factor_real - lactate_y * factor_imag
            lactate_y = old_x * factor_imag + lactate_y * factor_real

            if source_mode == 2:
                concentration_source_start = 0.0
                concentration_source_end = 0.0
                source_start = 0.0
                source_end = 0.0
            elif source_mode == 1 and has_spatial_sources:
                source_start = (
                    inflow_delivery[voxel]
                    * spatial_source_values[0, step, voxel])
                source_end = (
                    inflow_delivery[voxel]
                    * spatial_source_values[1, step, voxel])
                concentration_source_start = 0.0
                concentration_source_end = 0.0
            elif source_mode == 1:
                source_start = (
                    inflow_delivery[voxel] * inflow_first_start[step])
                source_end = (
                    inflow_delivery[voxel] * inflow_first_end[step])
                concentration_source_start = 0.0
                concentration_source_end = 0.0
            elif has_spatial_sources:
                concentration_source_start = (
                    inflow_delivery[voxel]
                    * spatial_source_values[0, step, voxel])
                concentration_source_end = (
                    inflow_delivery[voxel]
                    * spatial_source_values[1, step, voxel])
                source_start = (
                    concentration_source_start
                    * spatial_source_values[4, step, voxel])
                source_end = (
                    concentration_source_end
                    * spatial_source_values[5, step, voxel])
                if source_mode == 3:
                    concentration_source_start = 0.0
                    concentration_source_end = 0.0
            else:
                concentration_source_start = (
                    inflow_delivery[voxel] * inflow_first_start[step])
                concentration_source_end = (
                    inflow_delivery[voxel] * inflow_first_end[step])
                source_start = (
                    concentration_source_start
                    * polarization_first_start[step])
                source_end = (
                    concentration_source_end * polarization_first_end[step])
                if source_mode == 3:
                    concentration_source_start = 0.0
                    concentration_source_end = 0.0
            _advance_coupled_longitudinal_voxel(
                &pyruvate, &lactate,
                &concentration_pyruvate, &concentration_lactate,
                interval_kpl,
                exp_a_by_group[group, voxel], exp_b_by_group[group],
                difference_by_group[group, voxel],
                regular_by_group[group, voxel],
                f0_a_by_group[group, voxel], f1_a_by_group[group, voxel],
                j0_by_group[group, voxel], j1_by_group[group, voxel],
                concentration_exp_a_by_group[group, voxel],
                concentration_exp_b_by_group[group],
                concentration_difference_by_group[group, voxel],
                concentration_regular_by_group[group, voxel],
                concentration_f0_a_by_group[group, voxel],
                concentration_f1_a_by_group[group, voxel],
                concentration_j0_by_group[group, voxel],
                concentration_j1_by_group[group, voxel],
                source_start, source_end,
                concentration_source_start, concentration_source_end,
                duration_by_group[group], equilibrium,
                regular_mode_by_group[group],
                concentration_regular_mode_by_group[group])

            rf_group = rf_groups[step]
            if rf_group >= 0 and not spatial_rf:
                _rotate_rf_voxel(
                    &pyruvate_x, &pyruvate_y, &pyruvate,
                    rf_axis_x[rf_group, 0], rf_axis_y[rf_group, 0],
                    rf_cosine[rf_group, 0], rf_sine[rf_group, 0],
                    rf_one_minus_cosine[rf_group, 0])
                _rotate_rf_voxel(
                    &lactate_x, &lactate_y, &lactate,
                    rf_axis_x[rf_group, 0], rf_axis_y[rf_group, 0],
                    rf_cosine[rf_group, 0], rf_sine[rf_group, 0],
                    rf_one_minus_cosine[rf_group, 0])
            elif rf_group >= 0:
                _rotate_rf_voxel_plane(
                    &pyruvate_x, &pyruvate_y, &pyruvate,
                    rf_axis_x[rf_group, voxel], rf_axis_y[rf_group, voxel],
                    rf_cosine[rf_group, voxel], rf_sine[rf_group, voxel])
                _rotate_rf_voxel_plane(
                    &lactate_x, &lactate_y, &lactate,
                    rf_axis_x[rf_group, voxel], rf_axis_y[rf_group, voxel],
                    rf_cosine[rf_group, voxel], rf_sine[rf_group, voxel])

            factor_group = second_factor_groups[step]
            factor_offset = 2 * ((factor_group * 2) * nvoxels + voxel)
            factor_real = factor_data[factor_offset]
            factor_imag = factor_data[factor_offset + 1]
            old_x = pyruvate_x
            pyruvate_x = old_x * factor_real - pyruvate_y * factor_imag
            pyruvate_y = old_x * factor_imag + pyruvate_y * factor_real
            factor_offset = 2 * ((factor_group * 2 + 1) * nvoxels + voxel)
            factor_real = factor_data[factor_offset]
            factor_imag = factor_data[factor_offset + 1]
            old_x = lactate_x
            lactate_x = old_x * factor_real - lactate_y * factor_imag
            lactate_y = old_x * factor_imag + lactate_y * factor_real

            if source_mode == 2:
                concentration_source_start = 0.0
                concentration_source_end = 0.0
                source_start = 0.0
                source_end = 0.0
            elif source_mode == 1 and has_spatial_sources:
                source_start = (
                    inflow_delivery[voxel]
                    * spatial_source_values[2, step, voxel])
                source_end = (
                    inflow_delivery[voxel]
                    * spatial_source_values[3, step, voxel])
                concentration_source_start = 0.0
                concentration_source_end = 0.0
            elif source_mode == 1:
                source_start = (
                    inflow_delivery[voxel] * inflow_second_start[step])
                source_end = (
                    inflow_delivery[voxel] * inflow_second_end[step])
                concentration_source_start = 0.0
                concentration_source_end = 0.0
            elif has_spatial_sources:
                concentration_source_start = (
                    inflow_delivery[voxel]
                    * spatial_source_values[2, step, voxel])
                concentration_source_end = (
                    inflow_delivery[voxel]
                    * spatial_source_values[3, step, voxel])
                source_start = (
                    concentration_source_start
                    * spatial_source_values[6, step, voxel])
                source_end = (
                    concentration_source_end
                    * spatial_source_values[7, step, voxel])
                if source_mode == 3:
                    concentration_source_start = 0.0
                    concentration_source_end = 0.0
            else:
                concentration_source_start = (
                    inflow_delivery[voxel] * inflow_second_start[step])
                concentration_source_end = (
                    inflow_delivery[voxel] * inflow_second_end[step])
                source_start = (
                    concentration_source_start
                    * polarization_second_start[step])
                source_end = (
                    concentration_source_end * polarization_second_end[step])
                if source_mode == 3:
                    concentration_source_start = 0.0
                    concentration_source_end = 0.0
            _advance_coupled_longitudinal_voxel(
                &pyruvate, &lactate,
                &concentration_pyruvate, &concentration_lactate,
                interval_kpl,
                exp_a_by_group[group, voxel], exp_b_by_group[group],
                difference_by_group[group, voxel],
                regular_by_group[group, voxel],
                f0_a_by_group[group, voxel], f1_a_by_group[group, voxel],
                j0_by_group[group, voxel], j1_by_group[group, voxel],
                concentration_exp_a_by_group[group, voxel],
                concentration_exp_b_by_group[group],
                concentration_difference_by_group[group, voxel],
                concentration_regular_by_group[group, voxel],
                concentration_f0_a_by_group[group, voxel],
                concentration_f1_a_by_group[group, voxel],
                concentration_j0_by_group[group, voxel],
                concentration_j1_by_group[group, voxel],
                source_start, source_end,
                concentration_source_start, concentration_source_end,
                duration_by_group[group], equilibrium,
                regular_mode_by_group[group],
                concentration_regular_mode_by_group[group])

            if has_actions:
                for action in range(
                        action_offsets[step], action_offsets[step + 1]):
                    action_kind = action_kinds[action]
                    payload = action_payloads[action]
                    if action_kind == 1 and ideal_crushers:
                        pyruvate_x = 0.0
                        pyruvate_y = 0.0
                        lactate_x = 0.0
                        lactate_y = 0.0
                    elif action_kind == 2:
                        demod_real = demod_data[2 * payload]
                        demod_imag = demod_data[2 * payload + 1]
                        for coil in range(ncoils):
                            rx_real = rx_data[2 * (coil * nvoxels + voxel)]
                            rx_imag = rx_data[
                                2 * (coil * nvoxels + voxel) + 1]
                            transverse_real = (
                                pyruvate_x * rx_real - pyruvate_y * rx_imag)
                            transverse_imag = (
                                pyruvate_x * rx_imag + pyruvate_y * rx_real)
                            weighted_real = (
                                transverse_real * signal_weights[voxel]
                                * signal_scale)
                            weighted_imag = (
                                transverse_imag * signal_weights[voxel]
                                * signal_scale)
                            received_real = (
                                weighted_real * demod_real
                                - weighted_imag * demod_imag)
                            received_imag = (
                                weighted_real * demod_imag
                                + weighted_imag * demod_real)
                            signal_offset = 2 * (
                                ((parent * 2) * ncoils + coil) * nadc
                                + payload)
                            signal_data[signal_offset] = (
                                signal_data[signal_offset] + received_real)
                            signal_data[signal_offset + 1] = (
                                signal_data[signal_offset + 1] + received_imag)

                            transverse_real = (
                                lactate_x * rx_real - lactate_y * rx_imag)
                            transverse_imag = (
                                lactate_x * rx_imag + lactate_y * rx_real)
                            weighted_real = (
                                transverse_real * signal_weights[voxel]
                                * signal_scale)
                            weighted_imag = (
                                transverse_imag * signal_weights[voxel]
                                * signal_scale)
                            received_real = (
                                weighted_real * demod_real
                                - weighted_imag * demod_imag)
                            received_imag = (
                                weighted_real * demod_imag
                                + weighted_imag * demod_real)
                            signal_offset = 2 * (
                                ((parent * 2 + 1) * ncoils + coil) * nadc
                                + payload)
                            signal_data[signal_offset] = (
                                signal_data[signal_offset] + received_real)
                            signal_data[signal_offset + 1] = (
                                signal_data[signal_offset + 1] + received_imag)
                    elif action_kind == 3:
                        checkpoint_offset = 3 * (
                            (payload * 2) * nvoxels + voxel)
                        checkpoint_data[checkpoint_offset] = pyruvate_x
                        checkpoint_data[checkpoint_offset + 1] = pyruvate_y
                        checkpoint_data[checkpoint_offset + 2] = pyruvate
                        checkpoint_offset = 3 * (
                            (payload * 2 + 1) * nvoxels + voxel)
                        checkpoint_data[checkpoint_offset] = lactate_x
                        checkpoint_data[checkpoint_offset + 1] = lactate_y
                        checkpoint_data[checkpoint_offset + 2] = lactate

        state_data[pyruvate_offset] = pyruvate
        state_data[lactate_offset] = lactate
        concentration_data[pyruvate_offset] = concentration_pyruvate
        concentration_data[lactate_offset] = concentration_lactate
        transverse_offset = 2 * voxel
        transverse_data[transverse_offset] = pyruvate_x
        transverse_data[transverse_offset + 1] = pyruvate_y
        transverse_offset = 2 * (nvoxels + voxel)
        transverse_data[transverse_offset] = lactate_x
        transverse_data[transverse_offset + 1] = lactate_y


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def apply_longitudinal_block_no_inflow(
        np.ndarray[DTYPE_t, ndim=3] state,
        np.ndarray[DTYPE_t, ndim=1] kpl,
        np.ndarray[DTYPE_t, ndim=2] exp_a_by_group,
        np.ndarray[DTYPE_t, ndim=1] exp_b_by_group,
        np.ndarray[DTYPE_t, ndim=1] difference,
        np.ndarray[BOOL_t, ndim=1] regular,
        np.ndarray[np.int32_t, ndim=1] step_groups,
        np.ndarray[DTYPE_t, ndim=1] duration_by_group,
        int regular_mode,
        int num_threads):
    """Apply many longitudinal half-steps in one voxel-parallel region."""
    cdef Py_ssize_t nvoxels = kpl.shape[0]
    cdef Py_ssize_t ngroups = exp_a_by_group.shape[0]
    cdef Py_ssize_t nsteps = step_groups.shape[0]
    cdef Py_ssize_t voxel
    cdef Py_ssize_t step
    cdef Py_ssize_t group
    cdef Py_ssize_t state_offset
    cdef double pyruvate
    cdef double lactate
    cdef double transfer
    cdef double decay_delta
    cdef double exp_a
    cdef double exp_b
    cdef double *state_data
    cdef double *exp_a_data

    if state.shape[0] != 2 or state.shape[1] != nvoxels or state.shape[2] != 3:
        raise ValueError("state must have shape (2, nvoxels, 3)")
    if exp_a_by_group.shape[1] != nvoxels:
        raise ValueError("exp_a_by_group must have shape (ngroups, nvoxels)")
    if exp_b_by_group.shape[0] != ngroups or duration_by_group.shape[0] != ngroups:
        raise ValueError("group coefficient arrays must have equal lengths")
    if difference.shape[0] != nvoxels or regular.shape[0] != nvoxels:
        raise ValueError("voxel coefficient lengths must match nvoxels")
    if not state.flags.c_contiguous or not kpl.flags.c_contiguous:
        raise ValueError("state and kpl must be C-contiguous")
    if not exp_a_by_group.flags.c_contiguous:
        raise ValueError("exp_a_by_group must be C-contiguous")
    if not exp_b_by_group.flags.c_contiguous:
        raise ValueError("exp_b_by_group must be C-contiguous")
    if not difference.flags.c_contiguous or not regular.flags.c_contiguous:
        raise ValueError("voxel coefficients must be C-contiguous")
    if not step_groups.flags.c_contiguous or not duration_by_group.flags.c_contiguous:
        raise ValueError("block coefficient indices must be C-contiguous")
    if regular_mode < -1 or regular_mode > 1:
        raise ValueError("regular_mode must be -1, 0, or 1")
    if num_threads < 1:
        raise ValueError("num_threads must be positive")
    if nsteps and (
        np.min(step_groups) < 0 or np.max(step_groups) >= ngroups
    ):
        raise ValueError("step group index is out of range")

    state_data = <double *>state.data
    exp_a_data = <double *>exp_a_by_group.data

    for voxel in prange(
            nvoxels, nogil=True, schedule="static", num_threads=num_threads):
        state_offset = 3 * voxel + 2
        pyruvate = state_data[state_offset]
        state_offset = 3 * (nvoxels + voxel) + 2
        lactate = state_data[state_offset]
        for step in range(nsteps):
            group = step_groups[step]
            exp_a = exp_a_data[group * nvoxels + voxel]
            exp_b = exp_b_by_group[group]
            transfer = kpl[voxel] * pyruvate
            if regular_mode == 1:
                decay_delta = exp_b - exp_a
                transfer = transfer * decay_delta
                transfer = transfer / difference[voxel]
            elif regular_mode == -1:
                transfer = transfer * duration_by_group[group]
                transfer = transfer * exp_b
            elif regular[voxel]:
                decay_delta = exp_b - exp_a
                transfer = transfer * decay_delta
                transfer = transfer / difference[voxel]
            else:
                transfer = transfer * duration_by_group[group]
                transfer = transfer * exp_b
            pyruvate = pyruvate * exp_a
            lactate = lactate * exp_b
            lactate = lactate + transfer
        state_offset = 3 * voxel + 2
        state_data[state_offset] = pyruvate
        state_offset = 3 * (nvoxels + voxel) + 2
        state_data[state_offset] = lactate
