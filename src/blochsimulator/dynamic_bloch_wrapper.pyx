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


ctypedef np.float64_t DTYPE_t
ctypedef np.npy_bool BOOL_t


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
        np.ndarray[DTYPE_t, ndim=1] rf_axis_x,
        np.ndarray[DTYPE_t, ndim=1] rf_axis_y,
        np.ndarray[DTYPE_t, ndim=1] rf_cosine,
        np.ndarray[DTYPE_t, ndim=1] rf_sine,
        np.ndarray[DTYPE_t, ndim=1] rf_one_minus_cosine,
        np.ndarray[DTYPE_t, ndim=1] inflow_first_start,
        np.ndarray[DTYPE_t, ndim=1] inflow_first_end,
        np.ndarray[DTYPE_t, ndim=1] inflow_second_start,
        np.ndarray[DTYPE_t, ndim=1] inflow_second_end,
        np.ndarray[DTYPE_t, ndim=1] polarization_first_start,
        np.ndarray[DTYPE_t, ndim=1] polarization_first_end,
        np.ndarray[DTYPE_t, ndim=1] polarization_second_start,
        np.ndarray[DTYPE_t, ndim=1] polarization_second_end,
        double equilibrium,
        int num_threads):
    """Apply one complete RF raster block inside one persistent parallel region."""
    cdef Py_ssize_t nvoxels = kpl.shape[0]
    cdef Py_ssize_t nsteps = longitudinal_groups.shape[0]
    cdef Py_ssize_t voxel
    cdef Py_ssize_t step
    cdef Py_ssize_t group
    cdef Py_ssize_t factor_group
    cdef Py_ssize_t pyruvate_offset
    cdef Py_ssize_t lactate_offset
    cdef Py_ssize_t transverse_offset
    cdef Py_ssize_t factor_offset
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
    cdef double *state_data
    cdef double *concentration_data
    cdef double *transverse_data
    cdef double *factor_data

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
            rf_axis_x.shape[0] != nsteps or rf_axis_y.shape[0] != nsteps or
            rf_cosine.shape[0] != nsteps or rf_sine.shape[0] != nsteps or
            rf_one_minus_cosine.shape[0] != nsteps):
        raise ValueError("RF block arrays must have equal step counts")
    if (inflow_first_start.shape[0] != nsteps or
            inflow_first_end.shape[0] != nsteps or
            inflow_second_start.shape[0] != nsteps or
            inflow_second_end.shape[0] != nsteps or
            polarization_first_start.shape[0] != nsteps or
            polarization_first_end.shape[0] != nsteps or
            polarization_second_start.shape[0] != nsteps or
            polarization_second_end.shape[0] != nsteps):
        raise ValueError("source block arrays must have equal step counts")
    if num_threads < 1:
        raise ValueError("num_threads must be positive")
    if not state.flags.c_contiguous or not concentration_state.flags.c_contiguous:
        raise ValueError("state arrays must be C-contiguous")
    if not transverse_state.flags.c_contiguous or not transverse_factors.flags.c_contiguous:
        raise ValueError("transverse arrays must be C-contiguous")

    state_data = <double *>state.data
    concentration_data = <double *>concentration_state.data
    transverse_data = <double *>transverse_state.data
    factor_data = <double *>transverse_factors.data

    for voxel in prange(
            nvoxels, nogil=True, schedule="static", num_threads=num_threads):
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

            concentration_source_start = (
                inflow_delivery[voxel] * inflow_first_start[step])
            concentration_source_end = (
                inflow_delivery[voxel] * inflow_first_end[step])
            source_start = (
                concentration_source_start * polarization_first_start[step])
            source_end = concentration_source_end * polarization_first_end[step]
            _advance_coupled_longitudinal_voxel(
                &pyruvate, &lactate,
                &concentration_pyruvate, &concentration_lactate,
                kpl[voxel],
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

            _rotate_rf_voxel(
                &pyruvate_x, &pyruvate_y, &pyruvate,
                rf_axis_x[step], rf_axis_y[step], rf_cosine[step],
                rf_sine[step], rf_one_minus_cosine[step])
            _rotate_rf_voxel(
                &lactate_x, &lactate_y, &lactate,
                rf_axis_x[step], rf_axis_y[step], rf_cosine[step],
                rf_sine[step], rf_one_minus_cosine[step])

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

            concentration_source_start = (
                inflow_delivery[voxel] * inflow_second_start[step])
            concentration_source_end = (
                inflow_delivery[voxel] * inflow_second_end[step])
            source_start = (
                concentration_source_start * polarization_second_start[step])
            source_end = concentration_source_end * polarization_second_end[step]
            _advance_coupled_longitudinal_voxel(
                &pyruvate, &lactate,
                &concentration_pyruvate, &concentration_lactate,
                kpl[voxel],
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
