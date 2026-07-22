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
