"""Private Cython bridge for the Metal numerical-feasibility probe."""

from libc.stdint cimport uint32_t, uint64_t
from libc.stddef cimport size_t
import numpy as np
cimport numpy as np


cdef extern from "metal/dynamic_metal_probe.h":
    int bloch_metal_probe_capability(
        char *device_name,
        size_t device_name_size,
        char *reason,
        size_t reason_size,
        uint64_t *recommended_working_set_bytes,
        int *apple_gpu_family,
    ) nogil

    int bloch_metal_probe_run(
        const char *source,
        const float *interval_plan,
        uint32_t interval_count,
        const uint32_t *adc_state_indices,
        const float *adc_demodulation,
        uint32_t adc_count,
        const uint32_t *crusher_state_indices,
        uint32_t crusher_count,
        const float *initial_pool_state,
        const float *spatial_parameters,
        const float *kinetic_parameters,
        uint32_t spin_count,
        const float *physical_constants,
        uint32_t precision_mode,
        float *final_pool_state,
        float *per_spin_species_signal,
        double *pipeline_compile_seconds,
        double *simulation_seconds,
        char *error_message,
        size_t error_message_size,
    ) nogil


def capability():
    cdef char device_name[256]
    cdef char reason[512]
    cdef uint64_t recommended = 0
    cdef int family = 0
    cdef int available
    with nogil:
        available = bloch_metal_probe_capability(
            device_name, 256, reason, 512, &recommended, &family
        )
    return {
        "available": bool(available),
        "device_name": (<bytes>device_name).decode("utf-8", "replace") if available else None,
        "reason": None if available else (<bytes>reason).decode("utf-8", "replace"),
        "recommended_max_working_set_bytes": int(recommended),
        "apple_gpu_family": int(family),
    }


def run_probe(
    bytes source,
    np.ndarray[np.float32_t, ndim=2, mode="c"] interval_plan,
    np.ndarray[np.uint32_t, ndim=1, mode="c"] adc_state_indices,
    np.ndarray[np.float32_t, ndim=2, mode="c"] adc_demodulation,
    np.ndarray[np.uint32_t, ndim=1, mode="c"] crusher_state_indices,
    np.ndarray[np.float32_t, ndim=3, mode="c"] initial_pool_state,
    np.ndarray[np.float32_t, ndim=2, mode="c"] spatial_parameters,
    np.ndarray[np.float32_t, ndim=2, mode="c"] kinetic_parameters,
    np.ndarray[np.float32_t, ndim=1, mode="c"] physical_constants,
    int precision_mode=0,
):
    cdef uint32_t interval_count = <uint32_t>interval_plan.shape[0]
    cdef uint32_t adc_count = <uint32_t>adc_state_indices.shape[0]
    cdef uint32_t crusher_count = <uint32_t>crusher_state_indices.shape[0]
    cdef uint32_t spin_count = <uint32_t>initial_pool_state.shape[0]
    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] final_pool_state
    cdef np.ndarray[np.float32_t, ndim=4, mode="c"] per_spin_signal
    cdef double compile_seconds = 0.0
    cdef double simulation_seconds = 0.0
    cdef char error_message[1024]
    cdef int ok
    cdef const char *source_data = source

    if interval_plan.shape[1] != 16:
        raise ValueError("interval_plan must have shape (interval, 16)")
    if adc_demodulation.shape[0] != adc_count or adc_demodulation.shape[1] != 2:
        raise ValueError("adc_demodulation must have shape (adc, 2)")
    if initial_pool_state.shape[1] != 2 or initial_pool_state.shape[2] != 4:
        raise ValueError("initial_pool_state must have shape (spin, 2, 4)")
    if spatial_parameters.shape[0] != spin_count or spatial_parameters.shape[1] != 4:
        raise ValueError("spatial_parameters must have shape (spin, 4)")
    if kinetic_parameters.shape[0] != spin_count or kinetic_parameters.shape[1] != 4:
        raise ValueError("kinetic_parameters must have shape (spin, 4)")
    if physical_constants.shape[0] != 9:
        raise ValueError("physical_constants must contain nine values")
    if precision_mode not in (0, 1):
        raise ValueError("precision_mode must be 0 (float32) or 1 (double-single)")

    final_pool_state = np.empty((spin_count, 2, 4), dtype=np.float32)
    per_spin_signal = np.empty((spin_count, 2, adc_count, 2), dtype=np.float32)
    with nogil:
        ok = bloch_metal_probe_run(
            source_data,
            &interval_plan[0, 0] if interval_count else NULL,
            interval_count,
            &adc_state_indices[0] if adc_count else NULL,
            &adc_demodulation[0, 0] if adc_count else NULL,
            adc_count,
            &crusher_state_indices[0] if crusher_count else NULL,
            crusher_count,
            &initial_pool_state[0, 0, 0],
            &spatial_parameters[0, 0],
            &kinetic_parameters[0, 0],
            spin_count,
            &physical_constants[0],
            <uint32_t>precision_mode,
            &final_pool_state[0, 0, 0],
            &per_spin_signal[0, 0, 0, 0] if adc_count else NULL,
            &compile_seconds,
            &simulation_seconds,
            error_message,
            1024,
        )
    if not ok:
        raise RuntimeError((<bytes>error_message).decode("utf-8", "replace"))
    return {
        "final_pool_state": final_pool_state,
        "per_spin_species_signal": per_spin_signal,
        "pipeline_compile_seconds": compile_seconds,
        "simulation_seconds": simulation_seconds,
    }
