#ifndef BLOCHSIMULATOR_DYNAMIC_METAL_PROBE_H
#define BLOCHSIMULATOR_DYNAMIC_METAL_PROBE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int bloch_metal_probe_capability(
    char *device_name,
    size_t device_name_size,
    char *reason,
    size_t reason_size,
    uint64_t *recommended_working_set_bytes,
    int *apple_gpu_family);

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
    size_t error_message_size);

#ifdef __cplusplus
}
#endif

#endif
