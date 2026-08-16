#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <mutex>
#include <string>

#include "dynamic_metal_probe.h"

namespace {

struct ProbeParameters {
    uint32_t interval_count;
    uint32_t spin_count;
    uint32_t adc_count;
    uint32_t crusher_count;
    float pool_offset_hz[2];
    float r1_s_inv[2];
    float t2_s[2];
    float equilibrium_polarization;
    float signal_scale;
    uint32_t track_concentration;
    uint32_t reserved;
};

std::mutex pipeline_mutex;
id<MTLDevice> cached_device = nil;
id<MTLComputePipelineState> cached_pipeline = nil;
std::string cached_source;
uint32_t cached_precision_mode = UINT32_MAX;

void copy_message(char *target, size_t size, const std::string &message) {
    if (target == nullptr || size == 0) {
        return;
    }
    const size_t count = std::min(size - 1, message.size());
    std::memcpy(target, message.data(), count);
    target[count] = '\0';
}

std::string ns_error_message(NSError *error) {
    if (error == nil) {
        return "unknown Metal error";
    }
    NSString *description = error.localizedDescription;
    return description == nil ? "unknown Metal error" : description.UTF8String;
}

int highest_apple_family(id<MTLDevice> device) {
    int family = 0;
    for (int candidate = 1; candidate <= 10; ++candidate) {
        MTLGPUFamily value = static_cast<MTLGPUFamily>(1000 + candidate);
        if ([device supportsFamily:value]) {
            family = candidate;
        }
    }
    return family;
}

id<MTLBuffer> make_input_buffer(
    id<MTLDevice> device, const void *bytes, NSUInteger length) {
    if (length == 0) {
        length = 1;
        static const uint8_t zero = 0;
        bytes = &zero;
    }
    return [device newBufferWithBytes:bytes
                               length:length
                              options:MTLResourceStorageModeShared];
}

}  // namespace

int bloch_metal_probe_capability(
    char *device_name,
    size_t device_name_size,
    char *reason,
    size_t reason_size,
    uint64_t *recommended_working_set_bytes,
    int *apple_gpu_family) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            copy_message(reason, reason_size, "no Metal device is available");
            return 0;
        }
        copy_message(
            device_name,
            device_name_size,
            device.name == nil ? "Apple GPU" : device.name.UTF8String);
        if (reason != nullptr && reason_size > 0) {
            reason[0] = '\0';
        }
        if (recommended_working_set_bytes != nullptr) {
            *recommended_working_set_bytes =
                static_cast<uint64_t>(device.recommendedMaxWorkingSetSize);
        }
        if (apple_gpu_family != nullptr) {
            *apple_gpu_family = highest_apple_family(device);
        }
        return 1;
    }
}

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
    size_t error_message_size) {
    @autoreleasepool {
        if (source == nullptr || spin_count == 0 || physical_constants == nullptr ||
            precision_mode > 1) {
            copy_message(error_message, error_message_size, "invalid Metal probe input");
            return 0;
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            copy_message(error_message, error_message_size, "no Metal device is available");
            return 0;
        }

        id<MTLComputePipelineState> pipeline = nil;
        double compile_time = 0.0;
        {
            std::lock_guard<std::mutex> guard(pipeline_mutex);
            const std::string requested_source(source);
            if (cached_pipeline == nil || cached_source != requested_source ||
                cached_precision_mode != precision_mode ||
                cached_device.registryID != device.registryID) {
                const auto compile_start = std::chrono::steady_clock::now();
                MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
                options.fastMathEnabled = NO;
                NSError *library_error = nil;
                NSString *source_string = [NSString stringWithUTF8String:source];
                id<MTLLibrary> library = [device newLibraryWithSource:source_string
                                                               options:options
                                                                 error:&library_error];
                if (library == nil) {
                    copy_message(
                        error_message,
                        error_message_size,
                        "Metal library compilation failed: " +
                            ns_error_message(library_error));
                    return 0;
                }
                NSString *function_name = precision_mode == 0
                    ? @"dynamic_precision_probe"
                    : @"dynamic_precision_probe_double_single";
                id<MTLFunction> function = [library newFunctionWithName:function_name];
                if (function == nil) {
                    copy_message(
                        error_message,
                        error_message_size,
                        "Metal source does not define the requested precision probe");
                    return 0;
                }
                NSError *pipeline_error = nil;
                id<MTLComputePipelineState> new_pipeline =
                    [device newComputePipelineStateWithFunction:function
                                                          error:&pipeline_error];
                if (new_pipeline == nil) {
                    copy_message(
                        error_message,
                        error_message_size,
                        "Metal pipeline creation failed: " +
                            ns_error_message(pipeline_error));
                    return 0;
                }
                cached_device = device;
                cached_pipeline = new_pipeline;
                cached_source = requested_source;
                cached_precision_mode = precision_mode;
                const auto compile_end = std::chrono::steady_clock::now();
                compile_time = std::chrono::duration<double>(
                    compile_end - compile_start).count();
            }
            pipeline = cached_pipeline;
        }

        ProbeParameters parameters = {};
        parameters.interval_count = interval_count;
        parameters.spin_count = spin_count;
        parameters.adc_count = adc_count;
        parameters.crusher_count = crusher_count;
        parameters.pool_offset_hz[0] = physical_constants[0];
        parameters.pool_offset_hz[1] = physical_constants[1];
        parameters.r1_s_inv[0] = physical_constants[2];
        parameters.r1_s_inv[1] = physical_constants[3];
        parameters.t2_s[0] = physical_constants[4];
        parameters.t2_s[1] = physical_constants[5];
        parameters.equilibrium_polarization = physical_constants[6];
        parameters.signal_scale = physical_constants[7];
        parameters.track_concentration = physical_constants[8] != 0.0f;

        id<MTLBuffer> interval_buffer = make_input_buffer(
            device, interval_plan, static_cast<NSUInteger>(interval_count) * 16 * sizeof(float));
        id<MTLBuffer> adc_state_buffer = make_input_buffer(
            device, adc_state_indices, static_cast<NSUInteger>(adc_count) * sizeof(uint32_t));
        id<MTLBuffer> demod_buffer = make_input_buffer(
            device, adc_demodulation, static_cast<NSUInteger>(adc_count) * 2 * sizeof(float));
        id<MTLBuffer> crusher_buffer = make_input_buffer(
            device, crusher_state_indices, static_cast<NSUInteger>(crusher_count) * sizeof(uint32_t));
        id<MTLBuffer> initial_buffer = make_input_buffer(
            device, initial_pool_state, static_cast<NSUInteger>(spin_count) * 8 * sizeof(float));
        id<MTLBuffer> spatial_buffer = make_input_buffer(
            device, spatial_parameters, static_cast<NSUInteger>(spin_count) * 4 * sizeof(float));
        id<MTLBuffer> kinetic_buffer = make_input_buffer(
            device, kinetic_parameters, static_cast<NSUInteger>(spin_count) * 4 * sizeof(float));
        id<MTLBuffer> final_buffer = [device newBufferWithLength:
            static_cast<NSUInteger>(spin_count) * 8 * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
        const NSUInteger signal_bytes = static_cast<NSUInteger>(spin_count) * 2 *
            static_cast<NSUInteger>(adc_count) * 2 * sizeof(float);
        id<MTLBuffer> signal_buffer = [device newBufferWithLength:std::max<NSUInteger>(1, signal_bytes)
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> parameter_buffer = make_input_buffer(
            device, &parameters, sizeof(parameters));
        if (interval_buffer == nil || adc_state_buffer == nil || demod_buffer == nil ||
            crusher_buffer == nil || initial_buffer == nil || spatial_buffer == nil ||
            kinetic_buffer == nil || final_buffer == nil || signal_buffer == nil ||
            parameter_buffer == nil) {
            copy_message(error_message, error_message_size, "Metal buffer allocation failed");
            return 0;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:interval_buffer offset:0 atIndex:0];
        [encoder setBuffer:adc_state_buffer offset:0 atIndex:1];
        [encoder setBuffer:demod_buffer offset:0 atIndex:2];
        [encoder setBuffer:crusher_buffer offset:0 atIndex:3];
        [encoder setBuffer:initial_buffer offset:0 atIndex:4];
        [encoder setBuffer:spatial_buffer offset:0 atIndex:5];
        [encoder setBuffer:kinetic_buffer offset:0 atIndex:6];
        [encoder setBuffer:final_buffer offset:0 atIndex:7];
        [encoder setBuffer:signal_buffer offset:0 atIndex:8];
        [encoder setBuffer:parameter_buffer offset:0 atIndex:9];
        const NSUInteger width = std::min<NSUInteger>(
            64, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
        [encoder dispatchThreads:MTLSizeMake(spin_count, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        [encoder endEncoding];
        const auto simulation_start = std::chrono::steady_clock::now();
        [command commit];
        [command waitUntilCompleted];
        const auto simulation_end = std::chrono::steady_clock::now();
        if (command.status == MTLCommandBufferStatusError) {
            copy_message(
                error_message,
                error_message_size,
                "Metal command failed: " + ns_error_message(command.error));
            return 0;
        }

        std::memcpy(
            final_pool_state,
            final_buffer.contents,
            static_cast<size_t>(spin_count) * 8 * sizeof(float));
        if (signal_bytes > 0) {
            std::memcpy(per_spin_species_signal, signal_buffer.contents, signal_bytes);
        }
        if (pipeline_compile_seconds != nullptr) {
            *pipeline_compile_seconds = compile_time;
        }
        if (simulation_seconds != nullptr) {
            *simulation_seconds = std::chrono::duration<double>(
                simulation_end - simulation_start).count();
        }
        if (error_message != nullptr && error_message_size > 0) {
            error_message[0] = '\0';
        }
        return 1;
    }
}
