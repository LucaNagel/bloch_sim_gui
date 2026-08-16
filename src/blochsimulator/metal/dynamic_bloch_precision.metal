#include <metal_stdlib>
using namespace metal;

struct ProbeInterval {
    float4 gradient_dt;
    float4 rf_axis_cos_sin;
    float4 rf_one_rate;
    float4 polarization_conversion;
};

struct ProbeParameters {
    uint interval_count;
    uint spin_count;
    uint adc_count;
    uint crusher_count;
    float2 pool_offset_hz;
    float2 r1_s_inv;
    float2 t2_s;
    float equilibrium_polarization;
    float signal_scale;
    uint track_concentration;
    uint reserved;
};

inline float2 complex_multiply(float2 left, float2 right) {
    return float2(
        left.x * right.x - left.y * right.y,
        left.x * right.y + left.y * right.x
    );
}

inline float2 decay_convolution(float rate, float duration) {
    const float x = rate * duration;
    if (fabs(x) < 1.0e-5f) {
        const float x2 = x * x;
        const float x3 = x2 * x;
        return float2(
            duration * (1.0f - x * 0.5f + x2 / 6.0f - x3 / 24.0f),
            duration * duration *
                (0.5f - x / 6.0f + x2 / 24.0f - x3 / 120.0f)
        );
    }
    const float em1 = exp(-x) - 1.0f;
    return float2(
        duration * (-em1) / x,
        duration * duration * (x + em1) / (x * x)
    );
}

inline float2 equal_rate_exchange_convolution(float rate, float duration) {
    const float x = rate * duration;
    if (fabs(x) < 1.0e-4f) {
        const float x2 = x * x;
        const float x3 = x2 * x;
        return float2(
            duration * duration *
                (0.5f - x / 3.0f + x2 / 8.0f - x3 / 30.0f),
            duration * duration * duration *
                (1.0f / 6.0f - x / 12.0f + x2 / 40.0f - x3 / 180.0f)
        );
    }
    const float exp_x = exp(-x);
    return float2(
        duration * duration * (1.0f - exp_x * (1.0f + x)) / (x * x),
        duration * duration * duration *
            (x * (1.0f + exp_x) - 2.0f * (1.0f - exp_x)) /
            (x * x * x)
    );
}

inline float2 advance_zero_target(
    float2 longitudinal,
    float kpl,
    float r1_p,
    float r1_l,
    float duration,
    float source_start,
    float source_end
) {
    if (duration == 0.0f) {
        return longitudinal;
    }
    float pyruvate = longitudinal.x;
    float lactate = longitudinal.y;
    const float previous_pyruvate = pyruvate;
    const float a = r1_p + kpl;
    const float b = r1_l;
    const float exp_a = exp(-a * duration);
    const float exp_b = exp(-b * duration);
    const float difference = a - b;
    float transfer;
    if (fabs(difference) > 1.0e-12f) {
        transfer = kpl * previous_pyruvate * (exp_b - exp_a) / difference;
    } else {
        transfer = kpl * previous_pyruvate * duration * exp_b;
    }
    pyruvate = previous_pyruvate * exp_a;
    lactate = lactate * exp_b + transfer;

    const float source_slope = (source_end - source_start) / duration;
    const float2 fa = decay_convolution(a, duration);
    const float2 fb = decay_convolution(b, duration);
    float2 exchange;
    if (fabs(difference * duration) > 1.0e-7f) {
        exchange = (fb - fa) / difference;
    } else {
        exchange = equal_rate_exchange_convolution(
            0.5f * (a + b), duration
        );
    }
    pyruvate += source_start * fa.x + source_slope * fa.y;
    lactate += kpl * (source_start * exchange.x + source_slope * exchange.y);
    return float2(pyruvate, lactate);
}

inline void advance_longitudinal(
    thread float4 &pyruvate,
    thread float4 &lactate,
    float kpl,
    float2 r1,
    float duration,
    float source_start,
    float source_end,
    float concentration_source_start,
    float concentration_source_end,
    float equilibrium,
    bool track_concentration
) {
    if (!track_concentration) {
        const float2 next = advance_zero_target(
            float2(pyruvate.z, lactate.z),
            kpl,
            r1.x,
            r1.y,
            duration,
            source_start,
            source_end
        );
        pyruvate.z = next.x;
        lactate.z = next.y;
        return;
    }
    pyruvate.z -= equilibrium * pyruvate.w;
    lactate.z -= equilibrium * lactate.w;
    const float2 next_magnetization = advance_zero_target(
        float2(pyruvate.z, lactate.z),
        kpl,
        r1.x,
        r1.y,
        duration,
        source_start - equilibrium * concentration_source_start,
        source_end - equilibrium * concentration_source_end
    );
    pyruvate.z = next_magnetization.x;
    lactate.z = next_magnetization.y;
    const float2 next_concentration = advance_zero_target(
        float2(pyruvate.w, lactate.w),
        kpl,
        0.0f,
        0.0f,
        duration,
        concentration_source_start,
        concentration_source_end
    );
    pyruvate.w = next_concentration.x;
    lactate.w = next_concentration.y;
    pyruvate.z += equilibrium * pyruvate.w;
    lactate.z += equilibrium * lactate.w;
}

inline void advance_transverse(
    thread float4 &state,
    float frequency_hz,
    float t2_s,
    float duration
) {
    const float magnitude = exp(-duration / t2_s);
    const float angle = -2.0f * M_PI_F * frequency_hz * duration;
    const float2 factor = magnitude * float2(cos(angle), sin(angle));
    const float2 value = complex_multiply(state.xy, factor);
    state.x = value.x;
    state.y = value.y;
}

inline void rotate_rf(thread float4 &state, float4 coefficients, float one_minus) {
    const float axis_x = coefficients.x;
    const float axis_y = coefficients.y;
    const float cosine = coefficients.z;
    const float sine = coefficients.w;
    const float vx = state.x;
    const float vy = state.y;
    const float vz = state.z;
    const float projection = vx * axis_x + vy * axis_y;
    state.x = vx * cosine + axis_y * vz * sine +
        projection * axis_x * one_minus;
    state.y = vy * cosine - axis_x * vz * sine +
        projection * axis_y * one_minus;
    state.z = vz * cosine + (axis_x * vy - axis_y * vx) * sine;
}

inline void write_observations(
    uint spin,
    uint state_index,
    thread uint &adc_cursor,
    device const uint *adc_state_indices,
    device const float2 *adc_demodulation,
    device float2 *signals,
    float4 pyruvate,
    float4 lactate,
    float weight,
    constant ProbeParameters &parameters
) {
    while (adc_cursor < parameters.adc_count &&
           adc_state_indices[adc_cursor] == state_index) {
        const float scale = weight * parameters.signal_scale;
        const uint pyruvate_index =
            (spin * 2u) * parameters.adc_count + adc_cursor;
        const uint lactate_index = pyruvate_index + parameters.adc_count;
        signals[pyruvate_index] =
            complex_multiply(pyruvate.xy, adc_demodulation[adc_cursor]) * scale;
        signals[lactate_index] =
            complex_multiply(lactate.xy, adc_demodulation[adc_cursor]) * scale;
        ++adc_cursor;
    }
}

kernel void dynamic_precision_probe(
    device const ProbeInterval *intervals [[buffer(0)]],
    device const uint *adc_state_indices [[buffer(1)]],
    device const float2 *adc_demodulation [[buffer(2)]],
    device const uint *crusher_state_indices [[buffer(3)]],
    device const float4 *initial_pool_state [[buffer(4)]],
    device const float4 *spatial_parameters [[buffer(5)]],
    device const float4 *kinetic_parameters [[buffer(6)]],
    device float4 *final_pool_state [[buffer(7)]],
    device float2 *signals [[buffer(8)]],
    constant ProbeParameters &parameters [[buffer(9)]],
    uint spin [[thread_position_in_grid]]
) {
    if (spin >= parameters.spin_count) {
        return;
    }
    float4 pyruvate = initial_pool_state[spin * 2u];
    float4 lactate = initial_pool_state[spin * 2u + 1u];
    const float4 spatial = spatial_parameters[spin];
    const float4 kinetics = kinetic_parameters[spin];
    const float kpl = kinetics.x;
    const float delivery = kinetics.y;
    const float weight = kinetics.z;
    uint adc_cursor = 0;
    uint crusher_cursor = 0;

    if (parameters.crusher_count > 0 && crusher_state_indices[0] == 0u) {
        pyruvate.xy = 0.0f;
        lactate.xy = 0.0f;
        ++crusher_cursor;
    }
    write_observations(
        spin,
        0u,
        adc_cursor,
        adc_state_indices,
        adc_demodulation,
        signals,
        pyruvate,
        lactate,
        weight,
        parameters
    );

    for (uint interval_index = 0; interval_index < parameters.interval_count;
         ++interval_index) {
        const ProbeInterval interval = intervals[interval_index];
        const float duration = interval.gradient_dt.w;
        const float half_duration = duration * 0.5f;
        const float frequency = spatial.w +
            dot(spatial.xyz, interval.gradient_dt.xyz);
        const float interval_kpl =
            interval.polarization_conversion.w > 0.5f ? kpl : 0.0f;
        const float concentration_start = delivery * interval.rf_one_rate.y;
        const float concentration_mid = delivery * interval.rf_one_rate.z;
        const float concentration_end = delivery * interval.rf_one_rate.w;
        const float source_start =
            concentration_start * interval.polarization_conversion.x;
        const float source_mid =
            concentration_mid * interval.polarization_conversion.y;
        const float source_end =
            concentration_end * interval.polarization_conversion.z;

        advance_transverse(
            pyruvate,
            frequency + parameters.pool_offset_hz.x,
            parameters.t2_s.x,
            half_duration
        );
        advance_transverse(
            lactate,
            frequency + parameters.pool_offset_hz.y,
            parameters.t2_s.y,
            half_duration
        );
        advance_longitudinal(
            pyruvate,
            lactate,
            interval_kpl,
            parameters.r1_s_inv,
            half_duration,
            source_start,
            source_mid,
            concentration_start,
            concentration_mid,
            parameters.equilibrium_polarization,
            parameters.track_concentration != 0u
        );
        rotate_rf(
            pyruvate,
            interval.rf_axis_cos_sin,
            interval.rf_one_rate.x
        );
        rotate_rf(
            lactate,
            interval.rf_axis_cos_sin,
            interval.rf_one_rate.x
        );
        advance_transverse(
            pyruvate,
            frequency + parameters.pool_offset_hz.x,
            parameters.t2_s.x,
            half_duration
        );
        advance_transverse(
            lactate,
            frequency + parameters.pool_offset_hz.y,
            parameters.t2_s.y,
            half_duration
        );
        advance_longitudinal(
            pyruvate,
            lactate,
            interval_kpl,
            parameters.r1_s_inv,
            half_duration,
            source_mid,
            source_end,
            concentration_mid,
            concentration_end,
            parameters.equilibrium_polarization,
            parameters.track_concentration != 0u
        );

        const uint state_index = interval_index + 1u;
        while (crusher_cursor < parameters.crusher_count &&
               crusher_state_indices[crusher_cursor] == state_index) {
            pyruvate.xy = 0.0f;
            lactate.xy = 0.0f;
            ++crusher_cursor;
        }
        write_observations(
            spin,
            state_index,
            adc_cursor,
            adc_state_indices,
            adc_demodulation,
            signals,
            pyruvate,
            lactate,
            weight,
            parameters
        );
    }
    final_pool_state[spin * 2u] = pyruvate;
    final_pool_state[spin * 2u + 1u] = lactate;
}

struct DoubleSingle {
    float hi;
    float lo;
};

struct DoubleSingleState {
    DoubleSingle x;
    DoubleSingle y;
    DoubleSingle z;
    DoubleSingle w;
};

inline DoubleSingle ds_from_float(float value) {
    return DoubleSingle{value, 0.0f};
}

inline float ds_value(DoubleSingle value) {
    return value.hi + value.lo;
}

inline DoubleSingle ds_add(DoubleSingle left, DoubleSingle right) {
    const float sum = left.hi + right.hi;
    const float virtual_right = sum - left.hi;
    const float error =
        (left.hi - (sum - virtual_right)) + (right.hi - virtual_right) +
        left.lo + right.lo;
    const float hi = sum + error;
    return DoubleSingle{hi, error - (hi - sum)};
}

inline DoubleSingle ds_negate(DoubleSingle value) {
    return DoubleSingle{-value.hi, -value.lo};
}

inline DoubleSingle ds_subtract(DoubleSingle left, DoubleSingle right) {
    return ds_add(left, ds_negate(right));
}

inline DoubleSingle ds_multiply_float(DoubleSingle value, float factor) {
    const float product = value.hi * factor;
    const float error = fma(value.hi, factor, -product) + value.lo * factor;
    const float hi = product + error;
    return DoubleSingle{hi, error - (hi - product)};
}

inline DoubleSingle ds_add_float(DoubleSingle value, float addition) {
    return ds_add(value, ds_from_float(addition));
}

inline void ds_advance_zero_target(
    thread DoubleSingle &pyruvate,
    thread DoubleSingle &lactate,
    float kpl,
    float r1_p,
    float r1_l,
    float duration,
    float source_start,
    float source_end
) {
    if (duration == 0.0f) {
        return;
    }
    const DoubleSingle previous_pyruvate = pyruvate;
    const float a = r1_p + kpl;
    const float b = r1_l;
    const float exp_a = exp(-a * duration);
    const float exp_b = exp(-b * duration);
    const float difference = a - b;
    float transfer_coefficient;
    if (fabs(difference) > 1.0e-12f) {
        transfer_coefficient = kpl * (exp_b - exp_a) / difference;
    } else {
        transfer_coefficient = kpl * duration * exp_b;
    }
    pyruvate = ds_multiply_float(previous_pyruvate, exp_a);
    lactate = ds_add(
        ds_multiply_float(lactate, exp_b),
        ds_multiply_float(previous_pyruvate, transfer_coefficient)
    );

    const float source_slope = (source_end - source_start) / duration;
    const float2 fa = decay_convolution(a, duration);
    const float2 fb = decay_convolution(b, duration);
    float2 exchange;
    if (fabs(difference * duration) > 1.0e-7f) {
        exchange = (fb - fa) / difference;
    } else {
        exchange = equal_rate_exchange_convolution(
            0.5f * (a + b), duration
        );
    }
    pyruvate = ds_add_float(
        pyruvate, source_start * fa.x + source_slope * fa.y
    );
    lactate = ds_add_float(
        lactate,
        kpl * (source_start * exchange.x + source_slope * exchange.y)
    );
}

inline void ds_advance_longitudinal(
    thread DoubleSingleState &pyruvate,
    thread DoubleSingleState &lactate,
    float kpl,
    float2 r1,
    float duration,
    float source_start,
    float source_end,
    float concentration_source_start,
    float concentration_source_end,
    float equilibrium,
    bool track_concentration
) {
    if (!track_concentration) {
        ds_advance_zero_target(
            pyruvate.z,
            lactate.z,
            kpl,
            r1.x,
            r1.y,
            duration,
            source_start,
            source_end
        );
        return;
    }
    pyruvate.z = ds_subtract(
        pyruvate.z, ds_multiply_float(pyruvate.w, equilibrium)
    );
    lactate.z = ds_subtract(
        lactate.z, ds_multiply_float(lactate.w, equilibrium)
    );
    ds_advance_zero_target(
        pyruvate.z,
        lactate.z,
        kpl,
        r1.x,
        r1.y,
        duration,
        source_start - equilibrium * concentration_source_start,
        source_end - equilibrium * concentration_source_end
    );
    ds_advance_zero_target(
        pyruvate.w,
        lactate.w,
        kpl,
        0.0f,
        0.0f,
        duration,
        concentration_source_start,
        concentration_source_end
    );
    pyruvate.z = ds_add(
        pyruvate.z, ds_multiply_float(pyruvate.w, equilibrium)
    );
    lactate.z = ds_add(
        lactate.z, ds_multiply_float(lactate.w, equilibrium)
    );
}

inline void ds_advance_transverse(
    thread DoubleSingleState &state,
    float frequency_hz,
    float t2_s,
    float duration
) {
    const float magnitude = exp(-duration / t2_s);
    const float angle = -2.0f * M_PI_F * frequency_hz * duration;
    const float factor_x = magnitude * cos(angle);
    const float factor_y = magnitude * sin(angle);
    const DoubleSingle previous_x = state.x;
    const DoubleSingle previous_y = state.y;
    state.x = ds_subtract(
        ds_multiply_float(previous_x, factor_x),
        ds_multiply_float(previous_y, factor_y)
    );
    state.y = ds_add(
        ds_multiply_float(previous_x, factor_y),
        ds_multiply_float(previous_y, factor_x)
    );
}

inline void ds_rotate_rf(
    thread DoubleSingleState &state,
    float4 coefficients,
    float one_minus
) {
    const float axis_x = coefficients.x;
    const float axis_y = coefficients.y;
    const float cosine = coefficients.z;
    const float sine = coefficients.w;
    const DoubleSingle vx = state.x;
    const DoubleSingle vy = state.y;
    const DoubleSingle vz = state.z;
    const DoubleSingle projection = ds_add(
        ds_multiply_float(vx, axis_x),
        ds_multiply_float(vy, axis_y)
    );
    state.x = ds_add(
        ds_add(
            ds_multiply_float(vx, cosine),
            ds_multiply_float(vz, axis_y * sine)
        ),
        ds_multiply_float(projection, axis_x * one_minus)
    );
    state.y = ds_add(
        ds_subtract(
            ds_multiply_float(vy, cosine),
            ds_multiply_float(vz, axis_x * sine)
        ),
        ds_multiply_float(projection, axis_y * one_minus)
    );
    state.z = ds_add(
        ds_multiply_float(vz, cosine),
        ds_multiply_float(
            ds_subtract(
                ds_multiply_float(vy, axis_x),
                ds_multiply_float(vx, axis_y)
            ),
            sine
        )
    );
}

inline void ds_write_observations(
    uint spin,
    uint state_index,
    thread uint &adc_cursor,
    device const uint *adc_state_indices,
    device const float2 *adc_demodulation,
    device float2 *signals,
    DoubleSingleState pyruvate,
    DoubleSingleState lactate,
    float weight,
    constant ProbeParameters &parameters
) {
    while (adc_cursor < parameters.adc_count &&
           adc_state_indices[adc_cursor] == state_index) {
        const float scale = weight * parameters.signal_scale;
        const uint pyruvate_index =
            (spin * 2u) * parameters.adc_count + adc_cursor;
        const uint lactate_index = pyruvate_index + parameters.adc_count;
        signals[pyruvate_index] = complex_multiply(
            float2(ds_value(pyruvate.x), ds_value(pyruvate.y)),
            adc_demodulation[adc_cursor]
        ) * scale;
        signals[lactate_index] = complex_multiply(
            float2(ds_value(lactate.x), ds_value(lactate.y)),
            adc_demodulation[adc_cursor]
        ) * scale;
        ++adc_cursor;
    }
}

inline DoubleSingleState ds_state_from_float4(float4 value) {
    return DoubleSingleState{
        ds_from_float(value.x),
        ds_from_float(value.y),
        ds_from_float(value.z),
        ds_from_float(value.w)
    };
}

inline float4 ds_state_to_float4(DoubleSingleState value) {
    return float4(
        ds_value(value.x),
        ds_value(value.y),
        ds_value(value.z),
        ds_value(value.w)
    );
}

kernel void dynamic_precision_probe_double_single(
    device const ProbeInterval *intervals [[buffer(0)]],
    device const uint *adc_state_indices [[buffer(1)]],
    device const float2 *adc_demodulation [[buffer(2)]],
    device const uint *crusher_state_indices [[buffer(3)]],
    device const float4 *initial_pool_state [[buffer(4)]],
    device const float4 *spatial_parameters [[buffer(5)]],
    device const float4 *kinetic_parameters [[buffer(6)]],
    device float4 *final_pool_state [[buffer(7)]],
    device float2 *signals [[buffer(8)]],
    constant ProbeParameters &parameters [[buffer(9)]],
    uint spin [[thread_position_in_grid]]
) {
    if (spin >= parameters.spin_count) {
        return;
    }
    DoubleSingleState pyruvate = ds_state_from_float4(
        initial_pool_state[spin * 2u]
    );
    DoubleSingleState lactate = ds_state_from_float4(
        initial_pool_state[spin * 2u + 1u]
    );
    const float4 spatial = spatial_parameters[spin];
    const float4 kinetics = kinetic_parameters[spin];
    const float kpl = kinetics.x;
    const float delivery = kinetics.y;
    const float weight = kinetics.z;
    uint adc_cursor = 0;
    uint crusher_cursor = 0;

    if (parameters.crusher_count > 0 && crusher_state_indices[0] == 0u) {
        pyruvate.x = ds_from_float(0.0f);
        pyruvate.y = ds_from_float(0.0f);
        lactate.x = ds_from_float(0.0f);
        lactate.y = ds_from_float(0.0f);
        ++crusher_cursor;
    }
    ds_write_observations(
        spin,
        0u,
        adc_cursor,
        adc_state_indices,
        adc_demodulation,
        signals,
        pyruvate,
        lactate,
        weight,
        parameters
    );

    for (uint interval_index = 0; interval_index < parameters.interval_count;
         ++interval_index) {
        const ProbeInterval interval = intervals[interval_index];
        const float duration = interval.gradient_dt.w;
        const float half_duration = duration * 0.5f;
        const float frequency = spatial.w +
            dot(spatial.xyz, interval.gradient_dt.xyz);
        const float interval_kpl =
            interval.polarization_conversion.w > 0.5f ? kpl : 0.0f;
        const float concentration_start = delivery * interval.rf_one_rate.y;
        const float concentration_mid = delivery * interval.rf_one_rate.z;
        const float concentration_end = delivery * interval.rf_one_rate.w;
        const float source_start =
            concentration_start * interval.polarization_conversion.x;
        const float source_mid =
            concentration_mid * interval.polarization_conversion.y;
        const float source_end =
            concentration_end * interval.polarization_conversion.z;

        ds_advance_transverse(
            pyruvate,
            frequency + parameters.pool_offset_hz.x,
            parameters.t2_s.x,
            half_duration
        );
        ds_advance_transverse(
            lactate,
            frequency + parameters.pool_offset_hz.y,
            parameters.t2_s.y,
            half_duration
        );
        ds_advance_longitudinal(
            pyruvate,
            lactate,
            interval_kpl,
            parameters.r1_s_inv,
            half_duration,
            source_start,
            source_mid,
            concentration_start,
            concentration_mid,
            parameters.equilibrium_polarization,
            parameters.track_concentration != 0u
        );
        ds_rotate_rf(
            pyruvate,
            interval.rf_axis_cos_sin,
            interval.rf_one_rate.x
        );
        ds_rotate_rf(
            lactate,
            interval.rf_axis_cos_sin,
            interval.rf_one_rate.x
        );
        ds_advance_transverse(
            pyruvate,
            frequency + parameters.pool_offset_hz.x,
            parameters.t2_s.x,
            half_duration
        );
        ds_advance_transverse(
            lactate,
            frequency + parameters.pool_offset_hz.y,
            parameters.t2_s.y,
            half_duration
        );
        ds_advance_longitudinal(
            pyruvate,
            lactate,
            interval_kpl,
            parameters.r1_s_inv,
            half_duration,
            source_mid,
            source_end,
            concentration_mid,
            concentration_end,
            parameters.equilibrium_polarization,
            parameters.track_concentration != 0u
        );

        const uint state_index = interval_index + 1u;
        while (crusher_cursor < parameters.crusher_count &&
               crusher_state_indices[crusher_cursor] == state_index) {
            pyruvate.x = ds_from_float(0.0f);
            pyruvate.y = ds_from_float(0.0f);
            lactate.x = ds_from_float(0.0f);
            lactate.y = ds_from_float(0.0f);
            ++crusher_cursor;
        }
        ds_write_observations(
            spin,
            state_index,
            adc_cursor,
            adc_state_indices,
            adc_demodulation,
            signals,
            pyruvate,
            lactate,
            weight,
            parameters
        );
    }
    final_pool_state[spin * 2u] = ds_state_to_float4(pyruvate);
    final_pool_state[spin * 2u + 1u] = ds_state_to_float4(lactate);
}
