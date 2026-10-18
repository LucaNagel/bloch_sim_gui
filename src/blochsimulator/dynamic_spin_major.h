/* Spin-major strict-float64 executor for multi-pool dynamic Bloch plans.
 *
 * The executor walks a bounded slice of the dynamic execution plan for tiles
 * of DSM_TILE spins. Spin state stays in a structure-of-arrays tile buffer for
 * the whole slice, and per-spin interval coefficients are computed on the fly
 * into a small least-recently-used slot cache keyed by the plan's
 * (half-step, conversion) group. ADC samples are reduced into fixed work-block
 * buffers, so the signal does not depend on the thread count.
 *
 * Pool 0 is the precursor (pyruvate); every other pool is a product fed by a
 * per-spin precursor-to-product rate. Inflow either follows global source
 * rows or, with per-spin arrival delays, delayed time curves evaluated per
 * spin from the plan's interval times.
 */
#ifndef BLOCHSIMULATOR_DYNAMIC_SPIN_MAJOR_H
#define BLOCHSIMULATOR_DYNAMIC_SPIN_MAJOR_H

#include <stdint.h>

#define DSM_TILE 64
#define DSM_SLOTS 32
#define DSM_MAX_POOLS 6

#define DSM_ACTION_CRUSH 1
#define DSM_ACTION_ADC 2
#define DSM_ACTION_CHECKPOINT 3

/* Source modes of spatially delayed inflow (dynamic_phantom.py numbering). */
#define DSM_SOURCE_COUPLED 0
#define DSM_SOURCE_DIRECT 1
#define DSM_SOURCE_POLARIZED 3

/* Scalar time curve with TimeCurve.values_at semantics. */
typedef struct {
    int32_t n;               /* knots; 0 marks an absent polarization curve */
    int32_t step;            /* step interpolation, otherwise linear */
    int32_t hold;            /* hold the end values outside the knots */
    const double *times_s;   /* (n,), strictly increasing */
    const double *values;    /* (n,) */
} dsm_curve;

typedef struct {
    /* Problem size and execution options. */
    int64_t nspins;
    int32_t ncoils;
    int32_t n_threads;
    int32_t n_blocks;
    int32_t npools;
    int32_t track_concentration;
    int32_t has_sources;
    int32_t concentration_sources;
    int32_t ideal_crush;

    /* Pool constants. */
    double r1[DSM_MAX_POOLS];
    double t2_s[DSM_MAX_POOLS];
    double pool_offset_hz[DSM_MAX_POOLS];
    double equilibrium;

    /* Per-spin inputs. */
    const double *positions_m;   /* (nspins, 3) */
    const double *b0_hz;         /* (nspins,) */
    const double *conversion;    /* (npools - 1, nspins) precursor->product */
    const double *delivery;      /* (nspins,), NULL without sources */
    const double *rx;            /* (ncoils, nspins) complex */
    const double *weights;       /* (nspins,) */
    const double *tx;            /* (nspins,) complex, NULL for constant Tx */

    /* State, updated in place. */
    double *transverse;          /* (npools, nspins) complex */
    double *state;               /* (npools, nspins, 3); only z is used */
    double *concentration;       /* (npools, nspins, 3); NULL when untracked */
    double *checkpoints;         /* (n_checkpoints, npools, nspins, 3) or NULL */
    int64_t n_checkpoints;

    /* Coefficient groups keyed by (half step, conversion active). */
    int32_t n_groups;
    const double *group_half_s;  /* (n_groups,) */
    const int8_t *group_conversion;

    /* Plan slice. Table indices are -1 when an interval has no entry. */
    int32_t n_intervals;
    const int32_t *interval_group;
    const int32_t *interval_rf;
    const int32_t *interval_gradient;
    const int32_t *interval_source;
    const double *interval_phase_s; /* (n_intervals, 2): mid-start, end-mid */
    int32_t n_rf;
    int32_t n_gradient;
    int32_t n_source;
    const double *rf_table;         /* (n_rf, 5): ax, ay, cos, sin, 1-cos */
    const double *rf_inputs;        /* (n_rf, 3): real Hz, imag Hz, duration s */
    const double *gradient_table;   /* (n_gradient, 3) Hz/m */
    const double *source_table;     /* (n_source, 8): per half e0, de, c0, dc */

    /* Spatially delayed inflow. When enabled, every interval_source entry is
     * -1 and each half step uses a linear source between the delayed curve
     * values at its start and end (step curves: at its midpoint), exactly as
     * the interval-major spatial source table. */
    int32_t spatial_sources;
    int32_t spatial_mode;           /* DSM_SOURCE_* */
    const double *interval_time_s;  /* (n_intervals, 3): start, mid, end */
    const double *arrival_delay_s;  /* (nspins,) */
    dsm_curve rate_curve;
    dsm_curve polarization_curve;

    /* Boundary actions at state index interval + 1 (CSR layout). */
    int64_t n_actions;
    const int64_t *action_offsets;  /* (n_intervals + 1,) into action arrays */
    const int8_t *action_kinds;
    const int64_t *action_payloads;

    /* Unscaled, undemodulated ADC sums, accumulated in place. */
    int64_t adc_first_payload;
    int64_t adc_payload_count;
    double *adc_signal;             /* (npools, ncoils, adc_payload_count) complex */

    /* Optional cached RF segments (two pools without spatial sources only).
     * Each segment covers intervals [first, end) without boundary actions
     * before its last interval and is replaced by one affine per-spin
     * propagator
     *     u' = A u + sum_j a_j S_j + sum_j c_j Q_j
     * over the tile state u. The source columns S_j and Q_j are the response
     * to the excess and concentration source shapes given by basis rows; the
     * weights a_j and c_j are per segment. A segment with a basis row index
     * builds its propagator entry before applying it. */
    int32_t n_segments;
    const int32_t *segment_first;   /* (n_segments,), sorted, disjoint */
    const int32_t *segment_end;
    const int32_t *segment_entry;   /* propagator entry index */
    const int64_t *segment_basis;   /* first basis row of a build, or -1 */
    const double *segment_sources;  /* (n_segments, 6): a0..a2, c0..c2 */
    int64_t n_basis_rows;
    const double *basis_rows;       /* (n_basis_rows, 3, 4) per node j */
    int32_t n_propagators;
    double *propagators;            /* (n_propagators, n_tiles, values, TILE) */
    double *build_seconds;          /* optional, summed over threads */
} dsm_chunk;

/* Returns 0 on success, -1 for invalid input and -2 for allocation failure. */
int dsm_run_chunk(const dsm_chunk *chunk);

/* Number of doubles per spin of one two-pool propagator entry. */
int32_t dsm_propagator_values(int32_t track_concentration, int32_t has_sources,
                              int32_t concentration_sources);

#endif
