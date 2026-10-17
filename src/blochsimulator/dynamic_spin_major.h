/* Spin-major strict-float64 executor for two-pool dynamic Bloch plans.
 *
 * The executor walks a bounded slice of the dynamic execution plan for tiles
 * of DSM_TILE spins. Spin state stays in a structure-of-arrays tile buffer for
 * the whole slice, and per-spin interval coefficients are computed on the fly
 * into a small least-recently-used slot cache keyed by the plan's
 * (half-step, conversion) group. ADC samples are reduced into fixed work-block
 * buffers, so the signal does not depend on the thread count.
 */
#ifndef BLOCHSIMULATOR_DYNAMIC_SPIN_MAJOR_H
#define BLOCHSIMULATOR_DYNAMIC_SPIN_MAJOR_H

#include <stdint.h>

#define DSM_TILE 64
#define DSM_SLOTS 32

#define DSM_ACTION_CRUSH 1
#define DSM_ACTION_ADC 2
#define DSM_ACTION_CHECKPOINT 3

typedef struct {
    /* Problem size and execution options. */
    int64_t nspins;
    int32_t ncoils;
    int32_t n_threads;
    int32_t n_blocks;
    int32_t track_concentration;
    int32_t has_sources;
    int32_t concentration_sources;
    int32_t ideal_crush;

    /* Pool constants. */
    double r1_pyruvate;
    double r1_lactate;
    double t2_s[2];
    double pool_offset_hz[2];
    double equilibrium;

    /* Per-spin inputs. */
    const double *positions_m;   /* (nspins, 3) */
    const double *b0_hz;         /* (nspins,) */
    const double *kpl;           /* (nspins,) */
    const double *delivery;      /* (nspins,), NULL without sources */
    const double *rx;            /* (ncoils, nspins) complex */
    const double *weights;       /* (nspins,) */
    const double *tx;            /* (nspins,) complex, NULL for constant Tx */

    /* State, updated in place. */
    double *transverse;          /* (2, nspins) complex */
    double *state;               /* (2, nspins, 3); only z is used */
    double *concentration;       /* (2, nspins, 3); NULL when untracked */
    double *checkpoints;         /* (n_checkpoints, 2, nspins, 3) or NULL */
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

    /* Boundary actions at state index interval + 1 (CSR layout). */
    int64_t n_actions;
    const int64_t *action_offsets;  /* (n_intervals + 1,) into action arrays */
    const int8_t *action_kinds;
    const int64_t *action_payloads;

    /* Unscaled, undemodulated ADC sums, accumulated in place. */
    int64_t adc_first_payload;
    int64_t adc_payload_count;
    double *adc_signal;             /* (2, ncoils, adc_payload_count) complex */

    /* Optional cached RF segments. Each segment covers intervals
     * [first, end) without boundary actions before its last interval and is
     * replaced by one affine per-spin propagator
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

/* Number of doubles per spin of one propagator entry. */
int32_t dsm_propagator_values(int32_t track_concentration, int32_t has_sources,
                              int32_t concentration_sources);

#endif
