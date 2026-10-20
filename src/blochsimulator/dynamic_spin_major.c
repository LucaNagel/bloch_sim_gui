/* Spin-major strict-float64 executor for multi-pool dynamic Bloch plans.
 *
 * Mirrors the interval order of the native full-plan drivers in
 * dynamic_bloch_wrapper.pyx: first transverse half, first longitudinal half,
 * RF rotation, second transverse half, second longitudinal half, then the
 * boundary actions at the next state index. Coefficients are evaluated in
 * float64 from the same formulas as dynamic_phantom.py, but per tile instead
 * of per plan block, so results are float64-close rather than bitwise equal.
 */
#include "dynamic_spin_major.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(_MSC_VER)
#include <malloc.h>
#define DSM_RESTRICT __restrict
#else
#define DSM_RESTRICT restrict
#endif

#define DSM_TWO_PI 6.283185307179586476925286766559
#define DSM_PHASE_SERIES_LIMIT 1e-12
#define DSM_MAX_PRODUCTS (DSM_MAX_POOLS - 1)

/* Per-spin coefficient fields of one cache slot for np pools: one complex
 * transverse factor per pool, the precursor decay and source weights, three
 * fields per product, then the same for concentration. 8 * np fields. */
static inline int field_fac_real(int pool) { return 2 * pool; }
static inline int field_fac_imag(int pool) { return 2 * pool + 1; }
static inline int field_ea(int np) { return 2 * np; }
static inline int field_f0a(int np) { return 2 * np + 1; }
static inline int field_f1a(int np) { return 2 * np + 2; }
static inline int field_kt(int np, int j) { return 2 * np + 3 + 3 * j; }
static inline int field_kj0(int np, int j) { return 2 * np + 4 + 3 * j; }
static inline int field_kj1(int np, int j) { return 2 * np + 5 + 3 * j; }
static inline int field_eac(int np) { return 5 * np; }
static inline int field_f0ac(int np) { return 5 * np + 1; }
static inline int field_f1ac(int np) { return 5 * np + 2; }
static inline int field_ktc(int np, int j) { return 5 * np + 3 + 3 * j; }
static inline int field_kj0c(int np, int j) { return 5 * np + 4 + 3 * j; }
static inline int field_kj1c(int np, int j) { return 5 * np + 5 + 3 * j; }
/* Washout factor exp(-loss * half / 2) of a quarter interval. */
static inline int field_washout(int np) { return 8 * np; }

static inline size_t slot_doubles(int np) {
    return (size_t)(8 * np + 1) * DSM_TILE;
}

typedef struct {
    double half;
    int conversion;
    double eb[DSM_MAX_PRODUCTS];
    double f0b[DSM_MAX_PRODUCTS];
    double f1b[DSM_MAX_PRODUCTS];
    double e2[DSM_MAX_POOLS];
    double f0bc;
    double f1bc;
} dsm_group;

/* A time curve as value = slope * (q - x0) + y0 on the regions
 * [bound[r], bound[r + 1]) for r = 0..n+1: before the first knot, between
 * consecutive knots, exactly at the last knot, and after it. This reproduces
 * np.interp with end values (linear) and the clipped searchsorted lookup
 * (step) of TimeCurve.values_at. */
typedef struct {
    int32_t n_regions;
    int step;
    double *bound;   /* (n_regions + 1,) */
    double *slope;   /* (n_regions,) */
    double *x0;
    double *y0;
} dsm_curve_table;

/* Region of the last query, per spin. A query outside [lo, hi) relocates. */
typedef struct {
    double lo[DSM_TILE];
    double hi[DSM_TILE];
    double slope[DSM_TILE];
    double x0[DSM_TILE];
    double y0[DSM_TILE];
} dsm_curve_state;

typedef struct {
    double px[DSM_TILE];
    double py[DSM_TILE];
    double pz[DSM_TILE];
    double fs[DSM_MAX_POOLS][DSM_TILE];
    double k[DSM_MAX_PRODUCTS][DSM_TILE];
    double d[DSM_TILE];
    double delay[DSM_TILE];
    double x[DSM_MAX_POOLS][DSM_TILE];
    double y[DSM_MAX_POOLS][DSM_TILE];
    double z[DSM_MAX_POOLS][DSM_TILE];
    double c[DSM_MAX_POOLS][DSM_TILE];
    double wg[DSM_TILE];
    double gc[DSM_TILE];
    double gs[DSM_TILE];
    double tx_real[DSM_TILE];
    double tx_imag[DSM_TILE];
    double washout_loss[DSM_TILE];
    int32_t washout_compartment[DSM_TILE]; /* -1 outside source compartments */
    int washout_sources;                   /* any valid spin in a compartment */
    dsm_curve_state curve[2];
} dsm_tile;

/* Per-spin linear source of one half step: precursor excess start value and
 * slope, concentration start value and slope. */
typedef struct {
    double start[DSM_TILE];
    double slope[DSM_TILE];
    double cstart[DSM_TILE];
    double cslope[DSM_TILE];
} dsm_source;

/* Read-only state shared by all threads of one call. */
typedef struct {
    const dsm_chunk *c;
    const dsm_group *groups;
    const int32_t *slot_of;
    const uint8_t *miss;
    size_t slot_doubles;
    dsm_curve_table curves[2];
    int polarized;
    int rate_nonzero;
    double support_lo;   /* the rate curve is zero for q < support_lo */
    double support_hi;   /* and for q >= support_hi */
} dsm_context;

/* Per-thread buffers. ``columns`` is NULL when the call builds no
 * propagator and ``build_slots`` when it has no cached segment. */
typedef struct {
    dsm_tile *tile;
    double *slots;
    double *rw;
    dsm_source *sources;
    dsm_tile *columns;
    double *build_slots;
    double build_seconds;
    int64_t stepwise_tiles;
} dsm_workspace;

static void decay_convolution(double rate, double h, double *f0, double *f1) {
    double x = rate * h;
    if (fabs(x) < 1e-5) {
        double x2 = x * x;
        double x3 = x2 * x;
        *f0 = h * (1.0 - x / 2.0 + x2 / 6.0 - x3 / 24.0);
        *f1 = (h * h) * (0.5 - x / 6.0 + x2 / 24.0 - x3 / 120.0);
    } else {
        double em = expm1(-x);
        *f0 = h * (-em) / x;
        *f1 = (h * h) * (x + em) / (x * x);
    }
}

static void equal_rate_convolution(double rate, double h, double *j0,
                                   double *j1) {
    double x = rate * h;
    double h2 = h * h;
    double h3 = h2 * h;
    if (fabs(x) < 1e-4) {
        double x2 = x * x;
        double x3 = x2 * x;
        *j0 = h2 * (0.5 - x / 3.0 + x2 / 8.0 - x3 / 30.0);
        *j1 = h3 * (1.0 / 6.0 - x / 12.0 + x2 / 40.0 - x3 / 180.0);
    } else {
        double ex = exp(-x);
        *j0 = h2 * (1.0 - ex * (1.0 + x)) / (x * x);
        *j1 = h3 * (x * (1.0 + ex) - 2.0 * (1.0 - ex)) / (x * x * x);
    }
}

static void prepare_group(const dsm_chunk *c, int32_t index, dsm_group *g) {
    double h = c->group_half_s[index];
    g->half = h;
    g->conversion = c->group_conversion[index] != 0;
    for (int j = 0; j < c->npools - 1; j++) {
        g->eb[j] = exp(-c->r1[j + 1] * h);
        decay_convolution(c->r1[j + 1], h, &g->f0b[j], &g->f1b[j]);
    }
    for (int pool = 0; pool < c->npools; pool++) {
        g->e2[pool] = exp(-h / c->t2_s[pool]);
    }
    decay_convolution(0.0, h, &g->f0bc, &g->f1bc);
}

/* Evaluate every per-spin coefficient of one group into a cache slot, with
 * the operation order of _prepare_multi_pool_longitudinal_step. */
static void fill_slot(const dsm_chunk *c, const dsm_group *g,
                      const dsm_tile *t, double *slot) {
    const double h = g->half;
    const int np = c->npools;
    const int nproducts = np - 1;
    for (int s = 0; s < DSM_TILE; s++) {
        double k[DSM_MAX_PRODUCTS];
        double ksum = 0.0;
        for (int j = 0; j < nproducts; j++) {
            k[j] = g->conversion ? t->k[j][s] : 0.0;
            ksum = ksum + k[j];
        }
        for (int pool = 0; pool < np; pool++) {
            double angle = -DSM_TWO_PI * (t->fs[pool][s] * h);
            slot[field_fac_real(pool) * DSM_TILE + s] = g->e2[pool] * cos(angle);
            slot[field_fac_imag(pool) * DSM_TILE + s] = g->e2[pool] * sin(angle);
        }

        slot[field_washout(np) * DSM_TILE + s] =
            c->washout_loss != NULL ? exp(-t->washout_loss[s] * (h / 2.0))
                                    : 1.0;

        double a = c->r1[0] + ksum;
        double ea = exp(-a * h);
        slot[field_ea(np) * DSM_TILE + s] = ea;
        for (int j = 0; j < nproducts; j++) {
            double difference = a - c->r1[j + 1];
            slot[field_kt(np, j) * DSM_TILE + s] =
                fabs(difference) > 1e-12
                    ? k[j] * ((g->eb[j] - ea) / difference)
                    : (k[j] * h) * g->eb[j];
        }
        if (c->has_sources) {
            double f0a, f1a;
            decay_convolution(a, h, &f0a, &f1a);
            slot[field_f0a(np) * DSM_TILE + s] = f0a;
            slot[field_f1a(np) * DSM_TILE + s] = f1a;
            for (int j = 0; j < nproducts; j++) {
                double difference = a - c->r1[j + 1];
                double j0, j1;
                if (fabs(difference * h) > 1e-7) {
                    j0 = (g->f0b[j] - f0a) / difference;
                    j1 = (g->f1b[j] - f1a) / difference;
                } else {
                    equal_rate_convolution(0.5 * (a + c->r1[j + 1]), h, &j0, &j1);
                }
                slot[field_kj0(np, j) * DSM_TILE + s] = k[j] * j0;
                slot[field_kj1(np, j) * DSM_TILE + s] = k[j] * j1;
            }
        }
        if (c->track_concentration) {
            double eac = exp(-ksum * h);
            slot[field_eac(np) * DSM_TILE + s] = eac;
            for (int j = 0; j < nproducts; j++) {
                slot[field_ktc(np, j) * DSM_TILE + s] =
                    fabs(ksum) > 1e-12 ? k[j] * ((1.0 - eac) / ksum)
                                       : (k[j] * h) * 1.0;
            }
            if (c->concentration_sources) {
                double f0ac, f1ac, j0c, j1c;
                decay_convolution(ksum, h, &f0ac, &f1ac);
                if (fabs(ksum * h) > 1e-7) {
                    j0c = (g->f0bc - f0ac) / ksum;
                    j1c = (g->f1bc - f1ac) / ksum;
                } else {
                    equal_rate_convolution(0.5 * ksum, h, &j0c, &j1c);
                }
                slot[field_f0ac(np) * DSM_TILE + s] = f0ac;
                slot[field_f1ac(np) * DSM_TILE + s] = f1ac;
                for (int j = 0; j < nproducts; j++) {
                    slot[field_kj0c(np, j) * DSM_TILE + s] = k[j] * j0c;
                    slot[field_kj1c(np, j) * DSM_TILE + s] = k[j] * j1c;
                }
            }
        }
    }
}

static inline void transverse_plain(dsm_tile *DSM_RESTRICT t, int np,
                                    const double *DSM_RESTRICT slot) {
    for (int pool = 0; pool < np; pool++) {
        const double *DSM_RESTRICT fr = slot + field_fac_real(pool) * DSM_TILE;
        const double *DSM_RESTRICT fi = slot + field_fac_imag(pool) * DSM_TILE;
        double *DSM_RESTRICT xs = t->x[pool];
        double *DSM_RESTRICT ys = t->y[pool];
        for (int s = 0; s < DSM_TILE; s++) {
            double x = xs[s];
            double y = ys[s];
            xs[s] = x * fr[s] - y * fi[s];
            ys[s] = x * fi[s] + y * fr[s];
        }
    }
}

static inline void phase_correction(double eps, int series, double *cr,
                                    double *ci) {
    if (series) {
        double eps2 = eps * eps;
        *cr = 1.0 - 0.5 * eps2;
        *ci = eps - eps2 * eps / 6.0;
    } else {
        *cr = cos(eps);
        *ci = sin(eps);
    }
}

/* Apply the cached static factor times the gradient rotation (when present)
 * times a small correction for the difference between the plan's phase
 * duration and the cached half step. */
static inline void transverse_corrected(dsm_tile *DSM_RESTRICT t, int np,
                                        const double *DSM_RESTRICT slot,
                                        int gradient, double da, double db,
                                        int correct) {
    const int series =
        fabs(da) <= DSM_PHASE_SERIES_LIMIT && fabs(db) <= DSM_PHASE_SERIES_LIMIT;
    for (int pool = 0; pool < np; pool++) {
        const double *DSM_RESTRICT f0r = slot + field_fac_real(pool) * DSM_TILE;
        const double *DSM_RESTRICT f0i = slot + field_fac_imag(pool) * DSM_TILE;
        const double *DSM_RESTRICT fs = t->fs[pool];
        double *DSM_RESTRICT xs = t->x[pool];
        double *DSM_RESTRICT ys = t->y[pool];
        for (int s = 0; s < DSM_TILE; s++) {
            double base_r = gradient ? t->gc[s] : 1.0;
            double base_i = gradient ? t->gs[s] : 0.0;
            double gradient_term = gradient ? t->wg[s] * db : 0.0;
            double rot_r = base_r, rot_i = base_i;
            if (correct) {
                double cr, ci;
                phase_correction(-DSM_TWO_PI * fs[s] * da + gradient_term,
                                 series, &cr, &ci);
                rot_r = base_r * cr - base_i * ci;
                rot_i = base_r * ci + base_i * cr;
            }
            double fr = f0r[s] * rot_r - f0i[s] * rot_i;
            double fi = f0r[s] * rot_i + f0i[s] * rot_r;
            double x = xs[s];
            double y = ys[s];
            xs[s] = x * fr - y * fi;
            ys[s] = x * fi + y * fr;
        }
    }
}

static inline void prepare_gradient(dsm_tile *DSM_RESTRICT t, const double *g,
                                    double phase_duration) {
    const double gx = g[0], gy = g[1], gz = g[2];
    for (int s = 0; s < DSM_TILE; s++) {
        double frequency = t->px[s] * gx;
        frequency = frequency + t->py[s] * gy;
        frequency = frequency + t->pz[s] * gz;
        double angle = -DSM_TWO_PI * (frequency * phase_duration);
        t->wg[s] = -DSM_TWO_PI * frequency;
        t->gc[s] = cos(angle);
        t->gs[s] = sin(angle);
    }
}

/* One free longitudinal half-step of every pool. ``source`` holds e0, de,
 * c0, dc (global rates without delivery), ``spatial`` per-spin source values;
 * both are NULL when the half step has no source. */
static inline void longitudinal_half(const dsm_chunk *c, const dsm_group *g,
                                     const double *DSM_RESTRICT slot,
                                     dsm_tile *DSM_RESTRICT t,
                                     const double *source,
                                     const dsm_source *spatial) {
    const int np = c->npools;
    const int nproducts = np - 1;
    const double equilibrium = c->equilibrium;
    const int track_concentration = c->track_concentration;
    const int coupled = track_concentration && equilibrium != 0.0;
    const int with_source = source != NULL || spatial != NULL;
    const int with_concentration_source =
        with_source && track_concentration && c->concentration_sources;
    double precursor[DSM_TILE];
    double start_values[DSM_TILE];
    double slope_values[DSM_TILE];

    if (coupled) {
        for (int pool = 0; pool < np; pool++) {
            double *DSM_RESTRICT z = t->z[pool];
            const double *DSM_RESTRICT cz = t->c[pool];
            for (int s = 0; s < DSM_TILE; s++) {
                z[s] = z[s] - equilibrium * cz[s];
            }
        }
    }
    {
        const double *DSM_RESTRICT ea = slot + field_ea(np) * DSM_TILE;
        double *DSM_RESTRICT z = t->z[0];
        for (int s = 0; s < DSM_TILE; s++) {
            precursor[s] = z[s];
            z[s] = z[s] * ea[s];
        }
    }
    for (int j = 0; j < nproducts; j++) {
        const double *DSM_RESTRICT kt = slot + field_kt(np, j) * DSM_TILE;
        const double eb = g->eb[j];
        double *DSM_RESTRICT z = t->z[j + 1];
        for (int s = 0; s < DSM_TILE; s++) {
            double transfer = kt[s] * precursor[s];
            double l = z[s] * eb;
            z[s] = l + transfer;
        }
    }
    if (with_source) {
        const double *DSM_RESTRICT start = start_values;
        const double *DSM_RESTRICT slope = slope_values;
        if (spatial != NULL) {
            start = spatial->start;
            slope = spatial->slope;
        } else {
            const double e0 = source[0];
            const double de = source[1];
            for (int s = 0; s < DSM_TILE; s++) {
                start_values[s] = t->d[s] * e0;
                slope_values[s] = t->d[s] * de;
            }
        }
        const double *DSM_RESTRICT f0a = slot + field_f0a(np) * DSM_TILE;
        const double *DSM_RESTRICT f1a = slot + field_f1a(np) * DSM_TILE;
        double *DSM_RESTRICT z = t->z[0];
        for (int s = 0; s < DSM_TILE; s++) {
            z[s] = z[s] + (start[s] * f0a[s] + slope[s] * f1a[s]);
        }
        for (int j = 0; j < nproducts; j++) {
            const double *DSM_RESTRICT kj0 = slot + field_kj0(np, j) * DSM_TILE;
            const double *DSM_RESTRICT kj1 = slot + field_kj1(np, j) * DSM_TILE;
            z = t->z[j + 1];
            for (int s = 0; s < DSM_TILE; s++) {
                z[s] = z[s] + (start[s] * kj0[s] + slope[s] * kj1[s]);
            }
        }
    }
    if (track_concentration) {
        {
            const double *DSM_RESTRICT eac = slot + field_eac(np) * DSM_TILE;
            double *DSM_RESTRICT cz = t->c[0];
            for (int s = 0; s < DSM_TILE; s++) {
                precursor[s] = cz[s];
                cz[s] = cz[s] * eac[s];
            }
        }
        for (int j = 0; j < nproducts; j++) {
            const double *DSM_RESTRICT ktc = slot + field_ktc(np, j) * DSM_TILE;
            double *DSM_RESTRICT cz = t->c[j + 1];
            for (int s = 0; s < DSM_TILE; s++) {
                double concentration_transfer = ktc[s] * precursor[s];
                cz[s] = cz[s] + concentration_transfer;
            }
        }
        if (with_concentration_source) {
            const double *DSM_RESTRICT start = start_values;
            const double *DSM_RESTRICT slope = slope_values;
            if (spatial != NULL) {
                start = spatial->cstart;
                slope = spatial->cslope;
            } else {
                const double c0 = source[2];
                const double dc = source[3];
                for (int s = 0; s < DSM_TILE; s++) {
                    start_values[s] = t->d[s] * c0;
                    slope_values[s] = t->d[s] * dc;
                }
            }
            const double *DSM_RESTRICT f0ac = slot + field_f0ac(np) * DSM_TILE;
            const double *DSM_RESTRICT f1ac = slot + field_f1ac(np) * DSM_TILE;
            double *DSM_RESTRICT cz = t->c[0];
            for (int s = 0; s < DSM_TILE; s++) {
                cz[s] = cz[s] + (start[s] * f0ac[s] + slope[s] * f1ac[s]);
            }
            for (int j = 0; j < nproducts; j++) {
                const double *DSM_RESTRICT kj0c =
                    slot + field_kj0c(np, j) * DSM_TILE;
                const double *DSM_RESTRICT kj1c =
                    slot + field_kj1c(np, j) * DSM_TILE;
                cz = t->c[j + 1];
                for (int s = 0; s < DSM_TILE; s++) {
                    cz[s] = cz[s] + (start[s] * kj0c[s] + slope[s] * kj1c[s]);
                }
            }
        }
        if (coupled) {
            for (int pool = 0; pool < np; pool++) {
                double *DSM_RESTRICT z = t->z[pool];
                const double *DSM_RESTRICT cz = t->c[pool];
                for (int s = 0; s < DSM_TILE; s++) {
                    z[s] = z[s] + equilibrium * cz[s];
                }
            }
        }
    }
}

static inline void rotate_rf(double *DSM_RESTRICT vx_out, double *DSM_RESTRICT vy_out,
                             double *DSM_RESTRICT vz_out, const double *rf) {
    const double ax = rf[0], ay = rf[1], cosine = rf[2], sine = rf[3],
                 one_minus_cosine = rf[4];
    for (int s = 0; s < DSM_TILE; s++) {
        double vx = vx_out[s];
        double vy = vy_out[s];
        double vz = vz_out[s];
        double cross_x = ay * vz;
        double cross_y = 0.0 - ax * vz;
        double cross_z = ax * vy - ay * vx;
        double projection = vx * ax;
        projection = projection + vy * ay;
        double x = vx * cosine;
        x = x + cross_x * sine;
        x = x + (projection * ax) * one_minus_cosine;
        double y = vy * cosine;
        y = y + cross_y * sine;
        y = y + (projection * ay) * one_minus_cosine;
        double z = vz * cosine;
        z = z + cross_z * sine;
        vx_out[s] = x;
        vy_out[s] = y;
        vz_out[s] = z;
    }
}

/* Prepare local complex RF fields once per tile and interval. Propagator
 * builds reuse these coefficients for every state/source column. */
static void prepare_spatial_rf(const dsm_tile *t, const double *input,
                               double *rf) {
    for (int s = 0; s < DSM_TILE; s++) {
        const double real = input[0] * t->tx_real[s] - input[1] * t->tx_imag[s];
        const double imag = input[0] * t->tx_imag[s] + input[1] * t->tx_real[s];
        const double nx = -DSM_TWO_PI * real * input[2];
        const double ny = DSM_TWO_PI * imag * input[2];
        const double angle = hypot(nx, ny);
        rf[s] = angle == 0.0 ? 1.0 : nx / angle;
        rf[DSM_TILE + s] = angle == 0.0 ? 0.0 : ny / angle;
        rf[2 * DSM_TILE + s] = cos(angle);
        rf[3 * DSM_TILE + s] = sin(angle);
    }
}

/* Same plane-basis operation order as the spatial NumPy RF rotation. */
static inline void rotate_spatial_rf(double *vx, double *vy, double *vz,
                                     const double *rf) {
    for (int s = 0; s < DSM_TILE; s++) {
        const double ax = rf[s], ay = rf[DSM_TILE + s];
        const double cosine = rf[2 * DSM_TILE + s], sine = rf[3 * DSM_TILE + s];
        const double x = vx[s], y = vy[s], z = vz[s];
        const double parallel = x * ax + y * ay;
        const double perpendicular = -x * ay + y * ax;
        const double rotated = perpendicular * cosine - z * sine;
        vx[s] = parallel * ax - rotated * ay;
        vy[s] = parallel * ay + rotated * ax;
        vz[s] = perpendicular * sine + z * cosine;
    }
}

static inline void accumulate_adc(const double *DSM_RESTRICT x,
                                  const double *DSM_RESTRICT y,
                                  const double *DSM_RESTRICT rw_r,
                                  const double *DSM_RESTRICT rw_i, double *out) {
    double sum_r[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double sum_i[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int base = 0; base < DSM_TILE; base += 8) {
        for (int lane = 0; lane < 8; lane++) {
            int s = base + lane;
            sum_r[lane] += x[s] * rw_r[s] - y[s] * rw_i[s];
            sum_i[lane] += x[s] * rw_i[s] + y[s] * rw_r[s];
        }
    }
    out[0] += ((sum_r[0] + sum_r[1]) + (sum_r[2] + sum_r[3])) +
              ((sum_r[4] + sum_r[5]) + (sum_r[6] + sum_r[7]));
    out[1] += ((sum_i[0] + sum_i[1]) + (sum_i[2] + sum_i[3])) +
              ((sum_i[4] + sum_i[5]) + (sum_i[6] + sum_i[7]));
}

/* Build the region table of one curve. Returns -1 for invalid knots. */
static int build_curve_table(const dsm_curve *curve, dsm_curve_table *table) {
    const int32_t n = curve->n;
    memset(table, 0, sizeof(*table));
    if (n < 1 || curve->times_s == NULL || curve->values == NULL) {
        return -1;
    }
    const double *times = curve->times_s;
    const double *values = curve->values;
    for (int32_t k = 0; k < n; k++) {
        if (!isfinite(times[k]) || !isfinite(values[k]) ||
            (k > 0 && !(times[k] > times[k - 1]))) {
            return -1;
        }
    }
    table->n_regions = n + 2;
    table->step = curve->step != 0;
    table->bound = malloc(sizeof(double) * (size_t)(n + 3));
    table->slope = malloc(sizeof(double) * (size_t)(n + 2));
    table->x0 = malloc(sizeof(double) * (size_t)(n + 2));
    table->y0 = malloc(sizeof(double) * (size_t)(n + 2));
    if (table->bound == NULL || table->slope == NULL || table->x0 == NULL ||
        table->y0 == NULL) {
        return -2;
    }
    table->bound[0] = -INFINITY;
    for (int32_t k = 0; k < n; k++) {
        table->bound[k + 1] = times[k];
    }
    table->bound[n + 1] = nextafter(times[n - 1], INFINITY);
    table->bound[n + 2] = INFINITY;

    table->slope[0] = 0.0;
    table->x0[0] = 0.0;
    table->y0[0] = curve->hold ? values[0] : 0.0;
    for (int32_t j = 0; j + 1 < n; j++) {
        table->slope[j + 1] =
            table->step ? 0.0
                        : (values[j + 1] - values[j]) / (times[j + 1] - times[j]);
        table->x0[j + 1] = times[j];
        table->y0[j + 1] = values[j];
    }
    table->slope[n] = 0.0;
    table->x0[n] = times[n - 1];
    table->y0[n] = values[n - 1];
    table->slope[n + 1] = 0.0;
    table->x0[n + 1] = 0.0;
    table->y0[n + 1] = curve->hold ? values[n - 1] : 0.0;
    return 0;
}

static void free_curve_table(dsm_curve_table *table) {
    free(table->bound);
    free(table->slope);
    free(table->x0);
    free(table->y0);
}

static void reset_curve_state(dsm_curve_state *state) {
    for (int s = 0; s < DSM_TILE; s++) {
        state->lo[s] = INFINITY;
        state->hi[s] = -INFINITY;
    }
}

static void locate_region(const dsm_curve_table *table, dsm_curve_state *state,
                          int s, double q) {
    int32_t lo = 0;
    int32_t hi = table->n_regions;
    while (hi - lo > 1) {
        const int32_t middle = lo + (hi - lo) / 2;
        if (table->bound[middle] <= q) {
            lo = middle;
        } else {
            hi = middle;
        }
    }
    state->lo[s] = table->bound[lo];
    state->hi[s] = table->bound[lo + 1];
    state->slope[s] = table->slope[lo];
    state->x0[s] = table->x0[lo];
    state->y0[s] = table->y0[lo];
}

/* Curve values at time - delay[s]. Queries mostly stay inside the region of
 * the previous one; the rare others are relocated by binary search. */
static void evaluate_curve(const dsm_curve_table *table,
                           dsm_curve_state *DSM_RESTRICT state,
                           const double *DSM_RESTRICT delay, double time,
                           double *DSM_RESTRICT out) {
    int outside = 0;
    for (int s = 0; s < DSM_TILE; s++) {
        const double q = time - delay[s];
        outside |= (q < state->lo[s]) | (q >= state->hi[s]);
    }
    if (outside) {
        for (int s = 0; s < DSM_TILE; s++) {
            const double q = time - delay[s];
            if (q < state->lo[s] || q >= state->hi[s]) {
                locate_region(table, state, s, q);
            }
        }
    }
    for (int s = 0; s < DSM_TILE; s++) {
        const double q = time - delay[s];
        out[s] = state->slope[s] * (q - state->x0[s]) + state->y0[s];
    }
}

/* Delayed curve values of interval i: endpoint values start, mid, end for
 * linear curves, half midpoints for step curves. */
static void sample_curve(const dsm_curve_table *table, dsm_curve_state *state,
                         const double *delay, const double *time,
                         double values[3][DSM_TILE]) {
    if (table->step) {
        evaluate_curve(table, state, delay, 0.5 * (time[0] + time[1]),
                       values[0]);
        evaluate_curve(table, state, delay, 0.5 * (time[1] + time[2]),
                       values[1]);
    } else {
        evaluate_curve(table, state, delay, time[0], values[0]);
        evaluate_curve(table, state, delay, time[1], values[1]);
        evaluate_curve(table, state, delay, time[2], values[2]);
    }
}

/* Per-spin linear sources of both halves of interval i, with the operation
 * order of the interval-major spatial source path. */
static void fill_spatial_sources(const dsm_context *x, dsm_tile *t, int32_t i,
                                 double h, dsm_source *out) {
    const dsm_chunk *c = x->c;
    const double *time = c->interval_time_s + 3 * (int64_t)i;
    const double equilibrium = c->equilibrium;
    const int mode = c->spatial_mode;
    double rate[3][DSM_TILE];
    double polarization[3][DSM_TILE];
    sample_curve(&x->curves[0], &t->curve[0], t->delay, time, rate);
    if (x->polarized) {
        sample_curve(&x->curves[1], &t->curve[1], t->delay, time, polarization);
    }
    for (int half = 0; half < 2; half++) {
        const double *ra = rate[half];
        const double *rb = x->curves[0].step ? rate[half] : rate[half + 1];
        const double *pa = polarization[half];
        const double *pb =
            x->curves[1].step ? polarization[half] : polarization[half + 1];
        dsm_source *o = out + half;
        for (int s = 0; s < DSM_TILE; s++) {
            const double d = t->d[s];
            double source_start, source_end, cs, ce;
            if (mode == DSM_SOURCE_DIRECT) {
                source_start = d * ra[s];
                source_end = d * rb[s];
                cs = 0.0;
                ce = 0.0;
            } else {
                cs = d * ra[s];
                ce = d * rb[s];
                source_start = x->polarized ? cs * pa[s] : cs * 1.0;
                source_end = x->polarized ? ce * pb[s] : ce * 1.0;
                if (mode == DSM_SOURCE_POLARIZED) {
                    cs = 0.0;
                    ce = 0.0;
                }
            }
            const double precursor_start = source_start - equilibrium * cs;
            const double precursor_end = source_end - equilibrium * ce;
            o->start[s] = precursor_start;
            o->slope[s] = (precursor_end - precursor_start) / h;
            o->cstart[s] = cs;
            o->cslope[s] = (ce - cs) / h;
        }
    }
}

static void load_tile(const dsm_chunk *c, int64_t first, int valid,
                      dsm_tile *t, double *rw) {
    const int64_t n = c->nspins;
    const int ncoils = c->ncoils;
    const int np = c->npools;
    memset(t, 0, sizeof(*t));
    memset(rw, 0, sizeof(double) * 2 * (size_t)ncoils * DSM_TILE);
    for (int s = 0; s < DSM_TILE; s++) {
        t->washout_compartment[s] = -1;
    }
    for (int s = 0; s < valid; s++) {
        int64_t spin = first + s;
        t->px[s] = c->positions_m[3 * spin];
        t->py[s] = c->positions_m[3 * spin + 1];
        t->pz[s] = c->positions_m[3 * spin + 2];
        for (int pool = 0; pool < np; pool++) {
            t->fs[pool][s] = c->b0_hz[spin] + c->pool_offset_hz[pool];
            t->x[pool][s] = c->transverse[2 * (pool * n + spin)];
            t->y[pool][s] = c->transverse[2 * (pool * n + spin) + 1];
            t->z[pool][s] = c->state[3 * (pool * n + spin) + 2];
            if (c->track_concentration) {
                t->c[pool][s] = c->concentration[3 * (pool * n + spin) + 2];
            }
        }
        for (int j = 0; j < np - 1; j++) {
            t->k[j][s] = c->conversion[j * n + spin];
        }
        if (c->tx != NULL) {
            t->tx_real[s] = c->tx[2 * spin];
            t->tx_imag[s] = c->tx[2 * spin + 1];
        }
        t->d[s] = c->delivery != NULL ? c->delivery[spin] : 0.0;
        if (c->washout_loss != NULL) {
            t->washout_loss[s] = c->washout_loss[spin];
        }
        if (c->washout_compartment != NULL) {
            t->washout_compartment[s] = c->washout_compartment[spin];
            t->washout_sources |= c->washout_compartment[spin] >= 0;
        }
        if (c->spatial_sources) {
            t->delay[s] = c->arrival_delay_s[spin];
        }
        for (int coil = 0; coil < ncoils; coil++) {
            const double *rx = c->rx + 2 * ((int64_t)coil * n + spin);
            rw[(2 * coil) * DSM_TILE + s] = rx[0] * c->weights[spin];
            rw[(2 * coil + 1) * DSM_TILE + s] = rx[1] * c->weights[spin];
        }
    }
    reset_curve_state(&t->curve[0]);
    reset_curve_state(&t->curve[1]);
}

static void store_tile(const dsm_chunk *c, int64_t first, int valid,
                       const dsm_tile *t) {
    const int64_t n = c->nspins;
    const int np = c->npools;
    for (int s = 0; s < valid; s++) {
        int64_t spin = first + s;
        for (int pool = 0; pool < np; pool++) {
            c->transverse[2 * (pool * n + spin)] = t->x[pool][s];
            c->transverse[2 * (pool * n + spin) + 1] = t->y[pool][s];
            c->state[3 * (pool * n + spin) + 2] = t->z[pool][s];
            if (c->track_concentration) {
                c->concentration[3 * (pool * n + spin) + 2] = t->c[pool][s];
            }
        }
    }
}

/* Scale x, y, z and c of every pool by the per-spin washout factor of the
 * slot, squared when ``twice``. */
static inline void washout_decay(const dsm_chunk *c, dsm_tile *DSM_RESTRICT t,
                                 const double *DSM_RESTRICT slot, int twice) {
    const int np = c->npools;
    const double *DSM_RESTRICT q = slot + field_washout(np) * DSM_TILE;
    double factor[DSM_TILE];
    for (int s = 0; s < DSM_TILE; s++) {
        factor[s] = twice ? q[s] * q[s] : q[s];
    }
    for (int pool = 0; pool < np; pool++) {
        double *DSM_RESTRICT xs = t->x[pool];
        double *DSM_RESTRICT ys = t->y[pool];
        double *DSM_RESTRICT zs = t->z[pool];
        for (int s = 0; s < DSM_TILE; s++) {
            xs[s] = xs[s] * factor[s];
            ys[s] = ys[s] * factor[s];
            zs[s] = zs[s] * factor[s];
        }
        if (c->track_concentration) {
            double *DSM_RESTRICT cs = t->c[pool];
            for (int s = 0; s < DSM_TILE; s++) {
                cs[s] = cs[s] * factor[s];
            }
        }
    }
}

/* Add one half step of compartment gains (compartment, pool, 2) to the
 * spins of source compartments. */
static inline void washout_gain(const dsm_chunk *c, dsm_tile *t,
                                const double *gains) {
    const int np = c->npools;
    for (int s = 0; s < DSM_TILE; s++) {
        const int32_t compartment = t->washout_compartment[s];
        if (compartment < 0) {
            continue;
        }
        const double *row = gains + 2 * (int64_t)compartment * np;
        for (int pool = 0; pool < np; pool++) {
            t->z[pool][s] = t->z[pool][s] + row[2 * pool];
            if (c->track_concentration) {
                t->c[pool][s] = t->c[pool][s] + row[2 * pool + 1];
            }
        }
    }
}

/* Washout gains of interval i for this tile, or NULL. */
static inline const double *washout_row(const dsm_chunk *c, const dsm_tile *t,
                                        int32_t i) {
    if (!t->washout_sources || c->interval_washout == NULL ||
        c->interval_washout[i] < 0) {
        return NULL;
    }
    return c->washout_table + (int64_t)c->interval_washout[i] * 2 *
                                  c->washout_compartments * c->npools * 2;
}

/* Advance a tile across plan interval i. Gradient factors must already be
 * prepared for the first half; ``source`` is an 8-value source row and
 * ``spatial`` holds both halves of per-spin sources, or both are NULL.
 * ``washout`` holds both halves of compartment gains or is NULL. */
static inline void step_interval(const dsm_chunk *c, const dsm_group *g,
                                 const double *slot, dsm_tile *t, int32_t i,
                                 const double *source, const dsm_source *spatial,
                                 const double *spatial_rf,
                                 const double *washout) {
    const int np = c->npools;
    const double h = g->half;
    const double pd1 = c->interval_phase_s[2 * i];
    const double pd2 = c->interval_phase_s[2 * i + 1];
    const int32_t rf_index = c->interval_rf[i];
    const int gradient = c->interval_gradient[i] >= 0;
    const int loss = c->washout_loss != NULL;

    if (loss) {
        washout_decay(c, t, slot, 0);
    }
    double da = pd1 - h;
    if (gradient || da != 0.0) {
        transverse_corrected(t, np, slot, gradient, da, 0.0, da != 0.0);
    } else {
        transverse_plain(t, np, slot);
    }
    longitudinal_half(c, g, slot, t, source, spatial);
    if (washout != NULL) {
        washout_gain(c, t, washout);
    }
    if (loss) {
        washout_decay(c, t, slot, 1);
    }

    if (rf_index >= 0) {
        if (spatial_rf != NULL) {
            for (int pool = 0; pool < np; pool++) {
                rotate_spatial_rf(t->x[pool], t->y[pool], t->z[pool], spatial_rf);
            }
        } else {
            const double *rf = c->rf_table + 5 * (int64_t)rf_index;
            for (int pool = 0; pool < np; pool++) {
                rotate_rf(t->x[pool], t->y[pool], t->z[pool], rf);
            }
        }
    }

    da = pd2 - h;
    double db = gradient ? pd2 - pd1 : 0.0;
    if (gradient || da != 0.0) {
        transverse_corrected(t, np, slot, gradient, da, db,
                             da != 0.0 || db != 0.0);
    } else {
        transverse_plain(t, np, slot);
    }
    longitudinal_half(c, g, slot, t, source != NULL ? source + 4 : NULL,
                      spatial != NULL ? spatial + 1 : NULL);
    if (washout != NULL) {
        washout_gain(c, t, washout + 2 * (int64_t)c->washout_compartments * np);
    }
    if (loss) {
        washout_decay(c, t, slot, 0);
    }
}

/* Boundary actions at state index i + 1. */
static void run_actions(const dsm_chunk *c, dsm_tile *t, const double *rw,
                        int64_t first, int valid, double *adc_block,
                        int32_t i) {
    const int64_t n = c->nspins;
    const int ncoils = c->ncoils;
    const int np = c->npools;
    const int64_t count = c->adc_payload_count;
    for (int64_t action = c->action_offsets[i];
         action < c->action_offsets[i + 1]; action++) {
        const int kind = c->action_kinds[action];
        const int64_t payload = c->action_payloads[action];
        if (kind == DSM_ACTION_CRUSH) {
            if (c->ideal_crush) {
                for (int pool = 0; pool < np; pool++) {
                    memset(t->x[pool], 0, sizeof(t->x[pool]));
                    memset(t->y[pool], 0, sizeof(t->y[pool]));
                }
            }
        } else if (kind == DSM_ACTION_ADC) {
            const int64_t index = payload - c->adc_first_payload;
            for (int pool = 0; pool < np; pool++) {
                for (int coil = 0; coil < ncoils; coil++) {
                    const double *rw_r = rw + (2 * coil) * DSM_TILE;
                    const double *rw_i = rw + (2 * coil + 1) * DSM_TILE;
                    accumulate_adc(
                        t->x[pool], t->y[pool], rw_r, rw_i,
                        adc_block +
                            2 * ((int64_t)(pool * ncoils + coil) * count + index));
                }
            }
        } else if (kind == DSM_ACTION_CHECKPOINT) {
            for (int pool = 0; pool < np; pool++) {
                double *values = c->checkpoints + 3 * ((payload * np + pool) * n);
                for (int s = 0; s < valid; s++) {
                    int64_t spin = first + s;
                    values[3 * spin] = t->x[pool][s];
                    values[3 * spin + 1] = t->y[pool][s];
                    values[3 * spin + 2] = t->z[pool][s];
                }
            }
        }
    }
}

/* Layout of a cached RF propagator. Pools couple only through the precursor:
 * pool 0 evolves on its own, and product p depends on itself and on pool 0
 * but never on another product. A build therefore advances, independent of
 * the pool count,
 *     ``block`` unit states of pool 0 (columns [0, block)),
 *     ``block`` unit states set in all products at once ([block, 2 block)),
 *     the excess and concentration source shapes (from 2 block on),
 * with block = 4 (x, y, z, c) under tracked concentration, else 3. Products
 * never feed each other, so the all-product unit columns give every
 * product's own block exactly, and their pool-0 response is exactly zero.
 * Concentration does not depend on magnetization, so the c row of a column
 * without concentration input is exactly zero and is not stored. */
#define DSM_SOURCE_NODES 3
#define DSM_WEIGHTS (2 * DSM_SOURCE_NODES)
#define DSM_MAX_BLOCK 4
#define DSM_MAX_COLUMNS (2 * DSM_MAX_BLOCK + DSM_WEIGHTS)
#define DSM_BUILD_SLOTS 4

typedef struct {
    int block;   /* state components per pool */
    int excess;  /* excess source shapes */
    int columns; /* build columns */
} dsm_layout;

static inline dsm_layout propagator_layout(int track_concentration,
                                           int has_sources,
                                           int concentration_sources) {
    dsm_layout l;
    l.block = track_concentration ? DSM_MAX_BLOCK : 3;
    l.excess = has_sources ? DSM_SOURCE_NODES : 0;
    l.columns = 2 * l.block + l.excess +
                (has_sources && track_concentration && concentration_sources
                     ? DSM_SOURCE_NODES
                     : 0);
    return l;
}

/* Stored rows of a build column: the c row only where concentration enters,
 * through a c unit state or a concentration source shape. */
static inline int column_rows(const dsm_layout *l, int column) {
    if (l->block == 3) {
        return 3;
    }
    if (column == 3 || column == 7) {
        return 4;
    }
    return column >= 2 * l->block + l->excess ? 4 : 3;
}

/* Build columns of the pool-0 block, or of a product block (own states, the
 * pool-0 states, then the sources), in storage order. Returns the count. */
static int block_columns(const dsm_layout *l, int product, int *out) {
    int n = 0;
    if (product) {
        for (int b = 0; b < l->block; b++) {
            out[n++] = l->block + b;
        }
    }
    for (int b = 0; b < l->block; b++) {
        out[n++] = b;
    }
    for (int column = 2 * l->block; column < l->columns; column++) {
        out[n++] = column;
    }
    return n;
}

static int block_values(const dsm_layout *l, int product) {
    int columns[DSM_MAX_COLUMNS];
    const int n = block_columns(l, product, columns);
    int values = 0;
    for (int k = 0; k < n; k++) {
        values += column_rows(l, columns[k]);
    }
    return values;
}

int32_t dsm_propagator_columns(int32_t track_concentration, int32_t has_sources,
                               int32_t concentration_sources) {
    const dsm_layout l = propagator_layout(
        track_concentration != 0, has_sources != 0, concentration_sources != 0);
    return (int32_t)l.columns;
}

int32_t dsm_propagator_values(int32_t npools, int32_t track_concentration,
                              int32_t has_sources,
                              int32_t concentration_sources) {
    if (npools < 2 || npools > DSM_MAX_POOLS) {
        return 0;
    }
    const dsm_layout l = propagator_layout(
        track_concentration != 0, has_sources != 0, concentration_sources != 0);
    return (int32_t)(block_values(&l, 0) + (npools - 1) * block_values(&l, 1));
}

/* State component x, y, z or c of one pool. */
static inline double *pool_state(dsm_tile *t, int pool, int component) {
    switch (component) {
    case 0:
        return t->x[pool];
    case 1:
        return t->y[pool];
    case 2:
        return t->z[pool];
    default:
        return t->c[pool];
    }
}

static double wall_seconds(void) {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return (double)clock() / CLOCKS_PER_SEC;
#endif
}

/* Least-recently-used coefficient slots local to one build or one stepwise
 * cached segment. Slot contents depend only on the spin and group. */
typedef struct {
    int32_t group[DSM_BUILD_SLOTS];
    int32_t last_used[DSM_BUILD_SLOTS];
} dsm_local_slots;

static void reset_local_slots(dsm_local_slots *l) {
    for (int slot = 0; slot < DSM_BUILD_SLOTS; slot++) {
        l->group[slot] = -1;
        l->last_used[slot] = -1;
    }
}

static const double *local_slot(const dsm_context *x, dsm_local_slots *l,
                                const dsm_tile *t, double *buffer, int32_t i) {
    const int32_t group = x->c->interval_group[i];
    int slot = 0;
    while (slot < DSM_BUILD_SLOTS && l->group[slot] != group) {
        slot++;
    }
    if (slot == DSM_BUILD_SLOTS) {
        slot = 0;
        for (int candidate = 1; candidate < DSM_BUILD_SLOTS; candidate++) {
            if (l->last_used[candidate] < l->last_used[slot]) {
                slot = candidate;
            }
        }
        l->group[slot] = group;
        fill_slot(x->c, &x->groups[group], t,
                  buffer + (size_t)slot * x->slot_doubles);
    }
    l->last_used[slot] = i;
    return buffer + (size_t)slot * x->slot_doubles;
}

/* Delay range of the spins of a tile that receive delayed inflow. */
typedef struct {
    int active;
    double delay_min;
    double delay_max;
} dsm_inflow;

/* Advance a tile stepwise across plan interval i with coefficient slot
 * ``slot``: gradient, local RF and delayed sources of the interval, then the
 * interval itself. An interval whose delayed query times all fall where the
 * rate curve is zero has exactly zero source and takes the source-free
 * update. */
static void advance_interval(const dsm_context *x, dsm_workspace *w,
                             dsm_tile *t, int32_t i, const double *slot,
                             const dsm_inflow *inflow) {
    const dsm_chunk *c = x->c;
    const dsm_group *g = &x->groups[c->interval_group[i]];
    const int32_t gradient_index = c->interval_gradient[i];
    const int32_t source_index = c->interval_source[i];
    if (gradient_index >= 0) {
        prepare_gradient(t, c->gradient_table + 3 * (int64_t)gradient_index,
                         c->interval_phase_s[2 * i]);
    }
    double spatial_rf[4 * DSM_TILE];
    const int32_t rf_index = c->interval_rf[i];
    const int spatial = c->tx != NULL && rf_index >= 0;
    if (spatial) {
        prepare_spatial_rf(t, c->rf_inputs + 3 * (int64_t)rf_index, spatial_rf);
    }
    const dsm_source *delayed = NULL;
    if (inflow->active) {
        const double *time = c->interval_time_s + 3 * (int64_t)i;
        if (!(time[2] - inflow->delay_min < x->support_lo ||
              time[0] - inflow->delay_max >= x->support_hi)) {
            fill_spatial_sources(x, t, i, g->half, w->sources);
            delayed = w->sources;
        }
    }
    step_interval(c, g, slot, t, i,
                  source_index >= 0 ? c->source_table + 8 * (int64_t)source_index
                                    : NULL,
                  delayed, spatial ? spatial_rf : NULL, washout_row(c, t, i));
}

/* Build the propagator of intervals [first, end) for one tile into ``out``:
 * the pool-0 block, then one block per product, each laid out as
 * (column, row, TILE) over its block columns. Every column advances unit
 * states or a unit source shape through the stepwise interval code, so an
 * entry depends only on the segment's plan values and the spin, not on where
 * it is built. */
static void build_propagator(const dsm_context *x, const dsm_tile *t,
                             int32_t first, int32_t end, const double *basis,
                             dsm_tile *columns, double *build_slots,
                             double *out) {
    const dsm_chunk *c = x->c;
    const int np = c->npools;
    const dsm_layout l = propagator_layout(
        c->track_concentration, c->has_sources, c->concentration_sources);
    for (int column = 0; column < l.columns; column++) {
        dsm_tile *u = columns + column;
        memcpy(u, t, sizeof(*u));
        for (int pool = 0; pool < np; pool++) {
            for (int component = 0; component < DSM_MAX_BLOCK; component++) {
                memset(pool_state(u, pool, component), 0,
                       sizeof(double) * DSM_TILE);
            }
        }
        if (column < 2 * l.block) {
            const int component = column % l.block;
            for (int pool = column < l.block ? 0 : 1;
                 pool < (column < l.block ? 1 : np); pool++) {
                double *unit = pool_state(u, pool, component);
                for (int s = 0; s < DSM_TILE; s++) {
                    unit[s] = 1.0;
                }
            }
        }
    }

    dsm_local_slots local;
    reset_local_slots(&local);
    for (int32_t i = first; i < end; i++) {
        const double *coefficients = local_slot(x, &local, t, build_slots, i);
        const dsm_group *g = &x->groups[c->interval_group[i]];
        const int32_t gradient_index = c->interval_gradient[i];
        if (gradient_index >= 0) {
            prepare_gradient(columns,
                             c->gradient_table + 3 * (int64_t)gradient_index,
                             c->interval_phase_s[2 * i]);
            for (int column = 1; column < l.columns; column++) {
                memcpy(columns[column].wg, columns[0].wg, sizeof(columns[0].wg));
                memcpy(columns[column].gc, columns[0].gc, sizeof(columns[0].gc));
                memcpy(columns[column].gs, columns[0].gs, sizeof(columns[0].gs));
            }
        }

        double spatial_rf[4 * DSM_TILE];
        const int32_t rf_index = c->interval_rf[i];
        const int spatial = c->tx != NULL && rf_index >= 0;
        if (spatial) {
            prepare_spatial_rf(t, c->rf_inputs + 3 * (int64_t)rf_index, spatial_rf);
        }
        /* Basis node j holds v0, dv0, v1, dv1 for both halves. Excess shapes
         * fill the e0/de entries of a source row, concentration shapes the
         * c0/dc entries. */
        const double *nodes = basis + (int64_t)(i - first) * DSM_SOURCE_NODES * 4;
        for (int column = 0; column < l.columns; column++) {
            double row[8];
            const double *source = NULL;
            if (column >= 2 * l.block) {
                const int shape = column - 2 * l.block;
                const double *node = nodes + 4 * (shape % DSM_SOURCE_NODES);
                const int offset = shape < l.excess ? 0 : 2;
                memset(row, 0, sizeof(row));
                row[offset] = node[0];
                row[offset + 1] = node[1];
                row[offset + 4] = node[2];
                row[offset + 5] = node[3];
                source = row;
            }
            /* Washout gains are affine and never enter a propagator. */
            step_interval(c, g, coefficients, columns + column, i, source, NULL,
                          spatial ? spatial_rf : NULL, NULL);
        }
    }

    double *o = out;
    for (int pool = 0; pool < np; pool++) {
        int block[DSM_MAX_COLUMNS];
        const int n = block_columns(&l, pool > 0, block);
        for (int k = 0; k < n; k++) {
            const int rows = column_rows(&l, block[k]);
            for (int row = 0; row < rows; row++) {
                memcpy(o, pool_state(columns + block[k], pool, row),
                       sizeof(double) * DSM_TILE);
                o += DSM_TILE;
            }
        }
    }
}

/* Rotate the transverse state of every pool about z by (cosine, sine). */
static void rotate_transverse(dsm_tile *t, int np, double cosine, double sine) {
    for (int pool = 0; pool < np; pool++) {
        double *DSM_RESTRICT xs = t->x[pool];
        double *DSM_RESTRICT ys = t->y[pool];
        for (int s = 0; s < DSM_TILE; s++) {
            const double x = xs[s];
            const double y = ys[s];
            xs[s] = x * cosine - y * sine;
            ys[s] = x * sine + y * cosine;
        }
    }
}

/* Replace the tile state by A u + sum_j w_j S_j with per-spin source weights.
 * Source columns whose weights are all zero contribute nothing and are
 * skipped. Every non-RF operation commutes with rotations about z, so with
 * ``phase`` = (cos t, sin t) the propagator of the same segment with all RF
 * phases advanced by -t is R_z(t) A R_z(-t), applied by rotating the
 * transverse input by -t and the output by t. */
static void apply_propagator(const dsm_chunk *c, dsm_tile *t,
                             const double *DSM_RESTRICT propagator,
                             double weights[DSM_WEIGHTS][DSM_TILE],
                             const double *phase) {
    const int np = c->npools;
    const dsm_layout l = propagator_layout(
        c->track_concentration, c->has_sources, c->concentration_sources);
    const int rotate = phase != NULL && !(phase[0] == 1.0 && phase[1] == 0.0);
    int active[DSM_WEIGHTS];
    for (int k = 0; k < DSM_WEIGHTS; k++) {
        active[k] = 0;
        for (int s = 0; s < DSM_TILE; s++) {
            active[k] |= weights[k][s] != 0.0;
        }
    }
    if (rotate) {
        rotate_transverse(t, np, phase[0], -phase[1]);
    }

    double result[DSM_MAX_POOLS][DSM_MAX_BLOCK][DSM_TILE];
    memset(result, 0, sizeof(result));
    const double *DSM_RESTRICT a = propagator;
    for (int pool = 0; pool < np; pool++) {
        int block[DSM_MAX_COLUMNS];
        const int n = block_columns(&l, pool > 0, block);
        for (int k = 0; k < n; k++) {
            const int column = block[k];
            const int rows = column_rows(&l, column);
            const double *DSM_RESTRICT input = NULL;
            if (column < l.block) {
                input = pool_state(t, 0, column);
            } else if (column < 2 * l.block) {
                input = pool_state(t, pool, column - l.block);
            } else {
                const int shape = column - 2 * l.block;
                const int index =
                    shape < l.excess ? shape : DSM_SOURCE_NODES + shape - l.excess;
                if (active[index]) {
                    input = weights[index];
                }
            }
            if (input != NULL) {
                for (int row = 0; row < rows; row++) {
                    double *DSM_RESTRICT r = result[pool][row];
                    const double *DSM_RESTRICT ar = a + (size_t)row * DSM_TILE;
                    for (int s = 0; s < DSM_TILE; s++) {
                        r[s] = r[s] + ar[s] * input[s];
                    }
                }
            }
            a += (size_t)rows * DSM_TILE;
        }
    }
    for (int pool = 0; pool < np; pool++) {
        for (int component = 0; component < l.block; component++) {
            memcpy(pool_state(t, pool, component), result[pool][component],
                   sizeof(double) * DSM_TILE);
        }
    }
    if (rotate) {
        rotate_transverse(t, np, phase[0], phase[1]);
    }
}

/* Query times of one delayed curve over intervals [first, end): the start of
 * the first and the end of the last interval for linear curves, the first and
 * last half midpoints for step curves. */
static void curve_query_range(const dsm_curve_table *table,
                              const double *time_first,
                              const double *time_last, double *first,
                              double *last) {
    if (table->step) {
        *first = 0.5 * (time_first[0] + time_first[1]);
        *last = 0.5 * (time_last[1] + time_last[2]);
    } else {
        *first = time_first[0];
        *last = time_last[2];
    }
}

/* Locate the region of q_first for spin s; returns nonzero unless q_last lies
 * in the same region, where the curve is one polynomial of degree <= 1. */
static int crosses_region(const dsm_curve_table *table, dsm_curve_state *state,
                          int s, double q_first, double q_last) {
    if (q_first < state->lo[s] || q_first >= state->hi[s]) {
        locate_region(table, state, s, q_first);
    }
    return q_last < state->lo[s] || q_last >= state->hi[s];
}

static inline double region_value(const dsm_curve_state *state, int s,
                                  double q) {
    return state->slope[s] * (q - state->x0[s]) + state->y0[s];
}

/* Per-spin source weights of a cached segment under delayed inflow: the
 * excess and concentration source values at the three basis nodes, without
 * delivery, which the propagator's source columns carry. When every delayed
 * query range of a receiving spin stays inside one region of each curve, its
 * sources are polynomials of degree <= 2 in time, which the quadratic basis
 * reproduces; otherwise the function returns 1 and the tile runs the segment
 * stepwise. */
static int segment_spatial_weights(const dsm_context *x, dsm_tile *t,
                                   const dsm_inflow *inflow, int32_t first,
                                   int32_t end,
                                   double weights[DSM_WEIGHTS][DSM_TILE]) {
    const dsm_chunk *c = x->c;
    memset(weights, 0, sizeof(double) * DSM_WEIGHTS * DSM_TILE);
    if (!inflow->active) {
        return 0;
    }
    const double *time_first = c->interval_time_s + 3 * (int64_t)first;
    const double *time_last = c->interval_time_s + 3 * (int64_t)(end - 1);
    const double *time_middle =
        c->interval_time_s + 3 * (int64_t)(first + (end - first) / 2);
    if (time_last[2] - inflow->delay_min < x->support_lo ||
        time_first[0] - inflow->delay_max >= x->support_hi) {
        return 0;
    }
    const double nodes[DSM_SOURCE_NODES] = {time_first[0], time_middle[1],
                                            time_last[2]};
    const double equilibrium = c->equilibrium;
    const int mode = c->spatial_mode;
    double rate_first, rate_last, polarization_first = 0.0,
                                  polarization_last = 0.0;
    curve_query_range(&x->curves[0], time_first, time_last, &rate_first,
                      &rate_last);
    if (x->polarized) {
        curve_query_range(&x->curves[1], time_first, time_last,
                          &polarization_first, &polarization_last);
    }
    for (int s = 0; s < DSM_TILE; s++) {
        if (t->d[s] == 0.0) {
            continue;
        }
        const double delay = t->delay[s];
        const double q_first = rate_first - delay;
        const double q_last = rate_last - delay;
        if (q_last < x->support_lo || q_first >= x->support_hi) {
            continue;
        }
        if (crosses_region(&x->curves[0], &t->curve[0], s, q_first, q_last)) {
            return 1;
        }
        if (x->polarized &&
            crosses_region(&x->curves[1], &t->curve[1], s,
                           polarization_first - delay,
                           polarization_last - delay)) {
            return 1;
        }
        for (int j = 0; j < DSM_SOURCE_NODES; j++) {
            const double q = nodes[j] - delay;
            const double rate = region_value(&t->curve[0], s, q);
            double magnetization, concentration;
            if (mode == DSM_SOURCE_DIRECT) {
                magnetization = rate;
                concentration = 0.0;
            } else {
                magnetization = x->polarized
                                    ? rate * region_value(&t->curve[1], s, q)
                                    : rate;
                concentration = mode == DSM_SOURCE_POLARIZED ? 0.0 : rate;
            }
            weights[j][s] = magnetization - equilibrium * concentration;
            weights[DSM_SOURCE_NODES + j][s] = concentration;
        }
    }
    return 0;
}

static void run_tile(const dsm_context *x, int64_t tile, int64_t n_tiles,
                     int32_t values, dsm_workspace *w, double *adc_block) {
    const dsm_chunk *c = x->c;
    const int64_t first = tile * DSM_TILE;
    const int64_t remaining = c->nspins - first;
    const int valid = (int)(remaining < DSM_TILE ? remaining : DSM_TILE);
    dsm_tile *t = w->tile;
    load_tile(c, first, valid, t, w->rw);

    dsm_inflow inflow;
    inflow.active = 0;
    inflow.delay_min = INFINITY;
    inflow.delay_max = -INFINITY;
    if (c->spatial_sources && x->rate_nonzero) {
        for (int s = 0; s < valid; s++) {
            if (t->d[s] != 0.0) {
                inflow.delay_min =
                    t->delay[s] < inflow.delay_min ? t->delay[s] : inflow.delay_min;
                inflow.delay_max =
                    t->delay[s] > inflow.delay_max ? t->delay[s] : inflow.delay_max;
            }
        }
        inflow.active = inflow.delay_min <= inflow.delay_max;
    }

    int32_t segment = 0;
    for (int32_t i = 0; i < c->n_intervals; i++) {
        if (segment < c->n_segments && c->segment_first[segment] == i) {
            const int32_t end = c->segment_end[segment];
            double *propagator =
                c->propagators +
                ((size_t)c->segment_entry[segment] * (size_t)n_tiles +
                 (size_t)tile) *
                    (size_t)values * DSM_TILE;
            const int64_t basis = c->segment_basis[segment];
            if (basis >= 0) {
                const double started = wall_seconds();
                build_propagator(x, t, i, end,
                                 c->basis_rows +
                                     basis * DSM_SOURCE_NODES * 4,
                                 w->columns, w->build_slots, propagator);
                w->build_seconds += wall_seconds() - started;
            }
            double weights[DSM_WEIGHTS][DSM_TILE];
            int stepwise = 0;
            /* Washout gains are affine and not part of the entry. */
            if (t->washout_sources && c->interval_washout != NULL) {
                for (int32_t k = i; k < end && !stepwise; k++) {
                    stepwise = c->interval_washout[k] >= 0;
                }
            }
            if (c->spatial_sources) {
                stepwise = stepwise ||
                           segment_spatial_weights(x, t, &inflow, i, end, weights);
            } else {
                const double *sources =
                    c->segment_sources + DSM_WEIGHTS * (int64_t)segment;
                for (int k = 0; k < DSM_WEIGHTS; k++) {
                    for (int s = 0; s < DSM_TILE; s++) {
                        weights[k][s] = sources[k];
                    }
                }
            }
            if (stepwise) {
                /* The entry stays valid for later occurrences; only this
                 * one runs stepwise, with local coefficient slots. */
                dsm_local_slots local;
                reset_local_slots(&local);
                for (int32_t k = i; k < end; k++) {
                    advance_interval(x, w, t, k,
                                     local_slot(x, &local, t, w->build_slots, k),
                                     &inflow);
                }
                w->stepwise_tiles++;
            } else {
                apply_propagator(c, t, propagator, weights,
                                 c->segment_phase != NULL
                                     ? c->segment_phase + 2 * (int64_t)segment
                                     : NULL);
            }
            i = end - 1;
            segment++;
        } else {
            double *slot = w->slots + (size_t)x->slot_of[i] * x->slot_doubles;
            if (x->miss[i]) {
                fill_slot(c, &x->groups[c->interval_group[i]], t, slot);
            }
            advance_interval(x, w, t, i, slot, &inflow);
        }
        run_actions(c, t, w->rw, first, valid, adc_block, i);
    }
    store_tile(c, first, valid, t);
}

static int validate(const dsm_chunk *c) {
    if (c->tx != NULL && c->rf_inputs == NULL) {
        return -1;
    }

    if (c->nspins < 0 || c->ncoils < 1 || c->n_threads < 1 ||
        c->n_blocks < 1 || c->n_intervals < 0 || c->n_groups < 1) {
        return -1;
    }
    if (c->npools < 2 || c->npools > DSM_MAX_POOLS || c->conversion == NULL) {
        return -1;
    }
    for (int pool = 0; pool < c->npools; pool++) {
        if (!(c->t2_s[pool] > 0.0)) {
            return -1;
        }
    }
    if (c->track_concentration && c->concentration == NULL) {
        return -1;
    }
    if (c->has_sources && c->delivery == NULL) {
        return -1;
    }
    if (c->adc_payload_count < 0 ||
        (c->adc_payload_count > 0 && c->adc_signal == NULL)) {
        return -1;
    }
    if (c->n_checkpoints > 0 && c->checkpoints == NULL) {
        return -1;
    }
    if (c->spatial_sources) {
        if (!c->has_sources || c->interval_time_s == NULL ||
            c->arrival_delay_s == NULL) {
            return -1;
        }
        if (c->spatial_mode != DSM_SOURCE_COUPLED &&
            c->spatial_mode != DSM_SOURCE_DIRECT &&
            c->spatial_mode != DSM_SOURCE_POLARIZED) {
            return -1;
        }
        if (c->concentration_sources !=
            (c->spatial_mode == DSM_SOURCE_COUPLED)) {
            return -1;
        }
        if (c->rate_curve.n < 1 || c->polarization_curve.n < 0) {
            return -1;
        }
        for (int64_t spin = 0; spin < c->nspins; spin++) {
            if (!isfinite(c->arrival_delay_s[spin])) {
                return -1;
            }
        }
    }
    if (c->washout_loss != NULL) {
        for (int64_t spin = 0; spin < c->nspins; spin++) {
            if (!(c->washout_loss[spin] >= 0.0) ||
                !isfinite(c->washout_loss[spin])) {
                return -1;
            }
        }
    }
    if (c->washout_compartment != NULL) {
        if (c->washout_compartments < 1 || c->n_washout < 0 ||
            (c->n_washout > 0 &&
             (c->washout_table == NULL || c->interval_washout == NULL))) {
            return -1;
        }
        for (int64_t spin = 0; spin < c->nspins; spin++) {
            if (c->washout_compartment[spin] < -1 ||
                c->washout_compartment[spin] >= c->washout_compartments) {
                return -1;
            }
        }
    }
    if (c->interval_washout != NULL) {
        if (c->washout_compartment == NULL) {
            return -1;
        }
        for (int32_t i = 0; i < c->n_intervals; i++) {
            if (c->interval_washout[i] < -1 ||
                c->interval_washout[i] >= c->n_washout) {
                return -1;
            }
        }
    }
    for (int32_t i = 0; i < c->n_intervals; i++) {
        if (c->interval_group[i] < 0 || c->interval_group[i] >= c->n_groups ||
            c->interval_rf[i] >= c->n_rf ||
            c->interval_gradient[i] >= c->n_gradient ||
            c->interval_source[i] >= c->n_source) {
            return -1;
        }
        if (c->interval_source[i] >= 0 &&
            (!c->has_sources || c->spatial_sources)) {
            return -1;
        }
        if (c->action_offsets[i] > c->action_offsets[i + 1]) {
            return -1;
        }
    }
    if (c->n_intervals > 0 &&
        (c->action_offsets[0] < 0 ||
         c->action_offsets[c->n_intervals] > c->n_actions)) {
        return -1;
    }
    if (c->n_intervals > 0) {
        for (int64_t action = c->action_offsets[0];
             action < c->action_offsets[c->n_intervals]; action++) {
            const int kind = c->action_kinds[action];
            const int64_t payload = c->action_payloads[action];
            if (kind == DSM_ACTION_ADC &&
                (payload < c->adc_first_payload ||
                 payload >= c->adc_first_payload + c->adc_payload_count)) {
                return -1;
            }
            if (kind == DSM_ACTION_CHECKPOINT &&
                (payload < 0 || payload >= c->n_checkpoints)) {
                return -1;
            }
        }
    }
    if (c->n_segments < 0 || c->n_basis_rows < 0 || c->n_propagators < 0) {
        return -1;
    }
    if (c->n_segments > 0) {
        if (c->segment_first == NULL || c->segment_end == NULL ||
            c->segment_entry == NULL || c->segment_basis == NULL ||
            c->segment_sources == NULL || c->propagators == NULL) {
            return -1;
        }
        int32_t previous_end = 0;
        for (int32_t k = 0; k < c->n_segments; k++) {
            const int32_t first = c->segment_first[k];
            const int32_t end = c->segment_end[k];
            const int64_t basis = c->segment_basis[k];
            if (first < previous_end || end <= first || end > c->n_intervals ||
                c->segment_entry[k] < 0 ||
                c->segment_entry[k] >= c->n_propagators || basis < -1) {
                return -1;
            }
            if (basis >= 0 && (c->basis_rows == NULL ||
                               basis > c->n_basis_rows - (end - first))) {
                return -1;
            }
            /* Only the last interval of a segment may carry actions. */
            if (c->action_offsets[first] != c->action_offsets[end - 1]) {
                return -1;
            }
            if (c->segment_phase != NULL &&
                !(isfinite(c->segment_phase[2 * (int64_t)k]) &&
                  isfinite(c->segment_phase[2 * (int64_t)k + 1]))) {
                return -1;
            }
            previous_end = end;
        }
    }
    return 0;
}

/* Plan least-recently-used slot assignments once per call. Every tile starts
 * with an empty cache and follows the same plan, so each per-spin coefficient
 * depends only on its spin and group. Propagated segments use no slots. */
static void plan_slots(const dsm_chunk *c, int32_t *group_slot,
                       int32_t *slot_of, uint8_t *miss) {
    int32_t slot_group[DSM_SLOTS];
    int64_t last_used[DSM_SLOTS];
    for (int slot = 0; slot < DSM_SLOTS; slot++) {
        slot_group[slot] = -1;
        last_used[slot] = -1;
    }
    for (int32_t g = 0; g < c->n_groups; g++) {
        group_slot[g] = -1;
    }
    int32_t segment = 0;
    for (int32_t i = 0; i < c->n_intervals; i++) {
        if (segment < c->n_segments && c->segment_first[segment] == i) {
            i = c->segment_end[segment] - 1;
            segment++;
            continue;
        }
        const int32_t g = c->interval_group[i];
        int32_t slot = group_slot[g];
        if (slot >= 0) {
            miss[i] = 0;
        } else {
            slot = 0;
            for (int candidate = 1; candidate < DSM_SLOTS; candidate++) {
                if (last_used[candidate] < last_used[slot]) {
                    slot = candidate;
                }
            }
            if (slot_group[slot] >= 0) {
                group_slot[slot_group[slot]] = -1;
            }
            slot_group[slot] = g;
            group_slot[g] = slot;
            miss[i] = 1;
        }
        last_used[slot] = i;
        slot_of[i] = slot;
    }
}

/* Region tables of the delayed curves and the support of the rate curve. */
static int prepare_curves(const dsm_chunk *c, dsm_context *x) {
    int status = build_curve_table(&c->rate_curve, &x->curves[0]);
    if (status != 0) {
        return status;
    }
    x->polarized = c->spatial_mode != DSM_SOURCE_DIRECT &&
                   c->polarization_curve.n > 0;
    if (x->polarized) {
        status = build_curve_table(&c->polarization_curve, &x->curves[1]);
        if (status != 0) {
            return status;
        }
    }
    const dsm_curve_table *rate = &x->curves[0];
    int32_t first_nonzero = -1;
    int32_t last_nonzero = -1;
    for (int32_t r = 0; r < rate->n_regions; r++) {
        if (rate->slope[r] != 0.0 || rate->y0[r] != 0.0) {
            if (first_nonzero < 0) {
                first_nonzero = r;
            }
            last_nonzero = r;
        }
    }
    x->rate_nonzero = first_nonzero >= 0;
    if (x->rate_nonzero) {
        x->support_lo = rate->bound[first_nonzero];
        x->support_hi = rate->bound[last_nonzero + 1];
    }
    return 0;
}

static void *aligned_buffer(size_t bytes) {
    void *pointer = NULL;
    if (bytes == 0) {
        bytes = 64;
    }
#if defined(_MSC_VER)
    pointer = _aligned_malloc(bytes, 64);
#else
    if (posix_memalign(&pointer, 64, bytes) != 0) {
        pointer = NULL;
    }
#endif
    return pointer;
}

static void aligned_free(void *pointer) {
#if defined(_MSC_VER)
    _aligned_free(pointer);
#else
    free(pointer);
#endif
}

int dsm_run_chunk(const dsm_chunk *c) {
    if (c == NULL || validate(c) != 0) {
        return -1;
    }
    if (c->nspins == 0 || c->n_intervals == 0) {
        return 0;
    }
    const int64_t n_tiles = (c->nspins + DSM_TILE - 1) / DSM_TILE;
    const int n_blocks = (int)(c->n_blocks < n_tiles ? c->n_blocks : n_tiles);
    const int n_threads = c->n_threads < n_blocks ? c->n_threads : n_blocks;
    const size_t slot_size = slot_doubles(c->npools);
    const size_t block_doubles = 2 * (size_t)c->npools * (size_t)c->ncoils *
                                 (size_t)c->adc_payload_count;
    const size_t thread_doubles =
        (size_t)DSM_SLOTS * slot_size + 2 * (size_t)c->ncoils * DSM_TILE;
    const int32_t values =
        dsm_propagator_values(c->npools, c->track_concentration,
                              c->has_sources, c->concentration_sources);
    const int n_columns =
        propagator_layout(c->track_concentration, c->has_sources,
                          c->concentration_sources)
            .columns;
    const int cached = c->n_segments > 0;
    int builds = 0;
    for (int32_t k = 0; k < c->n_segments; k++) {
        builds = builds || c->segment_basis[k] >= 0;
    }

    dsm_context context;
    memset(&context, 0, sizeof(context));
    dsm_group *groups = malloc(sizeof(dsm_group) * (size_t)c->n_groups);
    int32_t *group_slot = malloc(sizeof(int32_t) * (size_t)c->n_groups);
    int32_t *slot_of = malloc(sizeof(int32_t) * (size_t)c->n_intervals);
    uint8_t *miss = malloc((size_t)c->n_intervals);
    dsm_tile *tiles = aligned_buffer(sizeof(dsm_tile) * (size_t)n_threads);
    dsm_source *sources =
        aligned_buffer(sizeof(dsm_source) * 2 * (size_t)n_threads);
    double *thread_buffers =
        aligned_buffer(sizeof(double) * thread_doubles * (size_t)n_threads);
    double *adc_blocks =
        block_doubles > 0
            ? calloc(block_doubles * (size_t)n_blocks, sizeof(double))
            : NULL;
    dsm_tile *column_tiles =
        builds ? aligned_buffer(sizeof(dsm_tile) * (size_t)n_columns *
                                (size_t)n_threads)
               : NULL;
    double *build_buffers =
        cached ? aligned_buffer(sizeof(double) * DSM_BUILD_SLOTS * slot_size *
                                (size_t)n_threads)
               : NULL;
    double build_seconds = 0.0;
    int64_t stepwise_tiles = 0;
    int status = 0;
    if (groups == NULL || group_slot == NULL || slot_of == NULL ||
        miss == NULL || tiles == NULL || sources == NULL ||
        thread_buffers == NULL || (block_doubles > 0 && adc_blocks == NULL) ||
        (builds && column_tiles == NULL) || (cached && build_buffers == NULL)) {
        status = -2;
        goto cleanup;
    }
    if (c->spatial_sources) {
        status = prepare_curves(c, &context);
        if (status != 0) {
            goto cleanup;
        }
    }
    for (int32_t g = 0; g < c->n_groups; g++) {
        prepare_group(c, g, &groups[g]);
    }
    plan_slots(c, group_slot, slot_of, miss);
    context.c = c;
    context.groups = groups;
    context.slot_of = slot_of;
    context.miss = miss;
    context.slot_doubles = slot_size;

#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
#endif
    {
#ifdef _OPENMP
        const int thread = omp_get_thread_num();
#else
        const int thread = 0;
#endif
        dsm_workspace w;
        w.tile = tiles + thread;
        w.slots = thread_buffers + (size_t)thread * thread_doubles;
        w.rw = w.slots + (size_t)DSM_SLOTS * slot_size;
        w.sources = sources + 2 * (size_t)thread;
        w.columns = builds ? column_tiles + (size_t)thread * n_columns : NULL;
        w.build_slots =
            cached ? build_buffers + (size_t)thread * DSM_BUILD_SLOTS * slot_size
                   : NULL;
        w.build_seconds = 0.0;
        w.stepwise_tiles = 0;
        /* MSVC's C OpenMP parser requires a separately declared loop index. */
        int block;
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
        for (block = 0; block < n_blocks; block++) {
            const int64_t first_tile = (int64_t)block * n_tiles / n_blocks;
            const int64_t end_tile = (int64_t)(block + 1) * n_tiles / n_blocks;
            double *adc_block =
                adc_blocks != NULL ? adc_blocks + (size_t)block * block_doubles
                                   : NULL;
            for (int64_t tile = first_tile; tile < end_tile; tile++) {
                run_tile(&context, tile, n_tiles, values, &w, adc_block);
            }
        }
#ifdef _OPENMP
#pragma omp atomic
#endif
        build_seconds += w.build_seconds;
#ifdef _OPENMP
#pragma omp atomic
#endif
        stepwise_tiles += w.stepwise_tiles;
    }

    if (adc_blocks != NULL) {
        for (int block = 0; block < n_blocks; block++) {
            const double *source = adc_blocks + (size_t)block * block_doubles;
            for (size_t index = 0; index < block_doubles; index++) {
                c->adc_signal[index] += source[index];
            }
        }
    }
    if (c->build_seconds != NULL) {
        *c->build_seconds += build_seconds;
    }
    if (c->stepwise_tiles != NULL) {
        *c->stepwise_tiles += stepwise_tiles;
    }

cleanup:
    free_curve_table(&context.curves[0]);
    free_curve_table(&context.curves[1]);
    free(groups);
    free(group_slot);
    free(slot_of);
    free(miss);
    aligned_free(tiles);
    aligned_free(sources);
    aligned_free(thread_buffers);
    free(adc_blocks);
    aligned_free(column_tiles);
    aligned_free(build_buffers);
    return status;
}
