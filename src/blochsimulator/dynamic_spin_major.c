/* Spin-major strict-float64 executor for two-pool dynamic Bloch plans.
 *
 * Mirrors the interval order of the native full-plan driver in
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

enum {
    F_FAC0R,
    F_FAC0I,
    F_FAC1R,
    F_FAC1I,
    F_EA,
    F_KT,
    F_F0A,
    F_F1A,
    F_KJ0,
    F_KJ1,
    F_EAC,
    F_KTC,
    F_F0AC,
    F_F1AC,
    F_KJ0C,
    F_KJ1C,
    F_COUNT
};

#define DSM_SLOT_DOUBLES (F_COUNT * DSM_TILE)

/* Cached RF propagators: up to eight state components and three source nodes
 * each for excess magnetization and concentration. */
#define DSM_MAX_STATES 8
#define DSM_SOURCE_NODES 3
#define DSM_BUILD_SLOTS 4

typedef struct {
    double half;
    int conversion;
    double eb;
    double f0b;
    double f1b;
    double e2[2];
    double f0bc;
    double f1bc;
} dsm_group;

typedef struct {
    double px[DSM_TILE];
    double py[DSM_TILE];
    double pz[DSM_TILE];
    double fs0[DSM_TILE];
    double fs1[DSM_TILE];
    double kpl[DSM_TILE];
    double d[DSM_TILE];
    double x0[DSM_TILE];
    double y0[DSM_TILE];
    double x1[DSM_TILE];
    double y1[DSM_TILE];
    double P[DSM_TILE];
    double L[DSM_TILE];
    double Cp[DSM_TILE];
    double Cl[DSM_TILE];
    double wg[DSM_TILE];
    double gc[DSM_TILE];
    double gs[DSM_TILE];
    double tx_real[DSM_TILE];
    double tx_imag[DSM_TILE];
} dsm_tile;

/* Per-thread buffers. ``columns`` and ``build_slots`` are NULL when the call
 * builds no propagator. */
typedef struct {
    dsm_tile *tile;
    double *slots;
    double *rw;
    dsm_tile *columns;
    double *build_slots;
    double build_seconds;
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
    g->eb = exp(-c->r1_lactate * h);
    decay_convolution(c->r1_lactate, h, &g->f0b, &g->f1b);
    g->e2[0] = exp(-h / c->t2_s[0]);
    g->e2[1] = exp(-h / c->t2_s[1]);
    decay_convolution(0.0, h, &g->f0bc, &g->f1bc);
}

/* Evaluate every per-spin coefficient of one group into a cache slot. */
static void fill_slot(const dsm_chunk *c, const dsm_group *g,
                      const dsm_tile *t, double *slot) {
    const double h = g->half;
    const double r1l = c->r1_lactate;
    for (int s = 0; s < DSM_TILE; s++) {
        double k = g->conversion ? t->kpl[s] : 0.0;
        double angle = -DSM_TWO_PI * (t->fs0[s] * h);
        slot[F_FAC0R * DSM_TILE + s] = g->e2[0] * cos(angle);
        slot[F_FAC0I * DSM_TILE + s] = g->e2[0] * sin(angle);
        angle = -DSM_TWO_PI * (t->fs1[s] * h);
        slot[F_FAC1R * DSM_TILE + s] = g->e2[1] * cos(angle);
        slot[F_FAC1I * DSM_TILE + s] = g->e2[1] * sin(angle);

        double a = c->r1_pyruvate + k;
        double ea = exp(-a * h);
        double difference = a - r1l;
        slot[F_EA * DSM_TILE + s] = ea;
        slot[F_KT * DSM_TILE + s] = fabs(difference) > 1e-12
                                        ? k * ((g->eb - ea) / difference)
                                        : (k * h) * g->eb;
        if (c->has_sources) {
            double f0a, f1a, j0, j1;
            decay_convolution(a, h, &f0a, &f1a);
            if (fabs(difference * h) > 1e-7) {
                j0 = (g->f0b - f0a) / difference;
                j1 = (g->f1b - f1a) / difference;
            } else {
                equal_rate_convolution(0.5 * (a + r1l), h, &j0, &j1);
            }
            slot[F_F0A * DSM_TILE + s] = f0a;
            slot[F_F1A * DSM_TILE + s] = f1a;
            slot[F_KJ0 * DSM_TILE + s] = k * j0;
            slot[F_KJ1 * DSM_TILE + s] = k * j1;
        }
        if (c->track_concentration) {
            double eac = exp(-k * h);
            slot[F_EAC * DSM_TILE + s] = eac;
            slot[F_KTC * DSM_TILE + s] =
                fabs(k) > 1e-12 ? k * ((1.0 - eac) / k) : (k * h) * 1.0;
            if (c->concentration_sources) {
                double f0ac, f1ac, j0c, j1c;
                decay_convolution(k, h, &f0ac, &f1ac);
                if (fabs(k * h) > 1e-7) {
                    j0c = (g->f0bc - f0ac) / k;
                    j1c = (g->f1bc - f1ac) / k;
                } else {
                    equal_rate_convolution(0.5 * k, h, &j0c, &j1c);
                }
                slot[F_F0AC * DSM_TILE + s] = f0ac;
                slot[F_F1AC * DSM_TILE + s] = f1ac;
                slot[F_KJ0C * DSM_TILE + s] = k * j0c;
                slot[F_KJ1C * DSM_TILE + s] = k * j1c;
            }
        }
    }
}

static inline void transverse_plain(dsm_tile *DSM_RESTRICT t,
                                    const double *DSM_RESTRICT slot) {
    const double *DSM_RESTRICT f0r = slot + F_FAC0R * DSM_TILE;
    const double *DSM_RESTRICT f0i = slot + F_FAC0I * DSM_TILE;
    const double *DSM_RESTRICT f1r = slot + F_FAC1R * DSM_TILE;
    const double *DSM_RESTRICT f1i = slot + F_FAC1I * DSM_TILE;
    for (int s = 0; s < DSM_TILE; s++) {
        double x = t->x0[s];
        double y = t->y0[s];
        t->x0[s] = x * f0r[s] - y * f0i[s];
        t->y0[s] = x * f0i[s] + y * f0r[s];
        x = t->x1[s];
        y = t->y1[s];
        t->x1[s] = x * f1r[s] - y * f1i[s];
        t->y1[s] = x * f1i[s] + y * f1r[s];
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
static inline void transverse_corrected(dsm_tile *DSM_RESTRICT t,
                                        const double *DSM_RESTRICT slot,
                                        int gradient, double da, double db,
                                        int correct) {
    const double *DSM_RESTRICT f0r = slot + F_FAC0R * DSM_TILE;
    const double *DSM_RESTRICT f0i = slot + F_FAC0I * DSM_TILE;
    const double *DSM_RESTRICT f1r = slot + F_FAC1R * DSM_TILE;
    const double *DSM_RESTRICT f1i = slot + F_FAC1I * DSM_TILE;
    const int series =
        fabs(da) <= DSM_PHASE_SERIES_LIMIT && fabs(db) <= DSM_PHASE_SERIES_LIMIT;
    for (int s = 0; s < DSM_TILE; s++) {
        double base_r = gradient ? t->gc[s] : 1.0;
        double base_i = gradient ? t->gs[s] : 0.0;
        double gradient_term = gradient ? t->wg[s] * db : 0.0;
        double rot0_r = base_r, rot0_i = base_i;
        double rot1_r = base_r, rot1_i = base_i;
        if (correct) {
            double cr, ci;
            phase_correction(-DSM_TWO_PI * t->fs0[s] * da + gradient_term,
                             series, &cr, &ci);
            rot0_r = base_r * cr - base_i * ci;
            rot0_i = base_r * ci + base_i * cr;
            phase_correction(-DSM_TWO_PI * t->fs1[s] * da + gradient_term,
                             series, &cr, &ci);
            rot1_r = base_r * cr - base_i * ci;
            rot1_i = base_r * ci + base_i * cr;
        }
        double fr = f0r[s] * rot0_r - f0i[s] * rot0_i;
        double fi = f0r[s] * rot0_i + f0i[s] * rot0_r;
        double x = t->x0[s];
        double y = t->y0[s];
        t->x0[s] = x * fr - y * fi;
        t->y0[s] = x * fi + y * fr;
        fr = f1r[s] * rot1_r - f1i[s] * rot1_i;
        fi = f1r[s] * rot1_i + f1i[s] * rot1_r;
        x = t->x1[s];
        y = t->y1[s];
        t->x1[s] = x * fr - y * fi;
        t->y1[s] = x * fi + y * fr;
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

/* One free longitudinal half-step. ``source`` holds e0, de, c0, dc (global
 * rates without delivery) or is NULL when the interval has no source. */
static inline void longitudinal_half(dsm_tile *DSM_RESTRICT t,
                                     const double *DSM_RESTRICT slot, double eb,
                                     const double *source, double equilibrium,
                                     int track_concentration,
                                     int concentration_sources) {
    const double *DSM_RESTRICT ea = slot + F_EA * DSM_TILE;
    const double *DSM_RESTRICT kt = slot + F_KT * DSM_TILE;
    const double *DSM_RESTRICT f0a = slot + F_F0A * DSM_TILE;
    const double *DSM_RESTRICT f1a = slot + F_F1A * DSM_TILE;
    const double *DSM_RESTRICT kj0 = slot + F_KJ0 * DSM_TILE;
    const double *DSM_RESTRICT kj1 = slot + F_KJ1 * DSM_TILE;
    const double *DSM_RESTRICT eac = slot + F_EAC * DSM_TILE;
    const double *DSM_RESTRICT ktc = slot + F_KTC * DSM_TILE;
    const double *DSM_RESTRICT f0ac = slot + F_F0AC * DSM_TILE;
    const double *DSM_RESTRICT f1ac = slot + F_F1AC * DSM_TILE;
    const double *DSM_RESTRICT kj0c = slot + F_KJ0C * DSM_TILE;
    const double *DSM_RESTRICT kj1c = slot + F_KJ1C * DSM_TILE;
    const int coupled = track_concentration && equilibrium != 0.0;
    const int with_source = source != NULL;
    const int with_concentration_source =
        with_source && track_concentration && concentration_sources;
    const double e0 = with_source ? source[0] : 0.0;
    const double de = with_source ? source[1] : 0.0;
    const double c0 = with_source ? source[2] : 0.0;
    const double dc = with_source ? source[3] : 0.0;

    for (int s = 0; s < DSM_TILE; s++) {
        double p = t->P[s];
        double l = t->L[s];
        double cp = t->Cp[s];
        double cl = t->Cl[s];
        if (coupled) {
            p = p - equilibrium * cp;
            l = l - equilibrium * cl;
        }
        double transfer = kt[s] * p;
        p = p * ea[s];
        l = l * eb;
        l = l + transfer;
        if (with_source) {
            double start = t->d[s] * e0;
            double slope = t->d[s] * de;
            p = p + (start * f0a[s] + slope * f1a[s]);
            l = l + (start * kj0[s] + slope * kj1[s]);
        }
        if (track_concentration) {
            double concentration_transfer = ktc[s] * cp;
            cp = cp * eac[s];
            cl = cl + concentration_transfer;
            if (with_concentration_source) {
                double start = t->d[s] * c0;
                double slope = t->d[s] * dc;
                cp = cp + (start * f0ac[s] + slope * f1ac[s]);
                cl = cl + (start * kj0c[s] + slope * kj1c[s]);
            }
            if (coupled) {
                p = p + equilibrium * cp;
                l = l + equilibrium * cl;
            }
            t->Cp[s] = cp;
            t->Cl[s] = cl;
        }
        t->P[s] = p;
        t->L[s] = l;
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

static void load_tile(const dsm_chunk *c, int64_t first, int valid,
                      dsm_tile *t, double *rw) {
    const int64_t n = c->nspins;
    const int ncoils = c->ncoils;
    memset(t, 0, sizeof(*t));
    memset(rw, 0, sizeof(double) * 2 * (size_t)ncoils * DSM_TILE);
    for (int s = 0; s < valid; s++) {
        int64_t spin = first + s;
        t->px[s] = c->positions_m[3 * spin];
        t->py[s] = c->positions_m[3 * spin + 1];
        t->pz[s] = c->positions_m[3 * spin + 2];
        t->fs0[s] = c->b0_hz[spin] + c->pool_offset_hz[0];
        t->fs1[s] = c->b0_hz[spin] + c->pool_offset_hz[1];
        t->kpl[s] = c->kpl[spin];
        if (c->tx != NULL) {
            t->tx_real[s] = c->tx[2 * spin];
            t->tx_imag[s] = c->tx[2 * spin + 1];
        }
        t->d[s] = c->delivery != NULL ? c->delivery[spin] : 0.0;
        t->x0[s] = c->transverse[2 * spin];
        t->y0[s] = c->transverse[2 * spin + 1];
        t->x1[s] = c->transverse[2 * (n + spin)];
        t->y1[s] = c->transverse[2 * (n + spin) + 1];
        t->P[s] = c->state[3 * spin + 2];
        t->L[s] = c->state[3 * (n + spin) + 2];
        if (c->track_concentration) {
            t->Cp[s] = c->concentration[3 * spin + 2];
            t->Cl[s] = c->concentration[3 * (n + spin) + 2];
        }
        for (int coil = 0; coil < ncoils; coil++) {
            const double *rx = c->rx + 2 * ((int64_t)coil * n + spin);
            rw[(2 * coil) * DSM_TILE + s] = rx[0] * c->weights[spin];
            rw[(2 * coil + 1) * DSM_TILE + s] = rx[1] * c->weights[spin];
        }
    }
}

static void store_tile(const dsm_chunk *c, int64_t first, int valid,
                       const dsm_tile *t) {
    const int64_t n = c->nspins;
    for (int s = 0; s < valid; s++) {
        int64_t spin = first + s;
        c->transverse[2 * spin] = t->x0[s];
        c->transverse[2 * spin + 1] = t->y0[s];
        c->transverse[2 * (n + spin)] = t->x1[s];
        c->transverse[2 * (n + spin) + 1] = t->y1[s];
        c->state[3 * spin + 2] = t->P[s];
        c->state[3 * (n + spin) + 2] = t->L[s];
        if (c->track_concentration) {
            c->concentration[3 * spin + 2] = t->Cp[s];
            c->concentration[3 * (n + spin) + 2] = t->Cl[s];
        }
    }
}

/* Advance a tile across plan interval i. Gradient factors must already be
 * prepared for the first half; ``source`` is an 8-value source row or NULL. */
static inline void step_interval(const dsm_chunk *c, const dsm_group *g,
                                 const double *slot, dsm_tile *t, int32_t i,
                                 const double *source, const double *spatial_rf) {
    const double h = g->half;
    const double pd1 = c->interval_phase_s[2 * i];
    const double pd2 = c->interval_phase_s[2 * i + 1];
    const int32_t rf_index = c->interval_rf[i];
    const int gradient = c->interval_gradient[i] >= 0;

    double da = pd1 - h;
    if (gradient || da != 0.0) {
        transverse_corrected(t, slot, gradient, da, 0.0, da != 0.0);
    } else {
        transverse_plain(t, slot);
    }
    longitudinal_half(t, slot, g->eb, source, c->equilibrium,
                      c->track_concentration, c->concentration_sources);

    if (rf_index >= 0) {
        if (spatial_rf != NULL) {
            rotate_spatial_rf(t->x0, t->y0, t->P, spatial_rf);
            rotate_spatial_rf(t->x1, t->y1, t->L, spatial_rf);
        } else {
            const double *rf = c->rf_table + 5 * (int64_t)rf_index;
            rotate_rf(t->x0, t->y0, t->P, rf);
            rotate_rf(t->x1, t->y1, t->L, rf);
        }
    }

    da = pd2 - h;
    double db = gradient ? pd2 - pd1 : 0.0;
    if (gradient || da != 0.0) {
        transverse_corrected(t, slot, gradient, da, db,
                             da != 0.0 || db != 0.0);
    } else {
        transverse_plain(t, slot);
    }
    longitudinal_half(t, slot, g->eb, source != NULL ? source + 4 : NULL,
                      c->equilibrium, c->track_concentration,
                      c->concentration_sources);
}

/* Boundary actions at state index i + 1. */
static void run_actions(const dsm_chunk *c, dsm_tile *t, const double *rw,
                        int64_t first, int valid, double *adc_block,
                        int32_t i) {
    const int64_t n = c->nspins;
    const int ncoils = c->ncoils;
    const int64_t count = c->adc_payload_count;
    for (int64_t action = c->action_offsets[i];
         action < c->action_offsets[i + 1]; action++) {
        const int kind = c->action_kinds[action];
        const int64_t payload = c->action_payloads[action];
        if (kind == DSM_ACTION_CRUSH) {
            if (c->ideal_crush) {
                memset(t->x0, 0, sizeof(t->x0));
                memset(t->y0, 0, sizeof(t->y0));
                memset(t->x1, 0, sizeof(t->x1));
                memset(t->y1, 0, sizeof(t->y1));
            }
        } else if (kind == DSM_ACTION_ADC) {
            const int64_t index = payload - c->adc_first_payload;
            for (int coil = 0; coil < ncoils; coil++) {
                const double *rw_r = rw + (2 * coil) * DSM_TILE;
                const double *rw_i = rw + (2 * coil + 1) * DSM_TILE;
                accumulate_adc(t->x0, t->y0, rw_r, rw_i,
                               adc_block + 2 * ((int64_t)coil * count + index));
                accumulate_adc(
                    t->x1, t->y1, rw_r, rw_i,
                    adc_block + 2 * ((int64_t)(ncoils + coil) * count + index));
            }
        } else if (kind == DSM_ACTION_CHECKPOINT) {
            double *pyruvate = c->checkpoints + 3 * ((payload * 2) * n);
            double *lactate = c->checkpoints + 3 * ((payload * 2 + 1) * n);
            for (int s = 0; s < valid; s++) {
                int64_t spin = first + s;
                pyruvate[3 * spin] = t->x0[s];
                pyruvate[3 * spin + 1] = t->y0[s];
                pyruvate[3 * spin + 2] = t->P[s];
                lactate[3 * spin] = t->x1[s];
                lactate[3 * spin + 1] = t->y1[s];
                lactate[3 * spin + 2] = t->L[s];
            }
        }
    }
}

static inline int propagator_states(int track_concentration) {
    return track_concentration ? DSM_MAX_STATES : 6;
}

static inline int propagator_columns(int track_concentration, int has_sources,
                                     int concentration_sources) {
    int columns = propagator_states(track_concentration);
    if (has_sources) {
        columns += DSM_SOURCE_NODES;
        if (track_concentration && concentration_sources) {
            columns += DSM_SOURCE_NODES;
        }
    }
    return columns;
}

int32_t dsm_propagator_values(int32_t track_concentration, int32_t has_sources,
                              int32_t concentration_sources) {
    return (int32_t)(propagator_columns(track_concentration != 0,
                                        has_sources != 0,
                                        concentration_sources != 0) *
                     propagator_states(track_concentration != 0));
}

/* Tile state component in propagator order x0, y0, P, x1, y1, L, Cp, Cl. */
static inline double *tile_state(dsm_tile *t, int index) {
    switch (index) {
    case 0:
        return t->x0;
    case 1:
        return t->y0;
    case 2:
        return t->P;
    case 3:
        return t->x1;
    case 4:
        return t->y1;
    case 5:
        return t->L;
    case 6:
        return t->Cp;
    default:
        return t->Cl;
    }
}

static double wall_seconds(void) {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return (double)clock() / CLOCKS_PER_SEC;
#endif
}

/* Build the propagator of intervals [first, end) for one tile into ``out``
 * with layout (column, state, TILE). Every column advances a unit state or a
 * unit source shape through the stepwise interval code, so an entry depends
 * only on the segment's plan values and the spin, not on where it is built. */
static void build_propagator(const dsm_chunk *c, const dsm_group *groups,
                             const dsm_tile *t, int32_t first, int32_t end,
                             const double *basis, dsm_tile *columns,
                             double *build_slots, double *out) {
    const int n_state = propagator_states(c->track_concentration);
    const int n_columns = propagator_columns(
        c->track_concentration, c->has_sources, c->concentration_sources);
    int32_t slot_group[DSM_BUILD_SLOTS];
    int32_t last_used[DSM_BUILD_SLOTS];
    for (int slot = 0; slot < DSM_BUILD_SLOTS; slot++) {
        slot_group[slot] = -1;
        last_used[slot] = -1;
    }
    for (int column = 0; column < n_columns; column++) {
        dsm_tile *u = columns + column;
        memcpy(u, t, sizeof(*u));
        for (int row = 0; row < DSM_MAX_STATES; row++) {
            memset(tile_state(u, row), 0, sizeof(double) * DSM_TILE);
        }
        if (column < n_state) {
            double *unit = tile_state(u, column);
            for (int s = 0; s < DSM_TILE; s++) {
                unit[s] = 1.0;
            }
        }
    }

    for (int32_t i = first; i < end; i++) {
        const int32_t group = c->interval_group[i];
        int slot = 0;
        while (slot < DSM_BUILD_SLOTS && slot_group[slot] != group) {
            slot++;
        }
        if (slot == DSM_BUILD_SLOTS) {
            slot = 0;
            for (int candidate = 1; candidate < DSM_BUILD_SLOTS; candidate++) {
                if (last_used[candidate] < last_used[slot]) {
                    slot = candidate;
                }
            }
            slot_group[slot] = group;
            fill_slot(c, &groups[group], t,
                      build_slots + (size_t)slot * DSM_SLOT_DOUBLES);
        }
        last_used[slot] = i;
        const double *coefficients =
            build_slots + (size_t)slot * DSM_SLOT_DOUBLES;

        const int32_t gradient_index = c->interval_gradient[i];
        if (gradient_index >= 0) {
            prepare_gradient(columns,
                             c->gradient_table + 3 * (int64_t)gradient_index,
                             c->interval_phase_s[2 * i]);
            for (int column = 1; column < n_columns; column++) {
                memcpy(columns[column].wg, columns[0].wg, sizeof(columns[0].wg));
                memcpy(columns[column].gc, columns[0].gc, sizeof(columns[0].gc));
                memcpy(columns[column].gs, columns[0].gs, sizeof(columns[0].gs));
            }
        }

        /* Basis node j holds v0, dv0, v1, dv1 for both halves. Excess shapes
         * fill the e0/de entries of a source row, concentration shapes the
         * c0/dc entries. */
        double spatial_rf[4 * DSM_TILE];
        const int32_t rf_index = c->interval_rf[i];
        const int spatial = c->tx != NULL && rf_index >= 0;
        if (spatial) {
            prepare_spatial_rf(t, c->rf_inputs + 3 * (int64_t)rf_index, spatial_rf);
        }
        const double *nodes = basis + (int64_t)(i - first) * DSM_SOURCE_NODES * 4;
        for (int column = 0; column < n_columns; column++) {
            double row[8];
            const double *source = NULL;
            if (column >= n_state) {
                const int shape = column - n_state;
                const double *node = nodes + 4 * (shape % DSM_SOURCE_NODES);
                const int offset = shape < DSM_SOURCE_NODES ? 0 : 2;
                memset(row, 0, sizeof(row));
                row[offset] = node[0];
                row[offset + 1] = node[1];
                row[offset + 4] = node[2];
                row[offset + 5] = node[3];
                source = row;
            }
            step_interval(c, &groups[group], coefficients, columns + column, i,
                          source, spatial ? spatial_rf : NULL);
        }
    }

    for (int column = 0; column < n_columns; column++) {
        for (int row = 0; row < n_state; row++) {
            memcpy(out + ((size_t)column * n_state + row) * DSM_TILE,
                   tile_state(columns + column, row), sizeof(double) * DSM_TILE);
        }
    }
}

/* Replace the tile state by A u + sum_j w_j S_j. Source columns with a zero
 * weight contribute nothing and are skipped. */
static void apply_propagator(const dsm_chunk *c, dsm_tile *t,
                             const double *DSM_RESTRICT propagator,
                             const double *weights) {
    const int n_state = propagator_states(c->track_concentration);
    const int n_columns = propagator_columns(
        c->track_concentration, c->has_sources, c->concentration_sources);
    double result[DSM_MAX_STATES][DSM_TILE];
    const double *DSM_RESTRICT input = tile_state(t, 0);
    for (int row = 0; row < n_state; row++) {
        const double *DSM_RESTRICT a = propagator + (size_t)row * DSM_TILE;
        for (int s = 0; s < DSM_TILE; s++) {
            result[row][s] = a[s] * input[s];
        }
    }
    for (int column = 1; column < n_state; column++) {
        input = tile_state(t, column);
        for (int row = 0; row < n_state; row++) {
            const double *DSM_RESTRICT a =
                propagator + ((size_t)column * n_state + row) * DSM_TILE;
            for (int s = 0; s < DSM_TILE; s++) {
                result[row][s] = result[row][s] + a[s] * input[s];
            }
        }
    }
    for (int column = n_state; column < n_columns; column++) {
        const double weight = weights[column - n_state];
        if (weight == 0.0) {
            continue;
        }
        for (int row = 0; row < n_state; row++) {
            const double *DSM_RESTRICT a =
                propagator + ((size_t)column * n_state + row) * DSM_TILE;
            for (int s = 0; s < DSM_TILE; s++) {
                result[row][s] = result[row][s] + a[s] * weight;
            }
        }
    }
    for (int row = 0; row < n_state; row++) {
        memcpy(tile_state(t, row), result[row], sizeof(double) * DSM_TILE);
    }
}

static void run_tile(const dsm_chunk *c, const dsm_group *groups,
                     const int32_t *slot_of, const uint8_t *miss, int64_t tile,
                     int64_t n_tiles, int32_t values, dsm_workspace *w,
                     double *adc_block) {
    const int64_t first = tile * DSM_TILE;
    const int64_t remaining = c->nspins - first;
    const int valid = (int)(remaining < DSM_TILE ? remaining : DSM_TILE);
    dsm_tile *t = w->tile;
    load_tile(c, first, valid, t, w->rw);

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
                build_propagator(c, groups, t, i, end,
                                 c->basis_rows +
                                     basis * DSM_SOURCE_NODES * 4,
                                 w->columns, w->build_slots, propagator);
                w->build_seconds += wall_seconds() - started;
            }
            apply_propagator(c, t, propagator,
                             c->segment_sources + 6 * (int64_t)segment);
            i = end - 1;
            segment++;
        } else {
            const dsm_group *g = &groups[c->interval_group[i]];
            double *slot = w->slots + (size_t)slot_of[i] * DSM_SLOT_DOUBLES;
            if (miss[i]) {
                fill_slot(c, g, t, slot);
            }
            const int32_t gradient_index = c->interval_gradient[i];
            const int32_t source_index = c->interval_source[i];
            if (gradient_index >= 0) {
                prepare_gradient(t,
                                 c->gradient_table + 3 * (int64_t)gradient_index,
                                 c->interval_phase_s[2 * i]);
            }
            double spatial_rf[4 * DSM_TILE];
            const int32_t rf_index = c->interval_rf[i];
            const int spatial = c->tx != NULL && rf_index >= 0;
            if (spatial) {
                prepare_spatial_rf(t, c->rf_inputs + 3 * (int64_t)rf_index, spatial_rf);
            }
            step_interval(c, g, slot, t, i,
                          source_index >= 0
                              ? c->source_table + 8 * (int64_t)source_index
                              : NULL, spatial ? spatial_rf : NULL);
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
    for (int32_t i = 0; i < c->n_intervals; i++) {
        if (c->interval_group[i] < 0 || c->interval_group[i] >= c->n_groups ||
            c->interval_rf[i] >= c->n_rf ||
            c->interval_gradient[i] >= c->n_gradient ||
            c->interval_source[i] >= c->n_source) {
            return -1;
        }
        if (c->interval_source[i] >= 0 && !c->has_sources) {
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
    const size_t block_doubles =
        4 * (size_t)c->ncoils * (size_t)c->adc_payload_count;
    const size_t thread_doubles =
        (size_t)DSM_SLOTS * DSM_SLOT_DOUBLES + 2 * (size_t)c->ncoils * DSM_TILE;
    const int32_t values = dsm_propagator_values(
        c->track_concentration, c->has_sources, c->concentration_sources);
    const int n_columns = propagator_columns(
        c->track_concentration, c->has_sources, c->concentration_sources);
    int builds = 0;
    for (int32_t k = 0; k < c->n_segments; k++) {
        builds = builds || c->segment_basis[k] >= 0;
    }

    dsm_group *groups = malloc(sizeof(dsm_group) * (size_t)c->n_groups);
    int32_t *group_slot = malloc(sizeof(int32_t) * (size_t)c->n_groups);
    int32_t *slot_of = malloc(sizeof(int32_t) * (size_t)c->n_intervals);
    uint8_t *miss = malloc((size_t)c->n_intervals);
    dsm_tile *tiles = aligned_buffer(sizeof(dsm_tile) * (size_t)n_threads);
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
        builds ? aligned_buffer(sizeof(double) * DSM_BUILD_SLOTS *
                                DSM_SLOT_DOUBLES * (size_t)n_threads)
               : NULL;
    double build_seconds = 0.0;
    int status = 0;
    if (groups == NULL || group_slot == NULL || slot_of == NULL ||
        miss == NULL || tiles == NULL || thread_buffers == NULL ||
        (block_doubles > 0 && adc_blocks == NULL) ||
        (builds && (column_tiles == NULL || build_buffers == NULL))) {
        status = -2;
        goto cleanup;
    }
    for (int32_t g = 0; g < c->n_groups; g++) {
        prepare_group(c, g, &groups[g]);
    }
    plan_slots(c, group_slot, slot_of, miss);

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
        w.rw = w.slots + (size_t)DSM_SLOTS * DSM_SLOT_DOUBLES;
        w.columns = builds ? column_tiles + (size_t)thread * n_columns : NULL;
        w.build_slots =
            builds ? build_buffers + (size_t)thread * DSM_BUILD_SLOTS *
                                         DSM_SLOT_DOUBLES
                   : NULL;
        w.build_seconds = 0.0;
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
        for (int block = 0; block < n_blocks; block++) {
            const int64_t first_tile = (int64_t)block * n_tiles / n_blocks;
            const int64_t end_tile = (int64_t)(block + 1) * n_tiles / n_blocks;
            double *adc_block =
                adc_blocks != NULL ? adc_blocks + (size_t)block * block_doubles
                                   : NULL;
            for (int64_t tile = first_tile; tile < end_tile; tile++) {
                run_tile(c, groups, slot_of, miss, tile, n_tiles, values, &w,
                         adc_block);
            }
        }
#ifdef _OPENMP
#pragma omp atomic
#endif
        build_seconds += w.build_seconds;
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

cleanup:
    free(groups);
    free(group_slot);
    free(slot_of);
    free(miss);
    aligned_free(tiles);
    aligned_free(thread_buffers);
    free(adc_blocks);
    aligned_free(column_tiles);
    aligned_free(build_buffers);
    return status;
}
