/* bloch_core.h - Header file for Bloch simulator core functions
 * Modified for Python integration via Cython
 * Original by Brian Hargreaves, Stanford University
 */

#ifndef BLOCH_CORE_H
#define BLOCH_CORE_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* Constants */
#define GAMMA   26753.0    /* Gyromagnetic ratio for protons (rad/s/G) */
#define TWOPI   6.283185307179586476925286766559   /* 2*pi */

/* Matrix and vector operations */
void multmatvec(double *mat, double *vec, double *matvec);
void addvecs(double *vec1, double *vec2, double *vecsum);
void adjmat(double *mat, double *adj);
void zeromat(double *mat);
void eyemat(double *mat);
double detmat(double *mat);
void scalemat(double *mat, double scalar);
void invmat(double *mat, double *imat);
void addmats(double *mat1, double *mat2, double *matsum);
void multmats(double *mat1, double *mat2, double *matproduct);
void calcrotmat(double nx, double ny, double nz, double *rmat);
void zerovec(double *vec);

/* Time interval conversion */
int times2intervals(double *endtimes, double *intervals, long n);

/* Core Bloch simulation functions */
int blochsim(double *b1real, double *b1imag,
            double *xgrad, double *ygrad, double *zgrad, double *tsteps,
            int ntime, double *e1, double *e2, double df,
            double dx, double dy, double dz,
            double *mx, double *my, double *mz, int mode);

int blochsimfz(double *b1r, double *b1i, double *gx, double *gy, double *gz,
               double *tp, int ntime, double t1, double t2, double *df, int nf,
               double *dx, double *dy, double *dz, int npos,
               double *mx, double *my, double *mz, int mode);

/* New functions for Python interface */
void blochsim_batch(double *b1real, double *b1imag,
                    double *xgrad, double *ygrad, double *zgrad, double *tsteps,
                    int ntime, double t1, double t2,
                    double *df, int nf,
                    double *dx, double *dy, double *dz, int npos,
                    double *mx, double *my, double *mz,
                    int mode, int num_threads);

void blochsim_batch_optimized(double *b1real, double *b1imag,
                              double *xgrad, double *ygrad, double *zgrad, double *tsteps,
                              int ntime, double t1, double t2,
                              double *df, int nf,
                              double *dx, double *dy, double *dz, int npos,
                              double *mx, double *my, double *mz,
                              int mode, int num_threads);

/* Utility functions for Python */
void calculate_relaxation(double t1, double t2, double dt, double *e1, double *e2);
void set_equilibrium_magnetization(double *mx, double *my, double *mz, int n);

/* Heterogeneous phantom simulation - per-voxel T1/T2/df */
void blochsim_heterogeneous(
    double *b1real, double *b1imag,
    double *gx, double *gy, double *gz, double *tsteps,
    int ntime,
    double *t1_arr, double *t2_arr, double *df_arr,
    double *dx, double *dy, double *dz,
    double *mx_init, double *my_init, double *mz_init,
    int nvoxels,
    double *mx, double *my, double *mz,
    int mode, int num_threads);

/* Optimized version for grouped tissue types (fewer unique T1/T2 combinations) */
void blochsim_heterogeneous_grouped(
    double *b1real, double *b1imag,
    double *gx, double *gy, double *gz, double *tsteps,
    int ntime,
    int *tissue_labels,
    double *t1_per_label, double *t2_per_label,
    int nlabels,
    double *df_arr,
    double *dx, double *dy, double *dz,
    double *mx_init, double *my_init, double *mz_init,
    int nvoxels,
    double *mx, double *my, double *mz,
    int mode, int num_threads);

/* Streaming sequence simulation in canonical frequency units (Hz, Hz/m, m). */
int blochsim_sequence_streaming(
    double *rf_real_hz, double *rf_imag_hz,
    double *gx_hz_m, double *gy_hz_m, double *gz_hz_m,
    double *dt_s, int nintervals,
    double *t1_s, double *t2_s, double *df_hz,
    double *x_m, double *y_m, double *z_m, double *pd,
    double *tx_real, double *tx_imag,
    double *rx_real, double *rx_imag, int ncoils,
    double *mx_init, double *my_init, double *mz_init, int nspins,
    int *adc_state_indices, double *adc_demod_real, double *adc_demod_imag,
    int nadc, int *checkpoint_state_indices, int ncheckpoints,
    double *signal_real, double *signal_imag,
    double *mx_final, double *my_final, double *mz_final,
    double *mx_checkpoints, double *my_checkpoints, double *mz_checkpoints,
    int num_threads);

/* Optimized streaming kernel with the same numerical model and output layout. */
int blochsim_sequence_streaming_optimized(
    double *rf_real_hz, double *rf_imag_hz,
    double *gx_hz_m, double *gy_hz_m, double *gz_hz_m,
    double *dt_s, int nintervals,
    double *t1_s, double *t2_s, double *df_hz,
    double *x_m, double *y_m, double *z_m, double *pd,
    double *tx_real, double *tx_imag,
    double *rx_real, double *rx_imag, int ncoils,
    double *mx_init, double *my_init, double *mz_init, int nspins,
    int *adc_state_indices, double *adc_demod_real, double *adc_demod_imag,
    int nadc, int *checkpoint_state_indices, int ncheckpoints,
    double *signal_real, double *signal_imag,
    double *mx_final, double *my_final, double *mz_final,
    double *mx_checkpoints, double *my_checkpoints, double *mz_checkpoints,
    int num_threads);

#endif /* BLOCH_CORE_H */
