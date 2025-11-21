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
#define TWOPI   6.283185   /* 2*pi */

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
double multmats(double *mat1, double *mat2, double *matproduct);
double calcrotmat(double nx, double ny, double nz, double *rmat);
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

#endif /* BLOCH_CORE_H */
