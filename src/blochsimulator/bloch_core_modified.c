/* bloch_core_modified.c - Modified Bloch simulator for Python integration
 * Based on original code by Brian Hargreaves, Stanford University
 * Modified to remove MATLAB dependencies and add parallelization
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "bloch_core.h"

#ifdef _OPENMP
#include <omp.h>
#endif



void multmatvec(double *mat, double *vec, double *matvec)

	/* Multiply 3x3 matrix by 3x1 vector. */

{
*matvec++ = mat[0]*vec[0] + mat[3]*vec[1] + mat[6]*vec[2];
*matvec++ = mat[1]*vec[0] + mat[4]*vec[1] + mat[7]*vec[2];
*matvec++ = mat[2]*vec[0] + mat[5]*vec[1] + mat[8]*vec[2];
}



void addvecs(double *vec1, double *vec2, double *vecsum)

	/* Add two 3x1 Vectors */

{
*vecsum++ = *vec1++ + *vec2++;
*vecsum++ = *vec1++ + *vec2++;
*vecsum++ = *vec1++ + *vec2++;
}




void adjmat(double *mat, double *adj)

/* ======== Adjoint of a 3x3 matrix ========= */

{
*adj++ = (mat[4]*mat[8]-mat[7]*mat[5]);
*adj++ =-(mat[1]*mat[8]-mat[7]*mat[2]);
*adj++ = (mat[1]*mat[5]-mat[4]*mat[2]);
*adj++ =-(mat[3]*mat[8]-mat[6]*mat[5]);
*adj++ = (mat[0]*mat[8]-mat[6]*mat[2]);
*adj++ =-(mat[0]*mat[5]-mat[3]*mat[2]);
*adj++ = (mat[3]*mat[7]-mat[6]*mat[4]);
*adj++ =-(mat[0]*mat[7]-mat[6]*mat[1]);
*adj++ = (mat[0]*mat[4]-mat[3]*mat[1]);
}


void zeromat(double *mat)

/* ====== Set a 3x3 matrix to all zeros	======= */

{
*mat++=0;
*mat++=0;
*mat++=0;
*mat++=0;
*mat++=0;
*mat++=0;
*mat++=0;
*mat++=0;
*mat++=0;
}


void eyemat(double *mat)

/* ======== Return 3x3 Identity Matrix  ========= */

{
zeromat(mat);
mat[0]=1;
mat[4]=1;
mat[8]=1;

}

double detmat(double *mat)

/* ======== Determinant of a 3x3 matrix ======== */

{
double det;

det = mat[0]*mat[4]*mat[8];
det+= mat[3]*mat[7]*mat[2];
det+= mat[6]*mat[1]*mat[5];
det-= mat[0]*mat[7]*mat[5];
det-= mat[3]*mat[1]*mat[8];
det-= mat[6]*mat[4]*mat[2];

return det;
}


void scalemat(double *mat, double scalar)

/* ======== multiply a matrix by a scalar ========= */

{
*mat++ *= scalar;
*mat++ *= scalar;
*mat++ *= scalar;
*mat++ *= scalar;
*mat++ *= scalar;
*mat++ *= scalar;
*mat++ *= scalar;
*mat++ *= scalar;
*mat++ *= scalar;
}


void invmat(double *mat, double *imat)

/* ======== Inverse of a 3x3 matrix ========= */
/*	DO NOT MAKE THE OUTPUT THE SAME AS ONE OF THE INPUTS!! */

{
double det;
int count;

det = detmat(mat);	/* Determinant */
adjmat(mat, imat);	/* Adjoint */

for (count=0; count<9; count++) {
	*imat /= det;
	imat++;
}
}


void addmats(double *mat1, double *mat2, double *matsum)

/* ====== Add two 3x3 matrices.	====== */

{
*matsum++ = *mat1++ + *mat2++;
*matsum++ = *mat1++ + *mat2++;
*matsum++ = *mat1++ + *mat2++;
*matsum++ = *mat1++ + *mat2++;
*matsum++ = *mat1++ + *mat2++;
*matsum++ = *mat1++ + *mat2++;
*matsum++ = *mat1++ + *mat2++;
*matsum++ = *mat1++ + *mat2++;
*matsum++ = *mat1++ + *mat2++;
}


void multmats(double *mat1, double *mat2, double *matproduct)

/* ======= Multiply two 3x3 matrices. ====== */
/*	DO NOT MAKE THE OUTPUT THE SAME AS ONE OF THE INPUTS!! */

{
*matproduct++ = mat1[0]*mat2[0] + mat1[3]*mat2[1] + mat1[6]*mat2[2];
*matproduct++ = mat1[1]*mat2[0] + mat1[4]*mat2[1] + mat1[7]*mat2[2];
*matproduct++ = mat1[2]*mat2[0] + mat1[5]*mat2[1] + mat1[8]*mat2[2];
*matproduct++ = mat1[0]*mat2[3] + mat1[3]*mat2[4] + mat1[6]*mat2[5];
*matproduct++ = mat1[1]*mat2[3] + mat1[4]*mat2[4] + mat1[7]*mat2[5];
*matproduct++ = mat1[2]*mat2[3] + mat1[5]*mat2[4] + mat1[8]*mat2[5];
*matproduct++ = mat1[0]*mat2[6] + mat1[3]*mat2[7] + mat1[6]*mat2[8];
*matproduct++ = mat1[1]*mat2[6] + mat1[4]*mat2[7] + mat1[7]*mat2[8];
*matproduct++ = mat1[2]*mat2[6] + mat1[5]*mat2[7] + mat1[8]*mat2[8];
}


void calcrotmat(double nx, double ny, double nz, double *rmat)

	/* Find the rotation matrix that rotates |n| radians about
		the vector given by nx,ny,nz				*/
{
double ar, ai, br, bi, hp, cp, sp;
double arar, aiai, arai2, brbr, bibi, brbi2, arbi2, aibr2, arbr2, aibi2;
double phi;

phi = sqrt(nx*nx+ny*ny+nz*nz);

if (phi == 0.0)
	{
	*rmat++ = 1;
	*rmat++	= 0;
	*rmat++ = 0;
	*rmat++ = 0;
	*rmat++ = 1;
	*rmat++	= 0;
	*rmat++ = 0;
	*rmat++ = 0;
	*rmat++ = 1;
	}

/*printf("calcrotmat(%6.3f,%6.3f,%6.3f) -> phi = %6.3f\n",nx,ny,nz,phi);*/

else
	{
	/* First define Cayley-Klein parameters 	*/
	hp = phi/2;
	cp = cos(hp);
	sp = sin(hp)/phi;	/* /phi because n is unit length in defs. */
	ar = cp;
	ai = -nz*sp;
	br = ny*sp;
	bi = -nx*sp;

 	/* Make auxiliary variables to speed this up	*/

	arar = ar*ar;
	aiai = ai*ai;
	arai2 = 2*ar*ai;
	brbr = br*br;
	bibi = bi*bi;
	brbi2 = 2*br*bi;
	arbi2 = 2*ar*bi;
	aibr2 = 2*ai*br;
	arbr2 = 2*ar*br;
	aibi2 = 2*ai*bi;


	/* Make rotation matrix.	*/

	*rmat++ = arar-aiai-brbr+bibi;
	*rmat++ = -arai2-brbi2;
	*rmat++ = -arbr2+aibi2;
	*rmat++ =  arai2-brbi2;
	*rmat++ = arar-aiai+brbr-bibi;
	*rmat++ = -aibr2-arbi2;
	*rmat++ =  arbr2+aibi2;
	*rmat++ =  arbi2-aibr2;
	*rmat++ = arar+aiai-brbr-bibi;
	}
}



void zerovec(double *vec)

/*	Set a 3x1 vector to all zeros	*/

{
*vec++=0;
*vec++=0;
*vec++=0;
}


int times2intervals( double *endtimes, double *intervals, long n)
/* ------------------------------------------------------------
	Function takes the given endtimes of intervals, and
	returns the interval lengths in an array, assuming that
	the first interval starts at 0.

	If the intervals are all greater than 0, then this
	returns 1, otherwise it returns 0.
   ------------------------------------------------------------ */

{
int allpos;
int count;
double lasttime;

allpos=1;
lasttime = 0.0;

for (count = 0; count < n; count++)
	{
	intervals[count] = endtimes[count]-lasttime;
	lasttime = endtimes[count];
	if (intervals[count] <= 0)
		allpos =0;
	}

return (allpos);
}






int blochsim(double *b1real, double *b1imag,
		double *xgrad, double *ygrad, double *zgrad, double *tsteps,
		int ntime, double *e1, double *e2, double df,
		double dx, double dy, double dz,
		double *mx, double *my, double *mz, int mode)

	/* Go through time for one df and one dx,dy,dz.		*/

{
int tcount;
double gammadx;
double gammady;
double gammadz;
double rotmat[9];
double amat[9], bvec[3];	/* A and B propagation matrix and vector */
double arot[9], brot[3];	/* A and B after rotation step. */
double decmat[9];		/* Decay matrix for each time step. */
double decvec[3];		/* Recovery vector for each time step. */
double rotx,roty,rotz;		/* Rotation axis coordinates. */
double imat[9], mvec[3];
double mcurr0[3];		/* Current magnetization before rotation. */
double mcurr1[3];		/* Current magnetization before decay. */

eyemat(amat); 		/* A is the identity matrix.	*/
eyemat(imat); 		/* I is the identity matrix.	*/

zerovec(bvec);
zerovec(decvec);
zeromat(decmat);

gammadx = dx*GAMMA;	/* Convert to Hz/cm */
gammady = dy*GAMMA;	/* Convert to Hz/cm */
gammadz = dz*GAMMA;	/* Convert to Hz/cm */


mcurr0[0] = *mx;		/* Set starting x magnetization */
mcurr0[1] = *my;		/* Set starting y magnetization */
mcurr0[2] = *mz;		/* Set starting z magnetization */


for (tcount = 0; tcount < ntime; tcount++)
	{
		/*	Rotation 	*/

	rotz = -(*xgrad++ * gammadx + *ygrad++ * gammady + *zgrad++ * gammadz +
								df*TWOPI ) * *tsteps;
	rotx = (- *b1real++ * GAMMA * *tsteps);
	roty = (+ *b1imag++ * GAMMA * *tsteps++);
	calcrotmat(rotx, roty, rotz, rotmat);

	if (mode == 1)
		{
		multmats(rotmat,amat,arot);
		multmatvec(rotmat,bvec,brot);
		}
	else
		multmatvec(rotmat,mcurr0,mcurr1);


		/* 	Decay	*/

	decvec[2]= 1- *e1;
	decmat[0]= *e2;
	decmat[4]= *e2++;
	decmat[8]= *e1++;

	if (mode == 1)
		{
		multmats(decmat,arot,amat);
		multmatvec(decmat,brot,bvec);
		addvecs(bvec,decvec,bvec);
		}
	else
		{
		multmatvec(decmat,mcurr1,mcurr0);
		addvecs(mcurr0,decvec,mcurr0);
		}

	/*
	printf("rotmat = [%6.3f  %6.3f  %6.3f ] \n",rotmat[0],rotmat[3],
	  			rotmat[6]);
	printf("         [%6.3f  %6.3f  %6.3f ] \n",rotmat[1],rotmat[4],
				rotmat[7]);
	printf("         [%6.3f  %6.3f  %6.3f ] \n",rotmat[2],rotmat[5],
				rotmat[8]);
	printf("A = [%6.3f  %6.3f  %6.3f ] \n",amat[0],amat[3],amat[6]);
	printf("    [%6.3f  %6.3f  %6.3f ] \n",amat[1],amat[4],amat[7]);
	printf("    [%6.3f  %6.3f  %6.3f ] \n",amat[2],amat[5],amat[8]);
	printf(" B = <%6.3f,%6.3f,%6.3f> \n",bvec[0],bvec[1],bvec[2]);
	printf("<mx,my,mz> = <%6.3f,%6.3f,%6.3f> \n",
		amat[6] + bvec[0], amat[7] + bvec[1], amat[8] + bvec[2]);

	printf("\n");
	*/

	if (mode == 2)		/* Sample output at times.  */
					/* Only do this if transient! */
		{
		*mx = mcurr0[0];
		*my = mcurr0[1];
		*mz = mcurr0[2];

		mx++;
		my++;
		mz++;
		}
	}



	/* If only recording the endpoint, either store the last
		point, or calculate the steady-state endpoint. */

if (mode==0)		/* Indicates start at given m, or m0. */
	{
	*mx = mcurr0[0];
	*my = mcurr0[1];
	*mz = mcurr0[2];
	}

else if (mode==1)	/* Indicates to find steady-state magnetization */
	{
	scalemat(amat,-1.0);		/* Negate A matrix 	*/
	addmats(amat,imat,amat);	/* Now amat = (I-A)		*/
	invmat(amat,imat);		/* Reuse imat as inv(I-A) 	*/
	multmatvec(imat,bvec,mvec);	/* Now M = inv(I-A)*B		*/
	*mx = mvec[0];
	*my = mvec[1];
	*mz = mvec[2];
	}

return 0;
}



int blochsimfz(double *b1real, double *b1imag, double *xgrad, double *ygrad, double *zgrad,
		double *tsteps,
		int ntime, double t1, double t2, double *dfreq, int nfreq,
		double *dxpos, double *dypos, double *dzpos, int npos,
		double *mx, double *my, double *mz, int mode)


{
int count;
int poscount;
int fcount;
int totpoints;
int totcount = 0;

int ntout;

double *e1;
double *e2;
double *e1ptr;
double *e2ptr;
double *tstepsptr;
double *dxptr, *dyptr, *dzptr;


if (mode & 2)
	ntout = ntime;
else
	ntout = 1;

	/* First calculate the E1 and E2 values at each time step. */

e1 = (double *) malloc(ntime * sizeof(double));
e2 = (double *) malloc(ntime * sizeof(double));

e1ptr = e1;
e2ptr = e2;
tstepsptr = tsteps;

for (count=0; count < ntime; count++)
	{
	*e1ptr++ = exp(- *tstepsptr / t1);
	*e2ptr++ = exp(- *tstepsptr++ / t2);
	}

totpoints = npos*nfreq;

for (fcount=0; fcount < nfreq; fcount++)
    {
    dxptr = dxpos;
    dyptr = dypos;
    dzptr = dzpos;
    for (poscount=0; poscount < npos; poscount++)

	{

	if (mode == 3)	/* Steady state AND record all time points. */

		{	/* First go through and find steady state, then
				repeat as if transient starting at steady st.*/

		blochsim(b1real, b1imag, xgrad, ygrad, zgrad, tsteps, ntime,
			e1, e2, *dfreq, *dxptr, *dyptr,
			*dzptr, mx, my, mz, 1);

		blochsim(b1real, b1imag, xgrad, ygrad, zgrad, tsteps, ntime,
			e1, e2, *dfreq, *dxptr++, *dyptr++,
			*dzptr++, mx, my, mz, 2);
		}
	else
		{
		blochsim(b1real, b1imag, xgrad, ygrad, zgrad, tsteps, ntime,
			e1, e2, *dfreq, *dxptr++, *dyptr++,
			*dzptr++, mx, my, mz, mode);
		}

	mx += ntout;
	my += ntout;
	mz += ntout;

	totcount++;
	if ((totpoints > 40000) && ( ((10*totcount)/totpoints)> (10*(totcount-1)/totpoints) ))
		printf("%d%% Complete.\n",(100*totcount/totpoints));
	}
    dfreq++;
    }

free(e1);
free(e2);

return 0;
}


/* ======== Python Interface Functions ======== */

void calculate_relaxation(double t1, double t2, double dt, double *e1, double *e2)
{
    /* Calculate relaxation parameters for a given time step */
    if (t1 > 0)
        *e1 = exp(-dt / t1);
    else
        *e1 = 0.0;

    if (t2 > 0)
        *e2 = exp(-dt / t2);
    else
        *e2 = 0.0;
}

void set_equilibrium_magnetization(double *mx, double *my, double *mz, int n)
{
    /* Set magnetization to equilibrium state [0, 0, 1] */
    int i;
    for (i = 0; i < n; i++) {
        mx[i] = 0.0;
        my[i] = 0.0;
        mz[i] = 1.0;
    }
}

/* Parallel batch processing function */
void blochsim_batch(double *b1real, double *b1imag,
                    double *xgrad, double *ygrad, double *zgrad, double *tsteps,
                    int ntime, double t1, double t2,
                    double *df, int nf,
                    double *dx, double *dy, double *dz, int npos,
                    double *mx, double *my, double *mz,
                    int mode, int num_threads)
{
    #ifdef _OPENMP
    omp_set_num_threads(num_threads);
    #endif

    /* Call the existing blochsimfz function */
    blochsimfz(b1real, b1imag, xgrad, ygrad, zgrad, tsteps, ntime,
               t1, t2, df, nf, dx, dy, dz, npos, mx, my, mz, mode);
}

/* Optimized batch processing with shared relaxation arrays and collapsed loops */
void blochsim_batch_optimized(double *b1real, double *b1imag,
                              double *xgrad, double *ygrad, double *zgrad, double *tsteps,
                              int ntime, double t1, double t2,
                              double *df, int nf,
                              double *dx, double *dy, double *dz, int npos,
                              double *mx, double *my, double *mz,
                              int mode, int num_threads)
{
    int ntout = (mode & 2) ? ntime : 1;
    double *e1 = (double *) malloc(ntime * sizeof(double));
    double *e2 = (double *) malloc(ntime * sizeof(double));
    if (e1 == NULL || e2 == NULL) {
        if (e1) free(e1);
        if (e2) free(e2);
        return;
    }

    for (int t = 0; t < ntime; t++) {
        e1[t] = exp(-tsteps[t] / t1);
        e2[t] = exp(-tsteps[t] / t2);
    }

    int total_points = nf * npos;
    int i;

    #ifdef _OPENMP
    omp_set_num_threads(num_threads);
    #pragma omp parallel for schedule(static)
    #endif
    for (i = 0; i < total_points; i++) {
        int f = i / npos;
        int p = i % npos;

        int base = (f * npos + p) * ntout;
        double *mx_ptr = mx + base;
        double *my_ptr = my + base;
        double *mz_ptr = mz + base;

        if (mode == 3) {
            blochsim(b1real, b1imag, xgrad, ygrad, zgrad, tsteps, ntime,
                     e1, e2, df[f], dx[p], dy[p], dz[p],
                     mx_ptr, my_ptr, mz_ptr, 1);
            blochsim(b1real, b1imag, xgrad, ygrad, zgrad, tsteps, ntime,
                     e1, e2, df[f], dx[p], dy[p], dz[p],
                     mx_ptr, my_ptr, mz_ptr, 2);
        } else {
            blochsim(b1real, b1imag, xgrad, ygrad, zgrad, tsteps, ntime,
                     e1, e2, df[f], dx[p], dy[p], dz[p],
                     mx_ptr, my_ptr, mz_ptr, mode);
        }
    }

    free(e1);
    free(e2);
}
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
    int mode, int num_threads)
{
    int ntout = (mode & 2) ? ntime : 1;
    int v;

    #ifdef _OPENMP
    omp_set_num_threads(num_threads);
    #pragma omp parallel for schedule(dynamic, 64)
    #endif
    for (v = 0; v < nvoxels; v++) {
        /* Get per-voxel parameters */
        double t1 = t1_arr[v];
        double t2 = t2_arr[v];
        double df = df_arr[v];
        double pos_x = dx[v];
        double pos_y = dy[v];
        double pos_z = dz[v];

        /* Output pointer base for this voxel */
        int base = v * ntout;

        /* Skip if T1 or T2 are zero (background/masked voxels) */
        if (t1 <= 0.0 || t2 <= 0.0) {
            for (int t = 0; t < ntout; t++) {
                mx[base + t] = 0.0;
                my[base + t] = 0.0;
                mz[base + t] = 0.0;
            }
            continue;
        }

        /* Compute E1, E2 arrays for this voxel's T1/T2 */
        double *e1 = (double *)malloc(ntime * sizeof(double));
        double *e2 = (double *)malloc(ntime * sizeof(double));
        if (e1 == NULL || e2 == NULL) {
            if (e1) free(e1);
            if (e2) free(e2);
            continue;
        }

        for (int t = 0; t < ntime; t++) {
            e1[t] = exp(-tsteps[t] / t1);
            e2[t] = exp(-tsteps[t] / t2);
        }

        /* Get initial magnetization */
        double mx_start = mx_init ? mx_init[v] : 0.0;
        double my_start = my_init ? my_init[v] : 0.0;
        double mz_start = mz_init ? mz_init[v] : 1.0;

        /* Set initial values */
        mx[base] = mx_start;
        my[base] = my_start;
        mz[base] = mz_start;

        /* Run Bloch simulation for this voxel */
        blochsim(b1real, b1imag, gx, gy, gz, tsteps, ntime,
                 e1, e2, df, pos_x, pos_y, pos_z,
                 &mx[base], &my[base], &mz[base], mode);

        free(e1);
        free(e2);
    }
}

/* Optimized version for grouped tissue types */
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
    int mode, int num_threads)
{
    int ntout = (mode & 2) ? ntime : 1;

    /* Pre-compute E1/E2 for each tissue label - shared across all voxels with that label */
    double **e1_table = (double **)malloc(nlabels * sizeof(double *));
    double **e2_table = (double **)malloc(nlabels * sizeof(double *));

    if (e1_table == NULL || e2_table == NULL) {
        if (e1_table) free(e1_table);
        if (e2_table) free(e2_table);
        return;
    }

    for (int l = 0; l < nlabels; l++) {
        e1_table[l] = (double *)malloc(ntime * sizeof(double));
        e2_table[l] = (double *)malloc(ntime * sizeof(double));

        if (e1_table[l] == NULL || e2_table[l] == NULL) {
            /* Cleanup on allocation failure */
            for (int j = 0; j <= l; j++) {
                if (e1_table[j]) free(e1_table[j]);
                if (e2_table[j]) free(e2_table[j]);
            }
            free(e1_table);
            free(e2_table);
            return;
        }

        double t1 = t1_per_label[l];
        double t2 = t2_per_label[l];

        for (int t = 0; t < ntime; t++) {
            e1_table[l][t] = (t1 > 0) ? exp(-tsteps[t] / t1) : 0.0;
            e2_table[l][t] = (t2 > 0) ? exp(-tsteps[t] / t2) : 0.0;
        }
    }

    int v;

    #ifdef _OPENMP
    omp_set_num_threads(num_threads);
    #pragma omp parallel for schedule(dynamic, 64)
    #endif
    for (v = 0; v < nvoxels; v++) {
        int label = tissue_labels[v];
        int base = v * ntout;

        /* Skip background (label < 0) or invalid labels */
        if (label < 0 || label >= nlabels) {
            for (int t = 0; t < ntout; t++) {
                mx[base + t] = 0.0;
                my[base + t] = 0.0;
                mz[base + t] = 0.0;
            }
            continue;
        }

        /* Check if this tissue has valid T1/T2 */
        double t1 = t1_per_label[label];
        double t2 = t2_per_label[label];
        if (t1 <= 0.0 || t2 <= 0.0) {
            for (int t = 0; t < ntout; t++) {
                mx[base + t] = 0.0;
                my[base + t] = 0.0;
                mz[base + t] = 0.0;
            }
            continue;
        }

        double *e1 = e1_table[label];
        double *e2 = e2_table[label];
        double df = df_arr[v];

        /* Set initial magnetization */
        mx[base] = mx_init ? mx_init[v] : 0.0;
        my[base] = my_init ? my_init[v] : 0.0;
        mz[base] = mz_init ? mz_init[v] : 1.0;

        blochsim(b1real, b1imag, gx, gy, gz, tsteps, ntime,
                 e1, e2, df, dx[v], dy[v], dz[v],
                 &mx[base], &my[base], &mz[base], mode);
    }

    /* Cleanup */
    for (int l = 0; l < nlabels; l++) {
        free(e1_table[l]);
        free(e2_table[l]);
    }
    free(e1_table);
    free(e2_table);
}

/*
 * Streaming sequence simulation.
 *
 * Unlike the legacy transient mode this routine never allocates or returns a
 * time-by-spin magnetization array. Each spin is propagated through all
 * intervals and contributes only at requested ADC state boundaries. Checkpoint
 * arrays use checkpoint-major layout: checkpoint * nspins + spin.
 */
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
    int num_threads)
{
    int thread_count = num_threads > 0 ? num_threads : 1;
#ifndef _OPENMP
    thread_count = 1;
#endif
    size_t samples_per_thread = (size_t)ncoils * (size_t)(nadc > 0 ? nadc : 1);
    size_t signal_count = (size_t)thread_count * samples_per_thread;
    double *thread_signal_real = (double *)calloc(signal_count, sizeof(double));
    double *thread_signal_imag = (double *)calloc(signal_count, sizeof(double));
    if (thread_signal_real == NULL || thread_signal_imag == NULL) {
        if (thread_signal_real) free(thread_signal_real);
        if (thread_signal_imag) free(thread_signal_imag);
        return -1;
    }

    for (int sample = 0; sample < ncoils * nadc; sample++) {
        signal_real[sample] = 0.0;
        signal_imag[sample] = 0.0;
    }

#ifdef _OPENMP
    omp_set_num_threads(thread_count);
#pragma omp parallel for schedule(static)
#endif
    for (int spin = 0; spin < nspins; spin++) {
        int thread_id = 0;
#ifdef _OPENMP
        thread_id = omp_get_thread_num();
#endif
        double *local_real = thread_signal_real + (size_t)thread_id * samples_per_thread;
        double *local_imag = thread_signal_imag + (size_t)thread_id * samples_per_thread;
        double magnetization[3] = {mx_init[spin], my_init[spin], mz_init[spin]};
        int adc_cursor = 0;
        int checkpoint_cursor = 0;

        while (adc_cursor < nadc && adc_state_indices[adc_cursor] == 0) {
            double demod_real = adc_demod_real[adc_cursor];
            double demod_imag = adc_demod_imag[adc_cursor];
            for (int coil = 0; coil < ncoils; coil++) {
                size_t rx_offset = (size_t)coil * (size_t)nspins + (size_t)spin;
                size_t signal_offset = (size_t)coil * (size_t)(nadc > 0 ? nadc : 1) + (size_t)adc_cursor;
                double weighted_x = pd[spin] * (magnetization[0] * rx_real[rx_offset]
                                                 - magnetization[1] * rx_imag[rx_offset]);
                double weighted_y = pd[spin] * (magnetization[0] * rx_imag[rx_offset]
                                                 + magnetization[1] * rx_real[rx_offset]);
                local_real[signal_offset] += weighted_x * demod_real - weighted_y * demod_imag;
                local_imag[signal_offset] += weighted_x * demod_imag + weighted_y * demod_real;
            }
            adc_cursor++;
        }
        while (checkpoint_cursor < ncheckpoints && checkpoint_state_indices[checkpoint_cursor] == 0) {
            size_t offset = (size_t)checkpoint_cursor * (size_t)nspins + (size_t)spin;
            mx_checkpoints[offset] = magnetization[0];
            my_checkpoints[offset] = magnetization[1];
            mz_checkpoints[offset] = magnetization[2];
            checkpoint_cursor++;
        }

        for (int interval = 0; interval < nintervals; interval++) {
            double dt = dt_s[interval];
            double rotation[9];
            double rotated[3];
            double effective_rf_real = rf_real_hz[interval] * tx_real[spin]
                                     - rf_imag_hz[interval] * tx_imag[spin];
            double effective_rf_imag = rf_real_hz[interval] * tx_imag[spin]
                                     + rf_imag_hz[interval] * tx_real[spin];
            double rotx = -TWOPI * effective_rf_real * dt;
            double roty = +TWOPI * effective_rf_imag * dt;
            double frequency = gx_hz_m[interval] * x_m[spin]
                             + gy_hz_m[interval] * y_m[spin]
                             + gz_hz_m[interval] * z_m[spin]
                             + df_hz[spin];
            double rotz = -TWOPI * frequency * dt;
            calcrotmat(rotx, roty, rotz, rotation);
            multmatvec(rotation, magnetization, rotated);

            double e1 = exp(-dt / t1_s[spin]);
            double e2 = exp(-dt / t2_s[spin]);
            magnetization[0] = rotated[0] * e2;
            magnetization[1] = rotated[1] * e2;
            magnetization[2] = rotated[2] * e1 + (1.0 - e1);

            int state_index = interval + 1;
            while (adc_cursor < nadc && adc_state_indices[adc_cursor] == state_index) {
                double demod_real = adc_demod_real[adc_cursor];
                double demod_imag = adc_demod_imag[adc_cursor];
                for (int coil = 0; coil < ncoils; coil++) {
                    size_t rx_offset = (size_t)coil * (size_t)nspins + (size_t)spin;
                    size_t signal_offset = (size_t)coil * (size_t)(nadc > 0 ? nadc : 1) + (size_t)adc_cursor;
                    double weighted_x = pd[spin] * (magnetization[0] * rx_real[rx_offset]
                                                     - magnetization[1] * rx_imag[rx_offset]);
                    double weighted_y = pd[spin] * (magnetization[0] * rx_imag[rx_offset]
                                                     + magnetization[1] * rx_real[rx_offset]);
                    local_real[signal_offset] += weighted_x * demod_real - weighted_y * demod_imag;
                    local_imag[signal_offset] += weighted_x * demod_imag + weighted_y * demod_real;
                }
                adc_cursor++;
            }
            while (checkpoint_cursor < ncheckpoints && checkpoint_state_indices[checkpoint_cursor] == state_index) {
                size_t offset = (size_t)checkpoint_cursor * (size_t)nspins + (size_t)spin;
                mx_checkpoints[offset] = magnetization[0];
                my_checkpoints[offset] = magnetization[1];
                mz_checkpoints[offset] = magnetization[2];
                checkpoint_cursor++;
            }
        }

        mx_final[spin] = magnetization[0];
        my_final[spin] = magnetization[1];
        mz_final[spin] = magnetization[2];
    }

    for (int thread = 0; thread < thread_count; thread++) {
        size_t base = (size_t)thread * samples_per_thread;
        for (int coil = 0; coil < ncoils; coil++) {
            for (int sample = 0; sample < nadc; sample++) {
                size_t offset = (size_t)coil * (size_t)nadc + (size_t)sample;
                signal_real[offset] += thread_signal_real[base + offset];
                signal_imag[offset] += thread_signal_imag[base + offset];
            }
        }
    }

    free(thread_signal_real);
    free(thread_signal_imag);
    return 0;
}

/* End of file */
