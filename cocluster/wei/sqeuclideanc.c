/*
 * sqeuclideanc.c
 *
 * Compute the squared Euclidean distance for base 1-5 in
 * Bregman co-clustering algorithm.
 *
 * The calling syntax is:
 *		D = sqeuclideanc(X, C, W)
 * 
 * This is a MEX-file for MATLAB.
 *
 * Copyright 2008 Wei Tang (wtang@cs.utexas.edu)
 * $Id: sqeuclideanc.c,v 1.2 2008/05/02 21:03:33 wtang Exp $
 */

#include "mex.h"

/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
	/* declaration of variables */
	mwSize n, p, nclusts;
	mwIndex i, j, k;
	double *X, *C, *W, *D;

	/* check for proper number of arguments */
	if (nrhs != 3) {
		mexErrMsgTxt("Three inputs required.");
	}
	if (nlhs > 1) {
		mexErrMsgTxt("Too many output arguments.");
	}
    if (mxGetM(prhs[0]) != mxGetM(prhs[2])
		|| mxGetN(prhs[0]) != mxGetN(prhs[2])
        || mxGetN(prhs[1]) != mxGetN(prhs[0])) {
        mexErrMsgTxt("Dimensions of the weight and input matrices do not match.");
    }
	if (mxIsSparse(prhs[0]) || mxIsSparse(prhs[1]) || mxIsSparse(prhs[2])) {
		mexErrMsgTxt("All the inputs should be full matrix.");
	}
	
	/* read the input matrix */
	n = mxGetM(prhs[0]);
	p = mxGetN(prhs[0]);
    nclusts = mxGetM(prhs[1]);

    X = mxGetPr(prhs[0]);
    C = mxGetPr(prhs[1]);
    W = mxGetPr(prhs[2]);

	/* create the output matrix */
	plhs[0] = mxCreateDoubleMatrix(n, nclusts, mxREAL);
	D = mxGetPr(plhs[0]);
	
	/* squared Euclidean distance computation */
    for (i = 0; i < nclusts; i++) {
        for (k = 0; k < n; k++) {
            D[k+n*i] = W[k] * (X[k]-C[i]) * (X[k]-C[i]);
        }
        for (j = 1; j < p; j++) {
            for (k = 0; k < n; k++) {
                D[k+n*i] += W[k+j*n] * (X[k+j*n]-C[i+j*nclusts])
                    * (X[k+j*n]-C[i+j*nclusts]);
            }
        }
    }
}
