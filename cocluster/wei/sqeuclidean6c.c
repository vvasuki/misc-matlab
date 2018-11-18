/*
 * sqeuclidean6c.c
 *
 * Compute the squared Euclidean distance for base 6 in
 * Bregman co-clustering algorithm.
 *
 * The calling syntax is:
 *		D = sqeuclidean6c(X, Y, I, C, W)
 * 
 * This is a MEX-file for MATLAB.
 *
 * Copyright 2008 Wei Tang (wtang@cs.utexas.edu)
 * $Id: sqeuclidean6c.c,v 1.1.1.1 2008/04/18 00:52:19 wtang Exp $
 */

#include "mex.h"

/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
	/* declaration of variables */
	mwSize n, p, nclusts;
	mwIndex *xIr, *xJc, *yIr, *yJc, *wIr, *wJc, i, j, k;
	double *X, *Y, *I, *C, *W, *D, *wgt, *dat;

	/* check for proper number of arguments */
	if (nrhs != 5) {
		mexErrMsgTxt("Five inputs required.");
	}
	if (nlhs > 1) {
		mexErrMsgTxt("Too many output arguments.");
	}
    if (mxGetM(prhs[4]) != mxGetM(prhs[0])
        || mxGetN(prhs[4]) != mxGetN(prhs[0])
        || mxGetN(prhs[0]) != mxGetN(prhs[3])) {
        mexErrMsgTxt("Dimensions of the weight and input matrices do not match.");
    }
	if (mxIsSparse(prhs[3])) {
		mexErrMsgTxt("The fourth input should be a full matrix.");
	}
	
	/* read the input matrix */
	n = mxGetM(prhs[0]);
	p = mxGetN(prhs[0]);
    nclusts = mxGetM(prhs[3]);

    xIr = mxGetIr(prhs[0]);
    xJc = mxGetJc(prhs[0]);
    X = mxGetPr(prhs[0]);

    yIr = mxGetIr(prhs[1]);
    yJc = mxGetJc(prhs[1]);
    Y = mxGetPr(prhs[1]);

    I = mxGetPr(prhs[2]);
    C = mxGetPr(prhs[3]);

    wIr = mxGetIr(prhs[4]);
    wJc = mxGetJc(prhs[4]);
    W = mxGetPr(prhs[4]);

	/* create the output matrix */
	plhs[0] = mxCreateDoubleMatrix(n, nclusts, mxREAL);
	D = mxGetPr(plhs[0]);
	
	/* squared Euclidean distance computation */
    wgt = calloc(n, sizeof(double));
	dat = calloc(n, sizeof(double));
    for (i = 0; i < nclusts; i++) {
        for (j = 0; j < n; j++) {
            wgt[j] = 0.0;
        }
        for (j = wJc[0]; j < wJc[1]; j++) {
            wgt[wIr[j]] = W[j];
        }
        for (j = xJc[0]; j < xJc[1]; j++) {
            D[xIr[j]+n*i] = X[j];
        }
        for (j = yJc[(mwIndex)I[0]-1]; j < yJc[(mwIndex)I[0]]; j++) {
            D[yIr[j]+n*i] -= Y[j];
        }
        for (j = 0; j < n; j++) {
            D[j+n*i] = wgt[j] * (D[j+n*i]-C[i]) * (D[j+n*i]-C[i]);
        }
        for (k = 1; k < p; k++) {
            for (j = 0; j < n; j++) {
                wgt[j] = dat[j] = 0.0;
            }
            for (j = wJc[k]; j < wJc[k+1]; j++) {
                wgt[wIr[j]] = W[j];
            }
            for (j = xJc[k]; j < xJc[k+1]; j++) {
                dat[xIr[j]] = X[j];
            }
            for (j = yJc[(mwIndex)I[k]-1]; j < yJc[(mwIndex)I[k]]; j++) {
                dat[yIr[j]] -= Y[j];
            }
            for (j = 0; j < n; j++) {
                D[j+n*i] += wgt[j] * (dat[j]-C[i+k*nclusts])
                    * (dat[j]-C[i+k*nclusts]);
            }
        }
    }
	free(wgt);
	free(dat);
}
