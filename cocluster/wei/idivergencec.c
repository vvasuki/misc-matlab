/*
 * idivergencec.c
 *
 * Compute the I divergence base 1-5 in Bregman co-clustering algorithm.
 *
 * The calling syntax is:
 *		D = idivergencec(X, C, V, W)
 * 
 * This is a MEX-file for MATLAB.
 *
 * Copyright 2008 Wei Tang (wtang@cs.utexas.edu)
 * $Id: idivergencec.c,v 1.1.1.1 2008/04/18 00:52:19 wtang Exp $
 */

#include "mex.h"
#include <math.h>

/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
	/* declaration of variables */
	mwSize n, p, nclusts;
	mwIndex *xIr, *xJc, *cIr, *cJc, *wIr, *wJc, i, j, k;
	double *X, *C, *V, *W, *D, *wgt, *dat, *con;

	/* check for proper number of arguments */
	if (nrhs != 4) {
		mexErrMsgTxt("Four inputs required.");
	}
	if (nlhs > 1) {
		mexErrMsgTxt("Too many output arguments.");
	}
    if (mxGetM(prhs[0]) != mxGetM(prhs[3])
		|| mxGetN(prhs[0]) != mxGetN(prhs[3])
		|| mxGetM(prhs[0]) != mxGetM(prhs[1])
        || mxGetN(prhs[0]) != mxGetN(prhs[1])
		|| mxGetN(prhs[0]) != mxGetN(prhs[2])) {
        mexErrMsgTxt("Dimensions of the weight and input matrices do not match.");
    }
	if (mxIsSparse(prhs[2])) {
		mexErrMsgTxt("The third input should be a full matrix.");
	}
	
	/* read the input matrix */
	n = mxGetM(prhs[0]);
	p = mxGetN(prhs[0]);
    nclusts = mxGetM(prhs[2]);

    xIr = mxGetIr(prhs[0]);
    xJc = mxGetJc(prhs[0]);
    X = mxGetPr(prhs[0]);
	
    cIr = mxGetIr(prhs[1]);
    cJc = mxGetJc(prhs[1]);
    C = mxGetPr(prhs[1]);
	
	V = mxGetPr(prhs[2]);
	
    wIr = mxGetIr(prhs[3]);
    wJc = mxGetJc(prhs[3]);
    W = mxGetPr(prhs[3]);

	/* create the output matrix */
	plhs[0] = mxCreateDoubleMatrix(n, nclusts, mxREAL);
	D = mxGetPr(plhs[0]);
	
	/* squared Euclidean distance computation */
    wgt = calloc(n, sizeof(double));
	dat = calloc(n, sizeof(double));
	con = calloc(n, sizeof(double));
    for (i = 0; i < nclusts; i++) {
        for (j = 0; j < n; j++) {
            wgt[j] = dat[j] = 0.0;
        }
        for (j = wJc[0]; j < wJc[1]; j++) {
            wgt[wIr[j]] = W[j];
        }
        for (j = cJc[0]; j < cJc[1]; j++) {
            D[cIr[j]+n*i] = C[j];
        }
		for (j = xJc[0]; j < xJc[1]; j++) {
			dat[xIr[j]] = X[j];
		}
        for (j = 0; j < n; j++) {
            D[j+n*i] = wgt[j]*(D[j+n*i]*V[i]-dat[j]*log(V[i]));
        }
        for (k = 1; k < p; k++) {
            for (j = 0; j < n; j++) {
                wgt[j] = dat[j] = con[j] = 0.0;
            }
            for (j = wJc[k]; j < wJc[k+1]; j++) {
                wgt[wIr[j]] = W[j];
            }
			for (j = cJc[k]; j < cJc[k+1]; j++) {
				con[cIr[j]] = C[j];
			}
            for (j = xJc[k]; j < xJc[k+1]; j++) {
                dat[xIr[j]] = X[j];
            }
            for (j = 0; j < n; j++) {
                D[j+n*i] += wgt[j]*(con[j]*V[i+k*nclusts]-dat[j]*log(V[i+k*nclusts]));
            }
        }
    }
	free(wgt);
	free(dat);
	free(con);
}
