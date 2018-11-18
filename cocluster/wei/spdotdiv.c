/*
 * spdotdiv.c
 * Compute the elementwise division of two sparse matrix
 * The calling syntax:
 *       [o] = spdotdiv(a, b)
 *
 * This is a MEX-file for MATLAB.
 *
 * Copyright 2008 Wei Tang (wtang@cs.utexas.edu)
 * $Id: spdotdiv.c,v 1.1.1.1 2008/04/18 00:52:19 wtang Exp $
 */

#include "mex.h"

/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    const mxArray* a = prhs[0];
    const mxArray* b = prhs[1];
    
    if (!mxIsSparse(a) || !mxIsSparse(b)) {
		mexErrMsgTxt("Input matrix should be sparse matrix!\n");
	}
	
	if (mxIsComplex(a) || mxIsComplex(b)) {
		mexErrMsgTxt("Input matrix should be real matrix!\n");
	}

	size_t M = mxGetM(a), N = mxGetN(a);
	if (M != mxGetM(b) || N != mxGetN(b)) {
		mexErrMsgTxt("Input matrices should have the same size!\n");
	}

	mwSize Nz = mxGetNzmax(a);
    mwIndex *aIr = mxGetIr(a);
    mwIndex *aJc = mxGetJc(a);
    double *aPr = mxGetPr(a);

    mwIndex *bIr = mxGetIr(b);
    mwIndex *bJc = mxGetJc(b);
    double *bPr = mxGetPr(b);
	
    plhs[0] = mxCreateSparse(M, N, Nz, mxREAL);
    
    mwIndex* oIr = mxGetIr(plhs[0]);
    mwIndex* oJc = mxGetJc(plhs[0]);
    double* oPr = mxGetPr(plhs[0]);

    mwIndex i, j, k;
    for (i = 0; i < N+1; i++) {
        oJc[i] = (mwIndex) aJc[i];
    }
    for (i = 0; i < Nz; i++) {
        oIr[i] = (mwIndex) aIr[i];
    }
	
	for (i = 0; i < N; i++) {
		j = aJc[i]; k = bJc[i];
		while (j < aJc[i+1] && k < bJc[i+1]) {
			while (j < aJc[i+1] && aIr[j] < bIr[k]) j++;
			while (k < bJc[i+1] && bIr[k] < aIr[j]) k++;
			if (aIr[j] == bIr[k]) {
				oPr[j] = aPr[j] / bPr[k];
				j++;
				k++;
			}
		}
	}	
}
