/* Given a sparse matrix, get the Ir, Jc, Pr arrays.
 *
 * The calling syntax is:
 *		[mat] = getsparse(m, n, Ir, Jc, Pr), where [m n] = size(mat)
 *
 * This is a MEX file for MATLAB.
 *
 * Copyright 2008 Wei Tang (wtang@cs.utexas.edu)
 * $Id: getsparse.c,v 1.1 2008/05/18 00:28:32 wtang Exp $
 */

#include "mex.h"

/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    if (!(nrhs == 5)) return;
    if (!mxIsDouble(prhs[2]) || 
        !mxIsDouble(prhs[3]) || 
        !mxIsDouble(prhs[4])) return;
    if (!(nlhs == 1)) return;

    mwSize M = (mwSize) mxGetScalar(prhs[0]);
    mwSize N = (mwSize) mxGetScalar(prhs[1]);
    double* dIr = mxGetPr(prhs[2]);
    double* dJc = mxGetPr(prhs[3]);
    double* Pr = mxGetPr(prhs[4]);

    mwSize Nz = (mwSize) dJc[N];

    plhs[0] = mxCreateSparse(M, N, Nz, mxREAL);
    
    mwIndex* oIr = mxGetIr(plhs[0]);
    mwIndex* oJc = mxGetJc(plhs[0]);
    double* oPr = mxGetPr(plhs[0]);

    mwIndex i, j;
    for(i = 0; i <= N; i++) {
        oJc[i] = (mwIndex) dJc[i];
    }
    for (j = 0; j < Nz; j++) {
        oIr[j] = (mwIndex) dIr[j];
        oPr[j] = Pr[j];
    }
}
