# include <iostream>
# include <cmath>
# include "mex.h"
# include <ctime>

#define I(x,t,M) (((t)*(M)) + x)
using namespace std;

double flip(){
	double flipv = static_cast< double >( rand() ) / RAND_MAX;
	return(flipv);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	
	//START OF MAT INPUT
	if(nlhs != 1) {
		cout<<"Function needs to return 1 arguments\n";
		return;
	}

	if(nrhs != 5) {
		cout<<"Function takes 5 arguments\n";
		return;
	}	

	double* graph_temp = mxGetPr(prhs[0]);
	int p = (int)(mxGetScalar(prhs[1]));
	int n = (int)(mxGetScalar(prhs[2]));
	int burnin = (int)(mxGetScalar(prhs[3]));
	int sampint = (int)(mxGetScalar(prhs[4]));
	int** graph = new int*[p];
	for(int s = 0;s < p;s++)
		graph[s] = new int[p];

	for(int s = 0;s < p;s++)
		for(int t = 0;t < p;t++)
			graph[s][t] = (int)graph_temp[I(s,t,p)];

	cout<<"n,p,burnin,sampint: "<<n<<" "<<p<<" "<<burnin<<" "<<sampint<<endl;
	
	//END OF MAT INPUT
	
	srand(time(NULL));
	int** X = new int*[n];
	for(int l = 0;l < n;l++){
		X[l] = new int[p];
		for(int s = 0;s < p;s++){
			X[l][s] = 2 * (flip() > 0.5) - 1;
		}
	}
	
	int sampleSkip = 0;
	for(int l = 0;l < n;l++){
		if(l == 1){
			sampleSkip = burnin + sampint;
		}else{
			sampleSkip = sampint;
		}
	
		for(int ctr = 0;ctr < sampleSkip;ctr++){
			for(int s = 0;s < p;s++){
				double tmp1 = 0, tmp2 = 0;
				for(int t = 0;t < p;t++){
					if(t == s){
						continue;
					}
					tmp1 = tmp1 + graph[t][s] * X[l][t];
					tmp2 = tmp2 - graph[t][s] * X[l][t];
				}
				double lprob = exp(tmp1)/(exp(tmp1)+exp(tmp2));
			
				X[l][s] = 2*(flip() <= lprob) - 1;
			}
		}
	
	}
		
	
	plhs[0] = mxCreateDoubleMatrix(n,p,mxREAL);
	double* X_ret = mxGetPr(plhs[0]);
	for(int l = 0;l < n;l++){
		for(int s = 0;s < p;s++){
			X_ret[I(l,s,n)] = X[l][s];
		}
	}
}

