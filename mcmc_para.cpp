/*   PX-Bayesian Low-rank Graph Regression Model
 *   MCMCpara nested class, for parameter initialization
 *   Eunjee Lee
 *   C++ Implementation by Andre Zapico
 */


#include <iostream>
#include <armadillo>
#include "mcmc_para.h"

using namespace std;
using namespace arma;

void mcmc_para::set_niter(int input_niter){niter = input_niter;}
int mcmc_para::get_niter(void){return niter;}

void mcmc_para::set_burnin(int input_burnin){burnin = input_burnin;}
int  mcmc_para::get_burnin(void){return burnin;}

void mcmc_para::set_B(int V, int R){
    B.zeros(V, R);
}
mat mcmc_para::get_B(void){return B;}

void mcmc_para::set_Lambda(int n, int R){
    Lambda.zeros(R, R, n);
}
cube mcmc_para::get_Lambda(void){return Lambda;}

void mcmc_para::set_Gamma(int p, int R){
    int temp = R * (R + 1) / 2;
    Gamma.zeros(p, temp);
}
//mcmc parameters
mat mcmc_para::get_Gamma(void){return Gamma;}

void mcmc_para::set_sigma(double in = 1.0){sigma = in;}
double mcmc_para::get_sigma(void){return sigma;}

void mcmc_para::set_sig_gam(double in = 1.0){sig_gam = in;}
double mcmc_para::get_sig_gam(void){return sig_gam;}

void mcmc_para::set_b1(double in = .01){b1 = in;}
double mcmc_para::get_b1(void){return b1;}

void mcmc_para::set_b2(double in = .01){b2 = in;}
double mcmc_para::get_b2(void){return b2;}

void mcmc_para::set_c1(double in = .01){c1 = in;}
double mcmc_para::get_c1(void){return c1;}

void mcmc_para::set_c2(double in = .01){c2 = in;}
double mcmc_para::get_c2(void){return c2;}

void mcmc_para::set_va(double in = .5){va = in;}
double mcmc_para::get_va(void){return va;}

void mcmc_para::set_vb(double in = .5){vb = in;}
double mcmc_para::get_vb(void){return vb;}

//mcmc slicesample parameters (may not need, if use metropolis)
void mcmc_para::set_slice_niter(int in){slice_niter = in;}
int mcmc_para::get_slice_niter(void){return slice_niter;}

void mcmc_para::set_slice_burnin(int in){slice_burnin = in;}
int mcmc_para::get_slice_burnin(void){return slice_burnin;}

void mcmc_para::set_slice_width(int in){slice_width = in;}
int mcmc_para::get_slice_width(void){return slice_width;}

