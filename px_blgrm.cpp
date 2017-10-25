/*
%----------------------------------------------------------------------------------
%		PX-Bayesian Low-rank Graph Regression Model
%
%----------------------------------------------------------------------------------
%
% Decomposition model: L_i = B*Lambda_i*B + E		   
% Regression model: Lambda_i = X*Gamma + e 		   
%----------------------------------------------------------------------------------
%
% Data input:
%	L: V x V x n, input data array. 
%	   A three-dimensional array with n symmetric V x V matrices.
%	X: n x p, covariate matrix for the regression model.
%	n: the number of subjects.
%	V: the number of edges or the dimension of input matrix.	
%	R: the prespecified number of eigenvectors. We use BIC to determine R.	
%----------------------------------------------------------------------------------
%
% MCMC setting input:
%	MCMCpara: input structure of the total number of MCMC iterations, burn-in, 
%			  initial values, hyperparameters for MCMC.
%	 	
%		mcmc_para.Niter: the total number of MCMC iterations.
%		mcmc_para.burnin: the number of burn-in
%		mcmc_para.B: the initial value for B matrix.
%		mcmc_para.Lambda: the initial value for Lambda matrix.
%		mcmc_para.Gamma: the initial value for Gamma matrix.
%		mcmc_para.sigma: the initial value for sigma (decomposition model variance).
%		mcmc_para.sig_gam: the initial value for sig_gam (regression model variance).
%		mcmc_para.b1: the initial value for hyperparameter for priors of sigma.
%		mcmc_para.b2: the initial value for hyperparameter for priors of sigma.
%		mcmc_para.c1: the initial value for hyperparameter for priors of sig_gam.
%		mcmc_para.c2: the initial value for hyperparameter for priors of sig_gam.
%		mcmc_para.va: the initial value for parameter for priors of Psi.
%		mcmc_para.vb: the initial value for parameter for priors of Psi.
%----------------------------------------------------------------------------------
%
% Metropolis setting input:
%	slice: input structure of tuning parameters for metropolis sampling for B.
%		slice.miter: the maximum number of iterations for tuning. 
%		slice.burnin: the number of burnin during the tuing iterations. 
%----------------------------------------------------------------------------------
*/


#include <iostream>
#include <armadillo> 
#include "mcmc_para.h"
#include "px_blgrm.h"
#include "px_blgrm_helper.h"
#include "random_generator.h"

using namespace std;
using namespace arma;


void px_blgrm(cube L, mat X, int R, mcmc_para mcmc_parameters){

  // Initialization, we are passing B in by mcmc_para.B,   
  int niter = mcmc_parameters.get_niter();
  int burnin = mcmc_parameters.get_burnin();

  mat B = mcmc_parameters.B; // B points to mcmc_parameter on heap, B
  cube Lambda = mcmc_parameters.Lambda;
  mat Gamma = mcmc_parameters.Gamma;
  
  double sigma = mcmc_parameters.get_sigma();
  double sig_gam_temp = mcmc_parameters.get_sig_gam();
  mat sig_gam; sig_gam.eye(R * (R + 1) / 2, R * (R + 1) / 2) * sig_gam_temp;
  double b1 = mcmc_parameters.get_b1();
  double b2 = mcmc_parameters.get_b2();
  double c1 = mcmc_parameters.get_c1();
  double c2 = mcmc_parameters.get_c2();
  double va = mcmc_parameters.get_va();
  double vb = mcmc_parameters.get_vb();

  
  double devhat = 0; double sigma_hat = 0;
  int V = L.n_rows;
  int n = L.n_slices;
  int p = X.n_cols;

  double sigma_0 = 1.0; // indentifiability 

  // init Psi
  mat Psi; Psi.eye(R, R); Psi = 2 * Psi; // diagonal matrix of two
  // init lambb
  vec lambb; lambb = n_gamma(R, 1, 0.5);

  // init phi
  mat phi; phi.ones(V, R);
  for(int i = 0; i < R; ++i){
    for(int j = i + 1; j < R; ++j){
      phi(i, j) = 0;
    }
  }
  // init tau
  vec tau; tau.ones(R);

  // // allocation for PX parameters
  cube a_Lambda; a_Lambda.zeros(R, R, n);
  cube s_Lambda; s_Lambda.zeros(R, R, n);
  cube out_Lambda; out_Lambda.zeros(R, R, n);

  mat aa_B; aa_B.zeros(V, R);
  mat s_B; s_B.zeros(V, R);
  mat out_B; out_B.zeros(V, R);
  cube s_L; s_L.zeros(V, V, n);
  
  mat a_Gamma; a_Gamma.zeros(p, R * (R + 1) / 2);
  mat s_Gamma; s_Gamma.zeros(p, R * (R + 1) / 2);
  mat out_Gamma; out_Gamma.zeros(p, R * (R + 1) / 2);
  double reconstr_error;
  
  
  cube Delta; Delta.zeros(R, R, n);
  vec mu_g; mu_g.zeros(p * R * (R + 1) / 2);
  mat s_g; s_g.zeros(p * R * (R + 1) / 2, p * R * (R + 1) / 2);
  vec gam_tau; gam_tau.ones(p * R * (R + 1) / 2);
  vec rnd; rnd.ones(R);
  vec rndd; rndd.ones(R); rndd = 2 * rndd;
  mat lamb; lamb = n_norm(R * (R + 1) / 2) / sqrt(sigma_0) / sqrt(2);
  
  
  // parameters that only need exist inside this function
  mat S; mat Cov;
  
  
  mat D; init_D(D, R);

  for(int iter = 0; iter < mcmc_parameters.get_niter(); ++iter){ // temporarily disable all interations
    cout << "\n\n" << iter << "\n";
  
    cout << "update S\n";
    update_S(S, Cov, B, Psi, D, sigma, sigma_0); // to be initialized
   
    cout << "update_Lambda\n";
    update_Lambda(Lambda, mu_g, s_g, L, S, B, X, Psi, D,
    		  Cov, Delta, sigma, sigma_0, lamb);

    cout << "update_Gamma\n";
    update_Gamma(Gamma, R, p, sigma_0, sig_gam, mu_g, s_g);

    cout << "calculate_Delta\n";
    calculate_Delta(Delta, X, Gamma, R, n);

    cout << "update_B\n";
    update_B(B, Lambda, L, tau, phi, R, V, mcmc_parameters, iter); 

    cout << "update_hyper\n";
    update_hyperparameters(phi, tau, Psi, lambb, rndd, D, Lambda, Delta, B, sigma_0,
    			   V, R, mcmc_parameters);

    cout << "sign and scale \n";
    sign_scale_adjustment(B, Lambda, Gamma, aa_B, a_Lambda, a_Gamma,
			  s_B, s_Lambda, s_Gamma, s_L,
			  sigma, sig_gam, sigma_hat, L, Psi, p, iter,
			  mcmc_parameters);
    
    //cout << std::fixed << aa_B << "\n";
    write_to_file(aa_B, a_Lambda, a_Gamma, Delta,"outf");

    // temporary for reconstruction error
    if(iter >= mcmc_parameters.get_burnin()){
      s_B = s_B + aa_B;
      s_Lambda = s_Lambda + a_Lambda;
      s_Gamma = s_Gamma + a_Gamma;
    }
    
  }

  out_B = s_B / (mcmc_parameters.get_niter() - mcmc_parameters.get_burnin());
  out_Lambda = s_Lambda / (mcmc_parameters.get_niter() - mcmc_parameters.get_burnin());
  out_Gamma = s_Gamma / (mcmc_parameters.get_niter() - mcmc_parameters.get_burnin());
  reconstr_error = reconstruction_error(out_Lambda, out_B, L, n);
  reconstruction_error(a_Lambda, aa_B, L, n);
  
  
  ofstream outf;
  outf.open("outf_reconstruction_error.txt");
  outf << reconstr_error;
  outf.close();
 
}

  
