/* random_generator.cpp
 * Various Random Number Generators aiding the model
 * Andre Zapico
 */


#include <iostream> // std out
#include <random> // random number generators
#include <armadillo> // fast linear algebra
#include <cmath> // math
#include <algorithm> // random device
#include "px_blgrm_helper.h"

using namespace std;
using namespace arma;


vec n_unif(int n){
  // generate p uniform(0,1) distributed realizations
  // requires: out a arma::vec of length n
  
  random_device rd;
  default_random_engine gen(rd());
  uniform_real_distribution<double> unif(0.0, 1.0);
  vec out; out.zeros(n);

  for(int i = 0; i < n; ++i){
    out(i) = unif(gen);
  }
  return out;
}

vec n_unif_int_a_b(int n, int a, int b){

  random_device rd;
  default_random_engine gen(rd());
  uniform_int_distribution<int> int_unif(a, b);
  vec out; out.zeros(n);

  for(int i = 0; i < n; ++i){
    out(i) = int_unif(gen);
    cout << int_unif(gen);
  }
  return out;
}

vec n_norm(int n){
  // generate n standard normally distributed random variables
  // requires: out a arma::vec of length n
  random_device rd;
  default_random_engine gen(rd());
  normal_distribution<double> norm(0.0,1.0);
  vec out(n);
  
  for(int i = 0; i < n; ++i){
    out(i) = norm(gen);
  }
  return out;
}


vec n_norm_musig(int n, double mu, double sigma){
  // generate n standard normally distributed random variables
  // requires: out a arma::vec of length n
  random_device rd;
  default_random_engine gen(rd());
  normal_distribution<double> norm(mu, sigma);
  vec out(n);
  
  for(int i = 0; i < n; ++i){
    out(i) = norm(gen);
  }
  return out;
}



mat n_norm_mat(int n){
  mat out; out.zeros(n, n);
  
  for(int i = 0; i < n; ++i){
    for(int j = 0; j < n; ++j){
      out(i , j) = n_norm(1)(0);
    }
  }
  return out;
}

vec mv_norm(vec mu, mat Sigma){
  // return n x d matrix of random vectors with mean mu
  // and covariance sigma
  // requires: mu is n x d matrix of means
  //           Sigma d x d positive semi-definite symmetrix matrix

  int n = mu.n_elem;
  vec out(n);
  out = mu + chol(Sigma) * n_norm(n);
  //out = mu + cholesky_decomp(Sigma) * n_norm(n);
  return out;
}

void rpois(vec &out, vec mu){
  // this function need be edited

  // generate poisson distributed RV
  // requies: out a arma::vec of length n
  
  random_device rd;
  default_random_engine gen(rd());
  int n = mu.n_elem;

  for(int i = 0; i < n; ++i){
    poisson_distribution<int> rpois(mu(i));
    out(i) = rpois(gen);
  }
}



vec n_inv_gaussian(int n, double mu, double lambda){
  // return n inverse gaussian distributed RV
  // with specified parameters
  vec out(n);
  
  double y, x;
  vec z; z.zeros(1);
  vec u; u.zeros(1);

  
  for(int i = 0; i < n; ++i){
    z = n_norm(1); // get 1 norm(0,1) distributed RV
    y = z(0) * z(0);
    x = mu + 0.5 * mu * mu * y / lambda - 0.5 * (mu / lambda) *
      sqrt(4 * mu * lambda * y + mu * mu * y * y);
    u = n_unif(1); // get 1 runif(0,1) distributed RV
    if(u(0) <= (mu / (mu + x))){
      out(i) = x;
    }else{
      out(i) = mu * mu / x;
    }
  }
  return out;
}

vec n_gamma(int n, double shape, double scale){
  // n draws from gamma_distribution with
  // given shape and scale
  
  vec out; out.zeros(n); // init and allocate space
  
  random_device rd;
  default_random_engine gen(rd());
  gamma_distribution<double> rgamma(shape, scale);

  for(int i = 0; i < n; ++i){
    out(i) = rgamma(gen);
  }
  return out; 
}

vec n_exp(int n, double lambda){

  vec out; out.zeros(n);
  random_device rd;
  default_random_engine gen(rd());
  exponential_distribution<double> rexp(lambda);
  for(int i = 0; i < n; ++i){out(i) = rexp(gen);}
  return out;
}

mat ARWM_B(int Niter, int burnin, int n_samples, vec beta_init,
	   mat B, int i, double sigma, mat lam_l, mat lam_s,
	   int range, int range_0, mat VV0, int R, int V, int n){
  // A general adaptive random walk metropolis that does
  
  
  int p = beta_init.n_elem; // number of predictors
  mat C(p,p, fill::eye); // pxp diagonal matrix, 1's along diag
  //  cout << C;
  mat K = chol(C).t(); // cholesky decomp, transpose
  vec mu; mu.zeros(p); // vector of p zeros
  vec bmu(p);

  
  mat out; out.zeros(p, burnin + Niter); // out variables, beta + sig
  // so we can take n samples from this distribution
  mat beta_out; beta_out.zeros(p, n_samples);
  vec to_sample; to_sample.zeros(n_samples);
  
  // addaptive params
  int cptUpdate = 0; 
  int LUpdate = 10000; // turn to 500
  double lsig = -1.0;
  
  vec beta; beta = beta_init; // init beta to 0
  vec beta_prop;
  
  double lpi = log_pdf_B(beta, B, i - 1, sigma, lam_l, lam_s,
			 range, range_0, VV0, R, V, n);
  double lpi_prop;

  double Acc;
  double runif;
  double u;
  
  for(int j = 1; j < Niter + burnin; ++j){
    
    beta_prop = beta + exp(lsig) * (K * n_norm(p)); // adaptive step size
    lpi_prop = log_pdf_B(beta_prop, B, i - 1, sigma, lam_l,
  		     lam_s, range, range_0, VV0, R, V, n);
    
    Acc = min(1.0, exp(lpi_prop - lpi)); // log posterior ratio
    runif = n_unif(1)(0);
    
    if(runif <= Acc){
      beta = beta_prop;
      lpi = lpi_prop;
    }

    // update adaptive parameters
    lsig = lsig + (1/pow(j, 0.7)) * (Acc - 0.4); // step size
    mu = mu + 1 / (j * (beta - mu));
    bmu = (beta - mu);
    C = C + 1 / (j * (bmu * bmu.t() - C)); // covariance

    if(cptUpdate == LUpdate){
      // sometimes bugs out, but you can re-run it
      K = chol(C).t();
      cptUpdate = 0;
    }else{
      cptUpdate = cptUpdate + 1;
    }
    out.col(j) = beta;
  }
  // remove the burnin
  out = out.cols(burnin + 1, out.n_cols - 1);
  
  for(int j = 0; j < n_samples; ++j){
    u = n_unif(1)(0);
    
    beta_out.col(j) = out.col(floor(u * out.n_cols));
  }
  
  // returns sample of n_samples from this posterior
  return beta_out;
}


mat ARWM_Psi(int Niter, int burnin, int n_samples, vec beta_init, int n,
	     mat D, double sigma_0, mat BMM, double v_a, double v_b){
  // A general adaptive random walk metropolis
  
  int p = beta_init.n_elem; // number of predictors
  mat C(p,p, fill::eye); // pxp diagonal matrix, 1's along diag
  mat K = chol(C).t(); // cholesky decomp, transpose
  vec mu; mu.zeros(p); // vector of p zeros
  vec bmu(p);

  
  mat out; out.zeros(p, burnin + Niter); // out variables, beta + sig
  // so we can take n samples from this distribution
  mat beta_out; beta_out.zeros(p, n_samples);
  vec to_sample; to_sample.zeros(n_samples);
  
  // addaptive params
  int cptUpdate = 0; 
  int LUpdate = 500; // turn to 500
  double lsig = -1.0;
  
  vec beta; beta = beta_init; // init beta to 0
  vec beta_prop;
  
  double lpi = log_pdf_Psi(beta, n, D, sigma_0, BMM, v_a, v_b);
  double lpi_prop;

  double Acc;
  double runif;
  double u;
  
  for(int j = 1; j < Niter + burnin; ++j){
    
    beta_prop = beta + exp(lsig) * (K * n_norm(p) * .25); // adaptive step size
    for(int k = 0; k < beta_prop.n_elem; ++k){
      if(beta_prop(k) < 0){beta_prop(k) = -1 * beta_prop(k);}
    }
    lpi_prop = log_pdf_Psi(beta_prop, n, D, sigma_0, BMM, v_a, v_b);
    
    Acc = min(1.0, exp(lpi_prop - lpi)); // log posterior ratio
    runif = n_unif(1)(0);


    if(runif <= Acc){
      beta = beta_prop;
      cout << beta << "\n";
      lpi = lpi_prop;
    }

    // update adaptive parameters
    lsig = lsig + (1/pow(j, 0.7)) * (Acc - 0.4); // step size
    mu = mu + 1 / (j * (beta - mu));
    bmu = (beta - mu);
    C = C + 1 / (j * (bmu * bmu.t() - C)); // covariance

    if(cptUpdate == LUpdate){
      // sometimes bugs out, but you can re-run it
      K = chol(C).t();
      cptUpdate = 0;
    }else{
      cptUpdate = cptUpdate + 1;
    }
    out.col(j) = beta;
  }
  // remove the burnin
  out = out.cols(burnin + 1, out.n_cols - 1);
  
  for(int j = 0; j < n_samples; ++j){
    u = n_unif(1)(0);
    
    beta_out.col(j) = out.col(floor(u * out.n_cols));
  }
  
  // returns sample of n_samples from this posterior
  return beta_out;
}
