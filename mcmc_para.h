/*   PX-Bayesian Low-rank Graph Regression Model
 *   MCMCpara nested class, for parameter initialization
 *   Andre 
 *   mcmc_para.h
 *    
 */


#ifndef mcmc_para_H
#define mcmc_para_H

#include <armadillo>


class mcmc_para
{
  int niter;
  int burnin;
  double sigma;
  double sig_gam;
  double b1;
  double b2;
  double c1;
  double c2;
  double va;
  double vb;
  int slice_niter;
  int slice_burnin;
  int slice_width;

  
 public:
  arma::mat B;
  arma::cube Lambda;
  arma::mat Gamma;

  
  //mcmc parameters
  void set_niter(int input_niter);
  int get_niter(void);

  void set_burnin(int input_burnin);
  int get_burnin(void);

  void set_B(int V, int R);
  arma::mat get_B(void);

  void set_Lambda(int V, int R);
  arma::cube get_Lambda(void);

  void set_Gamma(int p, int R);
  arma::mat get_Gamma(void);

  void set_sigma(double in);
  double get_sigma(void);

  void set_sig_gam(double in);
  double get_sig_gam(void);
  
  void set_b1(double in);
  double get_b1(void);

  void set_b2(double in);
  double get_b2(void);

  void set_c1(double in);
  double get_c1(void);

  void set_c2(double in);
  double get_c2(void);

  void set_va(double in);
  double get_va(void);

  void set_vb(double in);
  double get_vb(void);

  // for slicesample, if needed (if not use metropolis)
  void set_slice_niter(int in);
  int get_slice_niter(void);

  void set_slice_burnin(int in);
  int get_slice_burnin(void);

  void set_slice_width(int in);
  int get_slice_width(void);
};


#endif
