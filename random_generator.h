/* Random Number Generators 
 * various functions for generating
 * random numbers
 */



#ifndef random_generator_H
#define random_generator_H

#include <random>
#include <armadillo>
#include <algorithm>



/* Generat n uniform(0,1) realizations */
arma::vec n_unif(int n);

/* Sample by index */
arma::vec n_unif_int_a_b(int n, int alpha, int beta);

/* generate n std. normal realizations */
arma::vec n_norm(int n);

/* generate n mu, sigma normal realizations */
arma::vec n_norm_musig(int n, double mu, double sigma);


/* generate n x n matrix of std norm realizations */
arma::mat n_norm_mat(int n);

/* generate n multivariate normally distributed RV,
   given vector of means mu and covariance matrix cov*/
arma::vec mv_norm(arma::vec mu, arma::mat cov);

/* generate n poission RV with parameter mu */
// this function need be edited
void rpois(arma::vec &out, int n);

/* Generate an inverse gaussian distributed random variable */
arma::vec n_inv_gaussian(int n, double mu, double lambda);

/* Generate n gamma RV with given shape and scale
   I am modifying these to output a vector,
   it makes the rest of the program more efficient */
arma::vec n_gamma(int n, double shape, double scale);

arma::vec n_exp(int n, double lambda);

/* Adaptive metropolis algorithm for log_b_pdf,
   couldn't figure out how to include a function with
   arbitrary parameters, so there is more than one metrop*/
arma::mat ARWM_B(int Niter, int burnin, int n_samples, arma::vec beta_init,
		 arma::mat B, int i, double sigma, arma::mat lam_l,
		 arma::mat lam_s, int range, int range_0, arma::mat VV0,
		 int R, int V, int n);

arma::mat ARWM_Psi(int Niter, int burnin, int n_samples, arma::vec beta_init,
		   int n, arma::mat D, double sigma_0, arma::mat BMM,
		   double v_a, double v_b);

arma::vec slicesample_B(arma::vec x_0, double w, arma::mat B, int i,
			double sigma, arma::mat lam_l, arma::mat lam_s,
			int range, int range_0, arma::mat VV0, int R,
			int V, int n);

arma::vec slicesample_Psi(arma::vec x_0,  double w, int n, arma::mat D, double sigma_0,
		    arma::mat BMM, double v_a, double v_b);

#endif
