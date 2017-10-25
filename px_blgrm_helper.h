/* PX-BLGRM Helper Functions
 * Eunjee Lee
 * Cpp Implementation by Andre Zapico
 */


#ifndef px_blgrm_helper_H
#define px_blgrm_helper_H

#include <armadillo>
#include "mcmc_para.h"

/* Create a row vector of the form 1,sqrt(2),...sqrt(2), for R elements,
   1,sqrt(2),...,sqrt(2), for R-1 elements, until 1 */
void init_svec(arma::vec &v, int R);

/* Creates a square R*(R+1)/2 Matrix with 1,sqrt(2),sqrt(2),...,sqrt(2), for R
   elements, 1,sqrt(2), sqrt(2),...,sqrt(2), for R-1 elements, until 1, as in
   equation (6) of Paper: BLGRM for Human Connectome Data */
void init_D(arma::mat &D, int R);

/* Creates an RxR matrix with 2 along the diagonal*/
void init_Psi(arma::mat &Psi, int R); 

/* Turn symmetric matrix into vector, including double elements,
   this is equivalent to math definition vec(M) */
void vec_symm_mat(arma::mat M, arma::vec &v);

/* Take vector w R(R+1)/2 elements and turn it into lower triangular matrix */
void vec_2_lowtri(arma::mat &out, arma::vec in, int R);

/* Vector with R(R+1)/2 elementer map to upper triangular */
void vec_2_uptri(arma::mat &out, arma::vec in, int R);

/* Map symmetric matrix onto vector by columns, where diagonal elements
   stay the same, and off diagonal elements are multiplied by sqrt(2) */
//void svec(arma::vec &v, arma::mat M);
arma::vec svec(arma::mat M);


/* Custom Function for cholesky decompositions, thanks
   Jian Kang  */
arma::mat cholesky_decomp(arma::mat A);

/* here we take svec(M) * vec(M) to compute the unique orthogonal matrix, P */
void init_mat_P(arma::mat &M, int n);

/* Calculates symmetric kronecker product of A, B and stores it in G */
arma::mat skron(arma::mat A, arma::mat B);

/* Removes certain elements of a column vector given a range */
void pop_vec(arma::vec &pop_vec, int start, int end);

/* Remove elements of vector not in the specified range */
void subset_vec(arma::vec &sub_vec, int start, int end);

/* Removes certain rows of a matrix */
void pop_row_mat(arma::mat &pop_mat, int start, int end);

/* Removes certain columns of a matrix */
void pop_col_mat(arma::mat &pop_mat, int start, int end);

/* Update S (eq 6), PX_BLGRM it is named C, */
void update_S(arma::mat &S, arma::mat &Cov, arma::mat B, arma::mat Psi,
	      arma::mat D, double sigma, double sigma_0);

/* The Part of the Gibbs sampler that updates Lambda 
   mu_g and s_g are also updated */
void update_Lambda(arma::cube &Lambda, arma::vec &mu_g, arma::mat &s_g,
		   arma::cube L, arma::mat S, arma::mat B, arma::mat X,
		   arma::mat Psi, arma::mat D, arma::mat Cov, arma::cube Delta,
		   double sigma, double sigma_0, arma::vec lamb);

/* Part of Gibbs that updates Gamma */
void update_Gamma(arma::mat &Gamma,
		  int R, int p, double sigma_0, arma::mat sig_gam,
		  arma::vec mu_g, arma::mat s_g);

/* Calculate D */
void calculate_Delta(arma::cube &Delta, arma::mat X, arma::mat Gamma,
		     int R, int n);


/* Helper function to imitate Eunjee's,
   take one column of matrix and turn it into the vector
   a, while remaninig lower-triangular */
arma::mat LTI(arma::vec a, arma::mat A1, int i, int V, int R);


// may not need to be a matrix output, this is temporary
double log_pdf_B(arma::vec b1, arma::mat B, int i, double sigma,
		    arma::mat lam_l, arma::mat lam_s, int range,
		    int range_0, arma::mat VV0, int R, int V, int n);


/* Update B using adaptive metropolis */
void update_B(arma::mat &B, arma::cube Lambda, arma::cube L, arma::mat tau,
	      arma::mat phi, int R, int V, mcmc_para parameters, int iter);



/* Function Eunjee uses for ?? */
arma::mat sk(arma::mat b);

/* log posterior of psi for psi = lambda|psi * psi in (13) */
//double log_pdf_Psi(arma::vec b1, int n, arma::mat D, double sigma_0,
//		   arma::mat BMM, double v_a, double v_b);
double log_pdf_Psi(arma::vec b1, int n, arma::mat D, double sigma_0,
		   arma::mat BMM, double v_a, double v_b);


/* Function to update all hyperparameters, also updates phi*Lambda with 
   metropolis (or slice sampling) */
void update_hyperparameters(arma::mat &phi, arma::vec &tau, arma::mat &Psi,
			    arma::vec &lambb, arma::vec &rndd,
			    arma::mat D, arma::cube Lambda,
			    arma::cube Delta, arma::mat B, double sigma_0,
			    int V, int R, mcmc_para parameters);

double trace_1(arma::cube in);

void sign_scale_adjustment(arma::mat &B, arma::cube &Lambda, arma::mat &Gamma,
			   arma::mat &aa_B, arma::cube &a_Lambda,
			   arma::mat &a_Gamma,
			   arma::mat &s_B, arma::cube &s_Lambda,
			   arma::mat &s_Gamma, arma::cube &s_L,
			   double &sigma, arma::mat &sig_gam, double &sigma_hat,
			   arma::cube L, arma::mat Psi, int p, int niter,
			   mcmc_para parameters);

void write_to_file(arma::mat B, arma::cube Lambda, arma::mat Gamma,
		   arma::cube Delta, std::string outf_name);

double reconstruction_error(arma::cube Lambda, arma::mat B, arma::cube L,
			    int n);

#endif

