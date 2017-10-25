/* PX-Bayesian Low-rank Graph Regression Model
 * Header File for px_blgrm.cpp
 * Eunjee Lee
 * cpp implementation by Andre Zapico 
 */

#ifndef px_blgrm_H
#define px_blgrm_H

#include "mcmc_para.h"
#include <armadillo>

/* Program that is the primary loop for PX_BLGRM */
void px_blgrm(arma::cube L, arma::mat X, int R, mcmc_para mcmc_parameters);

#endif
