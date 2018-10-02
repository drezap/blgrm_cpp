# blgrm_cpp
A Bayesian Low Rank Graph Regression Model for Brain Network Connectivity Data

Eunjee Lee.


The original model and MATLAB implementation is was designed and implemented by Eunjee Lee, PhD, UNC. The link to the MATLAB implementation be found here: https://github.com/BIG-S2/BLGRM. C++ implementation by Andre Zapico, of Sripada Lab, University of Michigan Department of Psychiatry. Please direct any bugs or questions to Andre Zapico: drezap@umich.edu

I copied this from Eunjee's implementation:

Programs for parameter-expanded Bayesian low-rank graph regression models.

We propose a Bayesian low-rank graph regression modeling (BLGRM) framework for the regression analysis of matrix response data across subjects. This development is motivated by performing detailed comparisons of functional and structural connectivity data across subjects, groups, and time and relating connections to particular behavioral measures. The BLGRM can be regarded as a novel integration of principal component analysis, tensor decomposition, and regression models. In BLGRM, we find a common low-dimensional subspace for efficiently representing all matrix responses. Based on such low-dimensional representation, we can easily quantify the effects of various predictors of interest, such as age and diagnosis status, and then perform regression analysis in the common subspace, leading to both substantial dimension reduction and much better prediction. Posterior computation proceeds via an efficient Markov chain Monte Carlo algorithm.

PX_BLGRM is a main function to perform parameter-expanded BLGRM on symmetric matrix responses. The MCMC outputs are stored in the struct "Output", which contains a reconstruction error and the posterior mean of the parameters.


I believe Eunjee quit Michigan.
