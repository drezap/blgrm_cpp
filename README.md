# blgrm_cpp
A Bayesian Low Rank Graph Regression Model for Brain Network Connectivity Data

Programs for parameter-expanded Bayesian low-rank graph regression models.

This px_blgrm package is written by Andre Zapico, of Chandra Sripada's lab.

We propose a Bayesian low-rank graph regression modeling (BLGRM) framework for the regression analysis of matrix response data across subjects. This development is motivated by performing detailed comparisons of functional and structural connectivity data across subjects, groups, and time and relating connections to particular behavioral measures. The BLGRM can be regarded as a novel integration of principal component analysis, tensor decomposition, and regression models. In BLGRM, we find a common low-dimensional subspace for efficiently representing all matrix responses. Based on such low-dimensional representation, we can easily quantify the effects of various predictors of interest, such as age and diagnosis status, and then perform regression analysis in the common subspace, leading to both substantial dimension reduction and much better prediction. Posterior computation proceeds via an efficient Markov chain Monte Carlo algorithm.

PX_BLGRM is a main function to perform parameter-expanded BLGRM on symmetric matrix responses. The MCMC outputs are stored in the struct "Output", which contains a reconstruction error and the posterior mean of the parameters.

Requirements:
Armadillo, a C++ linear algebra library, and it's required dependencies.
The g++ compiler.

How to Use:

Linux:
1) Clone all required files.
2) Compile the program with $./make. You may need to change file permissions with: $ chmod +x make
3) The program reads in Z-transformed vectorized connectivity matrices in CSV format, with 0's along the diagonal. The program will reshape these vectorized matrices into VxV symmetric PSD matrices. The X matrix is a standard design matrix. You will need to code a column of 1's for the intercept. The number of rows of both of these inputs should be the same.
4) Run the program with ./px_blgrm [network_matrices] [design matrix] [R, reduced dimsnion size] [V, original connectivity matrix size] [number of iterations] [number of burnin] [metropolis iterations] [metropolis burnin]


The program works best with very few metropolis iterations and 0 or few burnin iterations for the metropolis. High iterations for the metropolis will give you erroneous results. You may have to print out the 'aa_B' matrix in px_blgrm.cpp to ensure model is converging properly.

Please direct any bugs or questions to Andre Zapico: drezap@umich.edu


