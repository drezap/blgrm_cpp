# blgrm_cpp
A Bayesian Low Rank Graph Regression Model for Brain Network Connectivity Data

Eunjee Lee.


The original model and MATLAB implementation is was designed and implemented by Eunjee Lee, Michigan Biostatistics Department. The link to the MATLAB implementation be found here: https://github.com/BIG-S2/BLGRM. C++ implementation by Andre Zapico, of Sripada Lab, University of Michigan Department of Psychiatry. Please direct any bugs or questions to Andre Zapico: drezap@umich.edu


Programs for parameter-expanded Bayesian low-rank graph regression models.

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

Data is output in the following formats:
B: The B matrices stacked by column.

Gamma: For each iteration, there p vectorized rows of Gamma realizations. That is, each row is 1 x R(R+1)/2, and there are p * #iterations rows.

Lambda: For each iteration, we have n vectorized Lambda matrices. so we have n * #iter rows of 1xR^R matrices.


a sample run will look like this:
$ ./px_blgrm network_matrices.csv design_matric.csv 17 250 5500 500 10 0


The program works best with very few metropolis iterations and 0 or few burnin iterations for the metropolis. High iterations for the metropolis will give you erroneous results. You may have to print out the 'aa_B' matrix in px_blgrm.cpp to ensure model is converging properly.

