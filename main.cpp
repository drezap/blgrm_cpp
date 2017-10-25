/*  PX-Bayesian Low-rank Graph Regression Model
 *  main.cpp
 *  Eunjuee Lee
 *  C++ implementation by Andre Zapico
 */

#include <iostream>
#include <armadillo>
#include "px_blgrm.h"
#include "px_blgrm_helper.h"
#include "mcmc_para.h"
#include "random_generator.h"



using namespace std;
using namespace arma;


int main(int argc, char *argv[]){
  
  try{

    mat L_temp; L_temp.load(argv[1], csv_ascii);
    mat X; X.load(argv[2], csv_ascii);

    string R_in = argv[3];
    string V_in = argv[4];
    string niter_in = argv[5];
    string burnin_in = argv[6];
    string slice_niter_in = argv[7];
    string slice_burnin_in = argv[8];

    int R = stoi(R_in);
    int V = stoi(V_in);
    int n = X.n_rows; int p = X.n_cols;

    // load in L
    cube L; L.zeros(V, V, n);
    for(int i = 0; i < n; ++i){
      L.slice(i) = reshape(L_temp.row(i), V, V);
    }

    // set parameters
    mcmc_para parameters;
    parameters.set_niter(stoi(niter_in));
    parameters.set_burnin(stoi(burnin_in));
    parameters.set_B(V, R);
    parameters.set_Lambda(n, R);
    parameters.set_Gamma(p, R);
    parameters.set_sigma(1);
    parameters.set_sig_gam(1);
    parameters.set_b1(.01);
    parameters.set_b2(.01);
    parameters.set_c1(.01);
    parameters.set_c2(.01);
    parameters.set_va(.5);
    parameters.set_vb(.5);

    parameters.set_slice_niter(stoi(slice_niter_in));
    parameters.set_slice_burnin(stoi(slice_burnin_in));
    parameters.set_slice_width(10);
    cout << size(L) << "\n";
    cout << size(X) << "\n";
    
    px_blgrm(L, X, R, parameters);
  }catch(const std::exception& e){
    cout << "Error in inputs. Please correct inputs, or uncomment line 164 in main.cpp to run simulation\n";

    srand(1); // fix random seed

    // initialize all parameters, according to coni3_test.m
    int R = 10;
    int V = 50;
    int n = 100;
    int p = 2;

    double sigma_0 = 1.0; // Fix variance of Lambda prior for identifiability

    mcmc_para parameters;
    parameters.set_niter(5500);
    parameters.set_B(V, R);
    parameters.set_burnin(500);
    parameters.set_Lambda(n, R);


    parameters.set_Gamma(p, R); // p as in the simulation
    // init to defaults
    parameters.set_sigma(1);
    parameters.set_sig_gam(1);
    parameters.set_b1(.01);
    parameters.set_b2(.01);
    parameters.set_c1(.01);
    parameters.set_c2(.01);
    parameters.set_va(.5);
    parameters.set_vb(.5);

    // iterations for metrop/slice
    parameters.set_slice_niter(25);
    parameters.set_slice_burnin(10);
    parameters.set_slice_width(10);

    // now simulate data, L and X
    double sigma_error = 1.0;

    mat B(V, R);


    for(int i = 0; i < R; ++i){
      B.col(i) = n_norm(V);
    }


    // simulate data
    cube L; L.zeros(V, V, n);
    cube tL; tL.zeros(V, V, n);
    cube Lambda; Lambda.zeros(R, R, n);


    vec x; x = n_norm(n) + 0.5; // init x to 100 std norm, mean .5

    mat A; vec temp(R*(R+1)/2); temp = n_norm(R*(R+1)/2);
    mat Lamb; Lamb.zeros(R, R);
    mat AA; AA.zeros(V, V); vec a;
    mat noise; noise.zeros(50, 50);

    for(int i = 0; i < n; ++i){
      A.zeros(R, R);
      vec_2_uptri(A, n_norm(R * (R + 1) / 2) * (1 / sqrt(1)) + 1, R);
      A(0, 1) = A(0, 1) + x(i) * 4;
      A(1, 2) = A(1, 2) + x(i) * 4;
      Lamb = A + A.t() - 2 * diagmat(A.diag());
      Lamb.diag() = n_norm(R) + 1; // 3 norm(1,1) RV

      Lambda.slice(i) = Lamb;

      // need V*(V+1)/2 RB
      a = n_norm_musig(V * (V + 1) / 2, 1, sigma_error / sqrt(2));
      vec_2_uptri(AA, a, V);

      noise = AA + AA.t() - 2 * diagmat(AA.diag());

      noise.diag() = n_norm_musig(V, 0, sigma_error);

      tL.slice(i) = B * Lambda.slice(i) * B.t();
      L.slice(i) = B * Lambda.slice(i) * B.t() + noise;

    }

    // set up design matrix
    mat X; X.ones(n, 2);
    X.col(1) = x;


    a = n_norm_musig(V * (V + 1) / 2, 1, sigma_error);

    // write the actual B, Lambda, for testing
    ofstream outf;
    outf.open("B_true.txt");
    outf << B;
    outf.close();

    // begin loop and testing helper functions
    //px_blgrm(L, X, R, parameters);

  }

  return 0;
}
