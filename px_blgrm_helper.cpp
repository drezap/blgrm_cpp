/* PX-BLGRM Helper Functions
 * Eunjee Lee
 * Cpp Implementation by Andre Zapico
 */



#include <iostream>
#include <armadillo>
#include <math.h>
#include <algorithm>
#include <string>
#include "mcmc_para.h"
#include "random_generator.h"


using namespace std;
using namespace arma;

void init_svec(vec &v, int R){
  // create svec
  
  int index = 0;
  v.set_size(R * (R + 1) / 2); v.fill(sqrt(2));
  
  for(int i = R; i > 0; --i){
    for(int j = 1; j <= i; ++j){
      if(j == 1){v(index) = 1;}
      index++;
    }
  }
}

void init_D(mat &D, int R){
  
  vec svec; init_svec(svec, R);
  D = diagmat(svec);
}


void init_Psi(mat &Psi, int R){
  // Matrix Psi initialization, see PX_BLGRM.m inits
  Psi.eye(R, R);
  Psi = Psi * 2.0;
}


void vec_symm_mat(mat M, vec &v){
  // take each row and append to vector
  // v should have same amount of elements as M
  
  int I = size(M)(0);
  int index = 0;

  for(int i = 0; i < I; ++i){
    for(int j = 0; j < I; ++j){
      v(index) = M(i,j);
      index++;
    }
  }
}

void vec_2_lowtri(mat &out, vec in, int R){
  // Requires: length(in) == R(R+1)/2
  int index = 0;
  mat temp(R, R); temp.fill(0);
  
  for(int j = 0; j < R; ++j){
    for(int i = j; i < R; ++i){
      temp(i, j) = in(index);
      ++index;
    }
  }
  out = temp;
}

void vec_2_uptri(mat &out, vec in, int R){
  
  int index = 0;
  mat temp(R, R); temp.fill(0);

  for(int i = 0; i < R; ++i){
    for(int j = i; j < R; ++j){
      temp(i, j) = in(index);
      ++index;
    }
  }
  out = temp;
}

vec svec(mat M){
  // length v should be n*(n + 1) / 2
  // M should be symmetric
  
  int I = size(M)(0);
  int index = 0;
  vec v; v.zeros(I * (I + 1) / 2); // reset v for functional purposes
  
  for(int i = 0; i < I; ++i){
    for(int j = i; j < I; ++j){
      if(i == j){
	v(index) = M(i, j);
      }else{
	v(index) = sqrt(2) * M(i, j);
      }
      ++index;
    }
  }
  return v;
}

mat cholesky_decomp(mat A){
  int k, i, j;
  int num_col = A.n_cols;
  mat out;
  
  for(k = 0; k < num_col; ++k){
    if(A(k, k) <= 0){
      printf("cholesky decomp: error matrix not PSD\n");
      return 0;
    }
    A(k, k) = sqrt(A(k, k));

    for(j = k + 1; j < num_col; ++j){
      A(j, k) /= A(k, k);
    }
    
    for(j = k + 1; j < num_col; ++j){
      for(i = j; i < num_col; ++i){
	A(i, j) = A(i, j) - A(i, k) * A(j, k);
      }
    }
  }
  out = A;
  return out;
}

void init_mat_P(mat &M, int n){
  int a = n * (n + 1) / 2;
  int b = n * n;
  M.zeros(a, b);

  int index_1_k;
  int index_1_l = 0;
  int index_2_k;
  int index_2_l;

  for(int i = 0; i < b; ++i){ // 

    // get indeces from longer side
    index_1_k = (i + 1) % n;
    if((i + 1) % n == 0){index_1_k = n;}
    if(index_1_k == 1){++index_1_l;} // increase right index
    
    index_2_k = 0;
    index_2_l = 1;

    for(int j = 0; j < a; ++j){

      if(index_2_k < n){
	++index_2_k;
      }
      else if(index_2_k == n){
	index_2_l++;
	index_2_k = index_2_l;
      }

      // do equality testing here
      // Shacke, 2013, eq 3.3
      if(((index_1_k == index_2_k) &&
      	  (index_1_k != index_1_l) &&
      	  (index_1_l == index_2_l)) ||
      	 ((index_1_k == index_2_l) &&
      	  (index_2_l != index_1_l) &&
      	  (index_1_l == index_2_k))){
      	M(j, i) = 1 / sqrt(2);
      }
      else if((index_1_k == index_1_l) &&
      	      (index_1_l == index_2_k) &&
      	      (index_2_k == index_2_l)){
      	M(j, i) = 1;
      }   
    }
  }
}


mat skron(mat A, mat B){
  // symmetric kronecker product
  // A and B must be square
  int n = A.n_cols;
  mat P; init_mat_P(P, n);
  mat G;

  G = 0.5 * P * (kron(A, B) + kron(B, A)) * P.t();
  return G;
}

void pop_vec(vec &pop_vec, int start, int end){
  // Requires: length(end - start) < length(in)
  //           end < length(in) - 1,
  //           etc.
  // Indexing starts at 0

  int n = pop_vec.n_elem;
  vec temp; temp.zeros(n - (end - start) - 1);
  int index = 0;
  
  for(int i = 0; i < n; ++i){
    if(i < start){
      temp(index) = pop_vec(i);
      ++index;
    } else if(i > end){
      temp(index) = pop_vec(i);
      ++index;
    }
  }
  pop_vec = temp;
}

void subset_vec(vec &sub_vec, int start, int end){

  int n = sub_vec.n_elem;
  vec temp; temp.zeros(end - start + 1);
  int index = 0;

  for(int i = 0; i < n; ++i){
    if((start <= i) & (i <= end)){
      temp(index) = sub_vec(i);
      ++index;
      break;
    }
  }
  sub_vec = temp;
}

void pop_row_mat(mat &pop_mat, int start, int end){
  // Remove a specified number of rows from a matrix
  // Requires: #rows > (end - start)
  // etc...


  int n = pop_mat.n_rows;
  mat temp; temp.zeros(n - (end - start) - 1, pop_mat.n_cols);
  int index = 0;
  
  for(int i = 0; i < n; ++i){
    if(i < start){
      temp.row(index) = pop_mat.row(i);
      ++index;
    } else if(i > end){
      temp.row(index) = pop_mat.row(i);
      ++index;
    }
  }
  pop_mat = temp; 
}

void pop_col_mat(mat &pop_mat, int start, int end){
  // Remove a specified number of rows from a matrix
  // Requires: #rows > (end - start)
  // etc...


  int n = pop_mat.n_cols;
  mat temp; temp.zeros(pop_mat.n_rows, n - (end - start) - 1);
  int index = 0;
  
  for(int i = 0; i < n; ++i){
    if(i < start){
      temp.col(index) = pop_mat.col(i);
      ++index;
    } else if(i > end){
      temp.col(index) = pop_mat.col(i);
      ++index;
    }
  }
  pop_mat = temp; 
}



void update_S(mat &S, mat &Cov, mat B, mat Psi, mat D, double sigma,
	      double sigma_0){
  // S matrix update

  mat cov;
  mat U; U = B.t() * B;

  cov = D * skron(U, U) * D * sigma + sigma_0 * D * skron(Psi, Psi) * D;
  cov = trimatl(cov) + trimatl(cov).t() - diagmat(cov.diag()); // force to be symmetric

  // used for update Lambda
  Cov = inv(cov);  
  Cov = trimatl(Cov) + trimatl(Cov).t() - diagmat(Cov.diag());

  S = solve(cov, D);
}


void update_Lambda(cube &Lambda, vec &mu_g, mat &s_g, cube L, mat S,
		   mat B, mat X, mat Psi, mat D, mat Cov, cube Delta, double sigma,
		   double sigma_0, vec lamb){
  // update Lambda
  int p = X.n_cols;
  int n = X.n_rows; // number of observations
  int R = B.n_cols; // num eigenvectors
  mat W(R, R);  vec M(R * (R + 1) / 2); // pre allocate memory
  // lamb is normally distributed random numbers /sqrt(sigma0) / sqrt(2)
  vec gg; mat ss;
  vec temp_vec; mat temp_mat; // for subetting purposes
  int rr; // incrementer
  mat A;
  int beg; int end; // incrementers

  
  for(int i = 0; i < n; ++i){
  
    W = B.t() * L.slice(i) * B * sigma + Psi * Delta.slice(i) * Psi * sigma_0;
    M = S * svec(W); // S * (1 * a + sqrt(2) * b...)
    // the matlab code is: M = C*svecmex(W);

    beg = 0;
    for(int r = 0; r < R; ++r){
      gg = lamb; // lamb is normally distributed random
      beg = beg + r; // 0:0, 1:2, 3:5
      end = beg + r;      
      
      pop_vec(gg, beg, end); // pop out these elements of gg
      ss = Cov.rows(beg, end);
      pop_col_mat(ss, beg, end);

      lamb.rows(beg, end) = mv_norm(M.rows(beg, end) - ss * gg,
      				    Cov(span(beg, end), span(beg, end)));      
    }

    vec_2_lowtri(A, lamb, R); // set A as lower trangular matrix from lamb
    Lambda.slice(i) = A + A.t() - diagmat(A.diag());
    
    
    // mu_g, s_g are used for update_Gamma    
    temp_mat = (X.row(i).t() * svec(Psi * Lambda.slice(i) * Psi).t() * D);
    temp_mat.reshape(p * R * (R + 1) / 2, 1);
    mu_g = mu_g + temp_mat;

    s_g = s_g + kron(D * skron(Psi, Psi) * D, X.row(i).t() * X.row(i));
  }
}

void update_Gamma(mat &Gamma, int R, int p, double sigma_0, mat sig_gam, vec mu_g,
		  mat s_g){
  mat S_gam;
  mat k_eye; k_eye.eye(p, p);
  mat gg; mat ss;
  mat mu_gam;
  vec temp_vec;

  
   S_gam = inv(sigma_0 * s_g + kron(sig_gam, k_eye));
   S_gam = trimatl(S_gam) + trimatl(S_gam).t() - diagmat(S_gam.diag());
   mu_gam = sigma_0 * S_gam * mu_g;
   
  for(int i = 1; i <= (R * (R + 1) / 2); ++i){    
    gg = Gamma;
    gg.reshape(p * R * (R + 1) / 2, 1);
    
    pop_row_mat(gg, p * (i - 1), p * (i - 1) + p - 1);
    ss = S_gam.rows(span(p * (i - 1), p * (i - 1) + p - 1));
    pop_col_mat(ss, p * (i - 1), p * (i - 1) + p - 1);

    Gamma.col(i - 1) = mv_norm(mu_gam.rows(p * (i - 1), p * (i - 1) + p - 1) - ss * gg,
    			   S_gam(span(p * (i - 1), p * (i - 1) + p - 1),
    				 span(p * (i - 1), p * (i - 1) + p - 1) ) );
  }
}


void calculate_Delta(cube &Delta, mat X, mat Gamma, int R, int n){

  rowvec AD_vec; mat AD_mat;

  for(int i = 0; i < n; ++i){
    AD_vec = X.row(i) * Gamma;
    vec_2_lowtri(AD_mat, AD_vec.t(), R);
    Delta.slice(i) = AD_mat + AD_mat.t() - diagmat(AD_mat.diag());
  }
}

mat LTI(vec a, mat A1, int i, int V){
  // Turn ith column of A1 into A, remaining lower triangular
  // Requires: A1 be size:

  A1(span(i, V - 1), span(i, i)) = a; // change ith column into a
  return A1;
}


double log_pdf_B(vec b1, mat B, int i, double sigma, mat lam_l, mat lam_s,
	      int range, int range_0, mat VV0, int R, int V, int n){
  
  // we are plugging in A from LTI
  vec pdf_in;
  mat A; mat A_vr; mat A_rr; vec A_r2;
  mat A_temp; mat VV0_temp; mat bt_VV0_b;
  mat ret; double ret_double;
  
  A = LTI(b1, B, i, V);
  A_vr = A; A_vr.reshape(V * R, 1);
  A_temp = A.t() * A;
  A_rr = A_temp; A_rr.reshape(pow(R, 2), 1);
  
  VV0_temp = .5 * b1.t() * VV0(span(range_0 - 1, range - 1),
   			       span(range_0 - 1, range - 1)) * b1;
  
  ret = 0.5 * n * V * (V + 1) * log(sigma) - 0.5 * sigma  *
    (-2.0 * A_vr.t() * lam_l * A_vr + A_rr.t() * lam_s * A_rr) - VV0_temp;

  ret_double = ret(0, 0);
  return ret_double;
}

void update_B(mat &B, cube Lambda, cube L, mat tau, mat phi, int R, int V,
	      mcmc_para parameters, int iter){
  // update B using adaptive metropolis
  // we update B one column at a time

  mat lam_s; lam_s.zeros(pow(R, 2), pow(R, 2));
  mat lam_l; lam_l.zeros(V * R, V * R);
  int n; n = Lambda.n_slices;
  mat JJ; mat VV0;
  int range_0 = 1; int range;
  mat rnd; mat BB00;
  int temp; vec temp_vec;

  mat dummy_mat;
  
  for(int i = 0; i < n; ++i){
    lam_s = lam_s + kron(Lambda.slice(i), Lambda.slice(i));
    lam_l = lam_l + kron(Lambda.slice(i), L.slice(i));
  }
  
  JJ = phi % repmat(tau.t(), V, 1);

  VV0.eye(JJ.n_elem - JJ.n_cols * (JJ.n_cols - 1) / 2,
	  JJ.n_elem - JJ.n_cols * (JJ.n_cols - 1) / 2);

  
  for(int i = 1; i <= R; ++i){
    range = i * V - i * (i - 1) / 2;

    
    if(iter <= parameters.get_slice_niter()){
      // call adaptive metropolis
      rnd = ARWM_B(parameters.get_slice_niter(), parameters.get_slice_burnin(),
      		   1, B(span(i - 1, V- 1), span(i - 1, i - 1)), B, i,
       		   parameters.get_sigma(), lam_l, lam_s, range,
      		   range_0, VV0, R, V, n);
    }else{
      // call it again
      rnd = ARWM_B(parameters.get_slice_niter(), parameters.get_slice_burnin(),
      		   1, B(span(i - 1, V- 1), span(i - 1, i - 1)), B, i,
       		   parameters.get_sigma(), lam_l, lam_s, range,
      		   range_0, VV0, R, V, n);
    }
    B(span(i - 1, V - 1), span(i- 1, i - 1)) = rnd; // update B at the end
    range_0 = range + 1;
  }
}


mat sk(mat b){
  // debugged
  int k = b.n_elem;
  int R = k * (k + 1) / 2;
  mat S; S.eye(R, R);
  int l = 0;
  for(int i = 0; i < k; ++i){
    for(int j = i; j < k; ++j){
      S(l, l) = b(i) * b(j); // but b is h x 1 matrix
      l = l + 1;
    } 
  }
  return S;
}


double log_pdf_Psi(vec b1, int n, mat D, double sigma_0, mat BMM, double v_a,
		   double v_b){
  // debugged

  double temp = 0;
  double ret;
  
  ret = 0.5 * n  * log(det(D * sk(b1) * D * sigma_0)) - 0.5 * sigma_0 *
    trace(sk(b1) * BMM) - v_b * sum(1 / b1) - (v_a + 1) * sum(log(b1));

  for(int i = 0; i < b1.n_elem; ++i){
    if(b1(i) <= 0){
      temp = temp + 10000000;
    }
  }

  ret = ret - temp;
  return ret;
}


void update_hyperparameters(mat &phi, vec &tau, mat &Psi, vec &lambb, vec &rndd,
			    mat D, cube Lambda, cube Delta, mat B,
			    double sigma_0, int V, int R, mcmc_para parameters){  
  vec temp_vec; vec temp_vec_1;
  double temp_double;
  mat BMM; int m; mat sl;
  vec temp_vec_2;
  bool flag = false;
  
  // update phi
  for(int i = 0; i < V; ++i){
    m = min(i, R);
    for(int j = 0; j < m; ++j){      
      phi(i,j) = n_inv_gaussian(1, sqrt(lambb(j) / pow(B(i, j), 2) / tau(j)), lambb(j))(0);
    }
  }

  // update tau
  for(int i = 0; i < R; ++i){
    
    temp_vec = B(span(i, V - 1), span(i, i)).t() *
      diagmat(phi(span(i, V - 1), span(i, i))) *
      B(span(i, V - 1), span(i, i));

    temp_double = 1 /(temp_vec(0) / 2.0);
    
    temp_vec_1 = n_gamma(1, (V - (i + 1) + 1) / 2.0, temp_double);
    tau(i) = temp_vec_1(0); // update tau with first element of output vector
  }

  // update lambb
  for(int i = 0; i < R; ++i){
    temp_vec = sum(1/ phi(span(i, V- 1), span(i, i)) / 2.0);
    lambb(i) = n_gamma(1, 1.0 / 2.0 + (V - i + 1),
		       1 /(temp_vec(0) + 1))(0);
  }

  // begin update Psi
  BMM.zeros(R * (R + 1) / 2, R * (R + 1) / 2);
  int n = Lambda.n_slices;
  for(int i = 0; i < n; ++i){
    sl = svec(Lambda.slice(i) - Delta.slice(i));
    BMM = BMM + sl * sl.t();
  }
  
  // update_Psi;  
  rndd = ARWM_Psi(50, 0, 1,
		  rndd, n, D, sigma_0, BMM, parameters.get_va(), parameters.get_vb());
  
  Psi = diagmat(rndd);
  cout << Psi << '\n';
}


double trace_1(arma::cube in){
  int n; n = in.n_slices;
  double ret; ret = 0;
  for(int i = 0; i < n; ++i){
    ret = ret + trace(in.slice(i));
  }
  return ret;
}


void sign_scale_adjustment(mat &B, cube &Lambda, mat &Gamma,
			   mat &aa_B, cube &a_Lambda, mat &a_Gamma,
			   mat &s_B, cube &s_Lambda, mat &s_Gamma,
			   cube &s_L,
			   double &sigma, mat &sig_gam, double &sigma_hat,
			   cube L, mat Psi, int p, int niter,
			   mcmc_para parameters){
  int V = B.n_rows;
  int R = B.n_cols;
  int n = Lambda.n_slices;

  int burn_in = parameters.get_burnin();
  double b1 = parameters.get_b1();
  double b2 = parameters.get_b2();
  double c1 = parameters.get_c1();
  double c2 = parameters.get_c2();
  
  // keeping a note of these below
  mat temp; mat temp_2;
  vec temp_vec;
  double temp_double;
  // init variables, there is a different line for each section
  mat sign_B; mat sign_BB; 
  cube n_L; n_L.zeros(V, V, n);
  // she does not use these in other functions
  
  cube Long; Long.zeros(V, V, n);
  cube LL; LL.zeros(V, V, n);
  double vv;
  mat ss_k;
  double dev_hat; // this should come from outside
  
  sign_B = repmat(sign(B(span(0, R - 1), span(0, R - 1)).diag()).t(), V, 1);
  sign_BB = diagmat(sign(B(span(0, R - 1), span(0, R - 1)).diag()));

  temp = sqrt(Psi);
  // these for loops can possibly all be a signle for loop
  for(int i = 0; i < n; ++i){
    a_Lambda.slice(i) = sign_BB * temp * Lambda.slice(i) *
      temp * sign_BB;
  }
  
  aa_B = sign_B % B * diagmat(1 / sqrt(Psi.diag()));

  temp = sign_BB * temp; // this is sign_BB * sqrtmat(Psi)
  a_Gamma = Gamma * skron(temp, temp);

  for(int i = 0; i < n; ++i){
    n_L.slice(i) = aa_B * a_Lambda.slice(i) * aa_B.t();
  }

  Long = L - n_L;

  for(int i = 0; i < n; ++i){
    LL.slice(i) = Long.slice(i) * Long.slice(i);
  }

  vv = trace_1(LL);

  temp_vec = n_gamma(1, b1 + n * V * (V + 1) / 4.0, 1 / (0.5 * vv + b2));
  sigma = temp_vec(0);

  ss_k = skron(Psi, Psi);

  // we do not have variable c1
  for(int i = 0; i < R * (R + 1) / 2; ++i){

    // re using temp_vec could cause unexpected behavior
    temp_vec = n_gamma(1, c1 + p * R * (R + 1) / 4.0,
		       1 / (0.5 * trace(Gamma.t() * Gamma) + c2));
    sig_gam(i, i) = temp_vec(0);
  }
  
  if(niter > burn_in){
    dev_hat = dev_hat - 2.0 * (-0.5 * sigma * vv -
			     log(1/sigma * 2.0 * 3.1415926535897) *
			     n * V * (V + 1) / 4.0);
    sigma_hat = sigma_hat + sigma;
  }
}



void write_to_file(mat B, cube Lambda, mat Gamma, cube Delta, string outf_name){

  int n = Lambda.n_slices;
  int R = Lambda.n_rows;
  mat temp;
  
  ofstream outf;
  
  // append to B outfile
  outf.open(outf_name + "_B.txt", std::ios::app);
  outf << B;
  outf.close();

  // append Gamma
  outf.open(outf_name + "_Gamma.txt", std::ios::app);
  outf << Gamma;
  outf.close();


  // reshape and apend vectorized Lambda
  outf.open(outf_name + "_Lambda.txt", std::ios::app);
  for(int i = 0; i < n; ++i){
    temp = Lambda.slice(i);
    temp.reshape(1, R * R);
    outf << temp;
  }
  outf << "end_iter\n";
  outf.close();

  outf.open(outf_name + "_Delta.txt", std::ios::app);
  for(int i = 0; i < n; ++i){
    temp = Delta.slice(i);
    temp.reshape(1, R * R);
    outf << temp;
  }
  outf << "end_iter\n";
  outf.close();
}



double reconstruction_error(cube Lambda, mat B, cube L, int n){
  double error = 0;
  int V = B.n_rows;
  mat temp; temp.zeros(V, V);
  
  for(int i = 0; i < n; ++i){
    temp = L.slice(i) - B * Lambda.slice(i) * B.t();
    error = error + norm(temp, "fro") / norm(L.slice(i), "fro");
  }
  error = error / double(n);
  cout << "Reconstruction Error\n";
  cout << error << '\n';
  return error;
}
