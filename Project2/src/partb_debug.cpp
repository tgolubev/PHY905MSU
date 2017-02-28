//Project2 part b, Debug version: The non-interacting case
//Performs Jacobi rotation to find the eigenvalues and eigenvectors of matrix which is formed from the discretization of the Schrodinger equation
//for 1 electron in a simple harmonic oscillator potential given by vector V.
//Eigenvalues are also calculated by armadillo's eig_sym function to ensure the accuracy of Jacobi rotation algorithm.
//CPU times for the two eigenvalue  solvers are compared.

//Unit tests:
//1. Function for finding the largest non-diagonal element is tested on a 5x5 matrix.
//2. Jacobi rotation algo result is tested with armadillo's eig_sym fnc.
//3. Preservation of orthogonality of eigenvectors is verified. NEED TO ADD THIS STILL

//Input: The program requires the desired name of the output file and the number of mesh points (n) to use for discretization.
//(i.e. n=200-300 works well for the simple harmonic oscillator potential). An example input would be "out 200".

//Output: The lowest several (# = num_eig) eigenvalues are output to the terminal. The ground state eigenvector (wavefunction(rho)) is output to the output file.

//Coded by: Tim Golubev, Hao Lin, Xingze Mao

//Include statements for header files
#include <iostream>
#include <fstream>
#include <iomanip>
#include <new>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <chrono>               //srand can work from here
#include <armadillo.h>

using namespace std;
using namespace arma;
using namespace std::chrono; //for high res clock

double offdiag(mat A, int& p, int& q, int n);                 //pass by ref p and q
void Jacobi_rotate ( mat& A, mat& R, int k, int l, int n );

// object for output files, for input files use ifstream
ofstream ofile;

int main(int argc, char *argv[])
{

  // Declarations
  double rho_max = 6.0;              //problem will be solved over range [0, rho_max]. If set too low (i.e. < 6.0) it overestimates eigvalues
  int num_eig = 5;                   //num of min eigvalues want to output
  int meshpts;
  string filename;

  //If in command-line has less than 1 argument, write out an error.
  if( argc <= 2 ){
       cout << "Bad Usage: " <<
            " read also file name on same line and number of mesh points n" << endl;
        exit(1);
   }
   else{
      filename = argv[1];
      meshpts = atoi(argv[2]);             //atoi: convert ascii input to integer. Input number of mesh points to use.
   }

  //Discretization
  double h = rho_max/((double) meshpts);   //define h=step size
  int n = meshpts -1;                      //We want to skip the endpoints. They are known from boundary conditions.
  double hh = h*h;
  vec rho(n);                              //rho values vector ("x"-values)
  vec V(n);                                //vector for the potential

  //Fill vectors rho and V
  for(int i=0; i<n; i++){
      rho(i) = (i+1)*h;                    //1st element of rho will = h. We exclude the endpoints rho(0)=0 and rho(rho_max).
      V(i) = (rho(i))*(rho(i));            //harmonic oscillator potential
  }

  //Define and fill matrix from discretization of Schrodinger eqn.
  mat A(n,n);
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < n; j++){
      if (i==j)
      {A(i,j) = 2.0/hh + V(i);
      }else if (abs(i-j)==1)
      {A(i,j) = -1.0/hh;
      }else {
          A(i,j) = 0.0;
      }
    }
  }
//-------------------------------------------------------------------------------------------------
  //Test offdiag fnc for finding largest-nondiagonal element
  int p, q;
  double maxnondiag;
  arma_rng::set_seed_random();           //set the random # gen seed to random value
  mat test = randu(5,5);
  cout << "test matrix = " <<endl;
  cout << test << endl;
  maxnondiag = offdiag(test, p, q, 5);
  cout << "maxnondiag element = " << maxnondiag << endl<<endl;

  //-----------------------------------------------------------------------------------------------
  // Testing with eig_sym  solver
  //start clock timer
  high_resolution_clock::time_point start1 = high_resolution_clock::now();
  vec eigval = eig_sym(A);
  //stop clock timer and output time duration
  high_resolution_clock::time_point finish1 = high_resolution_clock::now();
  duration<double> time1 = duration_cast<duration<double>>(finish1-start1);
  cout << "Armadillo eig_sym CPU time = " << time1.count() << endl;

  //Output the 5 lowest eigvalues from eig_sym solver
  vec lowest_eigval(num_eig);
  for (int i=0;i<num_eig;i++){
      lowest_eigval(i) = eigval(i);
  }
  cout<< lowest_eigval<<endl;
  //-----------------------------------------------------------------------------------------------

  //Jacobi rotation algorithm


  //Declarations for Jacobi rotation
  mat R = eye(n,n);       //initialize matrix for eigenvectors as the identity matrix
  double tolerance = 1.0E-10;
  int iterations = 0;
  maxnondiag = 1.0;    //initialize
  int maxiter = n*n; //anything <n^2 for maxiter gives results that don't match eig_sym solver

  //start clock timer
  high_resolution_clock::time_point start2 = high_resolution_clock::now();
  //Perform Jacobi rotations always acting on the largest (abs value) non-diag element
  srand((unsigned)time(0));     //seed the random number generator, will generate random index for orthonormality test
  vec random_eigvector1;
  vec random_eigvector2;
  int random_index1;
  int random_index2;
  double norm_test;
  double dot_product;
  while ( maxnondiag > tolerance && iterations <= maxiter)
  {
     maxnondiag  = offdiag(A, p, q, n);
     Jacobi_rotate(A, R, p, q, n);
     if (iterations % 10 ==0){                      //if remainder (modulus operator) iterations/10 = 0, check orthonormality
         random_index1 = rand()%n;                  //rand()%n finds remainder, ensuring random int will be always <n
         random_index2 = rand()%n;
         while (random_index1 == random_index2)     //if random indices are the same, reselect an index until they are different
         {random_index2 = rand()%n;
         }
         random_eigvector1 = R.col(random_index1);
         random_eigvector2 = R.col(random_index2);
         norm_test = norm(random_eigvector1);

         //norm_test = dot (random_eigvector,random_eigvector);
         //cout << random_index1 << endl;
         //cout<< random_index2 << endl;
         //cout.precision(17);
         //cout << "norm test = " <<norm_test <<endl;
         //cout.precision(17);
         //cout << "norm -1 = " << abs(norm_test -1.0) << endl;

         if (abs(norm_test-1.0) > 10E-15){             //because of limited precision, norm can return values slightly different from 1. Use 10E-15 tolerance
                                                       //for deviation from 1
             cout << "Normality test failed: " << "norm = " << norm_test << endl;
         }

         dot_product = dot(random_eigvector1, random_eigvector2);     //test orthogonality
         if (dot_product > 10E-15){
             cout << "Orthagonality test failed: " << "dot_product = " << dot_product << endl;
         }
     }
         iterations++;
  }
  cout << "Orthonormality tests passed" << endl<< endl;

  //stop clock timer and output time duration
  high_resolution_clock::time_point finish2 = high_resolution_clock::now();
  duration<double> time2 = duration_cast<duration<double>>(finish2-start2);
  cout << "Jacobi rotate CPU time = " << time2.count() << endl;

  //Vector with eigenvalues filled from diagonalized matrix A
  vec eig_values(n);
  for (int i=0; i<n; i++){
     eig_values(i) = A(i,i);
  }

  //Find and output lowest eigenvalues
  double max_value;
  int index;
  int groundstate_index;
  vec groundstate_eigvector;
  vec min_eig_values(num_eig);                               //vector to store min eigvalues
  max_value= max(eig_values);
  for (int i=0; i<num_eig; i++){
      index = index_min(eig_values);                         //find index of minimum eig_value
      min_eig_values(i) = eig_values(index);
      if (i==0){                                             //i=0 corresponds to ground state
          groundstate_index = index;                         //save ground state index
          groundstate_eigvector = R.col(groundstate_index);  //arma synthax: extract column vector from matrix R
      }
      eig_values(index) = max_value;                         //set the minimum value found to = max value in matrix
  }
  cout<<min_eig_values<<endl;

   //Setup output file
   ofile.open(filename);
   ofile << setiosflags(ios::showpoint | ios::uppercase);    //sets to write i.e. 10^6 as E6

   // Write to file:
   double bndry_value = 0.0;
   mat Output(n,2);  //Define matrix for output of rho and ground state eigenvector
   Output.col(0) = rho;
   Output.col(1) = groundstate_eigvector;
   ofile << "   Rho " << "         G.S. Wavefnc" <<endl;
   ofile << setprecision(8) << "   " <<bndry_value << "   " << bndry_value <<endl;
   ofile << setw(13) << setprecision(8) << Output;                         //width 13 ensures alignment when add endpts
   ofile << setprecision(8) <<rho_max << "   " << bndry_value <<endl;
   ofile.close();

  return 0;
}

//---------------------------------------------------------------------------------------------------------------------------
//Functions

//  the offdiag function, using Armadillo

double offdiag(mat A, int& p, int& q, int n)       //pass by referenc p and q, allows them to be changed by fnc.
{
double max = 0.0;
for (int i = 0; i < n; ++i)
{
 for ( int j = i+1; j < n; ++j)
 {
     double aij = fabs(A(i,j));
     if ( aij > max)
     {
        max = aij;  p = i; q = j;
     }
 }
}
return max;
}


void Jacobi_rotate ( mat& A, mat& R, int k, int l, int n )
{
double s, c;
if ( A(k,l) != 0.0 ) {
double t, tau;
tau = (A(l,l) - A(k,k))/(2*A(k,l));

if ( tau >= 0 ) {
t = 1.0/(tau + sqrt(1.0 + tau*tau));
} else {
t = -1.0/(-tau +sqrt(1.0 + tau*tau));
}

c = 1/sqrt(1+t*t);
s = c*t;
} else {
c = 1.0;
s = 0.0;
}
double a_kk, a_ll, a_ik, a_il, r_ik, r_il;
a_kk = A(k,k);
a_ll = A(l,l);
A(k,k) = c*c*a_kk - 2.0*c*s*A(k,l) + s*s*a_ll;
A(l,l) = s*s*a_kk + 2.0*c*s*A(k,l) + c*c*a_ll;
A(k,l) = 0.0;  // hard-coding non-diagonal elements by hand
A(l,k) = 0.0;  // same here
for ( int i = 0; i < n; i++ ) {
if ( i != k && i != l ) {
a_ik = A(i,k);
a_il = A(i,l);
A(i,k) = c*a_ik - s*a_il;
A(k,i) = A(i,k);
A(i,l) = c*a_il + s*a_ik;
A(l,i) = A(i,l);
}
//  And finally the new eigenvectors
r_ik = R(i,k);
r_il = R(i,l);

R(i,k) = c*r_ik - s*r_il;
R(i,l) = c*r_il + s*r_ik;
}
return;
} // end of function jacobi_rotate

