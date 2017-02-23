//Project2 part b.  Performs Jacobi rotation to find the eigenvalues of matrix which is formed from the
//discretization of the Schrodinger equation of 1 electron in a potential given by vector V.
//Also eigenvalues are calculated by armadillo's eig_sym for comparison.
//Input: From the command prompt the number of mesh points (n) to use for discretization. (i.e. n=200-300 works well
//for the simple harmonic oscillator potential)

//Include statements for header files
#include <iostream>
#include <fstream>
#include <iomanip>
#include <new>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>    //  Mathematical functions like sin, cos etc
#include <string>   //  useful library to operate on characters
#include <chrono>  //for high resolution clock
#include <armadillo.h>

using namespace std;
using namespace arma;
using namespace std::chrono; //for high res clock

double offdiag(mat A, int& p, int& q, int n);   //pass by ref p and q
void Jacobi_rotate ( mat& A, mat& R, int k, int l, int n );

int main(int argc, char *argv[])
{
  int n;
  //If in command-line has less than 1 argument, write out an error.
  if( argc <= 1 ){
       cout << "Bad Usage: " << argv[0] <<
            " read also file name on same line and max power 10^n" << endl;
        exit(1);
   }
   else{
      n = atoi(argv[1]);  //atoi: convert ascii input to integer. Input number of mesh points to use.
   }

  //   Declare Variables, Pointers, and Arrays
  double rho_max = 6.0;  //problem will be solved over range [0, rho_max]. If set too low (i.e. < 6.0) it overestimates eigvalues
  double h = rho_max/((double) n);  //define h=step size
  double hh = h*h;
  vec rho(n);            //rho values vector ("x"-values)
  vec V(n);              //vector for the potential
  int num_eig = 7;   //num of min eigvalues want to output

  //Fill vectors rho and V
  for(int i=0; i<n; i++){
      rho(i) = (i+1)*h;         //1st element of rho will = h. We exclude the endpoints rho(0)=0 and rho(rho_max)=0.
      V(i) = (rho(i))*(rho(i));  //harmonic oscillator potential
  }

  //Define matrix
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

  //Define matrix for eigenvectors
  mat R = eye(n,n);       //initialize as the identity matrix

  //start clock timer
  high_resolution_clock::time_point start2 = high_resolution_clock::now();

  //Perform Jacobi rotations always acting on the largest (abs value) non-diag element
  double tolerance = 1.0E-10;
  int iterations = 0;
  double maxnondiag = 1.0;    //initialize
  int maxiter = n*n; //anything <n^2 for maxiter gives results that don't match eig_sym solver
  while ( maxnondiag > tolerance && iterations <= maxiter)
  {
     int p, q;
     maxnondiag  = offdiag(A, p, q, n);
     Jacobi_rotate(A, R, p, q, n);
     iterations++;
  }

  //stop clock timer and output time duration
  high_resolution_clock::time_point finish2 = high_resolution_clock::now();
  duration<double> time2 = duration_cast<duration<double>>(finish2-start2);
  cout << "Jacobi rotate CPU time = " << time2.count() << endl;

  //cout<< maxnondiag << endl;

  //Find and output lowest 5 eigenvalues
  vec eig_values(n);      //vector with eigenvalues read off of diagonal elements of A
  for (int i=0; i<n; i++){
     eig_values(i) = A(i,i);
  }
   double max_value;

   int index;
   vec min_eig_values(num_eig);   //vector to store min eigvalues
   max_value= max(eig_values);
   for (int i=0; i<num_eig; i++){
       min_eig_values(i) = min(eig_values);
       index = index_min(eig_values); //find index of minimum eig_value
       eig_values(index) = max_value;  //set the minimum value found to = max value in matrix
   }
   cout<<min_eig_values<<endl;
  return 0;
}


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

