//Project2 part d Version 2: The Interacting Case
//Performs Jacobi rotation to find the eigenvalues of matrix which is formed from the discretization of the Schrodinger equation
//for 2 interacting electrons in a harmonic oscillator potential. The Coulomb interaction and potential they are in is described by
//vector V. The calculation can be done for different oscillator 'frequencies' w_r (describes strenth of the potential) specified
//by user input in the terminal window.

//Eigenvalues are also calculated by armadillo's eig_sym function to ensure the accuracy of Jacobi rotation algorithm.
//CPU times for the two eigenvalue  solvers are compared.

//Input: The program requires the desired name of the output file, number of mesh points (n) to use for discretization, w_r value, and
//rho_max value. Rho_max describes the maximum radius to which the discretization will be carried out. Note: If rho_max is not sufficiently
//large, the results will be incorrect! The necessary value of rho_max varries with w_r (larger for smaller w_r).
// An example input would be "out 150 0.5 10".

//Output: The lowest several (#=num_eig) eigenvalues are output to the terminal.
//The ground state eigenvector (wavefunction(r)) is output to the output file.

//Coded by: Tim Golubev, Hao Lin, Xingze Mao
//-----------------------------------------------------------------------------------------------------------------------------------------------

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

// objects for output files, for input files use ifstream
ofstream eigvec;
ofstream eigvalues;

int main(int argc, char *argv[])
{
  int meshpts;
  string eigvec_filename;      //string for filename
  string eigvalue_filename;
  double w_r;
  double rho_max;

  //If in command-line has insufficient number of arguments, write out an error.
  //argv[0] stores the path and .exe name that is being run
  if( argc <= 5 ){
       cout << "Bad Usage: "
            " Input must be: filename1 filename2 meshpts w_r rho_max" << endl
       <<"filname1 stores the g.s. eigenvector, filename2 stores the eigenvalue"
       << endl <<"meshpts = number of meshpts" << endl << "w_r = oscillator frequency" << endl
       << "rho_max = defines right endpt of rho" << endl;
        exit(1);
   }
   else{
      eigvec_filename = argv[1];
      eigvalue_filename = argv[2];
      meshpts = atoi(argv[3]);  //atoi: convert ascii input string to integer. Input number of mesh points to use.
      w_r = stod(argv[4]); //convert string to double
      rho_max = stod(argv[5]);
   }

  // Declarations
  //double rho_max = 50.0;  //problem will be solved over range [0, rho_max]. If set too low (i.e. < 6.0) it overestimates eigvalues
  double alpha = 1; //set alpha=1
  int num_eig = 3;   //num of min eigvalues want to output

  double h = rho_max/((double) meshpts);  //define h=step size
  double hh = h*h;
  int n = meshpts -1;    //we skip the endpts
  vec rho(n);            //rho values vector ("x"-values)
  vec V(n);              //vector for the potential

  //Fill vectors rho and V
  for(int i=0; i<n; i++){
      rho(i) = (i+1)*h;         //1st element of rho will = h. We exclude the endpoints rho(0)=0 and rho(rho_max).
      V(i) = rho(i)*rho(i)*w_r*w_r + 1/rho(i);  //harmonic oscillator potential with repulsive Coulomb interaction btw. 2 electrons
      //V(i) = rho(i)*rho(i)*w_r*w_r; //harmonic oscillator potential WITHOUT Coulomb interaction FOR TESTING
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

  mat R = eye(n,n);       //initialize matrix for eigenvectors as the identity matrix
  double tolerance = 1.0E-10;
  int iterations = 0;
  double maxnondiag = 1.0;    //initialize
  int maxiter = n*n; //anything <n^2 for maxiter gives results that don't match eig_sym solver

  //start clock timer
  high_resolution_clock::time_point start2 = high_resolution_clock::now();
  //Perform Jacobi rotations always acting on the largest (abs value) non-diag element
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

  //Vector with eigenvalues read off of diagonal elements of A
  vec eig_values(n);
  for (int i=0; i<n; i++){
     eig_values(i) = A(i,i);
  }

   //Find and output lowest eigenvalues
   double max_value;
   int index;
   int groundstate_index;
   vec groundstate_eigvector;
   vec min_eig_values(num_eig);   //vector to store min eigvalues
   max_value= max(eig_values);
   for (int i=0; i<num_eig; i++){
       index = index_min(eig_values); //find index of minimum eig_value
       min_eig_values(i) = eig_values(index);
       if (i==0){
           groundstate_index = index;  //save ground state index (will correspond to column in R where it occurs) to use for groundstate eigvector output
           groundstate_eigvector = R.col(groundstate_index);  //arma synthax: extract column vector from matrix R
       }
       eig_values(index) = max_value;  //set the minimum value found to = max value in matrix

   }
   cout<<min_eig_values<<endl;

   //Setup output file
   eigvec.open(eigvec_filename);         //after initial opening of the output file, file is referred to by the ofstream object name.
   eigvec << setiosflags(ios::showpoint | ios::uppercase); //sets to write i.e. 10^6 as E6
   // Write to file:
   double bndry_value = 0.0;
   mat Output(n,2);  //Define matrix for output of rho, ground state eigvector, and lowest eigvalues
   Output.col(0) = rho*alpha;  //fill 1st column with r=rho*alpha values
   Output.col(1) = groundstate_eigvector;
   eigvec << "w_r = " << w_r<< "  meshpts = " << meshpts << endl;
   eigvec << "   Rel. Coord." << "  G.S. Wavefnc" << endl;
   eigvec << setprecision(8) << "   " <<bndry_value << "   " << bndry_value <<endl;
   eigvec << setw(13) << setprecision(8) << Output;
   eigvec << setprecision(8) <<rho_max << "   " << bndry_value <<endl;
   eigvec.close();

   eigvalues.open(eigvalue_filename);
   eigvalues << setiosflags(ios::showpoint | ios::uppercase); //sets to write i.e. 10^6 as E6
   eigvalues << "w_r = " << w_r<< "  meshpts = " << meshpts << "  rho_max = " << rho_max << endl;
   eigvalues << "Lowest eigenvalues = " << endl;
   eigvalues << setprecision(8) << min_eig_values << endl;
   eigvalues.close();

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

