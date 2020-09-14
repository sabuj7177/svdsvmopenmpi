#include <iostream>
#include <armadillo>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <algorithm>
#include <queue>
#include "mpi_helper.h"
#include <boost/lexical_cast.hpp>
#include"arma_mpi.h"
#include <chrono>
#include <cstdlib>
#include <sstream>
#include <string>

using namespace arma;
using namespace std;

double sqr(const double &t)
{
    return t*t;
}
double sign(double x)
{
    if (x > 0.0) return 1.0;
    if (x < 0.0) return -1.0;
    return 0;
}
// Routine to initialized time measurement
//
std::chrono::time_point<std::chrono::steady_clock> time_tic()
{
    return std::chrono::steady_clock::now();
}
// Routine to display time elapsed from the time "start" was initialized
//
void time_toc(std::string s, std::chrono::time_point<std::chrono::steady_clock> start)
{
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>
	(std::chrono::steady_clock::now() - start);
    std::cout.precision(4);
    std::cout<<".. Time ("<<s<<"): "<<std::fixed<<std::real(duration.count()/1e6)<<" seconds"<<std::endl;
}

float time_duration(std::chrono::time_point<std::chrono::steady_clock> start)
{
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>
	(std::chrono::steady_clock::now() - start);
    return std::real(duration.count()/1e6);
}
// Routine to compute Qt * x, where Q is stored in the form of
// Householder vectors in reflectors
//
void Qtx(arma::vec& x, arma::mat& reflectors)
{
    int n = x.n_rows;
    for(int i = 0;i<reflectors.n_cols;i++)
    {
	x(arma::span(i,n-1),0) = x(arma::span(i,n-1),0)
	    - 2*reflectors(arma::span(i,n-1),i)
	    *(reflectors(arma::span(i,n-1),i).t()*x(arma::span(i,n-1),0));
    }
}
// Routine to compute Q * x, where Q is stored in the form of
// Householder vectors in reflectors
//
void Qx(arma::vec& x,arma::mat& reflectors)
{
    int n = x.n_rows;
    int d = reflectors.n_cols;
    for(int i = d-1;i>-1;i--)
    {
	x(arma::span(i,n-1),0) = x(arma::span(i,n-1),0)
	    - 2*reflectors(arma::span(i,n-1),i)
	    *(reflectors(arma::span(i,n-1),i).t()*x(arma::span(i,n-1),0));
    }
}
// -----------------------------------------------------------------------------
// Routine to compute Qt * x, where Q = Q1 * Q2 such that Q1 is distributed
// across ALL processes while Q2 is stored on process 0 only. Further, Q1 and Q2
// are stored in the form of Householder vectors in v and vI, respectively
// buffer1, buffer2, and buffer_vec are work vectors used for MPI calls
//
void dist_Qtx(arma::vec& x, arma::mat& vI, arma::mat& v , int _nodes, int d, int rank, int count,
	double* buffer1, double* buffer2, arma::vec& buffer_vec)
{
    // Compute local component of Qt * x at each process
    Qtx( x, v );

    // MPI_Gather from ALL processes to Process 0
    for (int i = 0; i < d; i++) buffer2[i] = x(i);
    auto start_gather_Qtx = time_tic();
    MPI_Gather(buffer2, d, MPI_DOUBLE, buffer1, d, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if ((rank == 0) && (count%50==1)) {
            std::cout.precision(3);
            cout << "\t ";
            time_toc("gather_distQtx", start_gather_Qtx);
    }

    // On Process 0 only:
    // - Compute global component of Qt * x
    // - Initialize buffer for scatter after Qt * x has been computed
    if (rank == 0) {
	for (int i = 0; i < d * _nodes; i++) buffer_vec(i) = buffer1[i];
	Qtx( buffer_vec, vI );
	for (int i = 0; i < d * _nodes; i++) buffer1[i] = buffer_vec(i);
    }

    // MPI_Scatter global component Qt * x from Process 0 to ALL processes
    auto start_scatter_Qtx = time_tic();
    MPI_Scatter(buffer1, d, MPI_DOUBLE, buffer2, d, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if ((rank == 0) && (count%50==1)) {
            std::cout.precision(3);
            cout << "\t ";
            time_toc("scatter_distQtx", start_scatter_Qtx);
    }


    for (int i = 0; i < d; i++) x(i) = buffer2[i];
}
// Routine to compute Q * x, where Q = Q1 * Q2 such that Q1 is distributed
// across ALL processes while Q2 is stored on process 0 only. Further, Q1 and Q2
// are stored in the form of Householder vectors in v and vI, respectively
// buffer1, buffer2, and buffer_vec are work vectors used for MPI calls
//
void dist_Qx(arma::vec& x, arma::mat& vI, arma::mat& v , int _nodes, int d, int rank, int count,
	double* buffer1, double* buffer2, arma::vec& buffer_vec)
{
    // MPI_Gather from ALL processes to Process 0
    for (int i = 0; i < d; i++) buffer2[i] = x(i);
    auto start_gather_Qx = time_tic();
    MPI_Gather(buffer2, d, MPI_DOUBLE, buffer1, d, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if ((rank == 0) && (count%50==1)) {
            std::cout.precision(3);
            cout << "    ... Iteration: " << setw(4)<<count<< "\n\t ";
            time_toc("gather_distQx", start_gather_Qx);
    }

    // On Process 0 only:
    // - Compute global component of Q * x
    // - Initialize buffer for scatter after Q * x has been computed
    if (rank == 0) {
	for (int i = 0; i < d * _nodes; i++) buffer_vec(i) = buffer1[i];
	Qx( buffer_vec, vI);
	for (int i = 0; i < d * _nodes; i++) buffer1[i] = buffer_vec(i);
    }

    // MPI_Scatter global component Q * x from Process 0 to ALL processes
    auto start_scatter_Qx = time_tic();
    MPI_Scatter(buffer1, d, MPI_DOUBLE, buffer2, d, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if ((rank == 0) && (count%50==1)) {
            std::cout.precision(3);
            cout << "\t ";
            time_toc("scatter_distQx", start_scatter_Qx);
    }

    for (int i = 0; i < d; i++) x(i) = buffer2[i];

    // Compute local component of Q * x at each process
    Qx(x, v);
}
// -----------------------------------------------------------------------------

void qr_solver(const arma::mat& z, arma::mat& v, arma::mat& r, const int& n, const int&d)
{
    arma::mat z_transform = z;
    //float bc_duration = 0;
    //float abc_duration = 0;
    //float reflector_duration = 0;
    for(int i = 0; i<d; i++)
    {
	double norm_value = norm(z_transform(arma::span(i,n-1),i));
	v(arma::span(i,n-1),i) = z_transform(arma::span(i,n-1),i);
	v.at(i,i) = v.at(i,i) + ((z_transform.at(i,i)<0)?-1:1) * norm_value * 1;
	v(arma::span(i,n-1),i) = v(arma::span(i,n-1),i)/arma::as_scalar(norm(v(span(i,n-1),i)));

	//if(i<5){
    //    mat a = v(arma::span(i,n-1),i);
    //    mat b = v(arma::span(i,n-1),i).t();
    //    mat c = z_transform(arma::span(i,n-1),arma::span(i,d-1));
    //    auto bc = time_tic();
    //    mat mmm = b * c;
    //    bc_duration += std::chrono::duration_cast<std::chrono::microseconds>
	//(std::chrono::steady_clock::now() - bc).count()/1e6;
    //    auto abc = time_tic();
    //    mat nnn = a * b * c;
    //  abc_duration += std::chrono::duration_cast<std::chrono::microseconds>
	//(std::chrono::steady_clock::now() - abc).count()/1e6;
        //time_toc("reflector calculation time", start_reflector);
    //}
    //duration += time_duration(start_reflector);

    //auto start_reflector = time_tic();
	z_transform(arma::span(i,n-1),arma::span(i,d-1))
	    = z_transform(arma::span(i,n-1),arma::span(i,d-1))
	    - 2.0 * v(arma::span(i,n-1),i)
	    * v(arma::span(i,n-1),i).t() * z_transform(arma::span(i,n-1),arma::span(i,d-1));
    //reflector_duration += std::chrono::duration_cast<std::chrono::microseconds>
	//(std::chrono::steady_clock::now() - start_reflector).count()/1e6;

	for ( int j=i ; j<n ; j++)
	    for(int k=i ; k<d ; k++)
	    {
		if( abs(z_transform(j,k)) < 0.0001)
		    z_transform(j,k) = 0.0;
	    }
    }
    r = z_transform(arma::span(0,d-1),arma::span(0,d-1));
    //std::cout.precision(4);
    //std::cout<<".. Time (bc calculation time): "<<std::fixed<<bc_duration/5.0<<" seconds"<<std::endl;
    //std::cout<<".. Time (abc calculation time): "<<std::fixed<<abc_duration/5.0<<" seconds"<<std::endl;
    //std::cout<<".. Time (Per reflector calculation time): "<<std::fixed<<(reflector_duration*1.0)/d<<" seconds"<<std::endl;
    //std::cout<<".. Time (Total reflector calculation time): "<<std::fixed<<reflector_duration<<" seconds"<<std::endl;
}

void print_matrix(const arma:: mat& m, int startRow, int endRow)
{
    if(startRow==-1)
	startRow=0;
    if(endRow==-1)
	endRow=m.n_rows;
    for(int i=startRow;i<endRow;i++){
	for(int j=0;j<m.n_cols;j++){
	    cout<<m(i,j)<<" ";
	}
	cout<<endl;
    }
}

void print_vector(const arma:: vec& v, int startRow, int endRow)
{
    cout<<v.n_rows<<endl;
    if(startRow==-1)
	startRow=0;
    if(endRow==-1)
	endRow=v.n_rows;
    for(int i=startRow;i<endRow;i++){
	cout<<v(i,0)<<" ";
	cout<<endl;
    }
}

int main( int argc, char *argv[])
{

    // MPI Initializations
    MPI_helper* mpi = new MPI_helper();
    int _nodes = mpi->getSize();
    int _rank = mpi->getRank();

    // Input parameters

    double C, learnRate, thresh, gamma;
    C = atof(argv[1]);
    learnRate = atof(argv[2]);
    thresh = atof(argv[3]);

    // Actual streamQRSVM code
    int repeat_step = 10;
    //double C =1, learnRate = 0.02, thresh = 0.00001;

    auto start_load = time_tic();

    // Read Data file
    string fname_Dataset = "./partitions/data_part_" + std::to_string(_rank) + ".csv";
    ifstream read_Dataset(fname_Dataset.c_str());

    //load dataset
    int _n_PerNode, d;	//

    mat dataset,X;
    vec label;
    if (read_Dataset.is_open())
    {
	dataset.load(fname_Dataset,arma::csv_ascii);

	_n_PerNode = dataset.n_rows; // sample size
	d = dataset.n_cols - 1; // d: dimensionality , last column is class label

	// MUST ensure number of processes less than n/d, i.e., _nodes < n/d
	// Alternatively, each proc has at least d rows, i.e., d < _n_PerNode (=n/np)
	cout<<". Node " << _rank << ": _n_PerNode = " << _n_PerNode <<", d = "<< d;
	if (_n_PerNode <= d) {
	    cout << "; _n_PerNode < d ... Aborting" << endl;
	    exit(0);
	} else {
	    cout << "; d <= _n_PerNode ... No problem" << endl;
	}

	X = dataset( span::all , span(0,d-1) ); //X( span(first_row, last_row), span(first_col, last_col) )
	label = dataset( span::all , d );  // last column in dataset is for class label and has column index (d+1)-1=d
    }
    else
    {
	cout << "Unable to open file or file does not exist"<<endl;
	exit(1);
    }

    read_Dataset.close();

    // remove dataset
    dataset.shed_cols(0,d);
    dataset.shed_rows(0,_n_PerNode-1);

    // Synchronize all processes after reading inout in parallel (not needed)
    MPI_Barrier(MPI_COMM_WORLD);
    if (_rank ==0)
    {
	time_toc("loading partitioned dataset", start_load);
    }

    // Create Dy and X_hat
    auto start_Traintime = time_tic();

    vec x(_n_PerNode);
    X = join_horiz(X, x.ones(_n_PerNode)); //insert vector of 1s
    d = d+1; // dimensionality of X is now d+1 with insertion of Ones

    mat X_hat = zeros<mat>(_n_PerNode,d);
    for (int i=0; i<_n_PerNode; i++)
    {
	X_hat(i, span::all) = label(i)*X(i, span::all); //_n_perNodexd TS matrix
    }

    int qr_comm = 0;
    int da_comm = 0;

    // -------------------------------------------------------------------------
    // Initialize matrices for distributed QR
    // Compute X_hat = QR, where Q = Q1 * Q2
    // QR computed in two steps, a local one on each process
    // followed by global one on Process 0
    //   Local: compute X_hat = Q1 * R1 (R1 consists of _nodes number of
    //          stacked dxd upper triangular matrices arising from
    //          local QR calculations)
    //   Global: compute R1 = Q2 * R (QR calculation on Process 0)
    //
    // v: Householder vectors for local part of Q1 on each process
    // r: Local part of R1 on each process
    // rCombined: R1
    // vI: Householder vectors for Q2
    // rF: R
    //
    mat v = zeros<mat>(_n_PerNode,d);	// initialize on ALL processes
    mat r = zeros<mat>(d,d);		// initialize on ALL processes
    mat rCombined;			// initialize on Process 0 only,
    					// within MPI_Gather_matrix routine
    mat vI, rF; 			// initialize on Process 0 only

    mat D, rrt, L_D, U_D; 		// initialize on Process 0 only

    // Global variables used in MPI gather and scatter operations in
    // routines dist_Qx and dist_Qtx
    vec buffer_vec;			// initialize on Process 0 only
    double *buffer1 = nullptr; 		// initialize on Process 0 only
    double *buffer2 = new double[d]; 	// initialize on ALL processes

    // Allocate Process 0 variables
    auto startTotalQR = time_tic();
    if (_rank == 0) {
	vI = zeros<mat>(_nodes*d,d);
	rF = zeros<mat>(d,d);
	buffer1 = new double[_nodes * d];
	buffer_vec = zeros<vec>(d * _nodes);

    }
    // -------------------------------------------------------------------------

    // QR using Householder: X_hat = v * r
    auto startQR = time_tic();
    qr_solver( X_hat, v, r, _n_PerNode, d); // truncates r to dxd
    time_toc("QR on X_hat at node "+ std::to_string(_rank) , startQR);

    // remove X_hat. Use X and label for finding b later
    X_hat.shed_cols(0,d-1);
    X_hat.shed_rows(0, X_hat.n_rows-1);

    //  ***** Communication 1: All r's to be gathered in master to form (_nodes*d)xd rCombined *****
    auto startComm1 = time_tic();

    MPI_Gather_matrix(r, rCombined, d, d, MPI_DOUBLE, MPI_COMM_WORLD, _nodes, 0, _rank);
    qr_comm += (_nodes-1)*d*d;

    if (_rank == 0)
	time_toc("Gathering r at Master", startComm1);


    // Compute QR of rCombined using Householder
    if ( _rank == 0)
    {

	auto startQR_rCombined = time_tic();
	qr_solver( rCombined , vI, rF, _nodes*d, d); 	// automatically truncated rF to dxd
	time_toc("QR on rCombined at Master", startQR_rCombined );

	rCombined.shed_cols(0,(rCombined.n_cols-1));
	rCombined.shed_rows(0, (rCombined.n_rows-1));

	// Creating M1= rrt i.e. 1st block partition of RRt+diag(1/2C) . It is ensured: d < _n_perNode i.e. _nodes < n/d
	rrt.zeros(d,d);
	rrt = rF* rF.t(); //dense

	// For Master Node, coefficient matrix
	D.eye(d,d);
	D = (0.5/C) * D;
	D = rrt + D;

	// LU factorization of rrt
	lu( L_D, U_D, D );
	time_toc("Total qr time ", startTotalQR );
    }

    // ***** Distributed Dual Ascent *****

    vec e_svm(_n_PerNode), alpha_hat(_n_PerNode), dual_hat(_n_PerNode), dual_prev_hat(_n_PerNode), B(_n_PerNode);
    dual_hat.fill(1);
    dual_prev_hat.fill(0);

    e_svm.fill(-1);
    // Compute Qt * e
    dist_Qtx(e_svm, vI, v, _nodes, d, _rank, -1, buffer1, buffer2, buffer_vec);
    da_comm += (_nodes-1)*d*2;

    int count = 0;
    double optP, optStepSize;
    optP =  floor(1/(learnRate*C)) - 0.5; // For safety, subtract 0.5
    optStepSize= optP*learnRate;

    if (_rank == 0)
	cout<<endl<<". opt P: " <<optP<<" and optStepSize: "<<optStepSize<<endl<<endl;

    double error = 1, errorLocal;
    auto startDA = time_tic();
    uvec countNeg;
    mat invL_B;
    while( error > thresh  ) // alternatively can try: error > (thresh/_n_PerNode)
    {


        count++; // number of iterations

	dual_prev_hat = dual_hat;

	//Update 1: alpha_hat_i = -(RRt+Dc)_i \ (- (Q'*dual_prev)_i + (Q'*e_svm)_i );
	//Substitute: dual_prev_hat= Q'*dual_prev), e_svm = Q'*e_svm
	B = -dual_prev_hat + e_svm ;

	if (_rank == 0)
	{
	    //Subproblem 1
	    // using X = solve( A, B )  where X = U\L\B in MATLAB and A = LU
	    invL_B = arma::solve( L_D, B(span(0,d-1)), solve_opts::fast );
	    alpha_hat(span(0,d-1)) = arma::solve(-U_D, invL_B, solve_opts::fast );
	    // Subproblem 2
	    alpha_hat(span(d,_n_PerNode-1)) = -2*C*B(span(d,_n_PerNode-1));

	}
	else
	{
	    alpha_hat = -2*C*B;
	}

	// Update 2:  (Q'*dual)_i = (Q'*dual_prev)_i - learnRate*optP*(alpha_hat_i);
	dual_hat = dual_prev_hat - (optStepSize * alpha_hat) ;

	// Each element of dual needs to be >=0
    auto start_iteration = time_tic();
	dist_Qx(dual_hat, vI, v, _nodes, d, _rank, count, buffer1, buffer2, buffer_vec);
	if(_rank==0){
	    cout << "Iteration "<<count<<".. error=  "<<std::scientific<< error << endl<<endl;

	}
	//time_toc("iter", start_iteration);
	da_comm += (_nodes-1)*d*2;

	countNeg = find(dual_hat<0.0);

	dual_hat.elem( find(dual_hat < 0.0) ).zeros();

	dist_Qtx(dual_hat, vI, v, _nodes, d, _rank, count, buffer1, buffer2, buffer_vec);
	da_comm += (_nodes-1)*d*2;

	errorLocal = norm(dual_hat - dual_prev_hat,1);
	MPI_Allreduce(&errorLocal, &error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	//if ((_rank == 0) && (count%50==1)) {
	//    std::cout.precision(3);
	//    cout<<"\t ";
	//    time_toc("iter", start_iteration);
	//    cout << "\t "<<".. error=  "<<std::scientific<< error << endl<<endl;
	//}

    }
    // end of dual ascent solver

    auto DAduration = std::chrono::duration_cast<std::chrono::milliseconds> (std::chrono::steady_clock::now() - startDA);
    if(_rank == 0)
	time_toc("DA solver on processes: "+ std::to_string(_rank), startDA);

    // Find Support Vectors
    dist_Qx(alpha_hat, vI, v, _nodes, d, _rank, count, buffer1, buffer2, buffer_vec);

    vec sv1;
    uvec svIndex1 = find(alpha_hat > 0.001);
    sv1 = alpha_hat.elem(svIndex1);
    int svCount1 = svIndex1.n_rows, svCount_global;
    mat Xk = zeros<mat>(1,X.n_cols);

    auto Trainduration = std::chrono::duration_cast<std::chrono::milliseconds> (std::chrono::steady_clock::now() - start_Traintime);
    if(_rank == 0)
    {
	time_toc("Train time on processes: "+ std::to_string(_rank), start_Traintime);
	cout<<endl<<". Iterations: "<<setw(4)<<count;
	std::cout.precision(3);
	cout<<": Global dual_hat error:"<<std::scientific<<error;
	Xk = X(svIndex1(0), span::all);
    }

    // Calculate Bias
    double biaslocal=0, bias;
    MPI_Bcast_matrix(Xk, Xk.n_rows, Xk.n_cols, MPI_DOUBLE, MPI_COMM_WORLD, 0, _rank);
    double K_sv;
    int index1;
    mat Xnode = zeros<mat>(1, X.n_cols);
    for(int i=0; i<svCount1; i++)
    {
	index1 = svIndex1(i);
	Xnode = X(index1, span::all);
	K_sv =  as_scalar(Xnode* Xk.t());
	biaslocal = biaslocal +  ( sv1(i) * label(index1) * K_sv );
    }

    MPI_Allreduce(&biaslocal, &bias, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (_rank == 0)
    {
	bias = label(svIndex1(0)) - bias;
	cout<<": Bias: "<<bias<<endl;
    }

    // Final results for processing
    if (_rank == 0)
    {
	cout<< endl;
	cout<<"Results: n, nprocs, n per node, d, iters, bias, time: DA, train (sec)= ";
	cout<< setw(8) << _n_PerNode * _nodes << setw(4) << _nodes;
	cout<< setw(8) << _n_PerNode << setw(8) << d << setw(6) << count;
        std::cout.precision(2);
	cout<< " " << std::fixed <<bias << " " << std::real(DAduration.count())/1000.0 << " " << std::real(Trainduration.count())/1000.0;
	cout<< endl<<endl<<"***** End *****"<<endl<<endl;
	cout<<"Total qr communication "<<qr_comm<<endl;
    cout<<"Total dual ascent communication "<<da_comm<<endl;
    cout<<"Total communication "<<qr_comm+da_comm<<endl;
    }

    MPI_Finalize();
    return 0;
}
