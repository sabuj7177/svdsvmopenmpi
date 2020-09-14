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

void shedMatrix(arma::mat& m){
    m.shed_cols(0,m.n_cols-1);
    m.shed_rows(0,m.n_rows-1);
}

void read_test_data(arma:: mat& m, string dataset_name)
{
    string line;
    ifstream myfile (dataset_name+"_test.txt");
    string delimiter = " ";
    size_t pos = 0;
    string token;
    int nCols = m.n_cols-1;
    if (myfile.is_open())
    {
        int i = 0;
        while ( getline (myfile,line) )
        {
            for(int j=0;j<nCols-1;j++){
                pos = line.find(delimiter);
                token = line.substr(0, pos);
                m(i,j) = stod(token);
                line.erase(0, pos + delimiter.length());
            }
            m(i,nCols-1) = stod(line);
            m(i,nCols) = 1.0;
            i++;
        }
        myfile.close();
    }
}

void read_test_label(arma:: mat& m, string dataset_name)
{
    string line;
    ifstream myfile (dataset_name+"_test_label.txt");
    string delimiter = " ";
    size_t pos = 0;
    string token;
    int nCols = m.n_cols;
    if (myfile.is_open())
    {
        int i = 0;
        while ( getline (myfile,line) )
        {
            for(int j=0;j<nCols-1;j++){
                pos = line.find(delimiter);
                token = line.substr(0, pos);
                m(i,j) = stod(token);
                line.erase(0, pos + delimiter.length());
            }
            m(i,nCols-1) = stod(line);
            i++;
        }
        myfile.close();
    }
}

void sumUpMatrix(const arma:: mat& mainMat, arma::mat& combinedMat, const int& d, const int& _nodes)
{
    combinedMat.fill(0);
    for(int it=0;it<_nodes;it++){
        mat temp = mainMat(arma::span(it*d,(it+1)*d-1),span::all);
        combinedMat = combinedMat + temp;
    }
}

void getYUsingRandomMatrix(const arma:: mat& X_hat, arma::mat& y, const int& seed, const int& d, const int& targetRank){
    arma::arma_rng::set_seed(seed);
    mat randomMat = arma::mat(d, targetRank, arma::fill::randu);
    y = X_hat * randomMat;
    shedMatrix(randomMat);
}

void getQ(const arma:: mat& y, arma:: mat& qPart, const int& targetRank, const int& _nodes, const int& _rank, int& svd_comm){
    mat q,r;
    mat qCombined, rCombined;
    mat rGather;
    mat rInverse = zeros<mat>(targetRank, targetRank);
    if(_rank == 0){
        rGather = zeros<mat>(_nodes*targetRank,targetRank);
    }
    qr_econ(q,r,y);
    shedMatrix(q);
    MPI_Gather_matrix(r, rGather, targetRank, targetRank, MPI_DOUBLE, MPI_COMM_WORLD, _nodes, 0, _rank);
    svd_comm += (_nodes-1)*targetRank*targetRank;
    if(_rank == 0)
    {
        qr_econ(qCombined, rCombined, rGather);
        rInverse = inv(rCombined);
        shedMatrix(qCombined);
        shedMatrix(rCombined);
    }
    MPI_Bcast_matrix(rInverse, rInverse.n_rows, rInverse.n_cols, MPI_DOUBLE, MPI_COMM_WORLD, 0, _rank);
    svd_comm += (_nodes-1)*targetRank*targetRank;

    qPart = y * rInverse;
}

void getQ2(const arma:: mat& y, arma:: mat& qPart, const int& targetRank, const int& _nodes, const int& _rank, int& svd_comm){
    mat q,r;
    mat qCombined, rCombined;
    mat rGather;
    mat rInverse = zeros<mat>(targetRank, targetRank);
    if(_rank == 0){
        rGather = zeros<mat>(_nodes*targetRank,targetRank);
    }
    qr_econ(q,r,y);
    shedMatrix(q);
    MPI_Gather_matrix(r, rGather, targetRank, targetRank, MPI_DOUBLE, MPI_COMM_WORLD, _nodes, 0, _rank);
    svd_comm += (_nodes-1)*targetRank*targetRank;
    shedMatrix(r);
    if(_rank == 0)
    {
        qr_econ(qCombined, rCombined, rGather);
        rInverse = inv(rCombined);
        shedMatrix(qCombined);
        shedMatrix(rCombined);
    }
    MPI_Bcast_matrix(rInverse, rInverse.n_rows, rInverse.n_cols, MPI_DOUBLE, MPI_COMM_WORLD, 0, _rank);
    svd_comm += (_nodes-1)*targetRank*targetRank;

    qPart = y * rInverse;
    shedMatrix(rInverse);
}

void getAtqCombined(const arma:: mat& X_hat, const arma:: mat& qPart, arma:: mat& atqCombined, const int& d, const int& targetRank, const int& _nodes, const int& _rank, int& svd_comm){
    mat atqCombinedOverall;
    if(_rank == 0){
        atqCombinedOverall = zeros<mat>(_nodes*d,targetRank);
    }
    mat atq = X_hat.t() * qPart;
    MPI_Gather_matrix(atq, atqCombinedOverall, d, targetRank, MPI_DOUBLE, MPI_COMM_WORLD, _nodes, 0, _rank);
    svd_comm += (_nodes-1)*d*targetRank;
    if(_rank==0){
        sumUpMatrix(atqCombinedOverall, atqCombined, d, _nodes);
    }
    MPI_Bcast_matrix(atqCombined, atqCombined.n_rows, atqCombined.n_cols, MPI_DOUBLE, MPI_COMM_WORLD, 0, _rank);
    svd_comm += (_nodes-1)*d*targetRank;
    shedMatrix(atq);
    if(_rank==0){
        shedMatrix(atqCombinedOverall);
    }
}

void svdSolver(const arma:: mat& X_hat, arma:: mat& u, arma:: mat& s, arma:: mat& v, const int& seed, const int& powerIt, const int& _n_PerNode, const int& d, const int& targetRank, const int& _nodes, const int& _rank, int& svd_comm){
    mat y = zeros<mat>(_n_PerNode, targetRank);
    mat qPart = zeros<mat>(_n_PerNode, targetRank);
    mat atqCombined = zeros<mat>(d, targetRank);
    mat eigvec = zeros<mat>(targetRank,targetRank);

    getYUsingRandomMatrix(X_hat, y, seed, d, targetRank);
    getQ(y, qPart, targetRank, _nodes, _rank, svd_comm);

    for(int it = 0;it<powerIt; it++){
        getAtqCombined(X_hat, qPart, atqCombined, d, targetRank, _nodes, _rank, svd_comm);
        y = X_hat * atqCombined;
        getQ(y, qPart, targetRank, _nodes, _rank, svd_comm);
    }
    getAtqCombined(X_hat, qPart, atqCombined, d, targetRank, _nodes, _rank, svd_comm);
    if(_rank==0){
        mat temp = atqCombined.t()*atqCombined;
        vec eigval;
        eig_sym(eigval, eigvec, temp);
        s = zeros<mat>(targetRank, targetRank);
        for(int i=0;i<targetRank;i++){
            s(i,i) = sqrt(eigval(i));
        }
        v = (atqCombined * eigvec) * inv(s);
    }
    MPI_Bcast_matrix(eigvec, eigvec.n_rows, eigvec.n_cols, MPI_DOUBLE, MPI_COMM_WORLD, 0, _rank);
    svd_comm += (_nodes-1)*targetRank*targetRank;
    u = qPart * eigvec;

    shedMatrix(y);
    shedMatrix(qPart);
}

void getUtE(const arma:: mat& u, arma:: mat& utE, const int& _n_PerNode, const int& targetRank, const int& _nodes, const int& _rank, int& da_comm){
    mat e_svm = zeros<mat>(_n_PerNode, 1);
    e_svm.fill(-1);
    mat partialUtE = u.t() * e_svm;

    mat utEOverall;
    if(_rank == 0){
        utEOverall = zeros<mat>(_nodes*targetRank,1);
    }
    MPI_Gather_matrix(partialUtE, utEOverall, targetRank, 1, MPI_DOUBLE, MPI_COMM_WORLD, _nodes, 0, _rank);
    da_comm += (_nodes-1)*targetRank;

    if(_rank==0){
        sumUpMatrix(utEOverall, utE, targetRank, _nodes);
    }
}

void getInvD(const arma:: mat& s, arma:: mat& inv_D, const int& targetRank, const double& C){
    mat D;
    mat ssT = s * s.t();
    D.eye(targetRank,targetRank);
    D = (0.5/C) * D;
    D = ssT + D;
    D = (-1) * D;
    inv_D = inv(D);
}

void alphaBeta(const arma:: mat & inv_D, const arma:: mat& utE, const arma:: mat& prevBetaCap, arma:: mat& alpha_hat, arma:: mat& beta_hat, const double& optStepSize, const int& _rank){
    if(_rank==0){
        alpha_hat = inv_D * (utE - prevBetaCap);
        beta_hat = prevBetaCap - (optStepSize * alpha_hat);
    }
    MPI_Bcast_matrix(beta_hat, beta_hat.n_rows, beta_hat.n_cols, MPI_DOUBLE, MPI_COMM_WORLD, 0, _rank);
}

void calculateUBeta(const arma:: mat& u, const arma:: mat& beta_hat, arma:: mat& partial_beta_updated, const int& targetRank, const int& _nodes, const int& _rank, int& da_comm){
    partial_beta_updated = u * beta_hat;
    da_comm += (_nodes-1)*targetRank;
}

void nonNegativityCheck(arma:: mat& partial_beta_updated){
    for(int i=0;i<partial_beta_updated.n_rows;i++){
        if(partial_beta_updated(i,0)<0){
            partial_beta_updated(i,0) = 0;
        }
    }
}

void calculateUtBeta(const arma:: mat& u, arma:: mat& new_beta_cap, const arma:: mat& partial_beta_updated, const int& targetRank, const int& _nodes, const int& _rank, int& da_comm){
    mat new_beta_cap_overall;
    if(_rank == 0){
        new_beta_cap_overall = zeros<mat>(_nodes*targetRank,1);
    }
    mat partial_beta_cap = u.t() * partial_beta_updated;
    MPI_Gather_matrix(partial_beta_cap, new_beta_cap_overall, targetRank, 1, MPI_DOUBLE, MPI_COMM_WORLD, _nodes, 0, _rank);
    da_comm += (_nodes-1)*targetRank;
    if(_rank==0){
        sumUpMatrix(new_beta_cap_overall, new_beta_cap, targetRank, _nodes);
    }
}

void test(const arma:: mat& s, const arma:: mat& v, const arma:: mat& alpha_hat, const int& testDataSize, const int& d, const string dataset_name, const int& _rank){
    if(_rank==0){
        mat swithVt = s * v.t();
        mat test_data = zeros<mat>(testDataSize, d);
        mat test_label = zeros<mat>(testDataSize, 1);
        read_test_data(test_data, dataset_name);
        read_test_label(test_label, dataset_name);
        mat weight_mat = swithVt.t() * alpha_hat;
        mat result = test_data * weight_mat;

        int correct = 0;
        int wrong = 0;
        int truePos = 0;
        int trueNeg = 0;
        int falsePos = 0;
        int falseNeg = 0;

        for(int i=0;i<testDataSize;i++){
            if(result(i,0)>0){
                if(test_label(i,0)==1.0){
                    correct++;
                    truePos++;
                }
                else{
                    wrong++;
                    falsePos++;
                }
            }
            else{
                if(test_label(i,0)==1.0){
                    wrong++;
                   falseNeg++;
                }
                else{
                    correct++;
                    trueNeg++;
                }
            }
        }
        cout<<"Total correct is "<<correct<<endl;
        cout<<"Total wrong is "<<wrong<<endl;
        double acc = (correct*100.0)/testDataSize;
        cout<<"Accuracy is "<<acc<<endl;
        cout<<"True positive is "<<truePos<<endl;
        cout<<"False positive is "<<falsePos<<endl;
        cout<<"True negative is "<<trueNeg<<endl;
        cout<<"False negative is "<<falseNeg<<endl;
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
    int targetRank, testDataSize, powerIt;
    string dataset_name;
    C = atof(argv[1]);
    learnRate = atof(argv[2]);
    thresh = atof(argv[3]);
    targetRank = atoi(argv[4]);
    powerIt = atoi(argv[5]);
    testDataSize = atoi(argv[6]);
    dataset_name = argv[7];

    auto start_total_time = time_tic();
    auto start_load = time_tic();

    // Read Data file
    string fname_Dataset = "./partitions/data_part_" + std::to_string(_rank) + ".csv";
    ifstream read_Dataset(fname_Dataset.c_str());

    //load dataset
    int _n_PerNode, d;

    mat dataset,X;
    vec label;
    if (read_Dataset.is_open()){
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
    else{
        cout << "Unable to open file or file does not exist"<<endl;
        exit(1);
    }

    read_Dataset.close();

    // remove dataset
    dataset.shed_cols(0,d);
    dataset.shed_rows(0,_n_PerNode-1);

    if (_rank ==0)
    {
	time_toc("loading partitioned dataset "+to_string(_rank), start_load);
    }

    int svd_comm = 0;
    int da_comm = 0;

    vec x(_n_PerNode);
    X = join_horiz(X, x.ones(_n_PerNode)); //insert vector of 1s
    d = d+1; // dimensionality of X is now d+1 with insertion of Ones

    mat X_hat = zeros<mat>(_n_PerNode,d);
    for (int i=0; i<_n_PerNode; i++)
    {
        X_hat(i, span::all) = label(i)*X(i, span::all); //_n_perNodexd TS matrix
    }
    // Data preparation completes

    auto start_Traintime = time_tic();
    auto startTotalSVD = time_tic();
    int seed = 137;
    mat u = zeros<mat>(_n_PerNode,targetRank);
    mat s, v;
    if(_rank == 0){
        s = zeros<mat>(targetRank, targetRank);
        v = zeros<mat>(targetRank, d);
    }

    svdSolver(X_hat, u, s, v, seed, powerIt, _n_PerNode, d, targetRank, _nodes, _rank, svd_comm);
    shedMatrix(X_hat);

    time_toc("Total SVD time "+to_string(_rank), startTotalSVD );

    auto dual_ascent_start = time_tic();
    mat utE;
    if(_rank == 0){
        utE = zeros<mat>(targetRank,1);
    }
    getUtE(u, utE, _n_PerNode, targetRank, _nodes, _rank, da_comm);

    mat inv_D;
    if(_rank == 0){
        inv_D = zeros<mat>(targetRank,targetRank);
        getInvD(s, inv_D, targetRank, C);
    }

    double optP, optStepSize;
    optP =  floor(1/(learnRate*C)) - 0.5; // For safety, subtract 0.5
    optStepSize= optP*learnRate;

    int itCount = 0;
    double error = 1;
    mat prevBetaCap = zeros<mat>(targetRank,1);
    prevBetaCap.fill(1);
    mat alpha_hat;
    mat beta_hat = zeros<mat>(targetRank,1);
    mat partial_beta_updated = zeros<mat>(_n_PerNode,1);
    mat new_beta_cap;
    if(_rank == 0){
        alpha_hat = zeros<mat>(targetRank,1);
        new_beta_cap = zeros<mat>(targetRank,1);
    }

    while(error > thresh)
    {
        itCount++;
        alphaBeta(inv_D, utE, prevBetaCap, alpha_hat, beta_hat, optStepSize, _rank);
        calculateUBeta(u, beta_hat, partial_beta_updated, targetRank, _nodes, _rank, da_comm);
        nonNegativityCheck(partial_beta_updated);
        calculateUtBeta(u, new_beta_cap, partial_beta_updated, targetRank, _nodes, _rank, da_comm);
        if(_rank==0){
            error = norm(new_beta_cap - prevBetaCap,1);
            cout << "Iteration "<<itCount<<".. error=  "<<std::scientific<< error << endl<<endl;
            prevBetaCap = new_beta_cap;
        }
        MPI_Bcast(&error, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if(_rank==0){
        time_toc("Total dual ascent time "+to_string(_rank), dual_ascent_start );
        time_toc("Total time "+to_string(_rank), start_total_time );
        cout<<"Total svd communication "<<svd_comm<<endl;
        cout<<"Total dual ascent communication "<<da_comm<<endl;
        cout<<"Total communication "<<svd_comm+da_comm<<endl;
    }
    test(s, v, alpha_hat, testDataSize, d, dataset_name, _rank);

    MPI_Finalize();
    return 0;
}
