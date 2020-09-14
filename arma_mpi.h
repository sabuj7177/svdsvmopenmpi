#ifndef ARMA_MPI_H_
#define ARMA_MPI_H_

#include<armadillo>
#include<mpi.h>
#include<iostream>

using namespace std;

inline void MPI_Gather_matrix(arma::mat& a, arma::mat& b,int rows,int cols, MPI_Datatype data_type,MPI_Comm comm_world, int world_size, int root, int rank) {
	double* a_cpp = new double[rows*cols];

	for(int i = 0;i<rows;i++) {
		for(int j = 0;j<cols;j++) {
			a_cpp[(i*cols)+j] = a(i,j);
		}
	}

	double* b_cpp = nullptr;

	if(rank== root)
		b_cpp = new double[world_size*rows*cols];

	MPI_Gather(a_cpp,rows*cols,data_type,b_cpp,rows*cols,data_type,root,comm_world);

	if(rank== root) {
		b.zeros(world_size*rows,cols);
		for(int i = 0;i<world_size*rows;i++) {
			for(int j = 0; j<cols; j++) {
				b(i,j) = b_cpp[(i*cols)+j];
			}
		}

	}

	delete[] a_cpp;
	delete[] b_cpp;


}

inline void MPI_Gather_matrix_2(arma::mat& a, arma::mat& b,int rows,int cols, MPI_Datatype data_type,MPI_Comm comm_world, int total_rows, int root, int rank) {
	double* a_cpp = new double[rows*cols];

	for(int i = 0;i<rows;i++) {
		for(int j = 0;j<cols;j++) {
			a_cpp[(i*cols)+j] = a(i,j);
		}
	}

	double* b_cpp = nullptr;

	if(rank== root)
		b_cpp = new double[total_rows*cols];

	MPI_Gather(a_cpp,rows*cols,data_type,b_cpp,rows*cols,data_type,root,comm_world);

	if(rank== root) {
		b.zeros(total_rows,cols);
		for(int i = 0;i<total_rows;i++) {
			for(int j = 0; j<cols; j++) {
				b(i,j) = b_cpp[(i*cols)+j];
			}
		}

	}

	delete[] a_cpp;
	delete[] b_cpp;


}

inline void MPI_Scatter_matrix(arma::mat& b, arma::mat& a,int rows,int cols, MPI_Datatype data_type,MPI_Comm comm_world, int world_size, int root, int rank)
{
	double* b_cpp = nullptr;
	int n = world_size*rows*cols;
        if(rank == root)
	{
        	b_cpp = new double[n];

		for(int i = 0;i<world_size*rows;i++) {
                	for(int j = 0;j<cols;j++) {
                        	b_cpp[(i*cols)+j] = b(i,j);
                	}
        	}
		// cout<<endl<<"b_cpp[0]: "<<*b_cpp;
		// cout<<endl<<"b_cpp[rows]: "<<b_cpp[rows];
	}

	double* a_cpp = nullptr;
	a_cpp = new double[rows*cols];
        MPI_Scatter(b_cpp,rows*cols,data_type,a_cpp,rows*cols,data_type,root,comm_world);
//	cout<<endl<<"Node: "<<rank<<", a_cpp[0]: "<<a_cpp[0];

	 for(int i = 0;i<rows;i++) {
                for(int j = 0;j<cols;j++) {
                        a(i,j) = a_cpp[(i*cols)+j];
                }
        }
//	cout<<endl<<"Node: "<<rank<<", a[0]: "<<a[0]<<endl;
	delete[] a_cpp;
	delete[] b_cpp;
}

inline void MPI_Bcast_matrix(arma::mat& a, int rows, int cols, MPI_Datatype data_type, MPI_Comm comm_world,  int root, int rank) {
        double* a_cpp = new double[rows*cols];

	if(rank == root)
        {
		for(int i = 0;i<rows;i++) {
                	for(int j = 0;j<cols;j++) {
                        	a_cpp[(i*cols)+j] = a(i,j);
               		 }
        	}
	}

        MPI_Bcast(a_cpp, rows*cols, data_type, root, comm_world);

        if(rank != root)
	{
                for(int i = 0; i<rows; i++) {
                        for(int j = 0; j<cols; j++) {
                                a(i,j) = a_cpp[(i*cols)+j];
                        }
                }

        }

        delete[] a_cpp;



}

inline void MPI_Bcast_value(int& a, MPI_Datatype data_type, MPI_Comm comm_world,  int root, int rank) {
        double* a_cpp = new double[1];

	if(rank == root)
    {
        a_cpp[0] = a;
	}

    MPI_Bcast(a_cpp, 1, data_type, root, comm_world);

    if(rank != root)
	{
        a = a_cpp[0];
    }

        delete[] a_cpp;



}

#endif
