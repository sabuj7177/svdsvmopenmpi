#ifndef MPI_HELPER_H
#define MPI_HELPER_H

#include <mpi.h>
#include <string>


class MPI_helper{
  int rank;
  int initialized;
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int pname_len;
  int size;
  MPI_Comm commWorld;
public:
  MPI_helper(){
     mpiInit(); 
  }
  
  void mpiInit(){
    MPI_Initialized(&initialized);
    if(!initialized)
      MPI_Init(NULL,NULL);
    commWorld = MPI_COMM_WORLD;
    MPI_Comm_size (MPI_COMM_WORLD, &size);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name (processor_name, &pname_len); 
  }

  int getRank(){
    return this->rank;
  }

  int getSize(){
    return this->size;
  }

  int getInitialized(){
    return this->initialized;
  }

  MPI_Comm getCommWorld(){
    return this->commWorld;
  }

  std::string getProcessorName(){
    return std::string(processor_name);
  }

  ~MPI_helper(){
    MPI_Finalize();
  }
};


#endif
