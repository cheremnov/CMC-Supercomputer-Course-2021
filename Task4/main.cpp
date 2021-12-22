#include <cassert>
#include <fstream>
#include <iostream>

#include "utils.h"
#include "num_method.h"
/**
 * Receives the filename with the parameters
 * as the first argument
 */
int main(int argc, char **argv){
	assert(argc >= 2);
	std::ifstream ifn(argv[1]);
	assert(ifn.is_open());
	double N, T, tau, h, L_x, L_y, L_z;
	int rows, columns, z_columns, time_frames;
	//ifn >> N >> h >> T >> tau;
	ifn >> L_x >> h >> T >> tau;
	ifn.close();
	L_z = L_y = L_x;
	N = rows = columns = z_columns = 
		static_cast<int>(floor(L_x / h)) + 1;
	time_frames = static_cast<int>(floor(T / tau)) + 1;

	// Fill the informational structures
	GlobalMesh_t global_mesh(rows, columns, z_columns, time_frames);

	// MPI prologue
	MPI_Init(&argc, &argv);
	double start = MPI_Wtime();
	int comm_rank=0, comm_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
	int root_process = 0;
	std::vector<BlockInfo_t> blk_descrs = getBlkDescrs(global_mesh, comm_size);
	assert(comm_rank < blk_descrs.size());
	
	int rank = comm_rank;
	PdeSolver pde(blk_descrs[comm_rank], global_mesh, N, T, h, tau, L_x, L_y, L_z, blk_descrs);
	pde.solve();
	
	double global_estimation_error = 0;
	double estimation_error = pde.getEstimationError();
	MPI_Reduce(&estimation_error, &global_estimation_error, 1,
			   MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if( comm_rank == root_process )
	{
		std::cout << "Calculation error: " << std::fixed <<
			global_estimation_error << std::endl;
		std::cout << "Time elapsed: " << std::fixed <<  (MPI_Wtime() - start) << std::endl;
	}	
	MPI_Finalize();
}