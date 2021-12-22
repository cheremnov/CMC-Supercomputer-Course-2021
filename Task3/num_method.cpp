#include <iostream>
#include "num_method.h"

double phi(double x, double y, double z, double tau,
		   double L_x, double L_y, double L_z){
	double a_t = M_PI * sqrt(M_PI * 1.0 / (L_x * L_x) + M_PI * 4.0 / (L_y * L_y) +
							 M_PI * 9.0 / (L_z * L_z));
	return sin(M_PI * x / L_x) * sin(2 * M_PI * y / L_y) * sin(M_PI * z / L_z) * cos(a_t * tau);
}

/** 
 * Non-blocking send a block of the local mesh between
 * @gl1_idx and @gl2_idx (@gl2_idx > @gl1_idx)
 * to the process with a given @rank.
 *
 * Put the block elements in @sent_buf.
 * Increase @send_req_size by the number of block elements
 * Put the requests into @reqs vector.
 * Warning:
 * 	@sent_buf must allocate enough space to fit
 *  all the block elements
 *  @reqs must have enough space to fit all requests.
 */
void PdeSolver::sendBlock(GlobalIndex_t gl1_idx, GlobalIndex_t gl2_idx,
			   int rank, std::vector<MPI_Request>& reqs,
			   std::vector<double>& sent_buf,
			   int& send_req_size, int recv_req_size){
	Mesh& cur_mesh = meshes_[iteration_];
	for(int row_idx = gl1_idx.row_idx_; row_idx < gl2_idx.row_idx_; ++row_idx){
		for(int column_idx = gl1_idx.column_idx_; column_idx < gl2_idx.column_idx_; ++column_idx){
			for(int z_idx = gl1_idx.z_idx_; z_idx < gl2_idx.z_idx_; ++z_idx){
				GlobalIndex_t gl_idx = GlobalIndex_t(row_idx, column_idx, z_idx);
				assert(send_req_size < sent_buf.size());
				assert((send_req_size + recv_req_size) < reqs.size());
				sent_buf[send_req_size] = cur_mesh[gl_idx];
				//MPI_Send(&sent_buf[send_req_size], 1, MPI_DOUBLE, rank, gl_idx.hash(global_mesh_),
				//MPI_COMM_WORLD);
				MPI_Isend(&sent_buf[send_req_size], 1, MPI_DOUBLE, rank, gl_idx.hash(global_mesh_),
				MPI_COMM_WORLD, &reqs[send_req_size + recv_req_size]);
				++send_req_size;
			}
		}
	}
}

/** 
 * Non-blocking receive a block of the local mesh between
 * @gl1_idx and @gl2_idx (@gl2_idx > @gl1_idx)
 * from the process with a given @rank.
 *
 * Put the block elements in @received_buf.
 * Increase @send_req_size by the number of block elements
 * Put the requests into @reqs vector.
 * Warning:
 * 	@received_buf must allocate enough space to fit
 *  all the block elements
 *  @reqs must have enough space to fit all requests
 */
void PdeSolver::recvBlock(GlobalIndex_t gl1_idx, GlobalIndex_t gl2_idx,
			   int rank, std::vector<MPI_Request>& reqs,
			   std::vector<std::pair<GlobalIndex_t, double> >& received_buf,
			   int send_req_size, int& recv_req_size){
	MPI_Status status;
	Mesh& cur_mesh = meshes_[iteration_];
	for(int row_idx = gl1_idx.row_idx_; row_idx < gl2_idx.row_idx_; ++row_idx){
		for(int column_idx = gl1_idx.column_idx_; column_idx < gl2_idx.column_idx_; ++column_idx){
			for(int z_idx = gl1_idx.z_idx_; z_idx < gl2_idx.z_idx_; ++z_idx){
				GlobalIndex_t gl_idx = GlobalIndex_t(row_idx, column_idx, z_idx);
				assert(recv_req_size < received_buf.size());
				assert((send_req_size + recv_req_size) < reqs.size());
				received_buf[recv_req_size].first = gl_idx;
				//MPI_Recv(&received_buf[recv_req_size].second, 1, MPI_DOUBLE, rank, gl_idx.hash(global_mesh_),
				//MPI_COMM_WORLD, &status);
				MPI_Irecv(&received_buf[recv_req_size].second, 1, MPI_DOUBLE, rank, gl_idx.hash(global_mesh_),
				MPI_COMM_WORLD, &reqs[send_req_size + recv_req_size]);
				++recv_req_size;
			}
		}
	}
}
/**
 * Communicate the divided global mesh halo,
 * exchanging it with other processes
 */
void PdeSolver::communicate(){
	Mesh& cur_mesh = meshes_[iteration_];
	MeshMap_t& global_mesh_data_ = meshes_[iteration_].getGlobalMeshData();
	int cur_rank = blk_info_.getRank();
	std::vector<MPI_Request> reqs(4 * (blk_info_.blk_column_size_ * blk_info_.blk_row_size_ +
			    blk_info_.blk_row_size_ * blk_info_.blk_z_size_ +
			    blk_info_.blk_column_size_ * blk_info_.blk_z_size_));
	/**
	 * There is no guarantee a pointer to a map element remains stable.
	 * Use the temporary buffer for storing the received elements.
	 */
	std::vector<std::pair<GlobalIndex_t, double> > received_buf(
					  2 * (blk_info_.blk_row_size_ * blk_info_.blk_column_size_ +
					  blk_info_.blk_row_size_ * blk_info_.blk_z_size_ +
					  blk_info_.blk_column_size_ * blk_info_.blk_z_size_));
	std::vector<double> sent_buf(
				  2 * (blk_info_.blk_row_size_ * blk_info_.blk_column_size_ +
				  blk_info_.blk_row_size_ * blk_info_.blk_z_size_ +
				  blk_info_.blk_column_size_ * blk_info_.blk_z_size_));
	// Count receive and send requests separately
	int recv_req_size = 0, send_req_size = 0;
	if(blk_info_.blk_row_start_ + blk_info_.blk_row_size_ < global_mesh_.rows_){
		/**
		 * The CPU sends the elements
		 * mesh[blk_row_start_ + blk_row_size_ - 1][j][z],
		 * receives the elements
		 * mesh[blk_row_start_ + blk_rows_size_][j][z]
		 * j,z falls within the local mesh's borders
		 */
		int row_idx = blk_info_.blk_row_start_ + blk_info_.blk_row_size_ - 1;
		int right_row_neighbor = GlobalIndex_t(row_idx + 1, blk_info_.blk_column_start_,
		                                       blk_info_.blk_z_start_).getProcessorRank(blk_descrs_);
		assert(right_row_neighbor != -1);
		sendBlock(GlobalIndex_t(row_idx, blk_info_.blk_column_start_, blk_info_.blk_z_start_),
			      GlobalIndex_t(row_idx + 1, blk_info_.blk_column_start_ + blk_info_.blk_column_size_,
										 blk_info_.blk_z_start_ + blk_info_.blk_z_size_),
				  right_row_neighbor, reqs, sent_buf, send_req_size, recv_req_size);
		recvBlock(GlobalIndex_t(row_idx + 1, blk_info_.blk_column_start_, blk_info_.blk_z_start_),
				  GlobalIndex_t(row_idx + 2, blk_info_.blk_column_start_ + blk_info_.blk_column_size_,
										 blk_info_.blk_z_start_ + blk_info_.blk_z_size_),
				  right_row_neighbor, reqs, received_buf, send_req_size, recv_req_size);
	}
	if(blk_info_.blk_row_start_ != 0){
		int row_idx = blk_info_.blk_row_start_;
		/**
		 * The CPU receives the elements
		 * mesh[blk_row_start_ - 1][j][z],
		 * send the elements
		 * mesh[blk_row_start_][j][z]
		 * j,z falls within the local mesh's borders
		 */
		int left_row_neighbor = GlobalIndex_t(row_idx - 1, blk_info_.blk_column_start_,
		                                      blk_info_.blk_z_start_).getProcessorRank(blk_descrs_);
		assert(left_row_neighbor != -1);
		recvBlock(GlobalIndex_t(row_idx - 1, blk_info_.blk_column_start_, blk_info_.blk_z_start_),
				  GlobalIndex_t(row_idx, blk_info_.blk_column_start_ + blk_info_.blk_column_size_,
							    blk_info_.blk_z_start_ + blk_info_.blk_z_size_),
				  left_row_neighbor, reqs, received_buf, send_req_size, recv_req_size);
		sendBlock(GlobalIndex_t(row_idx, blk_info_.blk_column_start_, blk_info_.blk_z_start_),
			      GlobalIndex_t(row_idx + 1, blk_info_.blk_column_start_ + blk_info_.blk_column_size_,
								blk_info_.blk_z_start_ + blk_info_.blk_z_size_),
				  left_row_neighbor, reqs, sent_buf, send_req_size, recv_req_size);
	}
	if(blk_info_.blk_column_start_ + blk_info_.blk_column_size_ != global_mesh_.columns_){
		/**
		 * The CPU sends the elements
		 * mesh[i][blk_column_start_ + blk_column_size_ - 1][z],
		 * receives the elements
		 * mesh[i][blk_column_start_ + blk_column_size_][z]
		 * i,z falls within the local mesh's borders
		 */
		int column_idx = blk_info_.blk_column_start_ + blk_info_.blk_column_size_ - 1;
		//int right_column_neighbor = cur_rank + 1;
		int right_column_neighbor = GlobalIndex_t(blk_info_.blk_row_start_, column_idx + 1,
												  blk_info_.blk_z_start_).getProcessorRank(blk_descrs_);
		assert(right_column_neighbor == cur_rank + 1);
		sendBlock(GlobalIndex_t(blk_info_.blk_row_start_, column_idx, blk_info_.blk_z_start_),
			      GlobalIndex_t(blk_info_.blk_row_start_ + blk_info_.blk_row_size_, column_idx + 1,
								blk_info_.blk_z_start_ + blk_info_.blk_z_size_),
				  right_column_neighbor, reqs, sent_buf, send_req_size, recv_req_size);
		recvBlock(GlobalIndex_t(blk_info_.blk_row_start_, column_idx + 1, blk_info_.blk_z_start_),
				  GlobalIndex_t(blk_info_.blk_row_start_ + blk_info_.blk_row_size_, column_idx + 2,
										 blk_info_.blk_z_start_ + blk_info_.blk_z_size_),
				  right_column_neighbor, reqs, received_buf, send_req_size, recv_req_size);
	}
	if(blk_info_.blk_column_start_ != 0){
		/**
		 * The CPU receives the elements
		 * mesh[i][blk_column_start_ - 1][z],
		 * sends the elements
		 * mesh[i][blk_column_start_][z]
		 * i,z falls within the local mesh's borders
		 */
		int column_idx = blk_info_.blk_column_start_;
		// Left neighbor
		//int left_column_neighbor = cur_rank - 1;
		int left_column_neighbor = GlobalIndex_t(blk_info_.blk_row_start_, column_idx - 1,
												 blk_info_.blk_z_start_).getProcessorRank(blk_descrs_);
		assert(left_column_neighbor == cur_rank - 1);
		recvBlock(GlobalIndex_t(blk_info_.blk_row_start_, column_idx - 1, blk_info_.blk_z_start_),
				  GlobalIndex_t(blk_info_.blk_row_start_ + blk_info_.blk_row_size_, column_idx,
									 blk_info_.blk_z_start_ + blk_info_.blk_z_size_),
				  left_column_neighbor, reqs, received_buf, send_req_size, recv_req_size);
		sendBlock(GlobalIndex_t(blk_info_.blk_row_start_, column_idx, blk_info_.blk_z_start_),
			      GlobalIndex_t(blk_info_.blk_row_start_ + blk_info_.blk_row_size_, column_idx + 1,
								blk_info_.blk_z_start_ + blk_info_.blk_z_size_),
				  left_column_neighbor, reqs, sent_buf, send_req_size, recv_req_size);
	}
	
	if(blk_info_.blk_z_start_ + blk_info_.blk_z_size_ != global_mesh_.z_columns_){
		/**
		 * The CPU sends the elements
		 * mesh[i][j][blk_z_start_ + blk_z_size_ - 1],
		 * receives the elements
		 * mesh[i][j][blk_z_start_ + blk_z_size_]
		 * i,j falls within the local mesh's borders
		 */
		int z_idx = blk_info_.blk_z_start_ + blk_info_.blk_z_size_ - 1;
		//int right_z_neighbor = cur_rank + blk_info_.blk_rows_ * blk_info_.blk_columns_;
		int right_z_neighbor = GlobalIndex_t(blk_info_.blk_row_start_, blk_info_.blk_column_start_,
											 z_idx + 1).getProcessorRank(blk_descrs_);
		assert(right_z_neighbor == cur_rank + blk_info_.blk_rows_ * blk_info_.blk_columns_);
		sendBlock(GlobalIndex_t(blk_info_.blk_row_start_, blk_info_.blk_column_start_, z_idx),
			      GlobalIndex_t(blk_info_.blk_row_start_ + blk_info_.blk_row_size_,
								blk_info_.blk_column_start_ + blk_info_.blk_column_size_, z_idx + 1),
				  right_z_neighbor, reqs, sent_buf, send_req_size, recv_req_size);
		recvBlock(GlobalIndex_t(blk_info_.blk_row_start_, blk_info_.blk_column_start_, z_idx + 1),
				  GlobalIndex_t(blk_info_.blk_row_start_ + blk_info_.blk_row_size_,
								blk_info_.blk_column_start_ + blk_info_.blk_column_size_, z_idx + 2),
				  right_z_neighbor, reqs, received_buf, send_req_size, recv_req_size);
	}
	if(blk_info_.blk_z_start_ != 0){
		/**
		 * The CPU receives the elements
		 * mesh[i][j][blk_z_start_ - 1],
		 * sends the elements
		 * mesh[i][j][blk_z_start_]
		 * i,z falls within the local mesh's borders
		 */
		int z_idx = blk_info_.blk_z_start_;
		//int left_z_neighbor = cur_rank - blk_info_.blk_rows_ * blk_info_.blk_columns_;
		int left_z_neighbor = GlobalIndex_t(blk_info_.blk_row_start_, blk_info_.blk_column_start_,
											z_idx - 1).getProcessorRank(blk_descrs_);
		assert(left_z_neighbor == cur_rank - blk_info_.blk_rows_ * blk_info_.blk_columns_);
		recvBlock(GlobalIndex_t(blk_info_.blk_row_start_, blk_info_.blk_column_start_, z_idx - 1),
				  GlobalIndex_t(blk_info_.blk_row_start_ + blk_info_.blk_row_size_,
								blk_info_.blk_column_start_ + blk_info_.blk_column_size_, z_idx),
				  left_z_neighbor, reqs, received_buf, send_req_size, recv_req_size);
		sendBlock(GlobalIndex_t(blk_info_.blk_row_start_, blk_info_.blk_column_start_, z_idx),
			      GlobalIndex_t(blk_info_.blk_row_start_ + blk_info_.blk_row_size_,
								blk_info_.blk_column_start_ + blk_info_.blk_column_size_, z_idx + 1),
				  left_z_neighbor, reqs, sent_buf, send_req_size, recv_req_size);
	}
	std::vector<MPI_Status> statuses(recv_req_size + send_req_size);
	MPI_Waitall(recv_req_size + send_req_size, &reqs[0], &statuses[0]);
	// Construct a map from the key-value vector.
	for(int req_idx = 0; req_idx < recv_req_size; ++req_idx){
		GlobalIndex_t global_idx = received_buf[req_idx].first;
		assert(global_idx.isValid());
		global_mesh_data_[global_idx] = received_buf[req_idx].second;
		assert(global_mesh_data_.find(global_idx) != global_mesh_data_.end());
	}
}
void PdeSolver::computeBoundaries(){
	Mesh& cur_mesh = meshes_[iteration_];
	Mesh& next_mesh = meshes_[iteration_ + 1];
	// First boundary
	if( blk_info_.blk_row_start_ == 0 || blk_info_.blk_row_start_ + blk_info_.blk_row_size_ == global_mesh_.rows_){
		PRAGMA_OMP(_Pragma("omp parallel for"))
		for(int column_idx = blk_info_.blk_column_start_;
				column_idx < blk_info_.blk_column_start_ + blk_info_.blk_column_size_; ++column_idx){
			PRAGMA_OMP(_Pragma("omp parallel for"))
			for(int z_idx = blk_info_.blk_z_start_;
					z_idx < blk_info_.blk_z_start_ + blk_info_.blk_z_size_; ++z_idx){
				GlobalIndex_t bound1 = GlobalIndex_t(0, column_idx, z_idx);
				GlobalIndex_t bound2 = GlobalIndex_t(global_mesh_.rows_ - 1, column_idx, z_idx);
				if(bound1.isLocal(blk_info_)){
					next_mesh[bound1] = 0;
				}
				if(bound2.isLocal(blk_info_)){
					next_mesh[bound2] = 0;
				}
			}
		}
	}
	
	int cur_rank = blk_info_.getRank();
	/**
	 * Periodic boundary condition
	 * mesh(x, 0, z) = mesh(x, N, z) = mesh(x, 1, z) + mesh(x, N-1, z)
	 * mesh(x, 1, z) and mesh(x, N-1, z) may not lie in the same local mesh.
	 * If so, communicate their value via MPI blocking messages.
	 */
	MPI_Status status;
	if( blk_info_.blk_column_start_ == 0 ||
		blk_info_.blk_column_start_ + blk_info_.blk_column_size_ == global_mesh_.columns_){
		for(int row_idx = blk_info_.blk_row_start_;
				row_idx < blk_info_.blk_row_start_ + blk_info_.blk_row_size_; ++row_idx){
			for(int z_idx = blk_info_.blk_z_start_;
					z_idx < blk_info_.blk_z_start_ + blk_info_.blk_z_size_; ++z_idx){
				GlobalIndex_t period_bound1(row_idx, 1, z_idx);
				GlobalIndex_t period_bound2(row_idx, global_mesh_.columns_ - 2, z_idx);
				int period_bound1_rank = period_bound1.getProcessorRank(blk_descrs_);
				int period_bound2_rank = period_bound2.getProcessorRank(blk_descrs_);
				assert(cur_rank == period_bound1_rank || cur_rank == period_bound2_rank);
				if(period_bound1_rank != period_bound2_rank)
				{
					// The processors exchange the mesh(x, 1, z) and mesh(x, N-1, z)
					if(cur_rank == period_bound1_rank && cur_rank != period_bound2_rank){
						assert(period_bound1.isLocal(blk_info_));
						double period_bound2_val;
						MPI_Send(&cur_mesh[period_bound1], 1, MPI_DOUBLE, period_bound2_rank, period_bound1.hash(global_mesh_),
								 MPI_COMM_WORLD);
						MPI_Recv(&period_bound2_val, 1, MPI_DOUBLE, period_bound2_rank, period_bound2.hash(global_mesh_),
								 MPI_COMM_WORLD, &status);
						cur_mesh.insertGlobalMesh(period_bound2, period_bound2_val);
					} else if(cur_rank == period_bound2_rank && cur_rank != period_bound1_rank){
						assert(period_bound2.isLocal(blk_info_));
						double period_bound1_val;
						MPI_Recv(&period_bound1_val, 1, MPI_DOUBLE, period_bound1_rank, period_bound1.hash(global_mesh_),
								 MPI_COMM_WORLD, &status);
						MPI_Send(&cur_mesh[period_bound2], 1, MPI_DOUBLE, period_bound1_rank, period_bound2.hash(global_mesh_),
								 MPI_COMM_WORLD);
						cur_mesh.insertGlobalMesh(period_bound1, period_bound1_val);
					}
				}
				double bound_val = (cur_mesh[period_bound1] + cur_mesh[period_bound2]) / 2;
				
				GlobalIndex_t bound1 = GlobalIndex_t(row_idx, 0, z_idx);
				GlobalIndex_t bound2 = GlobalIndex_t(row_idx, global_mesh_.columns_ - 1, z_idx);
				if(bound1.isLocal(blk_info_)){
					next_mesh[bound1] = bound_val;
				}
				if(bound2.isLocal(blk_info_)){
					next_mesh[bound2] = bound_val;
				}
			}
		}
	}
	
	if( blk_info_.blk_z_start_ == 0 || blk_info_.blk_z_start_ + blk_info_.blk_z_size_ == global_mesh_.z_columns_){
		PRAGMA_OMP(_Pragma("omp parallel for"))
		for(int row_idx = blk_info_.blk_row_start_;
			row_idx < blk_info_.blk_row_start_ + blk_info_.blk_row_size_; ++row_idx){
			PRAGMA_OMP(_Pragma("omp parallel for"))
			for(int column_idx = blk_info_.blk_column_start_;
					column_idx < blk_info_.blk_column_start_ + blk_info_.blk_column_size_; ++column_idx){
				GlobalIndex_t bound1 = GlobalIndex_t(row_idx, column_idx, 0);
				GlobalIndex_t bound2 = GlobalIndex_t(row_idx, column_idx, global_mesh_.z_columns_ - 1);
				if(bound1.isLocal(blk_info_)){
					next_mesh[bound1] = 0;
				}
				if(bound2.isLocal(blk_info_)){
					next_mesh[bound2] = 0;
				}
			}
		}
	}
}
	
void PdeSolver::computeEstimationError(){
	total_estimation_error_ = 0;
	Mesh& next_mesh = meshes_[iteration_ + 1];
	PRAGMA_OMP(_Pragma("omp parallel for"))
	for(int row_idx = blk_info_.blk_row_start_;
			row_idx < blk_info_.blk_row_start_ + blk_info_.blk_row_size_; ++row_idx){
		PRAGMA_OMP(_Pragma("omp parallel for"))
		for(int column_idx = blk_info_.blk_column_start_;
				column_idx < blk_info_.blk_column_start_ + blk_info_.blk_column_size_; ++column_idx){
			PRAGMA_OMP(_Pragma("omp parallel for"))
			for(int z_idx = blk_info_.blk_z_start_;
					z_idx < blk_info_.blk_z_start_ + blk_info_.blk_z_size_; ++z_idx){
				double new_estimation_error = fabs(next_mesh[GlobalIndex_t(row_idx, column_idx, z_idx)] -
												   phi(row_idx, column_idx, z_idx, iteration_ * tau_,
												       L_x_, L_y_, L_z_));
				if(new_estimation_error > estimation_error_){
					estimation_error_ = new_estimation_error;
				}
				total_estimation_error_ += new_estimation_error;
			}
		}
	}
}
	
void PdeSolver::nextIter(){
	assert(iteration_ + 1 < global_mesh_.time_frames_);
	Mesh& prev_mesh = meshes_[iteration_ - 1];
	Mesh& cur_mesh = meshes_[iteration_];
	Mesh& next_mesh = meshes_[iteration_ + 1];
	communicate();
	PRAGMA_OMP(_Pragma("omp parallel for"))
	for(int row_idx = blk_info_.blk_row_start_;
			row_idx < blk_info_.blk_row_start_ + blk_info_.blk_row_size_; ++row_idx){
		PRAGMA_OMP(_Pragma("omp parallel for"))
		for(int column_idx = blk_info_.blk_column_start_;
				column_idx < blk_info_.blk_column_start_ + blk_info_.blk_column_size_; ++column_idx){
			PRAGMA_OMP(_Pragma("omp parallel for"))
			for(int z_idx = blk_info_.blk_z_start_;
					z_idx < blk_info_.blk_z_start_ + blk_info_.blk_z_size_; ++z_idx){
				GlobalIndex_t mesh_idx = GlobalIndex_t(row_idx, column_idx, z_idx);
				assert(mesh_idx.isLocal(blk_info_));
				if(!mesh_idx.isBoundary(global_mesh_)){
					next_mesh[mesh_idx] = 
						cur_mesh.laplace(mesh_idx, h_) * tau_ * tau_ -
						prev_mesh[mesh_idx] + 2 * cur_mesh[mesh_idx];
				}
			}
		}
	}
	computeBoundaries();
	computeEstimationError(); 
	double global_estimation_error = 0;
	MPI_Reduce(&estimation_error_, &global_estimation_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	double global_total_estimation_error = 0;
	MPI_Reduce(&total_estimation_error_, &global_total_estimation_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	double global_avg_estimation_error = (global_total_estimation_error /
									      (global_mesh_.rows_ * global_mesh_.columns_ * global_mesh_.z_columns_));
		
	if(blk_info_.getRank() == 0){
		std::cout << "Estimation error on iteration " << iteration_ << " : " << std::fixed << global_estimation_error << std::endl;
		std::cout << "Average estimation error: " << std::fixed << global_avg_estimation_error << std::endl; 
		/*GlobalIndex_t avg_idx(blk_info_.blk_row_start_ + blk_info_.blk_row_size_ / 2,
						      blk_info_.blk_column_start_ + blk_info_.blk_column_size_ / 2,
							  blk_info_.blk_z_start_ + blk_info_.blk_z_size_ / 2);
		std::cout << "Computed solution: " << std::fixed << next_mesh[avg_idx] << std::endl;
		std::cout << "Analytical solution: " << phi(avg_idx.row_idx_, avg_idx.column_idx_, avg_idx.z_idx_,
												    tau_ * iteration_, L_x_, L_y_, L_z_) << std::endl;*/
	}
	iteration_++;
}
void PdeSolver::solve(){
	while(iteration_ + 1 < global_mesh_.time_frames_){
		// Create a new mesh for an iteration
		meshes_.push_back(Mesh(blk_info_.blk_row_size_,
					   blk_info_.blk_column_size_,
					   blk_info_.blk_z_size_,
					   blk_info_,
					   global_mesh_));
		nextIter();
		if(iteration_ > 20){
			break;
		}
	}
}


/**
 * Seven-point discrete laplace operator value
 * for the given mesh element
 * Assume the mesh grid has the same spatial step for all axes.
 */
double Mesh::laplace(GlobalIndex_t global_idx, double spatial_step){
	assert(!global_idx.isBoundary(global_mesh_));
	int row_idx = global_idx.row_idx_;
	int column_idx = global_idx.column_idx_;
	int z_idx = global_idx.z_idx_;
			
	double row_part = (getByGlobalIdx(GlobalIndex_t(row_idx - 1, column_idx, z_idx)) - 
					   2 * getByGlobalIdx(GlobalIndex_t(row_idx, column_idx, z_idx)) +
					   getByGlobalIdx(GlobalIndex_t(row_idx + 1, column_idx, z_idx))) /
					  (spatial_step * spatial_step);
	double column_part = (getByGlobalIdx(GlobalIndex_t(row_idx, column_idx - 1, z_idx)) - 
						  2 * getByGlobalIdx(GlobalIndex_t(row_idx, column_idx, z_idx)) +
						  getByGlobalIdx(GlobalIndex_t(row_idx, column_idx + 1, z_idx))) /
						 (spatial_step * spatial_step);
	double z_part = (getByGlobalIdx(GlobalIndex_t(row_idx, column_idx, z_idx - 1)) - 
					 2 * getByGlobalIdx(GlobalIndex_t(row_idx, column_idx, z_idx)) +
					 getByGlobalIdx(GlobalIndex_t(row_idx, column_idx, z_idx + 1))) /
					(spatial_step * spatial_step);
	return row_part + column_part + z_part;
}