#ifndef NUMMETHOD_H
#define NUMMETHOD_H
#include <math.h>
#include <memory>
#include <cstring>
#include <vector>

#include "cuda.h"
#include "utils.h"

#ifdef OMP_ENABLED
#include <omp.h>
#define PRAGMA_OMP(pragma) pragma
#else
#define PRAGMA_OMP(pragma)
#endif

/** 
 * Unordered map is better suited for this task
 */
#if __cplusplus >= 201103L
#include <unordered_map>
typedef std::unordered_map<GlobalIndex_t, double, GlobalIdxHasher> MeshMap_t;
#else
#error "You should have C++11 or higher to compile the program"
#endif

int packIdx(int local_row_idx, int local_column_idx, int local_z_idx,
			int rows, int columns, int z_columns);

// It is simultaneously a boundary condition and an analytical solution
double phi(double x, double y, double z, double tau,
           double L_x, double L_y, double L_z);

class Mesh{
public:
    Mesh(int rows, int columns, int z_columns,
         BlockInfo_t blk_info, GlobalMesh_t global_mesh):
        rows_(rows), columns_(columns), z_columns_(z_columns),
        blk_info_(blk_info), global_mesh_(global_mesh),
        padded_capacity_((rows_ + 2) * (columns_ + 2) *
                         (z_columns_ + 2)),
        padded_local_mesh_(make_meshptr_unique(padded_capacity_)){
    }
    
    Mesh(const Mesh& other):
        rows_(other.rows_), columns_(other.columns_), z_columns_(other.z_columns_),
        blk_info_(other.blk_info_), global_mesh_(other.global_mesh_),
        global_mesh_data_(other.global_mesh_data_),
        padded_capacity_(other.padded_capacity_),
        padded_local_mesh_(make_meshptr_unique(padded_capacity_)){
        std::memcpy(padded_local_mesh_.get(), other.padded_local_mesh_.get(),
                    padded_capacity_ * sizeof(double));
    }
    
    Mesh& operator=(Mesh other){
        swap(*this, other);
        return *this;
    }

    
    /**
     * Get the mesh element by its global index
     */
    double& getByGlobalIdx(const GlobalIndex_t& global_idx){
        int local_row_idx, local_column_idx, local_z_idx;
        if (globalToLocal(global_idx, blk_info_,
                          local_row_idx, local_column_idx,
                          local_z_idx)){
            return padded_local_mesh_.get()[packIdx(local_row_idx, local_column_idx, local_z_idx,
                                                    rows_ + 2, columns_ + 2, z_columns_ + 2)];
        }
        // Throws an exception if the element isn't found
        return global_mesh_data_.at(global_idx);
    }

    /**
     * Get reference to the contiguous 3D local mesh,
     * padded on all dimensions with the padding 1.
     * The values are filled with boundary elements
     * when available, otherwise with zero.
     */
    double* getPaddedMesh(){
        return padded_local_mesh_.get();
    }

    double& operator[](const GlobalIndex_t& global_idx){
        return getByGlobalIdx(global_idx);
    }
    
    void insertGlobalMesh(GlobalIndex_t global_idx, double value){
        global_mesh_data_[global_idx] = value;
    }
    /**
     * Copy-and-swap idiom
     */
    friend void swap(Mesh& first, Mesh& second){
        std::swap(first.rows_, second.rows_);
        std::swap(first.columns_, second.columns_);
        std::swap(first.z_columns_, second.z_columns_);
        std::swap(first.padded_local_mesh_, second.padded_local_mesh_);
        std::swap(first.global_mesh_data_, second.global_mesh_data_);
    }
    
    double laplace(GlobalIndex_t global_idx, double spatial_step);
    int rows_;
    int columns_;
    int z_columns_;
    int padded_capacity_;
private:
	/**
	 * Pad the local mesh to contain
	 * the neighboring elements received from other processes
	 */
    MeshPtr_t padded_local_mesh_;
    /**
	 * The @padded_local_mesh_ can't contain
	 * all the global mesh elements.
	 */
    MeshMap_t global_mesh_data_;
    BlockInfo_t blk_info_;
    GlobalMesh_t global_mesh_;
};

class PdeSolver{
public:
    PdeSolver(BlockInfo_t blk_info, GlobalMesh_t global_mesh,
              double N, double T, double h, double tau,
              double L_x, double L_y, double L_z,
              const std::vector<BlockInfo_t>& blk_descrs):
        tau_(tau), h_(h), N_(N), T_(T), blk_descrs_(blk_descrs),
        global_mesh_(global_mesh), blk_info_(blk_info),
        L_x_(L_x), L_y_(L_y), L_z_(L_z) {
        /**
         * Tau is a time step, h is a spatial step.
         */
        assert(global_mesh.rows_ = static_cast<int>(floor(N / h)) + 1);
        assert(global_mesh.time_frames_ = static_cast<int>(floor(T / tau)) + 1);
        assert(global_mesh.rows_ >= 2 && global_mesh.columns_ >= 2
               && global_mesh.z_columns_ >= 2);
        assert(global_mesh.time_frames_ >= 2);
        
        /**
         * Don't allocate all meshes at once
         * Create them when they are required
         */
        meshes_ = std::vector<Mesh>(2,
                             Mesh(blk_info_.blk_row_size_,
                                  blk_info_.blk_column_size_,
                                  blk_info_.blk_z_size_,
                                  blk_info_,
                                  global_mesh_));
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
                    meshes_[0][mesh_idx] = phi(row_idx * h, column_idx * h,
                                               z_idx * h, 0, L_x, L_y, L_z);
                    meshes_[1][mesh_idx] = phi(row_idx * h, column_idx * h,
                                               z_idx * h, tau, L_x, L_y, L_z);
                }
            }
        }
        estimation_error_ = 0;
        iteration_ = 1;
        // Mesh capacity never changes during the solving process
        padded_capacity_ = meshes_[0].padded_capacity_;
        #ifdef CUDA_ENABLED
        mallocOnCuda(&device_memory_, padded_capacity_);
        #endif
    }
    ~PdeSolver(){
        #ifdef CUDA_ENABLED
        freeOnCuda(&device_memory_);
        #endif
    }
    void sendBlock(GlobalIndex_t gl1_idx, GlobalIndex_t gl2_idx,
               int rank, std::vector<MPI_Request>& reqs,
               std::vector<double>& sent_buf,
               int& send_req_size, int recv_req_size);
    void recvBlock(GlobalIndex_t gl1_idx, GlobalIndex_t gl2_idx,
               int rank, std::vector<MPI_Request>& reqs,
               std::vector<std::pair<GlobalIndex_t, double> >& received_buf,
               int send_req_size, int& recv_req_size);
    void communicate();
    void computeBoundaries();
    void computeEstimationError();
    void nextIter();
    void solve();
    double getEstimationError(){
        return estimation_error_;
    }
private:
    const BlockInfo_t blk_info_;
    const GlobalMesh_t global_mesh_;
    const std::vector<BlockInfo_t>& blk_descrs_;
    std::vector<Mesh> meshes_;
    double tau_;
    double h_;
    double T_;
    double N_;
    double L_x_;
    double L_y_;
    double L_z_;
    int iteration_;
    int padded_capacity_;
    // Sum of all estimation errors
    double total_estimation_error_;
    /**
     * Estimation_error contains the max estimation error
     * on the entire mesh
     */
    double estimation_error_;
    DeviceMemoryDescr_t device_memory_;
};


#endif