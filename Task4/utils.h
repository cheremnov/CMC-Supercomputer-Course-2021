#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif
#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include <cassert>
#include <memory>
#include <vector>
#include <math.h>

#include "mpi.h"

/**
 * Each CPU contains a part of the global mesh.
 * Local meshes divide the global mesh into blocks
 * alongside spatial dimensions.
 * This struct describes the blocks.
 */
class BlockInfo_t
{
public:
    // Indices of the block in the block grid
    int blk_row_idx_;
    int blk_column_idx_;
    int blk_z_idx_;
    
    // The block grid's parameters
    int blk_rows_;
    int blk_columns_;
    int blk_z_columns_;
    

    // The borders of the block
    int blk_row_start_;
    int blk_column_start_;
    int blk_z_start_;
    int blk_row_size_;
    int blk_column_size_;
    int blk_z_size_;
    /**
     * Get a rank of the CPU that owns
     * a part of mesh containing the element.
     */
    int getRank() const{
        /**
         * Example:
         * z_idx: 0
         * | 0 | 1 | 2 |
         * | 3 | 4 | 5 |
         * z_idx: 1
         * | 6 | 7  | 8 |
         * | 9 | 10 | 11|
         */
        return blk_z_idx_ * blk_columns_ * blk_rows_ +
               blk_row_idx_ * blk_columns_ + blk_column_idx_;
    }   
    
    void setBlkIdx(int blk_row_idx, int blk_column_idx, int blk_z_idx){
        blk_row_idx_ = blk_row_idx;
        blk_column_idx_ = blk_column_idx;
        blk_z_idx_ = blk_z_idx;
    }

    void setBlkStartPos(int blk_row_start, int blk_column_start, int blk_z_start){
        blk_row_start_ = blk_row_start;
        blk_column_start_ = blk_column_start;
        blk_z_start_ = blk_z_start;
    }
    
    void setBlkSize(int blk_row_size, int blk_column_size, int blk_z_size){
        blk_row_size_ = blk_row_size;
        blk_column_size_ = blk_column_size;
        blk_z_size_ = blk_z_size;
    }
    
    void print() const{
        std::cout << "Block info: " << std::endl;
        std::cout << " Block row idx: " << blk_row_idx_ << " Block column idx: " << blk_column_idx_ <<
                  " Block z idx: " << blk_z_idx_ << std::endl;
        std::cout << "Block rows: " << blk_rows_ << " Block columns: " << blk_columns_ <<
                  " Block z columns: " << blk_z_columns_ << std::endl;
        std::cout << "Block row start: " << blk_row_start_ << " Block column start: " << blk_column_start_ <<
                  " Block z column start: " << blk_z_start_ << std::endl;
        std::cout << "Block row size: " << blk_row_size_ << " Block column size: " << blk_column_size_ <<
                  " Block z column size: " << blk_z_size_ << std::endl;
    }
};

/** 
 * A global mesh is a 4-D tensor
 * with 3 spatial dimensions and 1 time dimension.
 */
struct GlobalMesh_t{
    int rows_;
    int columns_;
    int z_columns_;
    int time_frames_;
    GlobalMesh_t(int rows = 0, int columns = 0, int z_columns = 0, int time_frames = 0):
        rows_(rows), columns_(columns), z_columns_(z_columns), time_frames_(time_frames) {}
};

/**
 * An index of the element in a mesh
 */
struct GlobalIndex_t{
    int row_idx_;
    int column_idx_;
    int z_idx_; 
    GlobalIndex_t():
        row_idx_(-1), column_idx_(-1), z_idx_(-1) {}
    GlobalIndex_t(int row_idx, int column_idx, int z_idx):
        row_idx_(row_idx), column_idx_(column_idx), z_idx_(z_idx) {}
    GlobalIndex_t(const GlobalIndex_t& other):
        row_idx_(other.row_idx_), column_idx_(other.column_idx_), z_idx_(other.z_idx_) {}
        
    bool operator <(const GlobalIndex_t& gl_idx) const{
        if(z_idx_ != gl_idx.z_idx_){
            return z_idx_ < gl_idx.z_idx_;
        }
        if(row_idx_ != gl_idx.row_idx_){
            return row_idx_ < gl_idx.row_idx_;
        }
        return column_idx_ < gl_idx.column_idx_;
    }
    
    bool operator ==(const GlobalIndex_t& gl_idx) const{
        return row_idx_ == gl_idx.row_idx_ &&
               column_idx_ == gl_idx.column_idx_ &&
               z_idx_ == gl_idx.z_idx_;
    }
    /**
     * Check whether the element lies on 
     * the global mesh boundary
     */
    bool isBoundary(const GlobalMesh_t& global_mesh) const{
        return row_idx_ == 0 || column_idx_ == 0 || z_idx_ == 0 ||
               row_idx_ == global_mesh.rows_ - 1 ||
               column_idx_ == global_mesh.columns_ - 1 ||
               z_idx_ == global_mesh.z_columns_ - 1;
               
    }
    /**
     * Check if the element with the given index
     * lies in the local mesh
     */
    bool isLocal(const BlockInfo_t& blk_info) const{
        return blk_info.blk_row_start_ <= row_idx_ &&
               row_idx_ < blk_info.blk_row_start_ + blk_info.blk_row_size_ &&
               blk_info.blk_column_start_ <= column_idx_ &&
               column_idx_ < blk_info.blk_column_start_ + blk_info.blk_column_size_ &&
               blk_info.blk_z_start_ <= z_idx_ &&
               z_idx_ < blk_info.blk_z_start_ + blk_info.blk_z_size_;
    }
    
    bool isValid(){
        return row_idx_ >= 0 && column_idx_ >= 0 && z_idx_ >= 0;
    }
    /**
     * Traverse the block descriptors array
     * to find out which local mesh contains
     * the global index
     */
    int getProcessorRank(const std::vector<BlockInfo_t>& descriptors) const{
        for(std::vector<BlockInfo_t>::const_iterator descr_iter = descriptors.begin();
            descr_iter != descriptors.end(); descr_iter++){
            if(isLocal(*descr_iter)){
                return descr_iter->getRank();
            }
        }
        return -1;
    }
    
    int hash(const GlobalMesh_t& global_mesh){
        return z_idx_ * global_mesh.columns_ * global_mesh.rows_ + 
               row_idx_ * global_mesh.columns_ + column_idx_;
    }
    
    void print() const{
        std::cout << row_idx_ << " " << column_idx_ << " " << z_idx_;
    }
};

#if __cplusplus >= 201103L
	struct GlobalIdxHasher
	{
		std::size_t operator()(const GlobalIndex_t& k) const
		{

			return ((std::hash<int>()(k.row_idx_)
					 ^ (std::hash<int>()(k.column_idx_) << 1)) >> 1)
					 ^ (std::hash<int>()(k.z_idx_) << 1);
		}
	};
#endif

void setProcessorGrid(BlockInfo_t& blk_info, int process_cnt);
std::vector<BlockInfo_t> getBlkDescrs(GlobalMesh_t& global_mesh, int process_cnt);
bool globalToLocal(const GlobalIndex_t& global_idx, const BlockInfo_t& blk_info,
                   int& local_row_idx, int& local_column_idx, int& local_z_idx);
				   
typedef std::unique_ptr<double, decltype(std::free)*> MeshPtr_t;
MeshPtr_t make_meshptr_unique(size_t mesh_size);
#endif