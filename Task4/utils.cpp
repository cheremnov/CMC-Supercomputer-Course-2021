#include <iostream>
#include "utils.h"

/**
 * Get 3 dividers of the processor num, such as their sum is minimal.
 * It's necessary to optimize block partition.
 */
void setProcessorGrid(BlockInfo_t& blk_info, int process_cnt)
{
	assert(process_cnt > 0);
	blk_info.blk_rows_ = process_cnt;
	blk_info.blk_columns_ = blk_info.blk_z_columns_ = 1;
	int min_divisor_sum = process_cnt + 2;
	for(int first_divisor = static_cast<int>(floor(std::pow(process_cnt, 1/3.)));
			first_divisor > 1; --first_divisor){
		if(process_cnt % first_divisor == 0){
			int divided_cnt = process_cnt / first_divisor;
			int second_divisor = static_cast<int>(floor(std::pow(divided_cnt, 1/2.)));

			// Divided_cnt has the divisor 1, the cycle will stop
			for(; divided_cnt % second_divisor; --second_divisor){}
			int third_divisor = divided_cnt / second_divisor;
			if(first_divisor + second_divisor + third_divisor < min_divisor_sum){
				min_divisor_sum = first_divisor + second_divisor + third_divisor;
				blk_info.blk_rows_ = first_divisor;
				blk_info.blk_columns_ = second_divisor;
				blk_info.blk_z_columns_ = third_divisor;
			}
		}
	}
	assert(blk_info.blk_rows_ * blk_info.blk_columns_ * blk_info.blk_z_columns_ == process_cnt);
}

std::vector<BlockInfo_t> getBlkDescrs(GlobalMesh_t& global_mesh, int process_cnt){
	BlockInfo_t blk_info;
	std::vector<BlockInfo_t> blk_descrs;
	setProcessorGrid(blk_info, process_cnt);
	int blk_z_start = 0;
	int blk_z_size = global_mesh.z_columns_ / blk_info.blk_z_columns_;
	int blk_column_size = global_mesh.columns_ / blk_info.blk_columns_;
	int blk_row_size = global_mesh.rows_ / blk_info.blk_rows_;
	for(int blk_z_idx = 0; blk_z_idx < blk_info.blk_z_columns_; ++blk_z_idx){
		int blk_row_start = 0;
		for(int blk_row_idx = 0; blk_row_idx < blk_info.blk_rows_; ++blk_row_idx){
			int blk_column_start = 0;
			for(int blk_column_idx = 0; blk_column_idx < blk_info.blk_columns_; ++blk_column_idx){
				BlockInfo_t cur_blk_info = blk_info;
				cur_blk_info.setBlkStartPos(blk_row_start, blk_column_start, blk_z_start);
				cur_blk_info.setBlkSize(blk_row_size, blk_column_size, blk_z_size);
				cur_blk_info.setBlkIdx(blk_row_idx, blk_column_idx, blk_z_idx);
				/**
				 * Sometimes can't divide the mesh equally.
				 * Distribute the remainder among the first blocks along
				 * the row, column or z-column.
				 */
				if((global_mesh.rows_ - blk_row_start) %
				   (blk_info.blk_rows_ - blk_row_idx)){
					cur_blk_info.blk_row_size_++;
				}
				if((global_mesh.columns_ - blk_column_start) %
				   (blk_info.blk_columns_ - blk_column_idx)){
					cur_blk_info.blk_column_size_++;
				}
				if((global_mesh.z_columns_ - blk_z_start) %
				   (blk_info.blk_z_columns_ - blk_z_idx)){
					cur_blk_info.blk_z_size_++;
				}
				blk_descrs.push_back(cur_blk_info);
				blk_column_start += blk_descrs.back().blk_column_size_;
			}
			assert(blk_column_start == global_mesh.columns_);
			blk_row_start += blk_descrs.back().blk_row_size_;
		}
		assert(blk_row_start == global_mesh.rows_);
		blk_z_start += blk_descrs.back().blk_z_size_;
	}
	assert(blk_z_start == global_mesh.z_columns_);
	return blk_descrs;
}
						
/**
 * Find the index of element in the local mesh
 * by the @global_idx in the global mesh.
 * The returned local index respects the apdding.
 * Returns:
 *     True, if the element found.
 *     False, otherwisse
 */
bool globalToLocal(const GlobalIndex_t& global_idx, const BlockInfo_t& blk_info,
				   int& local_row_idx, int& local_column_idx, int& local_z_idx){
	// Rows, columns and z columns are padded.
	local_row_idx = global_idx.row_idx_ - blk_info.blk_row_start_ + 1;
	local_column_idx = global_idx.column_idx_ - blk_info.blk_column_start_ + 1;
	local_z_idx = global_idx.z_idx_ - blk_info.blk_z_start_ + 1;
	if( (local_row_idx < 0 || local_row_idx >= blk_info.blk_row_size_ + 2) ||
		(local_column_idx < 0 || local_column_idx >= blk_info.blk_column_size_ + 2) ||
		(local_z_idx < 0 || local_z_idx >= blk_info.blk_z_size_ + 2) ){
		return false;
	}
	return true;
}

/**
 * RAII for the array of doubles of @mesh_size.
 * Returns:
 *      A smart pointer to the first array element
 */
MeshPtr_t make_meshptr_unique(size_t mesh_size) {
	assert(mesh_size > 0);
    return MeshPtr_t{ reinterpret_cast<double*>(std::calloc(mesh_size, sizeof(double))), std::free };
}