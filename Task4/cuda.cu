#include <assert.h>

#include "cuda.h"

void mallocOnCuda(DeviceMemoryDescr_t* device_memory_p, int padded_capacity){
    cudaMalloc(&device_memory_p->prev_mesh_, padded_capacity * sizeof(double));
    cudaMalloc(&device_memory_p->cur_mesh_, padded_capacity * sizeof(double));
    cudaMalloc(&device_memory_p->next_mesh_, padded_capacity * sizeof(double));
}
void freeOnCuda(DeviceMemoryDescr_t* device_memory_p){
    cudaFree(device_memory_p->prev_mesh_);
    cudaFree(device_memory_p->cur_mesh_);
    cudaFree(device_memory_p->next_mesh_);
}

/**
 * Get the index in the 1D array of size @rows * @columns * @z_columns,
 * that represents the 3D array [@rows; @columns; @z_columns]
 */
__device__ int packCudaIdx(int local_row_idx, int local_column_idx, int local_z_idx,
                           int rows, int columns, int z_columns){
    int packed_idx = local_row_idx * columns * z_columns +
                     local_column_idx * z_columns +
                     local_z_idx;
    assert(0 <= packed_idx && packed_idx < rows * columns * z_columns);
    return packed_idx;
}




/**
 * Compute the laplace operator over the current mesh cell.
 * @local_row_idx, @local_column_idx, @local_z_idx respects the padding.
 */
__device__ double laplace(const double* mesh,
                          int local_row_idx, int local_column_idx, int local_z_idx,
                          int padded_rows, int padded_columns, int padded_z_columns,
                          double spatial_step){
    double row_part = (mesh[packCudaIdx(local_row_idx - 1, local_column_idx, local_z_idx,
                                    padded_rows, padded_columns, padded_z_columns)] - 
                       2 * mesh[packCudaIdx(local_row_idx, local_column_idx, local_z_idx,
                                        padded_rows, padded_columns, padded_z_columns)] +
                       mesh[packCudaIdx(local_row_idx + 1, local_column_idx, local_z_idx,
                                        padded_rows, padded_columns, padded_z_columns)]) /
                      (spatial_step * spatial_step);
    double column_part = (mesh[packCudaIdx(local_row_idx, local_column_idx - 1, local_z_idx,
                                       padded_rows, padded_columns, padded_z_columns)] - 
                          2 * mesh[packCudaIdx(local_row_idx, local_column_idx, local_z_idx,
                                           padded_rows, padded_columns, padded_z_columns)] +
                          mesh[packCudaIdx(local_row_idx, local_column_idx + 1, local_z_idx,
                                       padded_rows, padded_columns, padded_z_columns)]) /
                         (spatial_step * spatial_step);
    double z_part = (mesh[packCudaIdx(local_row_idx, local_column_idx, local_z_idx - 1,
                                  padded_rows, padded_columns, padded_z_columns)] - 
                     2 * mesh[packCudaIdx(local_row_idx, local_column_idx, local_z_idx,
                                      padded_rows, padded_columns, padded_z_columns)] +
                     mesh[packCudaIdx(local_row_idx, local_column_idx, local_z_idx + 1,
                                  padded_rows, padded_columns, padded_z_columns)]) /
                    (spatial_step * spatial_step);
    return row_part + column_part + z_part;
}
/**
 * A kernel for the numerical method.
 * Warning:
 *      Doesn't check if the index is boundary for the global mesh,
 *      fills these indexes with incorrect values.
 *      Compute global boundary elements only after calling this kernel.
 */
__global__ void methodIterKernel(const double* prev_mesh, const double* cur_mesh, double* next_mesh,
                                 int rows, int columns, int z_columns,
                                 double tau, double h){

    // A grid-striding loop over the flattened 3D mesh
    for(int row_idx = 1; row_idx <= rows; ++row_idx){
        for(int column_idx = 1 + blockIdx.x; column_idx <= columns; column_idx += gridDim.x){
            for(int z_idx = 1 + threadIdx.x; z_idx <= z_columns; z_idx += blockDim.x){
                int packed_idx = packCudaIdx(row_idx, column_idx, z_idx,
											 rows + 2, columns + 2, z_columns + 2);
                next_mesh[packed_idx] = tau * tau * laplace(cur_mesh, row_idx, column_idx, z_idx,
                                                            rows + 2, columns + 2, z_columns + 2, h)
                                        - prev_mesh[packed_idx] + 2 * cur_mesh[packed_idx];
            }
        }
    }
}

/**
 * A wrapper for the kernel call.
 * Copies the memory to device
 */
void callMethodIterKernel(DeviceMemoryDescr_t* device_memory_p,
                          HostMemoryDescr_t* host_memory_p,
                          int padded_capacity,
                          int rows, int columns, int z_columns,
                          double tau, double h){
    cudaMemcpy(device_memory_p->prev_mesh_, host_memory_p->prev_mesh_, padded_capacity * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(device_memory_p->cur_mesh_, host_memory_p->cur_mesh_, padded_capacity * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(device_memory_p->next_mesh_, host_memory_p->next_mesh_, padded_capacity * sizeof(double),
               cudaMemcpyHostToDevice);
    int threads_per_block = 1024;
    int blocks_per_grid = 2;
    methodIterKernel<<<blocks_per_grid, threads_per_block>>>(device_memory_p->prev_mesh_, device_memory_p->cur_mesh_, device_memory_p->next_mesh_,
                                                             rows, columns, z_columns, tau, h);
    cudaMemcpy(host_memory_p->next_mesh_, device_memory_p->next_mesh_, padded_capacity * sizeof(double),
               cudaMemcpyDeviceToHost);
}