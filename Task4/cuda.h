#ifndef CUDA_H
#define CUDA_H
#include <cassert>
/**
 * Describes the structs in the device memory.
 * The device stores three local (padded) meshes.
 *
 * On each iteration, the numerical method uses
 * only three latest meshes.
 */
struct DeviceMemoryDescr_t{
	double* prev_mesh_;
	double* cur_mesh_;
	double* next_mesh_;
};

/**
 * Describes the structs from the host memory
 * sent to the device.
 */
struct HostMemoryDescr_t{
	double* prev_mesh_;
	double* cur_mesh_;
	double* next_mesh_;
	HostMemoryDescr_t(double* prev_mesh, double* cur_mesh, double* next_mesh):
				      prev_mesh_(prev_mesh), cur_mesh_(cur_mesh), next_mesh_(next_mesh) {}
};

void mallocOnCuda(DeviceMemoryDescr_t* device_memory_p, int padded_capacity);
void freeOnCuda(DeviceMemoryDescr_t* device_memory_p);
void callMethodIterKernel(DeviceMemoryDescr_t* device_memory_p,
						  HostMemoryDescr_t* host_memory_p,
						  int padded_capacity,
					      int rows, int columns, int z_columns,
						  double tau, double h);
#endif