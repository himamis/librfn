#include "gpu_common.h"
#include <cuda_runtime.h>

int get_gpu_id() {
	int gpu_id, num_devices, device;
	cudaGetDeviceCount(&num_devices);
	if (num_devices > 1) {
		size_t max_freememory = 0;
		for (device = 0; device < num_devices; device++) {
			size_t free, total;
			cudaDeviceProp prop;
			cudaSetDevice(device);
			cudaMemGetInfo(&free, &total);
			cudaGetDeviceProperties(&prop, device);
			//printf("Found device %d (%s) with %d MiB of free memory\n",
			//    device, prop.name, free / (1024l*1024l));
			if (free > max_freememory) {
				max_freememory = free;
				gpu_id = device;
			}
			cudaDeviceReset();
		}
	}
	return gpu_id;
}
