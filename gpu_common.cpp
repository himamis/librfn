#include "gpu_common.h"

int get_gpu_id() {
	int gpu_id, num_devices, device;
	gpu_id = -1;
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

void get_grid_sizes(int problemsize, int* blocks, int* threads) {
    int min_threads = 32;
    int max_threads = 256;
    int max_blocks = 384;

    if (problemsize < min_threads) {
        *blocks = 1;
        *threads = min_threads;
    } else if (problemsize < max_blocks * min_threads) {
        *blocks = (problemsize + min_threads - 1) / min_threads;
        *threads = min_threads;
    } else if (problemsize < max_blocks * max_threads) {
        *blocks = max_blocks;
        int grp = (problemsize + min_threads - 1) / min_threads;
        *threads = ((grp + max_blocks - 1) / max_blocks) * min_threads;
    } else {
        *blocks = max_blocks;
        *threads = max_threads;
    }
}
