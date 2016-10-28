#include <cuda_runtime.h>
#include <cusparse.h>

class GPU_SparseOperations {

	cusparseHandle_t handle;

	GPU_SparseOperations(int n, int m, int k, unsigned long seed, int gpu_id);
	~GPU_SparseOperations();
};
