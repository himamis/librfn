#include "gpu_sparse_operations.h"

static const int RNG_THREADS = 128;
static const int RNG_BLOCKS = 128;


__global__ void setup_rng(curandState* rng_state, unsigned long seed)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, tid, 0, &rng_state[tid]);
}


GPU_SparseOperations::GPU_SparseOperations(const int n, const int m,
		const int k, unsigned long seed, int gpu_id) {

	// if no GPU was specified, try to pick the best one automatically
	if (gpu_id < 0) {
		gpu_id = get_gpu_id();
	}
	assert(gpu_id >= 0);
	cudaSetDevice(gpu_id);

	cusparseStatus_t status = cusparseCreate(&sparseHandle);

	if (status != CUSPARSE_STATUS_SUCCESS) {
		fprintf(stderr, "cuSparse: %d\n", status);
		cudaDeviceReset();
		throw std::runtime_error("cuSparse error");
	}
	CUSOLVER_CALL(cusolverSpCreate(&solverHandle));
	CUDA_CALL(
			cudaMalloc(&rngState,
					RNG_BLOCKS * RNG_THREADS * sizeof(curandState)));
	setup_rng<<<RNG_BLOCKS, RNG_THREADS>>>(rngState, seed);
	int ones_size = n > k ? n : k;
	ones = malloc(ones_size * sizeof(float));
	fill(ones, ones_size, 1.0f);
	CUDA_CALL(cudaMalloc(&devinfo, sizeof(int)));
}

GPU_Operations::~GPU_Operations() {
	free(devinfo);
	free(ones);
	for (auto i : buffer_map) {
		free(i.second);
	}
	CUSOLVER_CALL(cusolverDnDestroy(cudense_handle));
	CUBLAS_CALL(cublasDestroy(handle));
}
