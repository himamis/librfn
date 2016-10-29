#include "GPUSparseOperations.h"
#include "GPUSparseKernels.h"

GPUSparseOperations::GPUSparseOperations(const int n, const int m, const int k, unsigned long seed, int gpu_id) {
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
	CUDA_CALL(cudaMalloc(&rngState, RNG_BLOCKS * RNG_THREADS * sizeof(curandState)));
	setup_rng<<<RNG_BLOCKS, RNG_THREADS>>>(rngState, seed);
	/*int ones_size = n > k ? n : k;
	 ones = malloc(ones_size * sizeof(float));
	 fill(ones, ones_size, 1.0f);*/
	CUDA_CALL(cudaMalloc(&devinfo, sizeof(int)));
}

GPUSparseOperations::~GPUSparseOperations() {
	free(devinfo);
	//free(ones);
	//for (auto i : buffer_map) {
	//	free(i.second);
	//}
	CUSOLVER_CALL(cusolverSpDestroy(solverHandle));
	CUSPARSE_CALL(cusparseDestroy(sparseHandle));
}

void GPUSparseOperations::fill_eye(cusparseMatDescr_t, unsigned int n) const {

}

void GPUSparseOperations::gemm(const char *transa, const char *transb, const int m, const int n, const int k,
		const float alpha, const float *a, const int lda, const float *b, const int ldb, const float beta, float *c,
		const int ldc) const {
	//CUSPARSE_CALL(cusparseSgemmi())

}

void GPUSparseOperations::scale_rows(cusparseMatDescr_t, const unsigned nrows, const unsigned ncols, float* s) const {
	int threads, blocks;
	get_grid_sizes(ncols * nrows, &threads, &blocks);
}
