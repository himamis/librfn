/*
 Copyright © 2015 Thomas Unterthiner
 Licensed under GPL, version 2 or a later (see LICENSE.txt)
 */

#include "GPUDenseOperations.h"
#include "GPUDenseKernels.h"

GPUDenseOperations::GPUDenseOperations(const int n, const int m, const int k, unsigned long seed, int gpu_id) {

	// if no GPU was specified, try to pick the best one automatically
	if (gpu_id < 0) {
		gpu_id = get_gpu_id();
	}
	assert(gpu_id >= 0);
	cudaSetDevice(gpu_id);

	// the following call does not work if the current process has already
	// called into librfn previously. Then, this call will return
	// cudaErrorSetOnActiveProcess. Resetting the device won't work either,
	// because then the subsequent cublasCreate call will just fail with
	// CUBLAS_STATUS_NOT_INITIALIZED. I don't know why any of this is happening
	//CUDA_CALL(cudaSetDeviceFlags(cudaDeviceScheduleYield));


	CUSOLVER_CALL(cusolverDnCreate(&cudense_handle));
	CUDA_CALL(cudaMalloc(&rng_state, RNG_BLOCKS * RNG_THREADS * sizeof(curandState)));
	setup_rng<<<RNG_BLOCKS, RNG_THREADS>>>(rng_state, seed);
	int ones_size = n > k ? n : k;
	ones = malloc(ones_size * sizeof(float));
	fill(ones, ones_size, 1.0f);
	CUDA_CALL(cudaMalloc(&devinfo, sizeof(int)));
}

GPUDenseOperations::~GPUDenseOperations() {
	free(devinfo);
	free(ones);
	CUSOLVER_CALL(cusolverDnDestroy(cudense_handle));
	CUBLAS_CALL(cublasDestroy(handle));
}

float* GPUDenseOperations::to_device(const float* src, size_t size) const {
	float* dst = 0;
	CUDA_CALL(cudaMalloc(&dst, size));
	CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
	return dst;
}

void GPUDenseOperations::dropout(float* X, const unsigned size, const float dropout_rate) const {
	dropout_eltw<<<RNG_BLOCKS, RNG_THREADS>>>(X, size, dropout_rate, rng_state);
	assert(!cudaGetLastError());
}

void GPUDenseOperations::add_gauss_noise(float* X, const unsigned size, const float noise_rate) const {
	gauss_noise_eltw<<<RNG_BLOCKS, RNG_THREADS>>>(X, size, noise_rate, rng_state);
	assert(!cudaGetLastError());
}

void GPUDenseOperations::add_saltpepper_noise(float* X, const unsigned size, const float noise_rate) const {
	saltpepper_noise_eltw<<<RNG_BLOCKS, RNG_THREADS>>>(X, size, noise_rate, rng_state);
	assert(!cudaGetLastError());
}

void GPUDenseOperations::calculate_column_variance(float* X, const unsigned nrows, const unsigned ncols,
		float* variance) const {
	int threads, blocks;
	get_grid_sizes(ncols, &threads, &blocks);
	col_variance_kernel<<<threads, blocks>>>(X, variance, nrows, ncols);
}

void GPUDenseOperations::scale_columns(float* X, const unsigned nrows, const unsigned ncols, float* s) const {

	int threads, blocks;
	get_grid_sizes(ncols * nrows, &threads, &blocks);
	scale_columns_kernel<<<threads, blocks>>>(X, s, nrows, ncols);
}

void GPUDenseOperations::scale_rows(float* X, const unsigned nrows, const unsigned ncols, float* s) const {
	int threads, blocks;
	get_grid_sizes(ncols * nrows, &threads, &blocks);
	scale_rows_kernel<<<threads, blocks>>>(X, s, nrows, ncols);
}

void GPUDenseOperations::printMatrixRM(const float* a, int n, int m, const char* fmt) {
	const char* format = fmt == 0 ? "%1.3f " : fmt;
	size_t size = n * m * sizeof(float);
	float* tmp = (float*) std::malloc(size);
	CUDA_CALL(cudaMemcpy(tmp, a, size, cudaMemcpyDeviceToHost));
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j)
			printf(format, tmp[i * m + j]);
		printf("\n");
	}
	printf("\n");
	std::free(tmp);
}

void GPUDenseOperations::printMatrixCM(const float* a, int n, int m, const char* fmt) {
	const char* format = fmt == 0 ? "%1.3f " : fmt;
	size_t size = n * m * sizeof(float);
	float* tmp = (float*) std::malloc(size);
	CUDA_CALL(cudaMemcpy(tmp, a, size, cudaMemcpyDeviceToHost));
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j)
			printf(format, tmp[i + j * n]);
		printf("\n");
	}
	printf("\n");
	std::free(tmp);
}
