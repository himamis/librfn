/*
 Copyright © 2015 Thomas Unterthiner
 Licensed under GPL, version 2 or a later (see LICENSE.txt)
 */

#include "GPUDenseOperations.h"
#include "GPUDenseKernels.h"

GPUDenseOperations::GPUDenseOperations(const int n, const int m, const int k, unsigned long seed, int gpu_id) :
		GPUOperations(n, m, k, seed, gpu_id) { }

GPUDenseOperations::~GPUDenseOperations() { }

float* GPUDenseOperations::to_device(const float* src, size_t size) const {
	float* dst = 0;
	CUDA_CALL(cudaMalloc(&dst, size));
	CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
	return dst;
}

void GPUDenseOperations::dropout(float* X, const unsigned size, const float dropout_rate) const {
	dense_dropout_eltw<<<RNG_BLOCKS, RNG_THREADS>>>(X, size, dropout_rate, rng_state);
	assert(!cudaGetLastError());
}

void GPUDenseOperations::add_gauss_noise(float* X, const unsigned size, const float noise_rate) const {
	dense_gauss_noise_eltw<<<RNG_BLOCKS, RNG_THREADS>>>(X, size, noise_rate, rng_state);
	assert(!cudaGetLastError());
}

void GPUDenseOperations::add_saltpepper_noise(float* X, const unsigned size, const float noise_rate) const {
	dense_saltpepper_noise_eltw<<<RNG_BLOCKS, RNG_THREADS>>>(X, size, noise_rate, rng_state);
	assert(!cudaGetLastError());
}

void GPUDenseOperations::calculate_column_variance(float* X, const unsigned nrows, const unsigned ncols,
		float* variance) const {
	int threads, blocks;
	get_grid_sizes(ncols, &threads, &blocks);
	dense_col_variance_kernel<<<threads, blocks>>>(X, variance, nrows, ncols);
}

void GPUDenseOperations::scale_columns(float* X, const unsigned nrows, const unsigned ncols, float* s) const {

	int threads, blocks;
	get_grid_sizes(ncols * nrows, &threads, &blocks);
	dense_scale_columns_kernel<<<threads, blocks>>>(X, s, nrows, ncols);
}

void GPUDenseOperations::scale_rows(float* X, const unsigned nrows, const unsigned ncols, float* s) const {
	int threads, blocks;
	get_grid_sizes(ncols * nrows, &threads, &blocks);
	dense_scale_rows_kernel<<<threads, blocks>>>(X, s, nrows, ncols);
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
