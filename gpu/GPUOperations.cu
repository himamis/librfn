#include "GPUOperations.h"
#include "GPUCommonKernels.h"
#include "SparseMatrix.h"

template<typename MatrixType>
GPUOperations<MatrixType>::GPUOperations(const int n, const int m, const int k, unsigned long seed, int gpu_id) {
	// if no GPU was specified, try to pick the best one automatically
	if (gpu_id < 0) {
		gpu_id = get_gpu_id();
	}
	assert(gpu_id >= 0);
	cudaSetDevice (gpu_id);

	cublasStatus_t status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		const char* errmsg = cublasErrorString(status);
		fprintf(stderr, "CUBLAS initialization error: %s\n", errmsg);
		cudaDeviceReset();
		throw std::runtime_error(errmsg);
	}
	CUSOLVER_CALL(cusolverDnCreate(&cudense_handle));
	CUDA_CALL(cudaMalloc(&devinfo, sizeof(int)));
	CUDA_CALL(cudaMalloc(&rng_state, RNG_BLOCKS * RNG_THREADS * sizeof(curandState)));
	setup_rng<<<RNG_BLOCKS, RNG_THREADS>>>(rng_state, seed);
	int ones_size = n > k ? n : k;
	ones = malloc(ones_size * sizeof(float));
	fill(ones, ones_size, 1.0f);
}

template<typename MatrixType>
GPUOperations<MatrixType>::~GPUOperations() {
	free(devinfo);

	for (auto i : buffer_map) {
		free(i.second);
	}
	CUSOLVER_CALL(cusolverDnDestroy(cudense_handle));
	CUBLAS_CALL(cublasDestroy(handle));
}

template<typename MatrixType>
float* GPUOperations<MatrixType>::to_device(const float* src, size_t size) const {
	float* dst = 0;
	CUDA_CALL(cudaMalloc(&dst, size));
	CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
	return dst;
}

template<typename MatrixType>
void* GPUOperations<MatrixType>::memset(void* dest, int ch, size_t count) const {
	CUDA_CALL(cudaMemset(dest, ch, count));
	return dest;
}

template<typename MatrixType>
float* GPUOperations<MatrixType>::memcpy(void* dest, const void *src, size_t count) const {
	CUDA_CALL(cudaMemcpy(dest, src, count, cudaMemcpyDeviceToDevice));
	return 0;
}

template<typename MatrixType>
void GPUOperations<MatrixType>::free(void* ptr) const {
	if (ptr != 0)
		CUDA_CALL(cudaFree(ptr));
}

template<typename MatrixType>
void GPUOperations<MatrixType>::free_devicememory(void* ptr) const {
	if (ptr != 0)
		CUDA_CALL(cudaFree(ptr));
}

template<typename MatrixType>
float* GPUOperations<MatrixType>::malloc(size_t size) const {
	float* retval = 0;
	cudaError_t err = cudaMalloc(&retval, size);
	CUDA_CALL(err);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed\n");
		retval = 0;
	}
	return retval;
}

template<typename MatrixType>
void GPUOperations<MatrixType>::fill(float* X, const unsigned size, const float value) const {
	int threads, blocks;
	get_grid_sizes(size, &threads, &blocks);
	fill_eltw<<<blocks, threads>>>(X, size, value);
	assert(!cudaGetLastError());
}

template<typename MatrixType>
void GPUOperations<MatrixType>::invert(float* X, const unsigned size) const {
	int threads, blocks;
	get_grid_sizes(size, &threads, &blocks);
	invert_eltw<<<blocks, threads>>>(X, size);
	assert(!cudaGetLastError());
}

template<typename MatrixType>
void GPUOperations<MatrixType>::maximum(float* x, const float value, const unsigned size) const {
	int threads, blocks;
	get_grid_sizes(size, &threads, &blocks);
	maximum_eltw<<<blocks, threads>>>(x, value, size);
	assert(!cudaGetLastError());
}

template<typename MatrixType>
void GPUOperations<MatrixType>::leaky_relu(float* x, const float value, const unsigned size) const {
	int threads, blocks;
	get_grid_sizes(size, &threads, &blocks);
	leaky_relu_eltw<<<blocks, threads>>>(x, value, size);
	assert(!cudaGetLastError());
}

template<typename MatrixType>
void GPUOperations<MatrixType>::sigmoid(float* x, const unsigned size) const {
	int threads, blocks;
	get_grid_sizes(size, &threads, &blocks);
	sigmoid_eltw<<<blocks, threads>>>(x, size);
	assert(!cudaGetLastError());
}

template<typename MatrixType>
void GPUOperations<MatrixType>::tanh(float* x, const unsigned size) const {
	int threads, blocks;
	get_grid_sizes(size, &threads, &blocks);
	tanh_eltw<<<blocks, threads>>>(x, size);
	assert(!cudaGetLastError());
}

template<typename MatrixType>
void GPUOperations<MatrixType>::soft_threshold(float* x, const float alpha, const unsigned size) const {
	int threads, blocks;
	get_grid_sizes(size, &threads, &blocks);
	softthreshold_eltw<<<blocks, threads>>>(x, alpha, size);
	assert(!cudaGetLastError());
}

template<typename MatrixType>
void GPUOperations<MatrixType>::fill_eye(float* X, unsigned n) const {
	memset(X, 0, n * n * sizeof(float));
	axpy(n, 1.0f, ones, 0, X, n + 1);
}

template<typename MatrixType>
void GPUOperations<MatrixType>::invsqrt(float* s, const unsigned n) const {
	int t, b;
	get_grid_sizes(n, &t, &b);
	invsqrt_eltw<<<t, b>>>(s, n);
}

// force compiler to create related class
template class GPUOperations<float*>;
template class GPUOperations<sparse_matrix_csr>;
