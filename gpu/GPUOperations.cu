#include "GPUOperations.h"
#include "GPUCommon.h"

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
