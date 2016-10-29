#ifndef GPU_COMMON_H
#define GPU_COMMON_H

#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <cusolverDn.h>
#include <stdexcept>
#include <curand.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <cusolverSp.h>
#include <cusparse.h>


static const int RNG_THREADS = 128;
static const int RNG_BLOCKS = 128;

#ifndef DNDEBUG

#define CUDA_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = false) {
	if (code != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s %s:%d\n", cudaGetErrorString(code), file, line);
		if (abort)
			exit(code);
	}
}

static const char* cusolverErrorString(cusolverStatus_t error) {
	switch (error) {
	case CUSOLVER_STATUS_SUCCESS:
		return "CUSOLVER_STATUS_SUCCESS";
	case CUSOLVER_STATUS_NOT_INITIALIZED:
		return "CUSOLVER_STATUS_NOT_INITIALIZED";
	case CUSOLVER_STATUS_ALLOC_FAILED:
		return "CUSOLVER_STATUS_ALLOC_FAILED";
	case CUSOLVER_STATUS_INVALID_VALUE:
		return "CUSOLVER_STATUS_INVALID_VALUE";
	case CUSOLVER_STATUS_ARCH_MISMATCH:
		return "CUSOLVER_STATUS_ARCH_MISMATCH";
	case CUSOLVER_STATUS_EXECUTION_FAILED:
		return "CUSOLVER_STATUS_EXECUTION_FAILED";
	case CUSOLVER_STATUS_INTERNAL_ERROR:
		return "CUSOLVER_STATUS_INTERNAL_ERROR";
	case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
		return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
	default:
		return "<unknown>";
	}
}

static const char* cublasErrorString(cublasStatus_t error) {
	switch (error) {
	case CUBLAS_STATUS_SUCCESS:
		return "CUBLAS_STATUS_SUCCESS";
	case CUBLAS_STATUS_NOT_INITIALIZED:
		return "CUBLAS_STATUS_NOT_INITIALIZED";
	case CUBLAS_STATUS_ALLOC_FAILED:
		return "CUBLAS_STATUS_ALLOC_FAILED";
	case CUBLAS_STATUS_INVALID_VALUE:
		return "CUBLAS_STATUS_INVALID_VALUE";
	case CUBLAS_STATUS_ARCH_MISMATCH:
		return "CUBLAS_STATUS_ARCH_MISMATCH";
	case CUBLAS_STATUS_MAPPING_ERROR:
		return "CUBLAS_STATUS_MAPPING_ERROR";
	case CUBLAS_STATUS_EXECUTION_FAILED:
		return "CUBLAS_STATUS_EXECUTION_FAILED";
	case CUBLAS_STATUS_INTERNAL_ERROR:
		return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
	case CUBLAS_STATUS_NOT_SUPPORTED:
		return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
	default:
		return "<unknown>";
	}
}

#define CUSOLVER_CALL(ans) { cusolverAssert((ans), __FILE__, __LINE__); }
inline void cusolverAssert(cusolverStatus_t code, const char *file, int line) {
	//printf("%d (%s:%d)\n", code, file, line);
	if (code != CUSOLVER_STATUS_SUCCESS) {
		fprintf(stderr, "CUBLAS Error: %s %s:%d\n", cusolverErrorString(code), file, line);
		exit(code);
	}
}

#define CUBLAS_CALL(ans) { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t code, const char *file, int line) {
	//printf("%d (%s:%d)\n", code, file, line);
	if (code != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "CUBLAS Error: %s %s:%d\n", cublasErrorString(code), file, line);
		exit(code);
	}
}


#define CUSPARSE_CALL(ans) { cusparseAssert((ans), __FILE__, __LINE__); }
inline void cusparseAssert(cusparseStatus_t code, const char *file, int line) {
	// printf("%d (%s:%d)\n", code, file, line);
	if (code != CUSPARSE_STATUS_SUCCESS) {
		fprintf(stderr, "CUSPARSE Error: %s %s:%d\n", cusparseErrorString(code),
				file, line);
		exit(code);
	}
}

#else
#define CUDA_CALL(ans) (ans)
#define CUSOLVER_CALL(ans) (ans)
#define CUBLAS_CALL(ans) (ans)
#define CUSPARSE_CALL(ans) (ans)
#endif

/*
 * Returns a GPU id based on some criteria.
 */
int get_gpu_id();

/*
 * Taken from PyCUDA
 */
void get_grid_sizes(int problemsize, int* blocks, int* threads);

#endif /* GPU_COMMON_H */
