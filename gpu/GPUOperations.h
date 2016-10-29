#ifndef GPU_OPERATIONS_H
#define GPU_OPERATIONS_H

#include "GPUCommon.h"
#include <map>

inline cublasFillMode_t uplo_to_cublas(const char* uplo) {
	return tolower(uplo[0]) == 'l' ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
}

template<typename MatrixType>
class GPUOperations {

	cublasHandle_t handle;
	cusolverDnHandle_t cudense_handle;


	std::map<int, float*> buffer_map; // keeps track of buffers allocated for potrf
	int* devinfo; // cuSOLVER error reporting

public:
	curandState* rng_state;
	float* ones;

	GPUOperations(const int n, const int m, const int k, unsigned long seed, int gpu_id);
	virtual ~GPUOperations() = 0;

	virtual void calculate_column_variance(MatrixType X, const unsigned nrows, const unsigned ncols,
			float* variances) const = 0;
	virtual void scale_columns(MatrixType X, const unsigned nrows, const unsigned ncols, float* s) const = 0;
	virtual void scale_rows(MatrixType X, const unsigned nrows, const unsigned ncols, float* s) const = 0;
	virtual void dropout(MatrixType X, const unsigned size, const float dropout_rate) const = 0;
	virtual void add_saltpepper_noise(MatrixType X, const unsigned size, const float noise_rate) const = 0;
	virtual void add_gauss_noise(MatrixType X, const unsigned size, const float noise_rate) const = 0;

	void fill_eye(float* X, unsigned n) const;
	void maximum(float* x, const float value, const unsigned size) const;
	void leaky_relu(float* x, const float value, const unsigned size) const;
	void tanh(float* x, const unsigned size) const;
	void sigmoid(float* x, const unsigned size) const;
	void soft_threshold(float* x, const float alpha, const unsigned size) const;
	void invsqrt(float* s, const unsigned n) const;
	void invert(float* X, const unsigned size) const;

	void fill(float * x, const unsigned size, const float value) const;
	void* memset(void* dest, int ch, size_t count) const;
	float* memcpy(void* dest, const void *src, size_t count) const;
	void free(void* ptr) const;
	void free_devicememory(void* ptr) const;
	float* malloc(size_t size) const;

	void gemm(const char *transa, const char *transb, const int m, const int n, const int k, const float alpha,
			const float *a, const int lda, const float *b, const int ldb, const float beta, float *c,
			const int ldc) const {
		cublasOperation_t ta = tolower(transa[0]) == 'n' ? CUBLAS_OP_N : CUBLAS_OP_T;
		cublasOperation_t tb = tolower(transb[0]) == 'n' ? CUBLAS_OP_N : CUBLAS_OP_T;
		CUBLAS_CALL(cublasSgemm(handle, ta, tb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc));
	}

	void dgmm(const char* mode, const int m, const int n, const float* A, int lda, const float* x, int incx, float* C,
			int ldc) const {
		cublasSideMode_t lr = mode[0] == 'l' ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
		CUBLAS_CALL(cublasSdgmm(handle, lr, m, n, A, lda, x, incx, C, ldc));
	}

	void symm(const char *side, const char *uplo, const int m, const int n, const float alpha, const float *a,
			const int lda, const float *b, const int ldb, const float beta, float *c, const int ldc) const {
		cublasSideMode_t s = tolower(side[0]) == 'l' ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
		cublasFillMode_t ul = uplo_to_cublas(uplo);
		CUBLAS_CALL(cublasSsymm(handle, s, ul, m, n, &alpha, a, lda, b, ldb, &beta, c, ldc));
	}

	void axpy(const int n, const float alpha, const float* x, const int incx, float *y, const int incy) const {
		CUBLAS_CALL(cublasSaxpy(handle, n, &alpha, x, incx, y, incy));
	}

	int potrf(const char *uplo, int n, float* a, int lda) {
		cublasFillMode_t ul = uplo_to_cublas(uplo);
		int bufsize = 0;
		int info = 0;
		CUSOLVER_CALL(cusolverDnSpotrf_bufferSize(cudense_handle, ul, n, a, lda, &bufsize));

		// See if we already have a buffer of correct size, otherwise allocate
		float* buffer = 0;
		auto it = buffer_map.find(bufsize);
		if (it != buffer_map.end()) {
			buffer = it->second;
		} else {
			buffer = malloc(bufsize * sizeof(float));
			buffer_map[bufsize] = buffer;
		}

		CUSOLVER_CALL(cusolverDnSpotrf(cudense_handle, ul, n, a, lda, buffer, bufsize, devinfo));
		CUDA_CALL(cudaMemcpy(&info, devinfo, sizeof(info), cudaMemcpyDeviceToHost));
		return info;
	}

	int potrs(const char *uplo, int n, int nrhs, float * a, int lda, float *b, int ldb) const {
		int info;
		cublasFillMode_t ul = uplo_to_cublas(uplo);
		CUSOLVER_CALL(cusolverDnSpotrs(cudense_handle, ul, n, nrhs, a, lda, b, ldb, devinfo));
		CUDA_CALL(cudaMemcpy(&info, devinfo, sizeof(info), cudaMemcpyDeviceToHost));
		return info;
	}

	int posv(const char *uplo, int n, int nrhs, float * a, int lda, float *b, int ldb) {
		int info = potrf(uplo, n, a, lda);
		if (info == 0)
			info = potrs(uplo, n, nrhs, a, lda, b, ldb);
		return info;
	}
};

#endif
