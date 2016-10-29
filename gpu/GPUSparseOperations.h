#ifndef GPU_SPARSE_OPERATIONS_H
#define GPU_SPARSE_OPERATIONS_H

#include "GPUOperations.h"
#include "GPUCommon.h"

class GPUSparseOperations: public GPUOperations<cusparseMatDescr_t> {

	cusparseHandle_t sparseHandle;

public:

	GPUSparseOperations(int n, int m, int k, unsigned long seed, int gpu_id);
	~GPUSparseOperations();

	virtual void gemm(const char *transa, const char *transb, const int m, const int n, const int k, const float alpha,
			const float *a, const int lda, const float *b, const int ldb, const float beta, float *c,
			const int ldc) const;
	virtual void dgmm(const char* mode, const int m, const int n, const float* A, int lda, const float* x, int incx,
			float* C, int ldc) const;
	virtual void symm(const char *side, const char *uplo, const int m, const int n, const float alpha, const float *a,
			const int lda, const float *b, const int ldb, const float beta, float *c, const int ldc) const;
	virtual void axpy(const int n, const float alpha, const float* x, const int incx, float *y, const int incy) const;
	virtual int potrf(const char *uplo, int n, float* a, int lda) const;
	virtual int potrs(const char *uplo, int n, int nrhs, float * a, int lda, float *b, int ldb) const;
	virtual int posv(const char *uplo, int n, int nrhs, float * a, int lda, float *b, int ldb) const;

	virtual void fill_eye(cusparseMatDescr_t, unsigned n) const;
	virtual void maximum(cusparseMatDescr_t, const float value, const unsigned size) const;
	virtual void leaky_relu(cusparseMatDescr_t, const float value, const unsigned size) const;
	virtual void tanh(cusparseMatDescr_t, const unsigned size) const;
	virtual void sigmoid(cusparseMatDescr_t, const unsigned size) const;
	virtual void soft_threshold(cusparseMatDescr_t, const float alpha, const unsigned size) const;
	virtual void calculate_column_variance(cusparseMatDescr_t, const unsigned nrows, const unsigned ncols,
			float* variances) const;
	virtual void invsqrt(cusparseMatDescr_t, const unsigned n) const;
	virtual void scale_columns(cusparseMatDescr_t, const unsigned nrows, const unsigned ncols, float* s) const;
	virtual void scale_rows(cusparseMatDescr_t, const unsigned nrows, const unsigned ncols, float* s) const;
	virtual void dropout(cusparseMatDescr_t, const unsigned size, const float dropout_rate) const;
	virtual void add_saltpepper_noise(cusparseMatDescr_t, const unsigned size, const float noise_rate) const;
	virtual void add_gauss_noise(cusparseMatDescr_t, const unsigned size, const float noise_rate) const;

	virtual void invert(cusparseMatDescr_t, const unsigned size) const;

};

#endif
