#ifndef GPU_SPARSE_OPERATIONS_H
#define GPU_SPARSE_OPERATIONS_H

#include "GPUCommon.h"
#include "GPUOperations.h"

class GPUSparseOperations: public GPUOperations<cusparseMatDescr_t> {

	cusparseHandle_t sparseHandle;
	cusolverSpHandle_t solverHandle;
	curandState* rngState;

	int* devinfo; // cuSOLVER error reporting

public:
	//float* ones;

	GPUSparseOperations(int n, int m, int k, unsigned long seed, int gpu_id);
	~GPUSparseOperations();

	void fill_eye(cusparseMatDescr_t, unsigned n) const;
	void fill(cusparseMatDescr_t, const unsigned size, const float value) const;
	void maximum(cusparseMatDescr_t, const float value, const unsigned size) const;
	void leaky_relu(cusparseMatDescr_t, const float value, const unsigned size) const;
	void tanh(cusparseMatDescr_t, const unsigned size) const;
	void sigmoid(cusparseMatDescr_t, const unsigned size) const;
	void soft_threshold(cusparseMatDescr_t, const float alpha, const unsigned size) const;
	void calculate_column_variance(cusparseMatDescr_t, const unsigned nrows,
	                               const unsigned ncols, float* variances) const;
	void invsqrt(cusparseMatDescr_t, const unsigned n) const;
	void scale_columns(cusparseMatDescr_t, const unsigned nrows, const unsigned ncols, float* s) const;
	void scale_rows(cusparseMatDescr_t, const unsigned nrows, const unsigned ncols, float* s) const;
	void dropout(cusparseMatDescr_t, const unsigned size, const float dropout_rate) const;
	void add_saltpepper_noise(cusparseMatDescr_t, const unsigned size, const float noise_rate) const;
	void add_gauss_noise(cusparseMatDescr_t, const unsigned size, const float noise_rate) const;

	void invert(cusparseMatDescr_t, const unsigned size) const;

};

#endif
