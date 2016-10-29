#ifndef GPU_SPARSE_OPERATIONS_H
#define GPU_SPARSE_OPERATIONS_H

#include "GPUOperations.h"
#include "GPUCommon.h"

class GPUSparseOperations: public GPUOperations<cusparseMatDescr_t> {

	cusparseHandle_t sparseHandle;

public:

	GPUSparseOperations(int n, int m, int k, unsigned long seed, int gpu_id);
	~GPUSparseOperations();

	void calculate_column_variance(cusparseMatDescr_t X, const unsigned nrows, const unsigned ncols, float* variances) const;
	void scale_columns(cusparseMatDescr_t X, const unsigned nrows, const unsigned ncols, float* s) const;
	void scale_rows(cusparseMatDescr_t X, const unsigned nrows, const unsigned ncols, float* s) const;
	void dropout(cusparseMatDescr_t X, const unsigned size, const float dropout_rate) const;
	void add_saltpepper_noise(cusparseMatDescr_t X, const unsigned size, const float noise_rate) const;
	void add_gauss_noise(cusparseMatDescr_t X, const unsigned size, const float noise_rate) const;

};

#endif
