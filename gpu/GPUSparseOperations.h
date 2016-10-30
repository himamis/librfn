#ifndef GPU_SPARSE_OPERATIONS_H
#define GPU_SPARSE_OPERATIONS_H

#include "GPUOperations.h"
#include "GPUCommon.h"
#include "SparseMatrix.h"

class GPUSparseOperations: public GPUOperations<sparse_matrix_csr> {

	cusparseHandle_t sparseHandle;

public:

	GPUSparseOperations(int n, int m, int k, unsigned long seed, int gpu_id);
	~GPUSparseOperations();

	void calculate_column_variance(sparse_matrix_csr X, const unsigned nrows, const unsigned ncols, float* variances) const;
	void scale_columns(sparse_matrix_csr X, const unsigned nrows, const unsigned ncols, float* s) const;
	void scale_rows(sparse_matrix_csr X, const unsigned nrows, const unsigned ncols, float* s) const;
	void dropout(sparse_matrix_csr X, const unsigned size, const float dropout_rate) const;
	void add_saltpepper_noise(sparse_matrix_csr X, const unsigned size, const float noise_rate) const;
	void add_gauss_noise(sparse_matrix_csr X, const unsigned size, const float noise_rate) const;

	sparse_matrix_csr dense_to_sparse_csr(const float *X, const unsigned nrows, const unsigned ncols) const;
};

#endif
