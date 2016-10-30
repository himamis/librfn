#include "GPUSparseKernels.h"

__global__ void sparse_col_variance_kernel(const sparse_matrix_csr X, float* var, const unsigned nrows,
		const unsigned ncols) {
	const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned num_threads = blockDim.x * gridDim.x;
	for (unsigned i = tid; i < ncols; i += num_threads) {
		var[i] = 0.0;
		for (unsigned j = 0; j < X.nnz; ++j) {
			if (X.column_indices[j] == i) {
				var[i] += X.values[j];
			}
		}
		float m = var[i] / nrows;
		var[i] = 0.0;
		for (unsigned j = 0; j < X.nnz; ++j) {
			if (X.column_indices[j] == i) {
				float tmp = X.values[j] - m;
				var[i] += tmp * tmp;
			}
		}
		var[i] /= nrows;
	}
}

__global__ void sparse_scale_columns_kernel(sparse_matrix_csr X, float* a, const unsigned nrows, const unsigned ncols) {
	const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned num_threads = blockDim.x * gridDim.x;
	for (unsigned i = tid; i < X.nnz; i += num_threads) {
		X.values[i] *= a[X.column_indices[i]];
	}
}

__global__ void sparse_scale_rows_kernel(sparse_matrix_csr X, float* a, const unsigned nrows, const unsigned ncols) {
	//const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	// calculate row index from tid
	// TODO
}
