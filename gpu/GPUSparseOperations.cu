#include "GPUSparseOperations.h"
#include "GPUSparseKernels.h"

GPUSparseOperations::GPUSparseOperations(const int n, const int m, const int k, unsigned long seed, int gpu_id) :
		GPUOperations(n, m, k, seed, gpu_id) {
	cusparseStatus_t status = cusparseCreate(&sparseHandle);

	if (status != CUSPARSE_STATUS_SUCCESS) {
		fprintf(stderr, "cuSparse: %d\n", status);
		cudaDeviceReset();
		throw std::runtime_error("cuSparse error");
	}
}

GPUSparseOperations::~GPUSparseOperations() {
	CUSPARSE_CALL(cusparseDestroy(sparseHandle));
}

void GPUSparseOperations::calculate_column_variance(sparse_matrix_csr X, const unsigned nrows, const unsigned ncols,
		float* variances) const {
	int threads, blocks;
	get_grid_sizes(ncols, &threads, &blocks);
	sparse_col_variance_kernel<<<threads, blocks>>>(X, variances, nrows, ncols);
}

void GPUSparseOperations::scale_columns(sparse_matrix_csr X, const unsigned nrows, const unsigned ncols,
		float* s) const {
	int threads, blocks;
	get_grid_sizes(X.nnz, &threads, &blocks);
	sparse_scale_columns_kernel<<<threads, blocks>>>(X, s, nrows, ncols);
}

void GPUSparseOperations::scale_rows(sparse_matrix_csr X, const unsigned nrows, const unsigned ncols, float* s) const {
	int threads, blocks;
	get_grid_sizes(X.nnz, &threads, &blocks);
	sparse_scale_rows_kernel<<<threads, blocks>>>(X, s, nrows, ncols);
}

void GPUSparseOperations::dropout(sparse_matrix_csr X, const unsigned size, const float dropout_rate) const {
	dropout_eltw<<<RNG_BLOCKS, RNG_THREADS>>>(X.values, X.nnz, dropout_rate, rng_state);
	assert(!cudaGetLastError());
}

void GPUSparseOperations::add_saltpepper_noise(sparse_matrix_csr X, const unsigned size, const float noise_rate) const {
	saltpepper_noise_eltw<<<RNG_BLOCKS, RNG_THREADS>>>(X.values, X.nnz, noise_rate, rng_state);
	assert(!cudaGetLastError());
}

void GPUSparseOperations::add_gauss_noise(sparse_matrix_csr X, const unsigned size, const float noise_rate) const {
	gauss_noise_eltw<<<RNG_BLOCKS, RNG_THREADS>>>(X.values, X.nnz, noise_rate, rng_state);
	assert(!cudaGetLastError());
}

sparse_matrix_csr GPUSparseOperations::dense_to_sparse_csr(const float* X, const unsigned nrows,
		const unsigned ncols) const {
	sparse_matrix_csr sparse;
	size_t size = nrows * ncols * sizeof(float);
	float *X_host = std::malloc(size);
	to_host(X, X_host, size);

	const float eps = 1e-5;

	for (unsigned i = 0; i < nrows * ncols; ++i) {
		if (!isNearlyZero(X_host[i])) {
			sparse.nnz += 1;
		}
	}
	sparse.m = nrows;

	float *sp_values = std::malloc(sparse.nnz * sizeof(float));
	float *sp_column_indices = std::malloc(sparse.nnz * sizeof(float));
	float *sp_index_pointers = std::malloc((nrows + 1) * sizeof(float));

	sp_index_pointers[0] = 0;

	unsigned index = 0;
	for (unsigned i = 0; i < nrows; ++i) {
		sp_index_pointers[i + 1] = sp_index_pointers[i];
		for (unsigned j = 0; j < ncols; ++j) {
			float value = X_host[i * ncols + j];
			if (!isNearlyZero(value)) {
				sp_values[index] = value;
				sp_column_indices[index] = j;
				sp_index_pointers[i + 1] += 1;
			}
		}
	}

	sparse.values = to_device(sp_values, sparse.nnz * sizeof(float));
	sparse.index_pointers = to_device(sp_index_pointers, (nrows + 1) * sizeof(float));
	sparse.column_indices = to_device(sp_column_indices, sparse.nnz * sizeof(float));

	return sparse;
}

bool isNearlyZero(float x) {
	const float epsilon = 1e-5;
	return std::abs(x) <= epsilon;
}
