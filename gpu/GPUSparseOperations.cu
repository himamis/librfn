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

sparse_matrix_csr GPUSparseOperations::dense_to_sparse_csr(const float* X, const unsigned nrows, const unsigned ncols) const {
	sparse_matrix_csr sparse;



	return sparse;
}
