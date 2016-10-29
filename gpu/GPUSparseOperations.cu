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

void GPUSparseOperations::calculate_column_variance(cusparseMatDescr_t X, const unsigned nrows, const unsigned ncols,
			float* variances) const {

}

void GPUSparseOperations::scale_columns(cusparseMatDescr_t X, const unsigned nrows, const unsigned ncols, float* s) const {

}

void GPUSparseOperations::scale_rows(cusparseMatDescr_t X, const unsigned nrows, const unsigned ncols, float* s) const {

}

void GPUSparseOperations::dropout(cusparseMatDescr_t X, const unsigned size, const float dropout_rate) const {

}

void GPUSparseOperations::add_saltpepper_noise(cusparseMatDescr_t X, const unsigned size, const float noise_rate) const {

}

void GPUSparseOperations::add_gauss_noise(cusparseMatDescr_t X, const unsigned size, const float noise_rate) const {

}
