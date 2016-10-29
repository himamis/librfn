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

void GPUSparseOperations::fill_eye(cusparseMatDescr_t, unsigned int n) const {

}

void GPUSparseOperations::gemm(const char *transa, const char *transb, const int m, const int n, const int k,
		const float alpha, const float *a, const int lda, const float *b, const int ldb, const float beta, float *c,
		const int ldc) const {
	//CUSPARSE_CALL(cusparseSgemmi())

}

void GPUSparseOperations::scale_rows(cusparseMatDescr_t, const unsigned nrows, const unsigned ncols, float* s) const {
	int threads, blocks;
	get_grid_sizes(ncols * nrows, &threads, &blocks);
}
