#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusolverSp.h>

#include "gpu_common.h"

static const char* cusparseErrorString(cusparseStatus_t error) {
	switch (error) {
	case CUSPARSE_STATUS_SUCCESS:
		return "CUSPARSE_STATUS_SUCCESS";
	case CUSPARSE_STATUS_NOT_INITIALIZED:
		return "CUSPARSE_STATUS_NOT_INITIALIZED";
	case CUSPARSE_STATUS_ALLOC_FAILED:
		return "CUSPARSE_STATUS_ALLOC_FAILED";
	case CUSPARSE_STATUS_INVALID_VALUE:
		return "CUSPARSE_STATUS_INVALID_VALUE";
	case CUSPARSE_STATUS_ARCH_MISMATCH:
		return "CUSPARSE_STATUS_ARCH_MISMATCH";
	case CUSPARSE_STATUS_MAPPING_ERROR:
		return "CUSPARSE_STATUS_MAPPING_ERROR";
	case CUSPARSE_STATUS_EXECUTION_FAILED:
		return "CUSPARSE_STATUS_EXECUTION_FAILED";
	case CUSPARSE_STATUS_INTERNAL_ERROR:
		return "CUSPARSE_STATUS_INTERNAL_ERROR";
	case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
		return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
	case CUSPARSE_STATUS_ZERO_PIVOT:
		return "CUSPARSE_STATUS_ZERO_PIVOT";
	default:
		return "<unknown>";
	}
}

#ifndef DNDEBUG

#define CUSPARSE_CALL(ans) { cusparseAssert((ans), __FILE__, __LINE__); }
inline void cusparseAssert(cusparseStatus_t code, const char *file, int line) {
	// printf("%d (%s:%d)\n", code, file, line);
	if (code != CUSPARSE_STATUS_SUCCESS) {
		fprintf(stderr, "CUSPARSE Error: %s %s:%d\n", cusparseErrorString(code),
				file, line);
		exit(code);
	}
}

#else
#define CUBLAS_CALL(ans) (ans)
#endif

class GPU_SparseOperations {

	cusparseHandle_t sparseHandle;
	cusolverSpHandle_t solverHandle;
	curandState* rngState;

	int* devinfo; // cuSOLVER error reporting

public:
	float* ones;

	GPU_SparseOperations(int n, int m, int k, unsigned long seed, int gpu_id);
	~GPU_SparseOperations();

	void fill_eye(float* X, unsigned n) const;
	void fill(float* X, const unsigned size, const float value) const;
	void maximum(float* x, const float value, const unsigned size) const;
	void leaky_relu(float* x, const float value, const unsigned size) const;
	void tanh(float* x, const unsigned size) const;
	void sigmoid(float* x, const unsigned size) const;
	void soft_threshold(float* x, const float alpha, const unsigned size) const;
	void calculate_column_variance(const float* X, const unsigned nrows,
	                               const unsigned ncols, float* variances);
	void invsqrt(float* s, const unsigned n) const;
	void scale_columns(float* X, const unsigned nrows, const unsigned ncols, float* s) const;
	void scale_rows(float* X, const unsigned nrows, const unsigned ncols, float* s) const;
	void dropout(float* X, const unsigned size, const float dropout_rate) const;
	void add_saltpepper_noise(float* X, const unsigned size, const float noise_rate) const;
	void add_gauss_noise(float* X, const unsigned size, const float noise_rate) const;

	void invert(float* X, const unsigned size) const;

};
