#include "GPUDenseKernels.h"

__global__ void dense_col_variance_kernel(const float* X, float* var, const unsigned nrows, const unsigned ncols) {
	const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned num_threads = blockDim.x * gridDim.x;
	for (unsigned i = tid; i < ncols; i += num_threads) {
		var[i] = 0.0;
		for (unsigned j = 0; j < nrows; ++j) {
			var[i] += X[j * ncols + i];
		}
		float m = var[i] / nrows;
		var[i] = 0.0;
		for (unsigned j = 0; j < nrows; ++j) {
			float tmp = X[j * ncols + i] - m;
			var[i] += tmp * tmp;
		}
		var[i] /= nrows;
	}
}

__global__ void dense_scale_columns_kernel(float* X, float* a, const unsigned nrows, const unsigned ncols) {
	const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned num_threads = blockDim.x * gridDim.x;
	for (unsigned i = tid; i < ncols * nrows; i += num_threads) {
		X[i] *= a[i % ncols];
	}
}

__global__ void dense_scale_rows_kernel(float* X, float* a, const unsigned nrows, const unsigned ncols) {
	const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned num_threads = blockDim.x * gridDim.x;
	for (unsigned i = tid; i < ncols * nrows; i += num_threads) {
		X[i] *= a[i / ncols];
	}
}
