#include "GPUDenseKernels.h"

__global__ void dense_dropout_eltw(float* x, const unsigned size, const float dropout_rate, curandState* rng_state) {
	const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned num_threads = gridDim.x * blockDim.x;
	curandState localState = rng_state[tid];
	for (unsigned i = tid; i < size; i += num_threads)
		x[i] = (curand_uniform(&localState) < dropout_rate) ? 0.0 : x[i];
	rng_state[tid] = localState;
}

__global__ void dense_saltpepper_noise_eltw(float* x, const unsigned size, const float noise_rate, curandState* rng_state) {
	const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned num_threads = gridDim.x * blockDim.x;
	curandState localState = rng_state[tid];
	for (unsigned i = tid; i < size; i += num_threads)
		if (curand_uniform(&localState) < noise_rate) {
			x[i] = (curand_uniform(&localState) < 0.5f) ? 0.0f : 1.0f;
		}
	rng_state[tid] = localState;

}

__global__ void dense_gauss_noise_eltw(float* x, const unsigned size, const float noise_rate, curandState* rng_state) {
	const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned num_threads = gridDim.x * blockDim.x;
	curandState localState = rng_state[tid];
	for (unsigned i = tid; i < size; i += num_threads)
		x[i] += curand_normal(&localState) * noise_rate;
	rng_state[tid] = localState;

}

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
