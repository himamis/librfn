#include "GPUCommonKernels.h"

__global__ void dropout_eltw(float* x, const unsigned size, const float dropout_rate, curandState* rng_state) {
	const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned num_threads = gridDim.x * blockDim.x;
	curandState localState = rng_state[tid];
	for (unsigned i = tid; i < size; i += num_threads)
		x[i] = (curand_uniform(&localState) < dropout_rate) ? 0.0 : x[i];
	rng_state[tid] = localState;
}

__global__ void saltpepper_noise_eltw(float* x, const unsigned size, const float noise_rate,
		curandState* rng_state) {
	const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned num_threads = gridDim.x * blockDim.x;
	curandState localState = rng_state[tid];
	for (unsigned i = tid; i < size; i += num_threads)
		if (curand_uniform(&localState) < noise_rate) {
			x[i] = (curand_uniform(&localState) < 0.5f) ? 0.0f : 1.0f;
		}
	rng_state[tid] = localState;

}

__global__ void gauss_noise_eltw(float* x, const unsigned size, const float noise_rate, curandState* rng_state) {
	const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned num_threads = gridDim.x * blockDim.x;
	curandState localState = rng_state[tid];
	for (unsigned i = tid; i < size; i += num_threads)
		x[i] += curand_normal(&localState) * noise_rate;
	rng_state[tid] = localState;

}

__global__ void setup_rng(curandState* rng_state, unsigned long seed) {
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, tid, 0, &rng_state[tid]);
}

__global__ void invsqrt_eltw(float* x, const unsigned k) {
	const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned num_threads = blockDim.x * gridDim.x;
	for (unsigned i = tid; i < k; i += num_threads) {
		x[i] = (x[i] > 1e-7) ? rsqrtf(x[i]) : 1.0;
	}
}

__global__ void leaky_relu_eltw(float* x, const float value, const unsigned size) {
	const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned num_threads = gridDim.x * blockDim.x;
	for (unsigned i = tid; i < size; i += num_threads) {
		x[i] = (x[i] < 0.0f) ? x[i] * value : x[i];
	}
}

__global__ void maximum_eltw(float* x, const float value, const unsigned size) {
	const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned num_threads = gridDim.x * blockDim.x;
	for (unsigned i = tid; i < size; i += num_threads) {
		x[i] = fmaxf(x[i], value);
	}
}

__global__ void sigmoid_eltw(float* x, const unsigned size) {
	const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned num_threads = gridDim.x * blockDim.x;
	for (unsigned i = tid; i < size; i += num_threads) {
		x[i] = 1 / (1 + __expf(-x[i]));
	}
}

__global__ void tanh_eltw(float* x, const unsigned size) {
	const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned num_threads = gridDim.x * blockDim.x;
	for (unsigned i = tid; i < size; i += num_threads) {
		x[i] = tanhf(x[i]);
	}
}

__global__ void softthreshold_eltw(float* x, float alpha, const unsigned size) {
	const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned num_threads = gridDim.x * blockDim.x;
	for (unsigned i = tid; i < size; i += num_threads) {
		const float f = x[i];
		x[i] = f > 0 ? fmaxf(0., f - alpha) : fminf(0., f + alpha);
	}
}

__global__ void fill_eltw(float* x, const unsigned size, const float value) {
	const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned num_threads = gridDim.x * blockDim.x;
	for (unsigned i = tid; i < size; i += num_threads) {
		x[i] = value;
	}
}

__global__ void invert_eltw(float* x, const unsigned size) {
	const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned num_threads = gridDim.x * blockDim.x;
	for (unsigned i = tid; i < size; i += num_threads) {
		x[i] = 1.0f / x[i];
	}
}
