#include "GPUCommonKernels.h"

__global__ void setup_rng(curandState* rng_state, unsigned long seed) {
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, tid, 0, &rng_state[tid]);
}
