#ifndef GPU_COMMON_KERNELS_H
#define GPU_COMMON_KERNELS_H

#include <curand_kernel.h>

__global__ void setup_rng(curandState* rng_state, unsigned long seed);

#endif /*GPU_COMMON_KERNELS_H*/
