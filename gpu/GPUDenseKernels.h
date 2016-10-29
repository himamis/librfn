#ifndef GPU_DENSE_KERNELS_H
#define GPU_DENSE_KERNELS_H

#include "GPUCommonKernels.h"

__global__ void dropout_eltw(float* x, const unsigned size, const float dropout_rate, curandState* rng_state);

__global__ void saltpepper_noise_eltw(float* x, const unsigned size, const float noise_rate, curandState* rng_state);

__global__ void gauss_noise_eltw(float* x, const unsigned size, const float noise_rate, curandState* rng_state);

__global__ void col_variance_kernel(const float* X, float* var, const unsigned nrows, const unsigned ncols);

__global__ void scale_columns_kernel(float* X, float* a, const unsigned nrows, const unsigned ncols);

__global__ void scale_rows_kernel(float* X, float* a, const unsigned nrows, const unsigned ncols);

#endif /*GPU_DENSE_KERNELS_H*/
