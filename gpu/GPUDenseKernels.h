#ifndef GPU_DENSE_KERNELS_H
#define GPU_DENSE_KERNELS_H

#include "GPUCommonKernels.h"

__global__ void dropout_eltw(float* x, const unsigned size, const float dropout_rate, curandState* rng_state);

__global__ void saltpepper_noise_eltw(float* x, const unsigned size, const float noise_rate, curandState* rng_state);

__global__ void gauss_noise_eltw(float* x, const unsigned size, const float noise_rate, curandState* rng_state);

__global__ void leaky_relu_eltw(float* x, const float value, const unsigned size);

__global__ void maximum_eltw(float* x, const float value, const unsigned size);

__global__ void sigmoid_eltw(float* x, const unsigned size);

__global__ void tanh_eltw(float* x, const unsigned size);

__global__ void softthreshold_eltw(float* x, float alpha, const unsigned size);

__global__ void fill_eltw(float* x, const unsigned size, const float value);

__global__ void invert_eltw(float* x, const unsigned size);

__global__ void col_variance_kernel(const float* X, float* var, const unsigned nrows, const unsigned ncols);

__global__ void invsqrt_eltw(float* x, const unsigned k);

__global__ void scale_columns_kernel(float* X, float* a, const unsigned nrows, const unsigned ncols);

__global__ void scale_rows_kernel(float* X, float* a, const unsigned nrows, const unsigned ncols);

#endif /*GPU_DENSE_KERNELS_H*/
