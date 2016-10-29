#ifndef GPU_SPARSE_KERNELS_H
#define GPU_SPARSE_KERNELS_H

#include "GPUCommonKernels.h"

__global__ void sparse_dropout_eltw(cusparseMatDescr_t x, const unsigned size, const float dropout_rate, curandState* rng_state);

__global__ void sparse_saltpepper_noise_eltw(cusparseMatDescr_t x, const unsigned size, const float noise_rate, curandState* rng_state);

__global__ void sparse_gauss_noise_eltw(cusparseMatDescr_t x, const unsigned size, const float noise_rate, curandState* rng_state);

__global__ void sparse_col_variance_kernel(const cusparseMatDescr_t X, float* var, const unsigned nrows, const unsigned ncols);

__global__ void sparse_scale_columns_kernel(cusparseMatDescr_t X, float* a, const unsigned nrows, const unsigned ncols);

__global__ void sparse_scale_rows_kernel(cusparseMatDescr_t X, float* a, const unsigned nrows, const unsigned ncols);

#endif /*GPU_SPARSE_KERNELS_H*/
