#ifndef GPU_SPARSE_KERNELS_H
#define GPU_SPARSE_KERNELS_H

#include "GPUCommonKernels.h"
#include "SparseMatrix.h"

__global__ void sparse_col_variance_kernel(const sparse_matrix_csr X, float* var, const unsigned nrows, const unsigned ncols);

__global__ void sparse_scale_columns_kernel(sparse_matrix_csr X, float* a, const unsigned nrows, const unsigned ncols);

__global__ void sparse_scale_rows_kernel(sparse_matrix_csr X, float* a, const unsigned nrows, const unsigned ncols);

#endif /*GPU_SPARSE_KERNELS_H*/
