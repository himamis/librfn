/*
 Copyright © 2015 Thomas Unterthiner
 Licensed under GPL, version 2 or a later (see LICENSE.txt)
 */
#ifndef GPU_DENSE_OPERATIONS_H
#define GPU_DENSE_OPERATIONS_H

#include "GPUCommon.h"
#include "GPUOperations.h"

class GPUDenseOperations: public GPUOperations<float*> {

public:

	GPUDenseOperations(int n, int m, int k, unsigned long seed, int gpu_id);
	virtual ~GPUDenseOperations();

	float* to_device(const float* src, size_t size) const;

	float* to_host(float* src, float* dst, size_t size) const {
		CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
		free(src);
		return dst;
	}

	float* copy_to_host(const float* src, float* dst, size_t size) const {
		CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
		return dst;
	}

	void fill_eye(float* X, unsigned n) const;
	void maximum(float* x, const float value, const unsigned size) const;
	void leaky_relu(float* x, const float value, const unsigned size) const;
	void tanh(float* x, const unsigned size) const;
	void sigmoid(float* x, const unsigned size) const;
	void soft_threshold(float* x, const float alpha, const unsigned size) const;
	void calculate_column_variance(float* X, const unsigned nrows, const unsigned ncols, float* variances) const;
	void invsqrt(float* s, const unsigned n) const;
	void scale_columns(float* X, const unsigned nrows, const unsigned ncols, float* s) const;
	void scale_rows(float* X, const unsigned nrows, const unsigned ncols, float* s) const;
	void dropout(float* X, const unsigned size, const float dropout_rate) const;
	void add_saltpepper_noise(float* X, const unsigned size, const float noise_rate) const;
	void add_gauss_noise(float* X, const unsigned size, const float noise_rate) const;
	void invert(float* X, const unsigned size) const;

	// Useful for debugging
	void printMatrixCM(const float* a, int n, int m, const char* fmt);
	void printMatrixRM(const float* a, int n, int m, const char* fmt);
};

#endif /*GPU_DENSE_OPERATIONS_H*/
