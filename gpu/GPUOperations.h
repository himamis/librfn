#ifndef GPU_OPERATIONS_H
#define GPU_OPERATIONS_H

template<typename MatrixType>
class GPUOperations {

public:

	virtual ~GPUOperations() = 0;

	virtual void fill_eye(MatrixType X, unsigned n) const = 0;
	virtual void fill(MatrixType X, const unsigned size, const float value) const = 0;
	virtual void maximum(MatrixType x, const float value, const unsigned size) const = 0;
	virtual void leaky_relu(MatrixType x, const float value, const unsigned size) const = 0;
	virtual void tanh(MatrixType x, const unsigned size) const = 0;
	virtual void sigmoid(MatrixType x, const unsigned size) const = 0;
	virtual void soft_threshold(MatrixType x, const float alpha, const unsigned size) const = 0;
	virtual void calculate_column_variance(MatrixType X, const unsigned nrows, const unsigned ncols,
			float* variances) const = 0;
	virtual void invsqrt(MatrixType s, const unsigned n) const = 0;
	virtual void scale_columns(MatrixType X, const unsigned nrows, const unsigned ncols, float* s) const = 0;
	virtual void scale_rows(MatrixType X, const unsigned nrows, const unsigned ncols, float* s) const = 0;
	virtual void dropout(MatrixType X, const unsigned size, const float dropout_rate) const = 0;
	virtual void add_saltpepper_noise(MatrixType X, const unsigned size, const float noise_rate) const = 0;
	virtual void add_gauss_noise(MatrixType X, const unsigned size, const float noise_rate) const = 0;
	virtual void invert(MatrixType X, const unsigned size) const = 0;

	void* memset(void* dest, int ch, size_t count) const;
	float* memcpy(void* dest, const void *src, size_t count) const;
	void free(void* ptr) const;
	void free_devicememory(void* ptr) const;
	float* malloc(size_t size) const;
};

#endif
