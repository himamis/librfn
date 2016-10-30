#ifndef SPARSE_MATRIX_H_
#define SPARSE_MATRIX_H_

struct sparse_matrix_csr {
	float* values;
	unsigned* column_indices;
	unsigned* index_pointers;
	unsigned nnz;
	unsigned m;
};



#endif /* SPARSE_MATRIX_H_ */
