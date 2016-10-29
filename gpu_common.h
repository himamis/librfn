#ifndef GPU_COMMON
#define GPU_COMMON

#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <cusolverDn.h>
#include <stdexcept>
#include <curand.h>
#include <curand_kernel.h>



#ifndef DNDEBUG

#define CUDA_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false) {
   if (code != cudaSuccess) {
      fprintf(stderr,"CUDA Error: %s %s:%d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

static const char* cusolverErrorString(cusolverStatus_t error) {
    switch (error) {
        case CUSOLVER_STATUS_SUCCESS: return "CUSOLVER_STATUS_SUCCESS";
        case CUSOLVER_STATUS_NOT_INITIALIZED: return "CUSOLVER_STATUS_NOT_INITIALIZED";
        case CUSOLVER_STATUS_ALLOC_FAILED: return "CUSOLVER_STATUS_ALLOC_FAILED";
        case CUSOLVER_STATUS_INVALID_VALUE: return "CUSOLVER_STATUS_INVALID_VALUE";
        case CUSOLVER_STATUS_ARCH_MISMATCH: return "CUSOLVER_STATUS_ARCH_MISMATCH";
        case CUSOLVER_STATUS_EXECUTION_FAILED: return "CUSOLVER_STATUS_EXECUTION_FAILED";
        case CUSOLVER_STATUS_INTERNAL_ERROR: return "CUSOLVER_STATUS_INTERNAL_ERROR";
        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
        default: return "<unknown>";
    }
}

#define CUSOLVER_CALL(ans) { cusolverAssert((ans), __FILE__, __LINE__); }
    inline void cusolverAssert(cusolverStatus_t code, const char *file, int line) {
    //printf("%d (%s:%d)\n", code, file, line);
    if (code != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr,"CUBLAS Error: %s %s:%d\n", cusolverErrorString(code), file, line);
        exit(code);
    }
}

#else
#define CUDA_CALL(ans) (ans)
#define CUSOLVER_CALL(ans) (ans)
#endif



/*
 * Returns a GPU id based on some criteria.
 */
int get_gpu_id();


/*
 * Taken from PyCUDA
 */
void get_grid_sizes(int problemsize, int* blocks, int* threads);


#endif /* GPU_COMMON */
