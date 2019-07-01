#ifndef __CONTROLFLOW_HEADER__
#define __CONTROLFLOW_HEADER__

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define DATA_TYPE 0 // 0-SP, 1-INT, 2-DP

#define UNROLL 32
#define TILE_DIM 1024
#define VECTOR_SIZE 60000000
#define COMP_ITERATIONS 1024
#define KERNEL_CALLS 4

// template <class T> __global__ void simpleKernel(T *A, T *C1, T *C2, T *C3, T *C4, int size, int inner_reps, int tile_dim);
template <class T> void callKernel(dim3 , dim3 , T *, T *, T *, T *, T *, int , int , int );

#endif
