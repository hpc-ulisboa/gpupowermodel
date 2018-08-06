#ifndef __CONTROLFLOWSHIFT_HEADER__
#define __CONTROLFLOWSHIFT_HEADER__

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

// #define DATA_TYPE 0 // 0-SP, 1-INT, 2-DP

#define UNROLL 32
#define TILE_DIM 1024
#define VECTOR_SIZE 60000000
#define COMP_ITERATIONS 2048
#define KERNEL_CALLS 3
// template <class T> __global__ void simpleKernel(T *A, T *C1, T *C2, T *C3, T *C4, int size, int inner_reps, int tile_dim);
void callKernel(dim3 , dim3 , int *, int *, int *, int *, int *, int , int , int );

#endif
