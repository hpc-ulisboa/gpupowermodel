#include "cf_kernel.cuh"

template <class T> __global__ void simpleKernel(T *A, T *C1, T *C2, T *C3, T *C4, int size, int inner_reps, int tile_dim)
{
    int xIndex = blockIdx.x * tile_dim + threadIdx.x;


    if (xIndex < size) {
        T r0 = A[xIndex],
          r1 = r0,
          r2 = r0,
          r3 = r0;

        // __syncthreads();

        // rb=A[xIndex];
        for (int i=0;i<inner_reps;i++) {
            //v5.1
            if (xIndex % 2) {
                if (xIndex % 4) {
                    if (xIndex % 8) {
                        if (xIndex % 16) {
                            if (xIndex % 32) {
                                r0 = r0 * r0 + r1;//r0;
                            } else {
                                r0 = r1 * r1 + r0;
                            }
                        } else {
                            r0 = r0 * r0 + r2;//r0;
                        }
                    } else {
                        r0 = r0 * r0 + r3;//r0;
                    }
                } else {
                    r0 = r0 * r0 + r0;//r0;
                }
            } else {
                r0 = r0;
            }
            if (xIndex % 2) {
                if (xIndex % 4) {
                    if (xIndex % 8) {
                        if (xIndex % 16) {
                            if (xIndex % 32) {
                                r1 = r1 * r1 + r2;//r0;
                            } else {
                                r1 = r2 * r2 + r1;
                            }
                        } else {
                            r1 = r1 * r1 + r3;//r0;
                        }
                    } else {
                        r1 = r1 * r1 + r0;//r0;
                    }
                } else {
                    r1 = r1 * r1 + r1;//r0;
                }
            } else {
                r1 = r1;
            }
            if (xIndex % 2) {
                if (xIndex % 4) {
                    if (xIndex % 8) {
                        if (xIndex % 16) {
                            if (xIndex % 32) {
                                r2 = r2 * r2 + r3;//r0;
                            } else {
                                r2 = r3 * r3 + r2;
                            }
                        } else {
                            r2 = r1 * r2 + r0;//r0;
                        }
                    } else {
                        r2 = r2 * r2 + r1;//r0;
                    }
                } else {
                    r2 = r2 * r2 + r2;//r0;
                }
            } else {
                r2 = r2;
            }
            if (xIndex % 2) {
                if (xIndex % 4) {
                    if (xIndex % 8) {
                        if (xIndex % 16) {
                            if (xIndex % 32) {
                                r3 = r3 * r3 + r0;//r0;
                            } else {
                                r3 = r0 * r0 + r3;
                            }
                        } else {
                            r3 = r3 * r3 + r1;//r0;
                        }
                    } else {
                        r3 = r3 * r3 + r2;//r0;
                    }
                } else {
                    r3 = r3 * r3 + r3;//r0;
                }
            } else {
                r3 = r3;
            }
        }

        C1[xIndex]=r0;
        C2[xIndex]=r1;
        C3[xIndex]=r2;
        C4[xIndex]=r3;
    }
}

template <class T> void callKernel(dim3 threads, dim3 grid, T *d_iA, T *d_oC1, T *d_oC2, T *d_oC3, T *d_oC4, int vector_size, int inner_reps, int tile_dim) {
    simpleKernel<<<grid, threads>>>(d_iA, d_oC1, d_oC2, d_oC3, d_oC4, vector_size, inner_reps, tile_dim);
}

template void callKernel<float>(dim3 threads, dim3 grid, float *d_iA, float *d_oC1, float *d_oC2, float *d_oC3, float *d_oC4, int vector_size, int inner_reps, int tile_dim);
template void callKernel<int>(dim3 threads, dim3 grid, int *d_iA, int *d_oC1, int *d_oC2, int *d_oC3, int *d_oC4, int vector_size, int inner_reps, int tile_dim);
template void callKernel<double>(dim3 threads, dim3 grid, double *d_iA, double *d_oC1, double *d_oC2, double *d_oC3, double *d_oC4, int vector_size, int inner_reps, int tile_dim);
