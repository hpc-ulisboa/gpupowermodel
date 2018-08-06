#include "cf_shift_kernel.cuh"

__device__ int recursiveKernel(int in) {
    if (in > 0)
        return in + recursiveKernel(in << 2);
    else
        return 0;
}

//recursiveKernel
__global__ void simpleKernel(int *A, int *C1, int *C2, int *C3, int *C4, int size, int inner_reps, int tile_dim)
{
    int xIndex = blockIdx.x * tile_dim + threadIdx.x;


    if (xIndex < size) {
        int r0 = A[xIndex],
          r1 = r0,
          r2 = r0,
          r3 = r0;

        // __syncthreads();

        // rb=A[xIndex];
        for (int i=0;i<inner_reps;i++) {
            //v5.1
            if (i % 2 == 0) {
                if (i % 4 == 0) {
                    if (i % 8 == 0) {
                        if (i % 16 == 0) {
                            if (i % 32 == 0) {
                                r0 = recursiveKernel(r1);//r0;
                            } else {
                                r0 = recursiveKernel(r0);
                            }
                        } else {
                            r0 = recursiveKernel(r2);//r0;
                        }
                    } else {
                        r0 = recursiveKernel(r3);//r0;
                    }
                } else {
                    r0 = r0 << 1;//r0;
                }
            } else {
                r0 = r0;
            }
            if (i % 2 == 0) {
                if (i % 4 == 0) {
                    if (i % 8 == 0) {
                        if (i % 16 == 0) {
                            if (i % 32 == 0) {
                                r1 = recursiveKernel(r2);//r0;
                            } else {
                                r1 = recursiveKernel(r1);
                            }
                        } else {
                            r1 = recursiveKernel(r3);//r0;
                        }
                    } else {
                        r1 = recursiveKernel(r0);//r0;
                    }
                } else {
                    r1 = r1 << 1;//r0;
                }
            } else {
                r1 = r1;
            }
            if (i % 2 == 0) {
                if (i % 4 == 0) {
                    if (i % 8 == 0) {
                        if (i % 16 == 0) {
                            if (i % 32 == 0) {
                                r2 = recursiveKernel(r3);//r0;
                            } else {
                                r2 = recursiveKernel(r2);
                            }
                        } else {
                            r2 = recursiveKernel(r0);//r0;
                        }
                    } else {
                        r2 = recursiveKernel(r1);//r0;
                    }
                } else {
                    r2 = r2 << 1;//r0;
                }
            } else {
                r2 = r2;
            }
            if (i % 2 == 0) {
                if (i % 4 == 0) {
                    if (i % 8 == 0) {
                        if (i % 16 == 0) {
                            if (i % 32 == 0) {
                                r3 = recursiveKernel(r0);//r0;
                            } else {
                                r3 = recursiveKernel(r3);
                            }
                        } else {
                            r3 = recursiveKernel(r1);//r0;
                        }
                    } else {
                        r3 = recursiveKernel(r2);//r0;
                    }
                } else {
                    r3 = r3 << 1;//r0;
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

void callKernel(dim3 threads, dim3 grid, int *d_iA, int *d_oC1, int *d_oC2, int *d_oC3, int *d_oC4, int vector_size, int inner_reps, int tile_dim) {
    simpleKernel<<<grid, threads>>>(d_iA, d_oC1, d_oC2, d_oC3, d_oC4, vector_size, inner_reps, tile_dim);
}
