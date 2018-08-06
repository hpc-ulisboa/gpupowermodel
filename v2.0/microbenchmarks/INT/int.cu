#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define DATA_TYPE 1 // 0-SP, 1-INT, 2-DP

#define VECTOR_SIZE 60000000
#define TILE_DIM 1024
#define COMP_ITERATIONS 4096
#define KERNEL_CALLS 4

template <class T> __global__ void simpleKernel(T *A, T *C1, T *C2, T *C3, T *C4, int size, int inner_reps, int tile_dim)
{
    int xIndex = blockIdx.x * tile_dim + threadIdx.x;
    T ra, rb, rc, rd, re, rf, rg, rh;

    if (xIndex < size) {
        ra=A[xIndex];
        rb=A[size-xIndex];
        rc=A[xIndex];
        rd=A[size-xIndex];
        re=A[xIndex];
        rf=A[size-xIndex];
        rg=A[xIndex];
        rh=A[size-xIndex];

        // rb=A[xIndex];
        for (int i=0;i<inner_reps;i++) {
            //add_2regs
            ra=ra+rb;
            rb=rb+rc;
            rc=rc+rd;
            rd=rd+ra;
        }
        C1[xIndex]=ra-rf;
        C2[xIndex]=rb+re;
        C3[xIndex]=rc+rh;
        C4[xIndex]=rd-rg;

    }
}


int main(int argc, char **argv) {
    int compute_iters=COMP_ITERATIONS,
      kernel_calls=KERNEL_CALLS,
      vector_size=VECTOR_SIZE,
      tile_dim=TILE_DIM;

    if (argc > 3 || argc == 2) {
      printf("\nError: Wrong number of arguments.\n\n");
      printf("Usage:\n\t %s [inner_iterations] [kernel_calls]\n\t %s\n", argv[0], argv[0]);

      return -1;
    }

    if (argc == 3) {
      kernel_calls = atoi(argv[2]);
      compute_iters = atoi(argv[1]);
    }

    printf("Number of kernel launches: %d\n", kernel_calls);
    printf("Number of compute iterations: %d\n", compute_iters);

    // execution configuration parameters
    dim3 grid(vector_size/tile_dim, 1), threads(tile_dim, 1);

    // CUDA events
    cudaEvent_t start, stop;

    #if (DATA_TYPE == 0)
        size_t mem_size = static_cast<size_t>(sizeof(float) * vector_size);
        // allocate host memory
        float *h_iA = (float *) malloc(mem_size);
        float *h_oC1 = (float *) malloc(mem_size);
        float *h_oC2 = (float *) malloc(mem_size);
        float *h_oC3 = (float *) malloc(mem_size);
        float *h_oC4 = (float *) malloc(mem_size);
        // initalize host data
        for (int i = 0; i < vector_size; ++i)
        {
            h_iA[i] = (float) i+3;
            // h_iB[i] = (float) i+3;
        }
        // allocate device memory
        float *d_iA, *d_oC1, *d_oC2, *d_oC3, *d_oC4;
    #elif (DATA_TYPE == 1)
            size_t mem_size = static_cast<size_t>(sizeof(int) * vector_size);
            // allocate host memory
            int *h_iA = (int *) malloc(mem_size);
            int *h_oC1 = (int *) malloc(mem_size);
            int *h_oC2 = (int *) malloc(mem_size);
            int *h_oC3 = (int *) malloc(mem_size);
            int *h_oC4 = (int *) malloc(mem_size);
            // initalize host data
            for (int i = 0; i < vector_size; ++i)
            {
                h_iA[i] = (int) i+3;
                // h_iB[i] = (float) i+3;
            }
            // allocate device memory
            int *d_iA, *d_oC1, *d_oC2, *d_oC3, *d_oC4;
    #else
            size_t mem_size = static_cast<size_t>(sizeof(double) * vector_size);
            // allocate host memory
            double *h_iA = (double *) malloc(mem_size);
            double *h_oC1 = (double *) malloc(mem_size);
            double *h_oC2 = (double *) malloc(mem_size);
            double *h_oC3 = (double *) malloc(mem_size);
            double *h_oC4 = (double *) malloc(mem_size);
            // initalize host data
            for (int i = 0; i < vector_size; ++i)
            {
                h_iA[i] = (double) i+3;
                // h_iB[i] = (float) i+3;
            }
            // allocate device memory
            double *d_iA, *d_oC1, *d_oC2, *d_oC3, *d_oC4;
    #endif

    cudaMalloc((void **) &d_iA, mem_size);
    cudaMalloc((void **) &d_oC1, mem_size);
    cudaMalloc((void **) &d_oC2, mem_size);
    cudaMalloc((void **) &d_oC3, mem_size);
    cudaMalloc((void **) &d_oC4, mem_size);

    // copy host data to device
    cudaMemcpy(d_iA, h_iA, mem_size, cudaMemcpyHostToDevice);

    // print out common data for all kernels
    printf("\nVector size: %d  TotalBlocks: %d blockSize: %d\n\n", vector_size, grid.x, threads.x);

    // initialize events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // take measurements for loop over kernel launches
    cudaEventRecord(start, 0);

    for (int i=0; i < kernel_calls; i++)
    {
        #if (DATA_TYPE == 0)
            simpleKernel<float><<<grid, threads>>>(d_iA, d_oC1, d_oC2, d_oC3, d_oC4, vector_size, compute_iters, tile_dim);
        #elif (DATA_TYPE == 1)
            simpleKernel<int><<<grid, threads>>>(d_iA, d_oC1, d_oC2, d_oC3, d_oC4, vector_size, compute_iters, tile_dim);
        #else
            simpleKernel<double><<<grid, threads>>>(d_iA, d_oC1, d_oC2, d_oC3, d_oC4, vector_size, compute_iters, tile_dim);
        #endif        // Ensure no launch failure
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float kernelTime;
    cudaEventElapsedTime(&kernelTime, start, stop);

    // take measurements for loop inside kernel
    cudaMemcpy(h_oC1, d_oC1, mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_oC2, d_oC2, mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_oC3, d_oC3, mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_oC4, d_oC4, mem_size, cudaMemcpyDeviceToHost);

    printf("teste: %f\n", h_oC1[0]);

    // report effective bandwidths
    float kernelBandwidth = 2.0f * 1000.0f * mem_size/(1024*1024*1024)/(kernelTime/kernel_calls);
    printf("simpleKernel, Throughput = %.4f GB/s, Time = %.5f ms, Size = %u fp32 elements, NumDevsUsed = %u, Workgroup = %u\n",
           kernelBandwidth,
           kernelTime/kernel_calls,
           vector_size, 1, tile_dim * 1);

    free(h_iA);
    free(h_oC1);
    free(h_oC2);
    free(h_oC3);
    free(h_oC4);

    cudaFree(d_iA);
    cudaFree(d_oC1);
    cudaFree(d_oC2);
    cudaFree(d_oC3);
    cudaFree(d_oC4);


    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceReset();

    printf("Test passed\n");

    exit(EXIT_SUCCESS);
}
