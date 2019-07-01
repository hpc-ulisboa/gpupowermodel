#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define DATA_TYPE 0 // 0-SP, 1-INT, 2-DP

#define VECTOR_SIZE 60000000
#define TILE_DIM 1024
#define COMP_ITERATIONS 8192
#define KERNEL_CALLS 1

template <class T> __global__ void simpleKernel2(int size, int compute_iters, int tile_dim)
{
    __shared__ T shared[TILE_DIM];
    T r0;
    int xIndex = blockIdx.x * tile_dim + threadIdx.x;

    if (xIndex < size) {
        for (int i=0;i<compute_iters;i++) {
            r0 = shared[threadIdx.x];
            shared[TILE_DIM - threadIdx.x - 1] = r0;
        }
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

    printf("\nVector size: %d  TotalBlocks: %d blockSize: %d\n\n", vector_size, grid.x, threads.x);

    // initialize events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // take measurements for loop over kernel launches
    cudaEventRecord(start, 0);

    for (int i=0; i < kernel_calls; i++)
    {
        #if (DATA_TYPE == 0)
            simpleKernel2<float><<<grid, threads>>>(vector_size, compute_iters, tile_dim);
        #elif (DATA_TYPE == 1)
            simpleKernel2<int><<<grid, threads>>>(vector_size, compute_iters, tile_dim);
        #else
            simpleKernel2<double><<<grid, threads>>>(vector_size, compute_iters, tile_dim);
        #endif
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float kernelTime;
    cudaEventElapsedTime(&kernelTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceReset();

    printf("Test passed\n");

    exit(EXIT_SUCCESS);
}
