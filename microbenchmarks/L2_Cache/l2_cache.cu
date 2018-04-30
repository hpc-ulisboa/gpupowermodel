#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define CUDA_SAFE_CALL( call) {                                    \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } }

#define DRIVER_API_CALL(apiFuncCall)                                           \
do {                                                                           \
    CUresult _status = apiFuncCall;                                            \
    if (_status != CUDA_SUCCESS) {                                             \
        fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",   \
                __FILE__, __LINE__, #apiFuncCall, _status);                    \
        exit(-1);                                                              \
    }                                                                          \
} while (0)

#define DATA_TYPE 0 // 0-SP, 1-DP

#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

#define THREADS (1024)
#define BLOCKS (3276)
#define N 10

#define KERNEL_CALLS 1
#define COMP_ITERATIONS 512
#define UNROLL_ITERATIONS 32

#define REGBLOCK_SIZE (4)
#define deviceNum (0)


template <class T> __global__ void benchmark (T* cdin, T* cdout, int inner_reps, int unrolls){

	#ifdef OFFSET
		const int ite = blockIdx.x%4 * THREADS + threadIdx.x;
	#else
		const int ite = blockIdx.x * THREADS + threadIdx.x;
	#endif
	T r0;
	for (int k=0; k<N;k++){
		for(int j=0; j<inner_reps; j+=unrolls){
			#pragma unroll
			for(int i=0; i<unrolls; i++){
				r0 = cdin[ite];
				cdout[ite]=r0;
			}
		}
	}
	cdout[ite]=r0;
}

void initializeEvents(cudaEvent_t *start, cudaEvent_t *stop){
	CUDA_SAFE_CALL( cudaEventCreate(start) );
	CUDA_SAFE_CALL( cudaEventCreate(stop) );
	CUDA_SAFE_CALL( cudaEventRecord(*start, 0) );
}

float finalizeEvents(cudaEvent_t start, cudaEvent_t stop){
	CUDA_SAFE_CALL( cudaGetLastError() );
	CUDA_SAFE_CALL( cudaEventRecord(stop, 0) );
	CUDA_SAFE_CALL( cudaEventSynchronize(stop) );
	float kernel_time;
	CUDA_SAFE_CALL( cudaEventElapsedTime(&kernel_time, start, stop) );
	CUDA_SAFE_CALL( cudaEventDestroy(start) );
	CUDA_SAFE_CALL( cudaEventDestroy(stop) );
	return kernel_time;
}

void runbench(double* kernel_time, double* bandw,double* cdin,double* cdout, int compute_iters, int unrolls){

	cudaEvent_t start, stop;
	initializeEvents(&start, &stop);
	dim3 dimBlock(THREADS, 1, 1);
	dim3 dimGrid(BLOCKS, 1, 1);
    int type = DATA_TYPE;

    CUDA_SAFE_CALL( cudaGetLastError() );
    if (type==0){
		benchmark<float><<< dimGrid, dimBlock >>>((float*)cdin,(float*)cdout, compute_iters, unrolls);
	}else{
		benchmark<double><<< dimGrid, dimBlock >>>(cdin,cdout, compute_iters, unrolls);
	}
    CUDA_SAFE_CALL( cudaGetLastError() );

	long long shared_access = 2*(long long)(compute_iters)*N*THREADS*BLOCKS;

	cudaDeviceSynchronize();

	double time = finalizeEvents(start, stop);
	double result;
	if (type==0)
		result = ((double)shared_access)*4/(double)time*1000./(double)(1024*1024*1024);
	else
		result = ((double)shared_access)*8/(double)time*1000./(double)(1024*1024*1024);

	*kernel_time = time;
	*bandw=result;
}

int main(int argc, char *argv[]){
	CUdevice device = 0;
	int deviceCount;
	char deviceName[32];

    int kernel_calls=KERNEL_CALLS, compute_iters=COMP_ITERATIONS, unrolls=UNROLL_ITERATIONS;
	cudaDeviceProp deviceProp;

    if (argc != 4) {
        printf("\nError: Wrong number of arguments.\n\n");
        printf("Usage:\n\t %s [inner_iterations] [kernel_calls] [unrolls]\n\t %s\n", argv[0], argv[0]);

        return -1;
    } else  {
        compute_iters = atoi(argv[argc-3]);
        kernel_calls = atoi(argv[argc-2]);
        unrolls = atoi(argv[argc-1]);
    }

    printf("Number of kernel launches: %d\n", kernel_calls);
    printf("Number of compute iterations: %d\n", compute_iters);

	CUDA_SAFE_CALL(cudaSetDevice(deviceNum));
	double time[kernel_calls][2],value[kernel_calls][4];

	int size = (THREADS*BLOCKS)*sizeof(double);
	size_t freeCUDAMem, totalCUDAMem;
	CUDA_SAFE_CALL(cudaMemGetInfo(&freeCUDAMem, &totalCUDAMem));
	printf("Total GPU memory %lu, free %lu\n", totalCUDAMem, freeCUDAMem);
	printf("Buffer size: %dMB\n", size*sizeof(double)/(1024*1024));

	//Initialize Global Memory
	double *cdin;
	double *cdout;
	CUDA_SAFE_CALL(cudaMalloc((void**)&cdin, size));
	CUDA_SAFE_CALL(cudaMalloc((void**)&cdout, size));

	// Copy data to device memory
	CUDA_SAFE_CALL(cudaMemset(cdin, 0, size));  // initialize to zeros
	CUDA_SAFE_CALL(cudaMemset(cdout, 0, size));  // initialize to zeros
	// Synchronize in order to wait for memory operations to finish
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	// make sure activity is enabled before any CUDA API

	DRIVER_API_CALL(cuDeviceGetCount(&deviceCount));
	if (deviceCount == 0) {
        printf("There is no device supporting CUDA.\n");
        return -2;
	}

	printf("CUDA Device Number: %d\n", deviceNum);

	DRIVER_API_CALL(cuDeviceGet(&device, deviceNum));
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, device));
	DRIVER_API_CALL(cuDeviceGetName(deviceName, 32, device));

	int i;

    for (i=0;i<kernel_calls;i++){
		runbench(&time[0][0],&value[0][0],cdin,cdout,compute_iters, unrolls);

        printf("Registered time: %f ms\n",time[0][0]);
    }

    CUDA_SAFE_CALL( cudaDeviceReset());
	return 0;
}
