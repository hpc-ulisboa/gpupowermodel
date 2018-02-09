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

#define KERNEL_CALLS 1

#define COMP_ITERATIONS (1024)
#define THREADS (1024)
#define BLOCKS (32768)
#define STRIDE (64*1024)

#define REGBLOCK_SIZE (4)
#define UNROLL_ITERATIONS (32)

#define deviceNum (0)

//CODE
__global__ void warmup(short* cd){

	short r0 = 1.0,
	  r1 = r0+(short)(31),
	  r2 = r0+(short)(37),
	  r3 = r0+(short)(41);

	for(int j=0; j<COMP_ITERATIONS; j+=UNROLL_ITERATIONS){
		#pragma unroll
		for(int i=0; i<UNROLL_ITERATIONS; i++){
			// Each iteration maps to floating point 8 operations (4 multiplies + 4 additions)
			r0 = r0 * r0 + r1;//r0;
			r1 = r1 * r1 + r2;//r1;
			r2 = r2 * r2 + r3;//r2;
			r3 = r3 * r3 + r0;//r3;
		}
	}
	cd[blockIdx.x * 256 + threadIdx.x] = r0;
}

template <class T>
__global__ void benchmark( T* cdin,  T* cdout, int compute_iters){

	const long ite=blockIdx.x * THREADS + threadIdx.x;
	// const int ite = threadIdx.x+(BLOCKS-blockIdx.x)%BLOCKS*32+threadIdx.x/(int)32*32;
	T r0;
	// printf("(%d/%d) - %d\n", blockIdx.x,threadIdx.x,ite);
	for(int j=0; j<compute_iters; j+=UNROLL_ITERATIONS){
		#pragma unroll
		for(int i=0; i<UNROLL_ITERATIONS; i++){
			r0=cdin[ite+STRIDE*i];
			cdout[ite+STRIDE*i]=r0;
		}
	}

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

void runbench_warmup(short* cd){
	const int BLOCK_SIZE = 512;
	const int TOTAL_REDUCED_BLOCKS = 512;

	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimReducedGrid(TOTAL_REDUCED_BLOCKS, 1, 1);

	warmup<<< dimReducedGrid, dimBlock >>>(cd);
	CUDA_SAFE_CALL( cudaGetLastError() );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

void runbench(double* kernel_time, double* bandw,double* cdin,double* cdout,int L2size, int compute_iters){

	cudaEvent_t start, stop;
    int type = DATA_TYPE;
    dim3 dimBlock(THREADS, 1, 1);
    dim3 dimGrid(BLOCKS, 1, 1);
	initializeEvents(&start, &stop);
	if (type==0){
		benchmark<float><<< dimGrid, dimBlock >>>((float*)cdin,(float*)cdout, compute_iters);
	}else{
		benchmark<double><<< dimGrid, dimBlock >>>(cdin,cdout, compute_iters);
	}

	long long shared_access = 2*(long long)(compute_iters)*THREADS*BLOCKS;

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

    int kernel_calls=KERNEL_CALLS, compute_iters=COMP_ITERATIONS;
    cudaDeviceProp deviceProp;

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

	cudaSetDevice(deviceNum);

	double time[kernel_calls][2],value[kernel_calls][4];
	int L2size;
	int size = (THREADS*BLOCKS+32*STRIDE)*sizeof(double);
	size_t freeCUDAMem, totalCUDAMem;
	cudaMemGetInfo(&freeCUDAMem, &totalCUDAMem);
	printf("Total GPU memory %lu, free %lu\n", totalCUDAMem, freeCUDAMem);
	printf("Buffer size: %dMB\n", size*sizeof(double)/(1024*1024));

	//Initialize Global Memory
	double *cdin;
	double *cdout;
	// init = (int*)malloc(size);
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
	class type;

	for (i=0;i<10;i++){
		runbench_warmup((short*)cdin);
	}

    for (i=0;i<kernel_calls;i++){
    	runbench(&time[0][0],&value[0][0],cdin,cdout,L2size,compute_iters);
    	printf("Registered time: %f ms\n",time[0][0]);
	}

    CUDA_SAFE_CALL( cudaDeviceReset());
    printf("-----------------------------------------------------------------------\n");
	return 0;
}
