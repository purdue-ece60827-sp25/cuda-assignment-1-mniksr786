
#include "cudaLib.cuh"
#include "../src/cpuLib.cpp"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < size) {
		y[i] = scale * x[i] + y[i];
		}	
}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n\n";
	std::cout << "My First GPU Kernel!\n";
	std::cout << "Many more to come!\n";
	
	auto tStart = std::chrono::high_resolution_clock::now();
	//	Insert code here
	
	float *x, *y, *z;
	float *x_d, *y_d, *z_d;
	float a = 2.0f;
	int vector_mem = vectorSize * sizeof(float);
	
	// Space allocation for the float input vectors
	x = (float *) malloc(vector_mem);
	y = (float *) malloc(vector_mem);
	z = (float *) malloc(vector_mem);
	
	// Creating the vectors using the CPU functions only
	vectorInit(x, vectorSize);
	vectorInit(y, vectorSize);
	
	
	// Allocating space and copying data to the GPU
	cudaMalloc((void **) &x_d, vector_mem);
	cudaMalloc((void **) &y_d, vector_mem);
	cudaMalloc((void **) &z_d, vector_mem);
	
	cudaMemcpy(x_d, x, vector_mem, cudaMemcpyHostToDevice);
	cudaMemcpy(y_d, y, vector_mem, cudaMemcpyHostToDevice);
	cudaMemcpy(z_d, y, vector_mem, cudaMemcpyHostToDevice);
	
	
	// Calling the SAXPY kernel
	saxpy_gpu<<< (vectorSize/256), 256 >>>(x_d, z_d, a, vectorSize);
	
	// Transfering data back from GPU to CPU
	cudaMemcpy(z, z_d, vector_mem, cudaMemcpyDeviceToHost);
	cudaMemcpy(y, y_d, vector_mem, cudaMemcpyDeviceToHost);
	cudaMemcpy(x, x_d, vector_mem, cudaMemcpyDeviceToHost);

	// Free GPU memory
	cudaFree(x_d); cudaFree(y_d); cudaFree(z_d);
		
	// Verifying the error count
	int errorCount = verifyVector(x, y, z, a, vectorSize);
	std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t *pSums, uint64_t *pSumSize, uint64_t sampleSize) {
	//	Insert code here
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	float x, y;
	uint64_t hits =0;
	
	curandState_t rng;
	curand_init(clock64(), i, 0, &rng);    
    
	for (uint64_t idx = 0; idx < sampleSize; ++idx) {
		x = curand_uniform(&rng);
		y = curand_uniform(&rng);

		if ((x * x + y * y) <= 1.0f ) {
			hits++;
		}
	}
	atomicAdd(reinterpret_cast<unsigned long long*>(pSums), hits);
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	std::string str;
	uint64_t hitCount = 0;
	uint64_t *hitCount_d, *totalHitCount_d;
	
	auto tStart = std::chrono::high_resolution_clock::now();
	
	cudaMalloc((void **) &hitCount_d, sizeof(uint64_t));
	//cudaMalloc((void **) &totalHitCount_d, sizeof(uint64_t));
	
	cudaMemcpy(hitCount_d, &hitCount, sizeof(uint64_t), cudaMemcpyHostToDevice);
	//cudaMemcpy(totalHitCount_d, &totalHitCount, sizeof(uint64_t), cudaMemcpyHostToDevice);

	//	Main GPU Monte-Carlo Code
	generatePoints<<<(generateThreadCount/128), 128>>>(hitCount_d, totalHitCount_d, sampleSize);
	
	cudaMemcpy(&hitCount, hitCount_d, sizeof(uint64_t), cudaMemcpyDeviceToHost);
	//cudaMemcpy(&totalHitCount, totalHitCount_d, sizeof(uint64_t), cudaMemcpyDeviceToHost);

	cudaFree(hitCount_d); 
	//cudaFree(totalHitCount_d);
	
	
	float approxPi = (((double)hitCount) / (sampleSize * generateThreadCount));
	approxPi = approxPi * 4.0f;
		
	std::cout << std::setprecision(10);
	std::cout << "Estimated Pi = " << approxPi << "\n";
		
	/* float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";*/

	auto tEnd= std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;

	//      Insert code here
	// NOT USING THIS FUNCTION
	std::cout << "Sneaky, you are ...\n";
	std::cout << "Compute pi, you must!\n";
	return approxPi;
}
