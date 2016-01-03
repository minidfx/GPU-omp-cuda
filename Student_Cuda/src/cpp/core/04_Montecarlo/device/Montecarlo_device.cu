#include "cudaTools.h"
#include "curand_kernel.h"

__global__
void computePIWithMonteCarlo(int* ptrDevResult, curandState* ptrDevTabGenerators, int nbGen);
__device__
int monteCarloIntraThreadReduction(curandState* ptrDevTabGenerators, int nbGen);
__device__
void generateRandomPoint(curandState* ptrDevGenerator, float xMin, float xMax, float yMin, float yMax, float* x, float* y);

__device__
static float fpi(float x) {
 return 4 / (1 + x * x);
}

__device__ static void intraBlockSumReduction(int* arraySM, int size) {
  const int NB_THREADS_LOCAL = blockDim.x * blockDim.y * blockDim.z;
  const int TID_LOCAL = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;

  int n = size;
  int half = size / 2;
  while (half >= 1) {

    int s = TID_LOCAL;
    while (s < half) {
      arraySM[s] += arraySM[s + half];
      s += NB_THREADS_LOCAL;
    }

    __syncthreads();

    n = half;
    half = n / 2;
  }
}

__device__ static void interBlockSumReduction(int* arraySM, int* resultGM) {
  const int TID_LOCAL = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;

  if (TID_LOCAL == 0) {
    atomicAdd(resultGM, arraySM[0]);
  }
}

__global__
void computePIWithMonteCarlo(int* ptrDevResult, curandState* ptrDevTabGenerators, int nbGen) {
  const int NB_THREADS_LOCAL = blockDim.x;
  const int TID_LOCAL = threadIdx.x;

  __shared__ int ptrDevArraySM[1024];

  ptrDevArraySM[TID_LOCAL] = monteCarloIntraThreadReduction(ptrDevTabGenerators, nbGen);

  __syncthreads();

  intraBlockSumReduction(ptrDevArraySM, NB_THREADS_LOCAL);
  interBlockSumReduction(ptrDevArraySM, ptrDevResult);
}

__device__
int monteCarloIntraThreadReduction(curandState* ptrDevTabGenerators, int nbGen) {
  const int NB_THREADS = gridDim.x * blockDim.x;
  const int TID = threadIdx.x + blockIdx.x * blockDim.x;

  int threadSum = 0;
  float x;
  float y;
  int s = TID;
  while (s < nbGen) {

    generateRandomPoint(&ptrDevTabGenerators[TID], 0, 1, 0, 4, &x, &y);

    if (y <= fpi(x)) {
      threadSum++;
    }

    s += NB_THREADS;
  }

  return threadSum;
}

__device__
void generateRandomPoint(curandState* ptrDevGenerator, float xMin, float xMax, float yMin, float yMax, float* x, float* y) {
  *x = curand_uniform(ptrDevGenerator) * (xMax - xMin) + xMin;
  *y = curand_uniform(ptrDevGenerator) * (yMax - yMin) + yMin;
}
