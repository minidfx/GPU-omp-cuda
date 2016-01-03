#ifndef SAUCISSON_DEVICE
#define SAUCISSON_DEVICE

#include <stdio.h>
#include "Indice1D.h"

__global__
void saucissonDevice(float* ptrDevResult,int nbSaucisson,int n);
__device__
void initSM(float* tabSM,int n);
__device__
void fillSM(float* tabSM,int nbSaucisson);
__device__
static float fpi(float x) {
 return 4 / (1 + x * x);
}

__device__ static void intraBlockSumReduction(float* arraySM, int size) {
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

__device__ static void interBlockSumReduction(float* arraySM, float* resultGM) {
  const int TID_LOCAL = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;

  if (TID_LOCAL == 0) {
    atomicAdd(resultGM, arraySM[0]);
  }
}

__global__
void saucissonDevice(float* ptrDevResult,int nbSaucisson,int n)
{
    extern __shared__ float tabSM[];

    initSM(tabSM,n);

    __syncthreads();

    fillSM(tabSM,nbSaucisson);

    __syncthreads();

    intraBlockSumReduction(tabSM, n);
    interBlockSumReduction(tabSM, ptrDevResult);
}

__device__
void initSM(float* tabSM,int n)
{
    const int TID_LOCAL = Indice1D::tidLocal();
    const int NB_THREAD_LOCAL= Indice1D::nbThreadBlock();
    int s = TID_LOCAL;
    while(s < n)
    {
	tabSM[s]=0;
	s+=NB_THREAD_LOCAL;
    }
}

__device__
void fillSM(float* tabSM,int nbSaucisson)
{
    const int NB_THREAD=Indice1D::nbThread();
    const int TID = Indice1D::tid();
    const int TID_LOCAL = Indice1D::tidLocal();
    const float DX = 1.0f/(float)nbSaucisson;
    float s = TID;
    float sum = 0;

    while (s < nbSaucisson)
    {
	    double xs = s * DX;
	    sum += fpi(xs);
	    s += NB_THREAD;
    }

    tabSM[TID_LOCAL] = sum;
}

#endif
