#include <iostream>

#include "cudaTools.h"
#include "Device.h"
#include "curandTools.h"
#include "Montecarlo_host.h"

extern __global__ void computePIWithMonteCarlo(int* ptrDevResult, curandState* ptrDevTabGenerators, int nbGen);

bool isMontecarlo()
    {
    const int NB_BLOCS = 256;
    const int NB_THREADS_BY_BLOCK = 1024;
    const int NB_GENERATIONS = 999999;

    dim3 dg(NB_BLOCS, 1, 1);
    dim3 db(NB_THREADS_BY_BLOCK, 1, 1);

    Device::assertDim(dg, db);

    int nbPointsOnIntegralArea;
    int* ptrDevResult;
    curandState* ptrDevTabGenerators;

    HANDLE_ERROR(cudaMalloc(&ptrDevResult, sizeof(int)));
    HANDLE_ERROR(cudaMalloc(&ptrDevTabGenerators, NB_BLOCS * NB_THREADS_BY_BLOCK * sizeof(curandState)));
    HANDLE_ERROR(cudaMemset(ptrDevResult, 0, sizeof(int)));

    setup_kernel_rand<<<dg,db>>>(ptrDevTabGenerators, 0);
    computePIWithMonteCarlo<<<dg,db>>>(ptrDevResult, ptrDevTabGenerators, NB_GENERATIONS);

    HANDLE_ERROR(cudaMemcpy(&nbPointsOnIntegralArea, ptrDevResult, sizeof(int), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(ptrDevResult));
    HANDLE_ERROR(cudaFree(ptrDevTabGenerators));

    float piValue = nbPointsOnIntegralArea * 4.0 / NB_GENERATIONS;

    std::cout << "PI = " << piValue << " (Montecarlo)" << std::endl;

    return abs(piValue - 3.141592653589793f) < 0.001;
    }

bool isMontecarloMGPU()
    {
    const int NB_BLOCS = 128;
    const int NB_THREADS_BY_BLOCK = 512;
    const int NB_GENERATIONS_BY_DEVICE = 1000000;

    int nbDevice = Device::getDeviceCount();

    int deviceSums[nbDevice];

#pragma omp parallel for
    for (int deviceID = 0; deviceID < nbDevice; deviceID++)
	{
	HANDLE_ERROR(cudaSetDevice(deviceID));

	dim3 dg(NB_BLOCS, 1, 1);
	dim3 db(NB_THREADS_BY_BLOCK, 1, 1);

	Device::assertDim(dg, db);

	int* ptrDevResult;
	curandState* ptrDevTabGenerators;

	HANDLE_ERROR(cudaMalloc(&ptrDevResult, sizeof(int)));
	HANDLE_ERROR(cudaMalloc(&ptrDevTabGenerators, NB_BLOCS * NB_THREADS_BY_BLOCK * sizeof(curandState)));
	HANDLE_ERROR(cudaMemset(ptrDevResult, 0, sizeof(int)));

	setup_kernel_rand<<<dg,db>>>(ptrDevTabGenerators, deviceID);
	computePIWithMonteCarlo<<<dg,db>>>(ptrDevResult, ptrDevTabGenerators, NB_GENERATIONS_BY_DEVICE);

    	HANDLE_ERROR(cudaMemcpy(&deviceSums[deviceID], ptrDevResult, sizeof(int), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaFree(ptrDevResult));
	HANDLE_ERROR(cudaFree(ptrDevTabGenerators));
	}

    int finalSum = 0;
    for (int i = 0; i < nbDevice; i++)
	{
	finalSum += deviceSums[0];
	}

    float piValue = finalSum * 4.0 / (nbDevice * NB_GENERATIONS_BY_DEVICE);

    std::cout << "PI = " << piValue << " (Montecarlo Multi GPU)" << std::endl;

    return abs(piValue - 3.141592653589793f) < 0.001;
    }
