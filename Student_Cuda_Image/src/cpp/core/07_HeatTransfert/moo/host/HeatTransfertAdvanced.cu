#include <iostream>
#include <omp.h>

#include "cuda_runtime.h"
#include "Device.h"
#include "HeatTransfertAdvanced.h"
#include "IndiceTools.h"

__global__ void diffuseAdvanced(float* ptrImageInput, float* ptrImageOutput, unsigned int width, unsigned int height, float propagationSpeed);
__global__ void crushAdvanced(float* ptrImageHeater, float* ptrImage, unsigned int size);
__global__ void displayAdvanced(float* ptrImage, uchar4* ptrPixels, unsigned int size);

HeatTransfertAdvanced::HeatTransfertAdvanced(unsigned int width, unsigned int height, float propagationSpeed, string title)
{
    // Inputs
    this->width = width;
    this->height = height;
    this->totalPixels = width * height;
    this->title = title;

    // Tools
    this->iteration = 0;
    this->propagationSpeed = propagationSpeed;

    // Cuda grid dimensions
    this->dg = dim3(8, 8, 1);
    this->db = dim3(16, 16, 1);

    // Check
    Device::assertDim(dg, db);

    float imageInit[this->totalPixels];
    float imageHeater[this->totalPixels];

    unsigned int s = 0;
    while(s++ < this->totalPixels)
    {
        imageInit[s] = 0.0;

        int i, j;
        IndiceTools::toIJ(s, width, &i, &j);

        if (i >= 187 && i < 312 && j >= 187 && j < 312)
        {
            imageHeater[s] = 1.0;
        }
        else if ((i >= 111 && i < 121 && j >= 111 && j < 121) || (i >= 111 && i < 121 && j >= 378 && j < 388) || (i >= 378 && i < 388 && j >= 111 && j < 121)
        || (i >= 378 && i < 388 && j >= 378 && j < 388) || (i >= 378 && i < 388 && j >= 378 && j < 388) || (i >= 378 && i < 388 && j >= 378 && j < 388))
        {
            imageHeater[s] = 0.2;
        }
        else
        {
            imageHeater[s] = 0.0;
        }
    }

    // Size of all pixels of an image
    size_t arraySize = sizeof(float) * this->totalPixels;

    // Allocating memory to GPU
    HANDLE_ERROR(cudaMalloc(&this->ptrDevImageHeater, arraySize));
    HANDLE_ERROR(cudaMalloc(&this->ptrDevImageInit, arraySize));
    HANDLE_ERROR(cudaMalloc(&this->ptrDevImageA, arraySize));
    HANDLE_ERROR(cudaMalloc(&this->ptrDevImageB, arraySize));

    // Set a known value to any array representing an image
    HANDLE_ERROR(cudaMemset(ptrDevImageHeater, 0, arraySize));
    HANDLE_ERROR(cudaMemset(ptrDevImageInit, 0, arraySize));
    HANDLE_ERROR(cudaMemset(ptrDevImageA, 0, arraySize));
    HANDLE_ERROR(cudaMemset(ptrDevImageB, 0, arraySize));

    // Copy images from CPU to GPU
    HANDLE_ERROR(cudaMemcpy(this->ptrDevImageHeater, imageHeater, arraySize, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(this->ptrDevImageInit, imageInit, arraySize, cudaMemcpyHostToDevice));
}

HeatTransfertAdvanced::~HeatTransfertAdvanced()
{
    // Release resources GPU side
    HANDLE_ERROR(cudaFree(this->ptrDevImageHeater));
    HANDLE_ERROR(cudaFree(this->ptrDevImageInit));
    HANDLE_ERROR(cudaFree(this->ptrDevImageA));
    HANDLE_ERROR(cudaFree(this->ptrDevImageB));
}

/**
 * Override
 */
void HeatTransfertAdvanced::process(uchar4* ptrDevPixels, int width, int height)
{
    if (this->iteration % 2 == 0)
    {
        diffuseAdvanced<<<this->dg, this->db>>>(this->ptrDevImageA, this->ptrDevImageB, this->width, this->height, this->propagationSpeed);
        crushAdvanced<<<this->dg, this->db>>>(this->ptrDevImageHeater, this->ptrDevImageB, this->totalPixels);
        displayAdvanced<<<this->dg, this->db>>>(this->ptrDevImageB, ptrDevPixels, this->totalPixels);
    }
    else
    {
        diffuseAdvanced<<<this->dg, this->db>>>(this->ptrDevImageB, this->ptrDevImageA, this->width, this->height, this->propagationSpeed);
        crushAdvanced<<<this->dg, this->db>>>(this->ptrDevImageHeater, this->ptrDevImageA, this->totalPixels);
        displayAdvanced<<<this->dg, this->db>>>(this->ptrDevImageA, ptrDevPixels, this->totalPixels);
    }
}

/**
 * Override
 */
void HeatTransfertAdvanced::animationStep()
{
    this->iteration++;
}

/**
 * Override
 */
float HeatTransfertAdvanced::getAnimationPara()
{
    return this->iteration;
}

/**
 * Override
 */
int HeatTransfertAdvanced::getW()
{
    return this->width;
}

/**
 * Override
 */
int HeatTransfertAdvanced::getH()
{
    return this->height;
}

/**
 * Override
 */
string HeatTransfertAdvanced::getTitle()
{
    return this->title;
}
