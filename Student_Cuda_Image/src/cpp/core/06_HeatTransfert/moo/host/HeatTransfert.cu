#include <iostream>
#include <omp.h>

#include "cuda_runtime.h"
#include "Device.h"
#include "HeatTransfert.h"
#include "IndiceTools.h"

__global__ void diffuse(float* ptrImageInput, float* ptrImageOutput, unsigned int width, unsigned int height, float propagationSpeed);
__global__ void crush(float* ptrImageHeater, float* ptrImage, unsigned int size);
__global__ void display(float* ptrImage, uchar4* ptrPixels, unsigned int size);

HeatTransfert::HeatTransfert(unsigned int w, unsigned int h, float propagationSpeed, string title)
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

    for (int s = 0; s < this->totalPixels; s++)
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

    size_t arraySize = sizeof(float) * this->totalPixels;
    HANDLE_ERROR(cudaMalloc(&this->ptrDevImageHeater, arraySize));
    HANDLE_ERROR(cudaMalloc(&this->ptrDevImageInit, arraySize));
    HANDLE_ERROR(cudaMalloc(&this->ptrDevImageA, arraySize));
    HANDLE_ERROR(cudaMalloc(&this->ptrDevImageB, arraySize));
    HANDLE_ERROR(cudaMemcpy(this->ptrDevImageHeater, imageHeater, arraySize, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(this->ptrDevImageInit, imageInit, arraySize, cudaMemcpyHostToDevice));

    // Initialization
    crush<<<this->dg, this->db>>>(this->ptrDevImageHeater, this->ptrDevImageInit, this->totalPixels);
    diffuse<<<this->dg, this->db>>>(this->ptrDevImageInit, this->ptrDevImageA, width, height, propagationSpeed);
    crush<<<this->dg, this->db>>>(this->ptrDevImageHeater, this->ptrDevImageA, this->totalPixels);
}

HeatTransfert::~HeatTransfert()
{
    HANDLE_ERROR(cudaFree(this->ptrDevImageHeater));
    HANDLE_ERROR(cudaFree(this->ptrDevImageInit));
    HANDLE_ERROR(cudaFree(this->ptrDevImageA));
    HANDLE_ERROR(cudaFree(this->ptrDevImageB));
}

/**
 * Override
 */
void HeatTransfert::process(uchar4* ptrDevPixels, int width, int height)
{
    if (this->iteration % 2 == 0)
    {
        diffuse<<<dg,db>>>(this->ptrDevImageA, this->ptrDevImageB, this->width, this->height, this->propagationSpeed);
        crush<<<dg,db>>>(this->ptrDevImageHeater, this->ptrDevImageB, this->totalPixels);
        display<<<dg,db>>>(this->ptrDevImageB, ptrDevPixels, this->totalPixels);
    }
    else
    {
        diffuse<<<dg,db>>>(this->ptrDevImageB, this->ptrDevImageA, this->width, this->height, this->propagationSpeed);
        crush<<<dg,db>>>(this->ptrDevImageHeater, this->ptrDevImageA, this->totalPixels);
        display<<<dg,db>>>(this->ptrDevImageA, ptrDevPixels, this->totalPixels);
    }
}

/**
 * Override
 */
void HeatTransfert::animationStep()
{
    this->iteration++;
}

/**
 * Override
 */
float HeatTransfert::getAnimationPara()
{
    return this->iteration;
}

/**
 * Override
 */
int HeatTransfert::getW()
{
    return this->width;
}

/**
 * Override
 */
int HeatTransfert::getH()
{
    return this->height;
}

/**
 * Override
 */
string HeatTransfert::getTitle()
{
    return this->title;
}
