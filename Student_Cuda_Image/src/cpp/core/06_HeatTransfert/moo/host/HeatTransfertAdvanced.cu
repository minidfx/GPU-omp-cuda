#include <iostream>
#include <omp.h>
#include <climits>

#include "cuda_runtime.h"
#include "Device.h"
#include "IndiceTools.h"

#include "HeatTransfertAdvanced.h"

using cpu::IntervalI;

HeatTransfertAdvanced::HeatTransfertAdvanced(unsigned int width,
                                              unsigned int height,
                                              float propagationSpeed,
                                              string title,
                                              ComputeMode computeMode) : variateurN(IntervalI(0, INT_MAX), 1)
{
    // Inputs
    this->width = width;
    this->height = height;
    this->totalPixels = width * height;
    this->title = title;
    this->computeMode = computeMode;

    // Tools
    this->iteration = 0;
    this->propagationSpeed = propagationSpeed;
    this->NB_ITERATION_AVEUGLE = 20;
    this->isBufferA = true;

    // Cuda grid dimensions
    this->dg = dim3(8, 8, 1);
    this->db = dim3(16, 16, 1);

    // Check
    Device::assertDim(this->dg, this->db);
    Device::printAll();

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
    HANDLE_ERROR(cudaMemset(this->ptrDevImageHeater, 0, arraySize));
    HANDLE_ERROR(cudaMemset(this->ptrDevImageInit, 0, arraySize));
    HANDLE_ERROR(cudaMemset(this->ptrDevImageA, 0, arraySize));
    HANDLE_ERROR(cudaMemset(this->ptrDevImageB, 0, arraySize));

    // Copy images from CPU to GPU
    HANDLE_ERROR(cudaMemcpy(this->ptrDevImageHeater, imageHeater, arraySize, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(this->ptrDevImageInit, imageInit, arraySize, cudaMemcpyHostToDevice));

    this->listener();
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
  float* ptrImageOutput;
  float* ptrImageInput;

  if (this->isBufferA)
  {
    ptrImageInput = this->ptrDevImageA;
    ptrImageOutput = this->ptrDevImageB;
  }
  else
  {
    ptrImageInput = this->ptrDevImageB;
    ptrImageOutput = this->ptrDevImageA;
  }

  diffuseAdvanced<<<this->dg, this->db>>>(ptrImageInput, ptrImageOutput, this->width, this->height, this->propagationSpeed, this->computeMode);
  crushAdvanced<<<this->dg, this->db>>>(this->ptrDevImageHeater, ptrImageOutput, this->totalPixels);

  if(this->iteration % this->NB_ITERATION_AVEUGLE == 0)
  {
    displayAdvanced<<<this->dg, this->db>>>(ptrImageOutput, ptrDevPixels, this->totalPixels);
  }

  this->isBufferA = !this->isBufferA;
}

/**
 * Override
 */
void HeatTransfertAdvanced::animationStep()
{
    this->iteration = this->variateurN.varierAndGet();
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

void HeatTransfertAdvanced::listener()
{
  //setMouseListener(this->ptrMouseListener = new SimpleMouseListener());
}
