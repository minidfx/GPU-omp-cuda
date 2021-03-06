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
                                              ComputeMode computeMode,
                                              bool isMultiGPU) : variateurN(IntervalI(0, INT_MAX), 1)
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
    // this->db = dim3(32, 32, 1);
    // this->dg = dim3(16, 16, 1);

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

        if (i >= 191 && i < 319 && j >= 191 && j < 319)
        {
            imageHeater[s] = 1.0;
        }
        else if ((i >= 113 && i < 123 && j >= 113 && j < 123) || (i >= 113 && i < 123 && j >= 387 && j < 397) || (i >= 387 && i < 397 && j >= 113 && j < 123)
        || (i >= 387 && i < 397 && j >= 387 && j < 397) || (i >= 387 && i < 397 && j >= 387 && j < 397) || (i >= 387 && i < 397 && j >= 387 && j < 397))
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

    this->ptrProcessFunction = isMultiGPU ? &HeatTransfertAdvanced::processMultiGPU : &HeatTransfertAdvanced::processSingleGPU;
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
  (this->*ptrProcessFunction)(ptrDevPixels, width, height);
}

void HeatTransfertAdvanced::processSingleGPU(uchar4* ptrDevPixels, int width, int height)
{
  float* ptrImageOutput;
  float* ptrImageInputANew;

  if (this->isBufferA)
  {
    ptrImageInputANew = this->ptrDevImageA;
    ptrImageOutput = this->ptrDevImageB;
  }
  else
  {
    ptrImageInputANew = this->ptrDevImageB;
    ptrImageOutput = this->ptrDevImageA;
  }

  diffuseAdvanced<<<this->dg, this->db>>>(ptrImageInputANew, ptrImageOutput, this->width, this->height, this->propagationSpeed, this->computeMode);
  crushAdvanced<<<this->dg, this->db>>>(this->ptrDevImageHeater, ptrImageOutput, this->totalPixels);
  // diffusePerPixel<<<this->dg, this->db>>>(ptrImageInputANew, ptrImageOutput, this->width, this->height, this->propagationSpeed, this->computeMode);
  // crushPerPixel<<<this->dg, this->db>>>(this->ptrDevImageHeater, ptrImageOutput, this->totalPixels);

  if(this->iteration % this->NB_ITERATION_AVEUGLE == 0)
  {
    displayAdvanced<<<this->dg, this->db>>>(ptrImageOutput, ptrDevPixels, this->totalPixels);
    // displayPerPixel<<<this->dg, this->db>>>(ptrImageOutput, ptrDevPixels, this->totalPixels);
  }

  this->isBufferA = !this->isBufferA;
}

void HeatTransfertAdvanced::processMultiGPU(uchar4* ptrDevPixels, int width, int height)
{

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
