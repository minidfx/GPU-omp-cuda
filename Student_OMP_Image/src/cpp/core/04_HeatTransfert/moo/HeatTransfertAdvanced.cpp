#include <iostream>
#include <omp.h>
#include <climits>
#include <stdio.h>

#include "IndiceTools.h"

#include "HeatTransfertAdvanced.h"

using cpu::IntervalI;

void diffuseAdvanced(float* ptrImageInput, float* ptrImageOutput, unsigned int width, unsigned int height, float propagationSpeed);
void crushAdvanced(float* ptrImageHeater, float* ptrImage, unsigned int arraySize);
void displayAdvanced(float* ptrImage, uchar4* ptrPixels, unsigned int arraySize);

HeatTransfertAdvanced::HeatTransfertAdvanced(unsigned int width, unsigned int height, float propagationSpeed, string title) : variateurN(IntervalI(0, INT_MAX), 1)
{
    // Inputs
    this->width = width;
    this->height = height;
    this->totalPixels = width * height;
    this->title = title;

    // Tools
    this->iteration = 0;
    this->propagationSpeed = propagationSpeed;
    this->NB_ITERATION_AVEUGLE = 20;
    this->isBufferA = true;

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

    // Allocates memory on Host
    this->ptrDevImageHeater = (float*) malloc(arraySize);
    this->ptrDevImageInit = (float*) malloc(arraySize);
    this->ptrDevImageA = (float*) malloc(arraySize);
    this->ptrDevImageB = (float*) malloc(arraySize);

    // Set default values
    // memset(this->ptrDevImageHeater, 0, arraySize);
    // memset(this->ptrDevImageInit, 0, arraySize);
    // memset(this->ptrDevImageA, 0, arraySize);
    // memset(this->ptrDevImageB, 0, arraySize);

    this->listener();
}

HeatTransfertAdvanced::~HeatTransfertAdvanced()
{
    // Release resources GPU side
    free(this->ptrDevImageHeater);
    free(this->ptrDevImageInit);
    free(this->ptrDevImageA);
    free(this->ptrDevImageB);
}

/**
 * Override
 */
void HeatTransfertAdvanced::process(uchar4* ptrTabPixels, int width, int height)
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

  diffuseAdvanced(ptrImageInput, ptrImageOutput, this->width, this->height, this->propagationSpeed);
  crushAdvanced(this->ptrDevImageHeater, ptrImageOutput, this->totalPixels);

  if(this->iteration % this->NB_ITERATION_AVEUGLE == 0)
  {
    displayAdvanced(ptrImageOutput, ptrTabPixels, this->totalPixels);
  }

  this->isBufferA = !this->isBufferA;
}

void HeatTransfertAdvanced::setParallelPatern(ParallelPatern parallelPatternEnum)
{
    this->parallelPattern = parallelPatternEnum;
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
