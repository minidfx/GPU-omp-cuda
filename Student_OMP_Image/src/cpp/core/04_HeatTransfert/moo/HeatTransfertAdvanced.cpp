#include <iostream>
#include <omp.h>
#include <climits>
#include <cstring>

#include "IndiceTools.h"

#include "HeatTransfertAdvanced.h"

using cpu::IntervalI;

void diffuseEntrelacement(float* ptrImageInput, float* ptrImageOutput, unsigned int width, unsigned int height, float propagationSpeed);
void crushEntrelacement(float* ptrImageHeater, float* ptrImage, unsigned int arraySize);
void displayEntrelacement(float* ptrImage, uchar4* ptrPixels, unsigned int arraySize);

void diffuseAuto(float* ptrImageInput, float* ptrImageOutput, unsigned int width, unsigned int height, float propagationSpeed);
void crushAuto(float* ptrImageHeater, float* ptrImage, unsigned int arraySize);
void displayAuto(float* ptrImage, uchar4* ptrPixels, unsigned int arraySize);

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
    this->NB_ITERATION_AVEUGLE = 1;
    this->isBufferA = true;

    // Size of all pixels of an image
    size_t arraySize = sizeof(float) * this->totalPixels;

    // Allocates memory on Host
    this->ptrTabImageHeater = (float*) malloc(arraySize);
    this->ptrTabImageInit = (float*) malloc(arraySize);
    this->ptrTabImageA = (float*) malloc(arraySize);
    this->ptrTabImageB = (float*) malloc(arraySize);

    // Set default values
    memset(this->ptrTabImageHeater, 0, arraySize);
    memset(this->ptrTabImageInit, 0, arraySize);
    memset(this->ptrTabImageA, 0, arraySize);
    memset(this->ptrTabImageB, 0, arraySize);

    unsigned int s = 0;
    while(s++ < this->totalPixels)
    {
        this->ptrTabImageInit[s] = 0.0;

        int i, j;
        IndiceTools::toIJ(s, width, &i, &j);

        if (i >= 187 && i < 312 && j >= 187 && j < 312)
        {
            this->ptrTabImageHeater[s] = 1.0;
        }
        else if ((i >= 111 && i < 121 && j >= 111 && j < 121) || (i >= 111 && i < 121 && j >= 378 && j < 388) || (i >= 378 && i < 388 && j >= 111 && j < 121)
        || (i >= 378 && i < 388 && j >= 378 && j < 388) || (i >= 378 && i < 388 && j >= 378 && j < 388) || (i >= 378 && i < 388 && j >= 378 && j < 388))
        {
            this->ptrTabImageHeater[s] = 0.2;
        }
        else
        {
            this->ptrTabImageHeater[s] = 0.0;
        }
    }

    this->listener();
}

HeatTransfertAdvanced::~HeatTransfertAdvanced()
{
    // Release resources GPU side
    free(this->ptrTabImageHeater);
    free(this->ptrTabImageInit);
    free(this->ptrTabImageA);
    free(this->ptrTabImageB);
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
        ptrImageInput = this->ptrTabImageA;
        ptrImageOutput = this->ptrTabImageB;
    }
    else
    {
        ptrImageInput = this->ptrTabImageB;
        ptrImageOutput = this->ptrTabImageA;
    }

    switch (this->parallelPattern)
    {
        case OMP_ENTRELACEMENT:
            diffuseEntrelacement(ptrImageInput, ptrImageOutput, this->width, this->height, this->propagationSpeed);
            crushEntrelacement(this->ptrTabImageHeater, ptrImageOutput, this->totalPixels);

            if(this->iteration % this->NB_ITERATION_AVEUGLE == 0)
            {
                displayEntrelacement(ptrImageOutput, ptrTabPixels, this->totalPixels);
            }
        break;

        case OMP_FORAUTO:
            diffuseAuto(ptrImageInput, ptrImageOutput, this->width, this->height, this->propagationSpeed);
            crushAuto(this->ptrTabImageHeater, ptrImageOutput, this->totalPixels);

            if(this->iteration % this->NB_ITERATION_AVEUGLE == 0)
            {
                displayAuto(ptrImageOutput, ptrTabPixels, this->totalPixels);
            }
        break;
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
