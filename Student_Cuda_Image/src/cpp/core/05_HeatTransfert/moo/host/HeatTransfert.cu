#include <iostream>
#include <omp.h>

#include "cuda_runtime.h"
#include "Device.h"
#include "HeatTransfert.h"
#include "IndiceTools.h"

__global__ void diffuse(float* ptrImageInput, float* ptrImageOutput, unsigned int w, unsigned int h, float propSpeed);
__global__ void crush(float* ptrImageHeater, float* ptrImage, unsigned int size);
__global__ void display(float* ptrImage, uchar4* ptrPixels, unsigned int size);

HeatTransfert::HeatTransfert(unsigned int w, unsigned int h, float* ptrImageInit, float* ptrImageHeater, float propSpeed, string title)
    {
    // Inputs
    this->w = w;
    this->h = h;
    this->wh = w * h;

    // Tools
    this->iteration = 0;
    this->propSpeed = propSpeed;

    // Cuda grid dimensions
    this->dg = dim3(8, 8, 1);
    this->db = dim3(16, 16, 1);

    // Check
    Device::assertDim(dg, db);

    for (int s = 0; s < this->wh; s++)
	{
	ptrImageInit[s] = 0.0;

	int i, j;
	IndiceTools::toIJ(s, w, &i, &j);

	if (i >= 187 && i < 312 && j >= 187 && j < 312)
	    {
	    ptrImageHeater[s] = 1.0;
	    }
	else if ((i >= 111 && i < 121 && j >= 111 && j < 121) || (i >= 111 && i < 121 && j >= 378 && j < 388) || (i >= 378 && i < 388 && j >= 111 && j < 121)
		|| (i >= 378 && i < 388 && j >= 378 && j < 388) || (i >= 378 && i < 388 && j >= 378 && j < 388) || (i >= 378 && i < 388 && j >= 378 && j < 388))
	    {
	    ptrImageHeater[s] = 0.2;
	    }
	else
	    {
	    ptrImageHeater[s] = 0.0;
	    }
	}

    size_t arraySize = sizeof(float) * wh;
    HANDLE_ERROR(cudaMalloc(&this->ptrDevImageHeater, arraySize));
    HANDLE_ERROR(cudaMalloc(&this->ptrDevImageInit, arraySize));
    HANDLE_ERROR(cudaMalloc(&this->ptrDevImageA, arraySize));
    HANDLE_ERROR(cudaMalloc(&this->ptrDevImageB, arraySize));
    HANDLE_ERROR(cudaMemcpy(this->ptrDevImageHeater, ptrImageHeater, arraySize, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(this->ptrDevImageInit, ptrImageInit, arraySize, cudaMemcpyHostToDevice));

    // Initialization
crush<<<this->dg, this->db>>>(this->ptrDevImageHeater, this->ptrDevImageInit, this->wh);
diffuse<<<this->dg, this->db>>>(this->ptrDevImageInit, this->ptrDevImageA, this->w, this->h, this->propSpeed);
crush<<<this->dg, this->db>>>(this->ptrDevImageHeater, this->ptrDevImageA, this->wh);
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
void HeatTransfert::process(uchar4* ptrDevPixels, int w, int h)
{
if (this->iteration % 2 == 0)
    {
diffuse<<<dg,db>>>(this->ptrDevImageA, this->ptrDevImageB, this->w, this->h, this->propSpeed);
crush<<<dg,db>>>(this->ptrDevImageHeater, this->ptrDevImageB, this->wh);
display<<<dg,db>>>(this->ptrDevImageB, ptrDevPixels, this->wh);
}
else
{
diffuse<<<dg,db>>>(this->ptrDevImageB, this->ptrDevImageA, this->w, this->h, this->propSpeed);
crush<<<dg,db>>>(this->ptrDevImageHeater, this->ptrDevImageA, this->wh);
display<<<dg,db>>>(this->ptrDevImageA, ptrDevPixels, this->wh);
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
return this->w;
}

/**
 * Override
 */
int HeatTransfert::getH()
{
return this->h;
}

/**
 * Override
 */
string HeatTransfert::getTitle()
{
return "CUDA HeatTransfert";
}
