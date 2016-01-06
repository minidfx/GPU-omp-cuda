#include "../../../02_Mandelbrot_MultiGPU/moo/host/MandelbrotMGPU.h"

#include <assert.h>

#include "Device.h"
#include "MathTools.h"

using cpu::IntervalI;

extern __global__ void mandelbrot(uchar4* ptrDevPixels, int w, int h, DomaineMath domaineMath, int n);

MandelbrotMGPU::MandelbrotMGPU(int w, int h, int nMin, int nMax, string title) :
	variateurN(IntervalI(nMin, nMax), 1)
    {
    // Inputs
    this->w = w;
    this->h = h;
    this->n = nMax;

    this->ptrDomaineMathInit = new DomaineMath(-2.1, -1.3, 0.8, 1.3);

    // Tools
    this->dg = dim3(16, 8, 1); // disons a optimiser
    this->db = dim3(16, 16, 1); // disons a optimiser
    this->deviceID = Device::getDeviceId();
    this->deviceIDBottom = 5;
    this->size = sizeof(uchar4) * w * (h / 2);

    //Outputs
    this->title = title;

    // Memory management

    HANDLE_ERROR(cudaSetDevice(deviceIDBottom));
    HANDLE_ERROR(cudaMalloc(&ptrDevTab1, size));
    HANDLE_ERROR(cudaMemset(ptrDevTab1, 0, size));
    HANDLE_ERROR(cudaSetDevice(deviceID));

    // Check:
    Device::assertDim(dg, db);
    }

MandelbrotMGPU::~MandelbrotMGPU()
    {
    HANDLE_ERROR(cudaSetDevice(deviceIDBottom));
    HANDLE_ERROR(cudaFree(ptrDevTab1));
    HANDLE_ERROR(cudaSetDevice(deviceID));

    delete this->ptrDomaineMathInit;
    }
/**
 * Override
 */
void MandelbrotMGPU::process(uchar4* ptrDevPixels0, int w, int h, const DomaineMath& domaineMath)
    {
    DomaineMath dmTop = domaineMath;
    dmTop.y1 = domaineMath.y1 / 2;

    DomaineMath dmBottom = domaineMath;
    dmBottom.y0 = domaineMath.y0 + (domaineMath.y1 - domaineMath.y0) / 2;

#pragma omp parallel sections
	{
#pragma omp section
	    {
	mandelbrot<<<dg,db>>>(ptrDevPixels0,w,h/2,dmTop, n);
	}

#pragma omp section
	{
	int deviceID = Device::getDeviceId();
	HANDLE_ERROR(cudaSetDevice(deviceIDBottom));

	this->ptrDevBottomImage0 = ptrDevPixels0 + (w * (h / 2));

	// kernel
mandelbrot<<<dg,db>>>(ptrDevTab1,w,h/2,dmBottom, n);

	    // MM copie sur device0 (affichage)
	    		HANDLE_ERROR(cudaMemcpy(ptrDevBottomImage0, ptrDevTab1, size, cudaMemcpyDeviceToDevice));
	HANDLE_ERROR(cudaSetDevice(deviceID));
	}
    }
}

/**
 * Override
 * Call periodicly by the API
 */
void MandelbrotMGPU::animationStep()
{
this->n = variateurN.varierAndGet();
}

/*--------------*\
 |*	get	 *|
 \*--------------*/

/**
 * Override
 */
DomaineMath* MandelbrotMGPU::getDomaineMathInit(void)
{
return ptrDomaineMathInit;
}

/**
 * Override
 */
float MandelbrotMGPU::getAnimationPara(void)
{
return n;
}

/**
 * Override
 */
int MandelbrotMGPU::getW(void)
{
return w;
}

/**
 * Override
 */
int MandelbrotMGPU::getH(void)
{
return h;
}

/**
 * Override
 */
string MandelbrotMGPU::getTitle(void)
{
return title;
}
