#include <iostream>
#include <assert.h>

#include "RayTracing.h"
#include "Device.h"
#include "MathTools.h"
#include "SphereFactory.h"
#include "ConstantMemoryLink.h"

using cpu::IntervalI;

#define LENGTH 1000
__constant__ Sphere TAB_DATA_CM[LENGTH];

/*----------------------------------------------------------------------*\
|*			Declaration                     *|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
|*		Imported	    *|
 \*-------------------------------------*/

extern __global__ void rayTracing(uchar4* ptrDevPixels, int w, int h, Sphere* ptrDevSpheres, int n, float t);

extern __global__ void rayTracingSM(uchar4* ptrDevPixels, int w, int h, Sphere* ptrDevSpheres, int n, float t);

/*--------------------------------------*\
|*		Public			*|
 \*-------------------------------------*/

/*--------------------------------------*\
|*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
|*			Implementation                  *|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
|*		Public			*|
 \*-------------------------------------*/

/*-------------------------*\
|*	Constructeur	    *|
 \*-------------------------*/

// Global memory : 1
// Constant memory : 2
// Shared memory : 3
RayTracing::RayTracing(int w, int h, int nSphere, float dt, int memoryType, string title)
    {
    // Inputs
    this->w = w;
    this->h = h;
    this->nSphere = nSphere;
    this->dt = dt;

    // Tools
    this->dg = dim3(16, 2, 1); // disons a optimiser
    this->db = dim3(32, 4, 1); // disons a optimiser

    int margin = 50;
    this->ptrSpheres = SphereFactory::createSpheres(nSphere, w, h, margin);

    this->t = 0.0f;

    //Outputs
    this->title = title;

    // Check:
    Device::assertDim(dg, db);

    // CM
    if (memoryType == 2)
	{
	this->ptrProcessFunction = &RayTracing::processCM;

	copySpheresToConstantMemory();
	}

    sizeSpheres = sizeof(Sphere) * LENGTH;

    if (memoryType == 1 || memoryType == 3)
	{
	this->ptrProcessFunction = memoryType == 1 ? &RayTracing::processGM : &RayTracing::processSM;

	HANDLE_ERROR(cudaMalloc(&ptrDevSpheres, sizeSpheres));
	HANDLE_ERROR(cudaMemcpy(ptrDevSpheres, ptrSpheres, sizeSpheres, cudaMemcpyHostToDevice));
	}
    }

RayTracing::~RayTracing()
    {
    delete[] this->ptrSpheres;
    HANDLE_ERROR(cudaFree(ptrDevSpheres));
    }

/*-------------------------*\
|*	Methode		    *|
 \*-------------------------*/

ConstantMemoryLink constantMemoryLink(void)
    {
    Sphere* ptrDevTabData;
    size_t sizeAll = LENGTH * sizeof(Sphere);
    HANDLE_ERROR(cudaGetSymbolAddress((void ** )&ptrDevTabData, TAB_DATA_CM));
    ConstantMemoryLink cmLink =
	{
	(void**) ptrDevTabData, LENGTH, sizeAll
	};

    return cmLink;
    }

void RayTracing::copySpheresToConstantMemory()
    {
    ConstantMemoryLink cmLink = constantMemoryLink();
    this->ptrDevSpheres = (Sphere*) cmLink.ptrDevTab;
    size_t sizeALL = cmLink.sizeAll;

    HANDLE_ERROR(cudaMemcpy(ptrDevSpheres, ptrSpheres, sizeALL, cudaMemcpyHostToDevice));
    }

/**
 * Override
 * Call periodicly by the API
 */
void RayTracing::process(uchar4* ptrDevPixels, int w, int h)
    {
    (this->*ptrProcessFunction)(ptrDevPixels, w, h);
    }

void RayTracing::processGM(uchar4* ptrDevPixels, int w, int h)
    {
rayTracing<<<this->dg, this->db>>>(ptrDevPixels, w, h, this->ptrDevSpheres, this->nSphere, this->t);
}
void RayTracing::processCM(uchar4* ptrDevPixels, int w, int h)
{
rayTracing<<<this->dg, this->db>>>(ptrDevPixels, w, h, this->ptrDevSpheres, this->nSphere, this->t);
}
void RayTracing::processSM(uchar4* ptrDevPixels, int w, int h)
{
rayTracingSM<<<this->dg, this->db, sizeSpheres>>>(ptrDevPixels, w, h, this->ptrDevSpheres, this->nSphere, this->t);
}

/**
 * Override
 * Call periodicly by the API
 */
void RayTracing::animationStep()
{
t += dt;
}

/*--------------*\
|*	get	 *|
 \*--------------*/

/**
 * Override
 */
float RayTracing::getAnimationPara(void)
{
return t;
}

/**
 * Override
 */
int RayTracing::getW(void)
{
return w;
}

/**
 * Override
 */
int RayTracing::getH(void)
{
return h;
}

/**
 * Override
 */
string RayTracing::getTitle(void)
{
return title;
}

/*--------------------------------------*\
|*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
|*			End	                    *|
 \*---------------------------------------------------------------------*/
