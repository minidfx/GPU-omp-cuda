#include <assert.h>
#include <iostream>

#include "GraphicGenerator.h"

#include "MathTools.h"
#include "Device.h"

using cpu::IntervalI;
using std::cout;
using std::endl;
using std::string;

extern __global__ void processMandelbrot(uchar4* ptrTabPixels, int width, int height, DomaineMath mathDomain, int max);

extern __global__ void processJulia(uchar4* ptrDevPixels, int width, int height, DomaineMath mathDomain, int max);

// -------------------------
// Constructor & Destructor
// -------------------------
GraphicGenerator::GraphicGenerator(unsigned int width, unsigned int height, unsigned int nMin, unsigned int nMax, DomaineMath *ptrMathDomain, bool isMandelbrot,
	string title) :
	variateurN(IntervalI(nMin, nMax), 1)
    {
    // Basic settings
    this->width = width;
    this->height = height;
    this->title = title;
    this->max = nMax;

    // Mathematical settings
    this->ptrMathDomain = ptrMathDomain;
    this->isMandelbrot = isMandelbrot;
    this->ptrProcessFunction = isMandelbrot ? &GraphicGenerator::processMandelbrotOnGPU : &GraphicGenerator::processJuliaOnGPU;

    this->dg = dim3(8, 8, 1); // disons a optimiser
    this->db = dim3(16, 16, 1); // disons a optimiser

    Device::assertDim(this->dg, this->db);
    }

GraphicGenerator::~GraphicGenerator()
    {
    delete this->ptrMathDomain;
    }

// ------------------
// Overrides members
// ------------------
void GraphicGenerator::process(uchar4* ptrDevPixels, int width, int height, const DomaineMath& mathDomain)
    {
    (this->*ptrProcessFunction)(ptrDevPixels, width, height, mathDomain);
    }

void GraphicGenerator::processMandelbrotOnGPU(uchar4* ptrTabPixels, int width, int height, const DomaineMath& mathDomain)
    {
processMandelbrot<<<this->dg, this->db>>>(ptrTabPixels, width, height, mathDomain, this->max);
}

void GraphicGenerator::processJuliaOnGPU(uchar4* ptrTabPixels, int width, int height, const DomaineMath& mathDomain)
{
processJulia<<<this->dg, this->db>>>(ptrTabPixels, width, height, mathDomain, this->max);
}

void GraphicGenerator::animationStep()
{
this->max = variateurN.varierAndGet();
}

float GraphicGenerator::getAnimationPara()
{
return this->max;
}

string GraphicGenerator::getTitle()
{
return this->title;
}

int GraphicGenerator::getW()
{
return this->width;
}

int GraphicGenerator::getH()
{
return this->height;
}

DomaineMath* GraphicGenerator::getDomaineMathInit()
{
return this->ptrMathDomain;
}
