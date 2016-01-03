#include <assert.h>

#include "GraphicGenerator.h"

#include "MathTools.h"
#include "Device.h"

using cpu::IntervalI;
using std::cout;
using std::endl;
using std::string;

__global__
extern void calculate(uchar4* ptrTabPixels,
                      int width,
                      int height,
                      const DomaineMath mathDomain,
                      bool isMandelbrot,
                      int max);

// -------------------------
// Constructor & Destructor
// -------------------------
GraphicGenerator::GraphicGenerator(unsigned int width,
                                   unsigned int height,
                                   unsigned int nMin,
                                   unsigned int nMax,
                                   DomaineMath *ptrMathDomain,
                                   bool isMandelbrot,
                                   string title) : variateurN(IntervalI(nMin, nMax), 1)
{
    // Basic settings
    this->width = width;
    this->height = height;
    this->title = title;

    // Mathematical settings
    this->ptrMathDomain = ptrMathDomain;
    this->isMandelbrot = isMandelbrot;

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
void GraphicGenerator::process(uchar4* ptrTabPixels,
                               int width,
                               int height,
                               const DomaineMath& mathDomain)
{
    calculate<<<this->dg, this->db>>>(ptrTabPixels,
                                        width,
                                        height,
                                        mathDomain,
                                        this->isMandelbrot,
                                        this->max);
}

void GraphicGenerator::animationStep()
{
    this->max = variateurN.varierAndGet();
}

float GraphicGenerator::getAnimationPara()
{
    return variateurN.get();
}

string GraphicGenerator::getTitle()
{
    return title;
}

int GraphicGenerator::getW()
{
    return width;
}

int GraphicGenerator::getH()
{
    return height;
}

DomaineMath* GraphicGenerator::getDomaineMathInit()
{
    return this->ptrMathDomain;
}
