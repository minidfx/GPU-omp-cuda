#include <assert.h>

#include "Newton.h"
#include "Device.h"
#include "MathTools.h"

using cpu::IntervalI;

extern __global__ void newton(uchar4 *ptrDevPixels, int w, int h, DomaineMath domaineMath, int n);

Newton::Newton(int w, int h, int nMin, int nMax, string title) :
	variateurN(IntervalI(nMin, nMax), 1)
    {
    // Inputs
    this->w = w;
    this->h = h;

    this->ptrDomaineMathInit = new DomaineMath(-1.3, -1.4, 1.4, 1.3);

    // Tools
    this->dg = dim3(8, 8, 1); // disons a optimiser
    this->db = dim3(16, 16, 1); // disons a optimiser

    //Outputs
    this->title = title;

    // Check:
    Device::assertDim(dg, db);
    }

Newton::~Newton()
    {
    delete this->ptrDomaineMathInit;
    }

void Newton::process(uchar4 *ptrDevPixels, int w, int h, const DomaineMath &domaineMath)
    {
newton <<<this->dg, this->db>>> (ptrDevPixels, w, h, domaineMath, n);
}

void Newton::animationStep()
{
this->n = variateurN.varierAndGet(); // in [0,2pi]
}

DomaineMath *Newton::getDomaineMathInit(void)
{
return this->ptrDomaineMathInit;
}

float Newton::getAnimationPara(void)
{
return this->n;
}

int Newton::getW(void)
{
return this->w;
}

int Newton::getH(void)
{
return this->h;
}

string Newton::getTitle(void)
{
return this->title;
}
