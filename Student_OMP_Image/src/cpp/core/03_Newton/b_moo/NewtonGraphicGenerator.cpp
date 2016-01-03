#include <iostream>
#include <math.h>

#include "NewtonGraphicGenerator.h"

#include "OmpTools.h"
#include "MathTools.h"

#include "IndiceTools.h"
#include "NewtonMath.h"

using std::cout;
using std::endl;
using std::string;

// -------------------------
// Constructor & Destructor
// -------------------------
NewtonGraphicGenerator::NewtonGraphicGenerator(unsigned int width,
                                   unsigned int height,
                                   unsigned int max,
                                   DomaineMath *ptrMathDomain,
                                   NewtonMath *ptrNewtonMath,
                                   string title) : variateurN(IntervalI(1, max), 1)
{
    // Basic settings
    this->width = width;
    this->height = height;
    this->title = title;
    this->parallelPattern = OMP_MIXTE;

    // Mathematical settings
    this->ptrMathDomain = ptrMathDomain;
    this->ptrNewtonMath = ptrNewtonMath;

    // OMP Settings
    this->nbThread = OmpTools::setAndGetNaturalGranularity();
}

NewtonGraphicGenerator::~NewtonGraphicGenerator()
{
    delete this->ptrNewtonMath;
    delete this->ptrMathDomain;
}

// ------------------
// Overrides members
// ------------------
void NewtonGraphicGenerator::process(uchar4* ptrTabPixels,
                               int width,
                               int height,
                               const DomaineMath& mathDomain)
{
    switch (this->parallelPattern)
	{
        // Plus lent sur CPU
    	case OMP_ENTRELACEMENT:
    	    entrelacementOMP(ptrTabPixels, width, height, mathDomain);
	    break;

        // Plus rapide sur CPU
    	case OMP_FORAUTO:
    	    forAutoOMP(ptrTabPixels, width, height, mathDomain);
	    break;

        // Pour tester que les deux implementations fonctionnent
        // Note : Des saccades peuvent apparaitre à cause de la grande difference de fps entre la version entrelacer et auto
    	case OMP_MIXTE:
    	    static bool isEntrelacement = true;
    	    if (isEntrelacement)
    		{
                entrelacementOMP(ptrTabPixels, width, height, mathDomain);
    		}
    	    else
    		{
                forAutoOMP(ptrTabPixels, width, height, mathDomain);
    		}

            // Pour swithcer a chaque iteration
    	    isEntrelacement = !isEntrelacement;
	    break;
	}
}

void NewtonGraphicGenerator::animationStep()
{
    this->variateurN.varierAndGet();
}

float NewtonGraphicGenerator::getAnimationPara()
{
    return this->variateurN.get();
}

string NewtonGraphicGenerator::getTitle()
{
    return title;
}

int NewtonGraphicGenerator::getW()
{
    return width;
}

int NewtonGraphicGenerator::getH()
{
    return height;
}

DomaineMath* NewtonGraphicGenerator::getDomaineMathInit()
{
    return this->ptrMathDomain;
}

void NewtonGraphicGenerator::setParallelPatern(ParallelPatern parallelPatternEnum)
{
    this->parallelPattern = parallelPatternEnum;
}

void NewtonGraphicGenerator::forAutoOMP(uchar4* ptrTabPixels,
                                  int width,
                                  int height,
                                  const DomaineMath& mathDomain)
{
    #pragma omp parallel for
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            //int s = i * W + j;
            // i[0,H[ j[0,W[  --> s[0,W*H[
            int s = IndiceTools::toS(width, i, j);

            workPixel(&ptrTabPixels[s], i, j, mathDomain);
        }
    }
}

void NewtonGraphicGenerator::entrelacementOMP(uchar4* ptrTabPixels,
                                        int width,
                                        int height,
                                        const DomaineMath& mathDomain)
{
    const int totalPixels = width * height;

    #pragma omp parallel
    {
        const int threadId = OmpTools::getTid();
        int s = threadId;

        // Position horizontal of the pixel
        int i;
        // Position vertical of the pixel
        int j;

        while (s < totalPixels)
        {
            // s[0,W*H[ --> i[0,H[ j[0,W[
            IndiceTools::toIJ(s, width, &i, &j);

            workPixel(&ptrTabPixels[s], i, j, mathDomain);

            s += this->nbThread;
        }
    }
}

void NewtonGraphicGenerator::workPixel(uchar4* ptrColor,
                                 int i,
                                 int j,
                                 const DomaineMath& mathDomain)
{
    double x;
    double y;

    mathDomain.toXY(i, j, &x, &y);

    int max = this->variateurN.get();
    this->ptrNewtonMath->colorXY(ptrColor, x, y, max);
}
