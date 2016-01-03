#include <iostream>
#include <math.h>

#include "GraphicGenerator.h"

#include "OmpTools.h"
#include "MathTools.h"

#include "IndiceTools.h"
#include "FractalMathBase.h"

using std::cout;
using std::endl;
using std::string;

// -------------------------
// Constructor & Destructor
// -------------------------
GraphicGenerator::GraphicGenerator(unsigned int width,
                                   unsigned int height,
                                   unsigned int nMin,
                                   unsigned int nMax,
                                   DomaineMath *ptrMathDomain,
                                   FractalMathBase *ptrFractalMath,
                                   string title) : variateurN(IntervalI(nMin, nMax), 1)
{
    // Basic settings
    this->width = width;
    this->height = height;
    this->title = title;
    this->parallelPattern = OMP_MIXTE;

    // Mathematical settings
    this->ptrMathDomain = ptrMathDomain;
    this->ptrFractalMath = ptrFractalMath;

    // OMP Settings
    this->nbThread = OmpTools::setAndGetNaturalGranularity();
}

GraphicGenerator::~GraphicGenerator()
{
    delete this->ptrFractalMath;
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

void GraphicGenerator::animationStep()
{
    variateurN.varierAndGet();
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

void GraphicGenerator::setParallelPatern(ParallelPatern parallelPatternEnum)
{
    this->parallelPattern = parallelPatternEnum;
}

void GraphicGenerator::forAutoOMP(uchar4* ptrTabPixels,
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

void GraphicGenerator::entrelacementOMP(uchar4* ptrTabPixels,
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

void GraphicGenerator::workPixel(uchar4* ptrColor,
                                           int i,
                                           int j,
                                           const DomaineMath& mathDomain)
{
    // (i,j) domaine ecran dans N*
    // (x,y) domaine math dans R*

    double x;
    double y;

    mathDomain.toXY(i, j, &x, &y);

    int max = variateurN.get();
    this->ptrFractalMath->colorXY(ptrColor, x, y, max);
}
