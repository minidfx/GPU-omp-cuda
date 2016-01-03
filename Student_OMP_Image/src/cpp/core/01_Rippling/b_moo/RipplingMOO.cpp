#include <iostream>
#include <omp.h>

#include "RipplingMOO.h"
#include "OmpTools.h"
#include "IndiceTools.h"

using std::cout;
using std::endl;
using std::string;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

RipplingMOO::RipplingMOO(unsigned int w, unsigned int h, float dt) : math(*(new RipplingMath(w, h)))
    {
    // Inputs
    this->w = w;
    this->h = h;
    this->dt = dt;

    // Tools
    this->t = 0;
    this->parallelPatern = OMP_MIXTE;
    }

RipplingMOO::~RipplingMOO(void)
    {
    }

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

/**
 * Override
 */
void RipplingMOO::process(uchar4* ptrTabPixels,int w,int h)
    {
    switch (parallelPatern)
	{

	case OMP_ENTRELACEMENT: // Plus lent sur CPU
	    {
	    entrelacementOMP(ptrTabPixels, w, h);
	    break;
	    }

	case OMP_FORAUTO: // Plus rapide sur CPU
	    {
	    forAutoOMP(ptrTabPixels, w, h);
	    break;
	    }

	case OMP_MIXTE: // Pour tester que les deux implementations fonctionnent
	    {
	    // Note : Des saccades peuvent apparaitre � cause de la grande difference de fps entre la version entrelacer et auto
	    static bool isEntrelacement = true;
	    if (isEntrelacement)
		{
		entrelacementOMP(ptrTabPixels, w, h);
		}
	    else
		{
		forAutoOMP(ptrTabPixels, w, h);
		}
	    isEntrelacement = !isEntrelacement; // Pour swithcer a chaque iteration
	    break;
	    }
	}
    }

/**
 * Override
 */
void RipplingMOO::animationStep()
    {
    t += dt;
    }

/*--------------*\
 |*	get	*|
 \*-------------*/

/**
 * Override
 */
float RipplingMOO::getAnimationPara()
    {
    return t;
    }

/**
 * Override
 */
int RipplingMOO::getW()
    {
    return w;
    }

/**
 * Override
 */
int RipplingMOO::getH()
    {
    return h;
    }

/**
 * Override
 */
string RipplingMOO::getTitle()
    {
    return "OMP Rippling";
    }

/*-------------*\
 |*     set	*|
 \*------------*/

void RipplingMOO::setParallelPatern(ParallelPatern parallelPatern)
    {
    this->parallelPatern = parallelPatern;
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/**
 * Code entrainement Cuda
 */
void RipplingMOO::entrelacementOMP(uchar4* ptrTabPixels, int w, int h) {

  const int NB_THREADS = OmpTools::setAndGetNaturalGranularity();

  #pragma omp parallel
  {
    const int TID = OmpTools::getTid();
    const int n = w * h;
    int s = TID;
    while ( s < n ) {
      int i, j;
      IndiceTools::toIJ(s, this->w, &i, &j);
      this->math.colorIJ(&ptrTabPixels[s], i, j, this->t);
      s += NB_THREADS;
    }
  }
}

/**
 * Code naturel et direct OMP
 */
void RipplingMOO::forAutoOMP(uchar4* ptrTabPixels, int w, int h)
    {
	const int n = w * h;

#pragma omp parallel for
	for (int s = 0; s < n; s ++)
	    {
		int i, j;
		IndiceTools::toIJ(s, this->w, &i, &j);
		this->math.colorIJ(&ptrTabPixels[s], i, j, this->t);
	    }
    }
/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
