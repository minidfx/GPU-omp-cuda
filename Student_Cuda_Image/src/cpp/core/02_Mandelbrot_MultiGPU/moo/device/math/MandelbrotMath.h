#ifndef MANDELBROT_MATH_H_
#define MANDELBROT_MATH_H_

#include <math.h>

#include "../../../../02_Mandelbrot_MultiGPU/moo/device/math/MandelbrotMathBase.h"
#include "ColorTools.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

class MandelbrotMath : public MandelbrotMathBase
    {

	/*--------------------------------------*\
	|*		Constructor		*|
	 \*-------------------------------------*/

    public:

	__device__
    MandelbrotMath(int n) : MandelbrotMathBase(n)
	    {
        // nothing
	    }

	__device__
    virtual ~MandelbrotMath(void)
	    {
	    //nothing
	    }

	/*--------------------------------------*\
	|*		Methodes		*|
	 \*-------------------------------------*/

    protected:

	__device__
	virtual int getK(float x, float y)
	    {
	    float a = 0;
	    float b = 0;

	    int k = 0;

	    while (!isDivergent(a, b) && k <= this->n)
		{
		float aCopy = a;
		a = (aCopy * aCopy - b * b) + x;
		b = 2.0 * aCopy * b + y;

		k++;
		}

	    return k;
	    }

	/*--------------------------------------*\
	|*		Attributs		*|
	 \*-------------------------------------*/

    };

#endif

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
