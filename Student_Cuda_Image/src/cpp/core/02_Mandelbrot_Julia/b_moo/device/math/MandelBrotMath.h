#ifndef MANDELBROT_MATH_H_
#define MANDELBROT_MATH_H_

#include <iostream>

#include "CalibreurF.h"
#include "ColorTools.h"
#include <math.h>
#include <list>

#include "FractalMathBase.h"

class MandelBrotMath: public FractalMathBase
    {
    public:

	__device__ MandelBrotMath() :
		FractalMathBase()
	    {
	    }

    protected:

	__device__
	virtual bool isDivergent(float a, float b)
	    {
	    return a * a + b * b > 4;
	    }

	__device__
	virtual int getK(float x, float y, int max)
	    {
	    float a = 0;
	    float b = 0;

	    int k = 0;

	    while (!this->isDivergent(a, b) && k < max)
		{
		float aCopy = a;
		a = (aCopy * aCopy - b * b) + x;
		b = 2. * aCopy * b + y;

		k++;
		}

	    return k;
	    }
    };

#endif
