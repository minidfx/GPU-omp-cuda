#ifndef MANDELBROT_MATH_H_
#define MANDELBROT_MATH_H_

#include <iostream>

#include "CalibreurF.h"
#include "ColorTools.h"
#include <math.h>
#include <list>

#include "FractalMathBase.h"

class MandelBrotMath : public FractalMathBase
{
    public:

        __device__
        MandelBrotMath() : FractalMathBase()
        {
        }

    protected:

        __device__
        virtual bool isDivergent(double a, double b)
        {
            return a*a + b*b > 4;
        }

        __device__
        virtual int getK(double x, double y, int max)
        {
            double a = 0;
    	    double b = 0;

    	    int k = 0;

    	    while(!this->isDivergent(a, b) && k < max)
    		{
    			double aCopy = a;
    			a = (aCopy*aCopy - b*b) + x;
    			b = 2. * aCopy * b + y;

    			k++;
    		}

    	    return k;
        }
};

#endif
