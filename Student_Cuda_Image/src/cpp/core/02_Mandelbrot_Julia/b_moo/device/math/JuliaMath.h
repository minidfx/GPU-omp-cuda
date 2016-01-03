#ifndef JULIA_MATH_H_
#define JULIA_MATH_H_

#include <iostream>

#include "CalibreurF.h"
#include "ColorTools.h"
#include <math.h>
#include <list>

#include "FractalMathBase.h"

class JuliaMath : public FractalMathBase
{
    public:

        __device__
        JuliaMath(double constantA, double constantB) : FractalMathBase()
        {
            this->constantA = constantA;
            this->constantB = constantB;
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
            double a = x;
    	    double b = y;

    	    int k = 0;

    	    while(!this->isDivergent(a, b) && k < max)
    		{
    			double aCopy = a;
    			a = (aCopy*aCopy - b*b) + this->constantA;
    			b = 2. * aCopy * b + this->constantB;

    			k++;
    		}

    	    return k;
        }

    private:

        double constantA;
        double constantB;
};

#endif
