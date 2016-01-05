#ifndef FRACTAL_MATH_H_
#define FRACTAL_MATH_H_

#include "CalibreurF.h"
#include "ColorTools.h"
#include <math.h>

// -----------------------------------
// Base class for calculating fractal
// -----------------------------------
class FractalMathBase
    {
    public:

	// ---------------------------
	// Constructor & Destructor
	// ---------------------------
	__device__ FractalMathBase() :
		calibreur(IntervalF(-1, 1), IntervalF(0, 1))
	    {
	    }

	// -------------------------------------------------
	// Method for coloring the pixel passed as argument
	// -------------------------------------------------
	__device__
	void colorXY(uchar4* ptrColor, float x, float y, int max)
	    {
	    int k = this->getK(x, y, max);

	    if (k < max)
		{
		float hue = (1. / max) * k;

		ColorTools::HSB_TO_RVB(hue, ptrColor);
		}
	    else
		{
		ptrColor->x = 0;
		ptrColor->y = 0;
		ptrColor->z = 0;
		}

	    ptrColor->w = 255;
	    }

    protected:

	// ----------------------------------------------
	// Determines whether the pixel divergent or not passing
	// an element of the list built by the method buildZ.
	// ----------------------------------------------
	__device__
	virtual bool isDivergent(float x, float y) = 0;

	// ------------------
	// Builds the Z list
	// ------------------
	__device__
	virtual int getK(float x, float y, int max) = 0;

	// -------------------------------------
	// Class for transforming color for HSV
	// -------------------------------------
	CalibreurF calibreur;
    };

#endif
