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
    	FractalMathBase() : calibreur(IntervalF(-1, 1), IntervalF(0, 1))
        {
        }

    // -------------------------------------------------
    // Method for coloring the pixel passed as argument
    // -------------------------------------------------
    void colorXY(uchar4* ptrColor, double x, double y, int max)
    {
        int k = this->getK(x, y, max);

        if(k < max)
        {
            double hue = (1./max) * k;

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
        virtual bool isDivergent(double x, double y) = 0;

        // ------------------
        // Builds the Z list
        // ------------------
        virtual int getK(double x, double y, int max) = 0;

        // ------------------------------
        // Class for calibrating the
        CalibreurF calibreur;
};

#endif
