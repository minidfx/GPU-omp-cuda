#include "Indice2D.h"
#include "IndiceTools.h"
#include "DomaineMath.h"
#include "cudaTools.h"
#include "Device.h"

#include "FractalMathBase.h"
#include "JuliaMath.h"
#include "MandelBrotMath.h"

// -------
// Headers
// -------
__global__
void calculate(uchar4* ptrTabPixels,
             int width,
             int height,
             const DomaineMath mathDomain,
             bool isMandelbrot,
             int max);

// ---------------
// Implementation
// ---------------
__global__
void calculate(uchar4* ptrTabPixels,
             int width,
             int height,
             const DomaineMath mathDomain,
             bool isMandelbrot,
             int max)
{
    const int totalPixels = width * height;

    const int threadId = Indice2D::tid();
    const int nbThread = Indice2D::nbThread();
    int s = threadId;

    // Position horizontal of the pixel
    int i;
    // Position vertical of the pixel
    int j;
    // Color of pixels
    uchar4 color;

    /*if(isMandelbrot)
    {*/
        MandelBrotMath fractalMath = MandelBrotMath();

        while (s < totalPixels)
        {
            IndiceTools::toIJ(s, width, &i, &j);

            double x;
            double y;

            mathDomain.toXY(i, j, &x, &y);
            fractalMath.colorXY(&color, x, y, max);

            ptrTabPixels[s] = color;

            s += nbThread;
        }
    /*}
    else
    {
        JuliaMath fractalMath = JuliaMath(-0.12, 0.85);

        while (s < totalPixels)
        {
            IndiceTools::toIJ(s, width, &i, &j);

            double x;
            double y;

            mathDomain.toXY(i, j, &x, &y);
            fractalMath.colorXY(&color, x, y, max);

            ptrTabPixels[s] = color;

            s += nbThread;
        }
    }*/
}
