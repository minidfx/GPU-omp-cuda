#include <stdio.h>

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
__global__ void processMandelbrot(uchar4* ptrDevPixels, int width, int height, DomaineMath mathDomain, int max);

__global__ void processJulia(uchar4* ptrDevPixels, int width, int height, DomaineMath mathDomain, int max);

__device__ void processFractal(uchar4* ptrDevPixels, int width, int height, DomaineMath mathDomain, int max, FractalMathBase* fractalMath);

// -------------------------
// Implementation Mandelbrot
// -------------------------
__global__ void processMandelbrot(uchar4* ptrDevPixels, int width, int height, DomaineMath mathDomain, int max)
    {
    FractalMathBase* fractalMath = new MandelBrotMath();

    processFractal(ptrDevPixels, width, height, mathDomain, max, fractalMath);

    delete fractalMath;
    }

// -------------------------
// Implementation Julia
// -------------------------
__global__ void processJulia(uchar4* ptrDevPixels, int width, int height, DomaineMath mathDomain, int max)
    {
    FractalMathBase* fractalMath = new JuliaMath(-0.12, 0.85);

    processFractal(ptrDevPixels, width, height, mathDomain, max, fractalMath);

    delete fractalMath;
    }

__device__ void processFractal(uchar4* ptrDevPixels, int width, int height, DomaineMath mathDomain, int max, FractalMathBase* fractalMath)
    {
    const int totalPixels = width * height;

    const int TID = Indice2D::tid();
    const int NB_THREADS = Indice2D::nbThread();
    int s = TID;

    // Position horizontal of the pixel
    int i;
    // Position vertical of the pixel
    int j;

    double x;
    double y;

    uchar4 color;

    while (s < totalPixels)
	{
	IndiceTools::toIJ(s, width, &i, &j);

	mathDomain.toXY(i, j, &x, &y);
	fractalMath->colorXY(&color, x, y, max);

	ptrDevPixels[s] = color;

	s += NB_THREADS;
	}
    }
