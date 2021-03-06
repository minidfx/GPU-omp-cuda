#include "Indice2D.h"
#include "IndiceTools.h"
#include "DomaineMath.h"
#include "cudaTools.h"
#include "Device.h"

#include "FractalMathBase.h"
#include "JuliaMath.h"
#include "MandelBrotMath.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Imported	 	*|
 \*-------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

__global__ void mandelbrot(uchar4* ptrDevPixels, int w, int h, DomaineMath domaineMath, int n);

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

__global__ void mandelbrot(uchar4* ptrDevPixels, int w, int h, DomaineMath domaineMath, int n)
    {
    FractalMathBase* mandelbrotMath = new MandelBrotMath();

    const int TID = Indice2D::tid();
    const int NB_THREAD = Indice2D::nbThread();

    const int WH = w * h;

    uchar4 color;

    double x;
    double y;

    int pixelI;
    int pixelJ;

    int s = TID;
    while (s < WH)
	{
	IndiceTools::toIJ(s, w, &pixelI, &pixelJ);

	domaineMath.toXY(pixelI, pixelJ, &x, &y);

	mandelbrotMath->colorXY(&color, x, y, n);

	ptrDevPixels[s] = color;

	s += NB_THREAD;
	}

    // must be present !
    delete mandelbrotMath;
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
