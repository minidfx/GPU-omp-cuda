#include "NewtonProvider.h"
#include "MathTools.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Imported	 	*|
 \*-------------------------------------*/

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

/*-----------------*\
 |*	static	   *|
 \*----------------*/

Newton* NewtonProvider::create()
    {
    int dw = 800;
    int dh = 500;

    float dt = 2 * PI / 8000;

    int nMin = 1;
    int nMax = 50;

    return new Newton(dw, dh, nMin, nMax, "CUDA Newton (zoom enabled)");
    }

ImageFonctionel* NewtonProvider::createGL(void)
    {
    ColorRGB_01* ptrColorTitre = new ColorRGB_01(0, 0, 0);

    return new ImageFonctionel(create(), ptrColorTitre);
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
