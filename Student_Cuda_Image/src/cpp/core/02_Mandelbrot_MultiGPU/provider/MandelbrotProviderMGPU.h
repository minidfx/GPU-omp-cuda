#ifndef MANDELBROT_MGPU_PROVIDER_H_
#define MANDELBROT_MGPU_PROVIDER_H_

#include "../../02_Mandelbrot_MultiGPU/moo/host/MandelbrotMGPU.h"
#include "ImageFonctionel.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

class MandelbrotProviderMGPU
    {
    public:

	static MandelbrotMGPU* create(void);
	static ImageFonctionel* createGL(void);
    };

#endif

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
