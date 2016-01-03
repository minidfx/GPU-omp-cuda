#ifndef RAYTRACING_PROVIDER_H_
#define RAYTRACING_PROVIDER_H_

#include "Raytracing.h"
#include "Image.h"
#include "ColorTools.h"


/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

class RaytracingProvider
    {
    public:

	static Raytracing* createMOO(void);
	static Image* createGL(void);

    };

#endif

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

