#include <iostream>
#include <stdlib.h>
#include <assert.h>

#include "cudaTools.h"
#include "Device.h"
#include "cudaTools.h"
#include "GLUTImageViewers.h"

using std::cout;
using std::endl;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Imported	 	*|
 \*-------------------------------------*/

extern int mainGL(void);
extern int mainFreeGL(void);

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

int main(int argc, char** argv);

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

static int start(void);
static void init(int deviceId);

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

int main(int argc, char** argv)
    {
    cout << "main" << endl;

    if (Device::isCuda())
	{
	GLUTImageViewers::init(argc, argv);

	//Device::printAll();
	Device::printAllSimple();

	// Server Cuda1: in [0,5]
	// Server Cuda2: in [0,2]
	int deviceId = 0;
	init(deviceId);
	int isOk = start();

	//cudaDeviceReset causes the driver to clean up all state.
	HANDLE_ERROR(cudaDeviceReset());

	return isOk;
	}
    else
	{
	return EXIT_FAILURE;
	}
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

void init(int deviceId)
    {
    // Check deviceId area
    int nbDevice = Device::getDeviceCount();
    assert(deviceId >= 0 && deviceId < nbDevice);

    // Choose current device  (state of host-thread)
    HANDLE_ERROR(cudaSetDevice(deviceId));

    HANDLE_ERROR(cudaGLSetGLDevice(deviceId));

    // It can be usefull to preload driver, by example to practice benchmarking! (sometimes slow under linux)
    Device::loadCudaDriver(deviceId);
    }

int start(void)
    {
    Device::printCurrent();

    bool IS_GL = true;

    if (IS_GL)
	{
	return mainGL(); // Bloquant, Tant qu'une fenetre est ouverte
	}
    else
	{
	return mainFreeGL();
	}
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

