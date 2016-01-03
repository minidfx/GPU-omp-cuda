#include <iostream>
#include <stdlib.h>
#include <string.h>

#include "GLUTImageViewers.h"

#include "Device.h"
#include "cudaTools.h"

#include "RipplingProvider.h"
#include "MandelbrotProvider.h"
#include "MandelbrotProviderMGPU.h"
#include "NewtonProvider.h"
#include "RayTracingProvider.h"
#include "HeatTransfertProvider.h"

using std::cout;
using std::endl;
using std::string;

int mainGL(void);

int mainGL(void)
    {
    cout << "\n[OpenGL] mode" << endl;

    Image* ptrRippling = RipplingProvider::createGL();
    ImageFonctionel* ptrMandelbrot = MandelbrotFactory::createGL();
    ImageFonctionel* ptrJulia = JuliaFactory::createGL();
    ImageFonctionel* ptrMandelbrotMGPU = MandelbrotProviderMGPU::createGL();
    ImageFonctionel* ptrNewton = NewtonProvider::createGL();
    Image* ptrRayTracing = RayTracingProvider::createGL();
    Image* ptrHeatTransfert = HeatTransfertProvider::createGL();

    //ImageViewer : (boolean,boolean) : (isAnimation,isSelectionEnable)
    GLUTImageViewers ripplingViewer(ptrRippling, true, false, 0, 0);
    GLUTImageViewers mandelbrotViewer(ptrMandelbrot, true, true, 0, 400);
    GLUTImageViewers mandelbrotMGPUViewer(ptrMandelbrotMGPU, true, true, 400, 0);
    GLUTImageViewers newtonViewer(ptrNewton,true, true, 400, 400);
    GLUTImageViewers rayTracingViewer(ptrRayTracing, true, false, 800, 0);
    GLUTImageViewers heatTransfertViewer(ptrHeatTransfert, true, false, 800, 400);
    GLUTImageViewers juliaViewer(ptrJulia, true, true, 800, 800);

    GLUTImageViewers::runALL(); // Bloquant, Tant qu'une fenetre est ouverte

    // destruction
	{
	delete ptrRippling;
	delete ptrMandelbrot;
	delete ptrJulia;
	delete ptrMandelbrotMGPU;
	delete ptrNewton;
	delete ptrRayTracing;
	delete ptrHeatTransfert;

	ptrRippling = NULL;
	ptrMandelbrot = NULL;
	ptrJulia = NULL;
	ptrMandelbrotMGPU = NULL;
	ptrNewton = NULL;
	ptrRayTracing = NULL;
	ptrHeatTransfert = NULL;
	}

    return EXIT_SUCCESS;
    }
