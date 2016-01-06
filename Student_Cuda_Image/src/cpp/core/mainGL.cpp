#include <iostream>
#include <stdlib.h>
#include <string.h>

#include "GLUTImageViewers.h"
#include "Option.h"
#include "Viewer.h"
#include "ViewerZoomable.h"

//#include "RipplingProvider.h"
//#include "MandelbrotFactory.h"
//#include "JuliaFactory.h"
#include "RayTracingProvider.h"
//#include "MandelbrotProviderMGPU.h"
//#include "NewtonProvider.h"
//#include "HeatTransfertProvider.h"

using std::cout;
using std::endl;
using std::string;

int mainGL(Option& option);

int mainGL(Option& option)
    {
    cout << "\n[OpenGL] mode" << endl;

    GLUTImageViewers::init(option.getArgc(), option.getArgv());

    //Viewer<RipplingProvider> rippling(10,10);
    //ViewerZoomable<MandelbrotFactory> mandelbrot(10, 10);
    //ViewerZoomable<JuliaFactory> julia(820, 0);
    Viewer<RayTracingProvider> raytracing(10, 10);
    //ViewerZoomable<MandelbrotProviderMGPU> mandelbrotMGPU(10, 10);
    //ViewerZoomable<NewtonProvider> newton(10, 10);
    //Viewer<HeatTransfertProvider> heatTransfert(0, 0);

    GLUTImageViewers::runALL(); // Bloquant, Tant qu'une fenetre est ouverte

    return EXIT_SUCCESS;
    }
