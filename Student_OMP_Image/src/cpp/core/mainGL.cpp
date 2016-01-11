#include <iostream>

#include "GLUTImageViewers.h"
#include "Settings.h"

#include "ViewerZoomable.h"
#include "Viewer.h"

#include "RipplingProvider.h"
#include "MandelbrotFactory.h"
#include "JuliaFactory.h"
#include "NewtonFactory.h"
#include "HeatTransfertProviderAdvanced.h"

using std::cout;
using std::endl;

int mainGL(Settings& settings);

int mainGL(Settings& settings)
    {
    cout << "\n[OpenGL] mode" << endl;

    GLUTImageViewers::init(settings.getArgc(), settings.getArgv());

    //Viewer<RipplingProvider> rippling(10, 10);
    //Viewer<HeatTransfertProviderAdvanced> rippling(10, 10);

    //ViewerZoomable<MandelbrotFactory> mandelbrot(10, 10);
    //ViewerZoomable<JuliaFactory> julia(820, 10);
    //ViewerZoomable<NewtonFactory> newton(10, 10);

    GLUTImageViewers::runALL();  // Bloquant, Tant qu'une fenetre est ouverte

    return EXIT_SUCCESS;
    }
