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
//#include "RaytracingProvider.h"
#include "MandelbrotProviderMGPU.h"
//#include "HeatTransfertProvider.h"

using std::cout;
using std::endl;
using std::string;

int mainGL(Option& option);

int mainGL(Option& option) {
	cout << "\n[OpenGL] mode" << endl;

	GLUTImageViewers::init(option.getArgc(), option.getArgv());

	//Viewer<RipplingProvider> rippling(0, 0);
	//ViewerZoomable<MandelbrotFactory> mandelbrot(0, 0);
	//ViewerZoomable<JuliaFactory> julia(800, 0);
	//ViewerZoomable<RaytracingProvider> raytracing(0, 0);
	ViewerZoomable<MandelbrotProviderMGPU> mandelbrotMGPU(0, 0);
	//Viewer<HeatTransfertProvider> heatTransfert(0, 0);

	GLUTImageViewers::runALL(); // Bloquant, Tant qu'une fenetre est ouverte

	return EXIT_SUCCESS;
}
