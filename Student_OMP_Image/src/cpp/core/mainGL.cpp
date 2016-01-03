#include <iostream>

#include "GLUTImageViewers.h"
#include "Settings.h"

#include "ViewerZoomable.h"
#include "Viewer.h"

#include "RipplingProvider.h"
#include "MandelbrotFactory.h"
#include "JuliaFactory.h"
#include "NewtonFactory.h"

using std::cout;
using std::endl;

int mainGL(Settings& settings);

int mainGL(Settings& settings) {
  cout << "\n[OpenGL] mode" << endl;

  GLUTImageViewers::init(settings.getArgc(), settings.getArgv());

  //Viewer<RipplingProvider> rippling(0, 0);
  //ViewerZoomable<MandelbrotFactory> mandelbrot(640, 0);
  //ViewerZoomable<JuliaFactory> julia(0, 640);
  ViewerZoomable<NewtonFactory> newton(0, 640);

  GLUTImageViewers::runALL();  // Bloquant, Tant qu'une fenetre est ouverte

  return EXIT_SUCCESS;
}
