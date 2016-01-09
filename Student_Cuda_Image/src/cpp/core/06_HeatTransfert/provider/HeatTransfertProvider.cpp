#include "HeatTransfertProvider.h"
#include "HeatTransfert.h"
#include "IndiceTools.h"

/**
 * Creates a window OpenGL containing the animation.
 */
Image* HeatTransfertProvider::createGL()
{
  Animable_I* ptrAnimable = HeatTransfertProvider::createMOO();
  return new Image(ptrAnimable);
}

/**
 * Creates the model responsible for calculating color of any pixels
 * displayed in the window OpenGL.
 */
Animable_I* HeatTransfertProvider::createMOO()
{
  unsigned int width = 500;
  unsigned int height = 500;
  unsigned int totalPixels = widht * height;

  float imageInit[WH];
  float imageHeater[WH];

  return new HeatTransfert(w, h, imageInit, imageHeater, 0.25, "CUDA HeatTransfert");
}
