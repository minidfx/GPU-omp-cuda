#include "IndiceTools.h"
#include "HeatTransfertProviderAdvanced.h"
#include "HeatTransfertAdvanced.h"
#include "HeatTransfertDeviceAdvanced.h"

/**
 * Creates a window OpenGL containing the animation.
 */
Image* HeatTransfertProviderAdvanced::createGL()
{
  Animable_I* ptrAnimable = HeatTransfertProviderAdvanced::createMOO();
  return new Image(ptrAnimable);
}

/**
 * Creates the model responsible for calculating color of any pixels
 * displayed in the window OpenGL.
 */
Animable_I* HeatTransfertProviderAdvanced::createMOO()
{
  unsigned int width = 512;
  unsigned int height = 512;

  return new HeatTransfertAdvanced(width, height, 0.25, "CUDA HeatTransfert (new)", ComputeMode2, false);
}
