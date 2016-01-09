#ifndef HEAT_TRANSFERT_PROVIDER_ADVANCED_H_
#define HEAT_TRANSFERT_PROVIDER_ADVANCED_H_

#include "Image.h"
#include "Animable_I.h"

/**
 * Service responsible for providing a model and an
 * image displaying the heat.
 */
class HeatTransfertProvider
{
  public:

    /**
     * Creates a window OpenGL containing the animation.
     */
  	static Image* createGL();

    /**
     * Creates the model responsible for calculating color of any pixels
     * displayed in the window OpenGL.
     */
  	static Animable_I* createMOO();
};

#endif
