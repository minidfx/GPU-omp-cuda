#ifndef MANDELBROT_PROVIDER_H_
#define MANDELBROT_PROVIDER_H_

#include "ImageFonctionel.h"
#include "AnimableFonctionel_I.h"
#include "VariateurI.h"

// ----------------------------------------------------
// Class for creating the Mandelbrot graphic generator
// ----------------------------------------------------
class MandelbrotFactory
{
    public:

        // ------------------
        // Creates the Image
        // ------------------
        static ImageFonctionel* createGL();

        // ------------------------------------------------------------------------
        // Creates the graphic generactor that will manage the content of the Image
        // ------------------------------------------------------------------------
        static AnimableFonctionel_I* createGenerator();
};

#endif
