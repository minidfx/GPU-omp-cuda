#ifndef MANDELBROT_MOO_H_
#define MANDELBROT_MOO_H_

#include "cudaType.h"
#include "AnimableFonctionel_I.h"
#include "MathTools.h"
#include "VariateurI.h"
#include "cudaTools.h"

#include "FractalMathBase.h"

// ----------------------------------------------------------------------
// Class responsible for generating a graphic through the Mandelbrot way.
// -----------------------------------------------------------------------
class GraphicGenerator : public AnimableFonctionel_I
{
    // ----------------------------
    // Public members
    // ----------------------------
    public:

        // -------------------------
        // Constructor & Destructor
        // -------------------------
        GraphicGenerator(unsigned int width,
                         unsigned int height,
                         unsigned int nMin,
                         unsigned int nMax,
                         DomaineMath *ptrMathDomain,
                         bool isMandelbrot,
                         string title);

        // ----------------------------
        // Virtual members
        // ----------------------------

        // --------------
        // Deconstructor
        // --------------
        virtual ~GraphicGenerator();

        // ----------------------------
        // Call periodicaly by the api
        // ----------------------------
        virtual void process(uchar4* ptrTabPixels, int width, int height, const DomaineMath& mathDomain);

        // ----------------------------
        // Call periodicaly by the api
        // ----------------------------
        virtual void animationStep();

        // ------------------------------------------------------
        // Gets the value to identify the state of the animation.
        // ------------------------------------------------------
    	virtual float getAnimationPara();

        // ------------------------------
        // Gets the width of the graphic
        // ------------------------------
    	virtual int getW();

        // -------------------------------
        // Gets the height of the graphic
        // -------------------------------
    	virtual int getH();

        // ------------------------------
        // Gets the title of the graphic
        // ------------------------------
    	virtual string getTitle();

        // -----------------------------------------------------
        // Gets the domain math used for generating the graphic
        // -----------------------------------------------------
    	virtual DomaineMath* getDomaineMathInit();

    // ----------------------------
    // Private members
    // ----------------------------
    private:

        dim3 dg;
        dim3 db;

        unsigned int max;

    // ----------------------------
    // Protected members
    // ----------------------------
    protected:

        // -------------------------
        // The widht of the graphic.
        // -------------------------
    	unsigned int width;

        // --------------------------
        // The height of the graphic.
        // --------------------------
    	unsigned int height;

        // -----------------------------------
        // Class representing the boundary of the graphic.
        // -----------------------------------
    	DomaineMath *ptrMathDomain;

        bool isMandelbrot;

    	// ----------------------------------------------
        // Members representing the title of the graphic
        // ----------------------------------------------
    	string title;

        // -------------------------------------------------------------
        // The variateur responsible for changing the fractal parameter.
        // -------------------------------------------------------------
    	VariateurI variateurN;
};

#endif
