#ifndef NEWTON_MOO_H_
#define NEWTON_MOO_H_

#include "cudaType.h"
#include "AnimableFonctionel_I.h"
#include "MathTools.h"
#include "VariateurI.h"

#include "NewtonMath.h"

// ----------------------------------------------------------------------
// Class responsible for generating a graphic through the Newton way.
// -----------------------------------------------------------------------
class NewtonGraphicGenerator : public AnimableFonctionel_I
{
    // ----------------------------
    // Public members
    // ----------------------------
    public:

        // -------------------------
        // Constructor & Destructor
        // -------------------------
        NewtonGraphicGenerator(unsigned int width,
                         unsigned int height,
                         unsigned int max,
                         DomaineMath *ptrMathDomain,
                         NewtonMath *ptrNewtonMath,
                         string title);

        // --------------
        // Deconstructor
        // --------------
        virtual ~NewtonGraphicGenerator();

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

        // --------------------------------------------------
        // Sets the pattern that the implementation will use.
        // --------------------------------------------------
    	virtual void setParallelPatern(ParallelPatern parallelPatern);

    // ----------------------------
    // Private members
    // ----------------------------
    private:

        // -------------------------------------------------------------
        // The method, which executes a loop for as parralel using OMP.
        // -------------------------------------------------------------
    	void forAutoOMP(uchar4* ptrTabPixels, int width, int height, const DomaineMath& mathDomain);

        // -------------------------------------------------------------
        // The method, which execute the default entrelacement pattern.
        // -------------------------------------------------------------
    	void entrelacementOMP(uchar4* ptrTabPixels, int width, int height, const DomaineMath& mathDomain);

        // ---------------------------------------------------------------------------
        // The method, which will work with the color of the pixel passed as argument.
        // ----------------------------------------------------------------------------
    	void workPixel(uchar4* ptrColorIJ,
                       int i,
                       int j,
                       const DomaineMath& mathDomain);

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

        // ----------------------------
        // Contains the math operations
        // ----------------------------
        NewtonMath *ptrNewtonMath;

    	// ----------------------------------------------
        // Members representing the title of the graphic
        // ----------------------------------------------
    	string title;

        // -----------------------------------
        // The current index of the animation
        // -----------------------------------
        unsigned int iteration;

        // -----------------------------------------------------
        // The pattern, which will use by the graphic generator.
        // -----------------------------------------------------
    	ParallelPatern parallelPattern;

        // -------------------------------------------------------------
        // The variateur responsible for changing the fractal parameter.
        // -------------------------------------------------------------
        VariateurI variateurN;

        // ---------------------------------------------------------------
        // Numbers of thread that will be used for generating the graphic
        // ---------------------------------------------------------------
        int nbThread;
};

#endif
