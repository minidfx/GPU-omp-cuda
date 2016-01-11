#ifndef HEAT_TRANSFERT_MOO_ADVANCED_H_
#define HEAT_TRANSFERT_MOO_ADVANCED_H_

#include "Animable_I.h"
#include "VariateurI.h"
#include "SimpleMouseListener.h"

/**
 * Service responsible for calculating the color of a pixels.
 */
class HeatTransfertAdvanced: public Animable_I
{
  public:

    /**
     * Constructs the service responsible for calculating heat between pixels which will be displayed into a window OpenGL.
     */
    HeatTransfertAdvanced(unsigned int width, unsigned int height, float propagationSpeed, string title);

    /**
     * Release resources initialized in the constructor.
     */
    virtual ~HeatTransfertAdvanced();

    /**
     * Override: Loop for processing the **ptrDevPixels** passed as argument.
     */
    virtual void process(uchar4* ptrDevPixels, int width, int height);

    virtual void animationStep();

    virtual float getAnimationPara();
    virtual int getW();
    virtual int getH();
    virtual string getTitle();

    virtual void setParallelPatern(ParallelPatern parallelPatern);

  private:

    // Inputs
    unsigned int width;
    unsigned int height;
    unsigned int totalPixels;
    string title;
    ParallelPatern parallelPattern;

    // Images
    float* ptrDevImageInit;
    float* ptrDevImageHeater;
    float* ptrDevImageA;
    float* ptrDevImageB;
    float propagationSpeed;

    // Tools
    unsigned int iteration;
    VariateurI variateurN;
    int NB_ITERATION_AVEUGLE;
    bool isBufferA;
	SimpleMouseListener* ptrMouseListener;

    void listener();
};

#endif
