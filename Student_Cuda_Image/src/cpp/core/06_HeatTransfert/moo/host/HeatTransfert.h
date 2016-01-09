#ifndef HEAT_TRANSFERT_MOO_H_
#define HEAT_TRANSFERT_MOO_H_

#include "cudaType.h"
#include "Animable_I.h"

class HeatTransfert: public Animable_I
{
  public:

    /**
     * Constructs the service responsible for calculating heat between pixels which will be displayed into a window OpenGL.
     */
    HeatTransfert(unsigned int width, unsigned int height, float propagationSpeed, string title);

    /**
     * Release
     */
    virtual ~HeatTransfert();

    virtual void process(uchar4* ptrDevPixels, int w, int h);
    virtual void animationStep();

    virtual float getAnimationPara();
    virtual int getW();
    virtual int getH();
    virtual string getTitle();

  private:

    // Inputs
    unsigned int width;
    unsigned int height;
    unsigned int totalPixels;
    string title;

    // Images
    float* ptrDevImageInit;
    float* ptrDevImageHeater;
    float* ptrDevImageA;
    float* ptrDevImageB;
    float propagationSpeed;

    // Tools
    unsigned int iteration;
    dim3 dg;
    dim3 db;
};

#endif
