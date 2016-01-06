#ifndef HEAT_TRANSFERT_MOO_H_
#define HEAT_TRANSFERT_MOO_H_

#include "cudaType.h"
#include "Animable_I.h"

class HeatTransfert: public Animable_I
    {

    public:
	HeatTransfert(unsigned int w, unsigned int h, float* ptrImageInit, float* ptrImageHeater, float propSpeed);
	virtual ~HeatTransfert();

	virtual void process(uchar4* ptrDevPixels, int w, int h);
	virtual void animationStep();

	virtual float getAnimationPara();
	virtual int getW();
	virtual int getH();
	virtual string getTitle();

    private:

	// Inputs
	unsigned int w;
	unsigned int h;
	unsigned int wh;

	// Images
	float* ptrDevImageInit;
	float* ptrDevImageHeater;
	float* ptrDevImageA;
	float* ptrDevImageB;
	float propSpeed;

	// Tools
	unsigned int iteration;
	dim3 dg;
	dim3 db;
    };

#endif
