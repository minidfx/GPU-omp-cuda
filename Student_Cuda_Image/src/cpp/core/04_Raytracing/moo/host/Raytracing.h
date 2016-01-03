#ifndef RAYTRACING_H_
#define RAYTRACING_H_

#include "cudaTools.h"
#include "Animable_I.h"
#include "MathTools.h"
#include "Sphere.h"
#include "VariateurF.h"

using cpu::IntervalF;
/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

class Raytracing: public Animable_I
    {
	/*--------------------------------------*\
	|*		Constructor		*|
	 \*-------------------------------------*/

    public:

	//Raytracing(int w, int h, float dt);
	Raytracing(int w, int h, IntervalF dt);
	virtual ~Raytracing(void);

	/*--------------------------------------*\
	 |*		Methodes		*|
	 \*-------------------------------------*/

    public:

	/*-------------------------*\
	|*   Override Animable_I   *|
	 \*------------------------*/

	virtual int RandomInRange(int x, int y);
	virtual float RandomFloat(float a, float b);
	/**
	 * Call periodicly by the api
	 */
	virtual void process(uchar4* ptrDevPixels, int w, int h);
	/**
	 * Call periodicly by the api
	 */
	virtual void animationStep();

	virtual float getAnimationPara();
	virtual string getTitle();
	virtual int getW();
	virtual int getH();



	/*--------------------------------------*\
	 |*		Attributs		*|
	 \*-------------------------------------*/

    private:

	// Inputs
	int w;
	int h;


	// Tools
	dim3 dg;
	dim3 db;
	float t;

	//Outputs
	string title;
	Sphere *ptrSpheres;
	Sphere *ptrDevSpheres;
	int spheresCount = 1535;

	VariateurF variateurF;


//SM
	size_t size;



    };

#endif

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
