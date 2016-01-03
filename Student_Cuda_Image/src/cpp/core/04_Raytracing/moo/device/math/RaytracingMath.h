#ifndef RAYTRACING_MATH_H_
#define RAYTRACING_MATH_H_
#include "math.h"
#include "Sphere.h"
#include "ColorTools.h"



/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

/**
 * Dans un header only pour preparer la version cuda
 */
class RaytracingMath
    {

	/*--------------------------------------*\
	|*		Constructeur		*|
	 \*-------------------------------------*/

    public:

	__device__ RaytracingMath(unsigned int w, unsigned int h)
	    {
	    this->dim2 = w / 2;
	    }
	__device__
	 virtual ~RaytracingMath(void)
	    {
	    //rien
	    }

	/*--------------------------------------*\
	|*		Methode			*|
	 \*-------------------------------------*/

    public:

	/**
	 * ptrColor represente le pixel (i,j) de l'image. uchar pour 4 cannaux color (r,g,b,alpha) chacun dans [0,255]
	 */
	__device__
	void colorIJ(uchar4* ptrColorIJ, int i, int j, float t, Sphere *ptrSpheres, int spheresCount)
	    {



			xySol.x = i;
			xySol.y = j;
			float hue = -1; // Pour faire du noir par d������faut

			// On itere le tableau de sphere pour savoir si il y en a par dessus le pixel
			for(int i = 0; i<spheresCount; i++){ // on a mis 2 en attendant...
				hCarre = ptrSpheres[i].hCarre(xySol);
				if(ptrSpheres[i].isEnDessous(hCarre)){
					dz = ptrSpheres[i].dz(hCarre);
					ptrSpheres[i].distance(dz);
					brightness = ptrSpheres[i].brightness(dz);
					//hue = ptrSpheres[i].c;
					//hue = ptrSpheres[i].getHueStart();
					//ptrSpheres[i].setT();
					hue = ptrSpheres[i].hue(t);
					//hue = fmodf(ptrSpheres[i].getHueStart()+t,1.0f);


				}
			}


			// Si il n'y en a pas la hue reste ������ sa valeur initiale et on colore en noir
			if(hue == -1){
				ptrColorIJ->x = 0;
				ptrColorIJ->y = 0;
				ptrColorIJ->z = 0;
			}
			else
			    {
			    ColorTools::HSB_TO_RVB(hue,1,brightness, ptrColorIJ);
			    }



		    ptrColorIJ->w = 255; // opaque
	    }

    private:
	/*--------------------------------------*\
	|*		Attribut		*|
	 \*-------------------------------------*/

	// Tools
	double dim2; //=dim/2
	float2 xySol;
	float hCarre;
	float dz;
	float brightness;

    };

#endif

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
