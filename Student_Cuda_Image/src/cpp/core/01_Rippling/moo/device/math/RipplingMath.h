#ifndef RIPPLING_MATH_H_
#define RIPPLING_MATH_H_

#include "MathTools.h"

class RipplingMath
{
    public:

        __device__
        RipplingMath(unsigned int w, unsigned int h)
        {
            this->dim2 = w / 2;
        }

        __device__
        void colorIJ(uchar4* ptrColor, int i, int j, float t)
        {
            uchar levelGris;
            f(i, j, t, &levelGris);

            ptrColor->x = levelGris;
            ptrColor->y = levelGris;
            ptrColor->z = levelGris;
        }

    private:

        __device__
    	void f(int i, int j, float t, uchar* ptrLevelGris)
	    {
    	    float dijResult;
    	    dij(i,j, &dijResult);

    	    *ptrLevelGris = 128 + 127 * ((cos((dijResult/(10.0))-(t/7.0))) / ((dijResult/10.0)+1));
	    }

        __device__
        void dij(int i, int j, float* ptrResult) // par exmple
        {
            float fi = i - this->dim2;
    	    float fj = j - this->dim2;

    	    *ptrResult = sqrt(fi*fi + fj*fj);
        }

        double dim2; //=dim/2
};

#endif
