#ifndef RIPPLING_MATH_H_
#define RIPPLING_MATH_H_


#include <math.h>

/**
 * Dans un header only pour preparer la version cuda
 */
class RipplingMath {

  public:
    RipplingMath(unsigned int w, unsigned int h) {
      this->dim2 = w / 2;
    }

    virtual ~RipplingMath(void) {
      //rien
    }

    /**
    * ptrColor represente le pixel (i,j) de l'image. uchar pour 4 cannaux color (r,g,b,alpha) chacun dans [0,255]
    */
    void colorIJ(uchar4* ptrColor, int i, int j, float t) {
      double d;

      dij(i, j, &d);

    	double color = 128 + 127 * cosf((d / 10.0) - t / 7.0) / ((d / 10.0) + 1.0);

      int icolor = (int) round(color);
      ptrColor->x = icolor;
      ptrColor->y = icolor;
      ptrColor->z = icolor;

      ptrColor->w = 255;
    }

  private:

    void dij(int i, int j, double* ptrResult) {
      *ptrResult = sqrtf(pow(i - this->dim2, 2) + pow(j - this->dim2, 2));
    }

    // Tools
    double dim2; //=dim/2
};

#endif
