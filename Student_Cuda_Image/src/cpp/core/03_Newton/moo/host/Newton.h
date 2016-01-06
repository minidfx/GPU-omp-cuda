#ifndef NEWTON_H_
#define NEWTON_H_

#include "cudaTools.h"
#include "AnimableFonctionel_I.h"
#include "MathTools.h"
#include "VariateurI.h"

class Newton : public AnimableFonctionel_I
{

  public:

      Newton(int w, int h, int nMin, int nMax, string title);

      virtual ~Newton(void);

  public:

      /**
       * Call periodicly by the api
       */
      virtual void process(uchar4 *ptrDevPixels, int w, int h, const DomaineMath &domaineMath);

      /**
       * Call periodicly by the api
       */
      virtual void animationStep();

      virtual float getAnimationPara();

      virtual int getW();

      virtual int getH();

      virtual string getTitle(void);

      virtual DomaineMath *getDomaineMathInit(void);

  private:

      // Inputs
      int w;
      int h;
      int n;

      // Tools
      dim3 dg;
      dim3 db;
      VariateurI variateurN; // varier n
      DomaineMath *ptrDomaineMathInit;

      //Outputs
      string title;
};

#endif
