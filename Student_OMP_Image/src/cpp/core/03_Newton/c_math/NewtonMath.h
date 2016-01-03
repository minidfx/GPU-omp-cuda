#ifndef NEWTON_MATH_H_
#define NEWTON_MATH_H_

#include <math.h>
#include <cfloat>
#include "ColorTools.h"

class NewtonMath
{
  public:

      NewtonMath(double epsilon)
      {
          this->epsilon = epsilon;
          this->colorFactor = 25;

          XA1 = 1;
          XA2 = 0;

          XB1 = -0.5f;
          XB2 = sqrtf(3) / 2.0f;

          XC1 = -0.5f;
          XC2 = -sqrtf(3) / 2.0f;
      }

      virtual ~NewtonMath()
      {
      }

      void colorXY(uchar4 *ptrColor, float x1, float x2, int n)
      {
          int i = 0;
          bool converged = false;

          float x1Copy = x1;
          float x2Copy = x2;

          do
          {
              this->computeNextStep(x1Copy, x2Copy, &x1, &x2);
              x1Copy = x1;
              x2Copy = x2;
              converged = this->hasConverged(x1, x2);
          }
          while (i++ <= n && !converged);

          if (converged)
          {
              float distSolA = getNorme(x1 - XA1, x2 - XA2);
              float distSolB = getNorme(x1 - XB1, x2 - XB2);
              float distSolC = getNorme(x1 - XC1, x2 - XC2);

              float colorEasing = ((i * this->colorFactor) % 255);

              if (distSolA < distSolB && distSolA < distSolC)
              {
                  // solA est la plus proche
                  ptrColor->x = 255 - colorEasing;
                  ptrColor->y = 0;
                  ptrColor->z = 0;
              }
              else if (distSolB < distSolC)
              {
                  // solB est la plus proche
                  ptrColor->x = 0;
                  ptrColor->y = 255 - colorEasing;
                  ptrColor->z = 0;
              }
              else
              {
                  //solC est la plus proche
                  ptrColor->x = 0;
                  ptrColor->y = 0;
                  ptrColor->z = 255 - colorEasing;
              }
          }
          else
          {
              ptrColor->x = 0;
              ptrColor->y = 0;
              ptrColor->z = 0;
          }

          ptrColor->w = 255;
      }

  private:

      bool hasConverged(float x1, float x2)
      {
          return this->isSolutionConverged(x1, x2, XA1, XA2) ||
                 this->isSolutionConverged(x1, x2, XB1, XB2) ||
                 this->isSolutionConverged(x1, x2, XC1, XC2);
      }

      bool isSolutionConverged(float x1, float x2, float x1Sol, float x2Sol)
      {
          return (getNorme(x1 - x1Sol, x2 - x2Sol) / getNorme(x1Sol, x2Sol)) < this->epsilon;
      }

      float getNorme(float x, float y)
      {
          return sqrtf(x * x + y * y);
      }

      float f1(float x, float y)
      {
          return x * x * x - 3.0f * x * y * y - 1.0f;
      }

      float f2(float x, float y)
      {
          return y * y * y - 3.0f * x * x * y;
      }

      float d1f1(float x, float y)
      {
          return 3.0f * (x * x - y * y);
      }

      float d2f1(float x, float y)
      {
          return -6.0f * x * y;
      }

      float d1f2(float x, float y)
      {
          return d2f1(x, y);
      }

      float d2f2(float x, float y)
      {
          return 3.0f * (y * y - x * x);
      }

      void computeNextStep(float x1, float x2, float *outX1, float *outX2)
      {
          float a = this->d1f1(x1, x2);
          float b = this->d1f2(x1, x2);
          float c = this->d2f1(x1, x2);
          float d = this->d2f2(x1, x2);

          float inversedJacobian = 1.0f / (a * d - b * c);

          float sa = inversedJacobian * d;
          float sb = -inversedJacobian * c;
          float sc = -inversedJacobian * b;
          float sd = inversedJacobian * a;

          *outX1 = x1 - (sa * this->f1(x1, x2) + sc * this->f2(x1, x2));
          *outX2 = x2 - (sb * this->f1(x1, x2) + sd * this->f2(x1, x2));
      }

      // Inputs
      float epsilon;

      // Tools
      int colorFactor;

      float XA1;
      float XA2;

      float XB1;
      float XB2;

      float XC1;
      float XC2;
};

#endif
