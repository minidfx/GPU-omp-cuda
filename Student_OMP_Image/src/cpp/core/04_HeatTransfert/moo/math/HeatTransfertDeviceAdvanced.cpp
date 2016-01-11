#include "OmpTools.h"
#include "IndiceTools.h"
#include "ColorTools.h"
#include "CalibreurF.h"

void diffuseAdvanced(float* ptrDevImageInput, float* ptrDevImageOutput, unsigned int width, unsigned int height, float propagationSpeed);
void crushAdvanced(float* ptrDevImageHeater, float* ptrDevImage, unsigned int arraySize);
void displayAdvanced(float* ptrDevImage, uchar4* ptrDevPixels, unsigned int arraySize);

float computeHeat1(float oldHeat, float* neighborPixels, float propagationSpeed);
float computeHeat2(float oldHeat, float* neighborPixels, float propagationSpeed);
float computeHeat3(float oldHeat, float* neighborPixels, float propagationSpeed);

void diffuseAdvanced(float* ptrDevImageInput, float* ptrDevImageOutput, unsigned int width, unsigned int height, float propagationSpeed)
{
    #pragma omp parallel
    {
        // Calucul threads available
        const int NB_THREADS = OmpTools::setAndGetNaturalGranularity();
        const int TID = OmpTools::getTid();

        // Init service and variable required
        unsigned int totalPixels = width * height;
        unsigned int s = TID;

        while (s < totalPixels)
        {
            int i, j;
            IndiceTools::toIJ(s, width, &i, &j);

            if (i > 0 && i < (height - 1) && j > 0 && j < (width - 1))
            {
                float neighborPixels[4];
                neighborPixels[0] = ptrDevImageInput[IndiceTools::toS(width, i - 1, j)];
                neighborPixels[1] = ptrDevImageInput[IndiceTools::toS(width, i + 1, j)];
                neighborPixels[2] = ptrDevImageInput[IndiceTools::toS(width, i, j - 1)];
                neighborPixels[3] = ptrDevImageInput[IndiceTools::toS(width, i, j + 1)];

                ptrDevImageOutput[s] = computeHeat3(ptrDevImageInput[s], neighborPixels, propagationSpeed);
            }
            else
            {
                ptrDevImageOutput[s] = ptrDevImageInput[s];
            }

            s += NB_THREADS;
        }
    }
}

void crushAdvanced(float* ptrDevImageHeater, float* ptrDevImage, unsigned int arraySize)
{
    #pragma omp parallel
    {
        const int NB_THREADS = OmpTools::setAndGetNaturalGranularity();
        const int TID = OmpTools::getTid();

        unsigned int s = TID;
        while (s < arraySize)
        {
            if (ptrDevImageHeater[s] > 0.0)
            {
                ptrDevImage[s] = ptrDevImageHeater[s];
            }

            s += NB_THREADS;
        }
    }
}

void displayAdvanced(float* ptrDevImage, uchar4* ptrDevPixels, unsigned int arraySize)
{
    #pragma omp parallel
    {
        const int NB_THREADS = OmpTools::setAndGetNaturalGranularity();
        const int TID = OmpTools::getTid();
        unsigned int s = TID;

        float heatMax = 1.0;
        float heatMin = 0.;
        float hueMax = 0.;
        float hueMin = 0.7;

        CalibreurF calibreur(IntervalF(heatMin, heatMax), IntervalF(hueMin, hueMax));

        while (s < arraySize)
        {
            float hue = ptrDevImage[s];
            calibreur.calibrer(hue);
            uchar4 p;
            ColorTools::HSB_TO_RVB(hue, 1, 1, &p.x, &p.y, &p.z);
            ptrDevPixels[s] = p;

            s += NB_THREADS;
        }
    }
}

float computeHeat3(float oldHeat, float* neighborPixels, float propagationSpeed)
{
  return oldHeat + propagationSpeed * (neighborPixels[0] + neighborPixels[1] + neighborPixels[2] + neighborPixels[3] - 4 * oldHeat);
}

float computeHeat2(float oldHeat, float* neighborPixels, float propagationSpeed)
{
  return neighborPixels[0] + neighborPixels[1] + neighborPixels[2] + neighborPixels[3] + 4 * oldHeat / 2 * 4;
}

float computeHeat1(float oldHeat, float* neighborPixels, float propagationSpeed)
{
  // Initialize the Heat with the previous one.
  float newHeat = oldHeat;

  for (int i = 0; i < 4; i++)
  {
    newHeat += propagationSpeed * (neighborPixels[i] - oldHeat);
  }

  return newHeat;
}
