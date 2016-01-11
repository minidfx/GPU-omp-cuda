#include "OmpTools.h"
#include "IndiceTools.h"
#include "ColorTools.h"
#include "CalibreurF.h"

void diffuse(float* ptrTabImageInput, float* ptrTabImageOutput, unsigned int width, unsigned int height, float propagationSpeed);
void crush(float* ptrTabImageHeater, float* ptrTabImage, unsigned int arraySize, unsigned int s);
void display(float* ptrTabImage, uchar4* ptrTabPixels, unsigned int arraySize, unsigned int s);

void diffuseEntrelacement(float* ptrTabImageInput, float* ptrTabImageOutput, unsigned int width, unsigned int height, float propagationSpeed);
void crushEntrelacement(float* ptrTabImageHeater, float* ptrTabImage, unsigned int arraySize);
void displayEntrelacement(float* ptrTabImage, uchar4* ptrDevPixels, unsigned int arraySize);

void diffuseAuto(float* ptrTabImageInput, float* ptrTabImageOutput, unsigned int width, unsigned int height, float propagationSpeed);
void crushAuto(float* ptrTabImageHeater, float* ptrTabImage, unsigned int arraySize);
void displayAuto(float* ptrTabImage, uchar4* ptrDevPixels, unsigned int arraySize);

float computeHeat1(float oldHeat, float* neighborPixels, float propagationSpeed);
float computeHeat2(float oldHeat, float* neighborPixels, float propagationSpeed);
float computeHeat3(float oldHeat, float* neighborPixels, float propagationSpeed);

void diffuse(float* ptrTabImageInput, float* ptrTabImageOutput, unsigned int width, unsigned int height, float propagationSpeed, unsigned int i, unsigned int j, unsigned int s)
{
    if (i > 0 && i < (height - 1) && j > 0 && j < (width - 1))
    {
        float neighborPixels[4];
        neighborPixels[0] = ptrTabImageInput[IndiceTools::toS(width, i - 1, j)];
        neighborPixels[1] = ptrTabImageInput[IndiceTools::toS(width, i + 1, j)];
        neighborPixels[2] = ptrTabImageInput[IndiceTools::toS(width, i, j - 1)];
        neighborPixels[3] = ptrTabImageInput[IndiceTools::toS(width, i, j + 1)];

        ptrTabImageOutput[s] = computeHeat3(ptrTabImageInput[s], neighborPixels, propagationSpeed);
    }
    else
    {
        ptrTabImageOutput[s] = ptrTabImageInput[s];
    }
}

void diffuseAuto(float* ptrTabImageInput, float* ptrTabImageOutput, unsigned int width, unsigned int height, float propagationSpeed)
{
    #pragma omp parallel for
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            int s = IndiceTools::toS(width, i, j);
            diffuse(ptrTabImageInput, ptrTabImageOutput, width, height, propagationSpeed, i, j, s);
        }
    }
}

void diffuseEntrelacement(float* ptrTabImageInput, float* ptrTabImageOutput, unsigned int width, unsigned int height, float propagationSpeed)
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

            diffuse(ptrTabImageInput, ptrTabImageOutput, width, height, propagationSpeed, i, j, s);
            s += NB_THREADS;
        }
    }
}

void crush(float* ptrTabImageHeater, float* ptrTabImage, unsigned int arraySize, unsigned int s)
{
    if (ptrTabImageHeater[s] > 0.0)
    {
        ptrTabImage[s] = ptrTabImageHeater[s];
    }
}

void crushAuto(float* ptrTabImageHeater, float* ptrTabImage, unsigned int arraySize)
{
    #pragma omp parallel for
    for (unsigned int i = 0; i < arraySize; i++)
    {
        crush(ptrTabImageHeater, ptrTabImage, arraySize, i);
    }
}

void crushEntrelacement(float* ptrTabImageHeater, float* ptrTabImage, unsigned int arraySize)
{
    #pragma omp parallel
    {
        const int NB_THREADS = OmpTools::setAndGetNaturalGranularity();
        const int TID = OmpTools::getTid();

        unsigned int s = TID;
        while (s < arraySize)
        {
            crush(ptrTabImageHeater, ptrTabImage, arraySize, s);
            s += NB_THREADS;
        }
    }
}

void display(float* ptrTabImage, uchar4* ptrTabPixels, unsigned int arraySize, unsigned int s)
{
    float heatMax = 1.0;
    float heatMin = 0.;
    float hueMax = 0.;
    float hueMin = 0.7;

    CalibreurF calibreur(IntervalF(heatMin, heatMax), IntervalF(hueMin, hueMax));

    float hue = ptrTabImage[s];
    calibreur.calibrer(hue);
    uchar4 p;
    ColorTools::HSB_TO_RVB(hue, 1, 1, &p.x, &p.y, &p.z);
    ptrTabPixels[s] = p;
}

void displayAuto(float* ptrTabImage, uchar4* ptrDevPixels, unsigned int arraySize)
{
    #pragma omp parallel for
    for (unsigned int i = 0; i < arraySize; i++)
    {
        display(ptrTabImage, ptrDevPixels, arraySize, i);
    }
}

void displayEntrelacement(float* ptrTabImage, uchar4* ptrTabPixels, unsigned int arraySize)
{
    #pragma omp parallel
    {
        const int NB_THREADS = OmpTools::setAndGetNaturalGranularity();
        const int TID = OmpTools::getTid();
        unsigned int s = TID;

        while (s < arraySize)
        {
            display(ptrTabImage, ptrTabPixels, arraySize, s);
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
