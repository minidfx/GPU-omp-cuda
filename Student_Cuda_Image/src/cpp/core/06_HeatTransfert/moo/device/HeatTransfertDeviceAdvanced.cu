#include "cudaType.h"

#include "Indice2D.h"
#include "IndiceTools.h"
#include "ColorTools.h"

__global__ void diffuseAdvanced(float* ptrDevImageInput, float* ptrDevImageOutput, unsigned int width, unsigned int height, float propagationSpeed);
__global__ void crushAdvanced(float* ptrDevImageHeater, float* ptrDevImage, unsigned int arraySize);
__global__ void displayAdvanced(float* ptrDevImage, uchar4* ptrDevPixels, unsigned int arraySize);

__device__ float computeHeat1(float oldHeat, float* neighborsHeat, unsigned int nbNeighbors, float propagationSpeed);
__device__ float computeHeat2(float oldHeat, float* neighborsHeat, unsigned int countNeighbors, float propagationSpeed);

__global__ void diffuseAdvanced(float* ptrDevImageInput, float* ptrDevImageOutput, unsigned int width, unsigned int height, float propagationSpeed)
{
  // Calucul threads available
  const int NB_THREADS = Indice2D::nbThread();
  const int TID = Indice2D::tid();

  // Init service and variable required
  unsigned int totalPixels = width * height;
  unsigned int s = TID;

  while (s < totalPixels)
  {
    int i, j;
    IndiceTools::toIJ(s, width, &i, &j);

    if (i > 0 && i < (height - 1) && j > 0 && j < (width - 1))
    {
      float neighborsHeat[4];
      neighborsHeat[0] = ptrDevImageInput[IndiceTools::toS(width, i - 1, j)];
      neighborsHeat[1] = ptrDevImageInput[IndiceTools::toS(width, i + 1, j)];
      neighborsHeat[2] = ptrDevImageInput[IndiceTools::toS(width, i, j - 1)];
      neighborsHeat[3] = ptrDevImageInput[IndiceTools::toS(width, i, j + 1)];

      ptrDevImageOutput[s] = computeHeat1(ptrDevImageInput[s], neighborsHeat, 4, propagationSpeed);
    }
    else
    {
      ptrDevImageOutput[s] = ptrDevImageInput[s];
    }

    s += NB_THREADS;
  }
}

__global__ void crushAdvanced(float* ptrDevImageHeater, float* ptrDevImage, unsigned int arraySize)
{
  const int NB_THREADS = Indice2D::nbThread();
  const int TID = Indice2D::tid();

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

__global__ void displayAdvanced(float* ptrDevImage, uchar4* ptrDevPixels, unsigned int arraySize)
{
  const int NB_THREADS = Indice2D::nbThread();
  const int TID = Indice2D::tid();

  unsigned int s = TID;
  while (s < arraySize)
  {
    float hue = 0.7 - ptrDevImage[s] * 0.7;
    ColorTools::HSB_TO_RVB(hue, 1, 1, &ptrDevPixels[s]);
    ptrDevPixels[s].w = 255;

    s += NB_THREADS;
  }
}

__device__ float computeHeat2(float oldHeat, float* neighborsHeat, unsigned int countNeighbors, float propagationSpeed)
{
  // Not implemented
}

__device__ float computeHeat1(float oldHeat, float* neighborsHeat, unsigned int countNeighbors, float propagationSpeed)
{
  // Initialize the Heat with the previous one.
  float newHeat = oldHeat;

  for (int i = 0; i < countNeighbors; i++)
  {
    newHeat += propagationSpeed * (neighborsHeat[i] - oldHeat);
  }

  return newHeat;
}
