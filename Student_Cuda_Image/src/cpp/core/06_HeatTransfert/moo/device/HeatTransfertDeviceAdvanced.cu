#include "cudaType.h"

#include "Indice2D.h"
#include "IndiceTools.h"
#include "ColorTools.h"
#include "CalibreurF.h"
#include "IntervalF_GPU.h"
#include "HeatTransfertDeviceAdvanced.h"

__global__ void diffusePerPixel(float* ptrDevImageInput,
                                float* ptrDevImageOutput,
                                unsigned int width,
                                unsigned int height,
                                float propagationSpeed,
                                ComputeMode computeMode)
{
  float (*computeMethod)(float oldHeat,
                         float* neighborPixels,
                         float propagationSpeed);

  switch(computeMode)
  {
      case ComputeMode1: computeMethod = &computeHeat1;
        break;

      case ComputeMode2: computeMethod = &computeHeat2;
        break;

      case ComputeMode3: computeMethod = &computeHeat3;
        break;
  }

  int i = threadIdx.y + blockIdx.y * blockDim.y;
  int j = threadIdx.x + blockIdx.x * blockDim.x;

  int s = j + gridDim.x * blockDim.x * (threadIdx.y + blockIdx.y * blockDim.y);

  if (i > 0 && i < (height - 1) && j > 0 && j < (width - 1))
  {
    float neighborPixels[4];
    neighborPixels[0] = ptrDevImageInput[IndiceTools::toS(width, i - 1, j)];
    neighborPixels[1] = ptrDevImageInput[IndiceTools::toS(width, i + 1, j)];
    neighborPixels[2] = ptrDevImageInput[IndiceTools::toS(width, i, j - 1)];
    neighborPixels[3] = ptrDevImageInput[IndiceTools::toS(width, i, j + 1)];

    ptrDevImageOutput[s] = (*computeMethod)(ptrDevImageInput[s], neighborPixels, propagationSpeed);
  }
  else
  {
    ptrDevImageOutput[s] = ptrDevImageInput[s];
  }
}

__global__ void crushPerPixel(float* ptrDevImageHeater,
                              float* ptrDevImage,
                              unsigned int arraySize)
{
  const int NB_THREADS = Indice2D::nbThread();
  const int TID = Indice2D::tid();

  //int i = threadIdx.y + blockIdx.y * blockDim.y;
  int j = threadIdx.x + blockIdx.x * blockDim.x;

  int s = j + gridDim.x * blockDim.x * (threadIdx.y + blockIdx.y * blockDim.y);

  if (ptrDevImageHeater[s] > 0.0)
  {
    ptrDevImage[s] = ptrDevImageHeater[s];
  }
}

__global__ void displayPerPixel(float* ptrDevImage,
                                uchar4* ptrDevPixels,
                                unsigned int arraySize)
{
  int j = threadIdx.x + blockIdx.x * blockDim.x;

  int s = j + gridDim.x * blockDim.x * (threadIdx.y + blockIdx.y * blockDim.y);

  float heatMax = 1.0;
  float heatMin = 0.;
  float hueMax = 0.;
  float hueMin = 0.7;

  CalibreurF calibreur(IntervalF(heatMin, heatMax), IntervalF(hueMin, hueMax));

  float hue = ptrDevImage[s];
  calibreur.calibrer(hue);
  uchar4 p;
  ColorTools::HSB_TO_RVB(hue, 1, 1, &p.x, &p.y, &p.z);
  ptrDevPixels[s] = p;
}

__global__ void diffuseAdvanced(float* ptrDevImageInput,
                                float* ptrDevImageOutput,
                                unsigned int width,
                                unsigned int height,
                                float propagationSpeed,
                                ComputeMode computeMode)
{
  // Calucul threads available
  const int NB_THREADS = Indice2D::nbThread();
  const int TID = Indice2D::tid();

  // Init service and variable required
  unsigned int totalPixels = width * height;
  unsigned int s = TID;

  float (*computeMethod)(float oldHeat,
                         float* neighborPixels,
                         float propagationSpeed);

  switch(computeMode)
  {
      case ComputeMode1: computeMethod = &computeHeat1;
        break;

      case ComputeMode2: computeMethod = &computeHeat2;
        break;

      case ComputeMode3: computeMethod = &computeHeat3;
        break;
  }

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

      ptrDevImageOutput[s] = (*computeMethod)(ptrDevImageInput[s], neighborPixels, propagationSpeed);
    }
    else
    {
      ptrDevImageOutput[s] = ptrDevImageInput[s];
    }

    s += NB_THREADS;
  }
}

__global__ void crushAdvanced(float* ptrDevImageHeater,
                              float* ptrDevImage,
                              unsigned int arraySize)
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

__global__ void displayAdvanced(float* ptrDevImage,
                                uchar4* ptrDevPixels,
                                unsigned int arraySize)
{
  const int NB_THREADS = Indice2D::nbThread();
  const int TID = Indice2D::tid();
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

__device__ float computeHeat3(float oldHeat,
                              float* neighborPixels,
                              float propagationSpeed)
{
  return oldHeat + propagationSpeed * (neighborPixels[0] + neighborPixels[1] + neighborPixels[2] + neighborPixels[3] - 4 * oldHeat);
}

__device__ float computeHeat2(float oldHeat,
                              float* neighborPixels,
                              float propagationSpeed)
{
  return neighborPixels[0] + neighborPixels[1] + neighborPixels[2] + neighborPixels[3] + 4 * oldHeat / 2 * 4;
}

__device__ float computeHeat1(float oldHeat,
                              float* neighborPixels,
                              float propagationSpeed)
{
  float newHeat = oldHeat;

  for (int i = 0; i < 4; i++)
  {
    newHeat += propagationSpeed * (neighborPixels[i] - oldHeat);
  }

  return newHeat;
}
