#include "cudaType.h"

#include "Indice2D.h"
#include "IndiceTools.h"
#include "ColorTools.h"
#include "HeatTransfertMath.h"

__global__ void diffuse(float* ptrDevImageInput, float* ptrDevImageOutput, unsigned int w, unsigned int h, float propSpeed);
__global__ void crush(float* ptrDevImageHeater, float* ptrDevImage, unsigned int size);
__global__ void toScreen(float* ptrDevImage, uchar4* ptrDevPixels, unsigned int size);

__global__ void diffuse(float* ptrDevImageInput, float* ptrDevImageOutput, unsigned int w, unsigned int h, float propSpeed) {
  const int NB_THREADS = Indice2D::nbThread();
	const int TID = Indice2D::tid();

  HeatTransfertMath math;
  unsigned int wh = w*h;
  unsigned int s = TID;
  while ( s < wh ) {
    int i, j;
		IndiceTools::toIJ(s, w, &i, &j);
    if (i > 0 && i < (h - 1) && j > 0 && j < (w - 1)) {

      float neighborsHeat[4];
      neighborsHeat[0] = ptrDevImageInput[IndiceTools::toS(w, i - 1, j)];
      neighborsHeat[1] = ptrDevImageInput[IndiceTools::toS(w, i + 1, j)];
      neighborsHeat[2] = ptrDevImageInput[IndiceTools::toS(w, i, j - 1)];
      neighborsHeat[3] = ptrDevImageInput[IndiceTools::toS(w, i, j + 1)];

      ptrDevImageOutput[s] = math.computeHeat(ptrDevImageInput[s], neighborsHeat, 4, propSpeed);
    } else {
      ptrDevImageOutput[s] = ptrDevImageInput[s];
    }

    s += NB_THREADS;
  }
}

__global__ void crush(float* ptrDevImageHeater, float* ptrDevImage, unsigned int size) {
  const int NB_THREADS = Indice2D::nbThread();
	const int TID = Indice2D::tid();

  unsigned int s = TID;
  while ( s < size ) {
    if (ptrDevImageHeater[s] > 0.0) {
      ptrDevImage[s] = ptrDevImageHeater[s];
    }
    s += NB_THREADS;
  }
}

__global__ void toScreen(float* ptrDevImage, uchar4* ptrDevPixels, unsigned int size) {
  const int NB_THREADS = Indice2D::nbThread();
	const int TID = Indice2D::tid();

  unsigned int s = TID;
  while ( s < size ) {

    float hue = 0.7 - ptrDevImage[s] * 0.7;
    ColorTools::HSB_TO_RVB(hue, 1, 1, &ptrDevPixels[s]);
    ptrDevPixels[s].w = 255;

    s += NB_THREADS;
  }
}
