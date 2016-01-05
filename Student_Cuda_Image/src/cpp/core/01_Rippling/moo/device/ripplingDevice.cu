#include <iostream>

#include "cudaTools.h"
#include "Device.h"

#include "Indice2D.h"
#include "IndiceTools.h"
#include "RipplingMath.h"

using std::cout;
using std::endl;

__global__
void rippling(uchar4* ptrDevPixels, int w, int h, float t);

__global__
void rippling(uchar4* ptrDevPixels, int w, int h, float t)
    {
    RipplingMath ripplingMath(w, h);

    //const int TID = Indice1D::tid(); // 1D
    //const int NB_THREAD = Indice1D::nbThread(); // 1D
    
    const int TID = Indice2D::tid(); // 2D
    const int NB_THREAD = Indice2D::nbThread(); // 2D

    const int WH = w * h;

    //int pixelI = threadIdx.y + blockIdx.y * blockDim.y;
    //int pixelJ = threadIdx.x + blockIdx.x * blockDim.x;

    //int s = pixelJ + gridDim.x * blockDim.x * (threadIdx.y + blockIdx.y * blockDim.y);

    //ripplingMath.colorIJ(&ptrDevPixels[s], pixelI, pixelJ, t);

    int s = TID;
    while (s < WH)
	{
	int pixelI;
	int pixelJ;
	uchar4 color;

	IndiceTools::toIJ(s, w, &pixelI, &pixelJ); // update (pixelI, pixelJ)

	ripplingMath.colorIJ(&color, pixelI, pixelJ, t); 	// update color
	ptrDevPixels[s] = color;

	s += NB_THREAD;
	}
    }
