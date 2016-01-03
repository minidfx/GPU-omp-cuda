#include <iostream>

#include "Indice2D.h"
#include "cudaTools.h"
#include "Device.h"

#include "IndiceTools.h"

using std::cout;
using std::endl;

__global__ void ecrasement(float* ptrImageInOutput, int w, int h);

__global__ void ecrasement(float* ptrImageInOutput, float* ptrImageHeater, int w, int h)
    {
    const int WH = w * h;

    const int NB_THREAD = Indice2D::nbThread();
    const int TID = Indice2D::tid();

    int s = TID;

    while (s < WH)
	{
	if(ptrImageHeater[s] != 0)
	    ptrImageInOutput[s] = ptrImageHeater[s];
	s += NB_THREAD;
	}
    }
