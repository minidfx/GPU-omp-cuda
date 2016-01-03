#include <iostream>
#include <assert.h>

#include "Rippling.h"
#include "Device.h"

using std::cout;
using std::endl;

__global__
extern void rippling(uchar4* ptrDevPixels, int w, int h, float t);

Rippling::Rippling(int w, int h, float dt)
{
    assert(w == h);

    // Inputs
    this->w = w;
    this->h = h;
    this->dt = dt;

    // Tools
    //this->dg = dim3(1024, 1, 1); // 1D
    //this->db = dim3(256, 1, 1); / 1D
    
 	//this->dg = dim3(96, 2, 1);
    //this->db = dim3(128, 4, 1);
    
   	this->dg = dim3(64, 64, 1); // Block
    this->db = dim3(16, 16, 1); // Threads
    
    this->t = 0;

    // Outputs
    this->title = "CUDA Rippling";

    //print(dg, db);
    Device::assertDim(dg, db);
}

Rippling::~Rippling()
{
}

void Rippling::process(uchar4* ptrDevPixels, int w, int h)
{
    rippling<<<dg,db>>>(ptrDevPixels, w, h, t);
}

void Rippling::animationStep()
{
    t += dt;
}

float Rippling::getAnimationPara()
{
    return t;
}

int Rippling::getW()
{
    return w;
}

int Rippling::getH()
{
    return  h;
}

string Rippling::getTitle()
{
    return title;
}
