#include <iostream>
#include <assert.h>

#include "Raytracing.h"
#include "Device.h"
#include "Sphere.h"
#include "math.h"
#include "ConstantMemoryLink.h"
#include <stdio.h>

using std::cout;
using std::endl;
using cpu::IntervalF;


//CM S1
//__device__  Sphere Spheres_DATA_CM[2300];

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Imported	 	*|
 \*-------------------------------------*/

//CM S1
//extern ConstantMemoryLink constantMemoryLink(void);
//CM s3
//extern void copyHost2DeviceCM(Sphere *ptrSpheres , int spheresCount);
//CM
//extern __global__ void raytracing(uchar4* ptrDevPixels, int w, int h, float t,  int spheresCount);


//GM
//extern __global__ void raytracing(uchar4* ptrDevPixels, int w, int h, float t,Sphere *ptrSpheres,  int spheresCount);

//SM
extern __global__ void raytracing(uchar4* ptrDevPixels, int w, int h, float t,Sphere *ptrSpheres,  int spheresCount);

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

/*-------------------------*\
 |*	Constructeur	    *|
 \*-------------------------*/

Raytracing::Raytracing(int w, int h, IntervalF dt):variateurF(dt,0.01)
    {
    assert(w == h);

    // Inputs
    this->w = w;
    this->h = h;
   // this->dt = dt;

    // Tools
  //  this->dg = dim3(16 , 2, 1); // disons a optimiser
   // this->db = dim3(16,  16, 1); // disons a optimiser
    this->dg = dim3(16*16  , 16*16 , 1);
    this->db = dim3(16*2 , 16 *2, 1);
    this->t = 0;


    // Outputs
   this->title = "Raytracing_Cuda" ;


    ptrSpheres = new Sphere[this->spheresCount];

    float3 centre;
    float rayon;
    float hue;
    int bord = 200;


    //Instanciate all the spheres in the array of spheres
    for(int i = 0; i<this->spheresCount; i++){
         centre.x = RandomInRange(0+bord, w-bord);
         centre.y = RandomInRange(bord, h-bord);
         centre.z = RandomInRange(10,2*w);
         rayon = RandomInRange(20, w/10);
         hue = RandomFloat(0,1);
         this->ptrSpheres[i] = Sphere(centre, rayon, hue);
         assert(centre.x >= 0 && centre.x<=w);
         assert(rayon >= 20);
         assert(centre.y >= 0 && centre.y<=h);
    }

//CM S1
    //ConstantMemoryLink cml = constantMemoryLink();
    //Sphere* ptrDevSphereCml =  (Sphere*)cml.ptrDevTab;
    //assert(spheresCount == cml.n);
    //HANDLE_ERROR(cudaMemcpy(ptrDevSphereCml, ptrSpheres ,spheresCount,cudaMemcpyHostToDevice ));

//CM S3
    //copyHost2DeviceCM(ptrSpheres,spheresCount);



//GM
      size =  this->spheresCount * sizeof(Sphere);
      printf("sizeofshpere %zu\n ",sizeof(Sphere));
      printf("numberofSpheres %zu\n ",this->spheresCount);
      printf("sizeofshperes %zu\n ",size);
      HANDLE_ERROR(cudaMalloc(&ptrDevSpheres, size));
      int initialValue = 0;
      HANDLE_ERROR(cudaMemset(ptrDevSpheres,initialValue,size));
      HANDLE_ERROR(cudaMemcpy(ptrDevSpheres,ptrSpheres,size,cudaMemcpyHostToDevice));

//SM
    //size =  this->spheresCount * sizeof(Sphere);

    Device::assertDim(dg, db);


    }

Raytracing::~Raytracing()
    {
//GM
    HANDLE_ERROR(cudaFree(ptrDevSpheres));
    // rien
    }

/*-------------------------*\
 |*	Methode		    *|
 \*-------------------------*/
float Raytracing::RandomFloat(float a, float b) {
    float random = ((float) rand()) / (float) RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}

int Raytracing::RandomInRange(int x, int y){
    return x+(rand()%(y-x+1));
}

/**
 * Override
 */



void Raytracing::process(uchar4* ptrDevPixels, int w, int h)
    {



//GM
   //raytracing<<<dg,db>>>(ptrDevPixels,w,h,t,ptrDevSpheres, this->spheresCount);

//CM
    //raytracing<<<dg,db>>>(ptrDevPixels,w,h,t, this->spheresCount);


//SM
   raytracing<<<dg,db,size>>>(ptrDevPixels,w,h,t,ptrDevSpheres, this->spheresCount);

    }




/**
 * Override
 */
void Raytracing::animationStep()
    {
	//t+=dt;
    t = variateurF.varierAndGet();
    }

/*--------------*\
 |*	get	 *|
 \*--------------*/

/**
 * Override
 */
float Raytracing::getAnimationPara(void)
    {
    return t;
    }

/**
 * Override
 */
int Raytracing::getW(void)
    {
    return w;
    }

/**
 * Override
 */
int Raytracing::getH(void)
    {
    return  h;
    }

/**
 * Override
 */
string Raytracing::getTitle(void)
    {
    return title;
    }


/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/


/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

