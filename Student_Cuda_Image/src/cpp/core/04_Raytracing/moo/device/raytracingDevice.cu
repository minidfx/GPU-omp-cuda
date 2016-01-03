#include <iostream>

#include "Indice2D.h"
#include "IndiceTools.h"
#include "cudaTools.h"
#include "Device.h"
#include "Sphere.h"
#include "ConstantMemoryLink.h"
#include "assert.h"

#include "RaytracingMath.h"

using std::cout;
using std::endl;


//CM
#define LENGTH 1535
__constant__ Sphere Spheres_DATA_CM[LENGTH];





/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Imported	 	*|
 \*-------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/


//CM S1
//ConstantMemoryLink constantMemoryLink(void);
//CM s3
//void copyHost2DeviceCM(Sphere *ptrTabSpheres , int spheresCount);

//CM
//__global__ void raytracing(uchar4* ptrDevPixels, int w, int h, float t,  int spheresCount);
//GM
//void copyGmToSm(Sphere* sphereSM, Sphere* shereGm,int n  );
__global__ void raytracing(uchar4* ptrDevPixels, int w, int h, float t, Sphere *ptrDevSpheres, int spheresCount);

//SM
//extern __global__ void raytracing(uchar4* ptrDevPixels, int w, int h, float t,Sphere *ptrSpheres,  int spheresCount);


/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

//CM S1
ConstantMemoryLink constantMemoryLink(void)
    {
    Sphere* ptrDevTab;
    size_t sizeAll = LENGTH * sizeof(Sphere);
    HANDLE_ERROR(cudaGetSymbolAddress((void ** ) &ptrDevTab, Spheres_DATA_CM));
    ConstantMemoryLink cmLink =
	{
	(void**) ptrDevTab, LENGTH, sizeAll
	};
    return cmLink;
    }

//CM s3
void copyHost2DeviceCM(Sphere *ptrTabSpheres , int spheresCount){
    assert(spheresCount == LENGTH);
    size_t size = LENGTH * sizeof(Sphere);
    int offset =0;
    HANDLE_ERROR(cudaMemcpyToSymbol(Spheres_DATA_CM ,ptrTabSpheres ,size,offset, cudaMemcpyHostToDevice));
}

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
__device__ void copyGmToSm(Sphere* sphereSM, Sphere* shereGm,int n  )
    {

       int s = Indice2D::tidLocal();
       int NB = Indice2D::nbThreadLocal();
       while(s<n)
       {
   	sphereSM[s] = shereGm[s];
   	s += NB;
        }

    }

__device__ void work(uchar4* ptrDevPixels, int w, int h, float t, Sphere *ptrDevSpheres, int spheresCount){

    RaytracingMath raytracingMath = RaytracingMath(w, h);
      // TODO pattern entrelacement
      const int WH = w * h;
      const int NB_THREAD = Indice2D::nbThread(); // dans region parallel

      const int TID = Indice2D::tid();
      int s = TID; // in [0,NB_THREAD[
      int i;
      int j;
        while (s < WH)
    	{
    	 IndiceTools::toIJ(s, w, &i, &j); // s[0,W*H[ --> i[0,H[ j[0,W[

    	 raytracingMath.colorIJ(&ptrDevPixels[s], i, j, t, ptrDevSpheres, spheresCount);
    	 s += NB_THREAD;
    	}
}

__global__ void raytracing(uchar4* ptrDevPixels, int w, int h, float t, Sphere *ptrDevSpheres, int spheresCount)

    {

    extern __shared__  Sphere sphereSM[];



    copyGmToSm(sphereSM,ptrDevSpheres,spheresCount);
    __syncthreads();
    //work(ptrDevPixels,w,h,t,ptrDevSpheres,spheresCount);
    work(ptrDevPixels,w,h,t,sphereSM,spheresCount);

    }


////GM
//__global__ void raytracing(uchar4* ptrDevPixels, int w, int h, float t, Sphere *ptrDevSpheres, int spheresCount)
//
////CM
////__global__ void raytracing(uchar4* ptrDevPixels, int w, int h, float t,  int spheresCount)
//
////SM
////__global__ void raytracing(uchar4* ptrDevPixels, int w, int h, float t,Sphere *ptrSpheres,  int spheresCount)
//{
//
//
//     extern __shared__  Sphere sphereSM[];
//     copyGmToSm(sphereSM,ptrDevSpheres,spheresCount);
//     __syncthreads();
//
//    RaytracingMath raytracingMath = RaytracingMath(w, h);
//    // TODO pattern entrelacement
//
//
//    int j1 = threadIdx.x + blockIdx.x * blockDim.x;
//          int i1 =  threadIdx.y + blockIdx.y * blockDim.y;
//
//          int s = i1 *w + j1;
//
//////GM
//         raytracingMath.colorIJ(&ptrDevPixels[s], i1, j1, t, sphereSM, spheresCount);
//
////// CM
//       // raytracingMath.colorIJ(&ptrDevPixels[s], i1, j1, t, Spheres_DATA_CM, spheresCount);
//
//////SM
//        //  extern __shared__  Sphere sphereSM[];
//        //  int s1 = threadIdx.x;
//        //  sphereSM[s1] = ptrSpheres[s1];
//        //  __syncthreads();
//        //  raytracingMath.colorIJ(&ptrDevPixels[s], i1, j1, t, sphereSM, spheresCount);
//    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

