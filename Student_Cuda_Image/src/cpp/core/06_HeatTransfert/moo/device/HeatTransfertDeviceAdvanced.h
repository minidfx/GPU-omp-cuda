#ifndef HEATTRANSFERTDEVICE
#define HEATTRANSFERTDEVICE

typedef enum {
  ComputeMode1,
  ComputeMode2,
  ComputeMode3
} ComputeMode;

__global__ void diffuseAdvanced(float* ptrDevImageInput,
                                float* ptrDevImageOutput,
                                unsigned int width,
                                unsigned int height,
                                float propagationSpeed,
                                ComputeMode computeMode);
__global__ void crushAdvanced(float* ptrDevImageHeater,
                              float* ptrDevImage,
                              unsigned int arraySize);
__global__ void displayAdvanced(float* ptrDevImage,
                                uchar4* ptrDevPixels,
                                unsigned int arraySize);

__global__ void diffusePerPixel(float* ptrDevImageInput,
                                float* ptrDevImageOutput,
                                unsigned int width,
                                unsigned int height,
                                float propagationSpeed,
                                ComputeMode computeMode);
__global__ void crushPerPixel(float* ptrDevImageHeater,
                              float* ptrDevImage,
                              unsigned int arraySize);
__global__ void displayPerPixel(float* ptrDevImage,
                                uchar4* ptrDevPixels,
                                unsigned int arraySize);

__device__ float computeHeat1(float oldHeat,
                              float* neighborPixels,
                              float propagationSpeed);
__device__ float computeHeat2(float oldHeat,
                              float* neighborPixels,
                              float propagationSpeed);
__device__ float computeHeat3(float oldHeat,
                              float* neighborPixels,
                              float propagationSpeed);

#endif
