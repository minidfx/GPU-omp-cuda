#ifndef HEAT_TRANSFERT_MATH_H_
#define HEAT_TRANSFERT_MATH_H_

/**
 * Dans un header only pour preparer la version cuda
 */
class HeatTransfertMath {

  public:
    __device__ HeatTransfertMath() {
    }

    __device__ ~HeatTransfertMath() {
      // Nothing special to do
    }

    __device__ float computeHeat(float oldHeat, float* neighborsHeat, unsigned int nbNeighbors, float propSpeed) {
      float newHeat = oldHeat;

      for (int i = 0 ; i < nbNeighbors ; i++) {
        newHeat += propSpeed * (neighborsHeat[i] - oldHeat);
      }

      return newHeat;
    }
};

#endif
