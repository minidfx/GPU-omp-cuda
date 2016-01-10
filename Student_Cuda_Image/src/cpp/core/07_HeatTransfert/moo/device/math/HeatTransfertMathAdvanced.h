#ifndef HEAT_TRANSFERT_MATH_ADVANCED_H_
#define HEAT_TRANSFERT_MATH_ADVANCED_H_

/**
 * Dans un header only pour preparer la version cuda
 */
class HeatTransfertMathAdvanced
    {

    public:
	__device__ HeatTransfertMathAdvanced()
	    {
	    }

	__device__ ~HeatTransfertMathAdvanced()
	    {
	    // Nothing special to do
	    }

	__device__
	float computeHeat(float oldHeat, float* neighborsHeat, unsigned int nbNeighbors, float propSpeed)
	    {
	    float newHeat = oldHeat;

	    for (int i = 0; i < nbNeighbors; i++)
		{
		newHeat += propSpeed * (neighborsHeat[i] - oldHeat);
		}

	    return newHeat;
	    }
    };

#endif
