#ifndef HEAT_TRANSFERT_PROVIDER_H_
#define HEAT_TRANSFERT_PROVIDER_H_

#include "Image.h"
#include "Animable_I.h"

class HeatTransfertProvider {

  public:
    static Image* createGL();
  	static Animable_I* createMOO();
};

#endif
