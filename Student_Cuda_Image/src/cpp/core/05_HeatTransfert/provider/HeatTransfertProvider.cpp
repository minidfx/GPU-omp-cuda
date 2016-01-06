#include "HeatTransfertProvider.h"
#include "HeatTransfert.h"
#include "IndiceTools.h"

Image* HeatTransfertProvider::createGL()
    {
    Animable_I* ptrAnimable = HeatTransfertProvider::createMOO();
    return new Image(ptrAnimable);
    }

Animable_I* HeatTransfertProvider::createMOO()
    {
    unsigned int w = 500;
    unsigned int h = 500;
    unsigned int WH = w * h;

    float imageInit[WH];
    float imageHeater[WH];

    return new HeatTransfert(w, h, imageInit, imageHeater, 0.25);
    }
