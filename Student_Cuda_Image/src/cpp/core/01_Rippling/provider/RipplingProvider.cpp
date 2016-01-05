#include "RipplingProvider.h"
#include "Rippling.h"

Image* RipplingProvider::createGL()
    {
    Animable_I* ptrAnimable = RipplingProvider::createMOO();

    return new Image(ptrAnimable);
    }

Animable_I* RipplingProvider::createMOO()
    {
    float dt = 1;

    int dw = 500;
    int dh = 500;

    return new Rippling(dw, dh, dt);
    }
