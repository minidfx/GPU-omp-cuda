#ifndef RIPPLING_PROVIDER_H_
#define RIPPLING_PROVIDER_H_

#include "Image.h"
#include "Animable_I.h"

class RipplingProvider
    {
    public:

	static Image* createGL(void);
	static Animable_I* createMOO(void);
    };

#endif
