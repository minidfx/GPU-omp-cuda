#include "NewtonFactory.h"
#include "NewtonGraphicGenerator.h"

#include "MathTools.h"
#include "NewtonMath.h"

ImageFonctionel *NewtonFactory::createGL()
{
    AnimableFonctionel_I *generator = NewtonFactory::createGenerator();

    return new ImageFonctionel(generator);
}

AnimableFonctionel_I *NewtonFactory::createGenerator()
{
    int width = 500;
    int height = 500;

    int nMin = 1;
    int nMax = 100;

    DomaineMath *ptrMathDomain = new DomaineMath(-2, -2, 2, 2);
    NewtonMath *ptrFractalMath = new NewtonMath(0.12);

    return new NewtonGraphicGenerator(width, height, nMax, ptrMathDomain, ptrFractalMath, "OMP Newton Graphic Generator");
}
