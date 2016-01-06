#include "JuliaFactory.h"
#include "GraphicGenerator.h"

#include "MathTools.h"
#include "JuliaMath.h"

ImageFonctionel* JuliaFactory::createGL()
{
    AnimableFonctionel_I* generator = JuliaFactory::createGenerator();

    return new ImageFonctionel(generator);
}

AnimableFonctionel_I* JuliaFactory::createGenerator()
{
    const int width = 0.8 * 1000;
    const int height = 0.5 * 1000;

    DomaineMath *ptrMathDomain = new DomaineMath(-2.1, -1.3, 0.8, 1.3);
    FractalMathBase *ptrFractalMath = new JuliaMath(-0.12, 0.85);

    const int nMin = 30;
    const int nMax = 100;

    return new GraphicGenerator(width, height, nMin, nMax, ptrMathDomain, ptrFractalMath, "OMP Julia (zoom enabled)");
}
