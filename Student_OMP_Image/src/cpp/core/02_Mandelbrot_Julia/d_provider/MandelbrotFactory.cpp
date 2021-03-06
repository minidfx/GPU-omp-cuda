#include "MandelbrotFactory.h"
#include "GraphicGenerator.h"

#include "MathTools.h"
#include "MandelBrotMath.h"

ImageFonctionel* MandelbrotFactory::createGL()
    {
    AnimableFonctionel_I* generator = MandelbrotFactory::createGenerator();

    return new ImageFonctionel(generator);
    }

AnimableFonctionel_I* MandelbrotFactory::createGenerator()
    {
    const int width = 0.8 * 1000;
    const int height = 0.5 * 1000;

    DomaineMath *ptrMathDomain = new DomaineMath(-2.1, -1.3, 0.8, 1.3);
    FractalMathBase *ptrFractalMath = new MandelBrotMath();

    const int nMin = 30;
    const int nMax = 100;

    return new GraphicGenerator(width, height, nMin, nMax, ptrMathDomain, ptrFractalMath, "OMP Mandelbrot (zoom enabled)");
    }
