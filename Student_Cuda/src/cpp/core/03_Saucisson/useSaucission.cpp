#include <iostream>
#include <limits.h>
#include <math.h>
#include "MathTools.h"
#include "Saucisson_host.h"

using std::cout;
using std::endl;

bool useSaucisson(void);

bool useSaucisson(void)
{
    cout << endl << "[Saucisson]" << endl;

    bool isOk = true;

    int n = INT_MAX/500;
    Saucisson saucisson(n);
    saucisson.process();

    cout << "saucisson pi: " << saucisson.getPi() << endl;

    isOk &= MathTools::isEquals(saucisson.getPi(), M_PI, 1e-3);

    return true;
}
