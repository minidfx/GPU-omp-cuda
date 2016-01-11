#include <iostream>
#include <stdlib.h>

#include "Settings.h"

#include "Animateur.h"
#include "AnimateurFonctionel.h"

#include "RipplingProvider.h"
#include "MandelbrotFactory.h"
#include "JuliaFactory.h"
#include "NewtonFactory.h"
#include "HeatTransfertProviderAdvanced.h"

using std::cout;
using std::endl;
using std::string;

int mainMOO(Settings& settings);

static void animeAndDestroy(Animable_I* ptrAnimable, int nbIteration);
static void animeAndDestroy(AnimableFonctionel_I* ptrAnimable, int nbIteration);

int mainMOO(Settings& settings) {
  cout << "\n[FreeGL] mode" << endl;

  const int NB_ITERATION = 1000;

  //animeAndDestroy(RipplingProvider::createMOO(), NB_ITERATION);
  //animeAndDestroy(MandelbrotFactory::createGenerator(), NB_ITERATION);
  //animeAndDestroy(JuliaFactory::createGenerator(), NB_ITERATION);
  //animeAndDestroy(NewtonFactory::createGenerator(), NB_ITERATION);
  animeAndDestroy(HeatTransfertProviderAdvanced::createMOO(), NB_ITERATION);

  cout << "\n[FreeGL] end" << endl;

  return EXIT_SUCCESS;
}

void animeAndDestroy(Animable_I* ptrAnimable, int nbIteration) {
  Animateur animateur(ptrAnimable, nbIteration);
  animateur.run();

  delete ptrAnimable;
}

void animeAndDestroy(AnimableFonctionel_I* ptrAnimable, int nbIteration) {
  AnimateurFonctionel animateur(ptrAnimable, nbIteration);
  animateur.run();

  delete ptrAnimable;
}
