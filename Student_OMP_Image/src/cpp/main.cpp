#include <iostream>
#include <stdlib.h>

#include "Settings.h"

using std::cout;
using std::endl;

extern int mainGL(Settings& settings);
extern int mainMOO(Settings& settings);
static int start(Settings& settings);

int main(int argc, char** argv) {
  cout<<"main"<<endl;

  bool IS_GL = false;
  Settings settings(IS_GL,argc,argv);

  return start(settings);
}

int start(Settings& settings) {
  if (settings.isGL()) {
    return mainGL(settings); // Bloquant, Tant qu'une fenetre est ouverte
  } else {
    return mainMOO(settings); // FreeGL
  }
}
