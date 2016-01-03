extern bool isMontecarlo();
extern bool isMontecarloMGPU();

bool useMontecarlo();
bool useMontecarloMGPU();

bool useMontecarlo() {
  return isMontecarlo();
}

bool useMontecarloMGPU() {
  return isMontecarloMGPU();
}
