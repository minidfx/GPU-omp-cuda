#!/bin/bash

bash build.sh

if [ $? != 0 ]; then
  exit 1
fi

ssh -C mse13@157.26.103.175 "cd ~/CUDA/toStudent/code/WCudaMSE/Student_Cuda_Image;cbicc cuda run"
