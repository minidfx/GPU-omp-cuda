#!/bin/bash

bash ./build.sh $1

if [ $? != 0 ]; then
  exit 1
fi

ssh -C mse13@157.26.103.175 "cd ~/CUDA/toStudent/code/WCudaMSE/Student_OMP_Image;cbicc $1 run"
