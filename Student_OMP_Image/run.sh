bash ./build.sh $1
ssh -C mse13@157.26.103.175 "cd ~/CUDA/toStudent/code/WCudaMSE/Student_OMP_Image;cbicc $1 run"
