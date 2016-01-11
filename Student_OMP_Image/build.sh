rsync -zrh --delete --exclude=*.json --exclude=*.sh --progress . mse13@157.26.103.175:CUDA/toStudent/code/WCudaMSE/Student_OMP_Image
ssh -C mse13@157.26.103.175 "cd ~/CUDA/toStudent/code/WCudaMSE/Student_OMP_Image;cbicc $1"
