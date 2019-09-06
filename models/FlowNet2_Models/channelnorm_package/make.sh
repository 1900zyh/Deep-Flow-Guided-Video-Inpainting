#!/usr/bin/env bash

export CUDA_PATH=/usr/local/cuda-9.0/
export CXXFLAGS="-std=c++11"
export CFLAGS="-std=c99"
export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
export CPATH=/usr/local/cuda-9.0/include${CPATH:+:${CPATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}


TORCH=$(python -c "import os; import torch; print(os.path.dirname(torch.__file__))")

cd src
echo "Compiling channelnorm kernels by nvcc..."
rm ChannelNorm_kernel.o
rm -r ../_ext

nvcc -c -o ChannelNorm_kernel.o ChannelNorm_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_35 -I ${TORCH}/lib/include/TH -I ${TORCH}/lib/include/THC
# nvcc -c -o ChannelNorm_kernel.o ChannelNorm_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52 -I ${TORCH}/lib/include/TH -I ${TORCH}/lib/include/THC

cd ../
python build.py
