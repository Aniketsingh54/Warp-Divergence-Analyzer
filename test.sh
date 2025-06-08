clang++-19 -x cuda \
  --cuda-path=/usr/local/cuda \
  --cuda-gpu-arch=sm_75 \
  --cuda-device-only \
  -O2 -g -emit-llvm -c test/kernels.cu \
  -o bin/kernels.bc
opt -load-pass-plugin ./build/WarpAnalysis.so \
    -passes=warp-analysis \
    bin/kernels.bc \
    -disable-output