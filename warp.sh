#!/bin/bash
set -e

show_help() {
  echo "Usage: ./warp.sh [OPTION]"
  echo ""
  echo "Options:"
  echo "  --build         Configure and build the LLVM plugin"
  echo "  --test          Compile CUDA test kernel and run WarpAnalysis pass"
  echo "  --start         Start Docker container for GPU analysis"
  echo "  --help          Show this help message"
}

build() {
  echo "[*] Cleaning previous build..."
  rm -rf build
  mkdir -p build
  cd build

  echo "[*] Running cmake and make..."
  cmake .. \
    -DCMAKE_CXX_FLAGS="-I/usr/lib/llvm-19/include -std=c++17" \
    -DCMAKE_EXE_LINKER_FLAGS="$(llvm-config --ldflags)"
  make -j$(nproc)
  echo "[*] Build completed successfully."

  cd ..
}

test() {
  echo "[*] Compiling CUDA kernel to LLVM bitcode..."
  mkdir -p bin
  clang++-19 -x cuda \
    --cuda-path=/usr/local/cuda \
    --cuda-gpu-arch=sm_75 \
    --cuda-device-only \
    -O2 -g -emit-llvm -c test/kernels.cu \
    -o bin/kernels.bc

  echo "[*] Running WarpAnalysis pass..."
  opt -load-pass-plugin ./build/WarpAnalysis.so \
      -passes=warp-analysis \
      bin/kernels.bc \
      -disable-output
}

start() {
  echo "[*] Starting Docker container with GPU access..."
  docker run --rm -it --gpus all -v "$PWD":/workspace warp-analysis-gpu
}

# Parse arguments
case "$1" in
  --build) build ;;
  --test) test ;;
  --start) start ;;
  --help | "" ) show_help ;;
  *) echo "Unknown option: $1"; show_help; exit 1 ;;
esac
