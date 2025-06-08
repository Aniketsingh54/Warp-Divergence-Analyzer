#!/bin/bash
set -e

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