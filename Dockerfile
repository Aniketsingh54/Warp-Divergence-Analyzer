# LLVM 19 + CUDA Dev Container
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install core dev tools
RUN apt-get update && apt-get install -y \
  build-essential cmake ninja-build curl wget git gnupg lsb-release \
  libz-dev libxml2-dev libedit-dev python3 python3-pip \
  && rm -rf /var/lib/apt/lists/*

# Add LLVM 19 from official repo
RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
    echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-19 main" > /etc/apt/sources.list.d/llvm19.list && \
    apt-get update && \
    apt-get install -y \
      llvm-19 llvm-19-dev llvm-19-tools \
      clang-19 libclang-common-19-dev && \
    rm -rf /var/lib/apt/lists/*

# Set default LLVM tools
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-19 100 && \
    update-alternatives --install /usr/bin/opt opt /usr/bin/opt-19 100 && \
    update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-19 100

# Set workspace
WORKDIR /workspace
COPY . .

ENTRYPOINT ["/bin/bash"]
