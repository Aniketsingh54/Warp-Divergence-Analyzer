# Warp Divergence Analyzer — LLVM Pass for GPU Kernel Divergence Detection

**Warp Divergence Analyzer** is a powerful LLVM-based static analysis tool designed to detect and analyze warp divergence in GPU kernels, particularly NVIDIA CUDA programs compiled to LLVM IR. Warp divergence — when threads within a GPU warp follow different control paths — significantly degrades GPU performance by serializing execution. This tool helps developers identify potential divergence points early in the compilation process and provides insights to optimize their GPU code.

## Key Features

- **Static Warp Divergence Detection:**  
  An LLVM IR pass that identifies branch instructions whose conditions cause threads within a warp to diverge, including simple and complex thread ID based conditions.

- **Control Flow and Data Flow Analysis:**  
  Tracks divergence through nested branches and loops to understand the overall impact on kernel execution.

- **Quantitative Divergence Metrics (Planned):**  
  Estimates the degree of divergence and its expected performance penalty.

- **Optimization Suggestions (Planned):**  
  Provides hints and code transformations to reduce warp divergence and improve GPU kernel efficiency.

- **Extensible Architecture:**  
  Designed to support multiple GPU architectures and integrate with LLVM’s evolving GPU toolchain.

- **Easy Integration:**  
  Compatible with LLVM 19+ and usable as a plugin with `opt` or integrated into custom compiler pipelines.

## Why Warp Divergence Matters

GPU warps execute instructions in lockstep, but divergence forces serialization, causing some threads to idle while others execute different code paths. Detecting and mitigating divergence early can yield significant performance gains in GPU-accelerated applications such as scientific simulations, machine learning, graphics, and more.

## Getting Started

1. Build the pass using CMake with LLVM 19+  
2. Run on LLVM IR generated from CUDA kernels  
3. Analyze pass output to identify warp divergence sites

## Future Roadmap

- Advanced divergence quantification and metrics  
- Automated code transformation suggestions  
- Integration with GPU performance models  
- Visualization and reporting tools  
- Support for AMD and Intel GPU architectures  

