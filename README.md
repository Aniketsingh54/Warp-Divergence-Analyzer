# Warp Insight

A comprehensive static-analysis framework for CUDA kernels, built on LLVM, that detects warp divergence, stall cycles, and memory‐access patterns — and presents them as structured JSON reports and in-editor annotations.


## Motivation

CUDA’s single-instruction multiple-thread (SIMT) model executes groups of 32 threads (“warps”) in lock-step. Any divergence in control flow or long-latency memory access within a warp can cause significant performance penalties.  

**Warp Insight** empowers developers to:

- Pinpoint **divergent branches** and **thread-dependent conditions**  
- Measure **stall cycles** due to memory and synchronization  
- Classify and map **memory operations** (global, shared, local)  
- Correlate stalls/divergences back to **source code locations**  

---

## Architecture Overview

1. **LLVM Analysis Pass**  
   - Parses each CUDA kernel’s LLVM IR  
   - Records branch instructions, dependencies, memory ops, and barriers  
   - Captures debug metadata for source‐line mapping  

2. **JSON Emitter**  
   - Collects per‐kernel statistics and source coordinates  
   - Emits one `.json` file per source file under `json/`  

3. **Dockerized Build & Execution**  
   - Ensures consistent, reproducible environment  
   - Users need only Docker and the `warp.sh` wrapper script  

4. **VS Code Extension**  
   - Reads the JSON reports  
   - Highlights divergent branches and memory‐access hotspots  
   - Supports undo/clear of decorations  

---

## Quick Start

```bash
# 1. Build the analysis tool
./warp.sh --build

# 2. Drop your CUDA files into 'test/'
cp path/to/your_kernel.cu test/

# 3. Run analysis
./warp.sh --test

# 4. Install VS Code extension (if not already)

# 5. Open your .cu in VS Code and run the 'Warp Insight: Highlight' command
```

