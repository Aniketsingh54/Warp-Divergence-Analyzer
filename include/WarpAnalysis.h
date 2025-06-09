#ifndef LLVM_WARP_ANALYSIS_H
#define LLVM_WARP_ANALYSIS_H

#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Passes/PassBuilder.h"

namespace llvm {

class WarpAnalysis : public PassInfoMixin<WarpAnalysis> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

} // namespace llvm

#endif // LLVM_WARP_ANALYSIS_H