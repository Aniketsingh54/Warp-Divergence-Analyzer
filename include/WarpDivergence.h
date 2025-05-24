#ifndef LLVM_WARP_DIVERGENCE_ANALYZER_H
#define LLVM_WARP_DIVERGENCE_ANALYZER_H

#include "llvm/IR/PassManager.h"
#include "llvm/IR/Function.h"

struct WarpDivergencePass : public llvm::PassInfoMixin<WarpDivergencePass> {
  llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM);
};

#endif // LLVM_WARP_DIVERGENCE_ANALYZER_H
