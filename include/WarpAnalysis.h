#ifndef LLVM_WARP_ANALYSIS_H
#define LLVM_WARP_ANALYSIS_H

#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"

namespace llvm {

/// \brief A ModulePass that analyzes each function for:
/// 1) total vs. divergent conditional branches (via PostDominatorTree)
/// 2) number of memory ops per address space (shared, local, global)
struct WarpAnalysis : public ModulePass {
  static char ID;
  WarpAnalysis();  // Constructor registers the pass ID

  bool runOnModule(Module &M) override;

private:
  void analyzeFunction(Function &F);
};

} // namespace llvm

#endif // LLVM_WARP_ANALYSIS_H
