#ifndef LLVM_WARP_ANALYSIS_H
#define LLVM_WARP_ANALYSIS_H

#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Passes/PassBuilder.h"

/// \brief Analysis pass for performing warp-level analysis on LLVM modules.
///
/// The WarpAnalysis class implements an LLVM pass that analyzes a given
/// Module to extract information relevant to warp-level execution or
/// transformations. It is designed to be used with the LLVM PassManager
/// infrastructure.
///
/// Usage:
///   - Instantiate and register the pass with the LLVM PassManager.
///   - The pass operates on LLVM Module objects.
///
/// Example:
///   llvm::WarpAnalysis warpAnalysis;
///   warpAnalysis.run(module, moduleAnalysisManager);
///
/// \see llvm::PassInfoMixin
namespace llvm {
class WarpAnalysis : public PassInfoMixin<WarpAnalysis> {
public:
  /// Runs the warp analysis pass on the given LLVM module.
  ///
  /// \param M The LLVM module to analyze.
  /// \param MAM The module analysis manager providing analysis results.
  /// \return A PreservedAnalyses object indicating which analyses are preserved.
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

} // namespace llvm

#endif // LLVM_WARP_ANALYSIS_H