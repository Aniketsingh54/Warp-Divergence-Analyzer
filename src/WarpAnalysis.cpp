#include "WarpAnalysis.h"
#include "AnalysisLogic.h"
#include "JsonEmitter.h"
using namespace llvm;

/**
 * @brief Runs the WarpAnalysis pass on the given LLVM module.
 *
 * This function iterates over all functions in the provided module, analyzes each
 * function using the warp::analyzeFunction utility, and collects results in a file map.
 * After analysis, it emits grouped file information using warp::emitFileGroups.
 *
 * @param M The LLVM module to analyze.
 * @param MAM The module analysis manager providing access to function analyses.
 * @return PreservedAnalyses indicating that all analyses are preserved.
 */
PreservedAnalyses WarpAnalysis::run(Module &M, ModuleAnalysisManager &MAM) {
  auto &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  warp::FileKernelMap fileMap;
  for (auto &F : M) {
    warp::analyzeFunction(F, FAM, fileMap);
  }
  warp::emitFileGroups(fileMap);
  return PreservedAnalyses::all();
}

/**
 * @brief Returns the PassPluginLibraryInfo for the WarpAnalysis LLVM pass plugin.
 *
 * This function is the entry point for the LLVM pass plugin. It provides
 * information about the plugin, such as its API version, name, and LLVM version.
 * It also registers the "warp-analysis" pass with the LLVM PassBuilder, allowing
 * it to be invoked via the new pass manager pipeline.
 *
 * @return PassPluginLibraryInfo structure containing plugin metadata and
 *         registration callbacks.
 */
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "WarpAnalysis", LLVM_VERSION_STRING,
    [](PassBuilder &PB) {
      PB.registerPipelineParsingCallback(
        [](StringRef Name, ModulePassManager &MPM, ArrayRef<PassBuilder::PipelineElement>) {
          if (Name == "warp-analysis") {
            MPM.addPass(WarpAnalysis());
            return true;
          }
          return false;
        }
      );
    }
  };
}