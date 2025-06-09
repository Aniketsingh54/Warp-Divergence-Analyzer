#include "WarpAnalysis.h"
#include "AnalysisLogic.h"
#include "JsonEmitter.h"
using namespace llvm;

PreservedAnalyses WarpAnalysis::run(Module &M, ModuleAnalysisManager &MAM) {
  auto &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  warp::FileKernelMap fileMap;
  for (auto &F : M) {
    warp::analyzeFunction(F, FAM, fileMap);
  }
  warp::emitFileGroups(fileMap);
  return PreservedAnalyses::all();
}

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