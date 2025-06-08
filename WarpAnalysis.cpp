#include "llvm/IR/PassManager.h"
#include "llvm/IR/Module.h" 
#include "llvm/IR/Function.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Analysis/PostDominators.h"

using namespace llvm;

namespace {

class WarpAnalysis : public PassInfoMixin<WarpAnalysis> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
    errs() << "[WarpAnalysis] Running on module: " << M.getName() << "\n";

    FunctionAnalysisManager &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

    for (Function &F : M) {
      if (F.isDeclaration() || F.isIntrinsic())
        continue;

      errs() << "\nAnalyzing Function: " << F.getName() << "\n";

      DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(F);
      PostDominatorTree &PDT = FAM.getResult<PostDominatorTreeAnalysis>(F);

      int branchCount = 0;
      int divergentBranches = 0;
      int memOps = 0;
      int sharedMemOps = 0;
      int globalMemOps = 0;
      int localMemOps = 0;

      for (auto &BB : F) {
        auto *term = BB.getTerminator();
        if (auto *br = dyn_cast<BranchInst>(term)) {
          if (br->isConditional()) {
            branchCount++;
            if (!PDT.dominates(br->getParent(), br->getSuccessor(0)) ||
                !PDT.dominates(br->getParent(), br->getSuccessor(1))) {
              divergentBranches++;
            }
          }
        }

        for (Instruction &I : BB) {
           if (auto *sel = dyn_cast<SelectInst>(&I)) {
            branchCount++;
            Value *cond = sel->getCondition();
            if (Instruction *condInst = dyn_cast<Instruction>(cond)) {
              if (condInst->getOpcode() == Instruction::ICmp) {
                divergentBranches++;
              }
            }
          }
          if (auto *load = dyn_cast<LoadInst>(&I)) {
            memOps++;
            switch (load->getPointerAddressSpace()) {
            case 3:
              sharedMemOps++;
              break;
            case 5:
              localMemOps++;
              break;
            default:
              globalMemOps++;
              break;
            }
          }
          if (auto *store = dyn_cast<StoreInst>(&I)) {
            memOps++;
            switch (store->getPointerAddressSpace()) {
            case 3:
              sharedMemOps++;
              break;
            case 5:
              localMemOps++;
              break;
            default:
              globalMemOps++;
              break;
            }
          }
        }
      }

      errs() << "Branches: " << branchCount << ", Divergent: " << divergentBranches << "\n";
      errs() << "Memory Ops: " << memOps << " (Shared: " << sharedMemOps
             << ", Local: " << localMemOps << ", Global: " << globalMemOps << ")\n";
    }

    return PreservedAnalyses::all();
  }
};

}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {
      LLVM_PLUGIN_API_VERSION, "WarpAnalysis", LLVM_VERSION_STRING,
      [](PassBuilder &PB) {
        PB.registerPipelineParsingCallback(
            [](StringRef Name, ModulePassManager &MPM,
               ArrayRef<PassBuilder::PipelineElement>) {
              if (Name == "warp-analysis") {
                MPM.addPass(WarpAnalysis());
                return true;
              }
              return false;
            });
      }};
}
