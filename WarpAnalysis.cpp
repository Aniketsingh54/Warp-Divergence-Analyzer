#include "llvm/IR/PassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstIterator.h"

using namespace llvm;

namespace {

bool isThreadIdxSource(Value *V) {
  if (auto *call = dyn_cast<CallBase>(V)) {
    if (Function *callee = call->getCalledFunction()) {
      StringRef name = callee->getName();
      return name.starts_with("llvm.nvvm.read.ptx.sreg.");
    }
  }
  return false;
}

bool dependsOnThreadIdx(Value *V, SmallPtrSetImpl<Value *> &visited) {
  if (!V || visited.contains(V)) return false;
  visited.insert(V);

  if (isThreadIdxSource(V)) return true;

  if (Instruction *I = dyn_cast<Instruction>(V)) {
    for (Use &U : I->operands()) {
      if (dependsOnThreadIdx(U.get(), visited)) {
        return true;
      }
    }
  }
  return false;
}

class WarpAnalysis : public PassInfoMixin<WarpAnalysis> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
    outs() << "[WarpAnalysis] Running on module: " << M.getName() << "\n";

    FunctionAnalysisManager &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

    outs() << "[\n";

    bool first = true;
    for (Function &F : M) {
      if (F.isDeclaration() || F.isIntrinsic()) continue;

      DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(F);
      PostDominatorTree &PDT = FAM.getResult<PostDominatorTreeAnalysis>(F);

      int branchCount = 0;
      int divergentBranches = 0;
      int threadDependentBranches = 0;
      int memOps = 0;
      int sharedMemOps = 0;
      int globalMemOps = 0;
      int localMemOps = 0;
      int barrierCalls = 0;

      for (auto &BB : F) {
        auto *term = BB.getTerminator();
        if (auto *br = dyn_cast<BranchInst>(term)) {
          if (br->isConditional()) {
            branchCount++;
            bool pddiv = !PDT.dominates(br->getParent(), br->getSuccessor(0)) ||
                         !PDT.dominates(br->getParent(), br->getSuccessor(1));

            SmallPtrSet<Value *, 8> visited;
            bool threadDep = dependsOnThreadIdx(br->getCondition(), visited);

            if (pddiv || threadDep) {
              divergentBranches++;
              if (threadDep) threadDependentBranches++;
            }
          }
        }

        for (Instruction &I : BB) {
          if (auto *sel = dyn_cast<SelectInst>(&I)) {
            branchCount++;
            SmallPtrSet<Value *, 8> visited;
            if (dependsOnThreadIdx(sel->getCondition(), visited)) {
              divergentBranches++;
              threadDependentBranches++;
            }
          }

          if (auto *call = dyn_cast<CallBase>(&I)) {
            if (Function *callee = call->getCalledFunction()) {
              if (callee->getName().starts_with("llvm.nvvm.barrier")) {
                barrierCalls++;
              }
            }
          }

          if (auto *load = dyn_cast<LoadInst>(&I)) {
            memOps++;
            switch (load->getPointerAddressSpace()) {
              case 3: sharedMemOps++; break;
              case 5: localMemOps++; break;
              default: globalMemOps++; break;
            }
          }

          if (auto *store = dyn_cast<StoreInst>(&I)) {
            memOps++;
            switch (store->getPointerAddressSpace()) {
              case 3: sharedMemOps++; break;
              case 5: localMemOps++; break;
              default: globalMemOps++; break;
            }
          }
        }
      }

      if (!first) outs() << ",\n";
      first = false;

      outs() << "  {\n"
             << "    \"function\": \"" << F.getName() << "\",\n"
             << "    \"branch_count\": " << branchCount << ",\n"
             << "    \"divergent_branches\": " << divergentBranches << ",\n"
             << "    \"thread_dependent\": " << threadDependentBranches << ",\n"
             << "    \"memory_ops\": {\n"
             << "      \"total\": " << memOps << ",\n"
             << "      \"shared\": " << sharedMemOps << ",\n"
             << "      \"local\": " << localMemOps << ",\n"
             << "      \"global\": " << globalMemOps << "\n"
             << "    },\n"
             << "    \"barriers\": " << barrierCalls << "\n"
             << "  }";
    }

    outs() << "\n]\n";
    return PreservedAnalyses::all();
  }
};

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION, "WarpAnalysis", LLVM_VERSION_STRING,
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
