#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/JSON.h"
#include "llvm/Passes/PassBuilder.h"
#include <set>

using namespace llvm;

namespace {

bool isThreadId(Value *V) {
  if (auto *Call = dyn_cast<CallInst>(V)) {
    if (Function *F = Call->getCalledFunction()) {
      auto Name = F->getName();
      return Name == "llvm.nvvm.read.ptx.sreg.tid.x" ||
             Name == "llvm.nvvm.read.ptx.sreg.tid.y" ||
             Name == "llvm.nvvm.read.ptx.sreg.tid.z";
    }
  }
  return false;
}

bool dependsOnThreadId(Value *V, std::set<const Value *> &Visited) {
  if (!V || Visited.count(V)) return false;
  Visited.insert(V);

  if (isThreadId(V)) return true;

  if (auto *Inst = dyn_cast<Instruction>(V)) {
    for (auto &Op : Inst->operands()) {
      if (dependsOnThreadId(Op.get(), Visited)) return true;
    }
  }
  return false;
}

struct WarpDivergencePass : public PassInfoMixin<WarpDivergencePass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    unsigned TotalBranches = 0;
    unsigned DivergentBranches = 0;

    json::Array BranchDetails;

    for (auto &BB : F) {
      for (auto &I : BB) {
        if (auto *BI = dyn_cast<BranchInst>(&I)) {
          if (BI->isConditional()) {
            TotalBranches++;
            std::set<const Value *> Visited;
            if (dependsOnThreadId(BI->getCondition(), Visited)) {
              DivergentBranches++;

              json::Object branchObj;
              if (const DebugLoc &Loc = I.getDebugLoc()) {
                branchObj["Line"] = Loc.getLine();
                branchObj["SourceFile"] = Loc->getFilename().str();
              } else {
                branchObj["Line"] = -1;
                branchObj["SourceFile"] = "no debug info";
              }

              BranchDetails.push_back(std::move(branchObj));
            }
          }
        }
      }
    }

    json::Object Report;
    Report["FunctionName"] = F.getName().str();
    Report["TotalBranches"] = TotalBranches;
    Report["DivergentBranches"] = DivergentBranches;
    Report["DivergenceRatio"] =
        TotalBranches ? (double)DivergentBranches / TotalBranches : 0.0;
    Report["Branches"] = std::move(BranchDetails);

    outs() << formatv("{0:2}\n", json::Value(std::move(Report)));

    return PreservedAnalyses::all();
  }
};

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {
      LLVM_PLUGIN_API_VERSION, "WarpDivergencePass", "v1",
      [](PassBuilder &PB) {
        PB.registerPipelineParsingCallback(
            [](StringRef Name, FunctionPassManager &FPM,
               ArrayRef<PassBuilder::PipelineElement>) {
              if (Name == "warp-divergence") {
                FPM.addPass(WarpDivergencePass());
                return true;
              }
              return false;
            });
      }};
}
