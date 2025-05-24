#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/JSON.h"
#include "llvm/Passes/PassBuilder.h"
#include <map>
#include <set>

using namespace llvm;

namespace {

bool isThreadId(Value *V) {
  if (auto *Call = dyn_cast<CallInst>(V)) {
    Function *CalledFunc = Call->getCalledFunction();
    if (CalledFunc && CalledFunc->getName().starts_with("llvm.nvvm.read.ptx.sreg.tid")) {
      return true;
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
  std::map<const Instruction *, unsigned> DivergenceCount;
  unsigned TotalBranches = 0;
  unsigned DivergentBranches = 0;

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    errs() << "Running WarpDivergencePass on function: " << F.getName() << "\n";

    for (auto &BB : F) {
      for (auto &I : BB) {
        if (auto *BI = dyn_cast<BranchInst>(&I)) {
          if (BI->isConditional()) {
            TotalBranches++;
            Value *Cond = BI->getCondition();
            std::set<const Value *> Visited;

            if (dependsOnThreadId(Cond, Visited)) {
              DivergentBranches++;
              DivergenceCount[BI]++;
            }
          }
        }
      }
    }

    // Emit report
    json::Object Report;
    Report["FunctionName"] = F.getName().str();
    Report["TotalBranches"] = TotalBranches;
    Report["DivergentBranches"] = DivergentBranches;
    Report["DivergenceRatio"] =
        TotalBranches ? (double)DivergentBranches / TotalBranches : 0.0;

    json::Array BranchDetails;
    for (auto &entry : DivergenceCount) {
      const Instruction *I = entry.first;
      unsigned count = entry.second;

      json::Object branchObj;
      if (const DebugLoc &Loc = I->getDebugLoc()) {
          unsigned line = Loc.getLine();
          std::string file = Loc->getFilename().str();
          branchObj["Line"] = line;
          branchObj["SourceFile"] = file;
      } else {
          branchObj["Line"] = -1;
          branchObj["SourceFile"] = "no debug info";
      }

      branchObj["DivergenceCount"] = count;
      BranchDetails.push_back(std::move(branchObj));
    }
    Report["Branches"] = std::move(BranchDetails);

    errs() << formatv("{0:2}", json::Value(std::move(Report))) << "\n";

    return PreservedAnalyses::all();
  }
};

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
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
