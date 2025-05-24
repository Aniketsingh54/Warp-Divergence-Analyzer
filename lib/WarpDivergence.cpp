#include "llvm/IR/PassManager.h"          // For PassInfoMixin
#include "llvm/Passes/PassPlugin.h"       // For PassPluginLibraryInfo and registration
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/JSON.h"
#include "llvm/Passes/PassBuilder.h"
#include <map>

using namespace llvm;

namespace {

struct WarpDivergencePass : public PassInfoMixin<WarpDivergencePass> {
  std::map<const Instruction*, unsigned> DivergenceCount;
  unsigned TotalBranches = 0;
  unsigned DivergentBranches = 0;

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    errs() << "Running WarpDivergencePass on function: " << F.getName() << "\n";

    // Fake example: any conditional branch counts as a branch
    for (auto &BB : F) {
      for (auto &I : BB) {
        if (auto *BI = dyn_cast<BranchInst>(&I)) {
          if (BI->isConditional()) {
            TotalBranches++;

            // Simulated logic: detect divergence here (replace with real warp logic)
            bool diverged = detectDivergence(BI);

            if (diverged) {
              DivergentBranches++;
              DivergenceCount[BI]++;
            }
          }
        }
      }
    }

    // At function end, dump JSON report
    json::Object Report;
    Report["FunctionName"] = F.getName().str();
    Report["TotalBranches"] = TotalBranches;
    Report["DivergentBranches"] = DivergentBranches;
    Report["DivergenceRatio"] = TotalBranches ? (double)DivergentBranches / TotalBranches : 0.0;

    json::Array BranchDetails;
    for (auto &entry : DivergenceCount) {
      const Instruction *I = entry.first;
      unsigned count = entry.second;

      json::Object branchObj;
      if (const DebugLoc &Loc = I->getDebugLoc()) {
        unsigned line = Loc.getLine();
        StringRef file = Loc->getFilename();
        branchObj["SourceFile"] = file.str();
        branchObj["Line"] = line;
      } else {
        branchObj["SourceFile"] = "unknown";
        branchObj["Line"] = 0;
      }
      branchObj["DivergenceCount"] = count;
      BranchDetails.push_back(std::move(branchObj));
    }
    Report["Branches"] = std::move(BranchDetails);

    errs() << formatv("{0:2}", json::Value(std::move(Report))) << "\n";

    return PreservedAnalyses::all();
  }

  bool detectDivergence(const BranchInst *BI) {
    // TODO: Insert your real warp divergence detection logic here
    // For demo, randomly treat even instructions as divergent
    return (BI->getOpcode() % 2) == 0;
  }
};

} // namespace

// Pass registration
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
    }
  };
}
