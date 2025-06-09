#include "AnalysisLogic.h"
#include "Utils.h"
#include "JsonEmitter.h"
using namespace llvm;

namespace warp {

void analyzeFunction(Function &F,
                     FunctionAnalysisManager &FAM,
                     FileKernelMap &fileKernelMap) {
  if (F.isDeclaration() || F.isIntrinsic()) return;

  auto &DT = FAM.getResult<DominatorTreeAnalysis>(F);
  auto &PDT = FAM.getResult<PostDominatorTreeAnalysis>(F);

  int branchCount = 0, divergentBranches = 0, threadDep = 0;
  int memOps = 0, shared = 0, local = 0, global = 0;
  int barriers = 0;
  std::vector<std::string> divLocs, memLocs;
  std::string sourceFile = "unknown";

  for (auto &BB : F) {
    if (auto *br = dyn_cast<BranchInst>(BB.getTerminator())) {
      if (br->isConditional()) {
        ++branchCount;
        bool pd = !PDT.dominates(br->getParent(), br->getSuccessor(0)) ||
                  !PDT.dominates(br->getParent(), br->getSuccessor(1));
        SmallPtrSet<Value *, 8> vis;
        bool td = dependsOnThreadIdx(br->getCondition(), vis);
        if (pd || td) {
          ++divergentBranches;
          if (td) ++threadDep;
          auto loc = getSourceLocation(*br);
          divLocs.push_back(loc);
          if (sourceFile == "unknown")
            sourceFile = loc.substr(0, loc.find(':'));
        }
      }
    }
    for (auto &I : BB) {
      if (auto *sel = dyn_cast<SelectInst>(&I)) {
        ++branchCount;
        SmallPtrSet<Value *, 8> vis;
        if (dependsOnThreadIdx(sel->getCondition(), vis)) {
          ++divergentBranches; ++threadDep;
          auto loc = getSourceLocation(I);
          divLocs.push_back(loc);
          if (sourceFile == "unknown")
            sourceFile = loc.substr(0, loc.find(':'));
        }
      }
      if (auto *call = dyn_cast<CallBase>(&I)) {
        if (auto *c = call->getCalledFunction())
          if (c->getName().starts_with("llvm.nvvm.barrier"))
            ++barriers;
      }
      if (auto *ld = dyn_cast<LoadInst>(&I)) {
        ++memOps; auto loc = getSourceLocation(I);
        memLocs.push_back(loc);
        if (sourceFile == "unknown") sourceFile = loc.substr(0, loc.find(':'));
        switch (ld->getPointerAddressSpace()) {
          case 3: ++shared; break;
          case 5: ++local; break;
          default: ++global; break;
        }
      }
      if (auto *st = dyn_cast<StoreInst>(&I)) {
        ++memOps; auto loc = getSourceLocation(I);
        memLocs.push_back(loc);
        if (sourceFile == "unknown") sourceFile = loc.substr(0, loc.find(':'));
        switch (st->getPointerAddressSpace()) {
          case 3: ++shared; break;
          case 5: ++local; break;
          default: ++global; break;
        }
      }
    }
  }

  std::string json = emitKernelJson(F.getName().str(),
                                    branchCount, divergentBranches,
                                    threadDep, memOps,
                                    shared, local, global,
                                    barriers, divLocs, memLocs);
  fileKernelMap[sourceFile].push_back(json);
}

} // namespace warp