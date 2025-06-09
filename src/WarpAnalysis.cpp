#include "WarpAnalysis.h"
using namespace llvm;

namespace {

// Helper to check if a value comes from threadIdx or similar
bool isThreadIdxSource(Value *V) {
  if (auto *call = dyn_cast<CallBase>(V)) {
    if (Function *callee = call->getCalledFunction()) {
      StringRef name = callee->getName();
      return name.starts_with("llvm.nvvm.read.ptx.sreg.");
    }
  }
  return false;
}

// Recursively determine if a value depends on threadIdx
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

// Extract source location (DWARF) from instruction
std::string getSourceLocation(const Instruction &I) {
  if (const DebugLoc &DL = I.getDebugLoc()) {
    const MDNode *ScopeNode = DL.getScope();
    const DIScope *Scope = dyn_cast_or_null<DIScope>(ScopeNode);
    if (Scope) {
      std::string file = Scope->getFilename().str();
      std::string dir = Scope->getDirectory().str();
      std::string path = dir.empty() ? file : (dir + "/" + file);
      return path + ":" + std::to_string(DL.getLine()) + ":" + std::to_string(DL.getCol());
    }
  }
  return "unknown";
}                                                         
static void parseLocation(const std::string &loc, std::string &path, int &line, int &column) {
  size_t first_colon = loc.find(':');
  size_t second_colon = loc.find(':', first_colon + 1);
  if (first_colon == std::string::npos || second_colon == std::string::npos) {
    path = loc;
    line = 0;
    column = 0;
    return;
  }
  path = loc.substr(0, first_colon);
  line = std::stoi(loc.substr(first_colon + 1, second_colon - first_colon - 1));
  column = std::stoi(loc.substr(second_colon + 1));
}

class WarpAnalysis : public PassInfoMixin<WarpAnalysis> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
  FunctionAnalysisManager &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  // Map: source_file -> vector of kernel JSON strings
  std::map<std::string, std::vector<std::string>> fileKernelMap;

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

    std::vector<std::string> divergentBranchLocs;
    std::vector<std::string> memoryAccessLocs;

    std::string sourceFile = "unknown";

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
            std::string loc = getSourceLocation(*br);
            divergentBranchLocs.push_back(loc);

            // Save source file
            if (sourceFile == "unknown") {
              size_t pos = loc.find(':');
              if (pos != std::string::npos)
                sourceFile = loc.substr(0, pos);
            }
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
            std::string loc = getSourceLocation(I);
            divergentBranchLocs.push_back(loc);

            if (sourceFile == "unknown") {
              size_t pos = loc.find(':');
              if (pos != std::string::npos)
                sourceFile = loc.substr(0, pos);
            }
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
          std::string loc = getSourceLocation(I);
          memoryAccessLocs.push_back(loc);

          if (sourceFile == "unknown") {
            size_t pos = loc.find(':');
            if (pos != std::string::npos)
              sourceFile = loc.substr(0, pos);
          }

          switch (load->getPointerAddressSpace()) {
            case 3: sharedMemOps++; break;
            case 5: localMemOps++; break;
            default: globalMemOps++; break;
          }
        }

        if (auto *store = dyn_cast<StoreInst>(&I)) {
          memOps++;
          std::string loc = getSourceLocation(I);
          memoryAccessLocs.push_back(loc);

          if (sourceFile == "unknown") {
            size_t pos = loc.find(':');
            if (pos != std::string::npos)
              sourceFile = loc.substr(0, pos);
          }

          switch (store->getPointerAddressSpace()) {
            case 3: sharedMemOps++; break;
            case 5: localMemOps++; break;
            default: globalMemOps++; break;
          }
        }
      }
    }

    // Build JSON for this kernel with arrays of {line, column}
    std::string kernelJson = "    {\n";
    kernelJson += "      \"function\": \"" + F.getName().str() + "\",\n";
    kernelJson += "      \"branch_count\": " + std::to_string(branchCount) + ",\n";
    kernelJson += "      \"divergent_branches\": " + std::to_string(divergentBranches) + ",\n";
    kernelJson += "      \"thread_dependent\": " + std::to_string(threadDependentBranches) + ",\n";
    kernelJson += "      \"memory_ops\": {\n";
    kernelJson += "        \"total\": " + std::to_string(memOps) + ",\n";
    kernelJson += "        \"shared\": " + std::to_string(sharedMemOps) + ",\n";
    kernelJson += "        \"local\": " + std::to_string(localMemOps) + ",\n";
    kernelJson += "        \"global\": " + std::to_string(globalMemOps) + "\n";
    kernelJson += "      },\n";
    kernelJson += "      \"barriers\": " + std::to_string(barrierCalls) + ",\n";

    kernelJson += "      \"divergent_branch_locs\": [\n";
    for (size_t i = 0; i < divergentBranchLocs.size(); ++i) {
      std::string path;
      int line, column;
      parseLocation(divergentBranchLocs[i], path, line, column);
      kernelJson += "        {\"line\": " + std::to_string(line) + ", \"column\": " + std::to_string(column) + "}";
      if (i + 1 != divergentBranchLocs.size()) kernelJson += ",";
      kernelJson += "\n";
    }
    kernelJson += "      ],\n";

    kernelJson += "      \"memory_access_locs\": [\n";
    for (size_t i = 0; i < memoryAccessLocs.size(); ++i) {
      std::string path;
      int line, column;
      parseLocation(memoryAccessLocs[i], path, line, column);
      kernelJson += "        {\"line\": " + std::to_string(line) + ", \"column\": " + std::to_string(column) + "}";
      if (i + 1 != memoryAccessLocs.size()) kernelJson += ",";
      kernelJson += "\n";
    }
    kernelJson += "      ]\n";

    kernelJson += "    }";

    fileKernelMap[sourceFile].push_back(kernelJson);
  }

  // Output JSON array of file groups
  outs() << "[\n";
  bool firstFile = true;
  for (const auto &fileEntry : fileKernelMap) {
    if (!firstFile) outs() << ",\n";
    firstFile = false;

    outs() << "  {\n";
    outs() << "    \"source_file\": \"" << fileEntry.first << "\",\n";
    outs() << "    \"kernels\": [\n";

    bool firstKernel = true;
    for (const auto &kjson : fileEntry.second) {
      if (!firstKernel) outs() << ",\n";
      firstKernel = false;
      outs() << kjson;
    }

    outs() << "\n    ]\n";
    outs() << "  }";
  }
  outs() << "\n]\n";

  return PreservedAnalyses::all();
}

};

} // namespace

// Register the plugin with LLVM
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
