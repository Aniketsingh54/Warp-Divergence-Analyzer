#ifndef LLVM_WARP_ANALYSIS_LOGIC_H
#define LLVM_WARP_ANALYSIS_LOGIC_H

#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"      // BranchInst, LoadInst, StoreInst
#include "llvm/IR/InstIterator.h"      // for-range loop
#include "llvm/Analysis/PostDominators.h"
#include <map>
#include <vector>
#include <string>

/**
 * @namespace warp
 * @brief Contains analysis logic and data structures for WarpInsight.
 *
 * The warp namespace encapsulates functions and types related to the analysis
 * of LLVM IR functions, specifically for mapping file names to kernel names.
 */
namespace warp {

using FileKernelMap = std::map<std::string, std::vector<std::string>>;

void analyzeFunction(llvm::Function &F,
                     llvm::FunctionAnalysisManager &FAM,
                     FileKernelMap &fileKernelMap);

} // namespace warp

#endif // LLVM_WARP_ANALYSIS_LOGIC_H