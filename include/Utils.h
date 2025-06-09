#ifndef LLVM_WARP_UTILS_H
#define LLVM_WARP_UTILS_H

#include "llvm/IR/Value.h"
#include "llvm/IR/Instruction.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include <string>

/**
 * @namespace warp
 * @brief Utility functions for analyzing LLVM IR in the context of thread index dependencies and source locations.
 *
 * This namespace provides helper functions to determine if a value is derived from thread indices,
 * analyze dependencies on thread indices, and extract or parse source code location information from LLVM instructions.
 */
namespace warp {

bool isThreadIdxSource(llvm::Value *V);
bool dependsOnThreadIdx(llvm::Value *V, llvm::SmallPtrSetImpl<llvm::Value *> &visited);
std::string getSourceLocation(const llvm::Instruction &I);
void parseLocation(const std::string &loc, std::string &path, int &line, int &column);

} // namespace warp

#endif // LLVM_WARP_UTILS_H