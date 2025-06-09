#ifndef LLVM_WARP_JSON_EMITTER_H
#define LLVM_WARP_JSON_EMITTER_H

#include <map>
#include <vector>
#include <string>
#include <sstream>
#include <llvm/Support/raw_ostream.h>

/**
 * @namespace warp
 * @brief Contains utilities for emitting JSON representations of kernel analysis and grouping files by kernels.
 *
 * The warp namespace provides functions to serialize kernel analysis data into JSON format and to emit
 * groupings of files associated with specific kernels. These utilities are intended to assist in the
 * analysis and reporting of kernel characteristics, such as branch divergence and memory usage.
 */
namespace warp {

std::string emitKernelJson(const std::string &funcName,
                           int branchCount,
                           int divergentBranches,
                           int threadDepBranches,
                           int totalMem,
                           int sharedMem,
                           int localMem,
                           int globalMem,
                           int barriers,
                           const std::vector<std::string> &divLocs,
                           const std::vector<std::string> &memLocs);
void emitFileGroups(const std::map<std::string, std::vector<std::string>> &fileKernelMap);

} // namespace warp

#endif // LLVM_WARP_JSON_EMITTER_H