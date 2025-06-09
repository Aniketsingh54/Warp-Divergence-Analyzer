#include "Utils.h"         // for parseLocation
#include "JsonEmitter.h"
using namespace llvm;

namespace warp {

/**
 * @brief Emits a JSON representation of kernel analysis data.
 *
 * This function generates a JSON object as a string containing various
 * statistics and metadata about a kernel function, such as branch counts,
 * memory operations, barrier usage, and source code locations of divergent
 * branches and memory accesses.
 *
 * @param funcName            Name of the kernel function.
 * @param branchCount         Total number of branches in the function.
 * @param divergentBranches   Number of divergent branches.
 * @param threadDepBranches   Number of thread-dependent branches.
 * @param totalMem            Total number of memory operations.
 * @param sharedMem           Number of shared memory operations.
 * @param localMem            Number of local memory operations.
 * @param globalMem           Number of global memory operations.
 * @param barriers            Number of barrier instructions.
 * @param divLocs             Vector of source locations for divergent branches (as strings).
 * @param memLocs             Vector of source locations for memory accesses (as strings).
 * @return std::string        JSON-formatted string representing the kernel analysis data.
 */
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
                           const std::vector<std::string> &memLocs) {
  std::ostringstream s;
  s << "    {\n";
  s << "      \"function\": \"" << funcName << "\",\n";
  s << "      \"branch_count\": " << branchCount << ",\n";
  s << "      \"divergent_branches\": " << divergentBranches << ",\n";
  s << "      \"thread_dependent\": " << threadDepBranches << ",\n";
  s << "      \"memory_ops\": {\n";
  s << "        \"total\": " << totalMem
    << ", \"shared\": " << sharedMem
    << ", \"local\": " << localMem
    << ", \"global\": " << globalMem << "\n";
  s << "      },\n";
  s << "      \"barriers\": " << barriers << ",\n";
  s << "      \"divergent_branch_locs\": [\n";
  for (size_t i = 0; i < divLocs.size(); ++i) {
    std::string path; int l, c;
    parseLocation(divLocs[i], path, l, c);
    s << "        {\"line\": " << l << ", \"column\": " << c << "}";
    if (i + 1 < divLocs.size()) s << ",";
    s << "\n";
  }
  s << "      ],\n";
  s << "      \"memory_access_locs\": [\n";
  for (size_t i = 0; i < memLocs.size(); ++i) {
    std::string path; int l, c;
    parseLocation(memLocs[i], path, l, c);
    s << "        {\"line\": " << l << ", \"column\": " << c << "}";
    if (i + 1 < memLocs.size()) s << ",";
    s << "\n";
  }
  s << "      ]\n";
  s << "    }";
  return s.str();
}

/**
 * @brief Emits a JSON array representing groups of source files and their associated kernels.
 *
 * This function takes a map where each key is a source file name and the corresponding value
 * is a vector of kernel names associated with that file. It outputs a JSON array to the
 * standard output stream, where each element is an object containing the source file name
 * and an array of its kernels.
 *
 * Example output:
 * [
 *   {
 *     "source_file": "file1.cpp",
 *     "kernels": [
 *       "kernelA",
 *       "kernelB"
 *     ]
 *   },
 *   ...
 * ]
 *
 * @param fileKernelMap A map from source file names to vectors of kernel names.
 */
void emitFileGroups(const std::map<std::string, std::vector<std::string>> &fileKernelMap) {
  // Create the directory once
  mkdir("json", 0777);

  for (auto &p : fileKernelMap) {
    const std::string &sourceFile = p.first;
    const std::vector<std::string> &kernels = p.second;
    size_t lastSlash = sourceFile.find_last_of("/\\");
    std::string baseName = (lastSlash != std::string::npos) ? sourceFile.substr(lastSlash + 1) : sourceFile;

    std::string path = "json/" + baseName + ".json";
    std::ofstream outFile(path);
    if (!outFile.is_open()) {
      llvm::errs() << "Failed to write file: " << path << "\n";
      continue;
    }

    outFile << "[\n";
    for (size_t i = 0; i < kernels.size(); ++i) {
      outFile << kernels[i];
      if (i + 1 < kernels.size()) outFile << ",\n";
    }
    outFile << "\n]\n";
    outFile.close();
    outs() << "[*] Emitted file: " << path << "\n";
  }

}

} // namespace warp