#include "Utils.h"         // for parseLocation
#include "JsonEmitter.h"
using namespace llvm;

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

void emitFileGroups(const std::map<std::string, std::vector<std::string>> &fileKernelMap) {
  outs() << "[\n";
  bool firstFile = true;
  for (auto &p : fileKernelMap) {
    if (!firstFile) outs() << ",\n";
    firstFile = false;
    outs() << "  {\n";
    outs() << "    \"source_file\": \"" << p.first << "\",\n";
    outs() << "    \"kernels\": [\n";
    for (size_t i = 0; i < p.second.size(); ++i) {
      outs() << p.second[i];
      if (i + 1 < p.second.size()) outs() << ",\n";
    }
    outs() << "\n    ]\n  }";
  }
  outs() << "\n]\n";
}

} // namespace warp