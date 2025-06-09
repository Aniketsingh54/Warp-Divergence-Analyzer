#include "Utils.h"
using namespace llvm;

namespace warp {

/**
 * @brief Checks if the given LLVM IR value represents a call to a PTX special register intrinsic.
 *
 * This function determines whether the provided LLVM IR value (`V`) is a call instruction
 * to a function whose name starts with "llvm.nvvm.read.ptx.sreg.", which typically corresponds
 * to reading special registers such as thread or block indices in NVIDIA PTX.
 *
 * @param V The LLVM IR value to check.
 * @return true if `V` is a call to a PTX special register intrinsic, false otherwise.
 */
bool isThreadIdxSource(Value *V) {
  if (auto *call = dyn_cast<CallBase>(V)) {
    if (auto *callee = call->getCalledFunction()) {
      return callee->getName().starts_with("llvm.nvvm.read.ptx.sreg.");
    }
  }
  return false;
}

/**
 * @brief Recursively determines if a given LLVM Value depends on the thread index.
 *
 * This function checks whether the provided Value, or any of its operand dependencies,
 * is ultimately derived from a thread index source (as determined by isThreadIdxSource).
 * It avoids revisiting already-checked Values using the provided visited set to prevent
 * infinite recursion in cyclic graphs.
 *
 * @param V The LLVM Value to analyze.
 * @param visited A set of Values already visited during recursion to avoid cycles.
 * @return true if V depends on the thread index, false otherwise.
 */
bool dependsOnThreadIdx(Value *V, SmallPtrSetImpl<Value *> &visited) {
  if (!V || visited.count(V)) return false;
  visited.insert(V);
  if (isThreadIdxSource(V)) return true;
  if (auto *I = dyn_cast<Instruction>(V)) {
    for (auto &op : I->operands())
      if (dependsOnThreadIdx(op.get(), visited))
        return true;
  }
  return false;
}

/**
 * @brief Retrieves the source code location of a given LLVM instruction.
 *
 * This function extracts debug location information from the provided LLVM
 * Instruction object. If debug information is available, it constructs a
 * string in the format "directory/filename:line:column". If the directory
 * is empty, only the filename is used. If no debug information is present,
 * the function returns "unknown".
 *
 * @param I The LLVM Instruction from which to extract the source location.
 * @return A string representing the source location, or "unknown" if unavailable.
 */
std::string getSourceLocation(const Instruction &I) {
  if (auto DL = I.getDebugLoc()) {
    if (auto *scope = dyn_cast_or_null<DIScope>(DL.getScope())) {
      auto file = scope->getFilename().str();
      auto dir = scope->getDirectory().str();
      return (dir.empty() ? file : dir + "/" + file) + ":" +
             std::to_string(DL.getLine()) + ":" + std::to_string(DL.getCol());
    }
  }
  return "unknown";
}

/**
 * @brief Parses a location string in the format "path:line:column" into its components.
 *
 * Given a string representing a file location (typically formatted as "path:line:column"),
 * this function extracts the file path, line number, and column number.
 * If the input string does not contain two ':' characters, the entire string is assigned to `path`,
 * and both `line` and `column` are set to 0.
 *
 * @param loc The input location string to parse.
 * @param path Reference to a string where the extracted file path will be stored.
 * @param line Reference to an integer where the extracted line number will be stored.
 * @param column Reference to an integer where the extracted column number will be stored.
 */
void parseLocation(const std::string &loc, std::string &path, int &line, int &column) {
  size_t a = loc.find(':'), b = loc.find(':', a + 1);
  if (a == std::string::npos || b == std::string::npos) {
    path = loc; line = column = 0; return;
  }
  path = loc.substr(0, a);
  line = std::stoi(loc.substr(a + 1, b - a - 1));
  column = std::stoi(loc.substr(b + 1));
}

} // namespace warp