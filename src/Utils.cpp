#include "Utils.h"
using namespace llvm;

namespace warp {

bool isThreadIdxSource(Value *V) {
  if (auto *call = dyn_cast<CallBase>(V)) {
    if (auto *callee = call->getCalledFunction()) {
      return callee->getName().starts_with("llvm.nvvm.read.ptx.sreg.");
    }
  }
  return false;
}

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