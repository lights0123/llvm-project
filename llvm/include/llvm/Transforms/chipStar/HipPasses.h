#ifndef LLVM_TRANSFORMS_HIPPASSES_HIPPASSES_H
#define LLVM_TRANSFORMS_HIPPASSES_HIPPASSES_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class HIPPassesPass : public PassInfoMixin<HIPPassesPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

// Pass creation function
std::unique_ptr<Pass> createHIPPassesPass();

} // namespace llvm

#endif // LLVM_TRANSFORMS_HIPPASSES_HIPPASSES_H
