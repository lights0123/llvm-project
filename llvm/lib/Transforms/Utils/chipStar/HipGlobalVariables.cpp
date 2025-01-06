//===- HipGlobalVariables.cpp ---------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// In HIP, __constant__ global scope variables in the device code can be
// accessed via a host API. Features and capabilities include:
//
// * Reading and writing the variables,
// * querying their size,
// * taking addresses of them and
// * passing the taken addresses to kernels (Note: HIP exclusive)
//
// This API can be implemented in Level Zero with zeModuleGetGlobalPointer API
// function. However, OpenCL does not have a corresponding API function. There
// is a non-public Intel extension, clGetDeviceGlobalVariablePointerINTEL, which
// brings the functionality of zeModuleGetGlobalPointer, but an address acquired
// by this API extension and being passed to kernels was tested to not work
// (possibly a bug in Intel’s OpenCL implementation). Nonetheless, use of the
// vendor extension would not be portable. Our implementation is designed such
// that it does not rely on vendor extensions and thus making it portable to
// various OpenCL drivers in addition to LZ. The implementation requires the
// OpenCL 2.0 coarse grained SVM capability from the OpenCL implementation.
//
// In OpenCL, constant objects in global scope are immutable in contrast to HIP
// and CUDA where they can be modified. Using constant global scope objects
// would block us from modifying them. This aspect has been already covered in
// the upstreamed HIP-Clang by mapping HIP/CUDA __constant__ address space
// objects to global address space in OpenCL, or more specifically
// - the CrossWorkGroup address space of the SPIR-V specification.
//
// (c) 2022 Parmance for Argonne National Laboratory
// (c) 2023 chipStar developers
//===----------------------------------------------------------------------===//

#include "HipGlobalVariables.h"

#include "LLVMSPIRV.h"
#include "common.hh"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ReplaceConstant.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

#define DEBUG_TYPE "hip-lower-gv"

using namespace llvm;

namespace {

using GVarMapT = std::map<GlobalVariable *, GlobalVariable *>;
using Const2InstMapT = std::map<Constant *, Instruction *>;
typedef llvm::SmallPtrSet<Function *, 16> FSet;
typedef llvm::SetVector<Function *> OrderedFSet;

// SPIR-V address spaces.
constexpr unsigned SpirvCrossWorkGroupAS = SPIRV_CROSSWORKGROUP_AS;

// Create kernel function stub, returns its return instruction.
static Instruction *createKernelStub(Module &M, StringRef Name,
                                     ArrayRef<Type *> ArgTypes = {}) {
  Function *F = cast<Function>(
      M.getOrInsertFunction(
           Name,
           FunctionType::get(Type::getVoidTy(M.getContext()), ArgTypes, false))
          .getCallee());
  F->setCallingConv(CallingConv::SPIR_KERNEL);
  // HIP-CLang marks kernels hidden. Do the same here for consistency.
  F->setVisibility(GlobalValue::HiddenVisibility);
  assert(F->empty() && "Function name clash?");
  IRBuilder<> B(BasicBlock::Create(M.getContext(), "entry", F));
  return B.CreateRetVoid();
}

// Returns a constant expression rewritten as instructions if needed.
//
// Global variable references found in the GVarMap are replaced with a load from
// the mapped pointer value. New instructions will be added at Builder's
// current insertion point.
static Value *expandConstant(Constant *C, GVarMapT &GVarMap,
                             IRBuilder<> &Builder, Const2InstMapT &InsnCache) {
  if (InsnCache.count(C))
    return InsnCache[C];

  if (isa<ConstantData>(C))
    return C;

  if (isa<ConstantAggregate>(C))
    return C;

  if (auto *GVar = dyn_cast<GlobalVariable>(C)) {
    if (GVarMap.count(GVar)) {
      // Replace with pointer load. All constant expressions depending
      // on this will be rewritten as instructions.
      auto *NewGVar = GVarMap[GVar];
      auto *LD = Builder.CreateLoad(NewGVar->getValueType(), NewGVar);
      InsnCache[GVar] = LD;
      return LD;
    }
    return GVar;
  }

  if (auto *CE = dyn_cast<ConstantExpr>(C)) {
    SmallVector<Value *, 4> Ops; // Collect potentially expanded operands.
    bool AnyOpExpanded = false;
    for (Value *Op : CE->operand_values()) {
      Value *V =
          expandConstant(cast<Constant>(Op), GVarMap, Builder, InsnCache);
      Ops.push_back(V);
      AnyOpExpanded |= !isa<Constant>(V);
    }

    if (!AnyOpExpanded)
      return CE;

    auto *AsInsn = Builder.Insert(CE->getAsInstruction());
    // Replace constant operands with expanded ones.
    for (auto &U : AsInsn->operands())
      U.set(Ops[U.getOperandNo()]);
    InsnCache[CE] = AsInsn;
    return AsInsn;
  }

  llvm_unreachable("Unexpected constant kind.");
}

// Emit a shadow kernel for initialing the global variable.
static void emitGlobalVarInitShadowKernel(Module &M, GlobalVariable *GVar,
                                          GVarMapT GVarMap) {
  // For original global variable in pseudo code:
  //
  //   SomeType Foo = SomeInit;
  //
  // Emit the following shadow kernel in pseudo code:
  //
  //   void <ChipVarInitPrefix>Foo(SomeType *Foo) {
  //     *Foo = SomeInit;
  //   }

  assert(GVar->hasInitializer());

  auto Name = std::string(ChipVarInitPrefix) + GVar->getName().str();
  IRBuilder<> Builder(createKernelStub(
#if LLVM_VERSION_MAJOR > 17
      M, Name, {PointerType::get(M.getContext(), SpirvCrossWorkGroupAS)}));
#else
      M, Name,
      {PointerType::get(GVar->getValueType(), SpirvCrossWorkGroupAS)}));
#endif

  Value *Ptr = Builder.GetInsertBlock()->getParent()->getArg(0);
  Ptr->setName(std::string(ChipVarPrefix) + GVar->getName().str());

  // Initializers are constant expressions. If they have references to a global
  // variables we are going to replace with load instructions so we need to
  // rewrite the constant expression as a sequence of instructions.
  LLVM_DEBUG(dbgs() << "May have runtime constants: " << *GVar << "\n");
  Const2InstMapT Cache;
  Value *Init = expandConstant(GVar->getInitializer(), GVarMap, Builder, Cache);

  // *Foo = SomeInit;
  Builder.CreateStore(Init, Ptr);
}

static bool shouldLower(const GlobalVariable &GVar) {
  if (!GVar.hasName())
    return false;

#if LLVM_VERSION_MAJOR < 19
  if (GVar.getName().startswith(ChipVarPrefix))
#else
  if (GVar.getName().starts_with(ChipVarPrefix))
#endif
    return false; // Already lowered.

  // All host accessible global device variables are marked to be externally
  // initialized and does not have COMDAT (so far).
  if (!GVar.isExternallyInitialized() || GVar.hasComdat())
    return false;

  // String literals get an unnamed_addr attribute, we know by it to
  // skip them.
  if (GVar.hasAtLeastLocalUnnamedAddr())
    return false;

  // Only objects in cross-workgroup address space are considered. LLVM IR
  // straight out from the HIP-Clang does not have objects in constant address
  // space so we don't look for them.
  if (GVar.getAddressSpace() != SpirvCrossWorkGroupAS)
    return false;

  // Catch globals with unexpected attributes.
  assert(!GVar.isThreadLocal());

  return true;
}

// get Function metadata "MDName" and append NN to it
static void appendMD(Function *F, StringRef MDName, MDNode *NN) {
  unsigned MDKind = F->getContext().getMDKindID(MDName);
  MDNode *OldMD = F->getMetadata(MDKind);

  assert(OldMD != nullptr && OldMD->getNumOperands() > 0);

  llvm::SmallVector<llvm::Metadata *, 8> NewMDNodes;
  // copy MDnodes for original args
  for (unsigned i = 0; i < (F->arg_size() - 1); ++i) {
    Metadata *N = cast<Metadata>(OldMD->getOperand(i).get());
    assert(N != nullptr);
    NewMDNodes.push_back(N);
  }
  NewMDNodes.push_back(NN->getOperand(0).get());
  F->setMetadata(MDKind, MDNode::get(F->getContext(), NewMDNodes));
}

static void updateFunctionMD(Function *F, Module &M,
                             PointerType *ArgTypeWithoutAS) {
  // No need to update if the function does not have kernel metadata to begin
  // with. We update the kernel metadata because the consumer of this code may
  // get confused if the metadata is not complete (level-zero is known to
  // crash).
  if (!F->hasMetadata("kernel_arg_addr_space"))
    // Assuming that other kernel metadata kinds are absent if this one is.
    return;

  IntegerType *I32Type = IntegerType::get(M.getContext(), 32);
  MDNode *MD = MDNode::get(
      M.getContext(),
      ConstantAsMetadata::get(ConstantInt::get(I32Type, SPIRV_GENERIC_AS)));
  appendMD(F, "kernel_arg_addr_space", MD);

  MD = MDNode::get(M.getContext(), MDString::get(M.getContext(), "none"));
  appendMD(F, "kernel_arg_access_qual", MD);

  std::string type_str;
  llvm::raw_string_ostream rso(type_str);
  ArgTypeWithoutAS->print(rso);
  std::string res(rso.str());

  MD = MDNode::get(M.getContext(), MDString::get(M.getContext(), res));
  appendMD(F, "kernel_arg_type", MD);
  appendMD(F, "kernel_arg_base_type", MD);

  MD = MDNode::get(M.getContext(), MDString::get(M.getContext(), ""));
  appendMD(F, "kernel_arg_type_qual", MD);
}

static void replaceGVarUsesWith(GlobalVariable *GV, Function *F, Value *Repl) {
  SmallVector<unsigned, 8> OperToReplace;
  for (Function::iterator BB = F->begin(); BB != F->end(); ++BB) {
    for (BasicBlock::iterator i = BB->begin(); i != BB->end(); ++i) {
      //
      // Scan through the operands of this instruction & check for GV
      //
      Instruction *I = &*i;
      OperToReplace.clear();
      for (unsigned index = 0; index < I->getNumOperands(); ++index) {
        if (GlobalVariable *ArgGV =
                dyn_cast<GlobalVariable>(I->getOperand(index))) {
          if (ArgGV == GV)
            OperToReplace.push_back(index);
        }
      }
      for (unsigned index : OperToReplace) {
        I->setOperand(index, Repl);
      }
    }
  }
}

static void recursivelyFindDirectUsers(Value *V, FSet &FS) {
  for (auto U : V->users()) {
    Instruction *Inst = dyn_cast<Instruction>(U);
    if (Inst) {
      Function *IF = Inst->getFunction();
      if (!IF)
        continue;
      FS.insert(IF);
    } else {
      recursivelyFindDirectUsers(U, FS);
    }
  }
}

static void recursivelyFindIndirectUsers(Value *V, OrderedFSet &FS) {
  OrderedFSet Temp;
  for (auto U : V->users()) {
    Instruction *Inst = dyn_cast<Instruction>(U);
    if (Inst) {
      Function *IF = Inst->getFunction();
      if (!IF)
        continue;
      if (FS.count(IF) == 0) {
        FS.insert(IF);
        Temp.insert(IF);
      }
    }
  }
  for (auto F : Temp) {
    recursivelyFindIndirectUsers(F, FS);
  }
}

static bool isGVarUsedInFunction(GlobalVariable *GV, Function *F) {
  for (Function::iterator BB = F->begin(); BB != F->end(); ++BB) {
    for (BasicBlock::iterator i = BB->begin(); i != BB->end(); ++i) {
      //
      // Scan through the operands of this instruction & check for GV
      //
      Instruction *I = &*i;
      for (unsigned index = 0; index < I->getNumOperands(); ++index) {
        if (GlobalVariable *ArgGV =
                dyn_cast<GlobalVariable>(I->getOperand(index))) {
          if (ArgGV == GV)
            return true;
        }
      }
    }
  }
  return false;
}

/* clones a function with an additional argument */
static Function *cloneFunctionWithDynMemArg(Function *F, Module &M,
                                            GlobalVariable *GV) {

  SmallVector<Type *, 8> Parameters;

  // [1024 * float] (value type is not pointer)
  Type *ElemT = GV->getValueType();

  // float addrspace(3)*
  PointerType *AS3_PTR = PointerType::get(ElemT, GV->getAddressSpace());

  for (Function::const_arg_iterator i = F->arg_begin(), e = F->arg_end();
       i != e; ++i) {
    Parameters.push_back(i->getType());
  }
  Parameters.push_back(AS3_PTR);

  // Create the new function.
  FunctionType *FT =
      FunctionType::get(F->getReturnType(), Parameters, F->isVarArg());
  Function *NewF =
      Function::Create(FT, F->getLinkage(), F->getAddressSpace(), "", &M);
  NewF->takeName(F);
  F->setName("old_replaced_func");


  Function::arg_iterator AI = NewF->arg_begin();
  ValueToValueMapTy VV;
  for (Function::const_arg_iterator i = F->arg_begin(), e = F->arg_end();
       i != e; ++i) {
    AI->setName(i->getName());
    VV[&*i] = &*AI;
    ++AI;
  }
  AI->setName(std::string(ChipVarPrefix) + GV->getName());

  SmallVector<ReturnInst *, 1> RI;

#if LLVM_VERSION_MAJOR > 11
  CloneFunctionInto(NewF, F, VV, CloneFunctionChangeType::GlobalChanges, RI);
#else
  CloneFunctionInto(NewF, F, VV, true, RI);
#endif
  IRBuilder<> B(M.getContext());

  // float* (without AS, for MDNode)
  PointerType *AS0_PTR = PointerType::get(ElemT, 0);
  updateFunctionMD(NewF, M, AS0_PTR);

  // insert new function with dynamic mem = last argument
  M.getOrInsertFunction(NewF->getName(), NewF->getFunctionType(),
                        NewF->getAttributes());


  // find all calls/uses of this function...
  std::vector<CallInst *> CallInstUses;
  for (const auto &U : F->users()) {
    CallInst *CI = dyn_cast<CallInst>(U);
    if (CI) {
      CallInstUses.push_back(CI);
    } else {
      llvm_unreachable("unknown instruction - bug");
    }
  }

  // ... and replace them with calls to new function
  for (CallInst *CI : CallInstUses) {
    llvm::SmallVector<Value *, 12> Args;
    Function *CallerF = CI->getCaller();
    assert(CallerF);
    assert(CallerF->arg_size() > 0);
    for (Value *V : CI->args()) {
      Args.push_back(V);
    }
    Argument *LastArg = CallerF->getArg(CallerF->arg_size() - 1);
    Args.push_back(LastArg);
    B.SetInsertPoint(CI);
    CallInst *NewCI = B.CreateCall(FT, NewF, Args);

    CI->replaceAllUsesWith(NewCI);
    CI->eraseFromParent();
  }

  // now we can safely delete the old function
  if (F->getNumUses() != 0)
    llvm_unreachable("old function still has uses - bug!");
  F->eraseFromParent();

  Argument *last_arg = NewF->arg_end();
  --last_arg;

  // if the function uses dynamic shared memory (via the GVar),
  // replace all uses of GVar inside function with the new dyn mem Argument
  if (isGVarUsedInFunction(GV, NewF)) {
    B.SetInsertPoint(NewF->getEntryBlock().getFirstNonPHI());

    replaceGVarUsesWith(GV, NewF, last_arg);
  }

  return NewF;
}

static void breakConstantExprs(const GVarMapT &GVars) {
#if LLVM_VERSION_MAJOR < 17
  for (auto &pair : GVars) {
    GlobalVariable *GV = pair.first;
    for (Value *U : GV->users()) {
      ConstantExpr *CE = dyn_cast<ConstantExpr>(U);
      if (!CE)
        continue;
      SmallVector<Instruction *, 4> IUsers;
      getInstUsers(CE, IUsers);
      for (Instruction *I : IUsers) {
        convertConstantExprsToInstructions(I, CE);
      }
    }
  }
#else
  SmallVector<Constant *, 8> Cnst;
  for (auto &pair : GVars) {
    GlobalVariable *GV = pair.first;
    Constant *CE = cast<Constant>(GV);
    Cnst.push_back(CE);
  }
  convertUsersOfConstantsToInstructions(Cnst);
#endif
}

static void convertGlobalsToParameters(Module &M, const GVarMapT &globals) {
  // Map globals to their new function parameters
  breakConstantExprs(globals);
  for (auto &pair : globals) {
    GlobalVariable *GV = pair.first;
    FSet DirectUserSet;

    // first, find functions that directly use the GVar. However, these may be
    // called from other functions, so we need to append the
    // dynamic shared memory argument recursively.
    recursivelyFindDirectUsers(GV, DirectUserSet);
    if (DirectUserSet.empty()) {
      continue;
    }

    OrderedFSet IndirectUserSet;
    for (Function *F : DirectUserSet) {
      recursivelyFindIndirectUsers(F, IndirectUserSet);
    }

    // find the functions that indirectly use the GVar. These will be processed
    // (cloned with dyn mem arg) before the direct users, so that the direct
    // users can rely on dyn mem argument being present in their caller.
    for (auto FI = IndirectUserSet.rbegin(); FI != IndirectUserSet.rend();
         ++FI) {
      Function *F = *FI;
      Function *NewF = cloneFunctionWithDynMemArg(F, M, GV);
      if (NewF == nullptr)
        llvm_unreachable("cloning failed");
    }

    // now clone the direct users and replace GVar references inside them
    for (Function *F : DirectUserSet) {

      Function *NewF = cloneFunctionWithDynMemArg(F, M, GV);
      if (NewF == nullptr)
        llvm_unreachable("cloning failed");
      // Modified = true;
    }
  }
}

static void eraseMappedGlobalVariables(GVarMapT &GVarMap) {
  for (auto &pair : GVarMap) {
    auto *OldGVar = pair.first;
    OldGVar->replaceAllUsesWith(PoisonValue::get(OldGVar->getType()));
    OldGVar->eraseFromParent();
  }
}

static bool lowerGlobalVariables(Module &M) {
  bool Changed = false;

  // Lower host accessible global device variables.
  GVarMapT GVarMap;
  for (GlobalVariable &GVar : M.globals())
    if (shouldLower(GVar)) {
      GVarMap.insert(std::make_pair(&GVar, nullptr));
    }

  if (!GVarMap.empty()) {
    for (auto Kv : GVarMap) {
      if (Kv.first->hasInitializer())
        emitGlobalVarInitShadowKernel(M, Kv.first, GVarMap);
    }
    convertGlobalsToParameters(M, GVarMap);
    eraseMappedGlobalVariables(GVarMap);
    Changed |= true;
  }

  return Changed;
}
} // namespace

PreservedAnalyses HipGlobalVariablesPass::run(Module &M,
                                              ModuleAnalysisManager &AM) {
  return lowerGlobalVariables(M) ? PreservedAnalyses::none()
                                 : PreservedAnalyses::all();
}

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "hip-lower-gv", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "hip-lower-gv") {
                    MPM.addPass(HipGlobalVariablesPass());
                    return true;
                  }
                  return false;
                });
          }};
}
