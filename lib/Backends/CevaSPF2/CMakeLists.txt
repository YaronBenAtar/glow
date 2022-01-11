add_subdirectory(ClassGen)

include(GlowAddLibjit)
include(GlowSerialize)

# LIBJIT LLVM common source files.
set(LIBJIT_CevaSPF2_SOURCE_FILES)
list(APPEND LIBJIT_CevaSPF2_SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/../../LLVMIRCodeGen/libjit/libjit.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../LLVMIRCodeGen/libjit/libjit_conv.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../LLVMIRCodeGen/libjit/libjit_matmul.cpp
)

# LIBJIT CevaSPF2 specific source files.
list(APPEND LIBJIT_CevaSPF2_SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/libjit_spf2/libjit_spf2.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/libjit_spf2/libjit_spf2_conv.cpp
)

# LIBJIT CevaSPF2 compile options.
set(LIBJIT_CevaSPF2_COMPILE_OPTIONS
  -I${CMAKE_CURRENT_SOURCE_DIR}/../../LLVMIRCodeGen/libjit
  -std=c++14
  #-g
  -O3
  # for SensProF2
  "-I${CEVA_TOOLS_PATH}/cevatools/lib/clang/9.0.1/senspro-unknown-unknown-elf/rtlv1.1.0-fp2-dpfp2/include/c++/v1"
  "-I${CEVA_TOOLS_PATH}/cevatools/lib/clang/9.0.1/senspro-unknown-unknown-elf/rtlv1.1.0-fp2-dpfp2/include"
  
  # --target=senspro500-unknown-unknown-elf
  --target=senspro-elf -mcpu=sensprov1.1.0
  -fglobal-restrict
  -mllvm -cxd-fast-arithmetic=true -ffast-math
  -mvu=1
  -mfp=2 -mdpfp=2 -mnonlinear=1 -mno-vbnn -mno-vhist -mlvu=1 -mfp=2 -mdpfp=2 -mvfp=1 -mvfpsimd
   -mno-vmpyv5 -mno-vmpyext
  -D__NEED_ssize_t

  # When building for a custom target, use this as an example of how to
  # set up cross-compilation correctly. Then, when building bundles,
  # do not forget to specify the -target= option with the same target
  # string. Also, when building bundles, specify the -llvm-compiler option
  # together with -llvm-compiler-opt to specify custom compilation flags for
  # cross-compilation.
  
  # --sysroot=/usr/arm-linux-gnueabihf/
  # -target armv7-neon-linux-gnueabihf
  # -I/usr/arm-linux-gnueabihf/include/c++/7.4.0/arm-linux-gnueabihf/
  # -emit-llvm
  -DDIM_T_32
  -fno-finite-math-only -ffast-math -DFFAST_MATH
  ${LLVMCevaSPF2RuntimeExtraFlags}

  # Include LIBJIT external functions (callbacks).
  -DGLOW_LIBJIT_EXTERNAL_FUNCTIONS
)

# Add CevaSPF2 LIBJIT.
glow_add_libjit(
  NAME "libjit_spf2"
  SOURCE_FILES ${LIBJIT_CevaSPF2_SOURCE_FILES}
  COMPILE_OPTIONS ${LIBJIT_CevaSPF2_COMPILE_OPTIONS}
)

# Add native CevaSPF2 LIBJIT library used for testing.
if (NOT MSVC)
  add_library(CevaSPF2RuntimeNative
              ${LIBJIT_CevaSPF2_SOURCE_FILES})
endif(NOT MSVC)

add_library(CevaSPF2Backend
            ${libjit_spf2_INCLUDE_FILE}
            CevaSPF2Backend.cpp
            CevaSPF2DeviceManager.cpp
            CevaSPF2Factory.cpp
            CevaSPF2Function.cpp
            CevaSPF2LLVMIRGen.cpp
            Transforms.cpp)

target_link_libraries(CevaSPF2Backend
                      PUBLIC
                        Backend
                        Base
                        CodeGen
                        Flags
                        Graph
                        IR
                        IROptimizer
                        IROptimizerPipeline
                        GraphOptimizerPipeline
                        QuantizationBase
                        Runtime
                        LLVMIRCodeGen)

add_dependencies(CevaSPF2Backend libjit_spf2_TARGET)

set(linked_backends ${linked_backends} CevaSPF2Backend PARENT_SCOPE)

# List the object files from "libobj_spf2" directory.
FILE(GLOB objects "${CMAKE_CURRENT_LIST_DIR}/libobj_spf2/*.o")

# Serialize external object files.
glow_serialize(
  INP_FILES ${objects}
  OUT_DIR "${GLOW_BINARY_DIR}/libobj/libobj_spf2"
  OUT_FILE "${GLOW_BINARY_DIR}/libobj/libobj_spf2/spf2ObjectRegistry.h"
  OUT_VAR "spf2ObjectRegistry"
  OUT_TARGET spf2ObjectRegistryTarget
)
add_dependencies(CevaSPF2Backend spf2ObjectRegistryTarget)
include_directories(${GLOW_BINARY_DIR}/libobj/libobj_spf2)
