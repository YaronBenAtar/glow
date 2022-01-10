/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "SPF2Function.h"

#include "glow/Graph/PlaceholderBindings.h"
#include "glow/Support/Compiler.h"
#include "glow/Support/Memory.h"

using namespace glow;

SPF2Function::SPF2Function(std::unique_ptr<llvm::orc::GlowJIT> JIT,
                         runtime::RuntimeBundle &&runtimeBundle)
    : LLVMCompiledFunction(std::move(JIT), std::move(runtimeBundle)) {}

Error SPF2Function::execute(ExecutionContext *context) {
  return LLVMCompiledFunction::execute(context);
}
