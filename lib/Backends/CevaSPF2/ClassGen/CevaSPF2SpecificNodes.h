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
//#ifdef GLOW_WITH_CevaSPF2

BB.newBackendSpecificNode("CevaSPF2MaxSplat")
    .addInput("Input")
    .addResult("Input.getType()")
    .addMember(MemberType::Float, "SplatValue")
    .setDocstring("A Max node with one splat input; CevaSPF2 specific.");

BB.newBackendSpecificNode("CevaSPF2ConvDKKC8")
    .addInput("Input")
    .addInput("Filter")
    .addInput("Bias")
    .addMember(MemberType::VectorUnsigned, "Kernels")
    .addMember(MemberType::VectorUnsigned, "Strides")
    .addMember(MemberType::VectorUnsigned, "Pads")
    .addMember(MemberType::Unsigned, "Group")
    .addResultFromCtorArg()
    .setDocstring("This is a cpu-specific convolution implementation where the "
                  "filter is transposed to the shape [D/8, K, K, C, 8]");

BB.includeBackendSpecificVerification("glow/CevaSPF2SpecificNodesVerification.h");

//#endif // GLOW_WITH_CevaSPF2
