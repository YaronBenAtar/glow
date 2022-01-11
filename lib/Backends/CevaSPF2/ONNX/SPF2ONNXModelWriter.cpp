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

Error ONNXModelWriter::writeCevaSPF2MaxSplat(const CevaSPF2MaxSplatNode *node,
                                        GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "value", node->getSplatValue());

  return writeAllWithNode("CevaSPF2MaxSplat", node, graph, proto);
}

Error ONNXModelWriter::writeCevaSPF2ConvDKKC8(const CevaSPF2ConvDKKC8Node *node,
                                         GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "kernel_shape", node->getKernels());
  addValueAttribute(proto, "strides", node->getStrides());
  addValueAttribute(proto, "pads", node->getPads());
  addValueAttribute(proto, "group", node->getGroup());

  return writeAllWithNode("CevaSPF2ConvDKKC8", node, graph, proto);
}
