# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import onnx
import tvm
import tvm.relay

@tvm.register_func("tvm_onnx_import_and_compile")
def onnx_compile(model_string, target, target_host, opt_level, input_shapes):
    model = onnx.load_model_from_string(bytes(model_string))

    input_shapes = {name : shape for (name, shape) in zip([i.name for i in model.graph.input], input_shapes)}

    irmod, params = tvm.relay.frontend.from_onnx(model, input_shapes, opset=11)
    with tvm.relay.build_config(opt_level=opt_level):
        graph, lib, params = tvm.relay.build(irmod, target_host=target_host, target=target, params=params)

    ctx = tvm.context(target, 0)
    m = tvm.contrib.graph_runtime.create(graph, lib, ctx)
    m.set_input(**params)
    return m.module
