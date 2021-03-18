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
import os
import onnx
import tvm
import tvm.relay
import tvm.autotvm as autotvm
import timeit
import numpy as np
import collections

@tvm.register_func("tvm_run_with_benchmark")
def run_with_benchmark(mod):
    run = mod.get_function('run')
    def benchmark(name):
        t = timeit.Timer(lambda: run()).repeat(repeat=5, number=5)
        ts = np.array(t) * 1000
        print("{} benchmark results: {:.2f}ms mean, {:.2f}ms median, {:.2f}ms std".format(
            name, np.mean(ts), np.median(ts), np.std(ts)
        ))
    if os.getenv("AUTOTVM_TUNING_LOG"):
        benchmark("Tuned")
    else:
        benchmark("Baseline")

@tvm.register_func("tvm_onnx_import_and_compile")
def onnx_compile(model_string, target, target_host, opt_level, input_shapes):
    model = onnx.load_model_from_string(bytes(model_string))

    input_mapping = [(name , shape) for (name, shape) in zip([i.name for i in model.graph.input], input_shapes)]
    # Using an ordereddict maintains input ordering.
    shape_dict = collections.OrderedDict(input_mapping)

    irmod, params = tvm.relay.frontend.from_onnx(model, shape_dict, opset=11)
    print(irmod)
    # import ipdb; ipdb.set_trace()
    with tvm.relay.build_config(opt_level=opt_level):
        tuning_logfile = os.getenv("AUTOTVM_TUNING_LOG")
        if tuning_logfile:
            with autotvm.apply_history_best(tuning_logfile):
                # XXX: do not pass parameters to relay.build otherwise they will be inline into the module
                lib = tvm.relay.build(irmod, target_host=target_host, target=target)
        else:
            lib = tvm.relay.build(irmod, target_host=target_host, target=target)

    print(lib.graph_json)
    ctx = tvm.context(target, 0)
    m = tvm.contrib.graph_runtime.GraphModule(lib["default"](ctx))
    # m.set_input(**params)
    return m.module
