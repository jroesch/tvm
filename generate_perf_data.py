import tvm
from tvm import parser
from tvm.relay.testing.resnet import get_workload



def generate_for_main(network_mod, params):
    def annotate(*args):
        return ""

    network_text = net.astext(annotate=annotate)
    network_mod = parser.parse(network_text)
    network_json = tvm.ir.save_json(network_mod["main"])
    # TODO execute and extract graph debug output
    return network_text, network_json, None

net, params = get_workload()
text, json, graph_debug_output = generate_for_main(net, params)

x = tvm.relay.var('x', shape=(10, 1))
fn = tvm.relay.Function([x], x)
print(tvm.ir.save_json(fn))

import pdb; pdb.set_trace()
