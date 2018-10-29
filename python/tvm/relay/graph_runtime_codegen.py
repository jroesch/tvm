"""
A compiler from a Relay expression to TVM's graph runtime.

The compiler is built from a few pieces.

First we define a compiler from a single Relay expression to the
graph langauge. We require the expression to be a function.
The function's parameters correpond to the placeholder/inputs
and model parameters found in the computation graph representation.
The body of the function represents the computation graph.

The compiler's output is a program in a the graph language. The
graph langauge is composed of Node, NodeRef, InputNode, OpNode.
This "little language" represents programs in TVM's graph format.

To connect to the graph runtime, we a printer which converts this
into TVM's JSON format. The resulting string can be loaded by
contrib.graph_runtime or any other TVM runtime comptatible system.

We expose this functionality in compile_to_tvm.
"""

from __future__ import absolute_import
import json
import attr
from . import ir_pass
from .op import Op
from .expr import Var, Function, Call, If, GlobalVar, Constant, Let
from ..build_module import build
from .. contrib import graph_runtime
from .ir_pass import infer_type
from .. import cpu

# @register_node
# class DictAttrs(object):
#     pass


class AbstractExprVisitor(object):
    """A visitor over Expr in Python."""

    # pylint: disable=no-else-return
    def visit(self, expr):
        """Apply the visitor to an expression."""
        if isinstance(expr, Function):
            return self.visit_function(expr)
        elif isinstance(expr, Call):
            return self.visit_call(expr)
        elif isinstance(expr, Let):
            return self.visit_let(expr)
        elif isinstance(expr, Var):
            return self.visit_var(expr)
        elif isinstance(expr, GlobalVar):
            return self.visit_global_var(expr)
        elif isinstance(expr, If):
            return self.visit_if(expr)
        elif isinstance(expr, Tuple):
            return self.visit_tuple(expr)
        elif isinstance(expr, Constant):
            return self.visit_constant(expr)
        else:
            raise Exception(f"warning unhandled case: {type(expr)}")

    def visit_function(self, _):
        raise Exception("Abstract method please implement me.")

    def visit_let(self, _):
        raise Exception("Abstract method please implement me.")

    def visit_call(self, _):
        raise Exception("Abstract method please implement me.")

    def visit_var(self, _):
        raise Exception("Abstract method please implement me.")

    def visit_type(self, typ):
        return typ

    def visit_if(self, _):
        raise Exception("Abstract method please implement me.")

    def visit_tuple(self, _):
        raise Exception("Abstract method please implement me.")

    def visit_constant(self, _):
        raise Exception("Abstract method please implement me.")

    def visit_global_var(self, _):
        raise Exception("Abstract method please implement me.")


class ExprMutator(AbstractExprVisitor):
    """A functional visitor over Expr in Python."""

    def visit_function(self, fn):
        new_body = self.visit(fn.body)
        return Function(
            list(fn.params),
            fn.ret_type, new_body,
            fn.type_params)

    def visit_let(self, let):
        new_var = self.visit(let.var)
        new_val = self.visit(let.value)
        new_body = self.visit(let.body)
        return Let(new_var, new_val, new_body)

    def visit_call(self, call):
        new_fn = self.visit(call.op)
        new_args = [self.visit(arg) for arg in call.args]
        return Call(new_fn, new_args, call.attrs)

    def visit_var(self, var):
        return var

    def visit_global_id(self, global_var):
        return global_var

    def visit_if(self, ite):
        return If(
            self.visit(ite.guard),
            self.visit(ite.true_b),
            self.visit(ite.false_b))

    def visit_tuple(self, tup):
        return Tuple([self.visit(field) for field in tup.fields])

    def visit_constant(self, const):
        return const


@attr.s
class NodeRef(object):
    """A reference to a node, used for constructing the graph."""
    ident = attr.ib()
    index = attr.ib(default=0)
    version = attr.ib(default=0)

    def to_json(self):
        return [self.ident, self.index, self.version]


@attr.s
class Node(object):
    """The base class for nodes in the TVM runtime system graph input."""
    name = attr.ib()
    attrs = attr.ib()
    is_output = attr.ib()

    def to_json(self) -> Any:
        raise Exception("Abstract method, please implement me.")


@attr.s
class InputNode(Node):
    """An input node in the TVM runtime system graph input."""
    name = attr.ib()
    attrs = attr.ib()
    is_output = attr.ib(default=False)

    def to_json(self):
        return {
            "op": "null",
            "name": self.name,
            "inputs": []
        }


@attr.s
class OpNode(Node):
    """An operator node in the TVM runtime system's graph input."""
    op_name = attr.ib()
    inputs = attr.ib()
    op_attrs = attr.ib()
    is_output = attr.ib(default=False)

    def to_json(self):
        attrs = dict.copy(self.op_attrs)
        # Extend ops with extra info.
        attrs['func_name'] = self.op_name
        # When do we flatten?
        attrs['flatten_data'] = "0"
        # Fix me!
        attrs['num_inputs'] = str(len(self.inputs))
        attrs['num_outputs'] = "1"

        return {
            "op": "tvm_op",
            "name": self.name,
            "attrs": attrs,
            "inputs": self.inputs
        }


def shape_to_json(shape):
    return [sh.value for sh in shape]


def from_tensor(typ):
    return (typ.dtype, shape_to_json(typ.shape))


class TVMRTSCompiler(ExprMutator):
    """The compiler from Relay to the TVM runtime system."""
    nodes = attr.ib()
    id_map = attr.ib()

    def __init__(self, env):
        self.nodes = []
        self.id_map = {}
        self.env = env

    def add_node(self, node):
        self.nodes.append(node)
        ident = len(self.nodes) - 1
        return NodeRef(ident)

    def add_binding(self, ident, ref):
        self.id_map[ident] = ref

    def let_bind(self, ident, node):
        ref = self.add_node(node)
        self.add_binding(ident, ref)
        return ref

    def get_node(self, ref):
        return self.nodes[ref.ident]

    def lookup(self, ident):
        return self.id_map[ident]

    def compile(self, func):
        """Compile a single function into a graph."""
        # First we convert all the parameters into input nodes.
        params = func.params

        for param in params:
            dtype, shape = from_tensor(param.type_annotation)
            node = InputNode(f"{param.name_hint}", {
                "shape": shape,
                "dtype": dtype,
            })
            self.let_bind(param, node)

        # Then we compile the body into a graph which can depend
        # on input variables.
        output_ref = self.visit(func.body)

        # Finally we retreive return value of program, which will
        # become our output node.
        self.get_node(output_ref).is_output = True

    def visit_let(self, let):
        """Visit the Let binding, by first traversing its value,
           then setting the metadata on the returned NodeRef.

           Finally visit the body, and return the NodeRef corresponding
           to it.
        """
        ident = let.var
        val = let.value
        body = let.body

        # Need to add type info?
        val_ref = self.visit(val)
        dtype, shape = from_tensor(val.checked_type())
        val_node = self.get_node(val_ref)
        val_node.attrs["dtype"] = dtype
        val_node.attrs["shape"] = shape
        self.add_binding(ident, val_ref)
        return self.visit(body)

    def visit_var(self, var):
        return self.lookup(var)

    def visit_call(self, call):
        """Transform a ::tvm.relay.Call into an operator in the TVM graph."""
        inputs = []
        for arg in call.args:
            inputs.append(self.visit(arg).to_json())

        if isinstance(call.op, Op):
            raise Exception(
                "Operators should be transformed away; try applying" +
                "the fuse_ops transformation to the expression.")
        elif isinstance(call.op, GlobalVar):
            func = self.env[call.op]
        elif isinstance(call.op, Function):
            func = call.op
        else:
            raise Exception(
                "TVM runtime does not support calls to {0}".format(type(call.op)))

        if int(func.attrs.Primitive) != 1:
            raise Exception(
                "TVM only support calls to primitive functions " +
                "(i.e functions composed of fusable operator invocations)")

        op_name = func.attrs.LoweredFunc.name

        attrs = {'shape': shape_to_json(call.checked_type.shape),
                 'dtype': call.checked_type.dtype}
        call_hash = str(ir_pass.structural_hash(call))
        op_node = OpNode("call_" + call_hash, attrs, op_name, inputs, {})
        return self.add_node(op_node)

    def to_json(self):
        """Convert the sequence of nodes stored by the compiler into the
           JSON format defined in: https://docs.tvm.ai/dev/nnvm_json_spec.html.
        """
        nodes = []
        # First we compute "nodes" field.
        for node in self.nodes:
            nodes.append(node.to_json())

        arg_nodes = []
        heads = []
        # Compute "arg_nodes" and "heads" fields.
        for i, node in enumerate(self.nodes):
            if isinstance(node, InputNode):
                arg_nodes.append(i)

            if node.is_output:
                # Need to fix this.
                heads.append(NodeRef(i).to_json())

        # Compute "node_row_ptr".
        # TODO

        # Compute "attrs" field.
        attrs = {}

        # These fields are mandatory.
        shapes = []
        storage_ids = []
        dtype = []
        dltype = []

        for i, node in enumerate(self.nodes):
            storage_ids.append(i)
            shapes.append(node.attrs['shape'])
            if node.attrs['dtype'] == 'float32':
                dtype.append(0)
                dltype.append('float32')

        attrs["shape"] = ["list_shape", shapes]
        attrs["storage_id"] = ["list_int", storage_ids]
        attrs["dtype"] = ["list_int", dtype]
        attrs["dltype"] = ["list_str", dltype]

        json_dict = {
            "nodes": nodes,
            "arg_nodes": arg_nodes,
            "heads": heads,
            "attrs": attrs
        }

        return json.dumps(json_dict)


def compile_to_tvm(env, func, target=None):
    """Compile a single function to the components needed by the
       TVM RTS.
    """
    if target is None:
        target = 'llvm'

    comp = TVMRTSCompiler(env)
    # NB(@jroesch) This creates lowered functions, and generates names for them
    #
    # We need these names to emit the correct graph as these are names of the
    # functions contained in the module.
    lowered_ops = ir_pass.lower_ops(env, func)
    mod = build([lf.lowered_func for lf in lowered_ops], target)

    # Therefore the call to compile must come after.
    comp.compile(func)
    graph_json = comp.to_json()
    return graph_json, mod, None  # params currently isn't supported by API


def evaluate_rts(env, func, *args):
    """
    Corresponding function to tvm.relay.eval.evaluate.

    This function evaluates a Relay expression on the
    TVM graph_runtime.

    Parameters
    ----------
    env: tvm.relay.Environment
        The global environment used.

    expr: tvm.relay.Expr
        The expression to evaluate.

    args: list of tvm.relay.Expr
        The arguments to apply to the expression, only works
        if the expression has a function type.

    Returns
    -------
    value: tvm.NDArray
        The output Tensor produced by evaluating the expression.
    """
    func = infer_type(func, env)
    func = ir_pass.fuse_ops(env, func)
    func = infer_type(func, env)
    graph_json, mod, params = compile_to_tvm(env, func)
    assert params is None
    # Temporary hack for node_row_ptr
    import nnvm
    graph = nnvm.graph.load_json(graph_json)
    gmodule = graph_runtime.create(graph, mod, cpu(0))
    # Create map of inputs.
    inputs = {}
    for i, arg in enumerate(args):
        inputs[func.params[i].name_hint] = arg
    # Set the inputs here.
    gmodule.set_input(**inputs)
    # Run the module, and fetch the output.
    gmodule.run()
    return gmodule.get_output(0)
