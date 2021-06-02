from pathlib import Path
import objectgen
from objectgen import ObjectGenConfig, ObjectDefinition, ObjectField, in_ns
# from tvm.runtime.object import Object

config = ObjectGenConfig(
    python_root = Path("./python/tvm/"),
    cpp_include_root = Path("./include/tvm"),
    cpp_source_root = Path("./src/"),
    root_namespace = [])

RegName = "tvm::runtime::vm::RegName"
Index = "tvm::runtime::vm::Index"

relax_expr_imports = [
    ["tvm", "ir", "span"],
    ["tvm", "node", "node"],
    ["tvm", "runtime", "container"],
    ["tvm", "runtime", "object"],
    ["tvm", "relay", "expr"],
    ["tvm", "ir", "expr"],
    ["tvm", "tir", "expr"]
]

relax_expr = in_ns(["relax", "expr"], relax_expr_imports, [
    ObjectDefinition(
        name="Type",
        fields=[
            ObjectField("span", "Span")
        ],
        final = False,
    ),
    ObjectDefinition(
        name="Expr",
        fields=[
            ObjectField("span", "Span")
        ],
        final = False,
    ),
    ObjectDefinition(
        name="Var",
        inherits_from="Expr",
        fields=[
            ObjectField("id", "relay::Id"),
            ObjectField("ty", "Optional<Type>"),
        ],
    ),
    ObjectDefinition(
        name="GlobalVar",
        inherits_from="Expr",
        fields=[
            ObjectField("id", "relay::Id"),
            ObjectField("ty", "Optional<Type>"),
        ],
    ),
    ObjectDefinition(
        name="Intrinsic",
        inherits_from="Expr",
        fields=[
            ObjectField("name", "runtime::String"),
        ]
    ),
    ObjectDefinition(
        name="Binding",
        fields=[
            ObjectField("var", "Var"),
            ObjectField("val", "Expr"),
        ]
    ),
    ObjectDefinition(
        name="Let",
        inherits_from="Expr",
        fields=[
            ObjectField("bindings", "runtime::Array<Binding>"),
            ObjectField("body", "Expr"),
        ]
    ),
    ObjectDefinition(
        name="Call",
        inherits_from="Expr",
        fields=[
            ObjectField("fn", "Expr"),
            ObjectField("args", "runtime::Array<Expr>"),
        ]
    ),
    ObjectDefinition(
        name="Function",
        inherits_from="Expr",
        fields=[
            ObjectField("name", "Optional<runtime::String>"),
            ObjectField("params", "runtime::Array<Var>"),
            ObjectField("body", "Expr"),
            ObjectField("ret_type", "Type"),
        ]
    ),
    ObjectDefinition(
        name="BroadcastShape",
        inherits_from="Expr",
        fields=[
            ObjectField("lhs", "Expr"),
            ObjectField("rhs", "Expr"),
        ]
    ),
    ObjectDefinition(
        name="ShapeOf",
        inherits_from="Expr",
        fields=[
            ObjectField("tensor", "Expr"),
        ]
    ),
    ObjectDefinition(
        name="TensorSlice",
        inherits_from="Expr",
        fields=[
            ObjectField("tensor", "Expr"),
            ObjectField("slice", "Array<Expr>")
        ]
    ),
    ObjectDefinition(
        name="Compute",
        inherits_from="Expr",
        fields=[
            ObjectField("out_shape", "Expr"),
            ObjectField("compute_body", "Expr")
        ]
    ),
    ObjectDefinition(
        name="Add",
        inherits_from="Expr",
        fields=[
            ObjectField("lhs", "Expr"),
            ObjectField("rhs", "Expr")
        ]
    ),
    ObjectDefinition(
        name="TIRExpr",
        inherits_from="Expr",
        fields=[
            ObjectField("expr", "PrimExpr")
        ]
    ),
    ObjectDefinition(
        name="Tuple",
        inherits_from="Expr",
        fields=[
            ObjectField("elements", "runtime::Array<Expr>")
        ]
    ),
    ObjectDefinition(
        name="RelayPrimFn",
        inherits_from="Expr",
        fields=[
            ObjectField("elements", "relay::Expr"),
        ]
    ),
    ObjectDefinition(
        name="Dim",
        inherits_from="Type",
        fields=[],
    ),
    ObjectDefinition(
        name="Shape",
        inherits_from="Type",
        fields=[],
    ),
    ObjectDefinition(
        name="Tensor",
        inherits_from="Type",
        fields=[
            ObjectField("shape", "Optional<Expr>"),
            ObjectField("dtype", "Optional<Expr>")
        ]
    ),
])

relax_vm_imports = [
    ["tvm", "ir", "span"],
    ["tvm", "node", "node"],
    ["tvm", "runtime", "container"],
    ["tvm", "runtime", "object"],
    ["tvm", "relay", "expr"],
    ["tvm", "ir", "expr"],
    ["tvm", "tir", "expr"],
    ["tvm", "runtime", "vm"],
]

relax_vm = in_ns(["relax", "vm"], relax_vm_imports, [
    ObjectDefinition(
        name="Instruction",
        fields=[
            ObjectField("span", "Span")
        ],
        final = False,
    ),
    ObjectDefinition(
        name="AllocTensor",
        inherits_from="Instruction",
        fields=[
            ObjectField("storage", RegName),
            ObjectField("offset", Index),
            ObjectField("shape_register", RegName),
            ObjectField("dtype", "DLDataType")
        ]
    ),
])

objectgen.from_python(config, relax_expr + relax_vm)

# struct Instruction {
#   /*! \brief The instruction opcode. */
#   Opcode op;

#   /*! \brief The destination register. */
#   RegName dst;

#   union {

#     struct /* InvokeClosure Operands */ {
#       /*! \brief The register containing the closure. */
#       RegName closure;
#       /*! \brief The number of arguments to the closure. */
#       Index num_closure_args;
#       /*! \brief The closure arguments as an array. */
#       RegName* closure_args;
#     };
#     struct /* Return Operands */ {
#       /*! \brief The register to return. */
#       RegName result;
#     };
#     struct /* Move Operands */ {
#       /*! \brief The source register for a move operation. */
#       RegName from;
#     };
#     struct /* InvokePacked Operands */ {
#       /*! \brief The index into the packed function table. */
#       Index packed_index;
#       /*! \brief The arity of the packed function. */
#       Index arity;
#       /*! \brief The number of outputs produced by the packed function. */
#       Index output_size;
#       /*! \brief The arguments to pass to the packed function. */
#       RegName* packed_args;
#     };
#     struct /* If Operands */ {
#       /*! \brief The register containing the test value. */
#       RegName test;
#       /*! \brief The register containing the target value. */
#       RegName target;
#       /*! \brief The program counter offset for the true branch. */
#       Index true_offset;
#       /*! \brief The program counter offset for the false branch. */
#       Index false_offset;
#     } if_op;
#     struct /* Invoke Operands */ {
#       /*! \brief The function to call. */
#       Index func_index;
#       /*! \brief The number of arguments to the function. */
#       Index num_args;
#       /*! \brief The registers containing the arguments. */
#       RegName* invoke_args_registers;
#     };
#     struct /* LoadConst Operands */ {
#       /* \brief The index into the constant pool. */
#       Index const_index;
#     };
#     struct /* LoadConsti Operands */ {
#       /* \brief The index into the constant pool. */
#       Index val;
#     } load_consti;
#     struct /* Jump Operands */ {
#       /*! \brief The jump offset. */
#       Index pc_offset;
#     };
#     struct /* Proj Operands */ {
#       /*! \brief The register to project from. */
#       RegName object;
#       /*! \brief The field to read out. */
#       Index field_index;
#     };
#     struct /* GetTag Operands */ {
#       /*! \brief The register to project from. */
#       RegName object;
#     } get_tag;
#     struct /* AllocADT Operands */ {
#       /*! \brief The datatype's constructor tag. */
#       Index constructor_tag;
#       /*! \brief The number of fields to store in the datatype. */
#       Index num_fields;
#       /*! \brief The fields as an array. */
#       RegName* datatype_fields;
#     };
#     struct /* AllocClosure Operands */ {
#       /*! \brief The index into the function table. */
#       Index clo_index;
#       /*! \brief The number of free variables to capture. */
#       Index num_freevar;
#       /*! \brief The free variables as an array. */
#       RegName* free_vars;
#     };
#     struct /* AllocStorage Operands */ {
#       /*! \brief The size of the allocation. */
#       RegName allocation_size;
#       /*! \brief The alignment of the allocation. */
#       Index alignment;
#       /*! \brief The hint of the dtype. */
#       DLDataType dtype_hint;
#       /*! \brief The device type of the allocation. */
#       Index device_type;
#     } alloc_storage;
#     struct /* ShapeOf Operands */ {
#       RegName tensor;
#     } shape_of;
#     struct /* ReshapeTensor Operands */ {
#       RegName tensor;
#       RegName newshape;
#     } reshape_tensor;
#     struct /* DeviceCopy Operands */ {
#       RegName src;
#       /*! \brief The source device type. */
#       Index src_device_type;
#       /*! \brief The destination device type. */
#       Index dst_device_type;
#     };
#   };

# objectgen.from_python(config,
# in_ns(["relax", "expr"], [], [
#     ObjectDefinition(
#         name="Type",
#         fields=[
#             ObjectField("span", "Span")
#         ],
#         final = False,
#     ),
#     ObjectDefinition(
#         name="Expr",
#         fields=[
#             ObjectField("span", "Span")
#         ],
#         final = False,
#     ),
#     ObjectDefinition(
#         name="Var",
#         inherits_from="Expr",
#         fields=[
#             ObjectField("id", "relay::Id"),
#             ObjectField("ty", "Type"),
#         ],
#     ),
#     ObjectDefinition(
#         name="Let",
#         inherits_from="Expr",
#         fields=[
#             ObjectField("bindings", "runtime::Array<Var>"),
#             ObjectField("body", "Expr"),
#         ]
#     ),
#     ObjectDefinition(
#         name="Function",
#         inherits_from="Expr",
#         fields=[
#             ObjectField("name", "runtime::String"),
#             ObjectField("params", "runtime::Array<Var>"),
#             ObjectField("body", "Expr"),
#             ObjectField("ret_type", "Type"),
#         ]
#     ),
#     ObjectDefinition(
#         name="BroadcastShape",
#         inherits_from="Expr",
#         fields=[
#             ObjectField("lhs", "Expr"),
#             ObjectField("rhs", "Expr"),
#         ]
#     ),
# ]) + in_ns(
#     ["relax", "ty"],
#     [["relax", "expr"]],
#     [ObjectDefinition(
#         name="Dim",
#         inherits_from="Type",
#         fields=[],
#     ),
#     ObjectDefinition(
#         name="Shape",
#         inherits_from="Type",
#         fields=[],
#     ),
#     ObjectDefinition(
#         name="Tensor",
#         inherits_from="Type",
#         fields=[
#             ObjectField("shape", "Optional<expr::Expr>"),
#             ObjectField("dtype", "Optional<expr::Expr>")
#         ]
#     ),
# ]))
