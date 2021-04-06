from pathlib import Path
import objectgen
from objectgen import ObjectGenConfig, ObjectDefinition, ObjectField, in_ns

config = ObjectGenConfig(
    python_root = Path("./python/tvm/"),
    cpp_include_root = Path("./include/tvm"),
    cpp_source_root = Path("./src/"),
    root_namespace = [])

objectgen.from_python(config,
in_ns(["relax", "expr"], [], [
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
]))

# objectgen.from_python(config,
# in_ns(["relay2", "expr"], [], [
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
#     ["relay2", "ty"],
#     [["relay2", "expr"]],
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
