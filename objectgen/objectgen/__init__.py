import attr
import io
from pathlib import Path
from typing import List, Optional
from collections import defaultdict

LICENSE = """
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
"""

Type = str

@attr.s(auto_attribs=True)
class ObjectField:
    field_name: str
    field_type: Type

@attr.s
class ObjectMethod:
    pass

@attr.s(auto_attribs=True)
class ObjectDefinition:
    name: str
    fields: List[ObjectField]
    methods: List[ObjectMethod] = []
    inherits_from: str = "ObjectRef"
    namespace: List[str] = []
    final: bool = True
    docs: str = ""

    def ref_name(self):
        return self.name

    def payload_name(self):
        return self.name + "Node"

    def parent_payload_name(self):
        if self.inherits_from != "ObjectRef":
            return self.inherits_from + "Node"
        return "Object"

    def parent_ref_name(self):
        if self.inherits_from != "ObjectRef":
            return self.inherits_from
        return "ObjectRef"

    def type_key(self):
        return ".".join(self.namespace + [self.name])

@attr.s(auto_attribs=True)
class ObjectGenConfig:
    cpp_include_root: Optional[Path]
    cpp_source_root: Optional[Path]
    python_root: Optional[Path]
    root_namespace: List[str]

@attr.s(auto_attribs=True)
class Generator:
    config: ObjectGenConfig

    def qualified_path(self, defn):
        ns = self.config.root_namespace + defn.namespace
        return tuple(ns)

class CPPGenerator(Generator):
    def header_for(self, ns):
        ns = ns_to_path(ns)
        path = Path(self.config.cpp_include_root.joinpath(ns)).resolve()
        path.parents[0].mkdir(parents=True, exist_ok=True)
        return path.with_suffix(".h")

    def source_for(self, ns):
        ns = ns_to_path(ns)
        path = Path(self.config.cpp_source_root.joinpath(ns)).resolve()
        path.parents[0].mkdir(parents=True, exist_ok=True)
        return path.with_suffix(".cc")

    def generate(self, definitions):
        by_ns = defaultdict(list)

        # Group definitions by namespaces.
        for defn in definitions:
            ns = self.qualified_path(defn)
            by_ns[ns].append(defn)

        # Generate each NS to a set of files.
        for ns in by_ns:
            header = io.StringIO("")
            source = io.StringIO("")

            self.generate_ns(header, source, ns, by_ns[ns])

            # Ensure directory exists.
            header_file = self.header_for(ns)
            source_file = self.source_for(ns)
            print(f"HeaderFile: {header_file}")
            print(f"SourceFile: {source_file}")

            license_str =("\n").join([f"* {line}" for line in LICENSE.splitlines()])
            license_str = f"/{license_str}\n*/"
            with open(header_file, 'w') as file:
                file.seek(0)
                file.truncate()
                file.write(license_str)
                file.write(header.getvalue())

            with open(source_file, 'w') as file:
                file.seek(0)
                file.truncate()
                file.write(license_str)
                file.write("\n")
                file.write(source.getvalue())

    def generate_ns(self, header_buf, source_buf, namespace, defs):
        header_value = "_".join([ns.upper() for ns in namespace])
        header_value = f"TVM_{header_value}_H_"
        header_buf.write("\n")
        header_buf.write(f"#ifndef {header_value}\n")
        header_buf.write(f"#define {header_value}\n")

        includes = [
            "<tvm/ir/span.h>",
            "<tvm/ir/type.h>",
            "<tvm/node/node.h>",
            "<tvm/runtime/container.h>",
            "<tvm/runtime/object.h>",
            "<tvm/relay/expr.h>",
            f"\"{self.header_for(namespace)}\"",
        ]

        source_buf.write("\n")
        header_buf.write("\n")
        for include in includes:
            source_buf.write(f"#include {include}\n")
            header_buf.write(f"#include {include}\n")
        source_buf.write("\n")
        header_buf.write("\n")

        for ns in ["tvm"] + list(namespace):
            header_buf.write(f"namespace {ns} {{ \n")
            source_buf.write(f"namespace {ns} {{ \n")

        header_buf.write("\n")
        source_buf.write("\n")

        for defn in defs:
            self.generate_object_def(header_buf, source_buf, defn)

        for ns in reversed(["tvm"] + list(namespace)):
            header_buf.write(f"}} // namespace {ns} \n")
            source_buf.write(f"}} // namespace {ns} \n")

        header_buf.write(f"#endif  // {header_value}\n")

    def generate_object_def(self, header_buf, source_buf, object_def):
        header_buf.write(f"class {object_def.name};\n")

        if object_def.docs:
            header_buf.write(object_def.docs)
            header_buf.write("\n")

        self.generate_payload_decl(header_buf, object_def)
        self.generate_ref_decl(header_buf, object_def)
        self.generate_impl(source_buf, object_def)

    def generate_payload_decl(self, header_buf, object_def):
        ref = object_def.ref_name()
        payload = object_def.payload_name()
        parent_ref = object_def.parent_ref_name()
        parent_payload = object_def.parent_payload_name()

        header_buf.write(f"class {payload} : public {parent_payload} {{\n")
        header_buf.write(" public:\n")

        for field in object_def.fields:
            header_buf.write(f"{4 * ' '}{field.field_type} {field.field_name};\n")

        header_buf.write(f"{4 * ' '}void VisitAttrs(AttrVisitor* v) {{\n")
        for field in object_def.fields:
            header_buf.write(f"{8 * ' '}v->Visit(\"{field.field_name}\", &{field.field_name});\n")
        header_buf.write(f"{4 * ' '}}}\n")

        # Equality
        header_buf.write(f"{4 * ' '}bool SEqualReduce(const {object_def.payload_name()}* other, SEqualReducer equal) const {{\n")
        header_buf.write(f"{8 * ' '}return")
        for i, field in enumerate(object_def.fields):
            header_buf.write(f" equal({field.field_name}, other->{field.field_name})")
            if i != len(object_def.fields) - 1:
                header_buf.write(" && ")
        header_buf.write(";\n")
        header_buf.write(f"{4 * ' '}}}\n")

        # Hashing
        header_buf.write(f"{4 * ' '}void SHashReduce(SHashReducer hash_reduce) const {{\n")
        for field in object_def.fields:
            header_buf.write(f"{8 * ' '}hash_reduce({field.field_name});\n")
        header_buf.write(f"{4 * ' '}}}\n")

        header_buf.write(f"{4 * ' '}static constexpr const char* _type_key = \"{object_def.type_key()}\";\n")
        header_buf.write(f"{4 * ' '}static constexpr const bool _type_has_method_sequal_reduce = true;\n")
        header_buf.write(f"{4 * ' '}static constexpr const bool _type_has_method_shash_reduce = true;\n")

        if object_def.final:
            macro_name = "TVM_DECLARE_FINAL_OBJECT_INFO"
        else:
            macro_name = "TVM_DECLARE_BASE_OBJECT_INFO"

        header_buf.write(f"{4 * ' '}{macro_name}({object_def.payload_name()}, {parent_payload});\n")

        header_buf.write("};\n\n")

    def generate_ref_decl(self, header_buf, object_def):
        ref = object_def.ref_name()
        payload = object_def.payload_name()
        parent_ref = object_def.parent_ref_name()
        parent_payload = object_def.parent_payload_name()

        header_buf.write(f"class {ref} : public {parent_ref} {{\n")
        header_buf.write(" public:\n")

        header_buf.write(f"{4 * ' '}TVM_DLL {ref}(\n")
        for i, field in enumerate(object_def.fields):
            header_buf.write(f"{8 * ' '}{field.field_type} {field.field_name}")
            if i != len(object_def.fields) - 1:
                header_buf.write(f",\n")
        header_buf.write(f"{4 * ' '});\n")

        header_buf.write(f"{4 * ' '}TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS")

        header_buf.write(f"({ref}, {parent_ref}, {payload});\n")

        header_buf.write("};\n\n")

    def generate_impl(self, source_buf, object_def):
        ref = object_def.ref_name()
        payload = object_def.payload_name()
        parent_ref = object_def.parent_ref_name()
        parent_payload = object_def.parent_payload_name()


        source_buf.write(f"{ref}::{ref}(\n")
        for i, field in enumerate(object_def.fields):
            source_buf.write(f"{4 * ' '}{field.field_type} {field.field_name}")
            if i != len(object_def.fields) - 1:
                source_buf.write(f",\n")
        source_buf.write(f") {{\n")

        source_buf.write(f"{4 * ' '}ObjectPtr<{payload}> n = make_object<{payload}>();\n")

        for field in object_def.fields:
            name = field.field_name
            source_buf.write(f"{4 * ' '}n->{name} = std::move({name});\n")
        source_buf.write(f"{4 * ' '}data_ = std::move(n);\n")
        source_buf.write("}\n\n")

        source_buf.write(f"TVM_REGISTER_NODE_TYPE({payload});\n\n")

        source_buf.write(f"TVM_REGISTER_GLOBAL(\"{object_def.type_key()}\")")
        source_buf.write(f".set_body_typed([]")

        source_buf.write(f"(")
        for i, field in enumerate(object_def.fields):
            source_buf.write(f"{field.field_type} {field.field_name}")
            if i != len(object_def.fields) - 1:
                source_buf.write(f",")
        source_buf.write(f") {{\n")
        source_buf.write(f"{4 * ' '}return {ref}(")

        for i, field in enumerate(object_def.fields):
            source_buf.write(f"{field.field_name}")
            if i != len(object_def.fields) - 1:
                source_buf.write(f",")
        source_buf.write(");\n")
        source_buf.write(f"}});\n\n")




        # return Tuple(fields, span);
        #});

# TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
#     .set_dispatch<TupleNode>([](const ObjectRef& ref, ReprPrinter* p) {
#       auto* node = static_cast<const TupleNode*>(ref.get());
#       p->stream << "Tuple(" << node->fields << ")";
#     });


def ns_to_path(ns):
    return "/".join(ns)

def from_python(config, definitions):
    if config.cpp_include_root and config.cpp_source_root:
        cpp_gen = CPPGenerator(config)
        cpp_gen.generate(definitions)
    elif config.python_root:
        assert False
    else:
        assert False

def in_ns(ns, defs):
    for defn in defs:
        defn.namespace = ns + defn.namespace
    return defs
