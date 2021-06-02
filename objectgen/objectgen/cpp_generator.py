import io

from collections import defaultdict
from pathlib import Path

from .generator import Generator, ns_to_path, LICENSE

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

    def generate_gitignore(self, ns):
        # TODO(@jroesch): unify with above code
        source_ns = ns_to_path(ns[1:])
        source_path = Path(self.config.cpp_source_root.joinpath(source_ns)).resolve()
        source_path.parents[0].mkdir(parents=True, exist_ok=True)
        source_path = source_path.parents[0]

        header_ns = ns_to_path(ns)
        header_path = Path(self.config.cpp_include_root.joinpath(header_ns)).resolve()
        header_path.parents[0].mkdir(parents=True, exist_ok=True)
        header_path = header_path.parents[0]

        header_ignore = header_path.joinpath(".gitignore")
        source_ignore = source_path.joinpath(".gitignore")

        with open(header_ignore, 'w') as header_ignore:
            with open(source_ignore, 'w') as source_ignore:
                for file_name in self.generated_files:
                    if file_name.suffix == ".h":
                        file_to_ignore = file_name.relative_to(header_path)
                        header_ignore.write(f"{file_to_ignore}\n")
                    elif file_name.suffix == ".cc":
                        file_to_ignore = file_name.relative_to(source_path)
                        source_ignore.write(f"{file_to_ignore}\n")

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
            # NB: We want to map root namespace of any C++ project to src/
            # directory, so we slice the namespace and only pass, all namespaces
            # after the root ns.
            source_file = self.source_for(ns[1:])
            print(f"HeaderFile: {header_file}")
            print(f"SourceFile: {source_file}")

            license_str =("\n").join([f"* {line}" for line in LICENSE.splitlines()])
            license_str = f"/{license_str}\n*/"

            with self.open_file(header_file) as file:
                file.seek(0)
                file.truncate()
                file.write(license_str)
                file.write(header.getvalue())

            with self.open_file(source_file) as file:
                file.seek(0)
                file.truncate()
                file.write(license_str)
                file.write("\n")
                file.write(source.getvalue())

            self.generate_gitignore(ns)


    def generate_ns(self, header_buf, source_buf, namespace, defs):
        header_value = "_".join([ns.upper() for ns in namespace])
        header_value = f"TVM_{header_value}_H_"
        header_buf.write("\n")
        header_buf.write(f"#ifndef {header_value}\n")
        header_buf.write(f"#define {header_value}\n")

        includes = []
        seen = set()

        for defn in defs:
            for imp in defn.imports:
                if tuple(imp) not in seen:
                    header_path = "/".join(imp) + ".h"
                    includes += [f"<{header_path}>"]
                    seen.add(tuple(imp))

        header_include = self.header_for(namespace)
        header_include = header_include.relative_to(self.config.cpp_include_root.resolve())
        source_includes = [f"\"{header_include}\""]

        source_buf.write("\n")
        header_buf.write("\n")

        for include in includes:
            header_buf.write(f"#include {include}\n")

        for include in source_includes:
            source_buf.write(f"#include {include}\n")

        source_buf.write("\n")
        header_buf.write("\n")

        for ns in namespace:
            header_buf.write(f"namespace {ns} {{ \n")
            source_buf.write(f"namespace {ns} {{ \n")

        header_buf.write("\n")
        source_buf.write("\n")

        for defn in defs:
            self.generate_object_def(header_buf, source_buf, defn)

        for ns in reversed(namespace):
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

        self.generate_equal_and_hash(header_buf, object_def)

        header_buf.write(f"{4 * ' '}static constexpr const char* _type_key = \"{object_def.type_key()}\";\n")
        header_buf.write(f"{4 * ' '}static constexpr const bool _type_has_method_sequal_reduce = true;\n")
        header_buf.write(f"{4 * ' '}static constexpr const bool _type_has_method_shash_reduce = true;\n")

        if object_def.final:
            macro_name = "TVM_DECLARE_FINAL_OBJECT_INFO"
        else:
            macro_name = "TVM_DECLARE_BASE_OBJECT_INFO"

        header_buf.write(f"{4 * ' '}{macro_name}({object_def.payload_name()}, {parent_payload});\n")

        header_buf.write("};\n\n")

    def generate_equal_and_hash(self, header_buf, object_def):
        # Equality
        header_buf.write(f"{4 * ' '}bool SEqualReduce(const {object_def.payload_name()}* other, SEqualReducer equal) const {{\n")

        header_buf.write(f"{8 * ' '}return")
        if len(object_def.fields):
            for i, field in enumerate(object_def.fields):
                header_buf.write(f" equal({field.field_name}, other->{field.field_name})")
                if i != len(object_def.fields) - 1:
                    header_buf.write(" && ")
        else:
            header_buf.write(" true")

        header_buf.write(";\n")
        header_buf.write(f"{4 * ' '}}}\n")

        # Hashing
        header_buf.write(f"{4 * ' '}void SHashReduce(SHashReducer hash_reduce) const {{\n")
        for field in object_def.fields:
            header_buf.write(f"{8 * ' '}hash_reduce({field.field_name});\n")
        header_buf.write(f"{4 * ' '}}}\n")

    def generate_ref_decl(self, header_buf, object_def):
        ref = object_def.ref_name()
        payload = object_def.payload_name()
        parent_ref = object_def.parent_ref_name()
        parent_payload = object_def.parent_payload_name()

        header_buf.write(f"class {ref} : public {parent_ref} {{\n")
        header_buf.write(" public:\n")

        if len(object_def.fields):
            self.generate_ctor_decl(header_buf, object_def)

        # TODO(@jroesch): ast nodes should be non-nullable need to fix default ctor issue
        # header_buf.write(f"{4 * ' '}TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS")
        header_buf.write(f"{4 * ' '}TVM_DEFINE_OBJECT_REF_METHODS")

        header_buf.write(f"({ref}, {parent_ref}, {payload});\n")

        header_buf.write("};\n\n")

    def generate_ctor_decl(self, header_buf, object_def):
        ref = object_def.ref_name()
        header_buf.write(f"{4 * ' '}TVM_DLL {ref}(\n")
        for i, field in enumerate(object_def.fields):
            header_buf.write(f"{8 * ' '}{field.field_type} {field.field_name}")
            if i != len(object_def.fields) - 1:
                header_buf.write(f",\n")
        header_buf.write(f"{4 * ' '});\n")

    def generate_impl(self, source_buf, object_def):
        ref = object_def.ref_name()
        payload = object_def.payload_name()
        parent_ref = object_def.parent_ref_name()
        parent_payload = object_def.parent_payload_name()

        if len(object_def.fields):
            self.generate_ctor_impl(source_buf, object_def)

        source_buf.write(f"TVM_REGISTER_NODE_TYPE({payload});\n\n")

        source_buf.write(f"TVM_REGISTER_GLOBAL(\"{object_def.ctor_pf()}\")")
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

# TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
#     .set_dispatch<TupleNode>([](const ObjectRef& ref, ReprPrinter* p) {
#       auto* node = static_cast<const TupleNode*>(ref.get());
#       p->stream << "Tuple(" << node->fields << ")";
#     });
    def generate_ctor_impl(self, source_buf, object_def):
        ref = object_def.ref_name()
        payload = object_def.payload_name()

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
