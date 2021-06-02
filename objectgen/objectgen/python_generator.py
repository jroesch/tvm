import io

from collections import defaultdict
from pathlib import Path

from .generator import Generator, ns_to_path, LICENSE

class PythonGenerator(Generator):
    def source_for(self, ns):
        ns = ns_to_path(ns)
        path = Path(self.config.python_root.joinpath(ns)).resolve()
        path.parents[0].mkdir(parents=True, exist_ok=True)
        return path.with_suffix(".py")

    def ffi_for(self, ns):
        ns = ns_to_path(ns)
        path = Path(self.config.python_root.joinpath(ns)).resolve()
        path.parents[0].mkdir(parents=True, exist_ok=True)
        return path.with_suffix(".py")

    def generate(self, definitions):
        by_ns = defaultdict(list)

        # Group definitions by namespaces.
        for defn in definitions:
            ns = self.qualified_path(defn)
            by_ns[ns].append(defn)

        for ns in by_ns:
            ns = ns[:-1]
            ffi_file = self.ffi_for(list(ns) + ["_ffi_api.py"])
            api_ns = ".".join(ns)
            license_str =("\n").join([f"# {line}" for line in LICENSE.splitlines()])

            with self.open_file(ffi_file, 'w') as file:
                print(f"FFI File: {ffi_file}")
                file.seek(0)
                file.truncate()
                file.write(license_str)
                file.write("\nfrom tvm import _ffi\n")
                file.write(f"_ffi._init_api(\"{api_ns}\", __name__)\n")
                file.write("\n")

        # Generate each NS to a set of files.
        for ns in by_ns:
            source = io.StringIO("")

            self.generate_ns(source, ns, by_ns[ns])

            # Ensure directory exists.
            source_file = self.source_for(ns)
            print(f"SourceFile: {source_file}")

            license_str =("\n").join([f"# {line}" for line in LICENSE.splitlines()])

            with self.open_file(source_file, 'w') as file:
                file.seek(0)
                file.truncate()
                file.write(license_str)
                file.write("\n")
                file.write("import tvm._ffi\n")
                file.write("from ..ir.base import Node\n")
                file.write("from . import _ffi_api\n")
                file.write("\n")
                file.write("ObjectRef = Node\n")
                file.write(source.getvalue())

    def generate_ns(self, source_buf, namespace, defs):
        for defn in defs:
            source_buf.write(f"@tvm._ffi.register_object(\"{defn.type_key()}\")\n")
            source_buf.write(f"class {defn.ref_name()}({defn.parent_ref_name()}):\n")
            source_buf.write(f"{4 * ' '}def __init__(self, ")

            for i, field in enumerate(defn.fields):
                source_buf.write(f"{field.field_name}")
                if i != len(defn.fields) - 1:
                    source_buf.write(", ")

            source_buf.write("):\n")
            source_buf.write(f"{8 * ' '}self.__init_handle_by_constructor__(_ffi_api.{defn.ref_name()}, ")

            for i, field in enumerate(defn.fields):
                source_buf.write(f" {field.field_name}")
                if i != len(defn.fields) - 1:
                    source_buf.write(", ")

            source_buf.write(")\n")
            source_buf.write("\n\n")


    def generate_gitignore(self, ns):
        # TODO(@jroesch): unify with above code
        # TODO(@jroesch): the generated .gitignores are kind of broke
        ns = ns_to_path(ns)
        source_path = Path(self.config.python_root.joinpath(ns)).resolve()
        source_path.parents[0].mkdir(parents=True, exist_ok=True)
        source_path = source_path.parents[0]

        source_ignore = source_path.joinpath(".gitignore")

        with open(source_ignore, 'w') as source_ignore:
            for file_name in set(self.generated_files):
                file_to_ignore = file_name.relative_to(source_path)
                source_ignore.write(f"{file_to_ignore}\n")
