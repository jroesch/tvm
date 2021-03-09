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
    methods: List[ObjectMethod]
    namespace: List[str] = []


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
        return ns_to_path(ns)

class CPPGenerator(Generator):

    def generate(self, definitions):
        by_ns = defaultdict(lambda: (io.StringIO(""), io.StringIO("")))
        for defn in definitions:
            path = self.qualified_path(defn)
            print(path)
            (header, source) = by_ns[path]
            self.generate_object_def(header, source, defn)

        for ns in by_ns:
            # Ensure directory exists.
            Path(self.config.cpp_include_root.joinpath(ns)).parents[0].mkdir(parents=True, exist_ok=True)
            Path(self.config.cpp_source_root.joinpath(ns)).parents[0].mkdir(parents=True, exist_ok=True)

            (header, source) = by_ns[ns]
            header_file = self.config.cpp_include_root.joinpath(ns).with_suffix(".h")
            with open(header_file, 'w') as file:
                file.seek(0)
                file.truncate()
                file.write(LICENSE)
                file.write(header.getvalue())

            source_file = self.config.cpp_source_root.joinpath(ns).with_suffix(".cpp")
            with open(source_file, 'w') as file:
                file.seek(0)
                file.truncate()
                file.write(LICENSE)
                file.write(header.getvalue())

    def generate_object_def(self, header_buf, source_buf, object_def):
        header_value = "_".join([ns.upper() for ns in object_def.namespace])
        header_value = f"TVM_{header_value}_H_"
        header_buf.write("\n")
        header_buf.write(f"#ifndef {header_value}\n")
        header_buf.write(f"#define {header_value}\n")
        for ns in ["tvm"] + object_def.namespace:
            header_buf.write(f"namespace {ns} {{ \n")

        for ns in reversed(["tvm"] + object_def.namespace):
            header_buf.write(f"}} // namespace {ns} \n")

        header_buf.write(f"#endif  // {header_value}\n")
        pass

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
