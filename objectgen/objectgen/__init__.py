import attr
import io
from pathlib import Path
from typing import List, Optional
from collections import defaultdict
from .cpp_generator import CPPGenerator
from .python_generator import PythonGenerator
from .generator import ObjectGenConfig
from .object_def import ObjectDefinition, ObjectField, ObjectMethod

def resolve_parent_fields(definitions):
    parent_map = {}
    for defn in definitions:
        parent_map[defn.name] = defn

    for defn in definitions:
        if defn.inherits_from != "ObjectRef":
            parent_fields = parent_map[defn.inherits_from].fields
            defn.fields = defn.fields + parent_fields

def from_python(config, definitions):
    resolve_parent_fields(definitions)

    if config.cpp_include_root and config.cpp_source_root:
        cpp_gen = CPPGenerator(config)
        cpp_gen.generate(definitions)

    if config.python_root:
        py_gen = PythonGenerator(config)
        py_gen.generate(definitions)

def in_ns(ns, imports, defs):
    for defn in defs:
        defn.namespace = ns + defn.namespace
        defn.imports = imports + defn.imports

    return defs
