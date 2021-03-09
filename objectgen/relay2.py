from pathlib import Path
import objectgen
from objectgen import ObjectGenConfig, ObjectDefinition, in_ns

config = ObjectGenConfig(
    python_root = Path("../python/tvm/"),
    cpp_include_root = Path("../include/"),
    cpp_source_root = Path("../src/"),
    root_namespace = ["relay2"])

objectgen.from_python(config, in_ns(["expr"], [
    ObjectDefinition(
        name="Var",
        fields=[

        ],
        methods=[

        ])
]))
