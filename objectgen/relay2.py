from pathlib import Path
import objectgen
from objectgen import ObjectGenConfig, ObjectDefinition, ObjectField, in_ns

config = ObjectGenConfig(
    python_root = Path("./python/tvm/"),
    cpp_include_root = Path("./include/"),
    cpp_source_root = Path("./src/"),
    root_namespace = [])

objectgen.from_python(config, in_ns(["relay2", "expr"], [
    ObjectDefinition(
        name="Expr",
        fields=[
            ObjectField("span", "Span")
        ],
        final = False,
    ),
    ObjectDefinition(
        name="Var",
        fields=[
            ObjectField("id", "relay::Id")
        ],
    )
]))
