import attr
from typing import List, Optional
from collections import defaultdict

Namespace = List[str]
Type = str

# TODO:
# fix header
# normalize ns handling
# generate .gitignore

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
    methods: List[ObjectMethod] = attr.Factory(list)
    inherits_from: str = "ObjectRef"
    namespace: Namespace = attr.Factory(list)
    imports: List[Namespace] = attr.Factory(list)
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

    def ctor_pf(self):
        # this is a temporary hack, need to think about how to clean up ns handling
        return ".".join(self.namespace[:-1] + [self.name])
