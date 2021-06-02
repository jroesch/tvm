import attr
import io
from pathlib import Path
from typing import List, Optional
from collections import defaultdict
from .object_def import *

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

def ns_to_path(ns):
    return "/".join(ns)

@attr.s(auto_attribs=True)
class ObjectGenConfig:
    cpp_include_root: Optional[Path]
    cpp_source_root: Optional[Path]
    python_root: Optional[Path]
    root_namespace: List[str]

@attr.s(auto_attribs=True)
class Generator:
    config: ObjectGenConfig
    generated_files: List[Path] = attr.Factory(list)

    def qualified_path(self, defn):
        ns = self.config.root_namespace + defn.namespace
        return tuple(ns)

    def open_file(self, file_name, mode='w'):
        self.generated_files.append(file_name)
        return open(file_name, mode)

    def generate_gitignore(self, ns):
        raise NotImplementedError()
