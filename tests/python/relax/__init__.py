"""Test suite for Relay Next (Relax)"""
from __future__ import annotations
import tvm
from tvm.relay.base import Id
from tvm.relax import expr, r2

from typing import TypeVar, Generic, Union
from io import StringIO
import numpy
