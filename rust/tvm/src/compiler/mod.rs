use std::convert::TryInto;
use std::path::Path;
use std::io::Read;

use once_cell::sync::Lazy;
use thiserror::Error;
use tracing;

use crate::python;
use crate::ir::IRModule;
use crate::runtime::{Function, ObjectRef, Module as RtModule, String, NDArray, map::Map};
use crate::runtime::IsObjectRef;

pub mod graph_rt;

pub(self) static TVM_LOADED: Lazy<Function> = Lazy::new(|| {
    let ver = python::load().unwrap();
    python::import("tvm.relay").unwrap();
    Function::get("tvm.relay.build").unwrap()
});
