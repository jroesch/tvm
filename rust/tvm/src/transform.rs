/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

use crate::ir::relay::Function;
use crate::ir::module::IRModule;
use crate::ir::diagnostics::DiagnosticContext;
use crate::runtime::array::Array;
use crate::runtime::map::Map;
use crate::runtime::{
    external,
    function::{self, Result, ToFunction},
    String as TString,
};
use crate::runtime::{Object, ObjectPtr, ObjectRef};

use tvm_macros::Object;

pub type Pass = ObjectRef;

/// PassContext contains the information that a pass can rely on,
/// such as analysis results.

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "PassContext"]
#[type_key = "transform.PassContext"]
pub struct PassContextNode {
    pub base: Object,
    /// The default optimization level.
    pub opt_level: i32,
    /// The list of required passes. */
    pub required_pass: Array<TString>,
    /// The list of disabled passes.
    pub isabled_pass: Array<TString>,
    /// The diagnostic context.
    // mutable Optional<DiagnosticContext> diag_ctx;
    pub diag_ctx: DiagnosticContext,
    /// Pass specific configuration.
    pub config: Map<TString, ObjectRef>,
    // Trace function to be invoked before and after each pass.
    // trace_func: ObjectRef
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "PassInfo"]
#[type_key = "transform.PassInfo"]
pub struct PassInfoNode {
    pub base: Object,
    pub opt_level: i32,
    pub name: TString,
    pub required: Array<TString>,
}

impl PassInfo {
    pub fn new<S: Into<TString>>(opt_level: i32, name: S, required: Vec<String>) -> Result<PassInfo> {
        let required = required.into_iter().map(|name| name.into()).collect();

        let required = Array::from_vec(required)?;

        let node = PassInfoNode {
            base: Object::base::<PassInfoNode>(),
            opt_level,
            name: name.into(),
            required,
        };

        Ok(PassInfo(Some(ObjectPtr::new(node))))
    }
}

external! {
    #[name("relay._transform.MakeFunctionPass")]
    fn create_func_pass(func: function::Function, pass_info: PassInfo) -> Pass;

    #[name("transform.MakeModulePass")]
    fn create_module_pass(func: function::Function, pass_info: PassInfo) -> Pass;
}

pub fn function_pass<F: Fn(Function, IRModule, PassContext) -> Function + 'static>(
    pass_fn: F,
    pass_info: PassInfo,
) -> Result<Pass> {
    let func = pass_fn.to_function();
    create_func_pass(func, pass_info)
}

pub fn module_pass<F: Fn(IRModule, PassContext) -> IRModule + 'static>(
    pass_fn: F,
    pass_info: PassInfo,
) -> Result<Pass> {
    let mod_func = pass_fn.to_function();
    create_module_pass(mod_func, pass_info)
}
