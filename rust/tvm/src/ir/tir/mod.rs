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

use tvm_macros::Object;

use super::{PrimExpr, PrimExprNode};
use crate::runtime::{Object, ObjectPtr, ObjectRef};
use crate::DataType;

#[repr(C)]
#[derive(Object)]
#[ref_name = "IntImm"]
#[type_key = "IntImm"]
pub struct IntImmNode {
    pub base: PrimExprNode,
    pub value: i64,
}

impl IntImm {
    pub fn new(value: i64, datatype: DataType) -> IntImm {
        let node = IntImmNode {
            base: PrimExprNode::base::<IntImmNode>(datatype),
            value: value
        };
        IntImm(Some(ObjectPtr::new(node)))
    }

    pub fn to_prim_expr(self) -> PrimExpr {
        unsafe { PrimExpr(std::mem::transmute(self.0)) }
    }
}

#[repr(C)]
#[derive(Object)]
#[ref_name = "Add"]
#[type_key = "tir.Add"]
pub struct AddNode {
    pub base: PrimExprNode,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

impl Add {
    pub fn new(a: PrimExpr, b: PrimExpr, datatype: DataType) -> Add {
        let node = AddNode {
            base: PrimExprNode::base::<AddNode>(datatype),
            a, b
        };
        Add(Some(ObjectPtr::new(node)))
    }

    // TODO(@jroesch): Remove we with subtyping traits.
    pub fn to_prim_expr(self) -> PrimExpr {
        unsafe { PrimExpr(std::mem::transmute(self.0)) }
    }
}

 #[cfg(test)]
 mod tests {
     use super::*;
     use crate::ir::as_text;
     use anyhow::Result;

    #[test]
    fn test_int_imm() -> Result<()> {
        let dt = DataType::new(0, 32, 1);
        let int_imm = IntImm::new(1337, dt);
        let text = as_text(int_imm);
        panic!(text)
    }

     #[test]
     fn test_add() -> Result<()> {
        let dt = DataType::new(0, 32, 1);
        let lhs = IntImm::new(1337, dt.clone());
        let rhs = IntImm::new(1337, dt.clone());
        let add = Add::new(lhs.to_prim_expr(), rhs.to_prim_expr(), dt);
        let text = as_text(add.clone());
        panic!("{}", text);
        //assert!(text.contains("relay.Id"));
        // Ok(())
     }
 }
