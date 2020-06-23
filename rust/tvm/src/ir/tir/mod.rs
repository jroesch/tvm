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
use crate::runtime::{Object, ObjectPtr, ObjectRef, String as TVMString};
use crate::DataType;

macro_rules! define_node {
    ($name:ident, $node:ident, $ref:literal, $typekey:literal = $($id:ident : $t:ty),*) => {
        #[repr(C)]
        #[derive(Object)]
        #[ref_name = $ref]
        #[type_key = $typekey]
        pub struct $node {
            pub base: PrimExprNode,
            $(pub $id : $t),*
        }

        impl $name {
            pub fn new($($id : $t,)* datatype: DataType) -> $name {
                let base = PrimExprNode::base::<$node>(datatype);
                let node = $node { base, $($id),* };
                $name(Some(ObjectPtr::new(node)))
            }

            // TODO(@jroesch): Remove we with subtyping traits.
            pub fn to_prim_expr(self) -> PrimExpr {
                unsafe { PrimExpr(std::mem::transmute(self.0)) }
            }
        }
    }
}

define_node!(Var, VarNode, "Var", "tir.Var" = name_hint: TVMString);

define_node!(IntImm, IntImmNode, "IntImm", "IntImm" = value: i64);

define_node!(Add, AddNode, "Add", "tir.Add" = a: PrimExpr, b: PrimExpr);
define_node!(Sub, SubNode, "Sub", "tir.Sub" = a: PrimExpr, b: PrimExpr);
define_node!(Mul, MulNode, "Mul", "tir.Mul" = a: PrimExpr, b: PrimExpr);

define_node!(Div, DivNode, "Div", "tir.Div" = a: PrimExpr, b: PrimExpr);
define_node!(Mod, ModNode, "Mod", "tir.Mod" = a: PrimExpr, b: PrimExpr);
define_node!(FloorDiv, FloorDivNode, "FloorDiv", "tir.FloorDiv" = a: PrimExpr, b: PrimExpr);
define_node!(FloorMod, FloorModNode, "FloorMod", "tir.FloorMod" = a: PrimExpr, b: PrimExpr);

define_node!(Min, MinNode, "Min", "tir.Min" = a: PrimExpr, b: PrimExpr);
define_node!(Max, MaxNode, "Max", "tir.Max" = a: PrimExpr, b: PrimExpr);

// the new datatype is in the base expr
define_node!(Cast, CastNode, "Cast", "tir.Cast" = value: PrimExpr);

// renamed base to start to avoid name clash
define_node!(Ramp, RampNode, "Ramp", "tir.Ramp" = start: PrimExpr, stride: PrimExpr, lanes: i32);

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
        assert!(text.contains("1337"));
        Ok(())
    }

     #[test]
     fn test_add() -> Result<()> {
        let dt = DataType::new(0, 32, 1);
        let lhs = IntImm::new(1337, dt.clone());
        let rhs = IntImm::new(1337, dt.clone());
        let add = Add::new(lhs.to_prim_expr(), rhs.to_prim_expr(), dt);
        let text = as_text(add.clone());
        assert!(text.contains("1337"));
        assert!(text.contains("+"));
        Ok(())
     }
 }
