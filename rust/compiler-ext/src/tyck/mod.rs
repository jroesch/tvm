use std::collections::HashMap;

use tvm::transform::{Pass, PassContext, module_pass, PassInfo};
use tvm::ir::module::IRModule;
use tvm::ir::function::BaseFunc;
use tvm::ir::Spanned;
use tvm::ir::relay::{self, Expr};
use tvm::ir::ty::{Type, TensorType, FuncType};
use tvm::runtime::{IsObjectRef};
use tvm::ir::diagnostics::{DiagnosticContext, Diagnostic};
use tvm::export;

pub mod context;
pub mod error;

use self::context::Context;
use self::error::TypeError;

macro_rules! downcast_match {
    ($id:ident; { $($t:ty => $arm:expr $(,)? )+ , else => $default:expr }) => {
        $( if let Ok($id) = $id.clone().downcast::<$t>() { $arm } else )+
        { $default }
    }
}
struct TypeInferencer {
    module: IRModule,
    pass_context: PassContext,
    locals: Context<relay::Var, Type>,
    // TODO(@jroesch): refine the type here?
    local_types: Context<Type, Type>,
}

type TResult<T> = std::result::Result<T, TypeError>;

struct WithType<T>(T, Type);

impl<T> WithType<T> {
    pub fn new(expr: T, ty: Type) -> WithType<T>
    where
        T: IsObjectRef,
        T::Object: AsRef<relay::ExprNode>
    {
        let expr: Expr = expr.upcast();
        let expr = unsafe { expr.write_checked_type(ty.clone()) };
        WithType(expr.downcast::<T>().unwrap(), ty)
    }
}

// #[derive(Hash, PartialEq)]
// struct TypeId(i64);

// enum TypeInformation {

// }

// struct TypeTable {
//     inner: HashMap<TypeId,
// }

use tvm::runtime::Function as TFunction;
use std::convert::TryInto;

// fn get_ftvm_compute(call: relay::Call, op: Op) -> TFunction {
//     // let get_op_attr = TFunction::get("ir.OpGetAttr").unwrap();
//     let get_op_attr = TFunction::get("yolo_altan").unwrap();
//     let res = get_op_attr.invoke(vec![call.into(), op.into(), "FTVMCompute".into()]).unwrap();
//     res.try_into().unwrap()
// }

// TVM_REGISTER_GLOBAL("ir.OpGetAttr").set_body_typed([](Op op, String attr_name) -> TVMRetValue {
//   auto op_map = Op::GetAttrMap<TVMRetValue>(attr_name);
//   TVMRetValue rv;
//   if (op_map.count(op)) {
//     rv = op_map[op];
//   }
//     return rv;
//     });


use tvm::runtime::{Array};

type Shape = Array<tvm::ir::PrimExpr>;
type DType = tvm::runtime::string::String;

fn get_output_shape(call: relay::Call) -> TResult<Vec<(Shape, DType)>> {
    let get_output_shape = TFunction::get("tyck.compute_output_shape").unwrap();
    let res = get_output_shape.invoke(vec![call.into()])?;
    let ty_infos: tvm::runtime::Array<tvm::runtime::ObjectRef> = res.try_into()?;
    let outs = ty_infos
    .into_iter()
    .map(|ty_info| {
        let res =
            ty_info.downcast::<tvm::runtime::Array<tvm::runtime::ObjectRef>>()?;
        let shape = res.get(0)?;
        let dtype = res.get(1)?;
        Ok((shape.downcast()?, dtype.downcast()?))
    }).collect::<TResult<Vec<(Shape, DType)>>>()?;
    Ok(outs)
}

impl TypeInferencer {
    fn new(module: IRModule, pass_context: PassContext) -> Self {
        TypeInferencer {
            module,
            pass_context,
            locals: Context::new(),
            local_types: Context::new(),
        }
    }

    fn diag_ctx(&self) -> DiagnosticContext {
        self.pass_context.diag_ctx.clone()
    }

    fn infer_fn(&mut self, func: BaseFunc) -> TResult<relay::Function> {
        downcast_match!(func; {
            relay::Function => Ok(self.infer_relay_fn(func)?.0),
            else => Err(TypeError::TIRUnsupported)
        })
    }

    fn scoped<F, R>(&mut self, body: F) -> TResult<R>
        where F: FnOnce(&mut Self) -> TResult<R> {
            self.locals.push();
            self.local_types.push();
            body(self)
            // pop the scopes
    }

    fn declare_param(&mut self, var: relay::Var) -> TResult<()> {
        let ty = var.type_annotation.clone();
        self.locals.insert(var, ty);
        Ok(())
    }

    fn infer_relay_fn(&mut self, func: relay::Function) -> TResult<WithType<relay::Function>> {
        self.scoped(|infcx| {
            for param in func.params.clone() {
                infcx.declare_param(param.clone())?;
            }

            let WithType(body, body_ty) = infcx.infer_type(func.body.clone().upcast())?;

            let func = relay::Function::new(
                func.params.clone(),
                body,
                body_ty.clone(),
                tvm::runtime::array::Array::from_vec(vec![]).unwrap(),
                func.span());

            Ok(WithType::new(func, body_ty))
        })
    }

    fn infer_type(&mut self, e: Expr) -> TResult<WithType<Expr>> {
        downcast_match!(e; {
            relay::Var => {
                let ty = self.locals.lookup(&e).unwrap();
                Ok(WithType::new(e.upcast(), ty.clone()))
            },
            relay::Call => {
                let _op = e.op.clone();
                downcast_match!(_op; {
                    relay::Op => {
                        let out = get_output_shape(e.clone())?;
                        println!("{:?}", out);


                        let args = e.args.clone().into_iter().map(|arg| {
                            self.infer_type(arg)
                        }).collect::<TResult<Vec<WithType<Expr>>>>()?;

                        let (args, arg_tys) =
                            args.into_iter().map(|WithType(arg, arg_ty)| {
                                (arg, arg_ty)
                            }).unzip();

                        let args: Vec<Expr> = args;
                        let arg_tys: Vec<Type> = arg_tys;

                        let output_ty = if out.len() == 1 {
                            let (sh, dtype) = out[0].clone();
                            let dtype = dtype.as_str().unwrap();
                            let dtype = std::str::FromStr::from_str(dtype).unwrap();
                            TensorType::new(sh, dtype, e.span())
                        } else {
                            panic!()
                        };

                        let fn_ty = FuncType::new(arg_tys,
                            output_ty.upcast(), vec![], vec![], e.span());
                        Ok(WithType::new(e.upcast(), fn_ty.upcast()))
                    },
                    else => {
                        self.diag_ctx().emit(
                            Diagnostic::bug(e.span().clone()))?;

                        Ok(WithType::new(e.upcast(), Type::null()))
                    }
                })
            },
            relay::Let => {
                let var = e.var.clone();
                let annotated_ty = var.type_annotation.clone();
                let value = e.value.clone();
                let body = e.body.clone();
                let ty = annotated_ty; // todo: unify soon
                self.locals.insert(var.clone(), ty);
                let WithType(body, body_ty) = self.infer_type(body)?;
                Ok(WithType::new(e.clone().upcast(), body_ty))
            },
            else => {
                self.diag_ctx().emit(
                    Diagnostic::bug(e.base.span.clone()))?;
                Ok(WithType::new(e, Type::null()))
            }
        })
    }
}

fn pass_fn(mut module: IRModule, ctx: PassContext) -> TResult<IRModule> {
    let mut inferencer = TypeInferencer::new(module.clone(), ctx);
    let mut updates: HashMap<relay::GlobalVar, relay::Function> = HashMap::new();
    // Jared we should probably figure out how to make this safe?
    //
    // You can still do iterative invalidation here afaict.
    for (global_var, function) in module.functions.clone() {
        println!("Doing {:?}", global_var.name_hint);
        let checked_fn = inferencer.infer_fn(function)?;
        println!("Doing {:?}", checked_fn.clone().upcast::<Expr>().checked_type);
        updates.insert(global_var, checked_fn);
    }

    for (global_var, updated_function) in updates {
        module.add(global_var, updated_function)?;
    }

    Ok(module)
}

fn infer_type() -> Pass {
    let pass_info = PassInfo::new(0, "RustInferType", vec![]).unwrap();
    let mod_func = move |module, ctx| {
        // TODO: can return result here
        pass_fn(module, ctx).unwrap()
    };
    module_pass(mod_func, pass_info).unwrap()
}


export!(infer_type);
