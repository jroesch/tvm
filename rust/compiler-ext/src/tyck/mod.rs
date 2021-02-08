use std::collections::HashMap;

use tvm::transform::{Pass, PassContext, module_pass, PassInfo};
use tvm::ir::module::IRModule;
use tvm::ir::function::BaseFunc;
use tvm::ir::relay::{self, Expr};
use tvm::ir::ty::Type;
use tvm::runtime::object::IsObjectRef;
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

impl TypeInferencer {
    fn new(_module: IRModule) -> Self {
        TypeInferencer {
            locals: Context::new(),
            local_types: Context::new(),
        }
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
                tvm::runtime::array::Array::from_vec(vec![]).unwrap());

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
                panic!("call node {:?}", e);
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
            else => { panic!("unsupported case {:?}", e) }
        })
    }
}

fn pass_fn(mut module: IRModule, _ctx: PassContext) -> TResult<IRModule> {
    let mut inferencer = TypeInferencer::new(module.clone());
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
