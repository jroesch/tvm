use std::collections::HashMap;
use thiserror::Error;

use tvm::transform::{Pass, PassContext, module_pass, PassInfo};
use tvm::function::Result;
use tvm::ir::module::IRModule;
use tvm::ir::function::BaseFunc;
use tvm::ir::relay::{self, Expr};
use tvm::ir::ty::Type;
use tvm::runtime::object::IsObjectRef;
use tvm::export;

macro_rules! downcast_match {
    ($id:ident; { $($t:ty => $arm:expr $(,)? )+ , else => $default:expr }) => {
        $( if let Ok($id) = $id.clone().downcast::<$t>() { $arm } else )+
        { $default }
    }
}

#[derive(Error, Debug)]
enum TypeError {
    #[error("TIR functions are currently unsupported")]
    TIRUnsupported,
    #[error("an error occurred inside of external TVM code {0}")]
    TVM(#[from] tvm::errors::Error),
}

struct Context<K, V> {
    inner: Vec<HashMap<K, V>>,
}

impl<K: Eq + std::hash::Hash, V> Context<K, V> {
    fn new() -> Self {
        Context { inner: vec![] }
    }

    fn push(&mut self) {
        self.inner.push(HashMap::new())

    }

    fn pop(&mut self) -> Option<HashMap<K, V>> {
        self.inner.pop()
    }

    fn insert(&mut self, key: K, value: V) -> () {
        self.inner.first_mut().unwrap().insert(key, value);
    }

    fn lookup(&mut self, key: &K) -> Option<&V> {
       for scope in (&self.inner).into_iter().rev() {
           match scope.get(key) {
               None => continue,
               Some(ty) => return Some(ty),
           }
       }

       None
    }
}

struct TypeInferencer {
    locals: Context<relay::Var, Type>,
    // TODO(@jroesch): refine the type here?
    local_types: Context<Type, Type>,
}

type TResult<T> = std::result::Result<T, TypeError>;

impl TypeInferencer {
    fn new(module: IRModule) -> Self {
        TypeInferencer {
            locals: Context::new(),
            local_types: Context::new(),
        }
    }

    fn infer_fn(&mut self, func: BaseFunc) -> TResult<Type> {
        downcast_match!(func; {
            relay::Function => self.infer_relay_fn(func),
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

    fn infer_relay_fn(&mut self, func: relay::Function) -> TResult<Type> {
        self.scoped(|infcx| {
            for param in func.params.clone() {
                infcx.declare_param(param.clone())?;
            }

            let (body, body_ty) = infcx.infer_type(func.body.clone().upcast())?;

            Ok(body_ty)
        })
    }


    fn infer_type(&mut self, e: Expr) -> TResult<(Expr, Type)> {
        downcast_match!(e; {
            relay::Var => {
                let ty = self.locals.lookup(&e).unwrap();
                Ok((e.upcast(), ty.clone()))
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
                let (body, body_ty) = self.infer_type(body)?;
                Ok((e.clone().upcast(), body_ty))
            },
            else => { panic!("unsupported case {:?}", e) }
        })
    }
}

fn pass_fn(module: IRModule, ctx: PassContext) -> TResult<IRModule> {
    let mut inferencer = TypeInferencer::new(module.clone());
    inferencer.infer_fn(module.lookup_str("main")?)?;
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
