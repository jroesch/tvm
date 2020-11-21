use tvm::transform::{Pass, PassContext, module_pass, PassInfo};
use tvm::function::Result;
use tvm::ir::module::IRModule;
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

#[derive(Debug)]
enum TypeError {}

struct TypeInferencer {

}

impl TypeInferencer {
    fn new(module: IRModule) -> Self {
        TypeInferencer {}
    }

    fn infer_type(&mut self, e: Expr) -> std::result::Result<Type, TypeError> {
        downcast_match!(e; {
            relay::Var => { panic!("found var") },
            else => { panic!("unsupported case {:?}", e) }
        })
    }
}

fn the_pass_shit(module: IRModule, ctx: PassContext) -> Result<IRModule> {
    let mut inferencer = TypeInferencer::new(module.clone());
    inferencer.infer_type(module.lookup_str("main").unwrap().upcast()).unwrap();
    Ok(module)
}

fn infer_type() -> Pass {
    let pass_info = PassInfo::new(0, "RustInferType", vec![]).unwrap();
    let mod_func = move |module, ctx| {
        the_pass_shit(module, ctx).unwrap()
    };
    module_pass(mod_func, pass_info).unwrap()
}


export!(infer_type);
