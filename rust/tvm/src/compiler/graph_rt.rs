use std::convert::TryInto;
use std::path::Path;
use std::io::Read;

use thiserror::Error;

use crate::ir::IRModule;
use crate::runtime::{Function, ObjectRef, Module as RtModule, String, NDArray, map::Map};

use super::TVM_LOADED;

#[derive(Error, Debug)]
pub enum Error {
    #[error("{0}")]
    IO(#[from] std::io::Error),
    #[error("{0}")]
    TVM(#[from] crate::errors::Error),
}

// Raw API
type RelayBuildFn =
    dyn Fn(IRModule, String, String, Map<String, NDArray>, String);

fn _compile_module(module: IRModule, target: String, target_host: String, params: Map<String, NDArray>, module_name: String) -> Result<RtModule, Error> {
    let module = TVM_LOADED.invoke(vec![module.into(), target.into(), target_host.into(), params.into(), module_name.into()])?;
    let module: RtModule = module.try_into().unwrap();
    Ok(module)
}

#[derive(Debug)]
pub struct CompilerConfig {
    pub target: Option<String>,
    pub target_host: Option<String>,
    pub params: Map<String, NDArray>,
    pub module_name: Option<String>
}

impl Default for CompilerConfig {
    fn default() -> Self {
        CompilerConfig {
            target: None,
            target_host: None,
            params: Map::empty(),
            module_name: None,
        }
    }
}

/// Compile a module from a configuration and IRModule.
///
/// # Arguments
///
/// * `config` - The configuration for the compiler.
/// * `module` - The IRModule to compile.
pub fn compile_module(config: CompilerConfig, module: IRModule) -> Result<RtModule, Error> {
    let target = config.target.unwrap_or("llvm".into());
    _compile_module(module, target, "llvm".into(), Map::<String, NDArray>::empty(), "foo".into())
}

/// The interface to the compile subcommand.
pub fn compile_to_module<P1, P2>(config: CompilerConfig, input_module: P1, output_module: P2) -> Result<(), Error>
where P1: AsRef<Path>, P2: AsRef<Path> {
    let mut input_file =
        std::fs::File::open(input_module.as_ref())?;
    let mut input_module_text = std::string::String::new();
    input_file.read_to_string(&mut input_module_text)?;
    let input_module = IRModule::parse("name", input_module_text)?;
    println!("Before");
    // let rt_module = compile_module(config, input_module)?;
    // println!("Pointer {:p}\n", rt_module.handle() );
    // let output_file =
    //     std::fs::File::open(output_module.as_ref())?;
    // panic!()
    Ok(())
}
