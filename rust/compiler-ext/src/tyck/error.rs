use thiserror::Error;

#[derive(Error, Debug)]
pub enum TypeError {
    #[error("TIR functions are currently unsupported")]
    TIRUnsupported,
    #[error("an error occurred inside of external TVM code {0}")]
    TVM(#[from] tvm::errors::Error),
}
