mod accumulation;
#[allow(clippy::module_inception)]
mod backend;
mod blake2s;
mod column;
mod constraint_eval;
mod field;
mod fri;
mod jit;
mod logup;
mod lookups;
pub mod memory_witness;
mod pointer_vec;
mod poly;
mod quotient;
mod secure_column;

pub use backend::CudaBackend;
pub use logup::finalize_raw_logup;
pub(crate) use pointer_vec::{UploadedDevicePointerVec, UploadedUint32Vec};

pub(crate) use crate::columns::{BaseFieldVec, SecureFieldVec};
