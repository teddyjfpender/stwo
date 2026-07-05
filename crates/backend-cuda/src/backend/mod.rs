mod accumulation;
#[allow(clippy::module_inception)]
mod backend;
mod blake2s;
pub mod blake_witness;
mod column;
mod constraint_eval;
pub mod exec_tables;
mod field;
mod fri;
mod fused_commit;
mod jit;
pub mod jit_witness;
mod logup;
pub mod logup_pairs;
mod lookups;
pub mod memory_witness;
pub mod pedersen_table;
pub mod pedersen_witness;
mod pointer_vec;
mod poly;
mod quotient;
mod secure_column;

pub use backend::CudaBackend;
pub use logup::finalize_raw_logup;
pub(crate) use pointer_vec::{UploadedDevicePointerVec, UploadedUint32Vec};

pub(crate) use crate::columns::{BaseFieldVec, SecureFieldVec};
