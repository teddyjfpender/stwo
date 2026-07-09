mod accumulation;
pub mod aot;
#[allow(clippy::module_inception)]
mod backend;
mod blake2s;
pub mod blake_witness;
mod column;
pub mod commit_graph;
mod constraint_eval;
pub mod decommit_gather;
pub mod device_transcript;
pub mod exec_context;
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
pub mod pcs_driver;
pub mod pedersen_table;
pub mod pedersen_witness;
mod pointer_vec;
mod poly;
pub mod prepared_commit;
pub mod prepared_fri;
pub mod prepared_quotient;
mod quotient;
pub mod relation_graph;
mod secure_column;

pub use backend::CudaBackend;
pub use logup::finalize_raw_logup;
pub(crate) use pointer_vec::{UploadedDevicePointerVec, UploadedUint32Vec};

pub(crate) use crate::columns::{BaseFieldVec, SecureFieldVec};
