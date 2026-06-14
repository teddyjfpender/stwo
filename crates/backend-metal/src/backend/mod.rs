mod accumulation;
#[allow(clippy::module_inception)]
mod backend;
mod blake2s;
mod column;
mod fri;
mod jit;
mod line;
mod logup;
mod lookups;
mod poly;
mod quotient;
pub mod zero_copy_bridge;

pub use backend::MetalBackend;
pub use logup::{
    add_opcode_small_logup_inputs, add_opcode_small_logup_inputs_from_cols,
    assert_eq_opcode_logup_inputs, assert_eq_opcode_logup_inputs_from_cols,
    finalize_device_raw_logup, finalize_raw_logup, memory_addr_to_id_logup_inputs,
    memory_logup_inputs, memory_rc_pair_logup, ret_opcode_logup_inputs, MetalRawLogupColumn,
};

pub(crate) use crate::columns::BaseFieldVec as MetalBaseFieldVec;
