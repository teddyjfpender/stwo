use super::*;

const P: u64 = 2_147_483_647;

fn batch(index: u32, first: usize, columns: usize, log: u32) -> DirectRetainedB2nBatchPlan {
    DirectRetainedB2nBatchPlan {
        batch_index: index,
        source_log_size: log - 1,
        retained_log_size: log,
        canonical_columns: (first..first + columns).collect(),
        pointer_words: columns * POINTER_WORDS,
    }
}

fn lde_absorb(batch: &DirectRetainedB2nBatchPlan) -> [CompactDomainStep; 2] {
    let first = batch.canonical_columns[0] as u32;
    let columns = batch.canonical_columns.len() as u32;
    [
        CompactDomainStep {
            operation: CompactDomainOperation::LdeBatch {
                batch_index: batch.batch_index,
                first_column: first,
                columns,
                log_size: batch.retained_log_size,
            },
            traffic: CommitProgramTraffic::default(),
        },
        CompactDomainStep {
            operation: CompactDomainOperation::AbsorbDomainBatch {
                batch_index: batch.batch_index,
                first_column: first,
                columns,
                log_size: batch.retained_log_size,
                absorbed_columns_before: first,
                initializes_state: first == 0,
                reconstructed_tail: (first != 0).then_some(CompactDomainTail {
                    first_column: first
                        - stwo_backend_cuda_kernels::raw::blake2s_compact_tail_words(first),
                    columns: stwo_backend_cuda_kernels::raw::blake2s_compact_tail_words(first),
                }),
                leaf_compressions: 0,
            },
            traffic: CommitProgramTraffic::default(),
        },
    ]
}

#[test]
fn exact_pairs_compile_in_order_and_receipt_is_byte_exact() {
    let batches = [batch(0, 0, 16, 13), batch(1, 16, 16, 14)];
    let mut steps = Vec::new();
    steps.extend(lde_absorb(&batches[0]));
    steps.push(CompactDomainStep {
        operation: CompactDomainOperation::StateExpandInPlace {
            from_log_size: 13,
            to_log_size: 14,
            absorbed_columns: 16,
            bands: 1,
        },
        traffic: CommitProgramTraffic::default(),
    });
    steps.extend(lde_absorb(&batches[1]));
    steps.push(CompactDomainStep {
        operation: CompactDomainOperation::FinalizeInPlace {
            log_size: 14,
            absorbed_columns: 32,
            reconstructed_tail: CompactDomainTail {
                first_column: 16,
                columns: 16,
            },
            leaf_compressions: 32,
        },
        traffic: CommitProgramTraffic::default(),
    });
    let program = DirectCompactTerminalProgram::compile_steps(&steps, &batches).unwrap();
    assert_eq!(program.operations.len(), 4);
    assert_eq!(program.receipt.separate_absorb_launches_removed, 2);
    assert_eq!(program.receipt.fixed_terminal_launches, 2);
    assert_eq!(program.receipt.net_cuda_launches_removed, 2);
    let expected = ((1 << 13) * 16 + (1 << 14) * 16) * core::mem::size_of::<u32>();
    assert_eq!(
        program.receipt.separate_absorb_reread_bytes_removed,
        expected as u64
    );
    assert_eq!(program.receipt.net_read_bytes_removed, expected as u64);
    assert_eq!(program.receipt.net_device_bytes_removed, expected as u64);
    assert_eq!(program.receipt.compact_tail_reread_bytes_added, 0);
    assert_eq!(
        program.receipt.canonical_retained_write_bytes_before,
        program.receipt.canonical_retained_write_bytes_after
    );
    assert!(program.receipt.merkle_suffix_unchanged);
    assert!(!program.receipt.same_gpu_timing_credit_applied);
}

#[test]
fn fixed16_hybrid_is_selected_only_when_exact_traffic_stays_positive() {
    let favorable = batch(1, 5, 19, 13);
    let mode = batch_mode(5, 13, 19);
    assert_eq!(
        mode,
        DirectCompactTerminalBatchMode::Fixed16Hybrid {
            fixed_columns: 16,
            tiles: 1,
            generic_remainder_columns: 3,
        }
    );
    let receipt = batch_receipt(&favorable, mode).unwrap();
    let column_bytes = (1u64 << 13) * core::mem::size_of::<u32>() as u64;
    assert_eq!(receipt.terminal_prefinal_read_bytes_added, 3 * column_bytes);
    assert_eq!(receipt.compact_tail_reread_bytes_added, 5 * column_bytes);
    assert_eq!(receipt.net_read_bytes_removed, 11 * column_bytes);
    assert_eq!(receipt.net_device_bytes_removed, 8 * column_bytes);

    assert_eq!(
        batch_mode(0, 13, 31),
        DirectCompactTerminalBatchMode::Materialized
    );
    assert_eq!(
        batch_mode(0, 12, 16),
        DirectCompactTerminalBatchMode::Materialized
    );
}

#[test]
fn nonadjacent_absorb_bad_lift_log_and_width_fail_closed() {
    let plan = batch(0, 0, 1, 4);
    let mut steps = lde_absorb(&plan).to_vec();
    steps.swap(0, 1);
    assert!(matches!(
        DirectCompactTerminalProgram::compile_steps(&steps, &[plan.clone()]),
        Err(DirectCompactTerminalError::Fallback(
            DirectCompactTerminalFallbackReason::NonAdjacentAbsorb
        ))
    ));

    let bad_log = batch(0, 0, 1, 3);
    assert!(matches!(
        DirectCompactTerminalProgram::compile_steps(&lde_absorb(&bad_log), &[bad_log]),
        Err(DirectCompactTerminalError::Fallback(
            DirectCompactTerminalFallbackReason::UnsupportedLogSize(3)
        ))
    ));

    assert!(matches!(
        admit_batch(
            4,
            accounting::MAX_TERMINAL_BATCH_COLUMNS + 1,
            Default::default(),
        ),
        Err(DirectCompactTerminalError::Fallback(
            DirectCompactTerminalFallbackReason::UnsupportedBatchWidth(_)
        ))
    ));

    let prior = batch(0, 0, 1, 6);
    let current = batch(1, 1, 1, 5);
    let mut steps = lde_absorb(&prior).to_vec();
    steps.extend(lde_absorb(&current));
    assert!(matches!(
        DirectCompactTerminalProgram::compile_steps(&steps, &[prior, current]),
        Err(DirectCompactTerminalError::Fallback(
            DirectCompactTerminalFallbackReason::InvalidTailLift
        ))
    ));
}

fn add(left: u32, right: u32) -> u32 {
    ((u64::from(left) + u64::from(right)) % P) as u32
}

fn sub(left: u32, right: u32) -> u32 {
    ((u64::from(left) + P - u64::from(right)) % P) as u32
}

fn mul(left: u32, right: u32) -> u32 {
    ((u64::from(left) * u64::from(right)) % P) as u32
}

fn terminal_pair(left: u32, right: u32, twiddle: u32) -> (u32, u32) {
    let product = mul(twiddle, right);
    (add(left, product), sub(left, product))
}

fn scalar_terminal(index: usize, prefinal: [u32; 2], twiddle: u32) -> u32 {
    let product = mul(twiddle, prefinal[1]);
    if index == 0 {
        add(prefinal[0], product)
    } else {
        sub(prefinal[0], product)
    }
}

#[test]
fn paired_terminal_matches_scalar_for_zero_impulse_carry_and_random_words() {
    let mut cases = vec![
        ([0, 0], 0),
        ([1, 0], 1),
        ([0, 1], 1),
        ([(P - 1) as u32, (P - 2) as u32], (P - 1) as u32),
    ];
    let mut seed = 0x4d59_5df4_d0f3_3173u64;
    for _ in 0..256 {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let left = (seed % P) as u32;
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let right = (seed % P) as u32;
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        cases.push(([left, right], (seed % P) as u32));
    }
    for (prefinal, twiddle) in cases {
        let pair = terminal_pair(prefinal[0], prefinal[1], twiddle);
        assert_eq!(pair.0, scalar_terminal(0, prefinal, twiddle));
        assert_eq!(pair.1, scalar_terminal(1, prefinal, twiddle));
        assert_eq!(add(pair.0, pair.1), mul(2, prefinal[0]));
    }
}
