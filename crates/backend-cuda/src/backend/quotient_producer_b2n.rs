//! Compiled quotient-producer to inverse-NTT boundary.
//!
//! SN2 combines on a log-23 subdomain. The legacy prepared path writes that
//! image and then rereads and rewrites it once per B2N stage. This program owns
//! stages 1..7 in the producer CTA and completes the transform as two 8-stage
//! intervals. Compilation is pure and exact-shape; runtime never probes policy.

use num_traits::Zero;
use stwo::core::circle::CirclePoint;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::quotients::denominator_inverses;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::utils::bit_reverse_index;
use stwo::prover::backend::CpuBackend;
use stwo::prover::poly::circle::PolyOps;
use stwo::prover::poly::twiddles::TwiddleBuffer;

use super::prepared_quotient::{QuotientSampleConstants, QuotientWorkspaceConfig};

pub const QUOTIENT_PRODUCER_B2N_FIRST_STAGES: u32 = 7;
pub const QUOTIENT_PRODUCER_B2N_LAUNCH_THREADS: u32 = 128;
pub const QUOTIENT_PRODUCER_BATCH_INVERSE_CHUNK: u32 = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuotientProducerB2nSchedule {
    pub lifting_log_size: u32,
    pub subdomain_log_size: u32,
    pub sample_count: usize,
    pub producer_stages: u32,
    pub continuation_intervals: [u32; 2],
}

impl QuotientProducerB2nSchedule {
    pub const fn is_exact(self) -> bool {
        self.lifting_log_size == 25
            && self.subdomain_log_size == 23
            && self.sample_count > 0
            && self.producer_stages == QUOTIENT_PRODUCER_B2N_FIRST_STAGES
            && self.continuation_intervals[0] == 8
            && self.continuation_intervals[1] == 8
            && self.producer_stages
                + self.continuation_intervals[0]
                + self.continuation_intervals[1]
                == self.subdomain_log_size
    }
}

/// Static AOT resource contract. `scripts/cuda_compile_check.sh --resources`
/// rejects the cubin if ptxas exceeds this register cap or emits a spill.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuotientProducerB2nResourceContract {
    pub sm_arch: u32,
    pub cuda_toolkit_major: u32,
    pub cuda_toolkit_minor: u32,
    pub launch_threads: u32,
    pub min_blocks_per_sm: u32,
    pub ptxas_registers_per_thread: u32,
    pub max_registers_per_thread: u32,
    pub ptxas_stack_bytes: u32,
    pub ptxas_spill_store_bytes: u32,
    pub ptxas_spill_load_bytes: u32,
    pub static_shared_bytes: u32,
    pub zero_spills_required: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuotientProducerB2nTraffic {
    pub coordinate_image_bytes: u64,
    pub unchanged_partial_read_bytes: u64,
    pub denominator_factors: u64,
    pub batch_inverse_calls: u64,
    pub fallback_logical_bytes: u64,
    pub fused_logical_bytes: u64,
    pub eliminated_logical_bytes: u64,
    pub fallback_kernel_launches: u32,
    pub fused_kernel_launches: u32,
    pub eliminated_kernel_launches: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuotientProducerB2nReceipt {
    pub schedule: QuotientProducerB2nSchedule,
    pub resources: QuotientProducerB2nResourceContract,
    pub traffic: QuotientProducerB2nTraffic,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QuotientProducerB2nProgram {
    receipt: QuotientProducerB2nReceipt,
    partial_log_sizes: Vec<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum QuotientProducerB2nError {
    UnsupportedShape(QuotientWorkspaceConfig),
    EmptySources,
    TooManySources(usize),
    PartialLogSizeTooLarge {
        source: usize,
        log_size: u32,
        subdomain_log_size: u32,
    },
    ConstantsCountMismatch {
        expected: usize,
        actual: usize,
    },
    PartialLength {
        source: usize,
        expected: usize,
        actual: usize,
    },
    InvalidSchedule,
    SizeOverflow,
}

impl core::fmt::Display for QuotientProducerB2nError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "invalid quotient producer/B2N program: {self:?}")
    }
}

impl std::error::Error for QuotientProducerB2nError {}

impl QuotientProducerB2nProgram {
    pub fn compile(
        config: QuotientWorkspaceConfig,
        partial_log_sizes: &[u32],
    ) -> Result<Self, QuotientProducerB2nError> {
        if config.lifting_log_size != 25 || config.log_blowup_factor != 2 {
            return Err(QuotientProducerB2nError::UnsupportedShape(config));
        }
        validate_sources(23, partial_log_sizes)?;
        let schedule = QuotientProducerB2nSchedule {
            lifting_log_size: 25,
            subdomain_log_size: 23,
            sample_count: partial_log_sizes.len(),
            producer_stages: QUOTIENT_PRODUCER_B2N_FIRST_STAGES,
            continuation_intervals: [8, 8],
        };
        if !schedule.is_exact() {
            return Err(QuotientProducerB2nError::InvalidSchedule);
        }
        Ok(Self {
            receipt: QuotientProducerB2nReceipt {
                schedule,
                resources: QuotientProducerB2nResourceContract {
                    sm_arch: 90,
                    cuda_toolkit_major: 11,
                    cuda_toolkit_minor: 8,
                    launch_threads: QUOTIENT_PRODUCER_B2N_LAUNCH_THREADS,
                    min_blocks_per_sm: 4,
                    ptxas_registers_per_thread: 98,
                    max_registers_per_thread: 128,
                    ptxas_stack_bytes: 0,
                    ptxas_spill_store_bytes: 0,
                    ptxas_spill_load_bytes: 0,
                    static_shared_bytes: 4
                        * QUOTIENT_PRODUCER_B2N_LAUNCH_THREADS
                        * core::mem::size_of::<u32>() as u32,
                    zero_spills_required: true,
                },
                traffic: traffic(schedule)?,
            },
            partial_log_sizes: partial_log_sizes.to_vec(),
        })
    }

    pub const fn receipt(&self) -> QuotientProducerB2nReceipt {
        self.receipt
    }

    pub fn matches(&self, config: QuotientWorkspaceConfig, partial_log_sizes: &[u32]) -> bool {
        config.lifting_log_size == self.receipt.schedule.lifting_log_size
            && config.log_blowup_factor
                == self.receipt.schedule.lifting_log_size - self.receipt.schedule.subdomain_log_size
            && partial_log_sizes == self.partial_log_sizes
    }
}

fn traffic(
    schedule: QuotientProducerB2nSchedule,
) -> Result<QuotientProducerB2nTraffic, QuotientProducerB2nError> {
    let rows = 1u64
        .checked_shl(schedule.subdomain_log_size)
        .ok_or(QuotientProducerB2nError::SizeOverflow)?;
    let image = rows
        .checked_mul(4)
        .and_then(|words| words.checked_mul(core::mem::size_of::<u32>() as u64))
        .ok_or(QuotientProducerB2nError::SizeOverflow)?;
    let fallback_passes = 1 + 2 * u64::from(schedule.subdomain_log_size);
    let fused_passes = 1 + 2 * schedule.continuation_intervals.len() as u64;
    let fallback_logical_bytes = image
        .checked_mul(fallback_passes)
        .ok_or(QuotientProducerB2nError::SizeOverflow)?;
    let fused_logical_bytes = image
        .checked_mul(fused_passes)
        .ok_or(QuotientProducerB2nError::SizeOverflow)?;
    let fallback_kernel_launches = 1 + schedule.subdomain_log_size;
    let fused_kernel_launches = 1 + schedule.continuation_intervals.len() as u32;
    Ok(QuotientProducerB2nTraffic {
        coordinate_image_bytes: image,
        unchanged_partial_read_bytes: rows
            .checked_mul(schedule.sample_count as u64)
            .and_then(|values| values.checked_mul(4 * core::mem::size_of::<u32>() as u64))
            .ok_or(QuotientProducerB2nError::SizeOverflow)?,
        denominator_factors: rows
            .checked_mul(schedule.sample_count as u64)
            .ok_or(QuotientProducerB2nError::SizeOverflow)?,
        batch_inverse_calls: rows
            .checked_mul(
                (schedule.sample_count as u64)
                    .div_ceil(u64::from(QUOTIENT_PRODUCER_BATCH_INVERSE_CHUNK)),
            )
            .ok_or(QuotientProducerB2nError::SizeOverflow)?,
        fallback_logical_bytes,
        fused_logical_bytes,
        eliminated_logical_bytes: fallback_logical_bytes - fused_logical_bytes,
        fallback_kernel_launches,
        fused_kernel_launches,
        eliminated_kernel_launches: fallback_kernel_launches - fused_kernel_launches,
    })
}

fn validate_sources(
    subdomain_log_size: u32,
    partial_log_sizes: &[u32],
) -> Result<(), QuotientProducerB2nError> {
    if partial_log_sizes.is_empty() {
        return Err(QuotientProducerB2nError::EmptySources);
    }
    u32::try_from(partial_log_sizes.len())
        .map_err(|_| QuotientProducerB2nError::TooManySources(partial_log_sizes.len()))?;
    for (source, &log_size) in partial_log_sizes.iter().enumerate() {
        if log_size > subdomain_log_size {
            return Err(QuotientProducerB2nError::PartialLogSizeTooLarge {
                source,
                log_size,
                subdomain_log_size,
            });
        }
    }
    Ok(())
}

/// Independent scalar oracle for the exact producer boundary. It covers the
/// bit-reversed quotient-domain row, lifted low-log numerator indexing, sample
/// order, and all seven inverse-NTT stages without sharing device code.
pub fn quotient_producer_b2n_oracle(
    config: QuotientWorkspaceConfig,
    constants: &[QuotientSampleConstants],
    partials: &[Vec<SecureField>],
) -> Result<[Vec<u32>; 4], QuotientProducerB2nError> {
    let subdomain_log_size = config
        .lifting_log_size
        .checked_sub(config.log_blowup_factor)
        .ok_or(QuotientProducerB2nError::UnsupportedShape(config))?;
    if subdomain_log_size < QUOTIENT_PRODUCER_B2N_FIRST_STAGES {
        return Err(QuotientProducerB2nError::UnsupportedShape(config));
    }
    let mut logs = Vec::with_capacity(partials.len());
    for (source, values) in partials.iter().enumerate() {
        if values.is_empty() {
            return Err(QuotientProducerB2nError::PartialLength {
                source,
                expected: 1,
                actual: 0,
            });
        }
        logs.push(values.len().ilog2());
    }
    validate_sources(subdomain_log_size, &logs)?;
    if constants.len() != partials.len() {
        return Err(QuotientProducerB2nError::ConstantsCountMismatch {
            expected: partials.len(),
            actual: constants.len(),
        });
    }
    for (source, (values, &log_size)) in partials.iter().zip(&logs).enumerate() {
        let expected = 1usize
            .checked_shl(log_size)
            .ok_or(QuotientProducerB2nError::SizeOverflow)?;
        if values.len() != expected {
            return Err(QuotientProducerB2nError::PartialLength {
                source,
                expected,
                actual: values.len(),
            });
        }
    }

    let eval_domain = CanonicCoset::new(config.lifting_log_size).circle_domain();
    let (domain, _) = eval_domain.split(config.log_blowup_factor);
    let rows = domain.size();
    let sample_points = constants
        .iter()
        .map(|value| value.sample_point)
        .collect::<Vec<CirclePoint<SecureField>>>();
    let mut coordinates: [Vec<BaseField>; 4] = std::array::from_fn(|_| Vec::with_capacity(rows));
    for row in 0..rows {
        let point = domain.at(bit_reverse_index(row, subdomain_log_size));
        let inverses = denominator_inverses(&sample_points, point);
        let quotient = constants
            .iter()
            .zip(partials)
            .zip(&logs)
            .zip(inverses)
            .fold(
                SecureField::zero(),
                |sum, (((constants, partial), &log_size), inverse)| {
                    let ratio = subdomain_log_size - log_size;
                    let lifted = (row >> (ratio + 1) << 1) + (row & 1);
                    let numerator = partial[lifted] - constants.first_linear_term_acc * point.y;
                    sum + numerator.mul_cm31(inverse)
                },
            );
        for (coordinate, value) in quotient.to_m31_array().into_iter().enumerate() {
            coordinates[coordinate].push(value);
        }
    }

    let twiddles = CpuBackend::precompute_twiddles(eval_domain.half_coset)
        .itwiddles
        .extract_subdomain_twiddles(config.lifting_log_size, subdomain_log_size);
    let mut layer_size = rows / 2;
    let mut layer_offset = 0usize;
    for stage in 1..=QUOTIENT_PRODUCER_B2N_FIRST_STAGES {
        let stride = 1usize << (stage - 1);
        for gid in 0..rows / 2 {
            let group = gid & (stride - 1);
            let pair = gid >> (stage - 1);
            let left = group + pair * 2 * stride;
            let right = left + stride;
            let twiddle = if stage == 1 {
                circle_twiddle(&twiddles, pair)
            } else {
                twiddles[layer_offset + pair]
            };
            for values in &mut coordinates {
                let left_value = values[left];
                let right_value = values[right];
                values[left] = left_value + right_value;
                values[right] = (left_value - right_value) * twiddle;
            }
        }
        if stage >= 2 {
            layer_size /= 2;
            layer_offset += layer_size;
        }
    }
    Ok(coordinates.map(|values| values.into_iter().map(|value| value.0).collect()))
}

fn circle_twiddle(twiddles: &[BaseField], index: usize) -> BaseField {
    let pair = index / 4;
    match index % 4 {
        0 => twiddles[2 * pair + 1],
        1 => -twiddles[2 * pair + 1],
        2 => -twiddles[2 * pair],
        _ => twiddles[2 * pair],
    }
}

#[cfg(test)]
mod tests {
    use stwo::core::circle::SECURE_FIELD_CIRCLE_GEN;
    use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
    use stwo::prover::poly::BitReversedOrder;

    use super::*;

    #[test]
    fn exact_sn2_receipt_removes_twenty_one_launches_and_5_637_gb() {
        let program = QuotientProducerB2nProgram::compile(
            QuotientWorkspaceConfig {
                lifting_log_size: 25,
                log_blowup_factor: 2,
            },
            &[23; 19],
        )
        .unwrap();
        let receipt = program.receipt();
        assert!(receipt.schedule.is_exact());
        assert_eq!(receipt.traffic.coordinate_image_bytes, 134_217_728);
        assert_eq!(receipt.traffic.denominator_factors, 159_383_552);
        assert_eq!(receipt.traffic.batch_inverse_calls, 25_165_824);
        assert_eq!(receipt.traffic.unchanged_partial_read_bytes, 2_550_136_832);
        assert_eq!(receipt.traffic.fallback_logical_bytes, 6_308_233_216);
        assert_eq!(receipt.traffic.fused_logical_bytes, 671_088_640);
        assert_eq!(receipt.traffic.eliminated_logical_bytes, 5_637_144_576);
        assert_eq!(
            (
                receipt.traffic.fallback_kernel_launches,
                receipt.traffic.fused_kernel_launches,
                receipt.traffic.eliminated_kernel_launches,
            ),
            (24, 3, 21)
        );
        assert_eq!(receipt.resources.max_registers_per_thread, 128);
        assert_eq!(receipt.resources.ptxas_registers_per_thread, 98);
        assert_eq!(receipt.resources.static_shared_bytes, 2048);
    }

    #[test]
    fn compilation_fails_closed_outside_the_registered_shape() {
        for config in [
            QuotientWorkspaceConfig {
                lifting_log_size: 24,
                log_blowup_factor: 2,
            },
            QuotientWorkspaceConfig {
                lifting_log_size: 25,
                log_blowup_factor: 1,
            },
        ] {
            assert!(matches!(
                QuotientProducerB2nProgram::compile(config, &[23]),
                Err(QuotientProducerB2nError::UnsupportedShape(_))
            ));
        }
        let config = QuotientWorkspaceConfig {
            lifting_log_size: 25,
            log_blowup_factor: 2,
        };
        assert_eq!(
            QuotientProducerB2nProgram::compile(config, &[]),
            Err(QuotientProducerB2nError::EmptySources)
        );
        assert!(matches!(
            QuotientProducerB2nProgram::compile(config, &[24]),
            Err(QuotientProducerB2nError::PartialLogSizeTooLarge { .. })
        ));
    }

    #[test]
    fn scalar_oracle_matches_the_independent_cpu_inverse_transform() {
        let config = QuotientWorkspaceConfig {
            lifting_log_size: 9,
            log_blowup_factor: 2,
        };
        let constants = [
            QuotientSampleConstants {
                sample_point: SECURE_FIELD_CIRCLE_GEN.mul(3),
                first_linear_term_acc: SecureField::from_u32_unchecked(2, 3, 5, 7),
            },
            QuotientSampleConstants {
                sample_point: SECURE_FIELD_CIRCLE_GEN.mul(11),
                first_linear_term_acc: SecureField::from_u32_unchecked(13, 17, 19, 23),
            },
        ];
        let logs = [7u32, 5];
        let partials = logs
            .iter()
            .enumerate()
            .map(|(source, &log_size)| {
                (0..1usize << log_size)
                    .map(|row| {
                        let value = (source as u32 + 1) * 65_537 + row as u32 * 257;
                        SecureField::from_u32_unchecked(value, value + 1, value + 2, value + 3)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let oracle = quotient_producer_b2n_oracle(config, &constants, &partials).unwrap();

        let eval_domain = CanonicCoset::new(config.lifting_log_size).circle_domain();
        let (domain, _) = eval_domain.split(config.log_blowup_factor);
        let full_twiddles = CpuBackend::precompute_twiddles(eval_domain.half_coset);
        let inverse_twiddles = full_twiddles
            .itwiddles
            .extract_subdomain_twiddles(config.lifting_log_size, domain.log_size());
        let tree = stwo::prover::poly::twiddles::TwiddleTree {
            root_coset: domain.half_coset,
            twiddles: Vec::new(),
            itwiddles: inverse_twiddles,
        };

        let sample_points = constants
            .iter()
            .map(|value| value.sample_point)
            .collect::<Vec<_>>();
        let mut rows: [Vec<BaseField>; 4] = std::array::from_fn(|_| Vec::new());
        for row in 0..domain.size() {
            let point = domain.at(bit_reverse_index(row, domain.log_size()));
            let inverses = denominator_inverses(&sample_points, point);
            let quotient = constants
                .iter()
                .zip(&partials)
                .zip(logs)
                .zip(inverses)
                .fold(
                    SecureField::zero(),
                    |sum, (((constants, partial), log), inverse)| {
                        let ratio = domain.log_size() - log;
                        let lifted = (row >> (ratio + 1) << 1) + (row & 1);
                        sum + (partial[lifted] - constants.first_linear_term_acc * point.y)
                            .mul_cm31(inverse)
                    },
                );
            for (coordinate, value) in quotient.to_m31_array().into_iter().enumerate() {
                rows[coordinate].push(value);
            }
        }
        let scale = BaseField::from_u32_unchecked(domain.size() as u32).inverse();
        for coordinate in 0..4 {
            let expected = CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(
                domain,
                rows[coordinate].clone(),
            )
            .interpolate_with_twiddles(&tree)
            .coeffs;
            let actual = oracle[coordinate]
                .iter()
                .copied()
                .map(BaseField::from_u32_unchecked)
                .map(|value| value * scale)
                .collect::<Vec<_>>();
            assert_eq!(actual, expected, "coordinate {coordinate}");
        }
    }
}
