use itertools::Itertools;
use stwo::core::vcs::blake2_hash::Blake2sHash;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasherGeneric;
use stwo::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use stwo::prover::backend::{Column, ColumnOps};
use stwo::prover::vcs_lifted::ops::MerkleOpsLifted;

use crate::backend::{CudaBackend, UploadedDevicePointerVec, UploadedUint32Vec};
use crate::columns::base_field_vec::BaseFieldVec;
use crate::columns::bindings;
use crate::columns::blake_2s_hash_vec::Blake2sHashVec;

impl ColumnOps<Blake2sHash> for CudaBackend {
    type Column = Blake2sHashVec;

    fn bit_reverse_column(_column: &mut Self::Column) {
        unimplemented!("Blake2sHash column bit-reverse is not implemented on CudaBackend")
    }
}

impl<const IS_M31_OUTPUT: bool> MerkleOpsLifted<Blake2sMerkleHasherGeneric<IS_M31_OUTPUT>>
    for CudaBackend
{
    fn build_leaves(columns: &[&BaseFieldVec], lifting_log_size: u32) -> Blake2sHashVec {
        if columns.is_empty() {
            let hasher = Blake2sMerkleHasherGeneric::<IS_M31_OUTPUT>::default();
            return Blake2sHashVec::from_vec(vec![hasher.finalize()]);
        }
        if IS_M31_OUTPUT {
            // The GPU kernel implements the non-M31 hasher; the M31-output variant is
            // hashed on the host to match the reference byte stream exactly (same
            // decision as the Metal port).
            return build_leaves_host::<IS_M31_OUTPUT>(columns, lifting_log_size);
        }

        assert!(lifting_log_size < usize::BITS);
        assert!(columns[0].len() >= 2, "A column must be of length >= 2.");

        let mut previous_len = columns[0].len();
        let mut column_log_sizes = Vec::with_capacity(columns.len());
        for column in columns {
            let len = column.len();
            assert!(
                len.is_power_of_two(),
                "column lengths must be powers of two"
            );
            assert!(len >= 2, "A column must be of length >= 2.");
            assert!(
                previous_len <= len,
                "lifted Blake2s columns must be sorted increasingly by length"
            );
            let log_size = len.ilog2();
            assert!(
                log_size <= lifting_log_size,
                "lifting_log_size must be at least the largest column log size"
            );
            column_log_sizes.push(log_size);
            previous_len = len;
        }

        let size = 1usize << lifting_log_size;
        let result = Blake2sHashVec::new_uninitialized(size);
        unsafe {
            Self::commit_on_first_layer_lifted_using_gpu(
                columns,
                &column_log_sizes,
                lifting_log_size,
                result.device_ptr,
            );
        }
        result
    }

    /// Batched leaf-hash recompute for the pruned-tree decommit: one indexed kernel
    /// launch + one D2H of `indices.len()` hashes, replacing the per-(column, leaf)
    /// synchronous `raw_value` readbacks of the host path. Same byte stream as
    /// [`Self::build_leaves`] => byte-identical hashes (proof byte-equality gated).
    /// Kill switch: `STWO_CUDA_LEAF_HASHES=0` falls back to the host recompute.
    fn leaf_hashes_at(
        columns: &[&BaseFieldVec],
        lifting_log_size: u32,
        indices: &[usize],
    ) -> Option<Vec<Blake2sHash>> {
        if IS_M31_OUTPUT
            || columns.is_empty()
            || std::env::var("STWO_CUDA_LEAF_HASHES").as_deref() == Ok("0")
        {
            return None;
        }
        assert!(lifting_log_size < u32::BITS);
        let column_log_sizes = columns
            .iter()
            .map(|column| {
                let len = column.len();
                assert!(len.is_power_of_two() && len >= 2);
                len.ilog2()
            })
            .collect_vec();
        assert!(
            column_log_sizes.windows(2).all(|w| w[0] <= w[1]),
            "columns must be sorted increasingly by length"
        );
        assert!(*column_log_sizes.last().unwrap() <= lifting_log_size);

        let indices_u32 = indices
            .iter()
            .map(|&idx| {
                assert!(idx < 1usize << lifting_log_size);
                idx as u32
            })
            .collect_vec();
        let result = Blake2sHashVec::new_uninitialized(indices.len());
        unsafe {
            let device_column_pointers_vector: Vec<*const u32> =
                columns.iter().map(|column| column.device_ptr).collect();
            let device_column_pointers =
                UploadedDevicePointerVec::upload(&device_column_pointers_vector);
            let uploaded_column_log_sizes = UploadedUint32Vec::upload(&column_log_sizes);
            let uploaded_indices = UploadedUint32Vec::upload(&indices_u32);
            bindings::commit_on_first_layer_lifted_indexed(
                indices.len(),
                uploaded_indices.as_ptr(),
                columns.len(),
                device_column_pointers.as_ptr(),
                uploaded_column_log_sizes.as_ptr(),
                lifting_log_size,
                result.device_ptr as *mut Blake2sHash,
            );
        }
        Some(result.to_vec())
    }

    fn build_next_layer(prev_layer: &Blake2sHashVec) -> Blake2sHashVec {
        let size = prev_layer.len() / 2;
        if IS_M31_OUTPUT {
            let prev = prev_layer.to_vec();
            let hashes = (0..size)
                .map(|i| {
                    Blake2sMerkleHasherGeneric::<IS_M31_OUTPUT>::hash_children((
                        prev[2 * i],
                        prev[2 * i + 1],
                    ))
                })
                .collect();
            return Blake2sHashVec::from_vec(hashes);
        }
        let result = Blake2sHashVec::new_uninitialized(size);

        unsafe {
            Self::commit_on_layer_using_gpu(size, 0, &[], Some(prev_layer), result.device_ptr);
        }

        result
    }
}

/// Host-side leaf builder for the M31-output hasher: mirrors the CPU reference's byte
/// stream over downloaded column copies (chunked update of up to 16 columns per leaf).
fn build_leaves_host<const IS_M31_OUTPUT: bool>(
    columns: &[&BaseFieldVec],
    lifting_log_size: u32,
) -> Blake2sHashVec {
    use itertools::Itertools;
    let hasher = Blake2sMerkleHasherGeneric::<IS_M31_OUTPUT>::default();
    let host_columns: Vec<Vec<stwo::core::fields::m31::BaseField>> =
        columns.iter().map(|column| column.to_vec()).collect();

    let mut prev_layer: Vec<Blake2sMerkleHasherGeneric<IS_M31_OUTPUT>> = Vec::new();
    let mut prev_layer_log_size: u32 = 0;
    for (log_size, group) in
        &itertools::Itertools::group_by(host_columns.iter(), |column| column.len().ilog2())
    {
        let group_columns = group.collect_vec();
        if prev_layer.is_empty() {
            prev_layer = vec![hasher.clone(); 1 << log_size];
        } else {
            let log_ratio = log_size - prev_layer_log_size;
            prev_layer = (0..1usize << log_size)
                .map(|idx| prev_layer[(idx >> (log_ratio + 1) << 1) + (idx & 1)].clone())
                .collect();
        }
        for chunk_columns in group_columns.chunks(16) {
            for (i, hasher) in prev_layer.iter_mut().enumerate() {
                let mut chunk_bytes = [0u8; 16 * 4];
                let mut used = 0usize;
                for column in chunk_columns {
                    chunk_bytes[used..used + 4].copy_from_slice(&column[i].0.to_le_bytes());
                    used += 4;
                }
                hasher.update(&chunk_bytes[..used]);
            }
        }
        prev_layer_log_size = log_size;
    }

    let log_ratio = lifting_log_size - prev_layer_log_size;
    if log_ratio > 0 {
        prev_layer = (0..1usize << lifting_log_size)
            .map(|idx| prev_layer[(idx >> (log_ratio + 1) << 1) + (idx & 1)].clone())
            .collect();
    }
    Blake2sHashVec::from_vec(prev_layer.into_iter().map(|x| x.finalize()).collect())
}

impl CudaBackend {
    unsafe fn commit_on_first_layer_lifted_using_gpu(
        columns: &[&BaseFieldVec],
        column_log_sizes: &[u32],
        lifting_log_size: u32,
        result_pointer: *const Blake2sHash,
    ) {
        let size = 1usize << lifting_log_size;
        let device_column_pointers_vector: Vec<*const u32> =
            columns.iter().map(|column| column.device_ptr).collect();
        let device_column_pointers =
            UploadedDevicePointerVec::upload(&device_column_pointers_vector);
        let uploaded_column_log_sizes = UploadedUint32Vec::upload(column_log_sizes);

        bindings::commit_on_first_layer_lifted(
            size,
            columns.len(),
            device_column_pointers.as_ptr(),
            uploaded_column_log_sizes.as_ptr(),
            lifting_log_size,
            result_pointer as *mut Blake2sHash,
        );
    }

    unsafe fn commit_on_layer_using_gpu(
        size: usize,
        number_of_columns: usize,
        columns: &[&BaseFieldVec],
        prev_layer: Option<&Blake2sHashVec>,
        result_pointer: *const Blake2sHash,
    ) {
        let device_column_pointers_vector: Vec<*const u32> =
            columns.iter().map(|column| column.device_ptr).collect();

        let device_column_pointers =
            UploadedDevicePointerVec::upload(&device_column_pointers_vector);

        if let Some(previous_layer) = prev_layer {
            bindings::commit_on_layer_with_previous(
                size,
                number_of_columns,
                device_column_pointers.as_ptr(),
                previous_layer.device_ptr,
                result_pointer as *mut Blake2sHash,
            );
        } else {
            bindings::commit_on_first_layer(
                size,
                number_of_columns,
                device_column_pointers.as_ptr(),
                result_pointer as *mut Blake2sHash,
            );
        }
    }
}

#[cfg(any())] // legacy (pre-lifted-merkle) test, superseded by the testkit
mod tests {
    use stwo::core::fields::m31::{BaseField, M31};
    use stwo::core::vcs::blake2_merkle::Blake2sMerkleHasher;
    use stwo::prover::backend::{Column, CpuBackend};
    use stwo::prover::vcs::ops::MerkleOps;

    use crate::backend::CudaBackend;
    use crate::columns::base_field_vec::BaseFieldVec;
    use crate::columns::blake_2s_hash_vec::Blake2sHashVec;

    #[test]
    fn test_commit_on_first_layer_with_many_columns_compared_with_cpu() {
        let log_size = 16;
        let size = 1 << log_size;

        let cpu_columns_vector: Vec<Vec<BaseField>> = columns_test_vector(100, size);
        let gpu_columns_vector: Vec<BaseFieldVec> = gpu_columns_from(&cpu_columns_vector);

        let expected_result = <CpuBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(
            log_size,
            None,
            &cpu_columns_vector.iter().collect::<Vec<_>>(),
        );
        let result: Blake2sHashVec =
            <CudaBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(
                log_size,
                None,
                &gpu_columns_vector.iter().collect::<Vec<_>>(),
            );

        assert_eq!(result.to_cpu(), expected_result);
    }

    #[test]
    fn test_commit_on_layer_with_previous_layer_compared_with_cpu() {
        let current_layer_log_size = 10;
        let current_layer_size = 1 << current_layer_log_size;
        let previous_layer_log_size = current_layer_log_size + 1;
        let previous_layer_size = 1 << previous_layer_log_size;

        // First layer

        let cpu_columns_vector: Vec<Vec<BaseField>> = columns_test_vector(35, previous_layer_size);
        let gpu_columns_vector: Vec<BaseFieldVec> = gpu_columns_from(&cpu_columns_vector);

        let cpu_previous_layer = <CpuBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(
            previous_layer_log_size,
            None,
            &cpu_columns_vector.iter().collect::<Vec<_>>(),
        );
        let gpu_previous_layer: Blake2sHashVec =
            <CudaBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(
                previous_layer_log_size,
                None,
                &gpu_columns_vector.iter().collect::<Vec<_>>(),
            );

        // Current layer

        let cpu_columns_vector: Vec<Vec<BaseField>> = columns_test_vector(16, current_layer_size);
        let gpu_columns_vector: Vec<BaseFieldVec> = gpu_columns_from(&cpu_columns_vector);

        let expected_result = <CpuBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(
            current_layer_log_size,
            Some(&cpu_previous_layer),
            &cpu_columns_vector.iter().collect::<Vec<_>>(),
        );
        let result: Blake2sHashVec =
            <CudaBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(
                current_layer_log_size,
                Some(&gpu_previous_layer),
                &gpu_columns_vector.iter().collect::<Vec<_>>(),
            );

        assert_eq!(result.to_cpu(), expected_result);
    }

    fn gpu_columns_from(columns: &Vec<Vec<BaseField>>) -> Vec<BaseFieldVec> {
        columns
            .clone()
            .into_iter()
            .map(|vector| BaseFieldVec::from_vec(vector))
            .collect()
    }

    fn columns_test_vector(
        number_of_columns: usize,
        size_of_columns: usize,
    ) -> Vec<Vec<BaseField>> {
        (0..number_of_columns)
            .map(|index_of_column| {
                (0..size_of_columns)
                    .map(|index_in_column| M31::from(index_in_column * index_of_column))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_commit_on_first_layer_log24() {
        // Test at log_size=24 to check for size-related issues
        let log_size = 24u32;
        let size = 1 << log_size;

        // Use fewer columns to save memory - just 4 columns
        let cpu_columns_vector: Vec<Vec<BaseField>> = columns_test_vector(4, size);
        let gpu_columns_vector: Vec<BaseFieldVec> = gpu_columns_from(&cpu_columns_vector);

        let expected_result = <CpuBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(
            log_size,
            None,
            &cpu_columns_vector.iter().collect::<Vec<_>>(),
        );
        let result: Blake2sHashVec =
            <CudaBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(
                log_size,
                None,
                &gpu_columns_vector.iter().collect::<Vec<_>>(),
            );

        // Compare first and last hashes
        let cpu_hashes = expected_result.clone();
        let gpu_hashes = result.to_cpu();

        assert_eq!(gpu_hashes.len(), size);
        assert_eq!(cpu_hashes.len(), size);

        // Check first 100 hashes
        assert_eq!(
            gpu_hashes[..100],
            cpu_hashes[..100],
            "First 100 hashes mismatch"
        );
        // Check last 100 hashes
        assert_eq!(
            gpu_hashes[size - 100..],
            cpu_hashes[size - 100..],
            "Last 100 hashes mismatch"
        );
        // Full equality
        assert_eq!(gpu_hashes, cpu_hashes);
    }
}

#[cfg(all(test, stwo_cuda_link))]
mod lifted_tests {
    use stwo::core::fields::m31::{BaseField, M31};
    use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasher;
    use stwo::prover::backend::{Column, CpuBackend};
    use stwo::prover::vcs_lifted::ops::MerkleOpsLifted;

    use crate::backend::CudaBackend;
    use crate::columns::base_field_vec::BaseFieldVec;
    use crate::columns::blake_2s_hash_vec::Blake2sHashVec;

    #[test]
    fn test_build_leaves_with_mixed_column_sizes_compared_with_cpu() {
        let cpu_columns = vec![
            base_field_column(4, 1),
            base_field_column(4, 3),
            base_field_column(8, 5),
            base_field_column(8, 7),
            base_field_column(16, 11),
        ];
        let gpu_columns = gpu_columns_from(&cpu_columns);
        let expected = <CpuBackend as MerkleOpsLifted<Blake2sMerkleHasher>>::build_leaves(
            &cpu_columns.iter().collect::<Vec<_>>(),
            4,
        );
        let result: Blake2sHashVec =
            <CudaBackend as MerkleOpsLifted<Blake2sMerkleHasher>>::build_leaves(
                &gpu_columns.iter().collect::<Vec<_>>(),
                4,
            );

        assert_eq!(result.to_cpu(), expected);
    }

    #[test]
    fn test_build_leaves_with_additional_lifting_compared_with_cpu() {
        let cpu_columns = vec![
            base_field_column(4, 13),
            base_field_column(8, 17),
            base_field_column(8, 19),
        ];
        let gpu_columns = gpu_columns_from(&cpu_columns);
        let expected = <CpuBackend as MerkleOpsLifted<Blake2sMerkleHasher>>::build_leaves(
            &cpu_columns.iter().collect::<Vec<_>>(),
            5,
        );
        let result: Blake2sHashVec =
            <CudaBackend as MerkleOpsLifted<Blake2sMerkleHasher>>::build_leaves(
                &gpu_columns.iter().collect::<Vec<_>>(),
                5,
            );

        assert_eq!(result.to_cpu(), expected);
    }

    #[test]
    fn test_leaf_hashes_at_matches_cpu_build_leaves() {
        // Mixed sizes + extra lifting, scattered indices: the indexed kernel must
        // reproduce the exact leaves the full commit kernel (and the CPU reference)
        // produces at those positions.
        let cpu_columns = vec![
            base_field_column(8, 23),
            base_field_column(16, 29),
            base_field_column(16, 31),
            base_field_column(32, 37),
            base_field_column(64, 41),
        ];
        let gpu_columns = gpu_columns_from(&cpu_columns);
        let lifting_log_size = 7u32;
        let expected = <CpuBackend as MerkleOpsLifted<Blake2sMerkleHasher>>::build_leaves(
            &cpu_columns.iter().collect::<Vec<_>>(),
            lifting_log_size,
        );
        let indices = vec![0usize, 1, 5, 30, 31, 64, 99, 127];
        let result = <CudaBackend as MerkleOpsLifted<Blake2sMerkleHasher>>::leaf_hashes_at(
            &gpu_columns.iter().collect::<Vec<_>>(),
            lifting_log_size,
            &indices,
        )
        .expect("CUDA backend must provide the batched leaf-hash path");
        for (&idx, hash) in indices.iter().zip(&result) {
            assert_eq!(*hash, expected[idx], "leaf {idx}");
        }
    }

    fn gpu_columns_from(columns: &[Vec<BaseField>]) -> Vec<BaseFieldVec> {
        columns
            .iter()
            .cloned()
            .map(BaseFieldVec::from_vec)
            .collect()
    }

    fn base_field_column(size: usize, multiplier: u32) -> Vec<BaseField> {
        (0..size)
            .map(|index| M31::from(index as u32 * multiplier + multiplier))
            .collect()
    }
}

impl stwo::prover::vcs_lifted::ops::PackLeavesOps for CudaBackend {
    fn pack_leaves_input(
        values: &[&BaseFieldVec; stwo::core::fields::qm31::SECURE_EXTENSION_DEGREE],
    ) -> [BaseFieldVec;
           stwo::core::fields::qm31::SECURE_EXTENSION_DEGREE
               * stwo::core::vcs_lifted::verifier::PACKED_LEAF_SIZE] {
        use stwo::core::fields::qm31::SECURE_EXTENSION_DEGREE;
        use stwo::core::vcs_lifted::verifier::PACKED_LEAF_SIZE;
        // Mirrors the CPU reference over host copies (small FRI-commit path).
        let len = values[0].len();
        assert!(values.iter().all(|column| column.len() == len));
        assert!(len.is_multiple_of(PACKED_LEAF_SIZE));
        let packed_len = len / PACKED_LEAF_SIZE;

        let coords: [Vec<stwo::core::fields::m31::BaseField>; SECURE_EXTENSION_DEGREE] =
            std::array::from_fn(|coord| values[coord].to_vec());
        let mut packed: [Vec<stwo::core::fields::m31::BaseField>;
            SECURE_EXTENSION_DEGREE * PACKED_LEAF_SIZE] =
            std::array::from_fn(|_| Vec::with_capacity(packed_len));
        for packed_row in 0..packed_len {
            let row_start = packed_row * PACKED_LEAF_SIZE;
            for offset in 0..PACKED_LEAF_SIZE {
                for coord in 0..SECURE_EXTENSION_DEGREE {
                    packed[coord + offset * SECURE_EXTENSION_DEGREE]
                        .push(coords[coord][row_start + offset]);
                }
            }
        }
        packed.map(BaseFieldVec::from_vec)
    }
}
