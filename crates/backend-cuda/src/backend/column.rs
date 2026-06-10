use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::vcs::blake2_hash::Blake2sHash;
use stwo::prover::backend::{Column, ColumnOps};

use crate::backend::CudaBackend;
use crate::columns as interface;
use crate::columns::base_field_vec::BaseFieldVec;
use crate::columns::bindings;
use crate::columns::blake_2s_hash_vec::Blake2sHashVec;
use crate::columns::secure_field_vec::SecureFieldVec;

fn split_host_backed_vec<T, V>(values: Vec<T>, from_vec: fn(Vec<T>) -> V) -> (V, V) {
    assert!(
        values.len().is_multiple_of(2),
        "column split_at_mid requires an even-length column"
    );
    let mid = values.len() / 2;
    let mut values = values;
    let right = values.split_off(mid);
    (from_vec(values), from_vec(right))
}

impl ColumnOps<BaseField> for CudaBackend {
    type Column = BaseFieldVec;

    fn bit_reverse_column(column: &mut Self::Column) {
        let size = column.len();
        assert!(size.is_power_of_two() && size < u32::MAX as usize);

        unsafe {
            interface::bindings::bit_reverse_base_field(column.device_ptr, size);
        }
    }
}

impl ColumnOps<SecureField> for CudaBackend {
    type Column = SecureFieldVec;
    fn bit_reverse_column(column: &mut Self::Column) {
        let size = column.len();
        assert!(size.is_power_of_two() && size < u32::MAX as usize);

        unsafe {
            interface::bindings::bit_reverse_secure_field(column.device_ptr, size);
        }
    }
}

impl Column<BaseField> for interface::base_field_vec::BaseFieldVec {
    fn zeros(len: usize) -> Self {
        Self::new_zeroes(len)
    }

    fn to_cpu(&self) -> Vec<BaseField> {
        self.to_vec()
    }

    fn len(&self) -> usize {
        self.size
    }

    fn at(&self, index: usize) -> BaseField {
        Self::get_data(self, index)
    }

    fn set(&mut self, _index: usize, _value: BaseField) {
        Self::set_data(self, _index, _value);
    }

    fn split_at_mid(self) -> (Self, Self) {
        split_host_backed_vec(self.to_vec(), BaseFieldVec::from_vec)
    }

    unsafe fn uninitialized(len: usize) -> Self {
        Self {
            device_ptr: bindings::cuda_malloc_uint32_t(len as u32),
            size: len,
            owns_memory: true,
        }
    }
}

impl FromIterator<BaseField> for BaseFieldVec {
    fn from_iter<T: IntoIterator<Item = BaseField>>(iter: T) -> Self {
        let vec: Vec<BaseField> = iter.into_iter().collect();
        BaseFieldVec::from_vec(vec)
    }
}

impl IntoIterator for BaseFieldVec {
    type Item = BaseField;

    type IntoIter = std::vec::IntoIter<BaseField>;

    fn into_iter(self) -> Self::IntoIter {
        self.to_cpu().into_iter()
    }
}

impl Column<SecureField> for SecureFieldVec {
    fn zeros(_len: usize) -> Self {
        Self::new_zeroes(_len)
    }

    fn to_cpu(&self) -> Vec<SecureField> {
        self.to_vec()
    }

    fn len(&self) -> usize {
        self.size
    }

    fn at(&self, _index: usize) -> SecureField {
        Self::get_data(self, _index)
    }

    fn set(&mut self, _index: usize, _value: SecureField) {
        Self::set_data(self, _index, _value);
    }

    fn split_at_mid(self) -> (Self, Self) {
        split_host_backed_vec(self.to_vec(), SecureFieldVec::from_vec)
    }

    unsafe fn uninitialized(len: usize) -> Self {
        Self {
            device_ptr: bindings::cuda_malloc_uint32_t(4 * len as u32),
            size: len,
        }
    }
}

impl FromIterator<SecureField> for SecureFieldVec {
    fn from_iter<T: IntoIterator<Item = SecureField>>(iter: T) -> Self {
        let data: Vec<SecureField> = iter.into_iter().collect();
        Self::from_vec(data)
    }
}

impl Column<Blake2sHash> for Blake2sHashVec {
    fn zeros(len: usize) -> Self {
        Self::new_zeroes(len)
    }

    fn to_cpu(&self) -> Vec<Blake2sHash> {
        self.to_vec()
    }

    fn len(&self) -> usize {
        self.size
    }

    fn at(&self, index: usize) -> Blake2sHash {
        Self::get_data(self, index)
    }

    fn set(&mut self, _index: usize, _value: Blake2sHash) {
        Self::set_data(self, _index, _value);
    }

    unsafe fn uninitialized(len: usize) -> Self {
        Self {
            device_ptr: bindings::cuda_malloc_blake_2s_hash(len),
            size: len,
        }
    }

    fn split_at_mid(self) -> (Self, Self) {
        split_host_backed_vec(self.to_vec(), Blake2sHashVec::from_vec)
    }
}

impl FromIterator<Blake2sHash> for Blake2sHashVec {
    fn from_iter<T: IntoIterator<Item = Blake2sHash>>(iter: T) -> Self {
        let data: Vec<Blake2sHash> = iter.into_iter().collect();
        Self::from_vec(data)
    }
}
#[cfg(all(test, stwo_cuda_link))]
mod tests {
    use stwo::core::fields::m31::BaseField;
    use stwo::core::fields::qm31::SecureField;
    use stwo::prover::backend::{Column, ColumnOps, CpuBackend};

    use crate::backend::CudaBackend;
    use crate::columns::base_field_vec::BaseFieldVec;
    use crate::columns::secure_field_vec::SecureFieldVec;

    #[test]
    fn test_bit_reverse_base_field() {
        let size: usize = 1 << 10;
        let column_data = (0..size as u32).map(BaseField::from).collect::<Vec<_>>();
        let mut expected_result = column_data.clone();
        CpuBackend::bit_reverse_column(&mut expected_result);

        let mut column = BaseFieldVec::from_vec(column_data);
        <CudaBackend as ColumnOps<BaseField>>::bit_reverse_column(&mut column);

        assert_eq!(column.to_cpu(), expected_result);
    }

    #[test]
    fn test_bit_reverse_secure_field() {
        let size: usize = 1 << 16;

        let from_raw = (1..(size + 1) as u32).collect::<Vec<u32>>();
        let from_cpu = from_raw
            .chunks(4)
            .map(|a| SecureField::from_u32_unchecked(a[0], a[1], a[2], a[3]))
            .collect::<Vec<_>>();
        let mut array_expected = from_cpu.clone();

        CpuBackend::bit_reverse_column(&mut array_expected);

        let mut array = SecureFieldVec::from_vec(from_cpu.clone());
        <CudaBackend as ColumnOps<SecureField>>::bit_reverse_column(&mut array);

        assert_eq!(array.to_cpu(), array_expected);
    }
}
