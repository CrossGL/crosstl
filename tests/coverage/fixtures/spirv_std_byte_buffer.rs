// An excerpt of crates/spirv-std/src/byte_addressable_buffer.rs from Rust-GPU/rust-gpu 62401280 (spirv-std 0.11.0),
// read by tests/coverage/test_rust_spirv_std.py.
impl<'a> ByteAddressableBuffer<&'a [u32]> {
    #[inline]
    pub fn from_slice(data: &'a [u32]) -> Self {
        Self { data }
    }

    pub unsafe fn load<T>(&self, byte_index: u32) -> T {
        bounds_check::<T>(self.data, byte_index);
        unsafe { buffer_load_intrinsic(self.data, byte_index) }
    }
}

impl<'a> ByteAddressableBuffer<&'a mut [u32]> {
    #[inline]
    pub fn from_mut_slice(data: &'a mut [u32]) -> Self {
        Self { data }
    }

    pub fn as_ref(&self) -> ByteAddressableBuffer<&[u32]> {
        ByteAddressableBuffer { data: self.data }
    }

    pub unsafe fn load<T>(&self, byte_index: u32) -> T {
        unsafe { self.as_ref().load(byte_index) }
    }

    pub unsafe fn store<T>(&mut self, byte_index: u32, value: T) {
        bounds_check::<T>(self.data, byte_index);
        unsafe { buffer_store_intrinsic(self.data, byte_index, value) }
    }
}
