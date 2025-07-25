use crate::{bindings::infiniopHandle_t, AsRaw};
use std::ptr::null_mut;

#[repr(transparent)]
pub struct Handle(infiniopHandle_t);

impl Handle {
    pub fn new() -> Self {
        let mut ptr = null_mut();
        infiniop!(infiniopCreateHandle(&mut ptr));
        Self(ptr)
    }
}

impl Default for Handle {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        infiniop!(infiniopDestroyHandle(self.0))
    }
}

unsafe impl Send for Handle {}
unsafe impl Sync for Handle {}

impl AsRaw for Handle {
    type Raw = infiniopHandle_t;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0
    }
}
