use crate::{bindings::infinirtStream_t, AsRaw, Device};
use std::ptr::null_mut;

#[repr(transparent)]
pub struct Stream(infinirtStream_t);

impl Device {
    pub fn stream(&self) -> Stream {
        let mut stream = null_mut();
        infinirt!(infinirtStreamCreate(&mut stream));
        Stream(stream)
    }
}

unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

impl Drop for Stream {
    fn drop(&mut self) {
        infinirt!(infinirtStreamDestroy(self.0))
    }
}

impl AsRaw for Stream {
    type Raw = infinirtStream_t;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0
    }
}

impl Stream {
    #[inline]
    pub fn synchronize(&self) {
        infinirt!(infinirtStreamSynchronize(self.0))
    }
}
