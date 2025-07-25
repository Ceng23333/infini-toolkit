#![cfg(infini)]
#![deny(warnings)]

use infini_rt::{bindings::infiniDtype_t, DevByte, Stream};
use std::ffi::c_int;
use std::mem::transmute;
use std::ptr::null_mut;

#[macro_use]
#[allow(non_snake_case, non_camel_case_types)]
pub mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

    #[macro_export]
    macro_rules! infiniccl {
        ($f:expr) => {{
            #[allow(unused_imports)]
            use $crate::bindings::*;
            #[allow(unused_unsafe, clippy::macro_metavars_in_unsafe)]
            let err = unsafe { $f };
            assert_eq!(err, infiniStatus_t::INFINI_STATUS_SUCCESS);
        }};
    }
}

#[repr(transparent)]
pub struct Comm(bindings::infinicclComm_t);

impl Comm {
    pub fn init_all(ty: bindings::infiniDevice_t, indices: &[c_int]) -> Vec<Self> {
        let mut ans = vec![null_mut(); indices.len()];
        infiniccl!(infinicclCommInitAll(
            ty,
            ans.as_mut_ptr(),
            indices.len() as _,
            indices.as_ptr()
        ));
        ans.into_iter().map(Self).collect()
    }
}

impl Drop for Comm {
    fn drop(&mut self) {
        infiniccl!(infinicclCommDestroy(self.0))
    }
}

unsafe impl Send for Comm {}
unsafe impl Sync for Comm {}

impl AsRaw for Comm {
    type Raw = bindings::infinicclComm_t;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0
    }
}

impl Comm {
    pub fn allreduce(
        &self,
        recvbuf: &mut [DevByte],
        sendbuf: &[DevByte],
        dt: infiniDtype_t,
        op: bindings::infinicclReduceOp_t,
        stream: &Stream,
    ) {
        use infini_rt::AsRaw;
        infiniccl!(infinicclAllReduce(
            sendbuf.as_ptr() as *mut _,
            recvbuf.as_mut_ptr() as *mut _,
            sendbuf.len(),
            unsafe { transmute(dt) },
            op,
            self.as_raw(),
            stream.as_raw()
        ))
    }
}

/// 资源的原始形式的表示。通常来自底层库的定义。
pub trait AsRaw {
    /// 原始形式的类型。
    type Raw: Unpin + 'static;
    /// # Safety
    ///
    /// The caller must ensure that the returned item is dropped before the original item.
    unsafe fn as_raw(&self) -> Self::Raw;
}
