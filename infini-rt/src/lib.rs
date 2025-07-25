#![cfg(infini)]
#![deny(warnings)]

#[macro_use]
#[allow(non_camel_case_types)]
pub mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

    #[macro_export]
    macro_rules! infinirt {
        ($f:expr) => {{
            #[allow(unused_imports)]
            use $crate::bindings::*;
            #[allow(unused_unsafe, clippy::macro_metavars_in_unsafe)]
            let err = unsafe { $f };
            assert_eq!(err, infiniStatus_t::INFINI_STATUS_SUCCESS);
        }};
    }
}

pub use bindings::infiniDevice_t::{self, *};

#[inline]
pub fn init() {
    infinirt!(infinirtInit());
}

#[test]
fn test_init() {
    init()
}

mod device;
mod event;
mod memory;
mod stream;

pub use device::Device;
pub use event::Event;
pub use memory::{DevBlob, DevByte, HostBlob};
pub use stream::Stream;

/// 资源的原始形式的表示。通常来自底层库的定义。
pub trait AsRaw {
    /// 原始形式的类型。
    type Raw: Unpin + 'static;
    /// # Safety
    ///
    /// The caller must ensure that the returned item is dropped before the original item.
    unsafe fn as_raw(&self) -> Self::Raw;
}
