use crate::{AsRaw, Device, Stream};
use std::{
    alloc::Layout,
    mem::{forget, size_of_val},
    ops::{Deref, DerefMut},
    os::raw::c_void,
    ptr::{null_mut, NonNull},
    slice::{from_raw_parts, from_raw_parts_mut},
};

#[repr(transparent)]
pub struct DevByte(u8);

impl Device {
    #[inline]
    pub fn memcpy_d2d(&self, dst: &mut [DevByte], src: &[DevByte]) {
        let (dst, src, len) = memcpy_ptr(dst, src);
        if len > 0 {
            infinirt!(infinirtMemcpy(
                dst,
                src,
                len,
                infinirtMemcpyKind_t::INFINIRT_MEMCPY_D2D
            ))
        }
    }

    #[inline]
    pub fn memcpy_h2d<T: Copy>(&self, dst: &mut [DevByte], src: &[T]) {
        let (dst, src, len) = memcpy_ptr(dst, src);
        if len > 0 {
            infinirt!(infinirtMemcpy(
                dst,
                src,
                len,
                infinirtMemcpyKind_t::INFINIRT_MEMCPY_H2D
            ))
        }
    }

    #[inline]
    pub fn memcpy_d2h<T: Copy>(&self, dst: &mut [T], src: &[DevByte]) {
        let (dst, src, len) = memcpy_ptr(dst, src);
        if len > 0 {
            infinirt!(infinirtMemcpy(
                dst,
                src,
                len,
                infinirtMemcpyKind_t::INFINIRT_MEMCPY_D2H
            ))
        }
    }
}

impl Stream {
    #[inline]
    pub fn memcpy_d2d(&self, dst: &mut [DevByte], src: &[DevByte]) {
        let (dst, src, len) = memcpy_ptr(dst, src);
        if len > 0 {
            infinirt!(infinirtMemcpyAsync(
                dst,
                src,
                len,
                infinirtMemcpyKind_t::INFINIRT_MEMCPY_D2D,
                self.as_raw()
            ))
        }
    }

    #[inline]
    pub fn memcpy_h2d<T: Copy>(&self, dst: &mut [DevByte], src: &[T]) {
        let (dst, src, len) = memcpy_ptr(dst, src);
        if len > 0 {
            infinirt!(infinirtMemcpyAsync(
                dst,
                src,
                len,
                infinirtMemcpyKind_t::INFINIRT_MEMCPY_H2D,
                self.as_raw()
            ))
        }
    }
}

#[inline]
fn memcpy_ptr<T, U>(dst: &mut [T], src: &[U]) -> (*mut c_void, *const c_void, usize) {
    let len = size_of_val(dst);
    assert_eq!(len, size_of_val(src));
    (dst.as_mut_ptr().cast(), src.as_ptr().cast(), len)
}

pub struct DevBlob {
    ptr: NonNull<DevByte>,
    len: usize,
}

impl Device {
    pub fn malloc<T: Copy>(&self, len: usize) -> DevBlob {
        let layout = Layout::array::<T>(len).unwrap();
        let len = layout.size();

        DevBlob {
            ptr: if len == 0 {
                NonNull::dangling()
            } else {
                let mut ptr = null_mut();
                infinirt!(infinirtMalloc(&mut ptr, len));
                NonNull::new(ptr).unwrap().cast()
            },
            len,
        }
    }

    pub fn from_host<T: Copy>(&self, data: &[T]) -> DevBlob {
        let src = data.as_ptr().cast();
        let len = size_of_val(data);

        DevBlob {
            ptr: if len == 0 {
                NonNull::dangling()
            } else {
                let mut ptr = null_mut();
                infinirt!(infinirtMalloc(&mut ptr, len));
                infinirt!(infinirtMemcpy(
                    ptr,
                    src,
                    len,
                    infinirtMemcpyKind_t::INFINIRT_MEMCPY_H2D
                ));
                NonNull::new(ptr).unwrap().cast()
            },
            len,
        }
    }
}

impl Stream {
    pub fn malloc<T: Copy>(&self, len: usize) -> DevBlob {
        let layout = Layout::array::<T>(len).unwrap();
        let len = layout.size();

        DevBlob {
            ptr: if len == 0 {
                NonNull::dangling()
            } else {
                let raw = unsafe { self.as_raw() };
                let mut ptr = null_mut();
                infinirt!(infinirtMallocAsync(&mut ptr, len, raw));
                NonNull::new(ptr).unwrap().cast()
            },
            len,
        }
    }

    pub fn from_host<T: Copy>(&self, data: &[T]) -> DevBlob {
        let src = data.as_ptr().cast();
        let len = size_of_val(data);

        DevBlob {
            ptr: if len == 0 {
                NonNull::dangling()
            } else {
                let raw = unsafe { self.as_raw() };
                let mut ptr = null_mut();
                infinirt!(infinirtMallocAsync(&mut ptr, len, raw));
                infinirt!(infinirtMemcpyAsync(
                    ptr,
                    src,
                    len,
                    infinirtMemcpyKind_t::INFINIRT_MEMCPY_H2D,
                    raw
                ));
                NonNull::new(ptr).unwrap().cast()
            },
            len,
        }
    }

    pub fn free(&self, blob: DevBlob) {
        if blob.len == 0 {
            return;
        }

        let &DevBlob { ptr, .. } = &blob;
        forget(blob);

        infinirt!(infinirtFreeAsync(ptr.as_ptr().cast(), self.as_raw()))
    }
}

impl Drop for DevBlob {
    fn drop(&mut self) {
        if self.len == 0 {
            return;
        }

        infinirt!(infinirtFree(self.ptr.as_ptr().cast(),))
    }
}

unsafe impl Send for DevBlob {}
unsafe impl Sync for DevBlob {}

impl AsRaw for DevBlob {
    type Raw = *mut DevByte;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.ptr.as_ptr()
    }
}

impl Deref for DevBlob {
    type Target = [DevByte];
    #[inline]
    fn deref(&self) -> &Self::Target {
        if self.len == 0 {
            &[]
        } else {
            unsafe { from_raw_parts(self.ptr.as_ptr(), self.len) }
        }
    }
}

impl DerefMut for DevBlob {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        if self.len == 0 {
            &mut []
        } else {
            unsafe { from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
        }
    }
}

pub struct HostBlob {
    ptr: NonNull<u8>,
    len: usize,
}

impl Device {
    pub fn malloc_host<T: Copy>(&self, len: usize) -> HostBlob {
        let layout = Layout::array::<T>(len).unwrap();
        let len = layout.size();

        HostBlob {
            ptr: if len == 0 {
                NonNull::dangling()
            } else {
                let mut ptr = null_mut();
                infinirt!(infinirtMallocHost(&mut ptr, len));
                NonNull::new(ptr).unwrap().cast()
            },
            len,
        }
    }
}

impl Drop for HostBlob {
    fn drop(&mut self) {
        if self.len == 0 {
            return;
        }

        infinirt!(infinirtFreeHost(self.ptr.as_ptr().cast(),))
    }
}

unsafe impl Send for HostBlob {}
unsafe impl Sync for HostBlob {}

impl AsRaw for HostBlob {
    type Raw = *mut u8;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.ptr.as_ptr()
    }
}

impl Deref for HostBlob {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &Self::Target {
        if self.len == 0 {
            &[]
        } else {
            unsafe { from_raw_parts(self.ptr.as_ptr(), self.len) }
        }
    }
}

impl DerefMut for HostBlob {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        if self.len == 0 {
            &mut []
        } else {
            unsafe { from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
        }
    }
}
