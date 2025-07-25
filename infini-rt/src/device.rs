use crate::infiniDevice_t;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Device {
    pub ty: infiniDevice_t,
    pub id: u32,
}

impl Device {
    #[inline]
    pub fn synchronize(&self) {
        infinirt!(infinirtSetDevice(self.ty, self.id as i32));
        infinirt!(infinirtDeviceSynchronize());
    }
}
