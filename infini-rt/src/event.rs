use crate::{
    bindings::{infinirtEventStatus_t, infinirtEvent_t},
    AsRaw, Device, Stream,
};
use std::ptr::null_mut;

#[repr(transparent)]
pub struct Event(infinirtEvent_t);

impl Device {
    pub fn event(&self) -> Event {
        let mut event = null_mut();
        infinirt!(infinirtEventCreate(&mut event));
        Event(event)
    }
}

unsafe impl Send for Event {}
unsafe impl Sync for Event {}

impl Drop for Event {
    fn drop(&mut self) {
        infinirt!(infinirtEventDestroy(self.0))
    }
}

impl AsRaw for Event {
    type Raw = infinirtEvent_t;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0
    }
}

impl Event {
    #[inline]
    pub fn synchronize(&self) {
        infinirt!(infinirtEventSynchronize(self.0))
    }

    #[inline]
    pub fn is_complete(&self) -> bool {
        let mut status = infinirtEventStatus_t::INFINIRT_EVENT_NOT_READY;
        infinirt!(infinirtEventQuery(self.0, &mut status));
        matches!(status, infinirtEventStatus_t::INFINIRT_EVENT_COMPLETE)
    }
}

impl Stream {
    #[inline]
    pub fn record(&self, event: &mut Event) {
        infinirt!(infinirtEventRecord(event.0, self.as_raw()))
    }

    #[inline]
    pub fn wait(&self, event: &Event) {
        infinirt!(infinirtStreamWaitEvent(self.as_raw(), event.0))
    }
}
