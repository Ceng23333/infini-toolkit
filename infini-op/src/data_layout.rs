use crate::bindings::infiniDtype_t;
use digit_layout::{types as ty, DigitLayout};

pub(crate) fn data_layout(dt: DigitLayout) -> infiniDtype_t {
    use infiniDtype_t::*;
    match dt {
        ty::I8 => INFINI_DTYPE_I8,
        ty::I16 => INFINI_DTYPE_I16,
        ty::I32 => INFINI_DTYPE_I32,
        ty::I64 => INFINI_DTYPE_I64,
        ty::U8 => INFINI_DTYPE_U8,
        ty::U16 => INFINI_DTYPE_U16,
        ty::U32 => INFINI_DTYPE_U32,
        ty::U64 => INFINI_DTYPE_U64,
        ty::F16 => INFINI_DTYPE_F16,
        ty::BF16 => INFINI_DTYPE_BF16,
        ty::F32 => INFINI_DTYPE_F32,
        ty::F64 => INFINI_DTYPE_F64,
        _ => panic!("unsupported data type {:?}", dt),
    }
}

#[allow(dead_code)]
pub(crate) fn digit_layout(dt: infiniDtype_t) -> DigitLayout {
    use infiniDtype_t::*;
    match dt {
        INFINI_DTYPE_I8 => ty::I8,
        INFINI_DTYPE_I16 => ty::I16,
        INFINI_DTYPE_I32 => ty::I32,
        INFINI_DTYPE_I64 => ty::I64,
        INFINI_DTYPE_U8 => ty::U8,
        INFINI_DTYPE_U16 => ty::U16,
        INFINI_DTYPE_U32 => ty::U32,
        INFINI_DTYPE_U64 => ty::U64,
        INFINI_DTYPE_F16 => ty::F16,
        INFINI_DTYPE_BF16 => ty::BF16,
        INFINI_DTYPE_F32 => ty::F32,
        INFINI_DTYPE_F64 => ty::F64,
        _ => panic!("unsupported data type {:?}", dt),
    }
}
