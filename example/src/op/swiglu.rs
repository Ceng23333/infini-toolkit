use infini_op::{infiniop, AsRaw, Descriptor, Handle, Tensor};

pub fn swiglu() {
    const D1: usize = 13;
    const D2: usize = 4;
    let shape = [D1, D2];

    let a_data: Vec<f32> = (0..shape.iter().product()).map(|x| x as f32).collect();
    let b_data: Vec<f32> = (0..shape.iter().product()).map(|x| x as f32).collect();
    let mut c_data: Vec<f32> = vec![0.0; shape.iter().product()];

    // Reference implementation
    let mut c_ref: Vec<f32> = vec![0.0; shape.iter().product()];
    for i in 0..c_ref.len() {
        c_ref[i] = a_data[i] * b_data[i] / (1.0 + (-b_data[i]).exp());
    }

    let handle = Handle::new();
    let dt = digit_layout::types::F32;

    let a = Tensor::new(dt, shape, default_strides(&shape, dt.nbytes()));
    let b = Tensor::new(dt, shape, default_strides(&shape, dt.nbytes()));
    let c = Tensor::new(dt, shape, default_strides(&shape, dt.nbytes()));

    let op_desc = unsafe {
        let handle = handle.as_raw();
        Descriptor::new(
            |ptr| {
                infiniop! { infini_op::bindings::infiniopCreateSwiGLUDescriptor(
                    handle,
                    ptr,
                    c.as_raw(),
                    a.as_raw(),
                    b.as_raw(),
                ) }
            },
            infini_op::bindings::infiniopDestroySwiGLUDescriptor,
        )
    };

    let mut workspace_size: usize = 0;
    infiniop! { infini_op::bindings::infiniopGetSwiGLUWorkspaceSize(op_desc.as_raw(), &mut workspace_size) };
    let mut workspace: Vec<u8> = vec![0; workspace_size];

    infiniop! { infini_op::bindings::infiniopSwiGLU(
        op_desc.as_raw(),
        workspace.as_mut_ptr() as _,
        workspace_size,
        c_data.as_mut_ptr() as _,
        a_data.as_ptr() as _,
        b_data.as_ptr() as _,
        std::ptr::null_mut(),
    ) };

    for i in 0..c_data.len() {
        assert!((c_data[i] - c_ref[i]).abs() < 1e-3);
    }
    println!("SwiGLU test passed!");
}

fn default_strides(shape: &[usize], nbytes: usize) -> Vec<isize> {
    let mut strides = vec![0; shape.len()];
    if shape.is_empty() {
        return strides;
    }
    strides[shape.len() - 1] = nbytes as isize;
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1] as isize;
    }
    strides
}
