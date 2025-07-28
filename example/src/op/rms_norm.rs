use infini_op::{infiniop, AsRaw, Descriptor, Handle, Tensor};

pub fn rms_norm() {
    const D1: usize = 1;
    const D2: usize = 4;
    let x_shape = [D1, D2];
    let w_shape = [D2];

    let x_data: Vec<f32> = (0..x_shape.iter().product())
        .map(|x| (x as f32) * 0.01)
        .collect();
    let w_data: Vec<f32> = (0..w_shape.iter().product()).map(|x| x as f32).collect();
    let mut y_data: Vec<f32> = vec![0.0; x_shape.iter().product()];

    // Reference implementation
    let mut y_ref: Vec<f32> = vec![0.0; x_shape.iter().product()];
    let eps = 1e-6;
    for i in 0..D1 {
        let offset = i * D2;
        let x_slice = &x_data[offset..offset + D2];
        let mean_sq: f32 = x_slice.iter().map(|&x| x * x).sum::<f32>() / D2 as f32;
        let rrms = (mean_sq + eps).sqrt().recip();
        for j in 0..D2 {
            y_ref[offset + j] = x_slice[j] * rrms * w_data[j];
        }
    }

    let handle = Handle::new();
    let dt = digit_layout::types::F32;

    let x = Tensor::new(dt, x_shape, default_strides(&x_shape, dt.nbytes()));
    let y = Tensor::new(dt, x_shape, default_strides(&x_shape, dt.nbytes()));
    let w = Tensor::new(dt, w_shape, default_strides(&w_shape, dt.nbytes()));

    let op_desc = unsafe {
        let handle = handle.as_raw();
        Descriptor::new(
            |ptr| {
                infiniop! { infini_op::bindings::infiniopCreateRMSNormDescriptor(
                    handle,
                    ptr,
                    y.as_raw(),
                    x.as_raw(),
                    w.as_raw(),
                    eps
                ) }
            },
            infini_op::bindings::infiniopDestroyRMSNormDescriptor,
        )
    };

    let mut workspace_size: usize = 0;
    infiniop! { infini_op::bindings::infiniopGetRMSNormWorkspaceSize(op_desc.as_raw(), &mut workspace_size) };
    let mut workspace: Vec<u8> = vec![0; workspace_size];

    infiniop! { infini_op::bindings::infiniopRMSNorm(
        op_desc.as_raw(),
        workspace.as_mut_ptr() as _,
        workspace_size,
        y_data.as_mut_ptr() as _,
        x_data.as_ptr() as _,
        w_data.as_ptr() as _,
        std::ptr::null_mut(),
    ) };

    for i in 0..y_data.len() {
        assert!((y_data[i] - y_ref[i]).abs() < 1e-6);
    }
    println!("RMSNorm test passed!");
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
