use infini_op::{infiniop, AsRaw, Descriptor, Handle, Tensor};

pub fn clip() {
    const SHAPE: [usize; 1] = [10];
    const MIN_VAL: f32 = -1.0;
    const MAX_VAL: f32 = 1.0;

    let x_data: Vec<f32> = (-5..5).map(|x| x as f32).collect();
    let mut y_data: Vec<f32> = vec![0.0; SHAPE[0]];

    // Reference implementation
    let mut y_ref: Vec<f32> = vec![0.0; SHAPE[0]];
    for i in 0..SHAPE[0] {
        y_ref[i] = x_data[i].clamp(MIN_VAL, MAX_VAL);
    }

    let handle = Handle::new();
    let dt = digit_layout::types::F32;

    let x = Tensor::new(dt, SHAPE, default_strides(&SHAPE, dt.nbytes()));
    let y = Tensor::new(dt, SHAPE, default_strides(&SHAPE, dt.nbytes()));
    // For min/max, use zero strides for broadcasting
    let min_tensor = Tensor::new(dt, SHAPE, vec![0; SHAPE.len()]);
    let max_tensor = Tensor::new(dt, SHAPE, vec![0; SHAPE.len()]);

    let op_desc = unsafe {
        let handle = handle.as_raw();
        Descriptor::new(
            |ptr| {
                infiniop! { infini_op::bindings::infiniopCreateClipDescriptor(
                    handle,
                    ptr,
                    y.as_raw(),
                    x.as_raw(),
                    min_tensor.as_raw(),
                    max_tensor.as_raw()
                ) }
            },
            infini_op::bindings::infiniopDestroyClipDescriptor,
        )
    };

    let mut workspace_size: usize = 0;
    infiniop! { infini_op::bindings::infiniopGetClipWorkspaceSize(op_desc.as_raw(), &mut workspace_size) };
    let mut workspace: Vec<u8> = vec![0; workspace_size];

    let min_data = [MIN_VAL];
    let max_data = [MAX_VAL];

    infiniop! { infini_op::bindings::infiniopClip(
        op_desc.as_raw(),
        workspace.as_mut_ptr() as _,
        workspace_size,
        y_data.as_mut_ptr() as _,
        x_data.as_ptr() as _,
        min_data.as_ptr() as _,
        max_data.as_ptr() as _,
        std::ptr::null_mut(),
    ) };

    for i in 0..y_data.len() {
        assert!((y_data[i] - y_ref[i]).abs() < 1e-6);
    }
    println!("Clip test passed!");
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
