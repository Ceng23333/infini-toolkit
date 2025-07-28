use infini_op::{infiniop, AsRaw, Descriptor, Handle, Tensor};

pub fn mul() {
    const N: usize = 13;
    const C: usize = 4;
    let shape = [N, C];

    let a_data: Vec<f32> = (0..N * C).map(|x| x as f32).collect();
    let b_data: Vec<f32> = (0..N * C).map(|x| (x * 2) as f32).collect();
    let mut c_data: Vec<f32> = vec![0.0; N * C];

    let mut c_ref: Vec<f32> = vec![0.0; N * C];
    for i in 0..(N * C) {
        c_ref[i] = a_data[i] * b_data[i];
    }

    let handle = Handle::new();
    let dt = digit_layout::types::F32;
    let strides = [(C * dt.nbytes()) as isize, dt.nbytes() as isize];
    let a = Tensor::new(dt, shape, strides);
    let b = Tensor::new(dt, shape, strides);
    let c = Tensor::new(dt, shape, strides);

    let op_desc = unsafe {
        let handle = handle.as_raw();
        let a = a.as_raw();
        let b = b.as_raw();
        let c = c.as_raw();
        Descriptor::new(
            |ptr| infiniop! { infini_op::bindings::infiniopCreateMulDescriptor(handle, ptr, c, a, b) },
            infini_op::bindings::infiniopDestroyMulDescriptor,
        )
    };

    let mut workspace_size: usize = 0;
    infiniop! { infini_op::bindings::infiniopGetMulWorkspaceSize(op_desc.as_raw(), &mut workspace_size) };
    let mut workspace: Vec<u8> = vec![0; workspace_size];

    infiniop! { infini_op::bindings::infiniopMul(
        op_desc.as_raw(),
        workspace.as_mut_ptr() as _,
        workspace_size,
        c_data.as_mut_ptr() as _,
        a_data.as_ptr() as _,
        b_data.as_ptr() as _,
        std::ptr::null_mut(),
    ) };

    for i in 0..(N * C) {
        assert!((c_data[i] - c_ref[i]).abs() < 1e-6);
    }
    println!("Mul test passed!");
}
