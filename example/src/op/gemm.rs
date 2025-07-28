use infini_op::{infiniop, AsRaw, Descriptor, Handle, Tensor};

pub fn gemm() {
    const M: usize = 6;
    const K: usize = 2048;
    const N: usize = 2560;
    let a_shape = [M, K];
    let b_shape = [K, N];
    let c_shape = [M, N];

    let alpha = 1.0;
    let beta = 1.0;

    let a_data: Vec<f32> = (0..M * K).map(|x| x as f32).collect();
    let b_data: Vec<f32> = (0..K * N).map(|x| x as f32).collect();
    let mut c_data: Vec<f32> = (0..M * N).map(|_| 1.0).collect();

    let mut c_ref: Vec<f32> = vec![1.0; M * N];
    for i in 0..M {
        for j in 0..N {
            let mut sum = 0.0;
            for k in 0..K {
                sum += a_data[i * K + k] * b_data[j * K + k];
            }
            c_ref[i * N + j] = alpha * sum + beta * c_ref[i * N + j];
        }
    }

    let handle = Handle::new();
    let dt = digit_layout::types::F32;

    let a = Tensor::new(
        dt,
        a_shape,
        [(K * dt.nbytes()) as isize, dt.nbytes() as isize],
    );
    let b = Tensor::new(
        dt,
        b_shape,
        [dt.nbytes() as isize, (K * dt.nbytes()) as isize],
    );
    let c = Tensor::new(
        dt,
        c_shape,
        [(N * dt.nbytes()) as isize, dt.nbytes() as isize],
    );

    let op_desc = unsafe {
        let handle = handle.as_raw();
        let a = a.as_raw();
        let b = b.as_raw();
        let c = c.as_raw();
        Descriptor::new(
            |ptr| infiniop! { infini_op::bindings::infiniopCreateGemmDescriptor(handle, ptr, c, a, b) },
            infini_op::bindings::infiniopDestroyGemmDescriptor,
        )
    };

    let mut workspace_size: usize = 0;
    infiniop! { infini_op::bindings::infiniopGetGemmWorkspaceSize(op_desc.as_raw(), &mut workspace_size) };
    let mut workspace: Vec<u8> = vec![0; workspace_size];

    infiniop! { infini_op::bindings::infiniopGemm(
        op_desc.as_raw(),
        workspace.as_mut_ptr() as _,
        workspace_size,
        c_data.as_mut_ptr() as _,
        a_data.as_ptr() as _,
        b_data.as_ptr() as _,
        alpha,
        beta,
        std::ptr::null_mut(),
    ) };

    for i in 0..(M * N) {
        assert!((c_data[i] - c_ref[i]).abs() < 1e-3);
    }
    println!("Gemm test passed!");
}
