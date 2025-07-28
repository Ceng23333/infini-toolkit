use infini_op::{infiniop, AsRaw, Descriptor, Handle, Tensor};

pub fn causal_softmax() {
    const D1: usize = 2;
    const D2: usize = 3;
    const D3: usize = 4;
    let shape = [D1, D2, D3];

    let x_data: Vec<f32> = (0..shape.iter().product()).map(|x| x as f32).collect();
    let mut y_data: Vec<f32> = vec![0.0; shape.iter().product()];

    // Reference implementation
    let mut y_ref: Vec<f32> = vec![0.0; shape.iter().product()];
    for i in 0..D1 {
        for j in 0..D2 {
            let offset = (i * D2 + j) * D3;
            let input_slice = &x_data[offset..offset + D3];
            let mut masked_input = input_slice.to_vec();

            for c in 0..D3 {
                // This is the rust equivalent of the python code:
                // mask = torch.tril(torch.ones_like(x), diagonal=-1).flip(dims=[-2, -1])
                // The condition for the mask to be 1 at (j, c) is (D3 - 1 - c) < (D2 - 1 - j)
                if (D3 - 1 - c) < (D2 - 1 - j) {
                    masked_input[c] = f32::NEG_INFINITY;
                }
            }

            // Softmax
            let max_val = masked_input
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exps: Vec<f32> = masked_input.iter().map(|&x| (x - max_val).exp()).collect();
            let sum_exps: f32 = exps.iter().sum();
            let softmax_output: Vec<f32> = exps.iter().map(|&e| e / sum_exps).collect();

            y_ref[offset..offset + D3].copy_from_slice(&softmax_output);
        }
    }

    let handle = Handle::new();
    let dt = digit_layout::types::F32;

    let x = Tensor::new(dt, shape, default_strides(&shape, dt.nbytes()));
    let y = Tensor::new(dt, shape, default_strides(&shape, dt.nbytes()));

    let op_desc = unsafe {
        let handle = handle.as_raw();
        Descriptor::new(
            |ptr| {
                infiniop! { infini_op::bindings::infiniopCreateCausalSoftmaxDescriptor(handle, ptr, y.as_raw(), x.as_raw()) }
            },
            infini_op::bindings::infiniopDestroyCausalSoftmaxDescriptor,
        )
    };

    let mut workspace_size: usize = 0;
    infiniop! { infini_op::bindings::infiniopGetCausalSoftmaxWorkspaceSize(op_desc.as_raw(), &mut workspace_size) };
    let mut workspace: Vec<u8> = vec![0; workspace_size];

    infiniop! { infini_op::bindings::infiniopCausalSoftmax(
        op_desc.as_raw(),
        workspace.as_mut_ptr() as _,
        workspace_size,
        y_data.as_mut_ptr() as _,
        x_data.as_ptr() as _,
        std::ptr::null_mut(),
    ) };

    for i in 0..y_data.len() {
        assert!((y_data[i] - y_ref[i]).abs() < 1e-5);
    }
    println!("CausalSoftmax test passed!");
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
