use infini_op::{infiniop, AsRaw, Descriptor, Handle, Tensor};

pub fn rope() {
    const D1: usize = 1;
    const D2: usize = 32;
    const D3: usize = 128;
    let shape = [D1, D2, D3];

    let x_data: Vec<f32> = (0..shape.iter().product()).map(|x| x as f32).collect();
    let mut y_data: Vec<f32> = vec![0.0; shape.iter().product()];

    // Reference implementation
    let mut y_ref: Vec<f32> = vec![0.0; shape.iter().product()];
    let pos: Vec<i32> = (0..D1 as i32).collect();
    let theta = 1e5_f32;
    let dim = D3;
    let freqs: Vec<f32> = (0..dim)
        .step_by(2)
        .map(|i| 1.0 / (theta.powf((i as f32) / (dim as f32))))
        .collect();
    let angles: Vec<f32> = pos
        .iter()
        .flat_map(|&p| freqs.iter().map(move |&f| p as f32 * f))
        .collect();
    let sin_table: Vec<f32> = angles.iter().map(|&a| a.sin()).collect();
    let cos_table: Vec<f32> = angles.iter().map(|&a| a.cos()).collect();

    for i in 0..D1 {
        for j in 0..D2 {
            for k in 0..(D3 / 2) {
                let x_even_idx = i * D2 * D3 + j * D3 + 2 * k;
                let x_odd_idx = i * D2 * D3 + j * D3 + 2 * k + 1;
                let sin_idx = i * (D3 / 2) + k;
                let cos_idx = i * (D3 / 2) + k;

                let x_even = x_data[x_even_idx];
                let x_odd = x_data[x_odd_idx];
                let sin = sin_table[sin_idx];
                let cos = cos_table[cos_idx];

                y_ref[x_even_idx] = x_even * cos - x_odd * sin;
                y_ref[x_odd_idx] = x_even * sin + x_odd * cos;
            }
        }
    }

    let handle = Handle::new();
    let dt = digit_layout::types::F32;
    let pos_dt = digit_layout::types::I32;

    let x = Tensor::new(dt, shape, default_strides(&shape, dt.nbytes()));
    let y = Tensor::new(dt, shape, default_strides(&shape, dt.nbytes()));
    let pos_tensor = Tensor::new(pos_dt, [D1], default_strides(&[D1], pos_dt.nbytes()));
    let sin_cos_shape = [D1, D3 / 2];
    let sin_table_tensor = Tensor::new(
        dt,
        sin_cos_shape,
        default_strides(&sin_cos_shape, dt.nbytes()),
    );
    let cos_table_tensor = Tensor::new(
        dt,
        sin_cos_shape,
        default_strides(&sin_cos_shape, dt.nbytes()),
    );

    let op_desc = unsafe {
        let handle = handle.as_raw();
        Descriptor::new(
            |ptr| {
                infiniop! { infini_op::bindings::infiniopCreateRoPEDescriptor(
                    handle,
                    ptr,
                    y.as_raw(),
                    x.as_raw(),
                    pos_tensor.as_raw(),
                    sin_table_tensor.as_raw(),
                    cos_table_tensor.as_raw()
                ) }
            },
            infini_op::bindings::infiniopDestroyRoPEDescriptor,
        )
    };

    let mut workspace_size: usize = 0;
    infiniop! { infini_op::bindings::infiniopGetRoPEWorkspaceSize(op_desc.as_raw(), &mut workspace_size) };
    let mut workspace: Vec<u8> = vec![0; workspace_size];

    infiniop! { infini_op::bindings::infiniopRoPE(
        op_desc.as_raw(),
        workspace.as_mut_ptr() as _,
        workspace_size,
        y_data.as_mut_ptr() as _,
        x_data.as_ptr() as _,
        pos.as_ptr() as _,
        sin_table.as_ptr() as _,
        cos_table.as_ptr() as _,
        std::ptr::null_mut(),
    ) };

    for i in 0..y_data.len() {
        assert!((y_data[i] - y_ref[i]).abs() < 1e-3);
    }
    println!("RoPE test passed!");
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
