use infini_op::{infiniop, AsRaw, Descriptor, Handle, Tensor};

pub fn rearrange() {
    const D1: usize = 2;
    const D2: usize = 3;
    let shape = [D1, D2];

    let x_data: Vec<f32> = (0..shape.iter().product()).map(|x| x as f32).collect();
    let mut y_data: Vec<f32> = vec![0.0; shape.iter().product()];

    // Reference implementation
    let mut y_ref: Vec<f32> = vec![0.0; shape.iter().product()];
    for i in 0..D1 {
        for j in 0..D2 {
            y_ref[j * D1 + i] = x_data[i * D2 + j];
        }
    }

    let handle = Handle::new();
    let dt = digit_layout::types::F32;

    let x = Tensor::new(dt, shape, row_major_strides(&shape, dt.nbytes()));
    let y = Tensor::new(dt, shape, column_major_strides(&shape, dt.nbytes()));

    let op_desc = unsafe {
        let handle = handle.as_raw();
        Descriptor::new(
            |ptr| {
                infiniop! { infini_op::bindings::infiniopCreateRearrangeDescriptor(handle, ptr, y.as_raw(), x.as_raw()) }
            },
            infini_op::bindings::infiniopDestroyRearrangeDescriptor,
        )
    };

    infiniop! { infini_op::bindings::infiniopRearrange(
        op_desc.as_raw(),
        y_data.as_mut_ptr() as _,
        x_data.as_ptr() as _,
        std::ptr::null_mut(),
    ) };

    for i in 0..y_data.len() {
        assert!((y_data[i] - y_ref[i]).abs() < 1e-6);
    }
    println!("Rearrange test passed!");
}

fn row_major_strides(shape: &[usize], nbytes: usize) -> Vec<isize> {
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

fn column_major_strides(shape: &[usize], nbytes: usize) -> Vec<isize> {
    let mut strides = vec![0; shape.len()];
    if shape.is_empty() {
        return strides;
    }
    strides[0] = nbytes as isize;
    for i in 1..shape.len() {
        strides[i] = strides[i - 1] * shape[i - 1] as isize;
    }
    strides
}
