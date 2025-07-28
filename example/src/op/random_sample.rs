use infini_op::{infiniop, AsRaw, Descriptor, Handle, Tensor};

pub fn random_sample() {
    const VOC: usize = 512;
    const RANDOM_VAL: f32 = 0.8;
    const TOP_P: f32 = 0.8;
    const TOP_K: i32 = 3;
    const TEMPERATURE: f32 = 0.5;

    let logits_data: Vec<f32> = (0..VOC).map(|i| (i as f32) * 0.0001).collect();
    let mut indices_data: Vec<i32> = vec![0];

    let handle = Handle::new();
    let dt = digit_layout::types::F32;
    let indices_dt = digit_layout::types::I32;

    let logits = Tensor::new(dt, [VOC], default_strides(&[VOC], dt.nbytes()));
    let indices = Tensor::new(indices_dt, [], default_strides(&[], indices_dt.nbytes()));

    let op_desc = unsafe {
        let handle = handle.as_raw();
        Descriptor::new(
            |ptr| {
                infiniop! { infini_op::bindings::infiniopCreateRandomSampleDescriptor(
                    handle,
                    ptr,
                    indices.as_raw(),
                    logits.as_raw()
                ) }
            },
            infini_op::bindings::infiniopDestroyRandomSampleDescriptor,
        )
    };

    let mut workspace_size: usize = 0;
    infiniop! { infini_op::bindings::infiniopGetRandomSampleWorkspaceSize(op_desc.as_raw(), &mut workspace_size) };
    let mut workspace: Vec<u8> = vec![0; workspace_size];

    infiniop! { infini_op::bindings::infiniopRandomSample(
        op_desc.as_raw(),
        workspace.as_mut_ptr() as _,
        workspace_size,
        indices_data.as_mut_ptr() as _,
        logits_data.as_ptr() as _,
        RANDOM_VAL,
        TOP_P,
        TOP_K,
        TEMPERATURE,
        std::ptr::null_mut(),
    ) };

    println!(
        "RandomSample test ran. Output index: {} (verification skipped).",
        indices_data[0]
    );
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
