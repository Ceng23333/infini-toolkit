mod op;

use op::{
    add, attention, causal_softmax, clip, gemm, mul, random_sample, rearrange, rms_norm, rope, sub,
    swiglu,
};

fn main() {
    let cases = std::env::args().collect::<Vec<String>>();
    let all = cases.len() == 1;
    if all || cases.contains(&"add".to_string()) {
        add::add();
    }
    if all || cases.contains(&"gemm".to_string()) {
        gemm::gemm();
    }
    if all || cases.contains(&"mul".to_string()) {
        mul::mul();
    }
    if all || cases.contains(&"sub".to_string()) {
        sub::sub();
    }
    if all || cases.contains(&"attention".to_string()) {
        attention::attention();
    }
    if all || cases.contains(&"causal_softmax".to_string()) {
        causal_softmax::causal_softmax();
    }
    if all || cases.contains(&"clip".to_string()) {
        clip::clip();
    }
    if all || cases.contains(&"random_sample".to_string()) {
        random_sample::random_sample();
    }
    if all || cases.contains(&"rearrange".to_string()) {
        rearrange::rearrange();
    }
    if all || cases.contains(&"rms_norm".to_string()) {
        rms_norm::rms_norm();
    }
    if all || cases.contains(&"rope".to_string()) {
        rope::rope();
    }
    if all || cases.contains(&"swiglu".to_string()) {
        swiglu::swiglu();
    }
}
