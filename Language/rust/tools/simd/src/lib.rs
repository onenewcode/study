#![feature(portable_simd)]
#![feature(test)]
#![feature(f16)]
#![feature(stdarch_x86_avx512)]
#![feature(avx512_target_feature)]
#![feature(iter_array_chunks)]
use std::simd::{self, i32x4, i8x32, num::SimdInt};

use x86_64::{
    consts::{GGML_F16_ARR, GGML_F16_EPR, GGML_F16_STEP},
    ggml_f32_cx16_load, ggml_f32x16_reduce,
};
#[allow(dead_code)]
#[cfg(target_arch = "x86_64")]
mod x86_64;

#[repr(C, packed)]
#[derive(Debug, Clone)]
pub struct BlockQ8_0 {
    pub d: f16,       // delta
    pub qs: [i8; 32], // quants
}

pub fn vec_dot_q8_naive(n: usize, x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
    let mut result: f32 = 0.0;
    for i in 0..(n / 32) {
        let mut tmp = 0.0;
        for j in 0..32 {
            tmp += (x[i].qs[j] as i32 * y[i].qs[j] as i32) as f32;
        }
        result += tmp * x[i as usize].d as f32 * y[i as usize].d as f32;
    }
    result
}

pub fn vec_dot_q8_stdsimd(n: usize, x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
    let mut sumf: f32 = 0.0;
    for i in 0..n / 32 {
        let mut sumi: i32 = 0;
        for j in 0..8 {
            let ax = i32x4::from_array([
                x[i].qs[j * 4] as i32,
                x[i].qs[j * 4 + 1] as i32,
                x[i].qs[j * 4 + 2] as i32,
                x[i].qs[j * 4 + 3] as i32,
            ]);
            let bx = i32x4::from_array([
                y[i].qs[j * 4] as i32,
                y[i].qs[j * 4 + 1] as i32,
                y[i].qs[j * 4 + 2] as i32,
                y[i].qs[j * 4 + 3] as i32,
            ]);
            sumi += (ax * bx).reduce_sum();
        }
        sumf += sumi as f32 * x[i].d as f32  * y[i].d as f32 ;
    }

    sumf
}

#[cfg(target_os = "linux")]
#[link(name = "ggml")]
extern "C" {
    fn ggml_vec_dot_q8_0_q8_0(
        n: i32,               // number of elements
        s: *mut f32,          // result
        bs: usize,            // not used?
        vx: *const BlockQ8_0, // binary of quantized vec x
        bx: usize,            // not used?
        vy: *const BlockQ8_0, // binary of quantized vec y
        by: usize,            // not used?
        nrc: i32,             // always 1?
    );
}
#[cfg(target_os = "linux")]
pub fn vec_dot_q8_ggml(n: usize, x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
    let mut result: f32 = 0.0;
    unsafe {
        ggml_vec_dot_q8_0_q8_0(
            n as i32,
            &mut result as *mut f32,
            0,
            x.as_ptr(),
            0,
            y.as_ptr(),
            0,
            1,
        );
    }
    result
}
