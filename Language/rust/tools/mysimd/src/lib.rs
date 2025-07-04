#![feature(test)]
#![feature(f16)]
#![feature(portable_simd)]
#![allow(soft_unstable)]
#![feature(iter_array_chunks)]
mod x86_x64;
use std::simd::{i32x4, num::SimdInt};

use rand::Rng;

#[repr(C, packed)]
#[derive(Debug, Clone)]
pub struct BlockQ8_0 {
    d: f16,       // delta
    qs: [i8; 32], // quants
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
        sumf += sumi as f32 * x[i].d as f32 * y[i].d as f32;
    }

    sumf
}
// generate a random vector of BlockQ8_0
fn gen_rand_block_q8_0() -> BlockQ8_0 {
    let mut rng = rand::rng();
    let d: f32 = rng.random_range(-10.0..10.0);
    let mut qs: [i8; 32] = [0; 32];
    for i in 0..32 {
        qs[i] = rng.random();
    }
    BlockQ8_0 { d: d as f16, qs }
}

pub fn gen_rand_block_q8_0_vec(n: usize) -> Vec<BlockQ8_0> {
    let mut v: Vec<BlockQ8_0> = Vec::with_capacity(n);
    for _ in 0..n {
        v.push(gen_rand_block_q8_0());
    }
    v
}
#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    extern crate test;
    use test::Bencher;

    const TEST_BLOCKS: usize = 1000;
    const TEST_ELEMS: usize = TEST_BLOCKS * 32;

    #[test]
    fn test_vec_dot_q8() {
        let v1 = gen_rand_block_q8_0_vec(4);
        let v2 = gen_rand_block_q8_0_vec(4);

        let naive_result = vec_dot_q8_naive(64, &v1, &v2);
        let result = vec_dot_q8_stdsimd(64, &v1, &v2);
        assert!((result - naive_result).abs() < 1e-2);
    }
}
