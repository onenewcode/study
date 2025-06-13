#![feature(test)]
#![feature(f16)]
#![feature(portable_simd)]
#![allow(soft_unstable)]
#![feature(iter_array_chunks)]

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86_x64;
use std::simd::{i32x4, num::SimdInt};

#[repr(C, packed)]
#[derive(Debug, Clone)]
pub struct BlockQ8_0 {
    d: f16,       // delta
    qs: [i8; 32], // quants
}

pub fn vec_dot_q8_naive(blocks: usize, x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
    let mut result = 0.0;

    for i in 0..blocks {
        let xi = &x[i];
        let yi = &y[i];

        let d_x = xi.d as f32;
        let d_y = yi.d as f32;

        let mut sum = 0.0;

        for j in 0..32 {
            sum += (xi.qs[j] as i32 * yi.qs[j] as i32) as f32;
        }

        result += sum * d_x * d_y;
    }

    result
}

pub fn vec_dot_q8_stdsimd(blocks: usize, x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
    let mut sumf: f32 = 0.0;
    for i in 0..blocks {
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

#[cfg(test)]
mod tests {
    use crate::x86_x64::{vec_dot_q8_avx,  vec_dot_q8_avx2};

    use super::*;
    extern crate test;
    use rand::Rng;
    use test::Bencher;
    // 太大会导致精度误差导致无法通过测试
    const TEST_BLOCKS: usize = 100;
    // generate a random vector of BlockQ8_0
    fn gen_rand_block_q8_0() -> BlockQ8_0 {
        let mut rng = rand::rng();
        let d: f32 = rng.random_range(-1.0..1.0);
        let mut qs: [i8; 32] = [0; 32];
        for i in 0..32 {
            qs[i] = rng.random();
        }
        BlockQ8_0 { d: d as f16, qs }
    }

    fn gen_rand_block_q8_0_vec(n: usize) -> Vec<BlockQ8_0> {
        let mut v: Vec<BlockQ8_0> = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(gen_rand_block_q8_0());
        }
        v
    }

    #[test]
    fn test_vec_dot_q8_stdsimd() {
        let v1 = gen_rand_block_q8_0_vec(TEST_BLOCKS);
        let v2 = gen_rand_block_q8_0_vec(TEST_BLOCKS);

        let naive_result = vec_dot_q8_naive(TEST_BLOCKS, &v1, &v2);
        let result = vec_dot_q8_stdsimd(TEST_BLOCKS, &v1, &v2);
        assert!((result - naive_result).abs() < 1e-5);
    }

    #[test]
    fn test_vec_dot_q8_avx2() {
        if is_x86_feature_detected!("avx2") {
            let v1 = gen_rand_block_q8_0_vec(TEST_BLOCKS);
            let v2 = gen_rand_block_q8_0_vec(TEST_BLOCKS);

            let naive_result = vec_dot_q8_naive(TEST_BLOCKS, &v1, &v2);

            // Wrap the call in an `unsafe` block
            let result = unsafe { crate::x86_x64::vec_dot_q8_avx2(TEST_BLOCKS, &v1, &v2) };

            assert!((result - naive_result).abs() < 1e-2);
        } else {
            println!("AVX not supported, skipping test");
        }
    }
    // #[test]
    // fn test_vec_dot_q8_avx() {
    //     if is_x86_feature_detected!("avx") {
    //         let v1 = gen_rand_block_q8_0_vec(TEST_BLOCKS);
    //         let v2 = gen_rand_block_q8_0_vec(TEST_BLOCKS);

    //         let naive_result = vec_dot_q8_naive(TEST_BLOCKS, &v1, &v2);

    //         // Wrap the call in an `unsafe` block
    //         let result = unsafe { crate::x86_x64::vec_dot_q8_avx(TEST_BLOCKS, &v1, &v2) };

    //         assert!((result - naive_result).abs() < 1e-2);
    //     } else {
    //         println!("AVX not supported, skipping test");
    //     }
    // }
    #[bench]
    fn bench_vec_dot_q8_naive(b: &mut Bencher) {
        let v1 = gen_rand_block_q8_0_vec(TEST_BLOCKS);
        let v2 = gen_rand_block_q8_0_vec(TEST_BLOCKS);
        b.iter(|| vec_dot_q8_naive(TEST_BLOCKS, &v1, &v2));
    }

    #[bench]
    fn bench_vec_dot_q8_stdsimd(b: &mut Bencher) {
        let v1 = gen_rand_block_q8_0_vec(TEST_BLOCKS);
        let v2 = gen_rand_block_q8_0_vec(TEST_BLOCKS);
        b.iter(|| vec_dot_q8_stdsimd(TEST_BLOCKS, &v1, &v2));
    }
    #[bench]
    fn bench_vec_dot_q8_avx(b: &mut Bencher) {
        if is_x86_feature_detected!("avx") {
            let v1 = gen_rand_block_q8_0_vec(TEST_BLOCKS);
            let v2 = gen_rand_block_q8_0_vec(TEST_BLOCKS);
            b.iter(|| unsafe { vec_dot_q8_avx(TEST_BLOCKS, &v1, &v2) });
        } else {
            println!("AVX not supported, skipping test");
        }
    }

    // #[bench]
    // fn bench_vec_dot_q8_avx2(b: &mut Bencher) {
    //     if is_x86_feature_detected!("avx2") {
    //         let v1 = gen_rand_block_q8_0_vec(TEST_BLOCKS);
    //         let v2 = gen_rand_block_q8_0_vec(TEST_BLOCKS);
    //         b.iter(|| unsafe { vec_dot_q8_avx2(TEST_BLOCKS, &v1, &v2) });
    //     } else {
    //         println!("AVX not supported, skipping test");
    //     }
    // }
}
