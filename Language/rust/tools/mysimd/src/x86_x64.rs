use std::arch::x86_64::{
    __m256i, _mm_add_epi32, _mm_cvtsi64_si128, _mm_cvtsi128_si32, _mm_extract_epi16,
    _mm_hadd_epi16, _mm_setzero_si128, _mm_shuffle_epi32, _mm256_add_epi32, _mm256_castsi256_si128,
    _mm256_cvtepi8_epi32, _mm256_extracti128_si256, _mm256_load_si256, _mm256_loadu_si256,
    _mm256_maddubs_epi16, _mm256_mullo_epi32, _mm256_setzero_si256,
};

use crate::BlockQ8_0;

#[target_feature(enable = "avx2,avx")]
pub unsafe fn vec_dot_q8_avx2(blocks: usize, x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
    // 初始化累加器
    let mut acc = _mm256_setzero_ps();
    for i in 0..blocks {
        let xi = &x[i];
        let yi = &y[i];

        unsafe {
            let d = _mm256_set1_ps(xi.d as f32 * yi.d as f32);
            // avx支持加载256个元素
            let x_ptr = xi.qs.as_ptr() as *const __m256i;
            let y_ptr = yi.qs.as_ptr() as *const __m256i;
            let x_data = _mm256_loadu_si256(x_ptr);
            let y_data = _mm256_loadu_si256(y_ptr);
            let q = mul_sum_i8_pairs_float(x_data, y_data);
            acc = _mm256_fmadd_ps(d, q, acc);
        }
    }
    unsafe { hsum_float_8(acc) }
}

// 辅助函数：水平求和256位寄存器中的8个int32
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_epi32(v: __m256i) -> i32 {
    // 交换高/低128位并相加
    let v_high = _mm256_extracti128_si256(v, 1);
    let v_low = _mm256_castsi256_si128(v);
    let sum128 = _mm_add_epi32(v_high, v_low);

    // 继续在128位寄存器中水平求和
    let sum64 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0x4E));
    let sum32 = _mm_add_epi32(sum64, _mm_shuffle_epi32(sum64, 0xB1));

    _mm_cvtsi128_si32(sum32)
}
use std::arch::x86_64::*;

// 将两个i8向量相乘，两次成对相加结果，并以f32向量返回
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn mul_sum_i8_pairs_float(x: __m256i, y: __m256i) -> __m256 {
    if is_x86_feature_detected!("avxvnniint8") {
        let zero = _mm256_setzero_si256();
        let summed_pairs = _mm256_dpbssd_epi32(zero, x, y);
        _mm256_cvtepi32_ps(summed_pairs)
    } else {
        let ax = _mm256_sign_epi8(x, x); // x的绝对值
        let sy = _mm256_sign_epi8(y, x); // 带符号的y值
        mul_sum_us8_pairs_float(ax, sy)
    }
}
#[target_feature(enable = "avx2")]
// 无符号i8与有符号i8相乘并求和
#[inline]
unsafe fn mul_sum_us8_pairs_float(ax: __m256i, sy: __m256i) -> __m256 {
    if is_x86_feature_detected!("avx512vnni") {
        if is_x86_feature_detected!("avx512vl") {
            let zero = _mm256_setzero_si256();
            let summed_pairs = _mm256_dpbusd_epi32(zero, ax, sy);
            _mm256_cvtepi32_ps(summed_pairs)
        } else {
            let zero = _mm256_setzero_si256();
            let summed_pairs = _mm256_dpbusd_epi32(zero, ax, sy);
            _mm256_cvtepi32_ps(summed_pairs)
        }
    } else {
        let dot = _mm256_maddubs_epi16(ax, sy); // 乘加生成i16
        sum_i16_pairs_float(dot)
    }
}

// 将i16成对相加并转换为f32向量
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn sum_i16_pairs_float(x: __m256i) -> __m256 {
    let ones = _mm256_set1_epi16(1);
    let summed_pairs = _mm256_madd_epi16(ones, x); // 成对相乘并相加
    _mm256_cvtepi32_ps(summed_pairs)
}
#[target_feature(enable = "avx")]
pub unsafe fn hsum_float_8(x: __m256) -> f32 {
    // 提取高128位
    let high: __m128 = _mm256_extractf128_ps(x, 1);
    // 获取低128位并加上高128位
    let low: __m128 = _mm256_castps256_ps128(x);
    let mut sum: __m128 = _mm_add_ps(low, high);

    // 水平求和：高位两个元素 + 低位两个元素
    let high_half: __m128 = _mm_movehl_ps(sum, sum);
    sum = _mm_add_ps(sum, high_half);

    // 复制高位到低位并相加
    let duplicated: __m128 = _mm_movehdup_ps(sum);
    sum = _mm_add_ss(sum, duplicated);

    // 提取结果
    _mm_cvtss_f32(sum)
}
