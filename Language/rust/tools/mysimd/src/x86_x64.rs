use std::arch::x86_64::{
    __m256i, _mm_add_epi32, _mm_cvtsi128_si32, _mm_shuffle_epi32, _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_loadu_si256,
    _mm256_maddubs_epi16, _mm256_setzero_si256,
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
            // 设置f32，便于使用_mm256_fmadd_ps直接进行mat mul 操作
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
#[target_feature(enable = "avx,sse2")]
pub unsafe fn vec_dot_q8_avx(blocks: usize, x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
    // 初始化累加器
    let mut acc = _mm256_setzero_ps();
    for [f, s] in (0..blocks).array_chunks::<2>() {
        let x_ib = &x[f];
        let x_ib1 = &x[s];
        let y_ib = &y[f];
        let y_ib1 = &y[s];
        unsafe {
            // sse2  是使用128位宽
            let qx1_0 = _mm_loadu_si128(x_ib.qs.as_ptr() as *const __m128i);
            let qx1_1 = _mm_loadu_si128(x_ib.qs.as_ptr().add(16) as *const __m128i);
            let qx2_0 = _mm_loadu_si128(x_ib1.qs.as_ptr() as *const __m128i);
            let qx2_1 = _mm_loadu_si128(x_ib1.qs.as_ptr().add(16) as *const __m128i);

            let qy1_0 = _mm_loadu_si128(y_ib.qs.as_ptr() as *const __m128i);
            let qy1_1 = _mm_loadu_si128(y_ib.qs.as_ptr().add(16) as *const __m128i);
            let qy2_0 = _mm_loadu_si128(y_ib1.qs.as_ptr() as *const __m128i);
            let qy2_1 = _mm_loadu_si128(y_ib1.qs.as_ptr().add(16) as *const __m128i);
            let deltas =
                quad_fp16_delta_float(x[f].d as f32, y[f].d as f32, x[s].d as f32, y[s].d as f32);
            let p = mul_sum_i8_quad_float(qx1_0, qx1_1, qx2_0, qx2_1, qy1_0, qy1_1, qy2_0, qy2_1);
            acc = _mm256_add_ps(_mm256_mul_ps(deltas, p), acc);
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
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn mul_sum_i8_pairs_float(x: __m256i, y: __m256i) -> __m256 {
    unsafe {
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
}
// 无符号i8与有符号i8相乘并求和
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn mul_sum_us8_pairs_float(ax: __m256i, sy: __m256i) -> __m256 {
    unsafe {
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
}
// 将i16成对相加并转换为f32向量
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn sum_i16_pairs_float(x: __m256i) -> __m256 {
    let ones = _mm256_set1_epi16(1);
    let summed_pairs = _mm256_madd_epi16(ones, x); // 成对相乘并相加
    _mm256_cvtepi32_ps(summed_pairs)
}
#[inline]
#[target_feature(enable = "avx")]
pub unsafe fn hsum_float_8(x: __m256) -> f32 {
    let mut res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    _mm_cvtss_f32(res)
}
#[inline]
#[target_feature(enable = "avx")]
unsafe fn quad_fp16_delta_float(x0: f32, y0: f32, x1: f32, y1: f32) -> __m256 {
    // 使用 _mm256_set_m128 需要创建两个 __m128 类型的变量
    let upper = _mm_set1_ps(x1 * y1);
    let lower = _mm_set1_ps(x0 * y0);

    // 合并为一个 __m256 类型
    _mm256_set_m128(upper, lower)
}
// Compute dot products for two blocks
#[inline]
#[target_feature(enable = "sse2,avx,ssse3")]
unsafe fn mul_sum_i8_quad_float(
    x1_0: __m128i,
    x1_1: __m128i,
    x2_0: __m128i,
    x2_1: __m128i,
    y1_0: __m128i,
    y1_1: __m128i,
    y2_0: __m128i,
    y2_1: __m128i,
) -> __m256 {
    let mone = _mm_set1_epi16(1);

    let p16_1_0 = mul_add_epi8_sse(x1_0, y1_0);
    let p16_1_1 = mul_add_epi8_sse(x1_1, y1_1);
    let p16_2_0 = mul_add_epi8_sse(x2_0, y2_0);
    let p16_2_1 = mul_add_epi8_sse(x2_1, y2_1);

    let p_1_0 = _mm_madd_epi16(p16_1_0, mone);
    let p_1_1 = _mm_madd_epi16(p16_1_1, mone);
    let p_2_0 = _mm_madd_epi16(p16_2_0, mone);
    let p_2_1 = _mm_madd_epi16(p16_2_1, mone);

    let p_1 = _mm_add_epi32(p_1_0, p_1_1);
    let p_2 = _mm_add_epi32(p_2_0, p_2_1);
    _mm256_cvtepi32_ps(mm256_set_m128i(p_2, p_1))
}
#[inline]
#[target_feature(enable = "ssse3")]
fn mul_add_epi8_sse(x: __m128i, y: __m128i) -> __m128i {
    let ax = _mm_sign_epi8(x, x);
    let sy = _mm_sign_epi8(y, x);
    _mm_maddubs_epi16(ax, sy)
}
#[inline]
#[target_feature(enable = "avx")]
fn mm256_set_m128i(a: __m128i, b: __m128i) -> __m256i {
    _mm256_insertf128_si256::<1>(_mm256_castsi128_si256(b), a)
}
