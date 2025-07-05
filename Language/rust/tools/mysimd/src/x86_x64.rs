use std::arch::x86_64::{__m128, __m128i};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::BlockQ8_0;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
pub unsafe fn vec_dot_q8_avx2(n: usize, a: &[BlockQ8_0], b: &[BlockQ8_0]) -> f32 {
    use std::arch::x86_64::*;

    let mut sumv = _mm256_setzero_ps();
    let zero = _mm256_setzero_si256();

    for i in 0..n / 32 {
        let ab = a.get_unchecked(i);
        let bb = b.get_unchecked(i);

        // 加载32个int8值
        let a_vec = _mm256_loadu_si256(ab.qs.as_ptr() as *const __m256i);
        let b_vec = _mm256_loadu_si256(bb.qs.as_ptr() as *const __m256i);

        // 扩展int8到int16（有符号）
        let a_low = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a_vec, 0));
        let a_high = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a_vec, 1));
        let b_low = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_vec, 0));
        let b_high = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_vec, 1));

        // 计算点积（乘加）
        let prod_low = _mm256_madd_epi16(a_low, b_low);
        let prod_high = _mm256_madd_epi16(a_high, b_high);

        // 合并结果
        let sum_i32 = _mm256_add_epi32(prod_low, prod_high);

        // 转换为浮点数
        let sum_f32 = _mm256_cvtepi32_ps(sum_i32);

        // 应用缩放因子
        let scale = ab.d as f32 * bb.d as f32 ;
        let scale_vec = _mm256_set1_ps(scale);
        sumv = _mm256_fmadd_ps(sum_f32, scale_vec, sumv);
    }

    // 水平求和
    let sum_high = _mm256_extractf128_ps(sumv, 1);
    let sum_low = _mm256_castps256_ps128(sumv);
    let sum = _mm_add_ps(sum_high, sum_low);
    let sum = _mm_hadd_ps(sum, sum);
    let sum = _mm_hadd_ps(sum, sum);
    _mm_cvtss_f32(sum)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
pub unsafe fn vec_dot_q8_avx2_unrolled(n: usize, a: &[BlockQ8_0], b: &[BlockQ8_0]) -> f32 {
    use std::arch::x86_64::*;

    let mut sumv0 = _mm256_setzero_ps();
    let mut sumv1 = _mm256_setzero_ps();
    let zero = _mm256_setzero_si256();

    for i in (0..n / 32).step_by(2) {
        let ab0 = a.get_unchecked(i);
        let ab1 = a.get_unchecked(i + 1);
        let bb0 = b.get_unchecked(i);
        let bb1 = b.get_unchecked(i + 1);

        // 处理第一个块
        let a_vec0 = _mm256_loadu_si256(ab0.qs.as_ptr() as *const __m256i);
        let b_vec0 = _mm256_loadu_si256(bb0.qs.as_ptr() as *const __m256i);
        let a_low0 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a_vec0, 0));
        let a_high0 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a_vec0, 1));
        let b_low0 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_vec0, 0));
        let b_high0 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_vec0, 1));
        let prod_low0 = _mm256_madd_epi16(a_low0, b_low0);
        let prod_high0 = _mm256_madd_epi16(a_high0, b_high0);
        let sum_i320 = _mm256_add_epi32(prod_low0, prod_high0);
        let sum_f320 = _mm256_cvtepi32_ps(sum_i320);
        let scale0 = ab0.d as f32  * bb0.d as f32 ;
        let scale_vec0 = _mm256_set1_ps(scale0);
        sumv0 = _mm256_fmadd_ps(sum_f320, scale_vec0, sumv0);

        // 处理第二个块
        let a_vec1 = _mm256_loadu_si256(ab1.qs.as_ptr() as *const __m256i);
        let b_vec1 = _mm256_loadu_si256(bb1.qs.as_ptr() as *const __m256i);
        let a_low1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a_vec1, 0));
        let a_high1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a_vec1, 1));
        let b_low1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_vec1, 0));
        let b_high1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_vec1, 1));
        let prod_low1 = _mm256_madd_epi16(a_low1, b_low1);
        let prod_high1 = _mm256_madd_epi16(a_high1, b_high1);
        let sum_i321 = _mm256_add_epi32(prod_low1, prod_high1);
        let sum_f321 = _mm256_cvtepi32_ps(sum_i321);
        let scale1 = ab1.d as f32  * bb1.d as f32 ;
        let scale_vec1 = _mm256_set1_ps(scale1);
        sumv1 = _mm256_fmadd_ps(sum_f321, scale_vec1, sumv1);
    }

    // 合并结果
    let sumv = _mm256_add_ps(sumv0, sumv1);
    let sum_high = _mm256_extractf128_ps(sumv, 1);
    let sum_low = _mm256_castps256_ps128(sumv);
    let sum = _mm_add_ps(sum_high, sum_low);
    let sum = _mm_hadd_ps(sum, sum);
    let sum = _mm_hadd_ps(sum, sum);
    _mm_cvtss_f32(sum)
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse4.1")]
#[target_feature(enable = "fma")]
pub unsafe fn vec_dot_q8_avx(blocks: usize, a: &[BlockQ8_0], b: &[BlockQ8_0]) -> f32 {
    use std::arch::x86_64::*;

    let mut sumv = _mm_setzero_ps();
    let mut sum_acc = _mm_setzero_ps();

    for i in 0..n  {
        let ab = a.get_unchecked(i);
        let bb = b.get_unchecked(i);

        // 加载16个int8值（第一部分）
        let a_vec0 = _mm_loadu_si128(ab.qs.as_ptr() as *const __m128i);
        let b_vec0 = _mm_loadu_si128(bb.qs.as_ptr() as *const __m128i);

        // 加载16个int8值（第二部分）
        let a_vec1 = _mm_loadu_si128(ab.qs.as_ptr().add(16) as *const __m128i);
        let b_vec1 = _mm_loadu_si128(bb.qs.as_ptr().add(16) as *const __m128i);

        // 计算第一部分点积
        let dot0 = dot_product_sse(a_vec0, b_vec0);

        // 计算第二部分点积
        let dot1 = dot_product_sse(a_vec1, b_vec1);

        // 合并点积结果
        let dot_sum = _mm_add_epi32(dot0, dot1);

        // 转换为浮点数
        let dot_f32 = _mm_cvtepi32_ps(dot_sum);

        // 应用缩放因子
        let scale = ab.d as f32 * bb.d as f32;
        let scale_vec = _mm_set1_ps(scale);

        // 乘积累加
        sumv = _mm_fmadd_ps(dot_f32, scale_vec, sumv);
    }

    // 水平求和
    horizontal_sum_sse(sumv)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse4.1")]

pub unsafe fn vec_dot_q8_avx_unrolled(n: usize, a: &[BlockQ8_0], b: &[BlockQ8_0]) -> f32 {
    use std::arch::x86_64::*;

    let mut sumv0 = _mm_setzero_ps();
    let mut sumv1 = _mm_setzero_ps();

    for i in (0..n).step_by(2) {
        let ab0 = a.get_unchecked(i);
        let ab1 = a.get_unchecked(i + 1);
        let bb0 = b.get_unchecked(i);
        let bb1 = b.get_unchecked(i + 1);

        // 处理第一个块（第一部分）
        let a00 = _mm_loadu_si128(ab0.qs.as_ptr() as *const __m128i);
        let b00 = _mm_loadu_si128(bb0.qs.as_ptr() as *const __m128i);
        let a01 = _mm_loadu_si128(ab0.qs.as_ptr().add(16) as *const __m128i);
        let b01 = _mm_loadu_si128(bb0.qs.as_ptr().add(16) as *const __m128i);

        // 处理第二个块（第一部分）
        let a10 = _mm_loadu_si128(ab1.qs.as_ptr() as *const __m128i);
        let b10 = _mm_loadu_si128(bb1.qs.as_ptr() as *const __m128i);
        let a11 = _mm_loadu_si128(ab1.qs.as_ptr().add(16) as *const __m128i);
        let b11 = _mm_loadu_si128(bb1.qs.as_ptr().add(16) as *const __m128i);

        // 计算点积
        let dot00 = dot_product_sse(a00, b00);
        let dot01 = dot_product_sse(a01, b01);
        let dot0 = _mm_add_epi32(dot00, dot01);

        let dot10 = dot_product_sse(a10, b10);
        let dot11 = dot_product_sse(a11, b11);
        let dot1 = _mm_add_epi32(dot10, dot11);

        // 应用缩放因子并累加
        let scale0 = ab0.d as f32 * bb0.d as f32;
        let scale1 = ab1.d as f32* bb1.d as f32;

        let dot0_f32 = _mm_cvtepi32_ps(dot0);
        let dot1_f32 = _mm_cvtepi32_ps(dot1);

        sumv0 = _mm_fmadd_ps(dot0_f32, _mm_set1_ps(scale0), sumv0);
        sumv1 = _mm_fmadd_ps(dot1_f32, _mm_set1_ps(scale1), sumv1);
    }

    // 合并结果
    let sumv = _mm_add_ps(sumv0, sumv1);
    horizontal_sum_sse(sumv)
}

// SSE点积辅助函数
#[target_feature(enable = "sse4.1")]
unsafe fn dot_product_sse(a: __m128i, b: __m128i) -> __m128i {
    use std::arch::x86_64::*;

    // 解包低8位到16位
    let a_low = _mm_cvtepi8_epi16(a);
    let b_low = _mm_cvtepi8_epi16(b);

    // 解包高8位到16位
    let a_high = _mm_cvtepi8_epi16(_mm_srli_si128(a, 8));
    let b_high = _mm_cvtepi8_epi16(_mm_srli_si128(b, 8));

    // 计算乘加
    let prod_low = _mm_madd_epi16(a_low, b_low);
    let prod_high = _mm_madd_epi16(a_high, b_high);

    // 合并结果
    _mm_add_epi32(prod_low, prod_high)
}

// SSE水平求和辅助函数
#[target_feature(enable = "sse,sse3")]
unsafe fn horizontal_sum_sse(v: __m128) -> f32 {
    use std::arch::x86_64::*;

    // 水平求和：交换高低64位并相加
    let shuf = _mm_movehdup_ps(v);   // 复制高部分到低部分 (b,b,a,a)
    let sum1 = _mm_add_ps(v, shuf);  // (a+b, b+a, ...)

    // 移动高64位到低64位
    let movhl = _mm_movehl_ps(shuf, sum1); // 移动高64位到低64位
    let sum2 = _mm_add_ss(sum1, movhl);    // 标量加法

    _mm_cvtss_f32(sum2)
}
