use std::arch::x86_64::{__m256i, _mm256_add_epi32, _mm256_castsi256_si128, _mm256_cvtepi8_epi32, _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_maddubs_epi16, _mm256_mullo_epi32, _mm256_setzero_si256, _mm_add_epi32, _mm_cvtsi128_si32, _mm_cvtsi64_si128, _mm_extract_epi16, _mm_hadd_epi16, _mm_setzero_si128, _mm_shuffle_epi32};

use crate::BlockQ8_0;

#[target_feature(enable = "avx2")]
pub unsafe fn vec_dot_q8_avx2(blocks: usize, x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
    let mut result = 0.0_f32;

    for i in 0..blocks {
        let xi = &x[i];
        let yi = &y[i];

        // 计算标量因子 (delta_x * delta_y)
        let d = (xi.d  as f32 * yi.d  as f32);

        // 获取量化值数组指针
        let x_ptr = xi.qs.as_ptr();
        let y_ptr = yi.qs.as_ptr();

        // 初始化累加器
        let mut acc = _mm256_setzero_si256();

        // 每次处理8个元素 (共4组)
        for j in (0..32).step_by(8) {
            // 加载8个int8值（64位）
            let x_bits = std::ptr::read_unaligned(x_ptr.add(j) as *const u64);
            let y_bits = std::ptr::read_unaligned(y_ptr.add(j) as *const u64);

            // 解包int8到int32 (8个值)
            let x_vec = _mm256_cvtepi8_epi32(_mm_cvtsi64_si128(x_bits as i64));
            let y_vec = _mm256_cvtepi8_epi32(_mm_cvtsi64_si128(y_bits as i64));

            // 乘积累加
            let prod = _mm256_mullo_epi32(x_vec, y_vec);
            acc = _mm256_add_epi32(acc, prod);
        }

        // 水平求和累加器中的8个int32
        let sum = horizontal_sum_epi32(acc);
        result += d * sum as f32;
    }

    result
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
