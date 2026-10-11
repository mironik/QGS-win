//! Audited SIMD kernels for the single-core decoder backend.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[allow(clippy::too_many_arguments)]
pub(crate) fn blend_average_u16(
    src0: &[u16],
    stride0: usize,
    x0: usize,
    y0: usize,
    src1: &[u16],
    stride1: usize,
    x1: usize,
    y1: usize,
    dst: &mut [u16],
    dst_stride: usize,
    dst_x: usize,
    dst_y: usize,
    width: usize,
    height: usize,
) -> bool {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 was detected and callers provide validated regions.
        unsafe {
            blend_average_u16_avx2(
                src0, stride0, x0, y0, src1, stride1, x1, y1, dst, dst_stride, dst_x, dst_y, width,
                height,
            )
        }
        return true;
    }
    false
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn blend_average_u16_avx2(
    src0: &[u16],
    stride0: usize,
    x0: usize,
    y0: usize,
    src1: &[u16],
    stride1: usize,
    x1: usize,
    y1: usize,
    dst: &mut [u16],
    dst_stride: usize,
    dst_x: usize,
    dst_y: usize,
    width: usize,
    height: usize,
) {
    for row in 0..height {
        let row0 = (y0 + row) * stride0 + x0;
        let row1 = (y1 + row) * stride1 + x1;
        let dst_row = (dst_y + row) * dst_stride + dst_x;
        let mut column = 0;
        while column + 16 <= width {
            unsafe {
                let a = _mm256_loadu_si256(src0.as_ptr().add(row0 + column).cast());
                let b = _mm256_loadu_si256(src1.as_ptr().add(row1 + column).cast());
                _mm256_storeu_si256(
                    dst.as_mut_ptr().add(dst_row + column).cast(),
                    _mm256_avg_epu16(a, b),
                );
            }
            column += 16;
        }
        while column + 8 <= width {
            unsafe {
                let a = _mm_loadu_si128(src0.as_ptr().add(row0 + column).cast());
                let b = _mm_loadu_si128(src1.as_ptr().add(row1 + column).cast());
                _mm_storeu_si128(
                    dst.as_mut_ptr().add(dst_row + column).cast(),
                    _mm_avg_epu16(a, b),
                );
            }
            column += 8;
        }
        while column < width {
            dst[dst_row + column] =
                ((u32::from(src0[row0 + column]) + u32::from(src1[row1 + column]) + 1) >> 1) as u16;
            column += 1;
        }
    }
}
