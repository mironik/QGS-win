//! Block motion compensation for High 4:2:2 10-bit.
//!
//! Fractional positions match the per-sample path: luma uses the H.264 six-tap
//! filter, quarter-pel samples average one side of the half-pel square, and
//! chroma uses the eighth-pel bilinear filter. Out-of-frame taps repeat the
//! edge sample.

use std::cell::RefCell;

use crate::frame::Plane422P10;
use crate::motion::{MotionCompensationError, MotionVectorQuarterPel};
use wide::{i32x8, u16x8};

const TILE: usize = 16;
const BUF_STRIDE: usize = TILE + 1;

struct BiScratch {
    first: [u16; TILE * TILE],
    second: [u16; TILE * TILE],
}

struct LumaGridScratch {
    first: [u16; BUF_STRIDE * TILE],
    second: [u16; BUF_STRIDE * TILE],
}

thread_local! {
    static BI_SCRATCH: RefCell<BiScratch> = const { RefCell::new(BiScratch {
        first: [0; TILE * TILE],
        second: [0; TILE * TILE],
    }) };
    static LUMA_GRID_SCRATCH: RefCell<LumaGridScratch> = const { RefCell::new(LumaGridScratch {
        first: [0; BUF_STRIDE * TILE],
        second: [0; BUF_STRIDE * TILE],
    }) };
}

pub(crate) fn put_luma(
    reference: &Plane422P10,
    target: &mut Plane422P10,
    dst_x: usize,
    dst_y: usize,
    width: usize,
    height: usize,
    motion: MotionVectorQuarterPel,
) -> Result<(), MotionCompensationError> {
    if width == 0 || height == 0 {
        return Ok(());
    }
    check_luma_motion(dst_x, dst_y, width, height, motion)?;
    check_destination(&target, dst_x, dst_y, width, height)?;
    let (ox, oy, fx, fy) = luma_origin(dst_x, dst_y, motion)?;
    let src_w = i32::try_from(reference.width).map_err(|_| MotionCompensationError::OutOfBounds)?;
    let src_h =
        i32::try_from(reference.height).map_err(|_| MotionCompensationError::OutOfBounds)?;
    if src_w == 0 || src_h == 0 {
        return Err(MotionCompensationError::OutOfBounds);
    }
    let src = reference.samples();
    let dst_stride = target.width;
    let dst = target.samples_mut();
    tile_luma(
        src, src_w, src_h, dst, dst_stride, dst_x, dst_y, width, height, ox, oy, fx, fy,
    );
    Ok(())
}

/// Produces luma prediction for an absolute destination region without writing
/// it to the decoded frame. This is the entry point used by fused
/// prediction-plus-residual reconstruction.
pub(crate) fn predict_luma_block(
    reference: &Plane422P10,
    output: &mut [u16],
    output_stride: usize,
    dst_x: usize,
    dst_y: usize,
    width: usize,
    height: usize,
    motion: MotionVectorQuarterPel,
) -> Result<(), MotionCompensationError> {
    if width == 0 || height == 0 {
        return Ok(());
    }
    if output_stride < width
        || output.len()
            < output_stride
                .saturating_mul(height.saturating_sub(1))
                .saturating_add(width)
    {
        return Err(MotionCompensationError::OutOfBounds);
    }
    check_luma_motion(dst_x, dst_y, width, height, motion)?;
    let (ox, oy, fx, fy) = luma_origin(dst_x, dst_y, motion)?;
    let src_w = i32::try_from(reference.width).map_err(|_| MotionCompensationError::OutOfBounds)?;
    let src_h =
        i32::try_from(reference.height).map_err(|_| MotionCompensationError::OutOfBounds)?;
    if src_w == 0 || src_h == 0 {
        return Err(MotionCompensationError::OutOfBounds);
    }
    tile_luma(
        reference.samples(),
        src_w,
        src_h,
        output,
        output_stride,
        0,
        0,
        width,
        height,
        ox,
        oy,
        fx,
        fy,
    );
    Ok(())
}

pub(crate) fn put_luma_bi(
    list0: &Plane422P10,
    list1: &Plane422P10,
    target: &mut Plane422P10,
    dst_x: usize,
    dst_y: usize,
    width: usize,
    height: usize,
    motion0: MotionVectorQuarterPel,
    motion1: MotionVectorQuarterPel,
    weight0: i32,
    weight1: i32,
) -> Result<(), MotionCompensationError> {
    if width == 0 || height == 0 {
        return Ok(());
    }
    check_luma_motion(dst_x, dst_y, width, height, motion0)?;
    check_luma_motion(dst_x, dst_y, width, height, motion1)?;
    check_destination(&target, dst_x, dst_y, width, height)?;
    let src0_w = i32::try_from(list0.width).map_err(|_| MotionCompensationError::OutOfBounds)?;
    let src0_h = i32::try_from(list0.height).map_err(|_| MotionCompensationError::OutOfBounds)?;
    let src1_w = i32::try_from(list1.width).map_err(|_| MotionCompensationError::OutOfBounds)?;
    let src1_h = i32::try_from(list1.height).map_err(|_| MotionCompensationError::OutOfBounds)?;
    if src0_w == 0 || src0_h == 0 || src1_w == 0 || src1_h == 0 {
        return Err(MotionCompensationError::OutOfBounds);
    }
    let (ox0, oy0, fx0, fy0) = luma_origin(dst_x, dst_y, motion0)?;
    let (ox1, oy1, fx1, fy1) = luma_origin(dst_x, dst_y, motion1)?;
    let src0 = list0.samples();
    let src1 = list1.samples();
    let dst_stride = target.width;
    let dst = target.samples_mut();

    if fx0 == 0
        && fy0 == 0
        && fx1 == 0
        && fy1 == 0
        && integer_region_inside(ox0, oy0, width, height, src0_w, src0_h)
        && integer_region_inside(ox1, oy1, width, height, src1_w, src1_h)
    {
        blend_integer_luma(
            src0,
            src0_w as usize,
            ox0 as usize,
            oy0 as usize,
            src1,
            src1_w as usize,
            ox1 as usize,
            oy1 as usize,
            dst,
            dst_stride,
            dst_x,
            dst_y,
            width,
            height,
            weight0,
            weight1,
        );
        return Ok(());
    }

    let mut y = 0;
    while y < height {
        let tile_h = (height - y).min(TILE);
        let mut x = 0;
        while x < width {
            let tile_w = (width - x).min(TILE);
            BI_SCRATCH.with(|scratch| {
                let mut scratch = scratch.borrow_mut();
                let BiScratch { first, second } = &mut *scratch;
                tile_luma(
                    src0,
                    src0_w,
                    src0_h,
                    first,
                    tile_w,
                    0,
                    0,
                    tile_w,
                    tile_h,
                    ox0 + x as i32,
                    oy0 + y as i32,
                    fx0,
                    fy0,
                );
                tile_luma(
                    src1,
                    src1_w,
                    src1_h,
                    second,
                    tile_w,
                    0,
                    0,
                    tile_w,
                    tile_h,
                    ox1 + x as i32,
                    oy1 + y as i32,
                    fx1,
                    fy1,
                );
                for row in 0..tile_h {
                    let dst_index = (dst_y + y + row) * dst_stride + dst_x + x;
                    let mut column = 0;
                    while column + 8 <= tile_w {
                        let index = row * tile_w + column;
                        let values = weighted_avg_x8(
                            load_u16x8(first, index),
                            load_u16x8(second, index),
                            weight0,
                            weight1,
                        );
                        dst[dst_index + column..dst_index + column + 8].copy_from_slice(&values);
                        column += 8;
                    }
                    while column < tile_w {
                        let index = row * tile_w + column;
                        dst[dst_index + column] =
                            weighted_avg(first[index], second[index], weight0, weight1);
                        column += 1;
                    }
                }
            });
            x += tile_w;
        }
        y += tile_h;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn predict_luma_bi_block(
    list0: &Plane422P10,
    list1: &Plane422P10,
    output: &mut [u16; TILE * TILE],
    dst_x: usize,
    dst_y: usize,
    motion0: MotionVectorQuarterPel,
    motion1: MotionVectorQuarterPel,
    weight0: i32,
    weight1: i32,
) -> Result<(), MotionCompensationError> {
    check_luma_motion(dst_x, dst_y, TILE, TILE, motion0)?;
    check_luma_motion(dst_x, dst_y, TILE, TILE, motion1)?;
    let src0_w = i32::try_from(list0.width).map_err(|_| MotionCompensationError::OutOfBounds)?;
    let src0_h = i32::try_from(list0.height).map_err(|_| MotionCompensationError::OutOfBounds)?;
    let src1_w = i32::try_from(list1.width).map_err(|_| MotionCompensationError::OutOfBounds)?;
    let src1_h = i32::try_from(list1.height).map_err(|_| MotionCompensationError::OutOfBounds)?;
    if src0_w == 0 || src0_h == 0 || src1_w == 0 || src1_h == 0 {
        return Err(MotionCompensationError::OutOfBounds);
    }
    let (ox0, oy0, fx0, fy0) = luma_origin(dst_x, dst_y, motion0)?;
    let (ox1, oy1, fx1, fy1) = luma_origin(dst_x, dst_y, motion1)?;
    let src0 = list0.samples();
    let src1 = list1.samples();
    if fx0 == 0
        && fy0 == 0
        && fx1 == 0
        && fy1 == 0
        && integer_region_inside(ox0, oy0, TILE, TILE, src0_w, src0_h)
        && integer_region_inside(ox1, oy1, TILE, TILE, src1_w, src1_h)
    {
        blend_integer_luma(
            src0,
            src0_w as usize,
            ox0 as usize,
            oy0 as usize,
            src1,
            src1_w as usize,
            ox1 as usize,
            oy1 as usize,
            output,
            TILE,
            0,
            0,
            TILE,
            TILE,
            weight0,
            weight1,
        );
        return Ok(());
    }
    BI_SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        let BiScratch { first, second } = &mut *scratch;
        tile_luma(
            src0, src0_w, src0_h, first, TILE, 0, 0, TILE, TILE, ox0, oy0, fx0, fy0,
        );
        tile_luma(
            src1, src1_w, src1_h, second, TILE, 0, 0, TILE, TILE, ox1, oy1, fx1, fy1,
        );
        for row in 0..TILE {
            let row_start = row * TILE;
            for column in (0..TILE).step_by(8) {
                let index = row_start + column;
                let values = weighted_avg_x8(
                    load_u16x8(first, index),
                    load_u16x8(second, index),
                    weight0,
                    weight1,
                );
                output[index..index + 8].copy_from_slice(&values);
            }
        }
    });
    Ok(())
}

#[inline]
fn integer_region_inside(
    x: i32,
    y: i32,
    width: usize,
    height: usize,
    source_width: i32,
    source_height: i32,
) -> bool {
    x >= 0 && y >= 0 && x + width as i32 <= source_width && y + height as i32 <= source_height
}

#[allow(clippy::too_many_arguments)]
fn blend_integer_luma(
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
    weight0: i32,
    weight1: i32,
) {
    if weight0 == 32
        && weight1 == 32
        && crate::simd::blend_average_u16(
            src0, stride0, x0, y0, src1, stride1, x1, y1, dst, dst_stride, dst_x, dst_y, width,
            height,
        )
    {
        return;
    }
    for row in 0..height {
        let row0 = (y0 + row) * stride0 + x0;
        let row1 = (y1 + row) * stride1 + x1;
        let dst_row = (dst_y + row) * dst_stride + dst_x;
        let mut column = 0;
        while column + 8 <= width {
            let values = weighted_avg_x8(
                load_u16x8(src0, row0 + column),
                load_u16x8(src1, row1 + column),
                weight0,
                weight1,
            );
            dst[dst_row + column..dst_row + column + 8].copy_from_slice(&values);
            column += 8;
        }
        while column < width {
            dst[dst_row + column] =
                weighted_avg(src0[row0 + column], src1[row1 + column], weight0, weight1);
            column += 1;
        }
    }
}

pub(crate) fn put_chroma(
    cb_ref: &Plane422P10,
    cr_ref: &Plane422P10,
    cb_dst: &mut Plane422P10,
    cr_dst: &mut Plane422P10,
    dst_x: usize,
    dst_y: usize,
    width: usize,
    height: usize,
    motion: MotionVectorQuarterPel,
) -> Result<(), MotionCompensationError> {
    if width == 0 || height == 0 {
        return Ok(());
    }
    let (ox, oy, dx, dy) = chroma_origin(dst_x, dst_y, width, height, motion)?;
    check_destination(cb_dst, dst_x, dst_y, width, height)?;
    check_destination(cr_dst, dst_x, dst_y, width, height)?;
    let cb_w = i32::try_from(cb_ref.width).map_err(|_| MotionCompensationError::OutOfBounds)?;
    let cb_h = i32::try_from(cb_ref.height).map_err(|_| MotionCompensationError::OutOfBounds)?;
    let cr_w = i32::try_from(cr_ref.width).map_err(|_| MotionCompensationError::OutOfBounds)?;
    let cr_h = i32::try_from(cr_ref.height).map_err(|_| MotionCompensationError::OutOfBounds)?;
    if cb_w == 0 || cb_h == 0 || cr_w == 0 || cr_h == 0 {
        return Err(MotionCompensationError::OutOfBounds);
    }
    let cb_src = cb_ref.samples();
    let cr_src = cr_ref.samples();
    let cb_stride = cb_dst.width;
    let cr_stride = cr_dst.width;
    // Split the mutable borrows: the two destination planes are distinct fields.
    let cb = cb_dst.samples_mut();
    tile_chroma(
        cb_src, cb_w, cb_h, cb, cb_stride, dst_x, dst_y, width, height, ox, oy, dx, dy,
    );
    let cr = cr_dst.samples_mut();
    tile_chroma(
        cr_src, cr_w, cr_h, cr, cr_stride, dst_x, dst_y, width, height, ox, oy, dx, dy,
    );
    Ok(())
}

pub(crate) fn predict_chroma_block(
    cb_ref: &Plane422P10,
    cr_ref: &Plane422P10,
    cb: &mut [u16; 8 * 16],
    cr: &mut [u16; 8 * 16],
    dst_x: usize,
    dst_y: usize,
    motion: MotionVectorQuarterPel,
) -> Result<(), MotionCompensationError> {
    predict_chroma_region_into(cb_ref, cr_ref, cb, cr, 8, 0, 0, dst_x, dst_y, 8, 16, motion)
}

/// Writes a chroma rectangle into the local 8×16 MB buffers at `(local_x, local_y)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn predict_chroma_region_into(
    cb_ref: &Plane422P10,
    cr_ref: &Plane422P10,
    cb: &mut [u16],
    cr: &mut [u16],
    buffer_stride: usize,
    local_x: usize,
    local_y: usize,
    dst_x: usize,
    dst_y: usize,
    width: usize,
    height: usize,
    motion: MotionVectorQuarterPel,
) -> Result<(), MotionCompensationError> {
    if width == 0 || height == 0 {
        return Ok(());
    }
    if local_x + width > buffer_stride
        || cb.len()
            < buffer_stride
                .saturating_mul(local_y.saturating_add(height.saturating_sub(1)))
                .saturating_add(local_x + width)
        || cr.len()
            < buffer_stride
                .saturating_mul(local_y.saturating_add(height.saturating_sub(1)))
                .saturating_add(local_x + width)
    {
        return Err(MotionCompensationError::OutOfBounds);
    }
    let (ox, oy, dx, dy) = chroma_origin(dst_x, dst_y, width, height, motion)?;
    let cb_w = i32::try_from(cb_ref.width).map_err(|_| MotionCompensationError::OutOfBounds)?;
    let cb_h = i32::try_from(cb_ref.height).map_err(|_| MotionCompensationError::OutOfBounds)?;
    let cr_w = i32::try_from(cr_ref.width).map_err(|_| MotionCompensationError::OutOfBounds)?;
    let cr_h = i32::try_from(cr_ref.height).map_err(|_| MotionCompensationError::OutOfBounds)?;
    if cb_w == 0 || cb_h == 0 || cr_w == 0 || cr_h == 0 {
        return Err(MotionCompensationError::OutOfBounds);
    }
    tile_chroma(
        cb_ref.samples(),
        cb_w,
        cb_h,
        cb,
        buffer_stride,
        local_x,
        local_y,
        width,
        height,
        ox,
        oy,
        dx,
        dy,
    );
    tile_chroma(
        cr_ref.samples(),
        cr_w,
        cr_h,
        cr,
        buffer_stride,
        local_x,
        local_y,
        width,
        height,
        ox,
        oy,
        dx,
        dy,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn put_chroma_bi(
    cb0: &Plane422P10,
    cr0: &Plane422P10,
    cb1: &Plane422P10,
    cr1: &Plane422P10,
    cb_dst: &mut Plane422P10,
    cr_dst: &mut Plane422P10,
    dst_x: usize,
    dst_y: usize,
    width: usize,
    height: usize,
    motion0: MotionVectorQuarterPel,
    motion1: MotionVectorQuarterPel,
    weight0: i32,
    weight1: i32,
) -> Result<(), MotionCompensationError> {
    if width == 0 || height == 0 {
        return Ok(());
    }
    let (ox0, oy0, dx0, dy0) = chroma_origin(dst_x, dst_y, width, height, motion0)?;
    let (ox1, oy1, dx1, dy1) = chroma_origin(dst_x, dst_y, width, height, motion1)?;
    check_destination(cb_dst, dst_x, dst_y, width, height)?;
    check_destination(cr_dst, dst_x, dst_y, width, height)?;
    blend_chroma_plane(
        cb0, cb1, cb_dst, dst_x, dst_y, width, height, ox0, oy0, dx0, dy0, ox1, oy1, dx1, dy1,
        weight0, weight1,
    )?;
    blend_chroma_plane(
        cr0, cr1, cr_dst, dst_x, dst_y, width, height, ox0, oy0, dx0, dy0, ox1, oy1, dx1, dy1,
        weight0, weight1,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn predict_chroma_bi_block(
    cb0: &Plane422P10,
    cr0: &Plane422P10,
    cb1: &Plane422P10,
    cr1: &Plane422P10,
    cb: &mut [u16; 8 * 16],
    cr: &mut [u16; 8 * 16],
    dst_x: usize,
    dst_y: usize,
    motion0: MotionVectorQuarterPel,
    motion1: MotionVectorQuarterPel,
    weight0: i32,
    weight1: i32,
) -> Result<(), MotionCompensationError> {
    let (ox0, oy0, dx0, dy0) = chroma_origin(dst_x, dst_y, 8, 16, motion0)?;
    let (ox1, oy1, dx1, dy1) = chroma_origin(dst_x, dst_y, 8, 16, motion1)?;
    predict_chroma_plane_bi_block(
        cb0, cb1, cb, ox0, oy0, dx0, dy0, ox1, oy1, dx1, dy1, weight0, weight1,
    )?;
    predict_chroma_plane_bi_block(
        cr0, cr1, cr, ox0, oy0, dx0, dy0, ox1, oy1, dx1, dy1, weight0, weight1,
    )
}

#[allow(clippy::too_many_arguments)]
fn predict_chroma_plane_bi_block(
    list0: &Plane422P10,
    list1: &Plane422P10,
    output: &mut [u16; 8 * 16],
    ox0: i32,
    oy0: i32,
    dx0: i32,
    dy0: i32,
    ox1: i32,
    oy1: i32,
    dx1: i32,
    dy1: i32,
    weight0: i32,
    weight1: i32,
) -> Result<(), MotionCompensationError> {
    let w0 = i32::try_from(list0.width).map_err(|_| MotionCompensationError::OutOfBounds)?;
    let h0 = i32::try_from(list0.height).map_err(|_| MotionCompensationError::OutOfBounds)?;
    let w1 = i32::try_from(list1.width).map_err(|_| MotionCompensationError::OutOfBounds)?;
    let h1 = i32::try_from(list1.height).map_err(|_| MotionCompensationError::OutOfBounds)?;
    if w0 == 0 || h0 == 0 || w1 == 0 || h1 == 0 {
        return Err(MotionCompensationError::OutOfBounds);
    }
    BI_SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        let BiScratch { first, second } = &mut *scratch;
        interpolate_chroma_tile(
            list0.samples(),
            w0,
            h0,
            first,
            8,
            0,
            0,
            8,
            16,
            ox0,
            oy0,
            dx0,
            dy0,
        );
        interpolate_chroma_tile(
            list1.samples(),
            w1,
            h1,
            second,
            8,
            0,
            0,
            8,
            16,
            ox1,
            oy1,
            dx1,
            dy1,
        );
        for index in 0..8 * 16 {
            output[index] = weighted_avg(first[index], second[index], weight0, weight1);
        }
    });
    Ok(())
}

fn check_destination(
    plane: &Plane422P10,
    dst_x: usize,
    dst_y: usize,
    width: usize,
    height: usize,
) -> Result<(), MotionCompensationError> {
    if dst_x
        .checked_add(width)
        .ok_or(MotionCompensationError::OutOfBounds)?
        > plane.width
        || dst_y
            .checked_add(height)
            .ok_or(MotionCompensationError::OutOfBounds)?
            > plane.height
    {
        return Err(MotionCompensationError::OutOfBounds);
    }
    Ok(())
}

fn check_luma_motion(
    dst_x: usize,
    dst_y: usize,
    width: usize,
    height: usize,
    motion: MotionVectorQuarterPel,
) -> Result<(), MotionCompensationError> {
    let dst_x = i32::try_from(dst_x).map_err(|_| MotionCompensationError::OutOfBounds)?;
    let dst_y = i32::try_from(dst_y).map_err(|_| MotionCompensationError::OutOfBounds)?;
    let last_x = dst_x
        .checked_add(i32::try_from(width - 1).map_err(|_| MotionCompensationError::OutOfBounds)?)
        .ok_or(MotionCompensationError::OutOfBounds)?;
    let last_y = dst_y
        .checked_add(i32::try_from(height - 1).map_err(|_| MotionCompensationError::OutOfBounds)?)
        .ok_or(MotionCompensationError::OutOfBounds)?;
    dst_x
        .checked_mul(4)
        .and_then(|value| value.checked_add(motion.x))
        .ok_or(MotionCompensationError::OutOfBounds)?;
    last_x
        .checked_mul(4)
        .and_then(|value| value.checked_add(motion.x))
        .ok_or(MotionCompensationError::OutOfBounds)?;
    dst_y
        .checked_mul(4)
        .and_then(|value| value.checked_add(motion.y))
        .ok_or(MotionCompensationError::OutOfBounds)?;
    last_y
        .checked_mul(4)
        .and_then(|value| value.checked_add(motion.y))
        .ok_or(MotionCompensationError::OutOfBounds)?;
    Ok(())
}

fn luma_origin(
    dst_x: usize,
    dst_y: usize,
    motion: MotionVectorQuarterPel,
) -> Result<(i32, i32, i32, i32), MotionCompensationError> {
    let ox = i32::try_from(dst_x)
        .ok()
        .and_then(|value| value.checked_add(div_floor(motion.x, 4)))
        .ok_or(MotionCompensationError::OutOfBounds)?;
    let oy = i32::try_from(dst_y)
        .ok()
        .and_then(|value| value.checked_add(div_floor(motion.y, 4)))
        .ok_or(MotionCompensationError::OutOfBounds)?;
    Ok((ox, oy, motion.x.rem_euclid(4), motion.y.rem_euclid(4)))
}

fn chroma_origin(
    dst_x: usize,
    dst_y: usize,
    width: usize,
    height: usize,
    motion: MotionVectorQuarterPel,
) -> Result<(i32, i32, i32, i32), MotionCompensationError> {
    let dst_x_i = i32::try_from(dst_x).map_err(|_| MotionCompensationError::OutOfBounds)?;
    let dst_y_i = i32::try_from(dst_y).map_err(|_| MotionCompensationError::OutOfBounds)?;
    let last_x = dst_x_i
        .checked_add(i32::try_from(width - 1).map_err(|_| MotionCompensationError::OutOfBounds)?)
        .ok_or(MotionCompensationError::OutOfBounds)?;
    let last_y = dst_y_i
        .checked_add(i32::try_from(height - 1).map_err(|_| MotionCompensationError::OutOfBounds)?)
        .ok_or(MotionCompensationError::OutOfBounds)?;
    dst_x_i
        .checked_mul(8)
        .and_then(|value| value.checked_add(motion.x))
        .ok_or(MotionCompensationError::OutOfBounds)?;
    last_x
        .checked_mul(8)
        .and_then(|value| value.checked_add(motion.x))
        .ok_or(MotionCompensationError::OutOfBounds)?;
    let motion_y2 = motion
        .y
        .checked_mul(2)
        .ok_or(MotionCompensationError::OutOfBounds)?;
    dst_y_i
        .checked_mul(4)
        .and_then(|value| value.checked_add(motion.y))
        .and_then(|value| value.checked_mul(2))
        .ok_or(MotionCompensationError::OutOfBounds)?;
    last_y
        .checked_mul(4)
        .and_then(|value| value.checked_add(motion.y))
        .and_then(|value| value.checked_mul(2))
        .ok_or(MotionCompensationError::OutOfBounds)?;
    let ox = dst_x_i
        .checked_add(div_floor(motion.x, 8))
        .ok_or(MotionCompensationError::OutOfBounds)?;
    let oy = dst_y_i
        .checked_add(div_floor(motion_y2, 8))
        .ok_or(MotionCompensationError::OutOfBounds)?;
    Ok((ox, oy, motion.x.rem_euclid(8), motion_y2.rem_euclid(8)))
}

fn tile_luma(
    src: &[u16],
    src_w: i32,
    src_h: i32,
    dst: &mut [u16],
    dst_stride: usize,
    dst_x: usize,
    dst_y: usize,
    width: usize,
    height: usize,
    ox: i32,
    oy: i32,
    fx: i32,
    fy: i32,
) {
    let mut y = 0;
    while y < height {
        let tile_h = (height - y).min(TILE);
        let mut x = 0;
        while x < width {
            let tile_w = (width - x).min(TILE);
            interpolate_luma_tile(
                src,
                src_w,
                src_h,
                dst,
                dst_stride,
                dst_x + x,
                dst_y + y,
                tile_w,
                tile_h,
                ox + x as i32,
                oy + y as i32,
                fx,
                fy,
            );
            x += tile_w;
        }
        y += tile_h;
    }
}

fn interpolate_luma_tile(
    src: &[u16],
    src_w: i32,
    src_h: i32,
    dst: &mut [u16],
    dst_stride: usize,
    dst_x: usize,
    dst_y: usize,
    width: usize,
    height: usize,
    ox: i32,
    oy: i32,
    fx: i32,
    fy: i32,
) {
    debug_assert!(width <= TILE && height <= TILE);
    if fx == 0 && fy == 0 {
        copy_integer(
            src, src_w, src_h, dst, dst_stride, dst_x, dst_y, width, height, ox, oy,
        );
        return;
    }

    let (kind_a, ox_a, oy_a, kind_b, ox_b, oy_b) = luma_pair(ox, oy, fx, fy);
    if kind_b == Grid::None {
        let dst_index = dst_y * dst_stride + dst_x;
        fill_luma_grid(
            &mut dst[dst_index..],
            dst_stride,
            src,
            src_w,
            src_h,
            ox_a,
            oy_a,
            width,
            height,
            kind_a,
        );
        return;
    }
    LUMA_GRID_SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        let LumaGridScratch { first, second } = &mut *scratch;
        fill_luma_grid(
            first, BUF_STRIDE, src, src_w, src_h, ox_a, oy_a, width, height, kind_a,
        );
        fill_luma_grid(
            second, BUF_STRIDE, src, src_w, src_h, ox_b, oy_b, width, height, kind_b,
        );
        store_average(dst, dst_stride, dst_x, dst_y, width, height, first, second);
    });
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Grid {
    None,
    Integer,
    Horizontal,
    Vertical,
    Diagonal,
}

/// Which half-sample grids a quarter-pel offset averages.
fn luma_pair(ox: i32, oy: i32, fx: i32, fy: i32) -> (Grid, i32, i32, Grid, i32, i32) {
    match (fx, fy) {
        (0, 0) => (Grid::Integer, ox, oy, Grid::None, 0, 0),
        (1, 0) => (Grid::Integer, ox, oy, Grid::Horizontal, ox, oy),
        (2, 0) => (Grid::Horizontal, ox, oy, Grid::None, 0, 0),
        (3, 0) => (Grid::Horizontal, ox, oy, Grid::Integer, ox + 1, oy),
        (0, 1) => (Grid::Integer, ox, oy, Grid::Vertical, ox, oy),
        (0, 2) => (Grid::Vertical, ox, oy, Grid::None, 0, 0),
        (0, 3) => (Grid::Vertical, ox, oy, Grid::Integer, ox, oy + 1),
        (1, 1) => (Grid::Horizontal, ox, oy, Grid::Vertical, ox, oy),
        (1, 2) => (Grid::Vertical, ox, oy, Grid::Diagonal, ox, oy),
        (1, 3) => (Grid::Vertical, ox, oy, Grid::Horizontal, ox, oy + 1),
        (2, 1) => (Grid::Horizontal, ox, oy, Grid::Diagonal, ox, oy),
        (2, 2) => (Grid::Diagonal, ox, oy, Grid::None, 0, 0),
        (2, 3) => (Grid::Diagonal, ox, oy, Grid::Horizontal, ox, oy + 1),
        (3, 1) => (Grid::Horizontal, ox, oy, Grid::Vertical, ox + 1, oy),
        (3, 2) => (Grid::Diagonal, ox, oy, Grid::Vertical, ox + 1, oy),
        (3, 3) => (Grid::Vertical, ox + 1, oy, Grid::Horizontal, ox, oy + 1),
        _ => (Grid::Integer, ox, oy, Grid::None, 0, 0),
    }
}

fn fill_luma_grid(
    buf: &mut [u16],
    buf_stride: usize,
    src: &[u16],
    src_w: i32,
    src_h: i32,
    ox: i32,
    oy: i32,
    width: usize,
    height: usize,
    kind: Grid,
) {
    match kind {
        Grid::None => {}
        Grid::Integer => fill_integer(buf, buf_stride, src, src_w, src_h, ox, oy, width, height),
        Grid::Horizontal => {
            fill_horizontal(buf, buf_stride, src, src_w, src_h, ox, oy, width, height)
        }
        Grid::Vertical => fill_vertical(buf, buf_stride, src, src_w, src_h, ox, oy, width, height),
        Grid::Diagonal => fill_diagonal(buf, buf_stride, src, src_w, src_h, ox, oy, width, height),
    }
}

fn fill_integer(
    buf: &mut [u16],
    buf_stride: usize,
    src: &[u16],
    src_w: i32,
    src_h: i32,
    ox: i32,
    oy: i32,
    width: usize,
    height: usize,
) {
    let inside = ox >= 0 && oy >= 0 && ox + width as i32 <= src_w && oy + height as i32 <= src_h;
    if inside {
        let stride = src_w as usize;
        for row in 0..height {
            let src_index = (oy as usize + row) * stride + ox as usize;
            let dst_index = row * buf_stride;
            buf[dst_index..dst_index + width].copy_from_slice(&src[src_index..src_index + width]);
        }
        return;
    }
    for row in 0..height {
        for column in 0..width {
            buf[row * buf_stride + column] =
                sample_clamped(src, src_w, src_h, ox + column as i32, oy + row as i32);
        }
    }
}

fn fill_horizontal(
    buf: &mut [u16],
    buf_stride: usize,
    src: &[u16],
    src_w: i32,
    src_h: i32,
    ox: i32,
    oy: i32,
    width: usize,
    height: usize,
) {
    for row in 0..height {
        let y = oy + row as i32;
        let inside = ox >= 2 && y >= 0 && y < src_h && ox + width as i32 + 3 <= src_w;
        if inside {
            let base = (y as usize) * (src_w as usize) + (ox as usize - 2);
            let samples = &src[base..base + width + 5];
            let mut column = 0;
            while column + 8 <= width {
                let values = six_tap_clip_x8(
                    load_u16x8(samples, column),
                    load_u16x8(samples, column + 1),
                    load_u16x8(samples, column + 2),
                    load_u16x8(samples, column + 3),
                    load_u16x8(samples, column + 4),
                    load_u16x8(samples, column + 5),
                );
                buf[row * buf_stride + column..row * buf_stride + column + 8]
                    .copy_from_slice(&values);
                column += 8;
            }
            while column < width {
                buf[row * buf_stride + column] = six_tap_clip(
                    samples[column],
                    samples[column + 1],
                    samples[column + 2],
                    samples[column + 3],
                    samples[column + 4],
                    samples[column + 5],
                );
                column += 1;
            }
        } else {
            for column in 0..width {
                let x = ox + column as i32;
                buf[row * buf_stride + column] = six_tap_clip(
                    sample_clamped(src, src_w, src_h, x - 2, y),
                    sample_clamped(src, src_w, src_h, x - 1, y),
                    sample_clamped(src, src_w, src_h, x, y),
                    sample_clamped(src, src_w, src_h, x + 1, y),
                    sample_clamped(src, src_w, src_h, x + 2, y),
                    sample_clamped(src, src_w, src_h, x + 3, y),
                );
            }
        }
    }
}

fn fill_vertical(
    buf: &mut [u16],
    buf_stride: usize,
    src: &[u16],
    src_w: i32,
    src_h: i32,
    ox: i32,
    oy: i32,
    width: usize,
    height: usize,
) {
    let inside =
        ox >= 0 && oy >= 2 && ox + width as i32 <= src_w && oy + height as i32 + 3 <= src_h;
    if inside {
        let stride = src_w as usize;
        for row in 0..height {
            let y = oy as usize + row;
            let mut column = 0;
            while column + 8 <= width {
                let x = ox as usize + column;
                let values = six_tap_clip_x8(
                    load_u16x8(src, (y - 2) * stride + x),
                    load_u16x8(src, (y - 1) * stride + x),
                    load_u16x8(src, y * stride + x),
                    load_u16x8(src, (y + 1) * stride + x),
                    load_u16x8(src, (y + 2) * stride + x),
                    load_u16x8(src, (y + 3) * stride + x),
                );
                buf[row * buf_stride + column..row * buf_stride + column + 8]
                    .copy_from_slice(&values);
                column += 8;
            }
            while column < width {
                let x = ox as usize + column;
                buf[row * buf_stride + column] = six_tap_clip(
                    src[(y - 2) * stride + x],
                    src[(y - 1) * stride + x],
                    src[y * stride + x],
                    src[(y + 1) * stride + x],
                    src[(y + 2) * stride + x],
                    src[(y + 3) * stride + x],
                );
                column += 1;
            }
        }
        return;
    }
    for row in 0..height {
        let y = oy + row as i32;
        for column in 0..width {
            let x = ox + column as i32;
            buf[row * buf_stride + column] = six_tap_clip(
                sample_clamped(src, src_w, src_h, x, y - 2),
                sample_clamped(src, src_w, src_h, x, y - 1),
                sample_clamped(src, src_w, src_h, x, y),
                sample_clamped(src, src_w, src_h, x, y + 1),
                sample_clamped(src, src_w, src_h, x, y + 2),
                sample_clamped(src, src_w, src_h, x, y + 3),
            );
        }
    }
}

fn fill_diagonal(
    buf: &mut [u16],
    buf_stride: usize,
    src: &[u16],
    src_w: i32,
    src_h: i32,
    ox: i32,
    oy: i32,
    width: usize,
    height: usize,
) {
    // Unrounded horizontal six-tap for the six rows each output row consumes.
    let mut horizontal = [0_i32; (TILE + 5) * BUF_STRIDE];
    let rows = height + 5;
    let inside =
        ox >= 2 && oy >= 2 && ox + width as i32 + 3 <= src_w && oy + height as i32 + 3 <= src_h;
    if inside {
        let stride = src_w as usize;
        for row in 0..rows {
            let y = (oy as usize + row) - 2;
            let base = y * stride + (ox as usize - 2);
            let samples = &src[base..base + width + 5];
            let mut column = 0;
            while column + 8 <= width {
                let values = six_tap_sum_x8(
                    load_u16x8(samples, column),
                    load_u16x8(samples, column + 1),
                    load_u16x8(samples, column + 2),
                    load_u16x8(samples, column + 3),
                    load_u16x8(samples, column + 4),
                    load_u16x8(samples, column + 5),
                )
                .to_array();
                horizontal[row * BUF_STRIDE + column..row * BUF_STRIDE + column + 8]
                    .copy_from_slice(&values);
                column += 8;
            }
            while column < width {
                horizontal[row * BUF_STRIDE + column] = six_tap_sum(
                    samples[column],
                    samples[column + 1],
                    samples[column + 2],
                    samples[column + 3],
                    samples[column + 4],
                    samples[column + 5],
                );
                column += 1;
            }
        }
    } else {
        for row in 0..rows {
            let y = oy - 2 + row as i32;
            for column in 0..width {
                let x = ox + column as i32;
                horizontal[row * BUF_STRIDE + column] = six_tap_sum(
                    sample_clamped(src, src_w, src_h, x - 2, y),
                    sample_clamped(src, src_w, src_h, x - 1, y),
                    sample_clamped(src, src_w, src_h, x, y),
                    sample_clamped(src, src_w, src_h, x + 1, y),
                    sample_clamped(src, src_w, src_h, x + 2, y),
                    sample_clamped(src, src_w, src_h, x + 3, y),
                );
            }
        }
    }
    for row in 0..height {
        let mut column = 0;
        while column + 8 <= width {
            let temp_index = row * BUF_STRIDE + column;
            let value = load_i32x8(&horizontal, temp_index)
                - load_i32x8(&horizontal, temp_index + BUF_STRIDE) * 5
                + load_i32x8(&horizontal, temp_index + 2 * BUF_STRIDE) * 20
                + load_i32x8(&horizontal, temp_index + 3 * BUF_STRIDE) * 20
                - load_i32x8(&horizontal, temp_index + 4 * BUF_STRIDE) * 5
                + load_i32x8(&horizontal, temp_index + 5 * BUF_STRIDE);
            let values = clip10_x8((value + 512) >> 10);
            let dst_index = row * buf_stride + column;
            buf[dst_index..dst_index + 8].copy_from_slice(&values);
            column += 8;
        }
        while column < width {
            let temp_index = row * BUF_STRIDE + column;
            let value = horizontal[temp_index] - 5 * horizontal[temp_index + BUF_STRIDE]
                + 20 * horizontal[temp_index + 2 * BUF_STRIDE]
                + 20 * horizontal[temp_index + 3 * BUF_STRIDE]
                - 5 * horizontal[temp_index + 4 * BUF_STRIDE]
                + horizontal[temp_index + 5 * BUF_STRIDE];
            buf[row * buf_stride + column] = clip10((value + 512) >> 10);
            column += 1;
        }
    }
}

fn copy_integer(
    src: &[u16],
    src_w: i32,
    src_h: i32,
    dst: &mut [u16],
    dst_stride: usize,
    dst_x: usize,
    dst_y: usize,
    width: usize,
    height: usize,
    ox: i32,
    oy: i32,
) {
    let inside = ox >= 0 && oy >= 0 && ox + width as i32 <= src_w && oy + height as i32 <= src_h;
    if inside {
        let src_stride = src_w as usize;
        for row in 0..height {
            let src_index = (oy as usize + row) * src_stride + ox as usize;
            let dst_index = (dst_y + row) * dst_stride + dst_x;
            dst[dst_index..dst_index + width].copy_from_slice(&src[src_index..src_index + width]);
        }
        return;
    }
    for row in 0..height {
        let dst_index = (dst_y + row) * dst_stride + dst_x;
        for column in 0..width {
            dst[dst_index + column] =
                sample_clamped(src, src_w, src_h, ox + column as i32, oy + row as i32);
        }
    }
}

fn store_average(
    dst: &mut [u16],
    dst_stride: usize,
    dst_x: usize,
    dst_y: usize,
    width: usize,
    height: usize,
    left: &[u16],
    right: &[u16],
) {
    for row in 0..height {
        let dst_index = (dst_y + row) * dst_stride + dst_x;
        let src_index = row * BUF_STRIDE;
        let mut column = 0;
        while column + 8 <= width {
            let averaged = ((load_u16x8(left, src_index + column)
                + load_u16x8(right, src_index + column)
                + u16x8::splat(1))
                >> 1_u16)
                .to_array();
            dst[dst_index + column..dst_index + column + 8].copy_from_slice(&averaged);
            column += 8;
        }
        while column < width {
            dst[dst_index + column] =
                avg_round(left[src_index + column], right[src_index + column]);
            column += 1;
        }
    }
}

fn tile_chroma(
    src: &[u16],
    src_w: i32,
    src_h: i32,
    dst: &mut [u16],
    dst_stride: usize,
    dst_x: usize,
    dst_y: usize,
    width: usize,
    height: usize,
    ox: i32,
    oy: i32,
    dx: i32,
    dy: i32,
) {
    let mut y = 0;
    while y < height {
        let tile_h = (height - y).min(TILE);
        let mut x = 0;
        while x < width {
            let tile_w = (width - x).min(TILE);
            interpolate_chroma_tile(
                src,
                src_w,
                src_h,
                dst,
                dst_stride,
                dst_x + x,
                dst_y + y,
                tile_w,
                tile_h,
                ox + x as i32,
                oy + y as i32,
                dx,
                dy,
            );
            x += tile_w;
        }
        y += tile_h;
    }
}

fn interpolate_chroma_tile(
    src: &[u16],
    src_w: i32,
    src_h: i32,
    dst: &mut [u16],
    dst_stride: usize,
    dst_x: usize,
    dst_y: usize,
    width: usize,
    height: usize,
    ox: i32,
    oy: i32,
    dx: i32,
    dy: i32,
) {
    if dx == 0 && dy == 0 {
        copy_integer(
            src, src_w, src_h, dst, dst_stride, dst_x, dst_y, width, height, ox, oy,
        );
        return;
    }
    let need_x = i32::from(dx != 0);
    let need_y = i32::from(dy != 0);
    let inside = ox >= 0
        && oy >= 0
        && ox + width as i32 + need_x <= src_w
        && oy + height as i32 + need_y <= src_h;
    if inside {
        let stride = src_w as usize;
        let x0 = ox as usize;
        let y0 = oy as usize;
        for row in 0..height {
            let row_a = (y0 + row) * stride;
            let row_b = (y0 + row + need_y as usize) * stride;
            let dst_index = (dst_y + row) * dst_stride + dst_x;
            let mut column = 0;
            while column + 8 <= width {
                let x = x0 + column;
                let values = bilinear_x8(
                    load_u16x8(src, row_a + x),
                    load_u16x8(src, row_a + x + need_x as usize),
                    load_u16x8(src, row_b + x),
                    load_u16x8(src, row_b + x + need_x as usize),
                    dx,
                    dy,
                );
                dst[dst_index + column..dst_index + column + 8].copy_from_slice(&values);
                column += 8;
            }
            while column < width {
                let a = src[row_a + x0 + column];
                let b = src[row_a + x0 + column + need_x as usize];
                let c = src[row_b + x0 + column];
                let d = src[row_b + x0 + column + need_x as usize];
                dst[dst_index + column] = bilinear(a, b, c, d, dx, dy);
                column += 1;
            }
        }
        return;
    }
    for row in 0..height {
        let y = oy + row as i32;
        let dst_index = (dst_y + row) * dst_stride + dst_x;
        for column in 0..width {
            let x = ox + column as i32;
            dst[dst_index + column] = bilinear(
                sample_clamped(src, src_w, src_h, x, y),
                sample_clamped(src, src_w, src_h, x + 1, y),
                sample_clamped(src, src_w, src_h, x, y + 1),
                sample_clamped(src, src_w, src_h, x + 1, y + 1),
                dx,
                dy,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn blend_chroma_plane(
    list0: &Plane422P10,
    list1: &Plane422P10,
    target: &mut Plane422P10,
    dst_x: usize,
    dst_y: usize,
    width: usize,
    height: usize,
    ox0: i32,
    oy0: i32,
    dx0: i32,
    dy0: i32,
    ox1: i32,
    oy1: i32,
    dx1: i32,
    dy1: i32,
    weight0: i32,
    weight1: i32,
) -> Result<(), MotionCompensationError> {
    let src0_w = i32::try_from(list0.width).map_err(|_| MotionCompensationError::OutOfBounds)?;
    let src0_h = i32::try_from(list0.height).map_err(|_| MotionCompensationError::OutOfBounds)?;
    let src1_w = i32::try_from(list1.width).map_err(|_| MotionCompensationError::OutOfBounds)?;
    let src1_h = i32::try_from(list1.height).map_err(|_| MotionCompensationError::OutOfBounds)?;
    if src0_w == 0 || src0_h == 0 || src1_w == 0 || src1_h == 0 {
        return Err(MotionCompensationError::OutOfBounds);
    }
    let src0 = list0.samples();
    let src1 = list1.samples();
    let dst_stride = target.width;
    let dst = target.samples_mut();
    if dx0 == 0
        && dy0 == 0
        && dx1 == 0
        && dy1 == 0
        && integer_region_inside(ox0, oy0, width, height, src0_w, src0_h)
        && integer_region_inside(ox1, oy1, width, height, src1_w, src1_h)
    {
        blend_integer_luma(
            src0,
            src0_w as usize,
            ox0 as usize,
            oy0 as usize,
            src1,
            src1_w as usize,
            ox1 as usize,
            oy1 as usize,
            dst,
            dst_stride,
            dst_x,
            dst_y,
            width,
            height,
            weight0,
            weight1,
        );
        return Ok(());
    }
    let mut y = 0;
    while y < height {
        let tile_h = (height - y).min(TILE);
        let mut x = 0;
        while x < width {
            let tile_w = (width - x).min(TILE);
            BI_SCRATCH.with(|scratch| {
                let mut scratch = scratch.borrow_mut();
                let BiScratch { first, second } = &mut *scratch;
                interpolate_chroma_tile(
                    src0,
                    src0_w,
                    src0_h,
                    first,
                    tile_w,
                    0,
                    0,
                    tile_w,
                    tile_h,
                    ox0 + x as i32,
                    oy0 + y as i32,
                    dx0,
                    dy0,
                );
                interpolate_chroma_tile(
                    src1,
                    src1_w,
                    src1_h,
                    second,
                    tile_w,
                    0,
                    0,
                    tile_w,
                    tile_h,
                    ox1 + x as i32,
                    oy1 + y as i32,
                    dx1,
                    dy1,
                );
                for row in 0..tile_h {
                    let dst_index = (dst_y + y + row) * dst_stride + dst_x + x;
                    let mut column = 0;
                    while column + 8 <= tile_w {
                        let index = row * tile_w + column;
                        let values = weighted_avg_x8(
                            load_u16x8(first, index),
                            load_u16x8(second, index),
                            weight0,
                            weight1,
                        );
                        dst[dst_index + column..dst_index + column + 8].copy_from_slice(&values);
                        column += 8;
                    }
                    while column < tile_w {
                        let index = row * tile_w + column;
                        dst[dst_index + column] =
                            weighted_avg(first[index], second[index], weight0, weight1);
                        column += 1;
                    }
                }
            });
            x += tile_w;
        }
        y += tile_h;
    }
    Ok(())
}

#[inline(always)]
fn sample_clamped(samples: &[u16], width: i32, height: i32, x: i32, y: i32) -> u16 {
    let x = x.clamp(0, width - 1) as usize;
    let y = y.clamp(0, height - 1) as usize;
    samples[y * width as usize + x]
}

#[inline(always)]
fn six_tap_sum(s0: u16, s1: u16, s2: u16, s3: u16, s4: u16, s5: u16) -> i32 {
    i32::from(s0) - 5 * i32::from(s1) + 20 * i32::from(s2) + 20 * i32::from(s3) - 5 * i32::from(s4)
        + i32::from(s5)
}

#[inline(always)]
fn load_u16x8(samples: &[u16], start: usize) -> u16x8 {
    let mut values = [0_u16; 8];
    values.copy_from_slice(&samples[start..start + 8]);
    u16x8::new(values)
}

#[inline(always)]
fn load_i32x8(samples: &[i32], start: usize) -> i32x8 {
    let mut values = [0_i32; 8];
    values.copy_from_slice(&samples[start..start + 8]);
    i32x8::new(values)
}

#[inline(always)]
fn six_tap_sum_x8(s0: u16x8, s1: u16x8, s2: u16x8, s3: u16x8, s4: u16x8, s5: u16x8) -> i32x8 {
    i32x8::from_u16x8(s0) - i32x8::from_u16x8(s1) * 5
        + i32x8::from_u16x8(s2) * 20
        + i32x8::from_u16x8(s3) * 20
        - i32x8::from_u16x8(s4) * 5
        + i32x8::from_u16x8(s5)
}

#[inline(always)]
fn six_tap_clip_x8(s0: u16x8, s1: u16x8, s2: u16x8, s3: u16x8, s4: u16x8, s5: u16x8) -> [u16; 8] {
    clip10_x8((six_tap_sum_x8(s0, s1, s2, s3, s4, s5) + 16) >> 5)
}

#[inline(always)]
fn clip10_x8(values: i32x8) -> [u16; 8] {
    values
        .max(i32x8::splat(0))
        .min(i32x8::splat(1023))
        .to_array()
        .map(|value| value as u16)
}

#[inline(always)]
fn six_tap_clip(s0: u16, s1: u16, s2: u16, s3: u16, s4: u16, s5: u16) -> u16 {
    clip10((six_tap_sum(s0, s1, s2, s3, s4, s5) + 16) >> 5)
}

#[inline(always)]
fn bilinear(a: u16, b: u16, c: u16, d: u16, dx: i32, dy: i32) -> u16 {
    let a = i32::from(a);
    let b = i32::from(b);
    let c = i32::from(c);
    let d = i32::from(d);
    let value =
        ((8 - dx) * (8 - dy) * a + dx * (8 - dy) * b + (8 - dx) * dy * c + dx * dy * d + 32) >> 6;
    clip10(value)
}

#[inline(always)]
fn bilinear_x8(a: u16x8, b: u16x8, c: u16x8, d: u16x8, dx: i32, dy: i32) -> [u16; 8] {
    let ax = 8 - dx;
    let ay = 8 - dy;
    let values = (i32x8::from_u16x8(a) * (ax * ay)
        + i32x8::from_u16x8(b) * (dx * ay)
        + i32x8::from_u16x8(c) * (ax * dy)
        + i32x8::from_u16x8(d) * (dx * dy)
        + 32)
        >> 6;
    clip10_x8(values)
}

#[inline(always)]
fn avg_round(a: u16, b: u16) -> u16 {
    weighted_avg(a, b, 32, 32)
}

#[inline(always)]
fn weighted_avg(a: u16, b: u16, weight0: i32, weight1: i32) -> u16 {
    let value = (i32::from(a) * weight0 + i32::from(b) * weight1 + 32) >> 6;
    clip10(value)
}

#[inline(always)]
fn weighted_avg_x8(a: u16x8, b: u16x8, weight0: i32, weight1: i32) -> [u16; 8] {
    clip10_x8((i32x8::from_u16x8(a) * weight0 + i32x8::from_u16x8(b) * weight1 + 32) >> 6)
}

#[inline(always)]
fn clip10(value: i32) -> u16 {
    value.clamp(0, 1023) as u16
}

fn div_floor(value: i32, divisor: i32) -> i32 {
    let quotient = value / divisor;
    let remainder = value % divisor;
    if remainder != 0 && (remainder > 0) != (divisor > 0) {
        quotient - 1
    } else {
        quotient
    }
}
