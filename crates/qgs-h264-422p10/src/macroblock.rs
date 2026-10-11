use crate::frame::DecodedFrame422P10;
use crate::macroblock_type::{
    CodedBlockPatternChroma, ISliceMacroblockType, Intra16x16PredictionMode,
};
use crate::residual::{prepare_dequant_4x4, ResidualBlock4x4, ResidualBlock8x8};
use crate::transform::{
    clip10, intra16x16_dc_prediction, intra16x16_horizontal_prediction,
    intra16x16_plane_prediction, intra16x16_vertical_prediction,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChromaPlane {
    Cb,
    Cr,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Intra16x16Macroblock {
    pub prediction: Intra16x16PredictionMode,
    pub coded_block_pattern_chroma: CodedBlockPatternChroma,
    pub coded_block_pattern_luma: u8,
    pub qp_y: u8,
    pub luma: [ResidualBlock4x4; 16],
    pub cb: [ResidualBlock4x4; 8],
    pub cr: [ResidualBlock4x4; 8],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Intra4x4PredictionMode {
    Vertical = 0,
    Horizontal = 1,
    Dc = 2,
    DiagonalDownLeft = 3,
    DiagonalDownRight = 4,
    VerticalRight = 5,
    HorizontalDown = 6,
    VerticalLeft = 7,
    HorizontalUp = 8,
}

impl Intra4x4PredictionMode {
    pub fn from_index(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Vertical),
            1 => Some(Self::Horizontal),
            2 => Some(Self::Dc),
            3 => Some(Self::DiagonalDownLeft),
            4 => Some(Self::DiagonalDownRight),
            5 => Some(Self::VerticalRight),
            6 => Some(Self::HorizontalDown),
            7 => Some(Self::VerticalLeft),
            8 => Some(Self::HorizontalUp),
            _ => None,
        }
    }
}

impl Intra16x16Macroblock {
    pub fn from_type(
        mb_type: ISliceMacroblockType,
        qp_y: u8,
    ) -> Result<Self, MacroblockReconstructionError> {
        let ISliceMacroblockType::Intra16x16 {
            prediction,
            coded_block_pattern_chroma,
            coded_block_pattern_luma,
        } = mb_type
        else {
            return Err(MacroblockReconstructionError::UnsupportedMacroblockType);
        };
        Ok(Self {
            prediction,
            coded_block_pattern_chroma,
            coded_block_pattern_luma,
            qp_y,
            luma: std::array::from_fn(|_| ResidualBlock4x4::zero()),
            cb: std::array::from_fn(|_| ResidualBlock4x4::zero()),
            cr: std::array::from_fn(|_| ResidualBlock4x4::zero()),
        })
    }

    pub fn reconstruct_into(
        &self,
        frame: &mut DecodedFrame422P10,
        mb_x: usize,
        mb_y: usize,
        top_luma: Option<[u16; 16]>,
        left_luma: Option<[u16; 16]>,
    ) -> Result<(), MacroblockReconstructionError> {
        let top_left = if mb_x > 0 && mb_y > 0 {
            let x = mb_x * 16 - 1;
            let y = mb_y * 16 - 1;
            if frame.luma_in_slice(x, y) {
                frame.y.get(x, y)
            } else {
                None
            }
        } else {
            None
        };
        if !reconstruct_intra16x16_luma_dc(
            frame,
            mb_x,
            mb_y,
            self.qp_y,
            self.prediction,
            &self.luma,
            top_luma,
            left_luma,
            top_left,
        ) {
            return Err(MacroblockReconstructionError::OutOfBounds);
        }
        if !reconstruct_chroma_422_dc(frame, ChromaPlane::Cb, mb_x, mb_y, self.qp_y, &self.cb) {
            return Err(MacroblockReconstructionError::OutOfBounds);
        }
        if !reconstruct_chroma_422_dc(frame, ChromaPlane::Cr, mb_x, mb_y, self.qp_y, &self.cr) {
            return Err(MacroblockReconstructionError::OutOfBounds);
        }
        Ok(())
    }
}

pub fn reconstruct_intra16x16_luma_dc(
    frame: &mut DecodedFrame422P10,
    mb_x: usize,
    mb_y: usize,
    qp: u8,
    prediction_mode: Intra16x16PredictionMode,
    residuals: &[ResidualBlock4x4; 16],
    top: Option<[u16; 16]>,
    left: Option<[u16; 16]>,
    top_left: Option<u16>,
) -> bool {
    let Some(prediction) = intra16x16_luma_prediction(prediction_mode, top, left, top_left) else {
        return false;
    };
    reconstruct_intra16x16_luma(frame, mb_x, mb_y, qp, residuals, prediction)
}

pub fn intra16x16_luma_prediction(
    prediction_mode: Intra16x16PredictionMode,
    top: Option<[u16; 16]>,
    left: Option<[u16; 16]>,
    top_left: Option<u16>,
) -> Option<[[u16; 16]; 16]> {
    match prediction_mode {
        Intra16x16PredictionMode::Vertical => top
            .map(intra16x16_vertical_prediction)
            .or_else(|| Some(intra16x16_dc_prediction(top, left))),
        Intra16x16PredictionMode::Horizontal => left
            .map(intra16x16_horizontal_prediction)
            .or_else(|| Some(intra16x16_dc_prediction(top, left))),
        Intra16x16PredictionMode::Dc => Some(intra16x16_dc_prediction(top, left)),
        Intra16x16PredictionMode::Plane => match (top, left, top_left) {
            (Some(top), Some(left), Some(top_left)) => {
                Some(intra16x16_plane_prediction(top, left, top_left))
            }
            _ => Some(intra16x16_dc_prediction(top, left)),
        },
    }
}

pub fn reconstruct_intra16x16_luma(
    frame: &mut DecodedFrame422P10,
    mb_x: usize,
    mb_y: usize,
    qp: u8,
    residuals: &[ResidualBlock4x4; 16],
    prediction: [[u16; 16]; 16],
) -> bool {
    let weights = frame.scaling.weight4x4[0];
    let base_x = mb_x.saturating_mul(16);
    let base_y = mb_y.saturating_mul(16);
    if base_x + 16 > frame.y.width || base_y + 16 > frame.y.height {
        return false;
    }
    let mut nonzero_seen = false;
    let mut dequant = None;
    for (block_index, residual) in residuals.iter().enumerate() {
        let (block_x, block_y) = luma4x4_position(block_index);
        let block_prediction = prediction_block(&prediction, block_x, block_y);
        let reconstructed = if residual.is_zero() {
            block_prediction
        } else if nonzero_seen {
            let prepared = dequant.get_or_insert_with(|| prepare_dequant_4x4(qp, &weights));
            residual.reconstruct_classified_with_prescaled_dc_prepared(
                block_prediction,
                prepared,
                residual.is_dc_only(),
            )
        } else {
            nonzero_seen = true;
            residual.reconstruct_nonzero_with_prescaled_dc_scaled(qp, block_prediction, &weights)
        };
        let x = base_x + block_x * 4;
        let y = base_y + block_y * 4;
        if !frame.y.write_block(x, y, &reconstructed) {
            return false;
        }
    }
    true
}

pub fn add_luma_residual_16x16(
    frame: &mut DecodedFrame422P10,
    mb_x: usize,
    mb_y: usize,
    qp: u8,
    residuals: &[ResidualBlock4x4; 16],
) -> bool {
    add_luma_residual_16x16_impl(frame, mb_x, mb_y, qp, residuals, false)
}

pub(crate) fn add_luma_residual_16x16_predequantized(
    frame: &mut DecodedFrame422P10,
    mb_x: usize,
    mb_y: usize,
    residuals: &[ResidualBlock4x4; 16],
) -> bool {
    add_luma_residual_16x16_impl(frame, mb_x, mb_y, 0, residuals, true)
}

fn add_luma_residual_16x16_impl(
    frame: &mut DecodedFrame422P10,
    mb_x: usize,
    mb_y: usize,
    qp: u8,
    residuals: &[ResidualBlock4x4; 16],
    predequantized: bool,
) -> bool {
    let weights = frame.scaling.weight4x4[3];
    let mut dequant = None;
    let base_x = mb_x.saturating_mul(16);
    let base_y = mb_y.saturating_mul(16);
    if base_x + 16 > frame.y.width || base_y + 16 > frame.y.height {
        return false;
    }
    for (block_index, residual) in residuals.iter().enumerate() {
        if residual.is_zero() {
            continue;
        }
        let (block_x, block_y) = luma4x4_position(block_index);
        let x = base_x + block_x * 4;
        let y = base_y + block_y * 4;
        let Some(prediction) = read_luma_block(frame, x, y) else {
            return false;
        };
        let dc_only = residual.is_dc_only();
        let reconstructed = if predequantized {
            residual.reconstruct_predequantized(prediction, dc_only)
        } else {
            residual.reconstruct_classified_with_prediction_prepared(
                prediction,
                dequant.get_or_insert_with(|| prepare_dequant_4x4(qp, &weights)),
                dc_only,
            )
        };
        if !frame.y.write_block(x, y, &reconstructed) {
            return false;
        }
    }
    true
}

pub fn add_luma_residual_8x8(
    frame: &mut DecodedFrame422P10,
    mb_x: usize,
    mb_y: usize,
    qp: u8,
    residuals: &[ResidualBlock8x8; 4],
) -> bool {
    let weights = frame.scaling.weight8x8[1];
    let base_x = mb_x.saturating_mul(16);
    let base_y = mb_y.saturating_mul(16);
    if base_x + 16 > frame.y.width || base_y + 16 > frame.y.height {
        return false;
    }
    for block_y in 0..2 {
        for block_x in 0..2 {
            let block_index = block_y * 2 + block_x;
            if residuals[block_index].is_zero() {
                continue;
            }
            let x = base_x + block_x * 8;
            let y = base_y + block_y * 8;
            let Some(prediction) = read_luma8x8_block(frame, x, y) else {
                return false;
            };
            let reconstructed = residuals[block_index]
                .reconstruct_nonzero_with_prediction_scaled(qp, prediction, &weights);
            if !frame.y.write_block(x, y, &reconstructed) {
                return false;
            }
        }
    }
    true
}

pub fn reconstruct_inter_luma_16x16_local(
    frame: &mut DecodedFrame422P10,
    mb_x: usize,
    mb_y: usize,
    qp: u8,
    residuals: &[ResidualBlock4x4; 16],
    prediction: &mut [u16; 16 * 16],
) -> bool {
    reconstruct_inter_luma_16x16_local_impl(frame, mb_x, mb_y, qp, residuals, prediction, false)
}

pub(crate) fn reconstruct_inter_luma_16x16_local_predequantized(
    frame: &mut DecodedFrame422P10,
    mb_x: usize,
    mb_y: usize,
    residuals: &[ResidualBlock4x4; 16],
    prediction: &mut [u16; 16 * 16],
) -> bool {
    reconstruct_inter_luma_16x16_local_impl(frame, mb_x, mb_y, 0, residuals, prediction, true)
}

fn reconstruct_inter_luma_16x16_local_impl(
    frame: &mut DecodedFrame422P10,
    mb_x: usize,
    mb_y: usize,
    qp: u8,
    residuals: &[ResidualBlock4x4; 16],
    prediction: &mut [u16; 16 * 16],
    predequantized: bool,
) -> bool {
    let weights = frame.scaling.weight4x4[3];
    let mut dequant = None;
    for (block_index, residual) in residuals.iter().enumerate() {
        if residual.is_zero() {
            continue;
        }
        let (block_x, block_y) = luma4x4_position(block_index);
        let mut block = [[0_u16; 4]; 4];
        for row in 0..4 {
            let start = (block_y * 4 + row) * 16 + block_x * 4;
            block[row].copy_from_slice(&prediction[start..start + 4]);
        }
        let dc_only = residual.is_dc_only();
        let reconstructed = if predequantized {
            residual.reconstruct_predequantized(block, dc_only)
        } else {
            residual.reconstruct_classified_with_prediction_prepared(
                block,
                dequant.get_or_insert_with(|| prepare_dequant_4x4(qp, &weights)),
                dc_only,
            )
        };
        for row in 0..4 {
            let start = (block_y * 4 + row) * 16 + block_x * 4;
            prediction[start..start + 4].copy_from_slice(&reconstructed[row]);
        }
    }
    write_local_luma_16x16(frame, mb_x, mb_y, prediction)
}

pub fn reconstruct_inter_luma_8x8_local(
    frame: &mut DecodedFrame422P10,
    mb_x: usize,
    mb_y: usize,
    qp: u8,
    residuals: &[ResidualBlock8x8; 4],
    prediction: &mut [u16; 16 * 16],
) -> bool {
    let weights = frame.scaling.weight8x8[1];
    for block_y in 0..2 {
        for block_x in 0..2 {
            let residual = &residuals[block_y * 2 + block_x];
            if residual.is_zero() {
                continue;
            }
            let mut block = [[0_u16; 8]; 8];
            for row in 0..8 {
                let start = (block_y * 8 + row) * 16 + block_x * 8;
                block[row].copy_from_slice(&prediction[start..start + 8]);
            }
            let reconstructed =
                residual.reconstruct_nonzero_with_prediction_scaled(qp, block, &weights);
            for row in 0..8 {
                let start = (block_y * 8 + row) * 16 + block_x * 8;
                prediction[start..start + 8].copy_from_slice(&reconstructed[row]);
            }
        }
    }
    write_local_luma_16x16(frame, mb_x, mb_y, prediction)
}

pub(crate) fn write_local_luma_16x16(
    frame: &mut DecodedFrame422P10,
    mb_x: usize,
    mb_y: usize,
    prediction: &[u16; 16 * 16],
) -> bool {
    let base_x = mb_x.saturating_mul(16);
    let base_y = mb_y.saturating_mul(16);
    if base_x + 16 > frame.y.width || base_y + 16 > frame.y.height {
        return false;
    }
    let stride = frame.y.width;
    let samples = frame.y.samples_mut();
    for row in 0..16 {
        let target = (base_y + row) * stride + base_x;
        samples[target..target + 16].copy_from_slice(&prediction[row * 16..row * 16 + 16]);
    }
    true
}

pub(crate) fn write_local_chroma_8x16(
    frame: &mut DecodedFrame422P10,
    plane: ChromaPlane,
    mb_x: usize,
    mb_y: usize,
    prediction: &[u16; 8 * 16],
) -> bool {
    let base_x = mb_x.saturating_mul(8);
    let base_y = mb_y.saturating_mul(16);
    let plane = match plane {
        ChromaPlane::Cb => &mut frame.cb,
        ChromaPlane::Cr => &mut frame.cr,
    };
    if base_x + 8 > plane.width || base_y + 16 > plane.height {
        return false;
    }
    let stride = plane.width;
    let samples = plane.samples_mut();
    for row in 0..16 {
        let target = (base_y + row) * stride + base_x;
        samples[target..target + 8].copy_from_slice(&prediction[row * 8..row * 8 + 8]);
    }
    true
}

/// Stores a completed local Y/Cb/Cr prediction with no residual.
pub(crate) fn store_local_inter_prediction(
    frame: &mut DecodedFrame422P10,
    mb_x: usize,
    mb_y: usize,
    luma: &[u16; 16 * 16],
    cb: &[u16; 8 * 16],
    cr: &[u16; 8 * 16],
) -> bool {
    write_local_luma_16x16(frame, mb_x, mb_y, luma)
        && write_local_chroma_8x16(frame, ChromaPlane::Cb, mb_x, mb_y, cb)
        && write_local_chroma_8x16(frame, ChromaPlane::Cr, mb_x, mb_y, cr)
}

pub fn reconstruct_intra4x4_luma(
    frame: &mut DecodedFrame422P10,
    mb_x: usize,
    mb_y: usize,
    qp: u8,
    modes: &[Intra4x4PredictionMode; 16],
    residuals: &[ResidualBlock4x4; 16],
) -> Result<(), MacroblockReconstructionError> {
    let weights = frame.scaling.weight4x4[0];
    let base_x = mb_x.saturating_mul(16);
    let base_y = mb_y.saturating_mul(16);
    if base_x + 16 > frame.y.width || base_y + 16 > frame.y.height {
        return Err(MacroblockReconstructionError::OutOfBounds);
    }
    let mut nonzero_seen = false;
    let mut dequant = None;
    for block_index in 0..16 {
        let (block_x, block_y) = luma4x4_position(block_index);
        let x = base_x + block_x * 4;
        let y = base_y + block_y * 4;
        let prediction = intra4x4_prediction(frame, x, y, block_index, modes[block_index])?;
        let residual = &residuals[block_index];
        let reconstructed = if residual.is_zero() {
            prediction
        } else if nonzero_seen {
            let prepared = dequant.get_or_insert_with(|| prepare_dequant_4x4(qp, &weights));
            residual.reconstruct_classified_with_prediction_prepared(
                prediction,
                prepared,
                residual.is_dc_only(),
            )
        } else {
            nonzero_seen = true;
            residual.reconstruct_nonzero_with_prediction_scaled(qp, prediction, &weights)
        };
        if !frame.y.write_block(x, y, &reconstructed) {
            return Err(MacroblockReconstructionError::OutOfBounds);
        }
    }
    Ok(())
}

pub fn reconstruct_intra8x8_luma(
    frame: &mut DecodedFrame422P10,
    mb_x: usize,
    mb_y: usize,
    qp: u8,
    modes: &[Intra4x4PredictionMode; 4],
    residuals: &[ResidualBlock8x8; 4],
) -> Result<(), MacroblockReconstructionError> {
    let weights = frame.scaling.weight8x8[0];
    let base_x = mb_x.saturating_mul(16);
    let base_y = mb_y.saturating_mul(16);
    if base_x + 16 > frame.y.width || base_y + 16 > frame.y.height {
        return Err(MacroblockReconstructionError::OutOfBounds);
    }
    for block_y in 0..2 {
        for block_x in 0..2 {
            let block_index = block_y * 2 + block_x;
            let x = base_x + block_x * 8;
            let y = base_y + block_y * 8;
            let prediction = intra8x8_prediction(frame, x, y, block_index, modes[block_index])?;
            let reconstructed = residuals[block_index]
                .reconstruct_nonzero_with_prediction_scaled(qp, prediction, &weights);
            if !frame.y.write_block(x, y, &reconstructed) {
                return Err(MacroblockReconstructionError::OutOfBounds);
            }
        }
    }
    Ok(())
}

pub fn reconstruct_chroma_422_dc(
    frame: &mut DecodedFrame422P10,
    plane: ChromaPlane,
    mb_x: usize,
    mb_y: usize,
    qp: u8,
    residuals: &[ResidualBlock4x4; 8],
) -> bool {
    reconstruct_chroma_422_intra(frame, plane, mb_x, mb_y, qp, 0, residuals)
}

pub fn reconstruct_chroma_422_intra(
    frame: &mut DecodedFrame422P10,
    plane: ChromaPlane,
    mb_x: usize,
    mb_y: usize,
    qp: u8,
    prediction_mode: u8,
    residuals: &[ResidualBlock4x4; 8],
) -> bool {
    let list = match plane {
        ChromaPlane::Cb => 1,
        ChromaPlane::Cr => 2,
    };
    let weights = frame.scaling.weight4x4[list];
    let slice_first_mb = frame.slice_first_mb;
    let coded_width = frame.coded_width;
    let base_x = mb_x.saturating_mul(8);
    let base_y = mb_y.saturating_mul(16);
    let plane = match plane {
        ChromaPlane::Cb => &mut frame.cb,
        ChromaPlane::Cr => &mut frame.cr,
    };
    if base_x + 8 > plane.width || base_y + 16 > plane.height {
        return false;
    }
    let prediction = chroma_422_prediction(
        plane,
        base_x,
        base_y,
        prediction_mode,
        slice_first_mb,
        coded_width,
    );
    let mut nonzero_seen = false;
    let mut dequant = None;
    for block_y in 0..4 {
        for block_x in 0..2 {
            let block_index = block_y * 2 + block_x;
            let block_prediction = chroma_prediction_block(&prediction, block_x, block_y);
            let residual = &residuals[block_index];
            let reconstructed = if residual.is_zero() {
                block_prediction
            } else if nonzero_seen {
                let prepared = dequant.get_or_insert_with(|| prepare_dequant_4x4(qp, &weights));
                residual.reconstruct_classified_with_prescaled_dc_prepared(
                    block_prediction,
                    prepared,
                    residual.is_dc_only(),
                )
            } else {
                nonzero_seen = true;
                residual.reconstruct_nonzero_with_prescaled_dc_scaled(
                    qp,
                    block_prediction,
                    &weights,
                )
            };
            let x = base_x + block_x * 4;
            let y = base_y + block_y * 4;
            if !plane.write_block(x, y, &reconstructed) {
                return false;
            }
        }
    }
    true
}

fn chroma_422_prediction(
    plane: &crate::frame::Plane422P10,
    base_x: usize,
    base_y: usize,
    prediction_mode: u8,
    slice_first_mb: u32,
    coded_width: usize,
) -> [[u16; 8]; 16] {
    let in_slice = |x: usize, y: usize| {
        let mb_width = (coded_width / 16).max(1);
        let address = (y / 16).saturating_mul(mb_width) + (x / 8);
        address >= slice_first_mb as usize
    };
    let top = if base_y > 0 && in_slice(base_x, base_y - 1) {
        let mut samples = [0_u16; 8];
        let mut available = true;
        for (index, sample) in samples.iter_mut().enumerate() {
            if let Some(value) = plane.get(base_x + index, base_y - 1) {
                *sample = value;
            } else {
                available = false;
                break;
            }
        }
        available.then_some(samples)
    } else {
        None
    };
    let left = if base_x > 0 && in_slice(base_x - 1, base_y) {
        let mut samples = [0_u16; 16];
        let mut available = true;
        for (index, sample) in samples.iter_mut().enumerate() {
            if let Some(value) = plane.get(base_x - 1, base_y + index) {
                *sample = value;
            } else {
                available = false;
                break;
            }
        }
        available.then_some(samples)
    } else {
        None
    };
    let top_left = if base_x > 0 && base_y > 0 && in_slice(base_x - 1, base_y - 1) {
        plane.get(base_x - 1, base_y - 1)
    } else {
        None
    };

    match prediction_mode {
        0 => chroma_422_dc_prediction(top, left),
        1 => left
            .map(|left| std::array::from_fn(|y| [left[y]; 8]))
            .unwrap_or_else(|| chroma_422_dc_prediction(top, left)),
        2 => top
            .map(|top| std::array::from_fn(|_| top))
            .unwrap_or_else(|| chroma_422_dc_prediction(top, left)),
        3 => match (top, left) {
            (Some(top), Some(left)) => top_left
                .map(|top_left| chroma_422_plane_prediction(top, left, top_left))
                .unwrap_or_else(|| chroma_422_dc_prediction(Some(top), Some(left))),
            _ => chroma_422_dc_prediction(top, left),
        },
        _ => chroma_422_dc_prediction(top, left),
    }
}

fn chroma_422_dc_prediction(top: Option<[u16; 8]>, left: Option<[u16; 16]>) -> [[u16; 8]; 16] {
    let sum4 = |samples: &[u16]| samples.iter().map(|value| u32::from(*value)).sum::<u32>();
    match (top, left) {
        (Some(top), Some(left)) => {
            let top_left = sum4(&top[0..4]);
            let top_right = sum4(&top[4..8]);
            let left_bands = [
                sum4(&left[0..4]),
                sum4(&left[4..8]),
                sum4(&left[8..12]),
                sum4(&left[12..16]),
            ];
            let values = [
                ((top_left + left_bands[0] + 4) >> 3) as u16,
                ((top_right + 2) >> 2) as u16,
                ((left_bands[1] + 2) >> 2) as u16,
                ((top_right + left_bands[1] + 4) >> 3) as u16,
                ((left_bands[2] + 2) >> 2) as u16,
                ((top_right + left_bands[2] + 4) >> 3) as u16,
                ((left_bands[3] + 2) >> 2) as u16,
                ((top_right + left_bands[3] + 4) >> 3) as u16,
            ];
            std::array::from_fn(|row| {
                let band = row / 4;
                let left_dc = values[band * 2];
                let right_dc = values[band * 2 + 1];
                [
                    left_dc, left_dc, left_dc, left_dc, right_dc, right_dc, right_dc, right_dc,
                ]
            })
        }
        (Some(top), None) => {
            let left_dc = ((sum4(&top[0..4]) + 2) >> 2) as u16;
            let right_dc = ((sum4(&top[4..8]) + 2) >> 2) as u16;
            [[
                left_dc, left_dc, left_dc, left_dc, right_dc, right_dc, right_dc, right_dc,
            ]; 16]
        }
        (None, Some(left)) => std::array::from_fn(|row| {
            let band = sum4(&left[row / 4 * 4..row / 4 * 4 + 4]);
            let dc = ((band + 2) >> 2) as u16;
            [dc; 8]
        }),
        (None, None) => [[512; 8]; 16],
    }
}

fn chroma_422_plane_prediction(top: [u16; 8], left: [u16; 16], top_left: u16) -> [[u16; 8]; 16] {
    let mut h = 0_i32;
    let mut v = 0_i32;
    for i in 0..=3 {
        let mirrored = if i <= 2 { top[2 - i] } else { top_left };
        h += (i as i32 + 1) * (i32::from(top[4 + i]) - i32::from(mirrored));
    }
    for i in 0..=7 {
        let mirrored = if i <= 6 { left[6 - i] } else { top_left };
        v += (i as i32 + 1) * (i32::from(left[8 + i]) - i32::from(mirrored));
    }
    let a = 16 * (i32::from(left[15]) + i32::from(top[7]));
    let b = (34 * h + 32) >> 6;
    let c = (5 * v + 32) >> 6;
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            clip10((a + b * (column as i32 - 3) + c * (row as i32 - 7) + 16) >> 5)
        })
    })
}

pub fn add_chroma_residual_422(
    frame: &mut DecodedFrame422P10,
    plane: ChromaPlane,
    mb_x: usize,
    mb_y: usize,
    qp: u8,
    residuals: &[ResidualBlock4x4; 8],
) -> bool {
    let list = match plane {
        ChromaPlane::Cb => 1,
        ChromaPlane::Cr => 2,
    };
    let weights = frame.scaling.weight4x4[list];
    let mut dequant = None;
    let base_x = mb_x.saturating_mul(8);
    let base_y = mb_y.saturating_mul(16);
    let plane = match plane {
        ChromaPlane::Cb => &mut frame.cb,
        ChromaPlane::Cr => &mut frame.cr,
    };
    if base_x + 8 > plane.width || base_y + 16 > plane.height {
        return false;
    }
    for block_y in 0..4 {
        for block_x in 0..2 {
            let block_index = block_y * 2 + block_x;
            if residuals[block_index].is_zero() {
                continue;
            }
            let x = base_x + block_x * 4;
            let y = base_y + block_y * 4;
            let Some(prediction) = read_plane_block(plane, x, y) else {
                return false;
            };
            let reconstructed = residuals[block_index]
                .reconstruct_classified_with_prediction_prepared(
                    prediction,
                    dequant.get_or_insert_with(|| prepare_dequant_4x4(qp, &weights)),
                    residuals[block_index].is_dc_only(),
                );
            if !plane.write_block(x, y, &reconstructed) {
                return false;
            }
        }
    }
    true
}

pub fn add_chroma_residual_422_prescaled(
    frame: &mut DecodedFrame422P10,
    plane: ChromaPlane,
    mb_x: usize,
    mb_y: usize,
    qp: u8,
    residuals: &[ResidualBlock4x4; 8],
    intra: bool,
) -> bool {
    let list = match (intra, plane) {
        (true, ChromaPlane::Cb) => 1,
        (true, ChromaPlane::Cr) => 2,
        (false, ChromaPlane::Cb) => 4,
        (false, ChromaPlane::Cr) => 5,
    };
    let weights = frame.scaling.weight4x4[list];
    let mut dequant = None;
    let base_x = mb_x.saturating_mul(8);
    let base_y = mb_y.saturating_mul(16);
    let plane = match plane {
        ChromaPlane::Cb => &mut frame.cb,
        ChromaPlane::Cr => &mut frame.cr,
    };
    if base_x + 8 > plane.width || base_y + 16 > plane.height {
        return false;
    }
    for block_y in 0..4 {
        for block_x in 0..2 {
            let block_index = block_y * 2 + block_x;
            if residuals[block_index].is_zero() {
                continue;
            }
            let x = base_x + block_x * 4;
            let y = base_y + block_y * 4;
            let Some(prediction) = read_plane_block(plane, x, y) else {
                return false;
            };
            let reconstructed = residuals[block_index]
                .reconstruct_classified_with_prescaled_dc_prepared(
                    prediction,
                    dequant.get_or_insert_with(|| prepare_dequant_4x4(qp, &weights)),
                    residuals[block_index].is_dc_only(),
                );
            if !plane.write_block(x, y, &reconstructed) {
                return false;
            }
        }
    }
    true
}

pub fn reconstruct_inter_chroma_422_local(
    frame: &mut DecodedFrame422P10,
    plane_kind: ChromaPlane,
    mb_x: usize,
    mb_y: usize,
    qp: u8,
    residuals: &[ResidualBlock4x4; 8],
    prediction: &mut [u16; 8 * 16],
) -> bool {
    let list = match plane_kind {
        ChromaPlane::Cb => 4,
        ChromaPlane::Cr => 5,
    };
    let weights = frame.scaling.weight4x4[list];
    let mut dequant = None;
    for block_y in 0..4 {
        for block_x in 0..2 {
            let residual = &residuals[block_y * 2 + block_x];
            if residual.is_zero() {
                continue;
            }
            let mut block = [[0_u16; 4]; 4];
            for row in 0..4 {
                let start = (block_y * 4 + row) * 8 + block_x * 4;
                block[row].copy_from_slice(&prediction[start..start + 4]);
            }
            let reconstructed = residual.reconstruct_classified_with_prescaled_dc_prepared(
                block,
                dequant.get_or_insert_with(|| prepare_dequant_4x4(qp, &weights)),
                residual.is_dc_only(),
            );
            for row in 0..4 {
                let start = (block_y * 4 + row) * 8 + block_x * 4;
                prediction[start..start + 4].copy_from_slice(&reconstructed[row]);
            }
        }
    }
    let base_x = mb_x.saturating_mul(8);
    let base_y = mb_y.saturating_mul(16);
    let plane = match plane_kind {
        ChromaPlane::Cb => &mut frame.cb,
        ChromaPlane::Cr => &mut frame.cr,
    };
    if base_x + 8 > plane.width || base_y + 16 > plane.height {
        return false;
    }
    let stride = plane.width;
    let samples = plane.samples_mut();
    for row in 0..16 {
        let target = (base_y + row) * stride + base_x;
        samples[target..target + 8].copy_from_slice(&prediction[row * 8..row * 8 + 8]);
    }
    true
}

fn read_luma_block(frame: &DecodedFrame422P10, x: usize, y: usize) -> Option<[[u16; 4]; 4]> {
    read_plane_block(&frame.y, x, y)
}

fn read_luma8x8_block(frame: &DecodedFrame422P10, x: usize, y: usize) -> Option<[[u16; 8]; 8]> {
    if x + 8 > frame.y.width || y + 8 > frame.y.height {
        return None;
    }
    let samples = frame.y.samples();
    let stride = frame.y.width;
    let mut block = [[0_u16; 8]; 8];
    for (row_index, row) in block.iter_mut().enumerate() {
        let start = (y + row_index) * stride + x;
        row.copy_from_slice(&samples[start..start + 8]);
    }
    Some(block)
}

fn intra4x4_prediction(
    frame: &DecodedFrame422P10,
    x: usize,
    y: usize,
    block_index: usize,
    mode: Intra4x4PredictionMode,
) -> Result<[[u16; 4]; 4], MacroblockReconstructionError> {
    let top = (y > 0 && frame.luma_in_slice(x, y - 1))
        .then(|| {
            let mut samples = [0_u16; 4];
            for (index, sample) in samples.iter_mut().enumerate() {
                *sample = frame.y.get(x + index, y - 1)?;
            }
            Some(samples)
        })
        .flatten();
    let left = (x > 0 && frame.luma_in_slice(x - 1, y))
        .then(|| {
            let mut samples = [0_u16; 4];
            for (index, sample) in samples.iter_mut().enumerate() {
                *sample = frame.y.get(x - 1, y + index)?;
            }
            Some(samples)
        })
        .flatten();
    match mode {
        Intra4x4PredictionMode::Vertical => {
            if let Some(top) = top {
                Ok([top; 4])
            } else {
                Ok(intra4x4_dc_prediction(top, left))
            }
        }
        Intra4x4PredictionMode::Horizontal => {
            if let Some(left) = left {
                let mut prediction = [[0_u16; 4]; 4];
                for (row, value) in left.into_iter().enumerate() {
                    prediction[row] = [value; 4];
                }
                Ok(prediction)
            } else {
                Ok(intra4x4_dc_prediction(top, left))
            }
        }
        Intra4x4PredictionMode::Dc => Ok(intra4x4_dc_prediction(top, left)),
        Intra4x4PredictionMode::DiagonalDownLeft => {
            let Some(top8) = top8_samples(frame, x, y, block_index) else {
                return Ok(intra4x4_dc_prediction(top, left));
            };
            Ok(std::array::from_fn(|row| {
                std::array::from_fn(|column| {
                    let index = row + column;
                    if index >= 6 {
                        avg3(top8[6], top8[7], top8[7])
                    } else {
                        avg3(top8[index], top8[index + 1], top8[index + 2])
                    }
                })
            }))
        }
        Intra4x4PredictionMode::DiagonalDownRight => {
            let (Some(top), Some(left), Some(top_left)) = (top, left, top_left_sample(frame, x, y))
            else {
                return Ok(intra4x4_dc_prediction(top, left));
            };
            Ok(std::array::from_fn(|row| {
                std::array::from_fn(|column| match column.cmp(&row) {
                    std::cmp::Ordering::Greater => {
                        let offset = column - row - 1;
                        if offset == 0 {
                            avg3(top_left, top[0], top[1])
                        } else {
                            avg3(top[offset - 1], top[offset], top[offset + 1])
                        }
                    }
                    std::cmp::Ordering::Equal => avg3(left[0], top_left, top[0]),
                    std::cmp::Ordering::Less => {
                        let offset = row - column - 1;
                        if offset == 0 {
                            avg3(top_left, left[0], left[1])
                        } else {
                            avg3(left[offset - 1], left[offset], left[offset + 1])
                        }
                    }
                })
            }))
        }
        Intra4x4PredictionMode::VerticalRight => {
            let (Some(top), Some(left), Some(top_left)) = (top, left, top_left_sample(frame, x, y))
            else {
                return Ok(intra4x4_dc_prediction(top, left));
            };
            Ok(std::array::from_fn(|row| {
                std::array::from_fn(|column| {
                    let z = 2_i32 * column as i32 - row as i32;
                    if z >= 0 {
                        let index = (z / 2) as usize;
                        if z % 2 == 0 {
                            avg2(
                                sample_top_with_left(top, top_left, index),
                                top[index.min(3)],
                            )
                        } else {
                            avg3(
                                sample_top_with_left(top, top_left, index),
                                top[index.min(3)],
                                top[(index + 1).min(3)],
                            )
                        }
                    } else if z == -1 {
                        avg3(left[0], top_left, top[0])
                    } else {
                        let depth = (-z) as usize;
                        avg3(
                            sample_left_with_top(left, top_left, depth),
                            sample_left_with_top(left, top_left, depth - 1),
                            sample_left_with_top(left, top_left, depth - 2),
                        )
                    }
                })
            }))
        }
        Intra4x4PredictionMode::HorizontalDown => {
            let (Some(top), Some(left), Some(top_left)) = (top, left, top_left_sample(frame, x, y))
            else {
                return Ok(intra4x4_dc_prediction(top, left));
            };
            Ok(std::array::from_fn(|row| {
                std::array::from_fn(|column| {
                    let z = 2_i32 * row as i32 - column as i32;
                    if z >= 0 {
                        let index = (z / 2) as usize;
                        if z % 2 == 0 {
                            avg2(
                                sample_left_with_top(left, top_left, index),
                                left[index.min(3)],
                            )
                        } else {
                            avg3(
                                sample_left_with_top(left, top_left, index),
                                left[index.min(3)],
                                left[(index + 1).min(3)],
                            )
                        }
                    } else if z == -1 {
                        avg3(top[0], top_left, left[0])
                    } else {
                        let depth = (-z) as usize;
                        avg3(
                            sample_top_with_left(top, top_left, depth),
                            sample_top_with_left(top, top_left, depth - 1),
                            sample_top_with_left(top, top_left, depth - 2),
                        )
                    }
                })
            }))
        }
        Intra4x4PredictionMode::VerticalLeft => {
            let Some(top8) = top8_samples(frame, x, y, block_index) else {
                return Ok(intra4x4_dc_prediction(top, left));
            };
            Ok(std::array::from_fn(|row| {
                std::array::from_fn(|column| {
                    let index = column + (row >> 1);
                    if row % 2 == 0 {
                        avg2(top8[index], top8[(index + 1).min(7)])
                    } else {
                        avg3(
                            top8[index],
                            top8[(index + 1).min(7)],
                            top8[(index + 2).min(7)],
                        )
                    }
                })
            }))
        }
        Intra4x4PredictionMode::HorizontalUp => {
            let Some(left) = left else {
                return Ok(intra4x4_dc_prediction(top, left));
            };
            Ok(std::array::from_fn(|row| {
                std::array::from_fn(|column| {
                    let index = row + (column >> 1);
                    if index >= 3 {
                        left[3]
                    } else if column % 2 == 0 {
                        avg2(left[index], left[index + 1])
                    } else {
                        avg3(left[index], left[index + 1], left[(index + 2).min(3)])
                    }
                })
            }))
        }
    }
}

fn intra8x8_prediction(
    frame: &DecodedFrame422P10,
    x: usize,
    y: usize,
    block_index: usize,
    mode: Intra4x4PredictionMode,
) -> Result<[[u16; 8]; 8], MacroblockReconstructionError> {
    let raw_top_left = top_left_sample(frame, x, y);
    let raw_top16 = top16_samples(frame, x, y, block_index);
    let raw_left = (x > 0 && frame.luma_in_slice(x - 1, y))
        .then(|| {
            let mut samples = [0_u16; 8];
            for (index, sample) in samples.iter_mut().enumerate() {
                *sample = frame.y.get(x - 1, y + index)?;
            }
            Some(samples)
        })
        .flatten();
    let top16 = raw_top16
        .map(|(samples, has_top_right)| filter_intra8x8_top(samples, raw_top_left, has_top_right));
    let top = top16.map(|samples| {
        let mut top = [0_u16; 8];
        top.copy_from_slice(&samples[..8]);
        top
    });
    let left = raw_left.map(|samples| filter_intra8x8_left(samples, raw_top_left));
    let top_left = match (raw_top_left, raw_top16, raw_left) {
        (Some(top_left), Some((top, _)), Some(left)) => {
            Some(filter_intra8x8_top_left(top_left, top[0], left[0]))
        }
        _ => None,
    };
    match mode {
        Intra4x4PredictionMode::Vertical => {
            if let Some(top) = top {
                Ok([top; 8])
            } else {
                Ok(intra8x8_dc_prediction(top, left))
            }
        }
        Intra4x4PredictionMode::Horizontal => {
            if let Some(left) = left {
                Ok(std::array::from_fn(|row| {
                    let value = left[row];
                    [value; 8]
                }))
            } else {
                Ok(intra8x8_dc_prediction(top, left))
            }
        }
        Intra4x4PredictionMode::Dc => Ok(intra8x8_dc_prediction(top, left)),
        Intra4x4PredictionMode::DiagonalDownLeft => {
            let Some(top16) = top16 else {
                return Ok(intra8x8_dc_prediction(top, left));
            };
            Ok(std::array::from_fn(|row| {
                std::array::from_fn(|column| {
                    let index = row + column;
                    if index >= 14 {
                        avg3(top16[14], top16[15], top16[15])
                    } else {
                        avg3(top16[index], top16[index + 1], top16[index + 2])
                    }
                })
            }))
        }
        Intra4x4PredictionMode::DiagonalDownRight => {
            let (Some(top), Some(left), Some(top_left)) = (top, left, top_left) else {
                return Ok(intra8x8_dc_prediction(top, left));
            };
            Ok(std::array::from_fn(|row| {
                std::array::from_fn(|column| match column.cmp(&row) {
                    std::cmp::Ordering::Greater => {
                        let offset = column - row - 1;
                        if offset == 0 {
                            avg3(top_left, top[0], top[1])
                        } else {
                            avg3(
                                top[(offset - 1).min(7)],
                                top[offset.min(7)],
                                top[(offset + 1).min(7)],
                            )
                        }
                    }
                    std::cmp::Ordering::Equal => avg3(left[0], top_left, top[0]),
                    std::cmp::Ordering::Less => {
                        let offset = row - column - 1;
                        if offset == 0 {
                            avg3(top_left, left[0], left[1])
                        } else {
                            avg3(
                                left[(offset - 1).min(7)],
                                left[offset.min(7)],
                                left[(offset + 1).min(7)],
                            )
                        }
                    }
                })
            }))
        }
        Intra4x4PredictionMode::VerticalRight => {
            let (Some(top), Some(left), Some(top_left)) = (top, left, top_left) else {
                return Ok(intra8x8_dc_prediction(top, left));
            };
            Ok(std::array::from_fn(|row| {
                std::array::from_fn(|column| {
                    let z = 2_i32 * column as i32 - row as i32;
                    if z >= 0 {
                        let index = (z / 2) as usize;
                        if z % 2 == 0 {
                            avg2(
                                sample_top8_with_left(top, top_left, index),
                                top[index.min(7)],
                            )
                        } else {
                            avg3(
                                sample_top8_with_left(top, top_left, index),
                                top[index.min(7)],
                                top[(index + 1).min(7)],
                            )
                        }
                    } else if z == -1 {
                        avg3(left[0], top_left, top[0])
                    } else {
                        let depth = (-z) as usize;
                        avg3(
                            sample_left8_with_top(left, top_left, depth),
                            sample_left8_with_top(left, top_left, depth - 1),
                            sample_left8_with_top(left, top_left, depth - 2),
                        )
                    }
                })
            }))
        }
        Intra4x4PredictionMode::HorizontalDown => {
            let (Some(top), Some(left), Some(top_left)) = (top, left, top_left) else {
                return Ok(intra8x8_dc_prediction(top, left));
            };
            Ok(std::array::from_fn(|row| {
                std::array::from_fn(|column| {
                    let z = 2_i32 * row as i32 - column as i32;
                    if z >= 0 {
                        let index = (z / 2) as usize;
                        if z % 2 == 0 {
                            avg2(
                                sample_left8_with_top(left, top_left, index),
                                left[index.min(7)],
                            )
                        } else {
                            avg3(
                                sample_left8_with_top(left, top_left, index),
                                left[index.min(7)],
                                left[(index + 1).min(7)],
                            )
                        }
                    } else if z == -1 {
                        avg3(top[0], top_left, left[0])
                    } else {
                        let depth = (-z) as usize;
                        avg3(
                            sample_top8_with_left(top, top_left, depth),
                            sample_top8_with_left(top, top_left, depth - 1),
                            sample_top8_with_left(top, top_left, depth - 2),
                        )
                    }
                })
            }))
        }
        Intra4x4PredictionMode::VerticalLeft => {
            let Some(top16) = top16 else {
                return Ok(intra8x8_dc_prediction(top, left));
            };
            Ok(std::array::from_fn(|row| {
                std::array::from_fn(|column| {
                    let index = column + (row >> 1);
                    if row % 2 == 0 {
                        avg2(top16[index], top16[(index + 1).min(15)])
                    } else {
                        avg3(
                            top16[index],
                            top16[(index + 1).min(15)],
                            top16[(index + 2).min(15)],
                        )
                    }
                })
            }))
        }
        Intra4x4PredictionMode::HorizontalUp => {
            let Some(left) = left else {
                return Ok(intra8x8_dc_prediction(top, left));
            };
            Ok(std::array::from_fn(|row| {
                std::array::from_fn(|column| {
                    let index = row + (column >> 1);
                    if index >= 7 {
                        left[7]
                    } else if column % 2 == 0 {
                        avg2(left[index], left[index + 1])
                    } else {
                        avg3(left[index], left[index + 1], left[(index + 2).min(7)])
                    }
                })
            }))
        }
    }
}

fn intra8x8_dc_prediction(top: Option<[u16; 8]>, left: Option<[u16; 8]>) -> [[u16; 8]; 8] {
    let value = match (top, left) {
        (Some(top), Some(left)) => {
            ((top.into_iter().chain(left).map(u32::from).sum::<u32>() + 8) >> 4) as u16
        }
        (Some(top), None) => ((top.into_iter().map(u32::from).sum::<u32>() + 4) >> 3) as u16,
        (None, Some(left)) => ((left.into_iter().map(u32::from).sum::<u32>() + 4) >> 3) as u16,
        (None, None) => 512,
    };
    [[value; 8]; 8]
}

fn top16_samples(
    frame: &DecodedFrame422P10,
    x: usize,
    y: usize,
    block_index: usize,
) -> Option<([u16; 16], bool)> {
    if y == 0 {
        return None;
    }
    let first_unavailable_4x4 = block_index.saturating_add(1).saturating_mul(4);
    let mut samples = [0_u16; 16];
    let mut available = [false; 16];
    for index in 0..16 {
        let sample_x = x + index;
        if intra4x4_sample_available(frame, x, y, first_unavailable_4x4, sample_x, y - 1) {
            samples[index] = frame.y.get(sample_x, y - 1)?;
            available[index] = true;
        }
    }
    if !available[..8].iter().all(|sample| *sample) {
        return None;
    }
    // Spec 8.3.2.2.3: undecoded samples to the top-right repeat p[7, -1].
    let has_top_right = available[8..].iter().all(|sample| *sample);
    if !has_top_right {
        let replicate = samples[7];
        for sample in &mut samples[8..] {
            *sample = replicate;
        }
    }
    Some((samples, has_top_right))
}

fn filter_intra8x8_top(top: [u16; 16], top_left: Option<u16>, has_top_right: bool) -> [u16; 16] {
    let mut filtered = [0_u16; 16];
    filtered[0] = match top_left {
        Some(top_left) => {
            (u32::from(top_left) + 2 * u32::from(top[0]) + u32::from(top[1]) + 2) >> 2
        }
        None => (3 * u32::from(top[0]) + u32::from(top[1]) + 2) >> 2,
    } as u16;
    for index in 1..7 {
        filtered[index] = ((u32::from(top[index - 1])
            + 2 * u32::from(top[index])
            + u32::from(top[index + 1])
            + 2)
            >> 2) as u16;
    }
    let right = if has_top_right { top[8] } else { top[7] };
    filtered[7] = ((u32::from(right) + 2 * u32::from(top[7]) + u32::from(top[6]) + 2) >> 2) as u16;
    if has_top_right {
        for index in 8..15 {
            filtered[index] = ((u32::from(top[index - 1])
                + 2 * u32::from(top[index])
                + u32::from(top[index + 1])
                + 2)
                >> 2) as u16;
        }
        filtered[15] = ((u32::from(top[14]) + 3 * u32::from(top[15]) + 2) >> 2) as u16;
    } else {
        for sample in &mut filtered[8..] {
            *sample = top[7];
        }
    }
    filtered
}

fn filter_intra8x8_left(left: [u16; 8], top_left: Option<u16>) -> [u16; 8] {
    let mut filtered = [0_u16; 8];
    let corner = top_left.unwrap_or(left[0]);
    filtered[0] =
        ((u32::from(corner) + 2 * u32::from(left[0]) + u32::from(left[1]) + 2) >> 2) as u16;
    for index in 1..7 {
        filtered[index] = ((u32::from(left[index - 1])
            + 2 * u32::from(left[index])
            + u32::from(left[index + 1])
            + 2)
            >> 2) as u16;
    }
    filtered[7] = ((u32::from(left[6]) + 3 * u32::from(left[7]) + 2) >> 2) as u16;
    filtered
}

fn filter_intra8x8_top_left(top_left: u16, top: u16, left: u16) -> u16 {
    ((u32::from(left) + 2 * u32::from(top_left) + u32::from(top) + 2) >> 2) as u16
}

fn top8_samples(
    frame: &DecodedFrame422P10,
    x: usize,
    y: usize,
    block_index: usize,
) -> Option<[u16; 8]> {
    if y == 0 {
        return None;
    }
    let mut samples = [0_u16; 8];
    let mut available = [false; 8];
    for index in 0..8 {
        let sample_x = x + index;
        if intra4x4_sample_available(frame, x, y, block_index, sample_x, y - 1) {
            samples[index] = frame.y.get(sample_x, y - 1)?;
            available[index] = true;
        }
    }
    if !available[..4].iter().all(|sample| *sample) {
        return None;
    }
    // Spec 8.3.1.2.2: undecoded top-right samples repeat p[3, -1].
    if !available[4..].iter().all(|sample| *sample) {
        let replicate = samples[3];
        for sample in &mut samples[4..] {
            *sample = replicate;
        }
    }
    Some(samples)
}

fn intra4x4_sample_available(
    frame: &DecodedFrame422P10,
    current_x: usize,
    current_y: usize,
    current_block: usize,
    sample_x: usize,
    sample_y: usize,
) -> bool {
    if sample_x >= frame.y.width || sample_y >= frame.y.height {
        return false;
    }
    if !frame.luma_in_slice(sample_x, sample_y) {
        return false;
    }
    let mb_x = current_x / 16;
    let mb_y = current_y / 16;
    let sample_mb_x = sample_x / 16;
    let sample_mb_y = sample_y / 16;
    if sample_mb_y < mb_y || sample_mb_x < mb_x {
        return true;
    }
    if sample_mb_y > mb_y || sample_mb_x > mb_x {
        return false;
    }
    let block_x = (sample_x % 16) / 4;
    let block_y = (sample_y % 16) / 4;
    luma4x4_block_index(block_x, block_y) < current_block
}

fn luma4x4_block_index(block_x: usize, block_y: usize) -> usize {
    let block_x = block_x as u32;
    let block_y = block_y as u32;
    ((block_y & 1) << 1 | (block_x & 1) | ((block_x & 2) << 1) | ((block_y & 2) << 2)) as usize
}

fn intra4x4_dc_prediction(top: Option<[u16; 4]>, left: Option<[u16; 4]>) -> [[u16; 4]; 4] {
    let value = match (top, left) {
        (Some(top), Some(left)) => {
            ((top.into_iter().chain(left).map(u32::from).sum::<u32>() + 4) >> 3) as u16
        }
        (Some(top), None) => ((top.into_iter().map(u32::from).sum::<u32>() + 2) >> 2) as u16,
        (None, Some(left)) => ((left.into_iter().map(u32::from).sum::<u32>() + 2) >> 2) as u16,
        (None, None) => 512,
    };
    [[value; 4]; 4]
}

fn top_left_sample(frame: &DecodedFrame422P10, x: usize, y: usize) -> Option<u16> {
    if x == 0 || y == 0 || !frame.luma_in_slice(x - 1, y - 1) {
        return None;
    }
    frame.y.get(x - 1, y - 1)
}

fn sample_top_with_left(top: [u16; 4], top_left: u16, index: usize) -> u16 {
    if index == 0 {
        top_left
    } else {
        top[(index - 1).min(3)]
    }
}

fn sample_left_with_top(left: [u16; 4], top_left: u16, index: usize) -> u16 {
    if index == 0 {
        top_left
    } else {
        left[(index - 1).min(3)]
    }
}

fn sample_top8_with_left(top: [u16; 8], top_left: u16, index: usize) -> u16 {
    if index == 0 {
        top_left
    } else {
        top[(index - 1).min(7)]
    }
}

fn sample_left8_with_top(left: [u16; 8], top_left: u16, index: usize) -> u16 {
    if index == 0 {
        top_left
    } else {
        left[(index - 1).min(7)]
    }
}

fn avg2(a: u16, b: u16) -> u16 {
    ((u32::from(a) + u32::from(b) + 1) >> 1) as u16
}

fn avg3(a: u16, b: u16, c: u16) -> u16 {
    ((u32::from(a) + 2 * u32::from(b) + u32::from(c) + 2) >> 2) as u16
}

fn read_plane_block(
    plane: &crate::frame::Plane422P10,
    x: usize,
    y: usize,
) -> Option<[[u16; 4]; 4]> {
    if x + 4 > plane.width || y + 4 > plane.height {
        return None;
    }
    let samples = plane.samples();
    let stride = plane.width;
    let mut block = [[0_u16; 4]; 4];
    for (row_index, row) in block.iter_mut().enumerate() {
        let start = (y + row_index) * stride + x;
        row.copy_from_slice(&samples[start..start + 4]);
    }
    Some(block)
}

fn prediction_block(prediction: &[[u16; 16]; 16], block_x: usize, block_y: usize) -> [[u16; 4]; 4] {
    let mut block = [[0_u16; 4]; 4];
    for row in 0..4 {
        for column in 0..4 {
            block[row][column] = prediction[block_y * 4 + row][block_x * 4 + column];
        }
    }
    block
}

fn luma4x4_position(block_index: usize) -> (usize, usize) {
    let x = (block_index & 1) + ((block_index >> 2) & 1) * 2;
    let y = ((block_index >> 1) & 1) + ((block_index >> 3) & 1) * 2;
    (x, y)
}

fn chroma_prediction_block(
    prediction: &[[u16; 8]; 16],
    block_x: usize,
    block_y: usize,
) -> [[u16; 4]; 4] {
    let mut block = [[0_u16; 4]; 4];
    for row in 0..4 {
        for column in 0..4 {
            block[row][column] = prediction[block_y * 4 + row][block_x * 4 + column];
        }
    }
    block
}

#[derive(Debug)]
pub enum MacroblockReconstructionError {
    UnsupportedMacroblockType,
    UnsupportedPredictionMode,
    OutOfBounds,
}

impl std::fmt::Display for MacroblockReconstructionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedMacroblockType => write!(f, "unsupported macroblock type"),
            Self::UnsupportedPredictionMode => write!(f, "unsupported macroblock prediction mode"),
            Self::OutOfBounds => write!(f, "macroblock reconstruction writes out of bounds"),
        }
    }
}

impl std::error::Error for MacroblockReconstructionError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::macroblock_type::{CodedBlockPatternChroma, Intra16x16PredictionMode};

    #[test]
    fn reconstruct_intra16x16_luma_writes_decoded_samples_to_frame() {
        let mut frame = DecodedFrame422P10::new(16, 16, 16, 16);
        let mut residuals = std::array::from_fn(|_| ResidualBlock4x4::zero());
        residuals[0].coeffs[0][0] = 64;

        assert!(reconstruct_intra16x16_luma_dc(
            &mut frame,
            0,
            0,
            0,
            Intra16x16PredictionMode::Dc,
            &residuals,
            None,
            None,
            None,
        ));

        assert!(frame.y.get(0, 0).unwrap() > 512);
        assert_eq!(frame.cb.get(0, 0), Some(0));
        assert_eq!(frame.cr.get(0, 0), Some(0));
    }

    #[test]
    fn reconstruct_intra16x16_luma_rejects_out_of_bounds_macroblock() {
        let mut frame = DecodedFrame422P10::new(16, 16, 16, 16);
        let residuals = std::array::from_fn(|_| ResidualBlock4x4::zero());

        assert!(!reconstruct_intra16x16_luma_dc(
            &mut frame,
            1,
            0,
            0,
            Intra16x16PredictionMode::Dc,
            &residuals,
            None,
            None,
            None,
        ));
    }

    #[test]
    fn reconstruct_chroma_422_writes_eight_blocks_per_plane() {
        let mut frame = DecodedFrame422P10::new(16, 16, 16, 16);
        let mut residuals = std::array::from_fn(|_| ResidualBlock4x4::zero());
        residuals[7].coeffs[0][0] = 64;

        assert!(reconstruct_chroma_422_dc(
            &mut frame,
            ChromaPlane::Cb,
            0,
            0,
            0,
            &residuals
        ));

        assert_eq!(frame.cb.get(0, 0), Some(512));
        assert!(frame.cb.get(7, 15).unwrap() > 512);
        assert_eq!(frame.cr.get(7, 15), Some(0));
    }

    #[test]
    fn reconstruct_chroma_422_intra_horizontal_uses_left_samples() {
        let mut frame = DecodedFrame422P10::new(32, 16, 32, 16);
        let residuals = std::array::from_fn(|_| ResidualBlock4x4::zero());
        for y in 0..16 {
            assert!(frame.cr.set(7, y, 200 + y as u16));
        }

        assert!(reconstruct_chroma_422_intra(
            &mut frame,
            ChromaPlane::Cr,
            1,
            0,
            0,
            1,
            &residuals
        ));

        assert_eq!(frame.cr.get(8, 0), Some(200));
        assert_eq!(frame.cr.get(15, 15), Some(215));
    }

    #[test]
    fn reconstruct_chroma_422_intra_vertical_uses_top_samples() {
        let mut frame = DecodedFrame422P10::new(16, 32, 16, 32);
        let residuals = std::array::from_fn(|_| ResidualBlock4x4::zero());
        for x in 0..8 {
            assert!(frame.cb.set(x, 15, 100 + x as u16));
        }

        assert!(reconstruct_chroma_422_intra(
            &mut frame,
            ChromaPlane::Cb,
            0,
            1,
            0,
            2,
            &residuals
        ));

        assert_eq!(frame.cb.get(0, 16), Some(100));
        assert_eq!(frame.cb.get(7, 31), Some(107));
    }

    #[test]
    fn reconstruct_chroma_422_intra_plane_uses_top_and_left_gradients() {
        let mut frame = DecodedFrame422P10::new(32, 32, 32, 32);
        let residuals = std::array::from_fn(|_| ResidualBlock4x4::zero());
        for x in 0..8 {
            assert!(frame.cb.set(8 + x, 15, 500 + (x as u16 * 4)));
        }
        for y in 0..16 {
            assert!(frame.cb.set(7, 16 + y, 400 + (y as u16 * 2)));
        }

        assert!(reconstruct_chroma_422_intra(
            &mut frame,
            ChromaPlane::Cb,
            1,
            1,
            0,
            3,
            &residuals
        ));

        let top_left = frame.cb.get(8, 16).unwrap();
        let top_right = frame.cb.get(15, 16).unwrap();
        let bottom_left = frame.cb.get(8, 31).unwrap();

        assert!(top_right > top_left);
        assert!(bottom_left > top_left);
    }

    #[test]
    fn reconstruct_chroma_422_rejects_out_of_bounds_macroblock() {
        let mut frame = DecodedFrame422P10::new(16, 16, 16, 16);
        let residuals = std::array::from_fn(|_| ResidualBlock4x4::zero());

        assert!(!reconstruct_chroma_422_dc(
            &mut frame,
            ChromaPlane::Cr,
            2,
            0,
            0,
            &residuals
        ));
    }

    #[test]
    fn add_luma_residual_16x16_updates_inter_prediction_samples() {
        let mut frame = DecodedFrame422P10::new(16, 16, 16, 16);
        for y in 0..16 {
            for x in 0..16 {
                assert!(frame.y.set(x, y, 200));
            }
        }
        let mut residuals = std::array::from_fn(|_| ResidualBlock4x4::zero());
        residuals[0].coeffs[0][0] = 64;

        assert!(add_luma_residual_16x16(&mut frame, 0, 0, 0, &residuals));

        assert!(frame.y.get(0, 0).unwrap() > 200);
        assert_eq!(frame.y.get(15, 15), Some(200));
    }

    #[test]
    fn add_chroma_residual_422_updates_inter_prediction_samples() {
        let mut frame = DecodedFrame422P10::new(16, 16, 16, 16);
        for y in 0..16 {
            for x in 0..8 {
                assert!(frame.cb.set(x, y, 300));
            }
        }
        let mut residuals = std::array::from_fn(|_| ResidualBlock4x4::zero());
        residuals[3].coeffs[0][0] = 64;

        assert!(add_chroma_residual_422(
            &mut frame,
            ChromaPlane::Cb,
            0,
            0,
            0,
            &residuals
        ));

        assert_eq!(frame.cb.get(0, 0), Some(300));
        assert!(frame.cb.get(4, 4).unwrap() > 300);
    }

    #[test]
    fn reconstruct_intra4x4_luma_supports_dc_vertical_horizontal() {
        let mut frame = DecodedFrame422P10::new(16, 16, 16, 16);
        let modes = [Intra4x4PredictionMode::Dc; 16];
        let residuals = std::array::from_fn(|_| ResidualBlock4x4::zero());

        reconstruct_intra4x4_luma(&mut frame, 0, 0, 0, &modes, &residuals).unwrap();

        assert_eq!(frame.y.get(0, 0), Some(512));
        assert_eq!(frame.y.get(15, 15), Some(512));
    }

    #[test]
    fn intra8x8_top_right_repeats_the_last_available_sample() {
        let mut frame = DecodedFrame422P10::new(32, 16, 32, 16);
        for x in 8..16 {
            assert!(frame.y.set(x, 7, 400));
        }

        let predicted =
            intra8x8_prediction(&frame, 8, 8, 3, Intra4x4PredictionMode::DiagonalDownLeft).unwrap();

        assert_eq!(predicted[7][7], 400);
    }

    #[test]
    fn intra8x8_vertical_filters_the_reference_row() {
        let mut frame = DecodedFrame422P10::new(32, 16, 32, 16);
        assert!(frame.y.set(15, 7, 1000));

        let predicted =
            intra8x8_prediction(&frame, 8, 8, 3, Intra4x4PredictionMode::Vertical).unwrap();

        assert_eq!(predicted[0][7], 750);
    }

    #[test]
    fn intra4x4_vertical_right_uses_the_top_sample_on_the_left_diagonal() {
        let mut frame = DecodedFrame422P10::new(16, 16, 16, 16);
        assert!(frame.y.set(3, 3, 50));
        assert!(frame.y.set(4, 3, 400));
        assert!(frame.y.set(3, 4, 100));
        assert!(frame.y.set(3, 5, 200));
        assert!(frame.y.set(3, 6, 300));

        let predicted =
            intra4x4_prediction(&frame, 4, 4, 0, Intra4x4PredictionMode::VerticalRight).unwrap();

        assert_eq!(predicted[1][0], 150);
        assert_eq!(predicted[3][1], 150);
        assert_eq!(predicted[2][0], 113);
        assert_eq!(predicted[3][0], 200);
    }

    #[test]
    fn reconstruct_intra8x8_luma_supports_dc() {
        let mut frame = DecodedFrame422P10::new(16, 16, 16, 16);
        let modes = [Intra4x4PredictionMode::Dc; 4];
        let residuals = std::array::from_fn(|_| ResidualBlock8x8::zero());

        reconstruct_intra8x8_luma(&mut frame, 0, 0, 36, &modes, &residuals).unwrap();

        assert_eq!(frame.y.get(0, 0), Some(512));
        assert_eq!(frame.y.get(15, 15), Some(512));
    }

    #[test]
    fn intra16x16_macroblock_reconstructs_luma_and_chroma_planes() {
        let mb_type = ISliceMacroblockType::Intra16x16 {
            prediction: Intra16x16PredictionMode::Dc,
            coded_block_pattern_chroma: CodedBlockPatternChroma::DcAndAc,
            coded_block_pattern_luma: 15,
        };
        let mut macroblock = Intra16x16Macroblock::from_type(mb_type, 0).unwrap();
        macroblock.luma[0].coeffs[0][0] = 64;
        macroblock.cb[0].coeffs[0][0] = 64;
        macroblock.cr[0].coeffs[0][0] = 128;
        let mut frame = DecodedFrame422P10::new(16, 16, 16, 16);

        macroblock
            .reconstruct_into(&mut frame, 0, 0, None, None)
            .unwrap();

        assert!(frame.y.get(0, 0).unwrap() > 512);
        assert!(frame.cb.get(0, 0).unwrap() > 512);
        assert!(frame.cr.get(0, 0).unwrap() > frame.cb.get(0, 0).unwrap());
    }

    #[test]
    fn intra16x16_macroblock_falls_back_to_dc_when_prediction_samples_are_unavailable() {
        let mb_type = ISliceMacroblockType::Intra16x16 {
            prediction: Intra16x16PredictionMode::Vertical,
            coded_block_pattern_chroma: CodedBlockPatternChroma::Zero,
            coded_block_pattern_luma: 0,
        };
        let macroblock = Intra16x16Macroblock::from_type(mb_type, 0).unwrap();
        let mut frame = DecodedFrame422P10::new(16, 16, 16, 16);

        macroblock
            .reconstruct_into(&mut frame, 0, 0, None, None)
            .unwrap();

        assert_eq!(frame.y.get(0, 0), Some(512));
    }

    #[test]
    fn intra16x16_macroblock_reconstructs_plane_prediction() {
        let mb_type = ISliceMacroblockType::Intra16x16 {
            prediction: Intra16x16PredictionMode::Plane,
            coded_block_pattern_chroma: CodedBlockPatternChroma::Zero,
            coded_block_pattern_luma: 0,
        };
        let macroblock = Intra16x16Macroblock::from_type(mb_type, 0).unwrap();
        let mut frame = DecodedFrame422P10::new(16, 16, 16, 16);
        let top = std::array::from_fn(|index| 300 + index as u16);
        let left = std::array::from_fn(|index| 400 + index as u16);

        assert!(reconstruct_intra16x16_luma_dc(
            &mut frame,
            0,
            0,
            macroblock.qp_y,
            macroblock.prediction,
            &macroblock.luma,
            Some(top),
            Some(left),
            Some(299),
        ));

        assert!(frame.y.get(15, 15).unwrap() > frame.y.get(0, 0).unwrap());
    }

    #[test]
    fn intra16x16_macroblock_reconstructs_vertical_prediction() {
        let mb_type = ISliceMacroblockType::Intra16x16 {
            prediction: Intra16x16PredictionMode::Vertical,
            coded_block_pattern_chroma: CodedBlockPatternChroma::Zero,
            coded_block_pattern_luma: 0,
        };
        let macroblock = Intra16x16Macroblock::from_type(mb_type, 0).unwrap();
        let mut frame = DecodedFrame422P10::new(16, 16, 16, 16);
        let mut top = [0_u16; 16];
        top[7] = 700;

        macroblock
            .reconstruct_into(&mut frame, 0, 0, Some(top), None)
            .unwrap();

        assert_eq!(frame.y.get(7, 0), Some(700));
        assert_eq!(frame.y.get(7, 15), Some(700));
    }

    #[test]
    fn intra16x16_macroblock_reconstructs_horizontal_prediction() {
        let mb_type = ISliceMacroblockType::Intra16x16 {
            prediction: Intra16x16PredictionMode::Horizontal,
            coded_block_pattern_chroma: CodedBlockPatternChroma::Zero,
            coded_block_pattern_luma: 0,
        };
        let macroblock = Intra16x16Macroblock::from_type(mb_type, 0).unwrap();
        let mut frame = DecodedFrame422P10::new(16, 16, 16, 16);
        let mut left = [0_u16; 16];
        left[9] = 333;

        macroblock
            .reconstruct_into(&mut frame, 0, 0, None, Some(left))
            .unwrap();

        assert_eq!(frame.y.get(0, 9), Some(333));
        assert_eq!(frame.y.get(15, 9), Some(333));
    }
}
