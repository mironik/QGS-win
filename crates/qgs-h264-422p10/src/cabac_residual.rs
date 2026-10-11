use std::fmt;

use crate::cabac::{CabacContext, CabacDecoder, CabacError, CabacInitValue};
use crate::residual::{
    PreparedDequant4x4, ResidualBlock4x4, ResidualBlock8x8, ZIGZAG_4X4, ZIGZAG_8X8,
};
use rusty_h264_common::cabac_tables::CTX_INIT;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CabacResidual4x4Contexts {
    pub coded_block_flag: CabacContext,
    pub significant_coeff_flag: [CabacContext; 15],
    pub last_significant_coeff_flag: [CabacContext; 15],
    pub coeff_abs_level_greater1: CabacContext,
    pub coeff_abs_level_greater2: CabacContext,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CabacResidualCategory {
    Luma16Dc,
    Luma16Ac,
    Luma4x4,
    Chroma422Dc,
    Chroma422Ac,
}

#[derive(Clone, Debug)]
pub struct CabacResidualCategoryContexts {
    coded_block_flag_base: usize,
    significant_base: usize,
    last_base: usize,
    abs_base: usize,
    chroma422_dc: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CabacResidual8x8Contexts {
    pub significant_coeff_flag: [CabacContext; 15],
    pub last_significant_coeff_flag: [CabacContext; 9],
    pub coeff_abs_level_minus1: [CabacContext; 10],
}

impl CabacResidual4x4Contexts {
    pub fn flat(context: CabacContext) -> Self {
        Self {
            coded_block_flag: context,
            significant_coeff_flag: [context; 15],
            last_significant_coeff_flag: [context; 15],
            coeff_abs_level_greater1: context,
            coeff_abs_level_greater2: context,
        }
    }
}

impl CabacResidualCategoryContexts {
    pub fn i_slice(category: CabacResidualCategory, qp_y: u8) -> Self {
        let _ = qp_y;
        Self::new(category)
    }

    pub fn new(category: CabacResidualCategory) -> Self {
        let coded_block_flag_base = match category {
            CabacResidualCategory::Luma16Dc => 85,
            CabacResidualCategory::Luma16Ac => 89,
            CabacResidualCategory::Luma4x4 => 93,
            CabacResidualCategory::Chroma422Dc => 97,
            CabacResidualCategory::Chroma422Ac => 101,
        };
        let significant_base = match category {
            CabacResidualCategory::Luma16Dc => 105,
            CabacResidualCategory::Luma16Ac => 120,
            CabacResidualCategory::Luma4x4 => 134,
            CabacResidualCategory::Chroma422Dc => 149,
            CabacResidualCategory::Chroma422Ac => 152,
        };
        let last_base = match category {
            CabacResidualCategory::Luma16Dc => 166,
            CabacResidualCategory::Luma16Ac => 181,
            CabacResidualCategory::Luma4x4 => 195,
            CabacResidualCategory::Chroma422Dc => 210,
            CabacResidualCategory::Chroma422Ac => 213,
        };
        let abs_base = match category {
            CabacResidualCategory::Luma16Dc => 227,
            CabacResidualCategory::Luma16Ac => 237,
            CabacResidualCategory::Luma4x4 => 247,
            CabacResidualCategory::Chroma422Dc => 257,
            CabacResidualCategory::Chroma422Ac => 266,
        };

        Self {
            coded_block_flag_base,
            significant_base,
            last_base,
            abs_base,
            chroma422_dc: matches!(category, CabacResidualCategory::Chroma422Dc),
        }
    }
}

#[inline(always)]
fn decode_bank_bin(
    decoder: &mut CabacDecoder<'_>,
    bank: &mut [CabacContext],
    absolute_index: usize,
) -> bool {
    debug_assert!(absolute_index < bank.len());
    // Context bases live in 85..=266 and the largest offset used with them is 14.
    let context = unsafe { bank.get_unchecked_mut(absolute_index) };
    decoder.decode_bin(context)
}

#[inline(always)]
fn ensure_cabac(decoder: &CabacDecoder<'_>) -> Result<(), CabacResidualError> {
    if decoder.fault() {
        Err(CabacError::EndOfSlice.into())
    } else {
        Ok(())
    }
}

pub fn i_slice_residual_context_bank(qp_y: u8) -> [CabacContext; 460] {
    let qp = i16::from(qp_y);
    let mut bank = [CabacContext::new(0, false); 460];
    for index in 85..=275 {
        bank[index] = cabac_i_init(index).initialize(qp);
    }
    bank
}

pub fn pb_slice_residual_context_bank(qp_y: u8, cabac_init_idc: u8) -> [CabacContext; 460] {
    let qp = i16::from(qp_y);
    let model = usize::from(cabac_init_idc).saturating_add(1).min(3);
    let mut bank = [CabacContext::new(0, false); 460];
    for index in 85..=275 {
        let (m, n) = CTX_INIT[index][model];
        bank[index] = CabacInitValue::new(m as i8, n as i8).initialize(qp);
    }
    bank
}

impl CabacResidual8x8Contexts {
    pub fn i_slice(qp_y: u8) -> Self {
        let qp = i16::from(qp_y);
        Self {
            significant_coeff_flag: std::array::from_fn(|index| {
                cabac_i_init(402 + index).initialize(qp)
            }),
            last_significant_coeff_flag: std::array::from_fn(|index| {
                cabac_i_init(417 + index).initialize(qp)
            }),
            coeff_abs_level_minus1: std::array::from_fn(|index| {
                cabac_i_init(426 + index).initialize(qp)
            }),
        }
    }

    pub fn pb_slice(qp_y: u8, cabac_init_idc: u8) -> Self {
        let qp = i16::from(qp_y);
        let model = usize::from(cabac_init_idc).saturating_add(1).min(3);
        Self {
            significant_coeff_flag: std::array::from_fn(|index| {
                let (m, n) = CTX_INIT[402 + index][model];
                CabacInitValue::new(m as i8, n as i8).initialize(qp)
            }),
            last_significant_coeff_flag: std::array::from_fn(|index| {
                let (m, n) = CTX_INIT[417 + index][model];
                CabacInitValue::new(m as i8, n as i8).initialize(qp)
            }),
            coeff_abs_level_minus1: std::array::from_fn(|index| {
                let (m, n) = CTX_INIT[426 + index][model];
                CabacInitValue::new(m as i8, n as i8).initialize(qp)
            }),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CabacResidual4x4 {
    pub block: ResidualBlock4x4,
    pub non_zero_coefficients: u8,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CabacResidual8x8 {
    pub block: ResidualBlock8x8,
    pub non_zero_coefficients: u8,
}

pub fn decode_residual_4x4(
    decoder: &mut CabacDecoder<'_>,
    contexts: &mut CabacResidual4x4Contexts,
) -> Result<CabacResidual4x4, CabacResidualError> {
    decode_residual_4x4_subset(decoder, contexts, 0, 16)
}

pub fn decode_residual_4x4_ac(
    decoder: &mut CabacDecoder<'_>,
    contexts: &mut CabacResidual4x4Contexts,
) -> Result<CabacResidual4x4, CabacResidualError> {
    decode_residual_4x4_subset(decoder, contexts, 1, 15)
}

pub fn decode_residual_4x4_category(
    decoder: &mut CabacDecoder<'_>,
    bank: &mut [CabacContext],
    contexts: &CabacResidualCategoryContexts,
    coded_block_context: usize,
) -> Result<CabacResidual4x4, CabacResidualError> {
    decode_residual_category_subset(decoder, bank, contexts, coded_block_context, 0, 16)
}

pub(crate) fn decode_residual_4x4_category_dequantized(
    decoder: &mut CabacDecoder<'_>,
    bank: &mut [CabacContext],
    contexts: &CabacResidualCategoryContexts,
    coded_block_context: usize,
    prepared: &PreparedDequant4x4,
) -> Result<CabacResidual4x4, CabacResidualError> {
    let residual = decode_residual_category_levels(
        decoder,
        bank,
        contexts,
        coded_block_context,
        0,
        16,
        Some(prepared),
    )?;
    Ok(CabacResidual4x4 {
        block: ResidualBlock4x4::from_raster_array(residual.levels),
        non_zero_coefficients: residual.non_zero_coefficients,
    })
}

pub fn decode_residual_4x4_ac_category(
    decoder: &mut CabacDecoder<'_>,
    bank: &mut [CabacContext],
    contexts: &CabacResidualCategoryContexts,
    coded_block_context: usize,
) -> Result<CabacResidual4x4, CabacResidualError> {
    decode_residual_category_subset(decoder, bank, contexts, coded_block_context, 1, 15)
}

pub fn decode_residual_chroma422_dc_category(
    decoder: &mut CabacDecoder<'_>,
    bank: &mut [CabacContext],
    contexts: &CabacResidualCategoryContexts,
    coded_block_context: usize,
) -> Result<CabacResidual4x4, CabacResidualError> {
    let residual =
        decode_residual_category_levels(decoder, bank, contexts, coded_block_context, 0, 8, None)?;
    let mut compact = [0_i32; 16];
    for (scan_index, level) in residual.levels.iter().copied().enumerate() {
        if scan_index >= CHROMA422_DC_SCAN.len() {
            break;
        }
        let (row, column) = CHROMA422_DC_SCAN[scan_index];
        compact[row * 4 + column] = level;
    }
    Ok(CabacResidual4x4 {
        block: ResidualBlock4x4::from_raster_array(compact),
        non_zero_coefficients: residual.non_zero_coefficients,
    })
}

#[inline(never)]
pub fn decode_residual_8x8(
    decoder: &mut CabacDecoder<'_>,
    contexts: &mut CabacResidual8x8Contexts,
) -> Result<CabacResidual8x8, CabacResidualError> {
    let mut significant = [0_usize; 64];
    let mut significant_len = 0_usize;
    for local_scan_index in 0..63 {
        let significant_context = SIGNIFICANT_COEFF_FLAG_OFFSET_8X8[local_scan_index].min(14);
        if decoder.decode_bin(&mut contexts.significant_coeff_flag[significant_context]) {
            significant[significant_len] = local_scan_index;
            significant_len += 1;
            let last_context = LAST_COEFF_FLAG_OFFSET_8X8[local_scan_index].min(8);
            if decoder.decode_bin(&mut contexts.last_significant_coeff_flag[last_context]) {
                return decode_significant_levels_8x8(
                    decoder,
                    contexts,
                    &significant[..significant_len],
                );
            }
        }
    }
    significant[significant_len] = 63;
    significant_len += 1;
    decode_significant_levels_8x8(decoder, contexts, &significant[..significant_len])
}

fn decode_residual_4x4_subset(
    decoder: &mut CabacDecoder<'_>,
    contexts: &mut CabacResidual4x4Contexts,
    scan_offset: usize,
    max_coeff: usize,
) -> Result<CabacResidual4x4, CabacResidualError> {
    if !decoder.decode_decision(&mut contexts.coded_block_flag)? {
        return Ok(CabacResidual4x4 {
            block: ResidualBlock4x4::zero(),
            non_zero_coefficients: 0,
        });
    }

    let mut significant = [0_usize; 16];
    let mut significant_len = 0_usize;
    for local_scan_index in 0..max_coeff.saturating_sub(1) {
        let context_index =
            local_scan_index.min(contexts.significant_coeff_flag.len().saturating_sub(1));
        if decoder.decode_decision(&mut contexts.significant_coeff_flag[context_index])? {
            significant[significant_len] = scan_offset + local_scan_index;
            significant_len += 1;
            if decoder.decode_decision(&mut contexts.last_significant_coeff_flag[context_index])? {
                return decode_significant_levels(
                    decoder,
                    contexts,
                    &significant[..significant_len],
                );
            }
        }
    }
    significant[significant_len] = scan_offset + max_coeff.saturating_sub(1);
    significant_len += 1;
    decode_significant_levels(decoder, contexts, &significant[..significant_len])
}

fn decode_residual_category_subset(
    decoder: &mut CabacDecoder<'_>,
    bank: &mut [CabacContext],
    contexts: &CabacResidualCategoryContexts,
    coded_block_context: usize,
    scan_offset: usize,
    max_coeff: usize,
) -> Result<CabacResidual4x4, CabacResidualError> {
    let residual = decode_residual_category_levels(
        decoder,
        bank,
        contexts,
        coded_block_context,
        scan_offset,
        max_coeff,
        None,
    )?;
    Ok(CabacResidual4x4 {
        block: ResidualBlock4x4::from_raster_array(residual.levels),
        non_zero_coefficients: residual.non_zero_coefficients,
    })
}

struct CabacResidualLevels4x4 {
    levels: [i32; 16],
    non_zero_coefficients: u8,
}

#[inline(never)]
fn decode_residual_category_levels(
    decoder: &mut CabacDecoder<'_>,
    bank: &mut [CabacContext],
    contexts: &CabacResidualCategoryContexts,
    coded_block_context: usize,
    scan_offset: usize,
    max_coeff: usize,
    prepared: Option<&PreparedDequant4x4>,
) -> Result<CabacResidualLevels4x4, CabacResidualError> {
    let coded_block_context = coded_block_context.min(3);
    let coded_block_flag_base = contexts.coded_block_flag_base;
    let significant_base = contexts.significant_base;
    let last_base = contexts.last_base;
    let abs_base = contexts.abs_base;
    let chroma422_dc = contexts.chroma422_dc;
    if !decode_bank_bin(decoder, bank, coded_block_flag_base + coded_block_context) {
        ensure_cabac(decoder)?;
        return Ok(CabacResidualLevels4x4 {
            levels: [0; 16],
            non_zero_coefficients: 0,
        });
    }

    let mut significant = [0_usize; 16];
    let mut significant_len = 0_usize;
    for local_scan_index in 0..max_coeff.saturating_sub(1) {
        let scan_index = scan_offset + local_scan_index;
        let context_index = significant_context_index(chroma422_dc, local_scan_index).min(14);
        if decode_bank_bin(decoder, bank, significant_base + context_index) {
            significant[significant_len] = scan_index;
            significant_len += 1;
            if decode_bank_bin(decoder, bank, last_base + context_index) {
                return decode_significant_level_values_category(
                    decoder,
                    bank,
                    abs_base,
                    chroma422_dc,
                    &significant[..significant_len],
                    prepared,
                );
            }
        }
    }
    significant[significant_len] = scan_offset + max_coeff.saturating_sub(1);
    significant_len += 1;
    decode_significant_level_values_category(
        decoder,
        bank,
        abs_base,
        chroma422_dc,
        &significant[..significant_len],
        prepared,
    )
}

fn significant_context_index(chroma422_dc: bool, scan_index: usize) -> usize {
    if chroma422_dc {
        const CHROMA422_DC_SIG_CTX: [usize; 8] = [0, 0, 1, 1, 2, 2, 2, 2];
        CHROMA422_DC_SIG_CTX
            .get(scan_index)
            .copied()
            .unwrap_or(*CHROMA422_DC_SIG_CTX.last().unwrap())
    } else {
        scan_index
    }
}

fn decode_significant_levels(
    decoder: &mut CabacDecoder<'_>,
    contexts: &mut CabacResidual4x4Contexts,
    significant: &[usize],
) -> Result<CabacResidual4x4, CabacResidualError> {
    let mut levels = [0_i32; 16];
    let mut decoded_abs_level_eq1 = 0_u32;
    let mut decoded_abs_level_gt1 = 0_u32;

    for scan_index in significant.iter().rev().copied() {
        let abs_level_minus1 = decode_coeff_abs_level_minus1(
            decoder,
            contexts,
            decoded_abs_level_eq1,
            decoded_abs_level_gt1,
        )?;
        let abs_level = abs_level_minus1
            .checked_add(1)
            .ok_or(CabacResidualError::LevelOverflow)?;
        let sign = if decoder.decode_bypass()? { -1 } else { 1 };
        let (row, column) = ZIGZAG_4X4[scan_index];
        levels[row * 4 + column] =
            i32::try_from(abs_level).map_err(|_| CabacResidualError::LevelOverflow)? * sign;

        if abs_level == 1 {
            decoded_abs_level_eq1 = decoded_abs_level_eq1.saturating_add(1);
        } else {
            decoded_abs_level_gt1 = decoded_abs_level_gt1.saturating_add(1);
        }
    }

    Ok(CabacResidual4x4 {
        block: ResidualBlock4x4::from_raster_array(levels),
        non_zero_coefficients: significant.len() as u8,
    })
}

#[inline(always)]
fn decode_significant_level_values_category(
    decoder: &mut CabacDecoder<'_>,
    bank: &mut [CabacContext],
    abs_base: usize,
    chroma422_dc: bool,
    significant: &[usize],
    prepared: Option<&PreparedDequant4x4>,
) -> Result<CabacResidualLevels4x4, CabacResidualError> {
    let mut levels = [0_i32; 16];
    let mut node_context = 0_usize;

    for scan_index in significant.iter().rev().copied() {
        let first_context = COEFF_ABS_LEVEL1_CTX[node_context];
        let abs_level = if !decode_bank_bin(decoder, bank, abs_base + first_context) {
            node_context = COEFF_ABS_LEVEL_TRANSITION[0][node_context];
            1_u32
        } else {
            let follow_context = if chroma422_dc {
                COEFF_ABS_LEVELGT1_CTX_422DC[node_context]
            } else {
                COEFF_ABS_LEVELGT1_CTX[node_context]
            };
            let coeff_abs = 2_u32
                .checked_add(decode_ueg0_level_category(
                    decoder,
                    bank,
                    abs_base + follow_context,
                )?)
                .ok_or(CabacResidualError::LevelOverflow)?;
            node_context = COEFF_ABS_LEVEL_TRANSITION[1][node_context];
            coeff_abs
        };

        let sign = if decoder.decode_bypass_bin() { -1 } else { 1 };
        let level = i32::try_from(abs_level).map_err(|_| CabacResidualError::LevelOverflow)? * sign;
        if chroma422_dc {
            levels[scan_index] = level;
        } else {
            let (row, column) = ZIGZAG_4X4[scan_index];
            levels[row * 4 + column] = if let Some(prepared) = prepared {
                prepared.scale_coefficient(row, column, level)
            } else {
                level
            };
        }
    }

    ensure_cabac(decoder)?;
    Ok(CabacResidualLevels4x4 {
        levels,
        non_zero_coefficients: significant.len() as u8,
    })
}

const COEFF_ABS_LEVEL1_CTX: [usize; 8] = [1, 2, 3, 4, 0, 0, 0, 0];
const COEFF_ABS_LEVELGT1_CTX: [usize; 8] = [5, 5, 5, 5, 6, 7, 8, 9];
const COEFF_ABS_LEVELGT1_CTX_422DC: [usize; 8] = [5, 5, 5, 5, 6, 7, 8, 8];
const COEFF_ABS_LEVEL_TRANSITION: [[usize; 8]; 2] =
    [[1, 2, 3, 3, 4, 5, 6, 7], [4, 4, 4, 4, 5, 6, 7, 7]];

const CHROMA422_DC_SCAN: [(usize, usize); 8] = [
    (0, 0),
    (1, 0),
    (0, 1),
    (2, 0),
    (3, 0),
    (1, 1),
    (2, 1),
    (3, 1),
];

#[inline(always)]
fn decode_significant_levels_8x8(
    decoder: &mut CabacDecoder<'_>,
    contexts: &mut CabacResidual8x8Contexts,
    significant: &[usize],
) -> Result<CabacResidual8x8, CabacResidualError> {
    let mut levels = [0_i32; 64];
    let mut node_context = 0_usize;

    for scan_index in significant.iter().rev().copied() {
        let abs_level_minus1_first_context = COEFF_ABS_LEVEL1_CTX_8X8[node_context];
        let abs_level = if !decoder
            .decode_bin(&mut contexts.coeff_abs_level_minus1[abs_level_minus1_first_context])
        {
            node_context = COEFF_ABS_LEVEL_TRANSITION_8X8[0][node_context];
            1_u32
        } else {
            let follow_context = COEFF_ABS_LEVELGT1_CTX_8X8[node_context];
            let coeff_abs = 2_u32
                .checked_add(decode_ueg0_level(
                    decoder,
                    &mut contexts.coeff_abs_level_minus1[follow_context],
                )?)
                .ok_or(CabacResidualError::LevelOverflow)?;
            node_context = COEFF_ABS_LEVEL_TRANSITION_8X8[1][node_context];
            coeff_abs
        };

        let sign = if decoder.decode_bypass_bin() { -1 } else { 1 };
        let (row, column) = ZIGZAG_8X8[scan_index];
        levels[row * 8 + column] =
            i32::try_from(abs_level).map_err(|_| CabacResidualError::LevelOverflow)? * sign;
    }

    ensure_cabac(decoder)?;
    Ok(CabacResidual8x8 {
        block: ResidualBlock8x8::from_raster_array(levels),
        non_zero_coefficients: significant.len() as u8,
    })
}

const SIGNIFICANT_COEFF_FLAG_OFFSET_8X8: [usize; 64] = [
    0, 1, 2, 3, 4, 5, 5, 4, 4, 3, 3, 4, 4, 4, 5, 5, 4, 4, 4, 4, 3, 3, 6, 7, 7, 7, 8, 9, 10, 9, 8,
    7, 7, 6, 11, 12, 13, 11, 6, 7, 8, 9, 14, 10, 9, 8, 6, 11, 12, 13, 11, 6, 9, 14, 10, 9, 11, 12,
    13, 11, 14, 10, 12, 14,
];

const LAST_COEFF_FLAG_OFFSET_8X8: [usize; 63] = [
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8,
];

const COEFF_ABS_LEVEL1_CTX_8X8: [usize; 8] = [1, 2, 3, 4, 0, 0, 0, 0];
const COEFF_ABS_LEVELGT1_CTX_8X8: [usize; 8] = [5, 5, 5, 5, 6, 7, 8, 9];
const COEFF_ABS_LEVEL_TRANSITION_8X8: [[usize; 8]; 2] =
    [[1, 2, 3, 3, 4, 5, 6, 7], [4, 4, 4, 4, 5, 6, 7, 7]];

#[inline(always)]
fn decode_ueg0_level(
    decoder: &mut CabacDecoder<'_>,
    context: &mut CabacContext,
) -> Result<u32, CabacResidualError> {
    let mut coeff_abs = 2_u32;
    while coeff_abs < 15 {
        if !decoder.decode_bin(context) {
            return coeff_abs
                .checked_sub(2)
                .ok_or(CabacResidualError::LevelOverflow);
        }
        coeff_abs = coeff_abs
            .checked_add(1)
            .ok_or(CabacResidualError::LevelOverflow)?;
    }
    decode_large_coeff_abs_level_minus2(decoder)
}

#[inline(always)]
fn decode_ueg0_level_category(
    decoder: &mut CabacDecoder<'_>,
    bank: &mut [CabacContext],
    absolute_index: usize,
) -> Result<u32, CabacResidualError> {
    let mut coeff_abs = 2_u32;
    while coeff_abs < 15 {
        if !decode_bank_bin(decoder, bank, absolute_index) {
            return coeff_abs
                .checked_sub(2)
                .ok_or(CabacResidualError::LevelOverflow);
        }
        coeff_abs = coeff_abs
            .checked_add(1)
            .ok_or(CabacResidualError::LevelOverflow)?;
    }
    decode_large_coeff_abs_level_minus2(decoder)
}

#[inline(never)]
fn decode_large_coeff_abs_level_minus2(
    decoder: &mut CabacDecoder<'_>,
) -> Result<u32, CabacResidualError> {
    let mut suffix_len = 0_u32;
    while decoder.decode_bypass_bin() && suffix_len < 23 {
        suffix_len = suffix_len
            .checked_add(1)
            .ok_or(CabacResidualError::LevelOverflow)?;
    }

    let mut coeff_abs = 1_u32;
    while suffix_len > 0 {
        suffix_len -= 1;
        coeff_abs = coeff_abs
            .checked_add(coeff_abs)
            .ok_or(CabacResidualError::LevelOverflow)?;
        if decoder.decode_bypass_bin() {
            coeff_abs = coeff_abs
                .checked_add(1)
                .ok_or(CabacResidualError::LevelOverflow)?;
        }
    }

    ensure_cabac(decoder)?;
    coeff_abs
        .checked_add(14)
        .and_then(|value| value.checked_sub(2))
        .ok_or(CabacResidualError::LevelOverflow)
}

fn cabac_i_init(index: usize) -> CabacInitValue {
    const INIT_85_104: [(i8, i8); 20] = [
        (-17, 123),
        (-12, 115),
        (-16, 122),
        (-11, 115),
        (-12, 63),
        (-2, 68),
        (-15, 84),
        (-13, 104),
        (-3, 70),
        (-8, 93),
        (-10, 90),
        (-30, 127),
        (-1, 74),
        (-6, 97),
        (-7, 91),
        (-20, 127),
        (-4, 56),
        (-5, 82),
        (-7, 76),
        (-22, 125),
    ];
    const INIT_105_165: [(i8, i8); 61] = [
        (-7, 93),
        (-11, 87),
        (-3, 77),
        (-5, 71),
        (-4, 63),
        (-4, 68),
        (-12, 84),
        (-7, 62),
        (-7, 65),
        (8, 61),
        (5, 56),
        (-2, 66),
        (1, 64),
        (0, 61),
        (-2, 78),
        (1, 50),
        (7, 52),
        (10, 35),
        (0, 44),
        (11, 38),
        (1, 45),
        (0, 46),
        (5, 44),
        (31, 17),
        (1, 51),
        (7, 50),
        (28, 19),
        (16, 33),
        (14, 62),
        (-13, 108),
        (-15, 100),
        (-13, 101),
        (-13, 91),
        (-12, 94),
        (-10, 88),
        (-16, 84),
        (-10, 86),
        (-7, 83),
        (-13, 87),
        (-19, 94),
        (1, 70),
        (0, 72),
        (-5, 74),
        (18, 59),
        (-8, 102),
        (-15, 100),
        (0, 95),
        (-4, 75),
        (2, 72),
        (-11, 75),
        (-3, 71),
        (15, 46),
        (-13, 69),
        (0, 62),
        (0, 65),
        (21, 37),
        (-15, 72),
        (9, 57),
        (16, 54),
        (0, 62),
        (12, 72),
    ];
    const INIT_166_226: [(i8, i8); 61] = [
        (24, 0),
        (15, 9),
        (8, 25),
        (13, 18),
        (15, 9),
        (13, 19),
        (10, 37),
        (12, 18),
        (6, 29),
        (20, 33),
        (15, 30),
        (4, 45),
        (1, 58),
        (0, 62),
        (7, 61),
        (12, 38),
        (11, 45),
        (15, 39),
        (11, 42),
        (13, 44),
        (16, 45),
        (12, 41),
        (10, 49),
        (30, 34),
        (18, 42),
        (10, 55),
        (17, 51),
        (17, 46),
        (0, 89),
        (26, -19),
        (22, -17),
        (26, -17),
        (30, -25),
        (28, -20),
        (33, -23),
        (37, -27),
        (33, -23),
        (40, -28),
        (38, -17),
        (33, -11),
        (40, -15),
        (41, -6),
        (38, 1),
        (41, 17),
        (30, -6),
        (27, 3),
        (26, 22),
        (37, -16),
        (35, -4),
        (38, -8),
        (38, -3),
        (37, 3),
        (38, 5),
        (42, 0),
        (35, 16),
        (39, 22),
        (14, 48),
        (27, 37),
        (21, 60),
        (12, 68),
        (2, 97),
    ];
    const INIT_227_275: [(i8, i8); 49] = [
        (-3, 71),
        (-6, 42),
        (-5, 50),
        (-3, 54),
        (-2, 62),
        (0, 58),
        (1, 63),
        (-2, 72),
        (-1, 74),
        (-9, 91),
        (-5, 67),
        (-5, 27),
        (-3, 39),
        (-2, 44),
        (0, 46),
        (-16, 64),
        (-8, 68),
        (-10, 78),
        (-6, 77),
        (-10, 86),
        (-12, 92),
        (-15, 55),
        (-10, 60),
        (-6, 62),
        (-4, 65),
        (-12, 73),
        (-8, 76),
        (-7, 80),
        (-9, 88),
        (-17, 110),
        (-11, 97),
        (-20, 84),
        (-11, 79),
        (-6, 73),
        (-4, 74),
        (-13, 86),
        (-13, 96),
        (-11, 97),
        (-19, 117),
        (-8, 78),
        (-5, 33),
        (-4, 48),
        (-2, 53),
        (-3, 62),
        (-13, 71),
        (-10, 79),
        (-12, 86),
        (-13, 90),
        (-14, 97),
    ];
    const INIT_399_459: [(i8, i8); 61] = [
        (31, 21),
        (31, 31),
        (25, 50),
        (-17, 120),
        (-20, 112),
        (-18, 114),
        (-11, 85),
        (-15, 92),
        (-14, 89),
        (-26, 71),
        (-15, 81),
        (-14, 80),
        (0, 68),
        (-14, 70),
        (-24, 56),
        (-23, 68),
        (-24, 50),
        (-11, 74),
        (23, -13),
        (26, -13),
        (40, -15),
        (49, -14),
        (44, 3),
        (45, 6),
        (44, 34),
        (33, 54),
        (19, 82),
        (-3, 75),
        (-1, 23),
        (1, 34),
        (1, 43),
        (0, 54),
        (-2, 55),
        (0, 61),
        (1, 64),
        (0, 68),
        (-9, 92),
        (-14, 106),
        (-13, 97),
        (-15, 90),
        (-12, 90),
        (-18, 88),
        (-10, 73),
        (-9, 79),
        (-14, 86),
        (-10, 73),
        (-10, 70),
        (-10, 69),
        (-5, 66),
        (-9, 64),
        (-5, 58),
        (2, 59),
        (21, -10),
        (24, -11),
        (28, -8),
        (28, -1),
        (29, 3),
        (29, 9),
        (35, 20),
        (29, 36),
        (14, 67),
    ];
    let (m, n) = match index {
        85..=104 => INIT_85_104[index - 85],
        105..=165 => INIT_105_165[index - 105],
        166..=226 => INIT_166_226[index - 166],
        227..=275 => INIT_227_275[index - 227],
        399..=459 => INIT_399_459[index - 399],
        _ => (0, 0),
    };
    CabacInitValue::new(m, n)
}

fn decode_coeff_abs_level_minus1(
    decoder: &mut CabacDecoder<'_>,
    contexts: &mut CabacResidual4x4Contexts,
    decoded_abs_level_eq1: u32,
    decoded_abs_level_gt1: u32,
) -> Result<u32, CabacResidualError> {
    let first_context = if decoded_abs_level_gt1 == 0 {
        &mut contexts.coeff_abs_level_greater1
    } else {
        &mut contexts.coeff_abs_level_greater2
    };
    if !decoder.decode_decision(first_context)? {
        return Ok(0);
    }

    let mut suffix = 1_u32;
    let follow_context = if decoded_abs_level_eq1 < 4 {
        &mut contexts.coeff_abs_level_greater1
    } else {
        &mut contexts.coeff_abs_level_greater2
    };
    while suffix < 14 {
        if !decoder.decode_decision(follow_context)? {
            return Ok(suffix);
        }
        suffix += 1;
    }

    suffix
        .checked_add(decoder.decode_bypass_ue()?)
        .ok_or(CabacResidualError::LevelOverflow)
}

#[derive(Debug)]
pub enum CabacResidualError {
    Cabac(CabacError),
    LevelOverflow,
}

impl fmt::Display for CabacResidualError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cabac(error) => write!(f, "{error}"),
            Self::LevelOverflow => write!(f, "CABAC residual coefficient level overflowed"),
        }
    }
}

impl std::error::Error for CabacResidualError {}

impl From<CabacError> for CabacResidualError {
    fn from(value: CabacError) -> Self {
        Self::Cabac(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::residual::ZIGZAG_4X4;

    #[test]
    fn uncoded_residual_block_returns_zero_coefficients() {
        let mut decoder = CabacDecoder::new(&[0, 0, 0]).unwrap();
        let mut contexts = CabacResidual4x4Contexts::flat(CabacContext::new(0, false));

        let residual = decode_residual_4x4(&mut decoder, &mut contexts).unwrap();

        assert_eq!(residual.non_zero_coefficients, 0);
        assert_eq!(residual.block, ResidualBlock4x4::zero());
    }

    #[test]
    fn coded_residual_decodes_first_scan_coefficient() {
        let mut decoder = CabacDecoder::new(&[0x00, 0x00, 0x00, 0x00]).unwrap();
        let mut contexts = CabacResidual4x4Contexts::flat(CabacContext::new(0, true));
        contexts.coeff_abs_level_greater1 = CabacContext::new(0, false);
        contexts.coeff_abs_level_greater2 = CabacContext::new(0, false);

        let residual = decode_residual_4x4(&mut decoder, &mut contexts).unwrap();

        assert_eq!(residual.non_zero_coefficients, 1);
        let (row, column) = ZIGZAG_4X4[0];
        assert_ne!(residual.block.coeffs[row][column], 0);
    }

    #[test]
    fn residual_8x8_contexts_use_high_profile_init_values() {
        let contexts = CabacResidual8x8Contexts::i_slice(26);

        assert_eq!(
            contexts.significant_coeff_flag[0],
            CabacInitValue::new(-17, 120).initialize(26)
        );
        assert_eq!(
            contexts.last_significant_coeff_flag[0],
            CabacInitValue::new(23, -13).initialize(26)
        );
        assert_eq!(
            contexts.coeff_abs_level_minus1[0],
            CabacInitValue::new(-3, 75).initialize(26)
        );
    }

    #[test]
    fn coded_residual_8x8_decodes_first_scan_coefficient() {
        let mut decoder = CabacDecoder::new(&[0x00, 0x00, 0x00, 0x00]).unwrap();
        let mut contexts = CabacResidual8x8Contexts {
            significant_coeff_flag: [CabacContext::new(0, true); 15],
            last_significant_coeff_flag: [CabacContext::new(0, true); 9],
            coeff_abs_level_minus1: [CabacContext::new(0, false); 10],
        };

        let residual = decode_residual_8x8(&mut decoder, &mut contexts).unwrap();

        assert_eq!(residual.non_zero_coefficients, 1);
        assert_ne!(residual.block.coeffs[0][0], 0);
    }

    #[test]
    fn chroma422_dc_scan_maps_to_two_column_dc_plane() {
        let mut compact = [0_i32; 16];
        for (scan_index, (row, column)) in CHROMA422_DC_SCAN.iter().copied().enumerate() {
            compact[row * 4 + column] = (scan_index + 1) as i32;
        }

        let block = ResidualBlock4x4::from_raster(&compact);

        assert_eq!(block.coeffs[0][0], 1);
        assert_eq!(block.coeffs[1][0], 2);
        assert_eq!(block.coeffs[0][1], 3);
        assert_eq!(block.coeffs[2][0], 4);
        assert_eq!(block.coeffs[3][0], 5);
        assert_eq!(block.coeffs[1][1], 6);
        assert_eq!(block.coeffs[2][1], 7);
        assert_eq!(block.coeffs[3][1], 8);
    }
}
