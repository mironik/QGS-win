use std::fmt;

use crate::frame::DecodedFrame422P10;
use crate::macroblock_type::{MacroblockAddress, MacroblockGrid};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MotionVectorQuarterPel {
    pub x: i32,
    pub y: i32,
}

impl MotionVectorQuarterPel {
    pub const ZERO: Self = Self { x: 0, y: 0 };

    pub fn is_full_pel(self) -> bool {
        self.x % 4 == 0 && self.y % 4 == 0
    }

    pub fn checked_add(self, delta: Self) -> Result<Self, MotionCompensationError> {
        Ok(Self {
            x: self
                .x
                .checked_add(delta.x)
                .ok_or(MotionCompensationError::MotionVectorOverflow)?,
            y: self
                .y
                .checked_add(delta.y)
                .ok_or(MotionCompensationError::MotionVectorOverflow)?,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MotionSample {
    pub vector: MotionVectorQuarterPel,
    pub ref_index: u8,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MotionField {
    width_in_mbs: u32,
    height_in_mbs: u32,
    decoded: Vec<bool>,
    l0: Vec<MotionSample>,
    l1: Vec<MotionSample>,
    /// Per-macroblock bitmasks: bit N set means 4x4 scan block N has a sample.
    l0_present: Vec<u16>,
    l1_present: Vec<u16>,
    /// 4x4 blocks whose partition is already known and does not use that list.
    /// Same-macroblock neighbors must see these as ref -1, not as not-yet-decoded.
    l0_unused: Vec<u16>,
    l1_unused: Vec<u16>,
    slice_first_mb: u32,
}

impl MotionField {
    pub fn new(grid: MacroblockGrid) -> Result<Self, MotionCompensationError> {
        let macroblocks = usize::try_from(grid.macroblock_count())
            .map_err(|_| MotionCompensationError::OutOfBounds)?;
        let count = macroblocks.saturating_mul(16);
        Ok(Self {
            width_in_mbs: grid.width_in_mbs,
            height_in_mbs: grid.height_in_mbs,
            decoded: vec![false; macroblocks],
            l0: vec![
                MotionSample {
                    vector: MotionVectorQuarterPel::ZERO,
                    ref_index: 0,
                };
                count
            ],
            l1: vec![
                MotionSample {
                    vector: MotionVectorQuarterPel::ZERO,
                    ref_index: 0,
                };
                count
            ],
            l0_present: vec![0; macroblocks],
            l1_present: vec![0; macroblocks],
            l0_unused: vec![0; macroblocks],
            l1_unused: vec![0; macroblocks],
            slice_first_mb: 0,
        })
    }

    pub fn set_slice_first_mb(&mut self, slice_first_mb: u32) {
        self.slice_first_mb = slice_first_mb;
    }

    pub(crate) fn matches_grid(&self, grid: MacroblockGrid) -> bool {
        self.width_in_mbs == grid.width_in_mbs && self.height_in_mbs == grid.height_in_mbs
    }

    pub(crate) fn clear(&mut self) {
        self.decoded.fill(false);
        self.l0_present.fill(0);
        self.l1_present.fill(0);
        self.l0_unused.fill(0);
        self.l1_unused.fill(0);
        self.slice_first_mb = 0;
    }

    pub fn mark_list_unused(
        &mut self,
        address: MacroblockAddress,
        blocks: &[usize],
        use_l1: bool,
    ) -> Result<(), MotionCompensationError> {
        let index =
            usize::try_from(address.address).map_err(|_| MotionCompensationError::OutOfBounds)?;
        let mask = if use_l1 {
            &mut self.l1_unused
        } else {
            &mut self.l0_unused
        };
        let Some(slot) = mask.get_mut(index) else {
            return Err(MotionCompensationError::OutOfBounds);
        };
        for block in blocks {
            if *block >= 16 {
                return Err(MotionCompensationError::OutOfBounds);
            }
            *slot |= 1_u16 << *block;
        }
        Ok(())
    }

    pub fn set_inter(
        &mut self,
        address: MacroblockAddress,
        vector: MotionVectorQuarterPel,
        ref_index: u8,
    ) -> Result<(), MotionCompensationError> {
        self.set_l0(address, vector, ref_index)?;
        self.clear_l1(address)
    }

    pub fn set_l0(
        &mut self,
        address: MacroblockAddress,
        vector: MotionVectorQuarterPel,
        ref_index: u8,
    ) -> Result<(), MotionCompensationError> {
        self.write_blocks(address, &ALL_4X4_BLOCKS, vector, ref_index, false)
    }

    pub fn set_l1(
        &mut self,
        address: MacroblockAddress,
        vector: MotionVectorQuarterPel,
        ref_index: u8,
    ) -> Result<(), MotionCompensationError> {
        self.write_blocks(address, &ALL_4X4_BLOCKS, vector, ref_index, true)
    }

    pub fn set_l0_blocks(
        &mut self,
        address: MacroblockAddress,
        blocks: &[usize],
        vector: MotionVectorQuarterPel,
        ref_index: u8,
    ) -> Result<(), MotionCompensationError> {
        self.write_blocks(address, blocks, vector, ref_index, false)
    }

    pub fn set_l1_blocks(
        &mut self,
        address: MacroblockAddress,
        blocks: &[usize],
        vector: MotionVectorQuarterPel,
        ref_index: u8,
    ) -> Result<(), MotionCompensationError> {
        self.write_blocks(address, blocks, vector, ref_index, true)
    }

    pub fn clear_l0(&mut self, address: MacroblockAddress) -> Result<(), MotionCompensationError> {
        self.clear_blocks(address, &ALL_4X4_BLOCKS, false)
    }

    pub fn clear_l1(&mut self, address: MacroblockAddress) -> Result<(), MotionCompensationError> {
        self.clear_blocks(address, &ALL_4X4_BLOCKS, true)
    }

    pub fn set_intra(&mut self, address: MacroblockAddress) -> Result<(), MotionCompensationError> {
        self.clear_l0(address)?;
        self.clear_l1(address)?;
        self.mark_decoded(address)
    }

    pub fn l0_sample(&self, address: MacroblockAddress) -> Option<MotionSample> {
        self.sample_at(address.x, address.y, false)
    }

    pub fn l1_sample(&self, address: MacroblockAddress) -> Option<MotionSample> {
        self.sample_at(address.x, address.y, true)
    }

    pub fn block_motion(
        &self,
        address: MacroblockAddress,
        block: usize,
        use_l1: bool,
    ) -> Option<MotionSample> {
        self.block_sample(address.x, address.y, block, use_l1)
    }

    pub fn predict_p_skip(
        &self,
        address: MacroblockAddress,
    ) -> Result<MotionVectorQuarterPel, MotionCompensationError> {
        let left = self.neighbor_block(address, -1, 0, false);
        let top = self.neighbor_block(address, 0, -1, false);
        let zero_motion = |neighbor: MotionNeighbor| {
            neighbor.ref_index == 0 && neighbor.vector == MotionVectorQuarterPel::ZERO
        };
        if !left.partition_available
            || !top.partition_available
            || zero_motion(left)
            || zero_motion(top)
        {
            return Ok(MotionVectorQuarterPel::ZERO);
        }
        self.predict_l0_16x16(address, 0)
    }

    pub fn predict_l0_16x16(
        &self,
        address: MacroblockAddress,
        ref_index: u8,
    ) -> Result<MotionVectorQuarterPel, MotionCompensationError> {
        self.predict_list_16x16(address, ref_index, false)
    }

    pub fn predict_l1_16x16(
        &self,
        address: MacroblockAddress,
        ref_index: u8,
    ) -> Result<MotionVectorQuarterPel, MotionCompensationError> {
        self.predict_list_16x16(address, ref_index, true)
    }

    pub fn spatial_direct_16x16(
        &self,
        address: MacroblockAddress,
        colocated: Option<MotionSample>,
    ) -> Result<
        (
            Option<(MotionVectorQuarterPel, u8)>,
            Option<(MotionVectorQuarterPel, u8)>,
        ),
        MotionCompensationError,
    > {
        self.spatial_direct_at(address, 0, 0, 4, 4, colocated)
    }

    pub fn spatial_direct_at(
        &self,
        address: MacroblockAddress,
        origin_x: usize,
        origin_y: usize,
        width: usize,
        height: usize,
        colocated: Option<MotionSample>,
    ) -> Result<
        (
            Option<(MotionVectorQuarterPel, u8)>,
            Option<(MotionVectorQuarterPel, u8)>,
        ),
        MotionCompensationError,
    > {
        if origin_x >= 4
            || origin_y >= 4
            || width == 0
            || height == 0
            || origin_x + width > 4
            || origin_y + height > 4
        {
            return Err(MotionCompensationError::OutOfBounds);
        }
        let col_zero = colocated.is_some_and(|sample| {
            sample.ref_index == 0 && sample.vector.x.abs() <= 1 && sample.vector.y.abs() <= 1
        });
        let left = self.neighbor_pair(address, origin_x as i32 - 1, origin_y as i32);
        let top = self.neighbor_pair(address, origin_x as i32, origin_y as i32 - 1);
        let top_right = self.neighbor_pair(address, (origin_x + width) as i32, origin_y as i32 - 1);
        let top_left = self.neighbor_pair(address, origin_x as i32 - 1, origin_y as i32 - 1);
        let mut l0 =
            spatial_direct_from_neighbors(left.0, top.0, top_right.0, top_left.0, col_zero);
        let mut l1 =
            spatial_direct_from_neighbors(left.1, top.1, top_right.1, top_left.1, col_zero);
        if l0.is_none() && l1.is_none() {
            l0 = Some((MotionVectorQuarterPel::ZERO, 0));
            l1 = Some((MotionVectorQuarterPel::ZERO, 0));
        }
        Ok((l0, l1))
    }

    pub fn predict_partition(
        &self,
        address: MacroblockAddress,
        origin_x: usize,
        origin_y: usize,
        width: usize,
        height: usize,
        ref_index: u8,
        use_l1: bool,
    ) -> Result<MotionVectorQuarterPel, MotionCompensationError> {
        if origin_x >= 4
            || origin_y >= 4
            || width == 0
            || height == 0
            || origin_x + width > 4
            || origin_y + height > 4
        {
            return Err(MotionCompensationError::OutOfBounds);
        }
        let left = self.neighbor_block(address, origin_x as i32 - 1, origin_y as i32, use_l1);
        let top = self.neighbor_block(address, origin_x as i32, origin_y as i32 - 1, use_l1);
        let top_right = self.neighbor_block(
            address,
            (origin_x + width) as i32,
            origin_y as i32 - 1,
            use_l1,
        );
        let top_left =
            self.neighbor_block(address, origin_x as i32 - 1, origin_y as i32 - 1, use_l1);
        let diagonal = substitute_unavailable_diagonal(top_right, top_left);
        if let Some(vector) = directed_partition_motion(
            width, height, origin_x, origin_y, ref_index, left, top, diagonal,
        ) {
            return Ok(vector);
        }
        Ok(median_prediction(left, top, diagonal, ref_index))
    }

    fn predict_list_16x16(
        &self,
        address: MacroblockAddress,
        ref_index: u8,
        use_l1: bool,
    ) -> Result<MotionVectorQuarterPel, MotionCompensationError> {
        self.predict_partition(address, 0, 0, 4, 4, ref_index, use_l1)
    }

    fn neighbor_block(
        &self,
        address: MacroblockAddress,
        block_x: i32,
        block_y: i32,
        use_l1: bool,
    ) -> MotionNeighbor {
        let Some((macroblock, scan)) = self.neighbor_location(address, block_x, block_y) else {
            return MotionNeighbor::MISSING;
        };
        self.neighbor_at(address, macroblock, scan, use_l1)
    }

    #[inline(always)]
    fn neighbor_pair(
        &self,
        address: MacroblockAddress,
        block_x: i32,
        block_y: i32,
    ) -> (MotionNeighbor, MotionNeighbor) {
        let Some((macroblock, scan)) = self.neighbor_location(address, block_x, block_y) else {
            return (MotionNeighbor::MISSING, MotionNeighbor::MISSING);
        };
        (
            self.neighbor_at(address, macroblock, scan, false),
            self.neighbor_at(address, macroblock, scan, true),
        )
    }

    #[inline(always)]
    fn neighbor_location(
        &self,
        address: MacroblockAddress,
        block_x: i32,
        block_y: i32,
    ) -> Option<(usize, usize)> {
        let mut macroblock_x = i32::try_from(address.x).unwrap_or(i32::MAX);
        let mut macroblock_y = i32::try_from(address.y).unwrap_or(i32::MAX);
        let mut local_x = block_x;
        let mut local_y = block_y;
        if local_x < 0 {
            macroblock_x -= 1;
            local_x += 4;
        } else if local_x >= 4 {
            macroblock_x += 1;
            local_x -= 4;
        }
        if local_y < 0 {
            macroblock_y -= 1;
            local_y += 4;
        } else if local_y >= 4 {
            macroblock_y += 1;
            local_y -= 4;
        }
        if macroblock_x < 0
            || macroblock_y < 0
            || local_x < 0
            || local_y < 0
            || local_x >= 4
            || local_y >= 4
            || macroblock_x as u32 >= self.width_in_mbs
            || macroblock_y as u32 >= self.height_in_mbs
        {
            return None;
        }
        let scan = ((local_y as usize & 1) << 1)
            | (local_x as usize & 1)
            | ((local_x as usize & 2) << 1)
            | ((local_y as usize & 2) << 2);
        let macroblock = (macroblock_y as u32)
            .saturating_mul(self.width_in_mbs)
            .saturating_add(macroblock_x as u32);
        if macroblock < self.slice_first_mb {
            return None;
        }
        Some((macroblock as usize, scan))
    }

    #[inline(always)]
    fn neighbor_at(
        &self,
        address: MacroblockAddress,
        macroblock_index: usize,
        scan: usize,
        use_l1: bool,
    ) -> MotionNeighbor {
        let block_index = macroblock_index * 16 + scan;
        let bit = 1_u16 << scan;
        let (primary, primary_present, secondary_present, unused) = if use_l1 {
            (
                &self.l1,
                &self.l1_present,
                &self.l0_present,
                &self.l1_unused,
            )
        } else {
            (
                &self.l0,
                &self.l0_present,
                &self.l1_present,
                &self.l0_unused,
            )
        };
        if primary_present
            .get(macroblock_index)
            .is_some_and(|mask| mask & bit != 0)
        {
            if let Some(sample) = primary.get(block_index).copied() {
                return MotionNeighbor {
                    partition_available: true,
                    vector: sample.vector,
                    ref_index: i16::from(sample.ref_index),
                };
            }
        }
        if unused
            .get(macroblock_index)
            .is_some_and(|mask| mask & bit != 0)
        {
            return MotionNeighbor::INTRA;
        }
        if secondary_present
            .get(macroblock_index)
            .is_some_and(|mask| mask & bit != 0)
        {
            return MotionNeighbor::INTRA;
        }
        let finished = self.decoded.get(macroblock_index).copied().unwrap_or(false)
            && macroblock_index != address.address as usize;
        if finished {
            MotionNeighbor::INTRA
        } else {
            MotionNeighbor::MISSING
        }
    }

    fn mark_decoded(&mut self, address: MacroblockAddress) -> Result<(), MotionCompensationError> {
        let index =
            usize::try_from(address.address).map_err(|_| MotionCompensationError::OutOfBounds)?;
        let slot = self
            .decoded
            .get_mut(index)
            .ok_or(MotionCompensationError::OutOfBounds)?;
        *slot = true;
        Ok(())
    }

    fn sample_at(&self, x: u32, y: u32, use_l1: bool) -> Option<MotionSample> {
        self.block_sample(x, y, 0, use_l1)
    }

    fn block_sample(&self, x: u32, y: u32, block: usize, use_l1: bool) -> Option<MotionSample> {
        if x >= self.width_in_mbs || y >= self.height_in_mbs || block >= 16 {
            return None;
        }
        let macroblock =
            usize::try_from(y.saturating_mul(self.width_in_mbs).saturating_add(x)).ok()?;
        let present = if use_l1 {
            &self.l1_present
        } else {
            &self.l0_present
        };
        if present.get(macroblock).copied().unwrap_or(0) & (1_u16 << block) == 0 {
            return None;
        }
        let samples = if use_l1 { &self.l1 } else { &self.l0 };
        samples.get(macroblock * 16 + block).copied()
    }

    fn write_blocks(
        &mut self,
        address: MacroblockAddress,
        blocks: &[usize],
        vector: MotionVectorQuarterPel,
        ref_index: u8,
        use_l1: bool,
    ) -> Result<(), MotionCompensationError> {
        let base = self.macroblock_base(address)?;
        let mb = address.address as usize;
        let (samples, present) = if use_l1 {
            (&mut self.l1, &mut self.l1_present)
        } else {
            (&mut self.l0, &mut self.l0_present)
        };
        let Some(mask) = present.get_mut(mb) else {
            return Err(MotionCompensationError::OutOfBounds);
        };
        for block in blocks {
            let Some(slot) = samples.get_mut(base + *block) else {
                return Err(MotionCompensationError::OutOfBounds);
            };
            *slot = MotionSample { vector, ref_index };
            *mask |= 1_u16 << *block;
        }
        self.mark_decoded(address)?;
        Ok(())
    }

    fn clear_blocks(
        &mut self,
        address: MacroblockAddress,
        blocks: &[usize],
        use_l1: bool,
    ) -> Result<(), MotionCompensationError> {
        let mb = address.address as usize;
        let present = if use_l1 {
            &mut self.l1_present
        } else {
            &mut self.l0_present
        };
        let Some(mask) = present.get_mut(mb) else {
            return Err(MotionCompensationError::OutOfBounds);
        };
        for block in blocks {
            if *block >= 16 {
                return Err(MotionCompensationError::OutOfBounds);
            }
            *mask &= !(1_u16 << *block);
        }
        Ok(())
    }

    fn macroblock_base(
        &self,
        address: MacroblockAddress,
    ) -> Result<usize, MotionCompensationError> {
        if address.x >= self.width_in_mbs || address.y >= self.height_in_mbs {
            return Err(MotionCompensationError::OutOfBounds);
        }
        usize::try_from(address.address)
            .map(|index| index.saturating_mul(16))
            .map_err(|_| MotionCompensationError::OutOfBounds)
    }
}

const ALL_4X4_BLOCKS: [usize; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

fn directed_partition_motion(
    width: usize,
    height: usize,
    origin_x: usize,
    origin_y: usize,
    ref_index: u8,
    left: MotionNeighbor,
    top: MotionNeighbor,
    diagonal: MotionNeighbor,
) -> Option<MotionVectorQuarterPel> {
    let same = |neighbor: MotionNeighbor| {
        (neighbor.ref_index == i16::from(ref_index)).then_some(neighbor.vector)
    };
    if width == 4 && height == 2 && origin_y == 0 {
        return same(top);
    }
    if width == 4 && height == 2 && origin_y == 2 {
        return same(left);
    }
    if width == 2 && height == 4 && origin_x == 0 {
        return same(left);
    }
    if width == 2 && height == 4 && origin_x == 2 {
        return same(diagonal);
    }
    None
}

#[derive(Clone, Copy)]
struct MotionNeighbor {
    partition_available: bool,
    vector: MotionVectorQuarterPel,
    ref_index: i16,
}

impl MotionNeighbor {
    const MISSING: Self = Self {
        partition_available: false,
        vector: MotionVectorQuarterPel::ZERO,
        ref_index: -1,
    };
    const INTRA: Self = Self {
        partition_available: true,
        vector: MotionVectorQuarterPel::ZERO,
        ref_index: -1,
    };
}

fn substitute_unavailable_diagonal(
    top_right: MotionNeighbor,
    top_left: MotionNeighbor,
) -> MotionNeighbor {
    if !top_right.partition_available && top_left.partition_available {
        top_left
    } else {
        top_right
    }
}

#[inline(always)]
fn spatial_direct_from_neighbors(
    left: MotionNeighbor,
    top: MotionNeighbor,
    top_right: MotionNeighbor,
    top_left: MotionNeighbor,
    col_zero: bool,
) -> Option<(MotionVectorQuarterPel, u8)> {
    let diagonal = substitute_unavailable_diagonal(top_right, top_left);
    let reference = min_positive(
        diagonal_ref(left),
        min_positive(diagonal_ref(top), diagonal_ref(diagonal)),
    );
    if reference < 0 {
        return None;
    }
    let mut motion = median_prediction(left, top, diagonal, reference as u8);
    if col_zero && reference == 0 {
        motion = MotionVectorQuarterPel::ZERO;
    }
    Some((motion, reference as u8))
}

fn diagonal_ref(neighbor: MotionNeighbor) -> i16 {
    neighbor.ref_index
}

fn median_prediction(
    left: MotionNeighbor,
    top: MotionNeighbor,
    diagonal: MotionNeighbor,
    ref_index: u8,
) -> MotionVectorQuarterPel {
    let (left, top, diagonal) =
        if !top.partition_available && !diagonal.partition_available && left.partition_available {
            (left, left, left)
        } else {
            (left, top, diagonal)
        };
    let current = i16::from(ref_index);
    let left_match = left.ref_index == current;
    let top_match = top.ref_index == current;
    let diagonal_match = diagonal.ref_index == current;
    let matches = u8::from(left_match) + u8::from(top_match) + u8::from(diagonal_match);
    if matches == 1 {
        if left_match {
            return left.vector;
        }
        if top_match {
            return top.vector;
        }
        return diagonal.vector;
    }
    MotionVectorQuarterPel {
        x: median3(left.vector.x, top.vector.x, diagonal.vector.x),
        y: median3(left.vector.y, top.vector.y, diagonal.vector.y),
    }
}

fn min_positive(left: i16, right: i16) -> i16 {
    match (left >= 0, right >= 0) {
        (true, true) => left.min(right),
        (true, false) => left,
        (false, true) => right,
        (false, false) => -1,
    }
}

fn median3(a: i32, b: i32, c: i32) -> i32 {
    a + b + c - a.min(b).min(c) - a.max(b).max(c)
}

pub fn predict_inter_16x16(
    reference: &DecodedFrame422P10,
    target: &mut DecodedFrame422P10,
    dst_mb_x: usize,
    dst_mb_y: usize,
    motion: MotionVectorQuarterPel,
) -> Result<(), MotionCompensationError> {
    let dst_luma_x = dst_mb_x
        .checked_mul(16)
        .ok_or(MotionCompensationError::OutOfBounds)?;
    let dst_luma_y = dst_mb_y
        .checked_mul(16)
        .ok_or(MotionCompensationError::OutOfBounds)?;
    copy_luma_prediction(reference, target, dst_luma_x, dst_luma_y, motion)?;

    let dst_chroma_x = dst_mb_x
        .checked_mul(8)
        .ok_or(MotionCompensationError::OutOfBounds)?;
    let dst_chroma_y = dst_mb_y
        .checked_mul(16)
        .ok_or(MotionCompensationError::OutOfBounds)?;
    copy_chroma_prediction(reference, target, dst_chroma_x, dst_chroma_y, motion)?;
    Ok(())
}

pub(crate) fn predict_inter_luma_16x16_block(
    reference: &DecodedFrame422P10,
    dst_mb_x: usize,
    dst_mb_y: usize,
    motion: MotionVectorQuarterPel,
    output: &mut [u16; 16 * 16],
) -> Result<(), MotionCompensationError> {
    let dst_x = dst_mb_x
        .checked_mul(16)
        .ok_or(MotionCompensationError::OutOfBounds)?;
    let dst_y = dst_mb_y
        .checked_mul(16)
        .ok_or(MotionCompensationError::OutOfBounds)?;
    crate::mc_interp::predict_luma_block(&reference.y, output, 16, dst_x, dst_y, 16, 16, motion)
}

pub(crate) fn predict_inter_chroma_16x16_blocks(
    reference: &DecodedFrame422P10,
    dst_mb_x: usize,
    dst_mb_y: usize,
    motion: MotionVectorQuarterPel,
    cb: &mut [u16; 8 * 16],
    cr: &mut [u16; 8 * 16],
) -> Result<(), MotionCompensationError> {
    let dst_x = dst_mb_x
        .checked_mul(8)
        .ok_or(MotionCompensationError::OutOfBounds)?;
    let dst_y = dst_mb_y
        .checked_mul(16)
        .ok_or(MotionCompensationError::OutOfBounds)?;
    crate::mc_interp::predict_chroma_block(
        &reference.cb,
        &reference.cr,
        cb,
        cr,
        dst_x,
        dst_y,
        motion,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn predict_inter_luma_region_into(
    reference: &DecodedFrame422P10,
    dst_luma_x: usize,
    dst_luma_y: usize,
    width: usize,
    height: usize,
    motion: MotionVectorQuarterPel,
    output: &mut [u16],
    output_stride: usize,
) -> Result<(), MotionCompensationError> {
    crate::mc_interp::predict_luma_block(
        &reference.y,
        output,
        output_stride,
        dst_luma_x,
        dst_luma_y,
        width,
        height,
        motion,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn predict_inter_chroma_region_into(
    reference: &DecodedFrame422P10,
    dst_chroma_x: usize,
    dst_chroma_y: usize,
    local_x: usize,
    local_y: usize,
    width: usize,
    height: usize,
    motion: MotionVectorQuarterPel,
    cb: &mut [u16; 8 * 16],
    cr: &mut [u16; 8 * 16],
) -> Result<(), MotionCompensationError> {
    crate::mc_interp::predict_chroma_region_into(
        &reference.cb,
        &reference.cr,
        cb,
        cr,
        8,
        local_x,
        local_y,
        dst_chroma_x,
        dst_chroma_y,
        width,
        height,
        motion,
    )
}

pub fn predict_bi_inter_16x16(
    list0: &DecodedFrame422P10,
    list1: &DecodedFrame422P10,
    target: &mut DecodedFrame422P10,
    dst_mb_x: usize,
    dst_mb_y: usize,
    motion0: MotionVectorQuarterPel,
    motion1: MotionVectorQuarterPel,
    weight0: i32,
    weight1: i32,
) -> Result<(), MotionCompensationError> {
    let dst_luma_x = dst_mb_x
        .checked_mul(16)
        .ok_or(MotionCompensationError::OutOfBounds)?;
    let dst_luma_y = dst_mb_y
        .checked_mul(16)
        .ok_or(MotionCompensationError::OutOfBounds)?;
    copy_bi_luma_prediction_region(
        list0, list1, target, dst_luma_x, dst_luma_y, 16, 16, motion0, motion1, weight0, weight1,
    )?;

    let dst_chroma_x = dst_mb_x
        .checked_mul(8)
        .ok_or(MotionCompensationError::OutOfBounds)?;
    let dst_chroma_y = dst_mb_y
        .checked_mul(16)
        .ok_or(MotionCompensationError::OutOfBounds)?;
    copy_bi_chroma_prediction_region(
        list0,
        list1,
        target,
        dst_chroma_x,
        dst_chroma_y,
        8,
        16,
        motion0,
        motion1,
        weight0,
        weight1,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn predict_bi_inter_luma_16x16_block(
    list0: &DecodedFrame422P10,
    list1: &DecodedFrame422P10,
    dst_mb_x: usize,
    dst_mb_y: usize,
    motion0: MotionVectorQuarterPel,
    motion1: MotionVectorQuarterPel,
    weight0: i32,
    weight1: i32,
    output: &mut [u16; 16 * 16],
) -> Result<(), MotionCompensationError> {
    let dst_x = dst_mb_x
        .checked_mul(16)
        .ok_or(MotionCompensationError::OutOfBounds)?;
    let dst_y = dst_mb_y
        .checked_mul(16)
        .ok_or(MotionCompensationError::OutOfBounds)?;
    crate::mc_interp::predict_luma_bi_block(
        &list0.y, &list1.y, output, dst_x, dst_y, motion0, motion1, weight0, weight1,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn predict_bi_inter_chroma_16x16_blocks(
    list0: &DecodedFrame422P10,
    list1: &DecodedFrame422P10,
    dst_mb_x: usize,
    dst_mb_y: usize,
    motion0: MotionVectorQuarterPel,
    motion1: MotionVectorQuarterPel,
    weight0: i32,
    weight1: i32,
    cb: &mut [u16; 8 * 16],
    cr: &mut [u16; 8 * 16],
) -> Result<(), MotionCompensationError> {
    let dst_x = dst_mb_x
        .checked_mul(8)
        .ok_or(MotionCompensationError::OutOfBounds)?;
    let dst_y = dst_mb_y
        .checked_mul(16)
        .ok_or(MotionCompensationError::OutOfBounds)?;
    crate::mc_interp::predict_chroma_bi_block(
        &list0.cb, &list0.cr, &list1.cb, &list1.cr, cb, cr, dst_x, dst_y, motion0, motion1,
        weight0, weight1,
    )
}

pub fn predict_inter_16x8(
    reference: &DecodedFrame422P10,
    target: &mut DecodedFrame422P10,
    dst_mb_x: usize,
    dst_mb_y: usize,
    partition_index: usize,
    motion: MotionVectorQuarterPel,
) -> Result<(), MotionCompensationError> {
    if partition_index >= 2 {
        return Err(MotionCompensationError::OutOfBounds);
    }
    let dst_luma_x = dst_mb_x
        .checked_mul(16)
        .ok_or(MotionCompensationError::OutOfBounds)?;
    let dst_luma_y = dst_mb_y
        .checked_mul(16)
        .and_then(|value| value.checked_add(partition_index * 8))
        .ok_or(MotionCompensationError::OutOfBounds)?;
    copy_luma_prediction_region(reference, target, dst_luma_x, dst_luma_y, 16, 8, motion)?;

    let dst_chroma_x = dst_mb_x
        .checked_mul(8)
        .ok_or(MotionCompensationError::OutOfBounds)?;
    let dst_chroma_y = dst_mb_y
        .checked_mul(16)
        .and_then(|value| value.checked_add(partition_index * 8))
        .ok_or(MotionCompensationError::OutOfBounds)?;
    copy_chroma_prediction_region(reference, target, dst_chroma_x, dst_chroma_y, 8, 8, motion)?;
    Ok(())
}

pub fn predict_bi_inter_16x8(
    list0: &DecodedFrame422P10,
    list1: &DecodedFrame422P10,
    target: &mut DecodedFrame422P10,
    dst_mb_x: usize,
    dst_mb_y: usize,
    partition_index: usize,
    motion0: MotionVectorQuarterPel,
    motion1: MotionVectorQuarterPel,
    weight0: i32,
    weight1: i32,
) -> Result<(), MotionCompensationError> {
    if partition_index >= 2 {
        return Err(MotionCompensationError::OutOfBounds);
    }
    let dst_luma_x = dst_mb_x
        .checked_mul(16)
        .ok_or(MotionCompensationError::OutOfBounds)?;
    let dst_luma_y = dst_mb_y
        .checked_mul(16)
        .and_then(|value| value.checked_add(partition_index * 8))
        .ok_or(MotionCompensationError::OutOfBounds)?;
    copy_bi_luma_prediction_region(
        list0, list1, target, dst_luma_x, dst_luma_y, 16, 8, motion0, motion1, weight0, weight1,
    )?;

    let dst_chroma_x = dst_mb_x
        .checked_mul(8)
        .ok_or(MotionCompensationError::OutOfBounds)?;
    let dst_chroma_y = dst_mb_y
        .checked_mul(16)
        .and_then(|value| value.checked_add(partition_index * 8))
        .ok_or(MotionCompensationError::OutOfBounds)?;
    copy_bi_chroma_prediction_region(
        list0,
        list1,
        target,
        dst_chroma_x,
        dst_chroma_y,
        8,
        8,
        motion0,
        motion1,
        weight0,
        weight1,
    )?;
    Ok(())
}

pub fn predict_inter_8x16(
    reference: &DecodedFrame422P10,
    target: &mut DecodedFrame422P10,
    dst_mb_x: usize,
    dst_mb_y: usize,
    partition_index: usize,
    motion: MotionVectorQuarterPel,
) -> Result<(), MotionCompensationError> {
    if partition_index >= 2 {
        return Err(MotionCompensationError::OutOfBounds);
    }
    let dst_luma_x = dst_mb_x
        .checked_mul(16)
        .and_then(|value| value.checked_add(partition_index * 8))
        .ok_or(MotionCompensationError::OutOfBounds)?;
    let dst_luma_y = dst_mb_y
        .checked_mul(16)
        .ok_or(MotionCompensationError::OutOfBounds)?;
    copy_luma_prediction_region(reference, target, dst_luma_x, dst_luma_y, 8, 16, motion)?;

    let dst_chroma_x = dst_mb_x
        .checked_mul(8)
        .and_then(|value| value.checked_add(partition_index * 4))
        .ok_or(MotionCompensationError::OutOfBounds)?;
    let dst_chroma_y = dst_mb_y
        .checked_mul(16)
        .ok_or(MotionCompensationError::OutOfBounds)?;
    copy_chroma_prediction_region(reference, target, dst_chroma_x, dst_chroma_y, 4, 16, motion)?;
    Ok(())
}

pub fn predict_inter_8x8(
    reference: &DecodedFrame422P10,
    target: &mut DecodedFrame422P10,
    dst_mb_x: usize,
    dst_mb_y: usize,
    subblock_index: usize,
    motion: MotionVectorQuarterPel,
) -> Result<(), MotionCompensationError> {
    if subblock_index >= 4 {
        return Err(MotionCompensationError::OutOfBounds);
    }
    let sub_x = subblock_index % 2;
    let sub_y = subblock_index / 2;
    let dst_luma_x = dst_mb_x
        .checked_mul(16)
        .and_then(|value| value.checked_add(sub_x * 8))
        .ok_or(MotionCompensationError::OutOfBounds)?;
    let dst_luma_y = dst_mb_y
        .checked_mul(16)
        .and_then(|value| value.checked_add(sub_y * 8))
        .ok_or(MotionCompensationError::OutOfBounds)?;
    copy_luma_prediction_region(reference, target, dst_luma_x, dst_luma_y, 8, 8, motion)?;

    let dst_chroma_x = dst_mb_x
        .checked_mul(8)
        .and_then(|value| value.checked_add(sub_x * 4))
        .ok_or(MotionCompensationError::OutOfBounds)?;
    let dst_chroma_y = dst_mb_y
        .checked_mul(16)
        .and_then(|value| value.checked_add(sub_y * 8))
        .ok_or(MotionCompensationError::OutOfBounds)?;
    copy_chroma_prediction_region(reference, target, dst_chroma_x, dst_chroma_y, 4, 8, motion)?;
    Ok(())
}

pub fn predict_inter_region(
    reference: &DecodedFrame422P10,
    target: &mut DecodedFrame422P10,
    dst_luma_x: usize,
    dst_luma_y: usize,
    luma_width: usize,
    luma_height: usize,
    motion: MotionVectorQuarterPel,
) -> Result<(), MotionCompensationError> {
    if !luma_width.is_multiple_of(2) {
        return Err(MotionCompensationError::OutOfBounds);
    }
    copy_luma_prediction_region(
        reference,
        target,
        dst_luma_x,
        dst_luma_y,
        luma_width,
        luma_height,
        motion,
    )?;
    copy_chroma_prediction_region(
        reference,
        target,
        dst_luma_x / 2,
        dst_luma_y,
        luma_width / 2,
        luma_height,
        motion,
    )?;
    Ok(())
}

pub fn predict_bi_inter_8x16(
    list0: &DecodedFrame422P10,
    list1: &DecodedFrame422P10,
    target: &mut DecodedFrame422P10,
    dst_mb_x: usize,
    dst_mb_y: usize,
    partition_index: usize,
    motion0: MotionVectorQuarterPel,
    motion1: MotionVectorQuarterPel,
    weight0: i32,
    weight1: i32,
) -> Result<(), MotionCompensationError> {
    if partition_index >= 2 {
        return Err(MotionCompensationError::OutOfBounds);
    }
    let dst_luma_x = dst_mb_x
        .checked_mul(16)
        .and_then(|value| value.checked_add(partition_index * 8))
        .ok_or(MotionCompensationError::OutOfBounds)?;
    let dst_luma_y = dst_mb_y
        .checked_mul(16)
        .ok_or(MotionCompensationError::OutOfBounds)?;
    copy_bi_luma_prediction_region(
        list0, list1, target, dst_luma_x, dst_luma_y, 8, 16, motion0, motion1, weight0, weight1,
    )?;

    let dst_chroma_x = dst_mb_x
        .checked_mul(8)
        .and_then(|value| value.checked_add(partition_index * 4))
        .ok_or(MotionCompensationError::OutOfBounds)?;
    let dst_chroma_y = dst_mb_y
        .checked_mul(16)
        .ok_or(MotionCompensationError::OutOfBounds)?;
    copy_bi_chroma_prediction_region(
        list0,
        list1,
        target,
        dst_chroma_x,
        dst_chroma_y,
        4,
        16,
        motion0,
        motion1,
        weight0,
        weight1,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn predict_bi_inter_8x8(
    list0: &DecodedFrame422P10,
    list1: &DecodedFrame422P10,
    target: &mut DecodedFrame422P10,
    dst_mb_x: usize,
    dst_mb_y: usize,
    subblock_index: usize,
    motion0: MotionVectorQuarterPel,
    motion1: MotionVectorQuarterPel,
) -> Result<(), MotionCompensationError> {
    if subblock_index >= 4 {
        return Err(MotionCompensationError::OutOfBounds);
    }
    let sub_x = subblock_index % 2;
    let sub_y = subblock_index / 2;
    let dst_luma_x = dst_mb_x
        .checked_mul(16)
        .and_then(|value| value.checked_add(sub_x * 8))
        .ok_or(MotionCompensationError::OutOfBounds)?;
    let dst_luma_y = dst_mb_y
        .checked_mul(16)
        .and_then(|value| value.checked_add(sub_y * 8))
        .ok_or(MotionCompensationError::OutOfBounds)?;
    copy_bi_luma_prediction_region(
        list0, list1, target, dst_luma_x, dst_luma_y, 8, 8, motion0, motion1, 32, 32,
    )?;

    let dst_chroma_x = dst_mb_x
        .checked_mul(8)
        .and_then(|value| value.checked_add(sub_x * 4))
        .ok_or(MotionCompensationError::OutOfBounds)?;
    let dst_chroma_y = dst_mb_y
        .checked_mul(16)
        .and_then(|value| value.checked_add(sub_y * 8))
        .ok_or(MotionCompensationError::OutOfBounds)?;
    copy_bi_chroma_prediction_region(
        list0,
        list1,
        target,
        dst_chroma_x,
        dst_chroma_y,
        4,
        8,
        motion0,
        motion1,
        32,
        32,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn predict_bi_inter_region(
    list0: &DecodedFrame422P10,
    list1: &DecodedFrame422P10,
    target: &mut DecodedFrame422P10,
    dst_luma_x: usize,
    dst_luma_y: usize,
    luma_width: usize,
    luma_height: usize,
    motion0: MotionVectorQuarterPel,
    motion1: MotionVectorQuarterPel,
    weight0: i32,
    weight1: i32,
) -> Result<(), MotionCompensationError> {
    if !luma_width.is_multiple_of(2) {
        return Err(MotionCompensationError::OutOfBounds);
    }
    copy_bi_luma_prediction_region(
        list0,
        list1,
        target,
        dst_luma_x,
        dst_luma_y,
        luma_width,
        luma_height,
        motion0,
        motion1,
        weight0,
        weight1,
    )?;
    copy_bi_chroma_prediction_region(
        list0,
        list1,
        target,
        dst_luma_x / 2,
        dst_luma_y,
        luma_width / 2,
        luma_height,
        motion0,
        motion1,
        weight0,
        weight1,
    )?;
    Ok(())
}

fn copy_luma_prediction(
    reference: &DecodedFrame422P10,
    target: &mut DecodedFrame422P10,
    dst_x: usize,
    dst_y: usize,
    motion: MotionVectorQuarterPel,
) -> Result<(), MotionCompensationError> {
    copy_luma_prediction_region(reference, target, dst_x, dst_y, 16, 16, motion)
}

fn copy_luma_prediction_region(
    reference: &DecodedFrame422P10,
    target: &mut DecodedFrame422P10,
    dst_x: usize,
    dst_y: usize,
    width: usize,
    height: usize,
    motion: MotionVectorQuarterPel,
) -> Result<(), MotionCompensationError> {
    crate::mc_interp::put_luma(
        &reference.y,
        &mut target.y,
        dst_x,
        dst_y,
        width,
        height,
        motion,
    )
}

#[allow(clippy::too_many_arguments)]
fn copy_bi_luma_prediction_region(
    list0: &DecodedFrame422P10,
    list1: &DecodedFrame422P10,
    target: &mut DecodedFrame422P10,
    dst_x: usize,
    dst_y: usize,
    width: usize,
    height: usize,
    motion0: MotionVectorQuarterPel,
    motion1: MotionVectorQuarterPel,
    weight0: i32,
    weight1: i32,
) -> Result<(), MotionCompensationError> {
    crate::mc_interp::put_luma_bi(
        &list0.y,
        &list1.y,
        &mut target.y,
        dst_x,
        dst_y,
        width,
        height,
        motion0,
        motion1,
        weight0,
        weight1,
    )
}

fn copy_chroma_prediction(
    reference: &DecodedFrame422P10,
    target: &mut DecodedFrame422P10,
    dst_x: usize,
    dst_y: usize,
    motion: MotionVectorQuarterPel,
) -> Result<(), MotionCompensationError> {
    copy_chroma_prediction_region(reference, target, dst_x, dst_y, 8, 16, motion)
}

fn copy_chroma_prediction_region(
    reference: &DecodedFrame422P10,
    target: &mut DecodedFrame422P10,
    dst_x: usize,
    dst_y: usize,
    width: usize,
    height: usize,
    motion: MotionVectorQuarterPel,
) -> Result<(), MotionCompensationError> {
    crate::mc_interp::put_chroma(
        &reference.cb,
        &reference.cr,
        &mut target.cb,
        &mut target.cr,
        dst_x,
        dst_y,
        width,
        height,
        motion,
    )
}

#[allow(clippy::too_many_arguments)]
fn copy_bi_chroma_prediction_region(
    list0: &DecodedFrame422P10,
    list1: &DecodedFrame422P10,
    target: &mut DecodedFrame422P10,
    dst_x: usize,
    dst_y: usize,
    width: usize,
    height: usize,
    motion0: MotionVectorQuarterPel,
    motion1: MotionVectorQuarterPel,
    weight0: i32,
    weight1: i32,
) -> Result<(), MotionCompensationError> {
    crate::mc_interp::put_chroma_bi(
        &list0.cb,
        &list0.cr,
        &list1.cb,
        &list1.cr,
        &mut target.cb,
        &mut target.cr,
        dst_x,
        dst_y,
        width,
        height,
        motion0,
        motion1,
        weight0,
        weight1,
    )
}

#[cfg(test)]
fn luma_quarter_sample(
    reference: &DecodedFrame422P10,
    quarter_x: i32,
    quarter_y: i32,
) -> Result<u16, MotionCompensationError> {
    let half_x0 = div_floor(quarter_x, 2);
    let half_y0 = div_floor(quarter_y, 2);
    let half_x1 = if quarter_x.rem_euclid(2) == 0 {
        half_x0
    } else {
        half_x0 + 1
    };
    let half_y1 = if quarter_y.rem_euclid(2) == 0 {
        half_y0
    } else {
        half_y0 + 1
    };
    let a = luma_half_grid_sample(reference, half_x0, half_y0)?;
    if half_x0 == half_x1 && half_y0 == half_y1 {
        return Ok(a);
    }
    let b = luma_half_grid_sample(reference, half_x1, half_y0)?;
    if half_y0 == half_y1 {
        return Ok(avg_round(a, b));
    }
    let c = luma_half_grid_sample(reference, half_x0, half_y1)?;
    if half_x0 == half_x1 {
        return Ok(avg_round(a, c));
    }
    let d = luma_half_grid_sample(reference, half_x1, half_y1)?;
    // Quarter-samples e, g, p and r average one diagonal of the half-sample square.
    if (half_x0 ^ half_y0) & 1 == 0 {
        Ok(avg_round(b, c))
    } else {
        Ok(avg_round(a, d))
    }
}

#[cfg(test)]
fn luma_half_grid_sample(
    reference: &DecodedFrame422P10,
    half_x: i32,
    half_y: i32,
) -> Result<u16, MotionCompensationError> {
    luma_half_sample(
        reference,
        div_floor(half_x, 2),
        div_floor(half_y, 2),
        half_x.rem_euclid(2) != 0,
        half_y.rem_euclid(2) != 0,
    )
}

#[cfg(test)]
fn luma_half_sample(
    reference: &DecodedFrame422P10,
    x: i32,
    y: i32,
    half_x: bool,
    half_y: bool,
) -> Result<u16, MotionCompensationError> {
    match (half_x, half_y) {
        (false, false) => sample_luma(reference, x, y),
        (true, false) => {
            let value = six_tap([
                sample_luma(reference, x - 2, y)?,
                sample_luma(reference, x - 1, y)?,
                sample_luma(reference, x, y)?,
                sample_luma(reference, x + 1, y)?,
                sample_luma(reference, x + 2, y)?,
                sample_luma(reference, x + 3, y)?,
            ]);
            Ok(clip10(value))
        }
        (false, true) => {
            let value = six_tap([
                sample_luma(reference, x, y - 2)?,
                sample_luma(reference, x, y - 1)?,
                sample_luma(reference, x, y)?,
                sample_luma(reference, x, y + 1)?,
                sample_luma(reference, x, y + 2)?,
                sample_luma(reference, x, y + 3)?,
            ]);
            Ok(clip10(value))
        }
        (true, true) => {
            let mut horizontal = [0_i32; 6];
            for (index, row_offset) in (-2..=3).enumerate() {
                horizontal[index] = six_tap_sum([
                    sample_luma(reference, x - 2, y + row_offset)?,
                    sample_luma(reference, x - 1, y + row_offset)?,
                    sample_luma(reference, x, y + row_offset)?,
                    sample_luma(reference, x + 1, y + row_offset)?,
                    sample_luma(reference, x + 2, y + row_offset)?,
                    sample_luma(reference, x + 3, y + row_offset)?,
                ]);
            }
            Ok(clip10(
                (horizontal[0] - 5 * horizontal[1] + 20 * horizontal[2] + 20 * horizontal[3]
                    - 5 * horizontal[4]
                    + horizontal[5]
                    + 512)
                    >> 10,
            ))
        }
    }
}

#[cfg(test)]
fn chroma_eighth_sample(
    plane: &crate::frame::Plane422P10,
    eighth_x: i32,
    eighth_y: i32,
) -> Result<u16, MotionCompensationError> {
    let base_x = div_floor(eighth_x, 8);
    let base_y = div_floor(eighth_y, 8);
    let dx = eighth_x.rem_euclid(8);
    let dy = eighth_y.rem_euclid(8);
    let a = i32::from(sample_plane(plane, base_x, base_y)?);
    let b = i32::from(sample_plane(plane, base_x + 1, base_y)?);
    let c = i32::from(sample_plane(plane, base_x, base_y + 1)?);
    let d = i32::from(sample_plane(plane, base_x + 1, base_y + 1)?);
    let value =
        ((8 - dx) * (8 - dy) * a + dx * (8 - dy) * b + (8 - dx) * dy * c + dx * dy * d + 32) >> 6;
    Ok(clip10(value))
}

#[cfg(test)]
fn six_tap(samples: [u16; 6]) -> i32 {
    (six_tap_sum(samples) + 16) >> 5
}

#[cfg(test)]
fn six_tap_sum(samples: [u16; 6]) -> i32 {
    i32::from(samples[0]) - 5 * i32::from(samples[1])
        + 20 * i32::from(samples[2])
        + 20 * i32::from(samples[3])
        - 5 * i32::from(samples[4])
        + i32::from(samples[5])
}

#[cfg(test)]
fn sample_luma(
    reference: &DecodedFrame422P10,
    x: i32,
    y: i32,
) -> Result<u16, MotionCompensationError> {
    sample_plane(&reference.y, x, y)
}

#[cfg(test)]
fn sample_plane(
    plane: &crate::frame::Plane422P10,
    x: i32,
    y: i32,
) -> Result<u16, MotionCompensationError> {
    if plane.width == 0 || plane.height == 0 {
        return Err(MotionCompensationError::OutOfBounds);
    }
    let x = x.clamp(0, plane.width.saturating_sub(1) as i32) as usize;
    let y = y.clamp(0, plane.height.saturating_sub(1) as i32) as usize;
    plane.get(x, y).ok_or(MotionCompensationError::OutOfBounds)
}

#[cfg(test)]
fn avg_round(a: u16, b: u16) -> u16 {
    weighted_avg(a, b, 32, 32)
}

#[cfg(test)]
fn weighted_avg(a: u16, b: u16, weight0: i32, weight1: i32) -> u16 {
    let value = (i32::from(a) * weight0 + i32::from(b) * weight1 + 32) >> 6;
    clip10(value)
}

#[cfg(test)]
fn clip10(value: i32) -> u16 {
    value.clamp(0, 1023) as u16
}

#[cfg(test)]
fn div_floor(value: i32, divisor: i32) -> i32 {
    let quotient = value / divisor;
    let remainder = value % divisor;
    if remainder != 0 && (remainder > 0) != (divisor > 0) {
        quotient - 1
    } else {
        quotient
    }
}

#[derive(Debug)]
pub enum MotionCompensationError {
    MotionVectorOverflow,
    OutOfBounds,
}

impl fmt::Display for MotionCompensationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MotionVectorOverflow => write!(f, "motion vector arithmetic overflowed"),
            Self::OutOfBounds => write!(f, "motion compensation reference is out of bounds"),
        }
    }
}

impl std::error::Error for MotionCompensationError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_pel_inter_prediction_copies_422_macroblock_planes() {
        let mut reference = DecodedFrame422P10::new(32, 16, 32, 16);
        for y in 0..16 {
            for x in 0..32 {
                assert!(reference.y.set(x, y, (x + y) as u16));
            }
            for x in 0..16 {
                assert!(reference.cb.set(x, y, (100 + x + y) as u16));
                assert!(reference.cr.set(x, y, (200 + x + y) as u16));
            }
        }
        let mut target = DecodedFrame422P10::new(32, 16, 32, 16);

        predict_inter_16x16(
            &reference,
            &mut target,
            1,
            0,
            MotionVectorQuarterPel { x: -64, y: 0 },
        )
        .unwrap();

        assert_eq!(target.y.get(16, 0), reference.y.get(0, 0));
        assert_eq!(target.y.get(31, 15), reference.y.get(15, 15));
        assert_eq!(target.cb.get(8, 0), reference.cb.get(0, 0));
        assert_eq!(target.cr.get(15, 15), reference.cr.get(7, 15));
    }

    #[test]
    fn fractional_pel_motion_interpolates_luma_and_chroma() {
        let mut reference = DecodedFrame422P10::new(32, 32, 32, 32);
        for y in 0..32 {
            for x in 0..32 {
                assert!(reference.y.set(x, y, (x * 4 + y * 2) as u16));
            }
            for x in 0..16 {
                assert!(reference.cb.set(x, y, (100 + x * 3 + y) as u16));
                assert!(reference.cr.set(x, y, (200 + x * 3 + y) as u16));
            }
        }
        let mut target = DecodedFrame422P10::new(32, 32, 32, 32);

        predict_inter_16x16(
            &reference,
            &mut target,
            0,
            0,
            MotionVectorQuarterPel { x: 2, y: 0 },
        )
        .unwrap();

        assert!(target.y.get(4, 4).unwrap() > reference.y.get(4, 4).unwrap());
        assert!(target.cb.get(4, 4).unwrap() >= reference.cb.get(4, 4).unwrap());
        assert!(target.cr.get(4, 4).unwrap() >= reference.cr.get(4, 4).unwrap());
    }

    #[test]
    fn inter_16x8_prediction_updates_only_selected_partition() {
        let mut reference = DecodedFrame422P10::new(16, 16, 16, 16);
        for y in 0..16 {
            for x in 0..16 {
                assert!(reference.y.set(x, y, (100 + y) as u16));
            }
            for x in 0..8 {
                assert!(reference.cb.set(x, y, (200 + y) as u16));
                assert!(reference.cr.set(x, y, (300 + y) as u16));
            }
        }
        let mut target = DecodedFrame422P10::new(16, 16, 16, 16);

        predict_inter_16x8(
            &reference,
            &mut target,
            0,
            0,
            1,
            MotionVectorQuarterPel::ZERO,
        )
        .unwrap();

        assert_eq!(target.y.get(0, 7), Some(0));
        assert_eq!(target.y.get(0, 8), Some(108));
        assert_eq!(target.cb.get(0, 7), Some(0));
        assert_eq!(target.cb.get(0, 8), Some(208));
    }

    #[test]
    fn inter_8x16_prediction_updates_only_selected_partition() {
        let mut reference = DecodedFrame422P10::new(16, 16, 16, 16);
        for y in 0..16 {
            for x in 0..16 {
                assert!(reference.y.set(x, y, (100 + x) as u16));
            }
            for x in 0..8 {
                assert!(reference.cb.set(x, y, (200 + x) as u16));
                assert!(reference.cr.set(x, y, (300 + x) as u16));
            }
        }
        let mut target = DecodedFrame422P10::new(16, 16, 16, 16);

        predict_inter_8x16(
            &reference,
            &mut target,
            0,
            0,
            1,
            MotionVectorQuarterPel::ZERO,
        )
        .unwrap();

        assert_eq!(target.y.get(7, 0), Some(0));
        assert_eq!(target.y.get(8, 0), Some(108));
        assert_eq!(target.cb.get(3, 0), Some(0));
        assert_eq!(target.cb.get(4, 0), Some(204));
    }

    #[test]
    fn bi_inter_prediction_averages_two_reference_frames() {
        let mut list0 = DecodedFrame422P10::new(16, 16, 16, 16);
        let mut list1 = DecodedFrame422P10::new(16, 16, 16, 16);
        for y in 0..16 {
            for x in 0..16 {
                assert!(list0.y.set(x, y, 100));
                assert!(list1.y.set(x, y, 300));
            }
            for x in 0..8 {
                assert!(list0.cb.set(x, y, 200));
                assert!(list1.cb.set(x, y, 400));
                assert!(list0.cr.set(x, y, 500));
                assert!(list1.cr.set(x, y, 700));
            }
        }
        let mut target = DecodedFrame422P10::new(16, 16, 16, 16);

        predict_bi_inter_16x16(
            &list0,
            &list1,
            &mut target,
            0,
            0,
            MotionVectorQuarterPel::ZERO,
            MotionVectorQuarterPel::ZERO,
            32,
            32,
        )
        .unwrap();

        assert_eq!(target.y.get(0, 0), Some(200));
        assert_eq!(target.cb.get(0, 0), Some(300));
        assert_eq!(target.cr.get(0, 0), Some(600));
    }

    #[test]
    fn motion_field_predicts_first_macroblock_as_zero() {
        let grid = MacroblockGrid::new(32, 16).unwrap();
        let field = MotionField::new(grid).unwrap();
        let address = grid.address(0).unwrap();

        assert_eq!(
            field.predict_l0_16x16(address, 0).unwrap(),
            MotionVectorQuarterPel::ZERO
        );
    }

    #[test]
    fn motion_field_predicts_from_available_left_neighbor() {
        let grid = MacroblockGrid::new(32, 16).unwrap();
        let mut field = MotionField::new(grid).unwrap();
        field
            .set_inter(
                grid.address(0).unwrap(),
                MotionVectorQuarterPel { x: 8, y: -4 },
                0,
            )
            .unwrap();

        assert_eq!(
            field.predict_l0_16x16(grid.address(1).unwrap(), 0).unwrap(),
            MotionVectorQuarterPel { x: 8, y: -4 }
        );
    }

    #[test]
    fn motion_field_predicts_median_of_three_neighbors() {
        let grid = MacroblockGrid::new(48, 32).unwrap();
        let mut field = MotionField::new(grid).unwrap();
        field
            .set_inter(
                grid.address(3).unwrap(),
                MotionVectorQuarterPel { x: 2, y: 40 },
                0,
            )
            .unwrap();
        field
            .set_inter(
                grid.address(1).unwrap(),
                MotionVectorQuarterPel { x: 10, y: 4 },
                0,
            )
            .unwrap();
        field
            .set_inter(
                grid.address(2).unwrap(),
                MotionVectorQuarterPel { x: 6, y: 8 },
                0,
            )
            .unwrap();

        assert_eq!(
            field.predict_l0_16x16(grid.address(4).unwrap(), 0).unwrap(),
            MotionVectorQuarterPel { x: 6, y: 8 }
        );
    }

    #[test]
    fn intra_above_right_stays_in_the_median_as_zero() {
        let grid = MacroblockGrid::new(64, 32).unwrap();
        let mut field = MotionField::new(grid).unwrap();
        let above = grid.address(2).unwrap();
        field
            .set_l0_blocks(
                above,
                &[10, 11, 14, 15],
                MotionVectorQuarterPel { x: 18, y: -32 },
                0,
            )
            .unwrap();
        field.set_intra(grid.address(3).unwrap()).unwrap();
        let current = grid.address(6).unwrap();
        field
            .set_l0_blocks(
                current,
                &[0, 1, 2, 3, 8, 9, 10, 11],
                MotionVectorQuarterPel { x: -16, y: 0 },
                0,
            )
            .unwrap();

        assert_eq!(
            field
                .predict_partition(current, 2, 0, 2, 4, 0, false)
                .unwrap(),
            MotionVectorQuarterPel::ZERO
        );
    }

    #[test]
    fn p_skip_uses_zero_when_left_neighbor_motion_is_zero() {
        let grid = MacroblockGrid::new(48, 32).unwrap();
        let mut field = MotionField::new(grid).unwrap();
        field
            .set_inter(grid.address(3).unwrap(), MotionVectorQuarterPel::ZERO, 0)
            .unwrap();
        field
            .set_inter(
                grid.address(1).unwrap(),
                MotionVectorQuarterPel { x: 12, y: 8 },
                0,
            )
            .unwrap();
        field
            .set_inter(
                grid.address(2).unwrap(),
                MotionVectorQuarterPel { x: 20, y: 4 },
                0,
            )
            .unwrap();

        assert_eq!(
            field.predict_p_skip(grid.address(4).unwrap()).unwrap(),
            MotionVectorQuarterPel::ZERO
        );
    }

    #[test]
    fn quarter_sample_between_half_pels_averages_one_diagonal() {
        let mut reference = DecodedFrame422P10::new(32, 32, 32, 32);
        assert!(reference.y.set(10, 10, 1000));

        assert_eq!(
            luma_quarter_sample(&reference, 10 * 4 + 1, 10 * 4 + 1).unwrap(),
            625
        );
        assert_eq!(
            luma_quarter_sample(&reference, 10 * 4 + 3, 10 * 4 + 1).unwrap(),
            313
        );
    }

    #[test]
    fn diagonal_half_pel_keeps_a_flat_luma_field() {
        let mut reference = DecodedFrame422P10::new(16, 16, 16, 16);
        for y in 0..16 {
            for x in 0..16 {
                assert!(reference.y.set(x, y, 512));
            }
            for x in 0..8 {
                assert!(reference.cb.set(x, y, 256));
                assert!(reference.cr.set(x, y, 256));
            }
        }
        let mut target = DecodedFrame422P10::new(16, 16, 16, 16);
        predict_inter_16x16(
            &reference,
            &mut target,
            0,
            0,
            MotionVectorQuarterPel { x: 2, y: 2 },
        )
        .unwrap();

        assert_eq!(target.y.get(8, 8), Some(512));
    }

    #[test]
    fn same_macroblock_unused_list_stays_available_for_later_prediction() {
        let grid = MacroblockGrid::new(32, 16).unwrap();
        let mut field = MotionField::new(grid).unwrap();
        let left = grid.address(0).unwrap();
        let current = grid.address(1).unwrap();
        field
            .set_l0_blocks(left, &[5], MotionVectorQuarterPel { x: -10, y: -8 }, 0)
            .unwrap();
        field
            .set_l0_blocks(left, &[7], MotionVectorQuarterPel { x: -4, y: 0 }, 0)
            .unwrap();
        field
            .set_l0_blocks(current, &[2], MotionVectorQuarterPel::ZERO, 0)
            .unwrap();
        field
            .mark_list_unused(current, &[4, 5, 6, 7], false)
            .unwrap();

        let predicted = field
            .predict_partition(current, 0, 2, 2, 2, 1, false)
            .unwrap();

        assert_eq!(predicted, MotionVectorQuarterPel { x: 0, y: 0 });
    }

    #[test]
    fn block_interpolation_matches_per_sample_filter() {
        let mut reference = DecodedFrame422P10::new(48, 40, 48, 40);
        for y in 0..40 {
            for x in 0..48 {
                assert!(reference.y.set(x, y, ((x * 17 + y * 29) % 1024) as u16));
            }
            for x in 0..24 {
                assert!(reference.cb.set(x, y, ((x * 13 + y * 7) % 1024) as u16));
                assert!(reference
                    .cr
                    .set(x, y, ((x * 3 + y * 11 + 40) % 1024) as u16));
            }
        }

        let mut motions = Vec::new();
        for fy in 0..4 {
            for fx in 0..4 {
                motions.push(MotionVectorQuarterPel {
                    x: 8 + fx,
                    y: 4 + fy,
                });
                motions.push(MotionVectorQuarterPel {
                    x: -9 + fx,
                    y: -5 + fy,
                });
            }
        }
        motions.push(MotionVectorQuarterPel { x: 13, y: -3 });
        motions.push(MotionVectorQuarterPel { x: -2, y: 11 });

        for motion in motions {
            for (dst_x, dst_y, width, height) in [
                (4_usize, 8_usize, 16_usize, 8_usize),
                (0, 0, 8, 4),
                (32, 24, 16, 16),
            ] {
                let mut target = DecodedFrame422P10::new(48, 40, 48, 40);
                predict_inter_region(&reference, &mut target, dst_x, dst_y, width, height, motion)
                    .unwrap();
                for row in 0..height {
                    for column in 0..width {
                        let expected = luma_quarter_sample(
                            &reference,
                            (dst_x + column) as i32 * 4 + motion.x,
                            (dst_y + row) as i32 * 4 + motion.y,
                        )
                        .unwrap();
                        assert_eq!(
                            target.y.get(dst_x + column, dst_y + row),
                            Some(expected),
                            "luma {motion:?} at {},{}",
                            dst_x + column,
                            dst_y + row
                        );
                    }
                }
                for row in 0..height {
                    for column in 0..width / 2 {
                        let chroma_x = dst_x / 2 + column;
                        let sample_x = chroma_x as i32 * 8 + motion.x;
                        let sample_y = ((dst_y + row) as i32 * 4 + motion.y) * 2;
                        assert_eq!(
                            target.cb.get(chroma_x, dst_y + row),
                            Some(chroma_eighth_sample(&reference.cb, sample_x, sample_y).unwrap()),
                            "cb {motion:?} at {chroma_x},{}",
                            dst_y + row
                        );
                        assert_eq!(
                            target.cr.get(chroma_x, dst_y + row),
                            Some(chroma_eighth_sample(&reference.cr, sample_x, sample_y).unwrap()),
                            "cr {motion:?} at {chroma_x},{}",
                            dst_y + row
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn local_luma_prediction_matches_frame_prediction() {
        let mut reference = DecodedFrame422P10::new(48, 48, 48, 48);
        for y in 0..48 {
            for x in 0..48 {
                assert!(reference.y.set(x, y, ((x * 17 + y * 29) % 1024) as u16));
            }
        }
        for motion in [
            MotionVectorQuarterPel::ZERO,
            MotionVectorQuarterPel { x: 2, y: 0 },
            MotionVectorQuarterPel { x: 0, y: 2 },
            MotionVectorQuarterPel { x: 2, y: 2 },
            MotionVectorQuarterPel { x: 3, y: -1 },
        ] {
            let mut frame = DecodedFrame422P10::new(48, 48, 48, 48);
            predict_inter_16x16(&reference, &mut frame, 1, 1, motion).unwrap();
            let mut local = [0_u16; 16 * 16];
            predict_inter_luma_16x16_block(&reference, 1, 1, motion, &mut local).unwrap();
            for y in 0..16 {
                for x in 0..16 {
                    assert_eq!(local[y * 16 + x], frame.y.get(16 + x, 16 + y).unwrap());
                }
            }
        }
    }

    #[test]
    fn local_bi_luma_prediction_matches_frame_prediction() {
        let mut list0 = DecodedFrame422P10::new(48, 48, 48, 48);
        let mut list1 = DecodedFrame422P10::new(48, 48, 48, 48);
        for y in 0..48 {
            for x in 0..48 {
                assert!(list0.y.set(x, y, ((x * 11 + y * 23) % 1024) as u16));
                assert!(list1.y.set(x, y, ((x * 31 + y * 7 + 91) % 1024) as u16));
            }
        }
        let cases = [
            (MotionVectorQuarterPel::ZERO, MotionVectorQuarterPel::ZERO),
            (
                MotionVectorQuarterPel { x: 2, y: 0 },
                MotionVectorQuarterPel { x: 0, y: 2 },
            ),
            (
                MotionVectorQuarterPel { x: 2, y: 2 },
                MotionVectorQuarterPel { x: 3, y: -1 },
            ),
        ];
        for (motion0, motion1) in cases {
            let mut frame = DecodedFrame422P10::new(48, 48, 48, 48);
            predict_bi_inter_16x16(&list0, &list1, &mut frame, 1, 1, motion0, motion1, 40, 24)
                .unwrap();
            let mut local = [0_u16; 16 * 16];
            predict_bi_inter_luma_16x16_block(
                &list0, &list1, 1, 1, motion0, motion1, 40, 24, &mut local,
            )
            .unwrap();
            for y in 0..16 {
                for x in 0..16 {
                    assert_eq!(local[y * 16 + x], frame.y.get(16 + x, 16 + y).unwrap());
                }
            }
        }
    }
}
