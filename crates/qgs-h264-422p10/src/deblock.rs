use qgs_codec_h264::H264PictureId;

use crate::frame::{DecodedFrame422P10, Plane422P10};
use crate::macroblock_type::MacroblockAddress;
use crate::motion::{MotionField, MotionVectorQuarterPel};

const BIT_SHIFT: i32 = 2;

const ALPHA: [i32; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 17, 20,
    22, 25, 28, 32, 36, 40, 45, 50, 56, 63, 71, 80, 90, 101, 113, 127, 144, 162, 182, 203, 226,
    255, 255,
];

const BETA: [i32; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 6, 6, 7, 7, 8, 8,
    9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18,
];

const TC0: [[i32; 3]; 52] = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 1, 1],
    [0, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 2],
    [1, 1, 2],
    [1, 1, 2],
    [1, 1, 2],
    [1, 2, 3],
    [1, 2, 3],
    [2, 2, 3],
    [2, 2, 4],
    [2, 3, 4],
    [2, 3, 4],
    [3, 3, 5],
    [3, 4, 6],
    [3, 4, 6],
    [4, 5, 7],
    [4, 5, 8],
    [4, 6, 9],
    [5, 7, 10],
    [6, 8, 11],
    [6, 8, 13],
    [7, 10, 14],
    [8, 11, 16],
    [9, 12, 18],
    [10, 13, 20],
    [11, 15, 23],
    [13, 17, 25],
];

const ABOVE_29: [i16; 22] = [
    29, 30, 31, 32, 32, 33, 34, 34, 35, 35, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 39,
];

#[derive(Clone, Debug)]
struct DeblockSlice {
    disable_idc: u8,
    alpha_div2: i8,
    beta_div2: i8,
    chroma_offset_cb: i8,
    chroma_offset_cr: i8,
    list0: Vec<(u16, i32)>,
    list1: Vec<(u16, i32)>,
}

#[derive(Clone, Copy, Debug)]
struct DeblockMb {
    present: bool,
    qp_y: i16,
    intra: bool,
    transform_8x8: bool,
    luma_nnz: [u8; 16],
    slice: u16,
}

#[derive(Clone, Debug)]
pub(crate) struct DeblockGrid {
    width: u32,
    height: u32,
    mbs: Vec<DeblockMb>,
    slices: Vec<DeblockSlice>,
}

#[derive(Clone, Copy)]
struct Axis {
    x: i32,
    y: i32,
}

#[derive(Clone, Copy)]
struct Pred {
    picture: (u16, i32),
    vector: MotionVectorQuarterPel,
}

impl DeblockGrid {
    pub(crate) fn new(width_in_mbs: u32, height_in_mbs: u32) -> Self {
        let count = (width_in_mbs as usize).saturating_mul(height_in_mbs as usize);
        Self {
            width: width_in_mbs,
            height: height_in_mbs,
            mbs: vec![
                DeblockMb {
                    present: false,
                    qp_y: 0,
                    intra: false,
                    transform_8x8: false,
                    luma_nnz: [0; 16],
                    slice: 0,
                };
                count
            ],
            slices: Vec::new(),
        }
    }

    pub(crate) fn begin_slice(
        &mut self,
        disable_idc: u8,
        alpha_div2: i8,
        beta_div2: i8,
        chroma_offset_cb: i8,
        chroma_offset_cr: i8,
        list0: &[H264PictureId],
        list1: &[H264PictureId],
    ) -> u16 {
        let index = u16::try_from(self.slices.len()).unwrap_or(u16::MAX);
        self.slices.push(DeblockSlice {
            disable_idc,
            alpha_div2,
            beta_div2,
            chroma_offset_cb,
            chroma_offset_cr,
            list0: list0.iter().map(|id| (id.frame_num, id.poc)).collect(),
            list1: list1.iter().map(|id| (id.frame_num, id.poc)).collect(),
        });
        index
    }

    pub(crate) fn record(
        &mut self,
        slice_id: u16,
        address: MacroblockAddress,
        qp_y: i16,
        intra: bool,
        transform_8x8: bool,
        luma_nnz: [u8; 16],
    ) {
        let Some(slot) = self.mbs.get_mut(address.address as usize) else {
            return;
        };
        *slot = DeblockMb {
            present: true,
            qp_y,
            intra,
            transform_8x8,
            luma_nnz,
            slice: slice_id,
        };
    }

    pub(crate) fn apply(&self, frame: &mut DecodedFrame422P10, motion: &MotionField) {
        if self.slices.iter().all(|slice| slice.disable_idc == 1) {
            return;
        }
        for mb_y in 0..self.height {
            for mb_x in 0..self.width {
                self.filter_macroblock(frame, motion, mb_x, mb_y);
            }
        }
    }

    fn filter_macroblock(
        &self,
        frame: &mut DecodedFrame422P10,
        motion: &MotionField,
        mb_x: u32,
        mb_y: u32,
    ) {
        let Some(mb) = self.mb_at(mb_x as i32, mb_y as i32).copied() else {
            return;
        };
        if !mb.present {
            return;
        }
        if self
            .slices
            .get(mb.slice as usize)
            .is_some_and(|slice| slice.disable_idc == 1)
        {
            return;
        }
        for edge in 0..4 {
            let strengths = self.edge_strengths(motion, mb_x, mb_y, edge, true);
            let boundary = edge == 0;
            let neighbor = if boundary {
                self.mb_at(mb_x as i32 - 1, mb_y as i32).copied()
            } else {
                Some(mb)
            };
            if self.edge_allowed(&mb, neighbor.as_ref(), boundary) {
                let qp = edge_qp(
                    mb.qp_y,
                    neighbor.as_ref().map(|item| item.qp_y).unwrap_or(mb.qp_y),
                );
                if (edge != 1 && edge != 3) || !mb.transform_8x8 {
                    self.filter_luma_edge(
                        &mut frame.y,
                        (mb_x as i32) * 16 + edge * 4,
                        (mb_y as i32) * 16,
                        Axis { x: 1, y: 0 },
                        Axis { x: 0, y: 1 },
                        4,
                        strengths,
                        qp,
                        mb.slice,
                        false,
                    );
                }
                if edge % 2 == 0 {
                    let (qp_cb, qp_cr) = self.chroma_edge_qp(&mb, neighbor.as_ref());
                    self.filter_luma_edge(
                        &mut frame.cb,
                        (mb_x as i32) * 8 + (edge / 2) * 4,
                        (mb_y as i32) * 16,
                        Axis { x: 1, y: 0 },
                        Axis { x: 0, y: 1 },
                        4,
                        strengths,
                        qp_cb,
                        mb.slice,
                        true,
                    );
                    self.filter_luma_edge(
                        &mut frame.cr,
                        (mb_x as i32) * 8 + (edge / 2) * 4,
                        (mb_y as i32) * 16,
                        Axis { x: 1, y: 0 },
                        Axis { x: 0, y: 1 },
                        4,
                        strengths,
                        qp_cr,
                        mb.slice,
                        true,
                    );
                }
            }
        }
        for edge in 0..4 {
            let strengths = self.edge_strengths(motion, mb_x, mb_y, edge, false);
            let boundary = edge == 0;
            let neighbor = if boundary {
                self.mb_at(mb_x as i32, mb_y as i32 - 1).copied()
            } else {
                Some(mb)
            };
            if !self.edge_allowed(&mb, neighbor.as_ref(), boundary) {
                continue;
            }
            let qp = edge_qp(
                mb.qp_y,
                neighbor.as_ref().map(|item| item.qp_y).unwrap_or(mb.qp_y),
            );
            if (edge != 1 && edge != 3) || !mb.transform_8x8 {
                self.filter_luma_edge(
                    &mut frame.y,
                    (mb_x as i32) * 16,
                    (mb_y as i32) * 16 + edge * 4,
                    Axis { x: 0, y: 1 },
                    Axis { x: 1, y: 0 },
                    4,
                    strengths,
                    qp,
                    mb.slice,
                    false,
                );
            }
            let (qp_cb, qp_cr) = self.chroma_edge_qp(&mb, neighbor.as_ref());
            self.filter_luma_edge(
                &mut frame.cb,
                (mb_x as i32) * 8,
                (mb_y as i32) * 16 + edge * 4,
                Axis { x: 0, y: 1 },
                Axis { x: 1, y: 0 },
                2,
                strengths,
                qp_cb,
                mb.slice,
                true,
            );
            self.filter_luma_edge(
                &mut frame.cr,
                (mb_x as i32) * 8,
                (mb_y as i32) * 16 + edge * 4,
                Axis { x: 0, y: 1 },
                Axis { x: 1, y: 0 },
                2,
                strengths,
                qp_cr,
                mb.slice,
                true,
            );
        }
    }

    fn filter_luma_edge(
        &self,
        plane: &mut Plane422P10,
        q_x: i32,
        q_y: i32,
        across: Axis,
        along: Axis,
        group_len: usize,
        strengths: [u8; 4],
        qp: i16,
        slice_id: u16,
        chroma: bool,
    ) {
        if strengths == [0, 0, 0, 0] {
            return;
        }
        let Some(slice) = self.slices.get(slice_id as usize) else {
            return;
        };
        let index_a = threshold_index(qp, slice.alpha_div2);
        let index_b = threshold_index(qp, slice.beta_div2);
        let alpha = ALPHA[index_a as usize] << BIT_SHIFT;
        let beta = BETA[index_b as usize] << BIT_SHIFT;
        if alpha == 0 || beta == 0 {
            return;
        }
        let strong = strengths[0] == 4;
        for group in 0..4 {
            let strength = strengths[group];
            if !strong && strength == 0 {
                continue;
            }
            for offset in 0..group_len {
                let step = (group * group_len + offset) as i32;
                filter_sample(
                    plane,
                    q_x + along.x * step,
                    q_y + along.y * step,
                    across,
                    alpha,
                    beta,
                    index_a,
                    strength,
                    strong,
                    chroma,
                );
            }
        }
    }

    fn edge_strengths(
        &self,
        motion: &MotionField,
        mb_x: u32,
        mb_y: u32,
        edge: i32,
        vertical: bool,
    ) -> [u8; 4] {
        let mut strengths = [0; 4];
        for group in 0..4 {
            let (q_bx, q_by, p_bx, p_by) = if vertical {
                let q_bx = mb_x as i32 * 4 + edge;
                let q_by = mb_y as i32 * 4 + group;
                (q_bx, q_by, q_bx - 1, q_by)
            } else {
                let q_bx = mb_x as i32 * 4 + group;
                let q_by = mb_y as i32 * 4 + edge;
                (q_bx, q_by, q_bx, q_by - 1)
            };
            strengths[group as usize] =
                self.pair_strength(motion, p_bx, p_by, q_bx, q_by, edge == 0);
        }
        strengths
    }

    fn pair_strength(
        &self,
        motion: &MotionField,
        p_bx: i32,
        p_by: i32,
        q_bx: i32,
        q_by: i32,
        macroblock_edge: bool,
    ) -> u8 {
        let (Some((p_index, p_x, p_y)), Some((q_index, q_x, q_y))) =
            (self.block_local(p_bx, p_by), self.block_local(q_bx, q_by))
        else {
            return 0;
        };
        let (Some(p), Some(q)) = (self.mbs.get(p_index), self.mbs.get(q_index)) else {
            return 0;
        };
        if p.intra || q.intra {
            return if macroblock_edge { 4 } else { 3 };
        }
        let p_nnz = p.luma_nnz[scan4x4(p_x, p_y)];
        let q_nnz = q.luma_nnz[scan4x4(q_x, q_y)];
        if p_nnz > 0 || q_nnz > 0 {
            return 2;
        }
        let p0 = self.prediction(motion, p_index, p_x, p_y, p.slice, false);
        let p1 = self.prediction(motion, p_index, p_x, p_y, p.slice, true);
        let q0 = self.prediction(motion, q_index, q_x, q_y, q.slice, false);
        let q1 = self.prediction(motion, q_index, q_x, q_y, q.slice, true);
        if same_motion(p0, q0) && same_motion(p1, q1) || same_motion(p0, q1) && same_motion(p1, q0)
        {
            0
        } else {
            1
        }
    }

    fn prediction(
        &self,
        motion: &MotionField,
        mb_index: usize,
        local_x: usize,
        local_y: usize,
        slice_id: u16,
        list1: bool,
    ) -> Option<Pred> {
        let slice = self.slices.get(slice_id as usize)?;
        let mb_x = (mb_index as u32) % self.width;
        let mb_y = (mb_index as u32) / self.width;
        let sample = motion.block_motion(
            MacroblockAddress {
                address: mb_index as u32,
                x: mb_x,
                y: mb_y,
            },
            scan4x4(local_x, local_y),
            list1,
        )?;
        let list = if list1 { &slice.list1 } else { &slice.list0 };
        Some(Pred {
            picture: *list.get(usize::from(sample.ref_index))?,
            vector: sample.vector,
        })
    }

    fn chroma_edge_qp(&self, current: &DeblockMb, neighbor: Option<&DeblockMb>) -> (i16, i16) {
        let Some(slice) = self.slices.get(current.slice as usize) else {
            return (current.qp_y, current.qp_y);
        };
        let neighbor = neighbor.unwrap_or(current);
        let neighbor_slice = self.slices.get(neighbor.slice as usize).unwrap_or(slice);
        let cb = edge_qp(
            chroma_qp(current.qp_y, slice.chroma_offset_cb),
            chroma_qp(neighbor.qp_y, neighbor_slice.chroma_offset_cb),
        );
        let cr = edge_qp(
            chroma_qp(current.qp_y, slice.chroma_offset_cr),
            chroma_qp(neighbor.qp_y, neighbor_slice.chroma_offset_cr),
        );
        (cb, cr)
    }

    fn edge_allowed(
        &self,
        current: &DeblockMb,
        neighbor: Option<&DeblockMb>,
        boundary: bool,
    ) -> bool {
        let Some(slice) = self.slices.get(current.slice as usize) else {
            return false;
        };
        if slice.disable_idc == 1 {
            return false;
        }
        if !boundary {
            return true;
        }
        let Some(neighbor) = neighbor else {
            return false;
        };
        if !neighbor.present {
            return false;
        }
        slice.disable_idc != 2 || neighbor.slice == current.slice
    }

    fn mb_at(&self, mb_x: i32, mb_y: i32) -> Option<&DeblockMb> {
        if mb_x < 0 || mb_y < 0 || mb_x >= self.width as i32 || mb_y >= self.height as i32 {
            return None;
        }
        self.mbs
            .get((mb_y as u32 * self.width + mb_x as u32) as usize)
    }

    fn block_local(&self, block_x: i32, block_y: i32) -> Option<(usize, usize, usize)> {
        if block_x < 0 || block_y < 0 {
            return None;
        }
        let mb_x = block_x / 4;
        let mb_y = block_y / 4;
        if mb_x >= self.width as i32 || mb_y >= self.height as i32 {
            return None;
        }
        let index = (mb_y as u32 * self.width + mb_x as u32) as usize;
        Some((index, (block_x % 4) as usize, (block_y % 4) as usize))
    }
}

fn edge_qp(current: i16, neighbor: i16) -> i16 {
    (current + neighbor + 1) >> 1
}

fn chroma_qp(qp_y: i16, offset: i8) -> i16 {
    let qpi = (qp_y + i16::from(offset)).clamp(-12, 51);
    if qpi < 30 {
        qpi
    } else {
        ABOVE_29[(qpi - 30) as usize]
    }
}

fn threshold_index(qp: i16, offset_div2: i8) -> i32 {
    (i32::from(qp) + i32::from(offset_div2) * 2).clamp(0, 51)
}

fn scan4x4(x: usize, y: usize) -> usize {
    ((y & 1) << 1) | (x & 1) | ((x & 2) << 1) | ((y & 2) << 2)
}

fn same_motion(left: Option<Pred>, right: Option<Pred>) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => {
            left.picture == right.picture
                && (left.vector.x - right.vector.x).abs() < 4
                && (left.vector.y - right.vector.y).abs() < 4
        }
        _ => false,
    }
}

fn filter_sample(
    plane: &mut Plane422P10,
    q_x: i32,
    q_y: i32,
    across: Axis,
    alpha: i32,
    beta: i32,
    index_a: i32,
    strength: u8,
    strong: bool,
    chroma: bool,
) {
    let Some(samples) = edge_samples(plane, q_x, q_y, across) else {
        return;
    };
    let [p3, p2, p1, p0, q0, q1, q2, q3] = samples;
    if (p0 - q0).abs() >= alpha || (p1 - p0).abs() >= beta || (q1 - q0).abs() >= beta {
        return;
    }
    if strong {
        if chroma {
            write_sample(plane, q_x, q_y, across, -1, (2 * p1 + p0 + q1 + 2) >> 2);
            write_sample(plane, q_x, q_y, across, 0, (2 * q1 + q0 + p1 + 2) >> 2);
            return;
        }
        if (p0 - q0).abs() < ((alpha >> 2) + 2) {
            if (p2 - p0).abs() < beta {
                write_sample(
                    plane,
                    q_x,
                    q_y,
                    across,
                    -1,
                    (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3,
                );
                write_sample(plane, q_x, q_y, across, -2, (p2 + p1 + p0 + q0 + 2) >> 2);
                write_sample(
                    plane,
                    q_x,
                    q_y,
                    across,
                    -3,
                    (2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3,
                );
            } else {
                write_sample(plane, q_x, q_y, across, -1, (2 * p1 + p0 + q1 + 2) >> 2);
            }
            if (q2 - q0).abs() < beta {
                write_sample(
                    plane,
                    q_x,
                    q_y,
                    across,
                    0,
                    (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3,
                );
                write_sample(plane, q_x, q_y, across, 1, (p0 + q0 + q1 + q2 + 2) >> 2);
                write_sample(
                    plane,
                    q_x,
                    q_y,
                    across,
                    2,
                    (2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) >> 3,
                );
            } else {
                write_sample(plane, q_x, q_y, across, 0, (2 * q1 + q0 + p1 + 2) >> 2);
            }
        } else {
            write_sample(plane, q_x, q_y, across, -1, (2 * p1 + p0 + q1 + 2) >> 2);
            write_sample(plane, q_x, q_y, across, 0, (2 * q1 + q0 + p1 + 2) >> 2);
        }
        return;
    }

    let tc_orig = if chroma {
        (TC0[index_a as usize][(strength - 1) as usize] << BIT_SHIFT) + 1
    } else {
        TC0[index_a as usize][(strength - 1) as usize] << BIT_SHIFT
    };
    if !chroma && tc_orig < 0 {
        return;
    }
    let mut tc = tc_orig;
    if !chroma && (p2 - p0).abs() < beta {
        if tc_orig != 0 {
            let updated = p1 + ((((p2 + ((p0 + q0 + 1) >> 1)) >> 1) - p1).clamp(-tc_orig, tc_orig));
            write_sample(plane, q_x, q_y, across, -2, updated);
        }
        tc += 1;
    }
    if !chroma && (q2 - q0).abs() < beta {
        if tc_orig != 0 {
            let updated = q1 + ((((q2 + ((p0 + q0 + 1) >> 1)) >> 1) - q1).clamp(-tc_orig, tc_orig));
            write_sample(plane, q_x, q_y, across, 1, updated);
        }
        tc += 1;
    }
    let delta = ((((q0 - p0) * 4) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
    write_sample(plane, q_x, q_y, across, -1, p0 + delta);
    write_sample(plane, q_x, q_y, across, 0, q0 - delta);
}

fn edge_samples(plane: &Plane422P10, q_x: i32, q_y: i32, across: Axis) -> Option<[i32; 8]> {
    let mut samples = [0; 8];
    for (index, distance) in (-4..4).enumerate() {
        samples[index] = sample_at(plane, q_x + across.x * distance, q_y + across.y * distance)?;
    }
    Some(samples)
}

fn sample_at(plane: &Plane422P10, x: i32, y: i32) -> Option<i32> {
    if x < 0 || y < 0 {
        return None;
    }
    plane.get(x as usize, y as usize).map(i32::from)
}

fn write_sample(
    plane: &mut Plane422P10,
    q_x: i32,
    q_y: i32,
    across: Axis,
    distance: i32,
    value: i32,
) {
    let x = q_x + across.x * distance;
    let y = q_y + across.y * distance;
    if x >= 0 && y >= 0 {
        plane.set(x as usize, y as usize, value.clamp(0, 1023) as u16);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::macroblock_type::MacroblockGrid;
    use crate::motion::MotionCompensationError;

    fn grid_with_slice(width: u32, height: u32) -> DeblockGrid {
        let mut grid = DeblockGrid::new(width, height);
        grid.begin_slice(0, 0, 0, 0, 0, &[], &[]);
        grid
    }

    fn address(x: u32, y: u32, width: u32) -> MacroblockAddress {
        MacroblockAddress {
            address: y * width + x,
            x,
            y,
        }
    }

    fn fill(plane: &mut Plane422P10, value: u16) {
        for y in 0..plane.height {
            for x in 0..plane.width {
                plane.set(x, y, value);
            }
        }
    }

    fn motion(width: u32, height: u32) -> Result<MotionField, MotionCompensationError> {
        MotionField::new(MacroblockGrid::new(width * 16, height * 16).expect("grid"))
    }

    #[test]
    fn intra_macroblock_edge_pulls_luma_and_chroma_together() {
        let mut grid = grid_with_slice(2, 1);
        for x in 0..2 {
            grid.record(0, address(x, 0, 2), 30, true, false, [0; 16]);
        }
        let mut frame = DecodedFrame422P10::new(32, 16, 32, 16);
        fill(&mut frame.y, 100);
        fill(&mut frame.cb, 100);
        fill(&mut frame.cr, 100);
        for y in 0..16 {
            for x in 16..32 {
                frame.y.set(x, y, 140);
            }
            for x in 8..16 {
                frame.cb.set(x, y, 140);
                frame.cr.set(x, y, 140);
            }
        }
        grid.apply(&mut frame, &motion(2, 1).expect("motion"));

        assert_eq!(frame.y.get(15, 8), Some(110));
        assert_eq!(frame.y.get(16, 8), Some(130));
        assert_eq!(frame.y.get(14, 8), Some(100));
        assert_eq!(frame.cb.get(7, 8), Some(110));
        assert_eq!(frame.cb.get(8, 8), Some(130));
        assert_eq!(frame.cb.get(6, 8), Some(100));
    }

    #[test]
    fn transform_8x8_keeps_the_internal_four_sample_luma_edge() {
        let mut grid = grid_with_slice(1, 1);
        grid.record(0, address(0, 0, 1), 30, true, true, [0; 16]);
        let mut frame = DecodedFrame422P10::new(16, 16, 16, 16);
        fill(&mut frame.y, 140);
        fill(&mut frame.cb, 140);
        fill(&mut frame.cr, 140);
        for y in 0..16 {
            for x in 0..4 {
                frame.y.set(x, y, 100);
            }
        }
        grid.apply(&mut frame, &motion(1, 1).expect("motion"));

        assert_eq!(frame.y.get(3, 4), Some(100));
        assert_eq!(frame.y.get(4, 4), Some(140));
    }

    #[test]
    fn coded_inter_edge_uses_the_weak_filter() {
        let mut grid = grid_with_slice(2, 1);
        for x in 0..2 {
            grid.record(0, address(x, 0, 2), 30, false, false, [1; 16]);
        }
        let mut frame = DecodedFrame422P10::new(32, 16, 32, 16);
        fill(&mut frame.y, 100);
        for y in 0..16 {
            for x in 16..32 {
                frame.y.set(x, y, 140);
            }
        }
        grid.apply(&mut frame, &motion(2, 1).expect("motion"));

        assert_eq!(frame.y.get(14, 0), Some(104));
        assert_eq!(frame.y.get(15, 0), Some(106));
        assert_eq!(frame.y.get(16, 0), Some(134));
        assert_eq!(frame.y.get(17, 0), Some(136));
    }

    #[test]
    fn disabled_filter_leaves_the_edge() {
        let mut grid = DeblockGrid::new(2, 1);
        grid.begin_slice(1, 0, 0, 0, 0, &[], &[]);
        for x in 0..2 {
            grid.record(0, address(x, 0, 2), 30, true, false, [0; 16]);
        }
        let mut frame = DecodedFrame422P10::new(32, 16, 32, 16);
        fill(&mut frame.y, 100);
        for y in 0..16 {
            for x in 16..32 {
                frame.y.set(x, y, 140);
            }
        }
        grid.apply(&mut frame, &motion(2, 1).expect("motion"));

        assert_eq!(frame.y.get(15, 0), Some(100));
        assert_eq!(frame.y.get(16, 0), Some(140));
    }
}
