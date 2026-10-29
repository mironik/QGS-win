//! Compact single-thread reconstruction plan for dominant Sony inter MBs.
//!
//! CABAC emits commands and coefficient blocks into dense arenas; a later
//! execute pass runs homogeneous fused kernels. Intra macroblocks remain a
//! barrier that flushes the plan first.

use crate::frame::DecodedFrame422P10;
use crate::macroblock::{
    reconstruct_inter_chroma_422_local, reconstruct_inter_luma_16x16_local,
    reconstruct_inter_luma_16x16_local_predequantized, reconstruct_inter_luma_8x8_local,
    store_local_inter_prediction, write_local_chroma_8x16, write_local_luma_16x16, ChromaPlane,
};
use crate::macroblock_type::MacroblockAddress;
use crate::motion::{
    predict_bi_inter_16x16, predict_bi_inter_chroma_16x16_blocks,
    predict_bi_inter_luma_16x16_block, predict_inter_16x16, predict_inter_chroma_16x16_blocks,
    predict_inter_luma_16x16_block, MotionVectorQuarterPel,
};
use crate::residual::{ResidualBlock4x4, ResidualBlock8x8};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PlanLumaResidual {
    Absent,
    Blocks4x4(u32),
    Blocks4x4Dequantized(u32),
    Blocks8x8(u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PlanKernel {
    L0 {
        motion: MotionVectorQuarterPel,
        reference: u8,
    },
    L1 {
        motion: MotionVectorQuarterPel,
        reference: u8,
    },
    Bi {
        l0: MotionVectorQuarterPel,
        l1: MotionVectorQuarterPel,
        reference_l0: u8,
        reference_l1: u8,
        weight0: i32,
        weight1: i32,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PlanCommand {
    pub address: MacroblockAddress,
    pub kernel: PlanKernel,
    pub luma: PlanLumaResidual,
    pub chroma: Option<u32>,
    pub qp_y: u8,
    pub qp_cb: u8,
    pub qp_cr: u8,
}

#[derive(Debug, Default)]
pub(crate) struct SonySlicePlan {
    pub commands: Vec<PlanCommand>,
    pub luma4x4: Vec<[ResidualBlock4x4; 16]>,
    pub luma8x8: Vec<[ResidualBlock8x8; 4]>,
    pub chroma: Vec<([ResidualBlock4x4; 8], [ResidualBlock4x4; 8])>,
}

impl SonySlicePlan {
    pub fn with_capacity(macroblocks: usize) -> Self {
        Self {
            commands: Vec::with_capacity(macroblocks),
            luma4x4: Vec::with_capacity(macroblocks / 4 + 1),
            luma8x8: Vec::with_capacity(macroblocks / 8 + 1),
            chroma: Vec::with_capacity(macroblocks / 4 + 1),
        }
    }

    pub fn clear(&mut self) {
        self.commands.clear();
        self.luma4x4.clear();
        self.luma8x8.clear();
        self.chroma.clear();
    }

    pub fn push_luma4x4(&mut self, blocks: [ResidualBlock4x4; 16]) -> u32 {
        let index = self.luma4x4.len() as u32;
        self.luma4x4.push(blocks);
        index
    }

    pub fn push_luma4x4_dequantized(&mut self, blocks: [ResidualBlock4x4; 16]) -> u32 {
        self.push_luma4x4(blocks)
    }

    pub fn push_luma8x8(&mut self, blocks: [ResidualBlock8x8; 4]) -> u32 {
        let index = self.luma8x8.len() as u32;
        self.luma8x8.push(blocks);
        index
    }

    pub fn push_chroma(&mut self, blocks: ([ResidualBlock4x4; 8], [ResidualBlock4x4; 8])) -> u32 {
        let index = self.chroma.len() as u32;
        self.chroma.push(blocks);
        index
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }
}

#[derive(Debug)]
pub(crate) enum PlanExecuteError {
    OutOfBounds,
    MissingReference,
    CorruptPlan,
    MotionVectorOverflow,
}

impl From<crate::motion::MotionCompensationError> for PlanExecuteError {
    fn from(error: crate::motion::MotionCompensationError) -> Self {
        match error {
            crate::motion::MotionCompensationError::OutOfBounds => Self::OutOfBounds,
            crate::motion::MotionCompensationError::MotionVectorOverflow => {
                Self::MotionVectorOverflow
            }
        }
    }
}

#[inline(always)]
fn kernel_class(kernel: PlanKernel) -> u8 {
    match kernel {
        PlanKernel::L0 { .. } => 0,
        PlanKernel::L1 { .. } => 1,
        PlanKernel::Bi { .. } => 2,
    }
}

#[inline(always)]
fn has_residual(command: &PlanCommand) -> bool {
    !(matches!(command.luma, PlanLumaResidual::Absent) && command.chroma.is_none())
}

struct LocalPred {
    luma: [u16; 16 * 16],
    cb: [u16; 8 * 16],
    cr: [u16; 8 * 16],
}

impl LocalPred {
    fn zeroed() -> Self {
        Self {
            luma: [0; 16 * 16],
            cb: [0; 8 * 16],
            cr: [0; 8 * 16],
        }
    }
}

/// Executes the plan with homogeneous kernel batches on one core.
///
/// Commands are sorted in-place by (kernel class, residual flag) so L0/L1/Bi
/// absent runs form long monomorphic loops, then coded runs software-pipeline
/// predict(N+1) with residual(N) on double buffers.
pub(crate) fn execute_sony_slice_plan(
    plan: &mut SonySlicePlan,
    list0: &[&DecodedFrame422P10],
    list1: &[&DecodedFrame422P10],
    target: &mut DecodedFrame422P10,
) -> Result<(), PlanExecuteError> {
    if plan.commands.is_empty() {
        return Ok(());
    }

    plan.commands.sort_unstable_by_key(|command| {
        (
            kernel_class(command.kernel),
            u8::from(has_residual(command)),
        )
    });

    let mut index = 0;
    let total = plan.commands.len();
    while index < total {
        let class = kernel_class(plan.commands[index].kernel);
        let residual = has_residual(&plan.commands[index]);
        let mut end = index + 1;
        while end < total
            && kernel_class(plan.commands[end].kernel) == class
            && has_residual(&plan.commands[end]) == residual
        {
            end += 1;
        }

        if !residual {
            for command in &plan.commands[index..end] {
                predict_into_frame(command.kernel, list0, list1, command.address, target)?;
            }
        } else {
            let mut buf_a = LocalPred::zeroed();
            let mut buf_b = LocalPred::zeroed();
            let first = plan.commands[index];
            predict_command(
                first.kernel,
                list0,
                list1,
                first.address,
                &mut buf_a.luma,
                &mut buf_a.cb,
                &mut buf_a.cr,
            )?;
            let mut use_a = true;
            let count = end - index;
            for pos in 0..count {
                let command = plan.commands[index + pos];
                if pos + 1 < count {
                    let next = plan.commands[index + pos + 1];
                    let next_buf = if use_a { &mut buf_b } else { &mut buf_a };
                    predict_command(
                        next.kernel,
                        list0,
                        list1,
                        next.address,
                        &mut next_buf.luma,
                        &mut next_buf.cb,
                        &mut next_buf.cr,
                    )?;
                }
                let cur = if use_a { &mut buf_a } else { &mut buf_b };
                apply_residual(
                    plan,
                    command,
                    target,
                    &mut cur.luma,
                    &mut cur.cb,
                    &mut cur.cr,
                )?;
                use_a = !use_a;
            }
        }
        index = end;
    }

    plan.clear();
    Ok(())
}

fn predict_into_frame(
    kernel: PlanKernel,
    list0: &[&DecodedFrame422P10],
    list1: &[&DecodedFrame422P10],
    address: MacroblockAddress,
    target: &mut DecodedFrame422P10,
) -> Result<(), PlanExecuteError> {
    let x = address.x as usize;
    let y = address.y as usize;
    match kernel {
        PlanKernel::L0 { motion, reference } => {
            predict_inter_16x16(resolve(list0, reference)?, target, x, y, motion)?;
        }
        PlanKernel::L1 { motion, reference } => {
            predict_inter_16x16(resolve(list1, reference)?, target, x, y, motion)?;
        }
        PlanKernel::Bi {
            l0,
            l1,
            reference_l0,
            reference_l1,
            weight0,
            weight1,
        } => {
            predict_bi_inter_16x16(
                resolve(list0, reference_l0)?,
                resolve(list1, reference_l1)?,
                target,
                x,
                y,
                l0,
                l1,
                weight0,
                weight1,
            )?;
        }
    }
    Ok(())
}

fn predict_command(
    kernel: PlanKernel,
    list0: &[&DecodedFrame422P10],
    list1: &[&DecodedFrame422P10],
    address: MacroblockAddress,
    luma: &mut [u16; 16 * 16],
    cb: &mut [u16; 8 * 16],
    cr: &mut [u16; 8 * 16],
) -> Result<(), PlanExecuteError> {
    let x = address.x as usize;
    let y = address.y as usize;
    match kernel {
        PlanKernel::L0 { motion, reference } => {
            let reference = resolve(list0, reference)?;
            predict_inter_luma_16x16_block(reference, x, y, motion, luma)?;
            predict_inter_chroma_16x16_blocks(reference, x, y, motion, cb, cr)?;
        }
        PlanKernel::L1 { motion, reference } => {
            let reference = resolve(list1, reference)?;
            predict_inter_luma_16x16_block(reference, x, y, motion, luma)?;
            predict_inter_chroma_16x16_blocks(reference, x, y, motion, cb, cr)?;
        }
        PlanKernel::Bi {
            l0,
            l1,
            reference_l0,
            reference_l1,
            weight0,
            weight1,
        } => {
            let reference0 = resolve(list0, reference_l0)?;
            let reference1 = resolve(list1, reference_l1)?;
            predict_bi_inter_luma_16x16_block(
                reference0, reference1, x, y, l0, l1, weight0, weight1, luma,
            )?;
            predict_bi_inter_chroma_16x16_blocks(
                reference0, reference1, x, y, l0, l1, weight0, weight1, cb, cr,
            )?;
        }
    }
    Ok(())
}

fn apply_residual(
    plan: &SonySlicePlan,
    command: PlanCommand,
    target: &mut DecodedFrame422P10,
    luma: &mut [u16; 16 * 16],
    cb: &mut [u16; 8 * 16],
    cr: &mut [u16; 8 * 16],
) -> Result<(), PlanExecuteError> {
    let x = command.address.x as usize;
    let y = command.address.y as usize;
    if matches!(command.luma, PlanLumaResidual::Absent) && command.chroma.is_none() {
        if !store_local_inter_prediction(target, x, y, luma, cb, cr) {
            return Err(PlanExecuteError::OutOfBounds);
        }
        return Ok(());
    }

    let luma_ok = match command.luma {
        PlanLumaResidual::Absent => write_local_luma_16x16(target, x, y, luma),
        PlanLumaResidual::Blocks4x4(index) => {
            let blocks = plan
                .luma4x4
                .get(index as usize)
                .ok_or(PlanExecuteError::CorruptPlan)?;
            reconstruct_inter_luma_16x16_local(target, x, y, command.qp_y, blocks, luma)
        }
        PlanLumaResidual::Blocks4x4Dequantized(index) => {
            let blocks = plan
                .luma4x4
                .get(index as usize)
                .ok_or(PlanExecuteError::CorruptPlan)?;
            reconstruct_inter_luma_16x16_local_predequantized(target, x, y, blocks, luma)
        }
        PlanLumaResidual::Blocks8x8(index) => {
            let blocks = plan
                .luma8x8
                .get(index as usize)
                .ok_or(PlanExecuteError::CorruptPlan)?;
            reconstruct_inter_luma_8x8_local(target, x, y, command.qp_y, blocks, luma)
        }
    };
    if !luma_ok {
        return Err(PlanExecuteError::OutOfBounds);
    }

    if let Some(index) = command.chroma {
        let (cb_residual, cr_residual) = plan
            .chroma
            .get(index as usize)
            .ok_or(PlanExecuteError::CorruptPlan)?;
        if !reconstruct_inter_chroma_422_local(
            target,
            ChromaPlane::Cb,
            x,
            y,
            command.qp_cb,
            cb_residual,
            cb,
        ) || !reconstruct_inter_chroma_422_local(
            target,
            ChromaPlane::Cr,
            x,
            y,
            command.qp_cr,
            cr_residual,
            cr,
        ) {
            return Err(PlanExecuteError::OutOfBounds);
        }
    } else if !write_local_chroma_8x16(target, ChromaPlane::Cb, x, y, cb)
        || !write_local_chroma_8x16(target, ChromaPlane::Cr, x, y, cr)
    {
        return Err(PlanExecuteError::OutOfBounds);
    }
    Ok(())
}

fn resolve<'a>(
    list: &[&'a DecodedFrame422P10],
    index: u8,
) -> Result<&'a DecodedFrame422P10, PlanExecuteError> {
    list.get(usize::from(index))
        .copied()
        .ok_or(PlanExecuteError::MissingReference)
}
