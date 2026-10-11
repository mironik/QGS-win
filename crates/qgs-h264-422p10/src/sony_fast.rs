//! Single-thread reconstruction commands for the dominant Sony XAVC paths.
//!
//! Commands contain only already-decoded motion and reference information.
//! CABAC and generic macroblock enums stay outside this module so hot
//! reconstruction kernels can evolve independently.

use crate::frame::DecodedFrame422P10;
use crate::motion::{
    predict_bi_inter_16x16, predict_inter_16x16, MotionCompensationError, MotionVectorQuarterPel,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SonyDirectMotion {
    L0(MotionVectorQuarterPel),
    L1(MotionVectorQuarterPel),
    Bi {
        l0: MotionVectorQuarterPel,
        l1: MotionVectorQuarterPel,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SonyDirect16x16Command {
    pub motion: SonyDirectMotion,
}

impl SonyDirect16x16Command {
    #[allow(clippy::too_many_arguments)]
    pub fn reconstruct(
        self,
        reference_l0: Option<&DecodedFrame422P10>,
        reference_l1: Option<&DecodedFrame422P10>,
        target: &mut DecodedFrame422P10,
        mb_x: usize,
        mb_y: usize,
        weight0: i32,
        weight1: i32,
    ) -> Result<(), MotionCompensationError> {
        match self.motion {
            SonyDirectMotion::L0(motion) => predict_inter_16x16(
                reference_l0.ok_or(MotionCompensationError::OutOfBounds)?,
                target,
                mb_x,
                mb_y,
                motion,
            ),
            SonyDirectMotion::L1(motion) => predict_inter_16x16(
                reference_l1.ok_or(MotionCompensationError::OutOfBounds)?,
                target,
                mb_x,
                mb_y,
                motion,
            ),
            SonyDirectMotion::Bi { l0, l1 } => predict_bi_inter_16x16(
                reference_l0.ok_or(MotionCompensationError::OutOfBounds)?,
                reference_l1.ok_or(MotionCompensationError::OutOfBounds)?,
                target,
                mb_x,
                mb_y,
                l0,
                l1,
                weight0,
                weight1,
            ),
        }
    }
}
