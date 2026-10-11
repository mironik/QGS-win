use crate::cabac::CabacDecoder;
use crate::cabac_nonzero::CabacNonZeroState422;
use crate::cabac_residual::{
    decode_residual_4x4_ac_category, decode_residual_4x4_category,
    decode_residual_4x4_category_dequantized, decode_residual_chroma422_dc_category,
    i_slice_residual_context_bank, pb_slice_residual_context_bank, CabacResidualCategory,
    CabacResidualCategoryContexts, CabacResidualError,
};
use crate::macroblock_type::{CodedBlockPatternChroma, MacroblockAddress, MacroblockGrid};
use crate::residual::{prepare_dequant_4x4, PreparedDequant4x4, ResidualBlock4x4};
use crate::transform::{
    inverse_chroma422_dc_transform_weighted, inverse_intra16x16_dc_transform_weighted,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Intra16x16LumaResidual422 {
    pub dc: ResidualBlock4x4,
    pub ac: [ResidualBlock4x4; 16],
}

impl Intra16x16LumaResidual422 {
    pub fn into_reconstruction_blocks(self, qp_y: u8, dc_weight: i32) -> [ResidualBlock4x4; 16] {
        let dc = inverse_intra16x16_dc_transform_weighted(self.dc.coeffs, qp_y, dc_weight);
        std::array::from_fn(|block_index| {
            let mut block = self.ac[block_index].clone();
            let block_x = (block_index & 1) + ((block_index >> 2) & 1) * 2;
            let block_y = ((block_index >> 1) & 1) + ((block_index >> 3) & 1) * 2;
            block.coeffs[0][0] = dc[block_y][block_x];
            block
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Chroma422Residual {
    pub dc: [ResidualBlock4x4; 2],
    pub ac: [[ResidualBlock4x4; 8]; 2],
}

impl Chroma422Residual {
    pub fn into_reconstruction_blocks(
        self,
        qp_cb: u8,
        qp_cr: u8,
        dc_cb: i32,
        dc_cr: i32,
    ) -> [[ResidualBlock4x4; 8]; 2] {
        let quantizers = [(qp_cb, dc_cb), (qp_cr, dc_cr)];
        std::array::from_fn(|plane| {
            let dc_levels = chroma422_dc_levels(&self.dc[plane]);
            let dc = inverse_chroma422_dc_transform_weighted(
                dc_levels,
                quantizers[plane].0,
                quantizers[plane].1,
            );
            std::array::from_fn(|block_index| {
                let mut block = self.ac[plane][block_index].clone();
                block.coeffs[0][0] = dc[block_index / 2][block_index % 2];
                block
            })
        })
    }
}

#[derive(Clone, Debug)]
pub struct CabacResidualDecoder422 {
    nonzero: CabacNonZeroState422,
    bank: [crate::cabac::CabacContext; 460],
    luma16_dc: CabacResidualCategoryContexts,
    luma16_ac: CabacResidualCategoryContexts,
    luma4x4: CabacResidualCategoryContexts,
    chroma_dc: CabacResidualCategoryContexts,
    chroma_ac: CabacResidualCategoryContexts,
    inter_luma_dequant: Option<(u8, PreparedDequant4x4)>,
}

impl CabacResidualDecoder422 {
    pub fn new_i_slice(grid: MacroblockGrid, qp_y: u8) -> Self {
        let bank = i_slice_residual_context_bank(qp_y);
        Self::with_bank(grid, bank, true)
    }

    pub fn new_pb_slice(grid: MacroblockGrid, qp_y: u8, cabac_init_idc: u8) -> Self {
        let bank = pb_slice_residual_context_bank(qp_y, cabac_init_idc);
        Self::with_bank(grid, bank, false)
    }

    fn with_bank(
        grid: MacroblockGrid,
        bank: [crate::cabac::CabacContext; 460],
        unavailable_nonzero: bool,
    ) -> Self {
        let nonzero = if unavailable_nonzero {
            CabacNonZeroState422::new_i_slice(grid)
        } else {
            CabacNonZeroState422::new(grid)
        };
        Self {
            nonzero,
            bank,
            luma16_dc: CabacResidualCategoryContexts::new(CabacResidualCategory::Luma16Dc),
            luma16_ac: CabacResidualCategoryContexts::new(CabacResidualCategory::Luma16Ac),
            luma4x4: CabacResidualCategoryContexts::new(CabacResidualCategory::Luma4x4),
            chroma_dc: CabacResidualCategoryContexts::new(CabacResidualCategory::Chroma422Dc),
            chroma_ac: CabacResidualCategoryContexts::new(CabacResidualCategory::Chroma422Ac),
            inter_luma_dequant: None,
        }
    }

    pub fn nonzero(&self) -> &CabacNonZeroState422 {
        &self.nonzero
    }

    pub fn set_slice_first_mb(&mut self, slice_first_mb: u32) {
        self.nonzero.set_slice_first_mb(slice_first_mb);
    }

    pub fn set_current_macroblock_intra(&mut self, intra: bool) {
        self.nonzero.set_unavailable_nonzero(intra);
    }

    pub fn decode_intra16x16_luma(
        &mut self,
        cabac: &mut CabacDecoder<'_>,
        address: MacroblockAddress,
        coded_block_pattern_luma: u8,
    ) -> Result<Intra16x16LumaResidual422, CabacResidualError> {
        let dc_context = self.nonzero.luma16_dc_context(address);
        let dc = decode_residual_4x4_category(cabac, &mut self.bank, &self.luma16_dc, dc_context)?;
        self.record_luma16_dc(address, usize::from(dc.non_zero_coefficients));

        let mut ac = zero_blocks_16();
        for block_index in 0..16 {
            if coded_block_pattern_luma & (1 << (block_index / 4)) == 0 {
                self.nonzero.set_luma4x4_count(address, block_index, 0);
                continue;
            }
            let context = self.nonzero.luma4x4_context(address, block_index);
            let residual =
                decode_residual_4x4_ac_category(cabac, &mut self.bank, &self.luma16_ac, context)?;
            self.record_luma4x4(
                address,
                block_index,
                usize::from(residual.non_zero_coefficients),
            );
            ac[block_index] = residual.block;
        }

        Ok(Intra16x16LumaResidual422 { dc: dc.block, ac })
    }

    pub fn decode_intra4x4_luma(
        &mut self,
        cabac: &mut CabacDecoder<'_>,
        address: MacroblockAddress,
        coded_block_pattern_luma: u8,
    ) -> Result<[ResidualBlock4x4; 16], CabacResidualError> {
        let mut blocks = zero_blocks_16();
        for block_index in 0..16 {
            if coded_block_pattern_luma & (1 << (block_index / 4)) == 0 {
                self.nonzero.set_luma4x4_count(address, block_index, 0);
                continue;
            }
            let context = self.nonzero.luma4x4_context(address, block_index);
            let residual =
                decode_residual_4x4_category(cabac, &mut self.bank, &self.luma4x4, context)?;
            self.record_luma4x4(
                address,
                block_index,
                usize::from(residual.non_zero_coefficients),
            );
            blocks[block_index] = residual.block;
        }
        Ok(blocks)
    }

    pub fn decode_inter4x4_luma(
        &mut self,
        cabac: &mut CabacDecoder<'_>,
        address: MacroblockAddress,
        coded_block_pattern_luma: u8,
    ) -> Result<[ResidualBlock4x4; 16], CabacResidualError> {
        self.decode_intra4x4_luma(cabac, address, coded_block_pattern_luma)
    }

    pub(crate) fn decode_inter4x4_luma_dequantized(
        &mut self,
        cabac: &mut CabacDecoder<'_>,
        address: MacroblockAddress,
        coded_block_pattern_luma: u8,
        qp_y: u8,
        weights: &[[i32; 4]; 4],
    ) -> Result<[ResidualBlock4x4; 16], CabacResidualError> {
        if self.inter_luma_dequant.map(|(qp, _)| qp) != Some(qp_y) {
            self.inter_luma_dequant = Some((qp_y, prepare_dequant_4x4(qp_y, weights)));
        }
        let prepared = self
            .inter_luma_dequant
            .expect("dequant cache initialized")
            .1;
        let mut blocks = zero_blocks_16();
        for block_index in 0..16 {
            if coded_block_pattern_luma & (1 << (block_index / 4)) == 0 {
                self.nonzero.set_luma4x4_count(address, block_index, 0);
                continue;
            }
            let context = self.nonzero.luma4x4_context(address, block_index);
            let residual = decode_residual_4x4_category_dequantized(
                cabac,
                &mut self.bank,
                &self.luma4x4,
                context,
                &prepared,
            )?;
            self.record_luma4x4(
                address,
                block_index,
                usize::from(residual.non_zero_coefficients),
            );
            blocks[block_index] = residual.block;
        }
        Ok(blocks)
    }

    pub fn decode_chroma_422(
        &mut self,
        cabac: &mut CabacDecoder<'_>,
        address: MacroblockAddress,
        coded_block_pattern_chroma: CodedBlockPatternChroma,
    ) -> Result<Chroma422Residual, CabacResidualError> {
        let mut dc = [ResidualBlock4x4::zero(), ResidualBlock4x4::zero()];
        let mut ac = std::array::from_fn(|_| zero_blocks_8());

        if coded_block_pattern_chroma != CodedBlockPatternChroma::Zero {
            for plane in 0..2 {
                let context = self.nonzero.chroma422_dc_context(address, plane);
                let residual = decode_residual_chroma422_dc_category(
                    cabac,
                    &mut self.bank,
                    &self.chroma_dc,
                    context,
                )?;
                self.record_chroma422_dc(
                    address,
                    plane,
                    usize::from(residual.non_zero_coefficients),
                );
                dc[plane] = residual.block;
            }
        } else {
            for plane in 0..2 {
                self.nonzero.set_chroma422_dc_count(address, plane, 0);
            }
        }

        if coded_block_pattern_chroma == CodedBlockPatternChroma::DcAndAc {
            for plane in 0..2 {
                for block_index in 0..8 {
                    let context = self
                        .nonzero
                        .chroma422_ac_context(address, plane, block_index);
                    let residual = decode_residual_4x4_ac_category(
                        cabac,
                        &mut self.bank,
                        &self.chroma_ac,
                        context,
                    )?;
                    self.record_chroma422_ac(
                        address,
                        plane,
                        block_index,
                        usize::from(residual.non_zero_coefficients),
                    );
                    ac[plane][block_index] = residual.block;
                }
            }
        } else {
            for plane in 0..2 {
                for block_index in 0..8 {
                    self.nonzero
                        .set_chroma422_ac_count(address, plane, block_index, 0);
                }
            }
        }

        Ok(Chroma422Residual { dc, ac })
    }

    pub fn record_luma16_dc(&mut self, address: MacroblockAddress, non_zero_coefficients: usize) {
        self.nonzero
            .set_luma16_dc_count(address, non_zero_coefficients);
    }

    pub fn record_luma4x4(
        &mut self,
        address: MacroblockAddress,
        block_index: usize,
        non_zero_coefficients: usize,
    ) {
        self.nonzero
            .set_luma4x4_count(address, block_index, non_zero_coefficients);
    }

    pub fn record_luma8x8_count(
        &mut self,
        address: MacroblockAddress,
        block_index: usize,
        count: usize,
    ) {
        const LUMA8X8_BLOCKS: [[usize; 4]; 4] =
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]];
        if let Some(blocks) = LUMA8X8_BLOCKS.get(block_index) {
            for &block in blocks {
                self.nonzero.set_luma4x4_count(address, block, count);
            }
        }
    }

    pub fn record_inter_absent(&mut self, address: MacroblockAddress) {
        self.record_inter_luma_absent(address);
        self.record_inter_chroma_absent(address);
    }

    pub fn record_inter_luma_absent(&mut self, address: MacroblockAddress) {
        self.nonzero.clear_luma4x4(address);
    }

    pub fn record_inter_chroma_absent(&mut self, address: MacroblockAddress) {
        for plane in 0..2 {
            self.nonzero.set_chroma422_dc_count(address, plane, 0);
        }
        self.nonzero.clear_chroma422_ac(address);
    }

    pub fn record_chroma422_dc(
        &mut self,
        address: MacroblockAddress,
        plane: usize,
        non_zero_coefficients: usize,
    ) {
        self.nonzero
            .set_chroma422_dc_count(address, plane, non_zero_coefficients);
    }

    pub fn record_chroma422_ac(
        &mut self,
        address: MacroblockAddress,
        plane: usize,
        block_index: usize,
        non_zero_coefficients: usize,
    ) {
        self.nonzero
            .set_chroma422_ac_count(address, plane, block_index, non_zero_coefficients);
    }

    pub fn record_pcm(&mut self, address: MacroblockAddress) {
        self.nonzero.mark_pcm(address);
    }
}

fn zero_blocks_16() -> [ResidualBlock4x4; 16] {
    std::array::from_fn(|_| ResidualBlock4x4::zero())
}

fn zero_blocks_8() -> [ResidualBlock4x4; 8] {
    std::array::from_fn(|_| ResidualBlock4x4::zero())
}

fn chroma422_dc_levels(block: &ResidualBlock4x4) -> [[i32; 2]; 4] {
    std::array::from_fn(|row| std::array::from_fn(|column| block.coeffs[row][column]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> MacroblockGrid {
        MacroblockGrid::new(32, 32).unwrap()
    }

    fn addr(address: u32) -> MacroblockAddress {
        grid().address(address).unwrap()
    }

    #[test]
    fn records_intra16x16_luma_dc_for_neighbor_context() {
        let mut decoder = CabacResidualDecoder422::new_i_slice(grid(), 24);
        decoder.record_luma16_dc(addr(0), 3);

        assert_eq!(decoder.nonzero().luma16_dc_context(addr(1)), 3);
        assert_eq!(decoder.nonzero().luma16_dc_context(addr(2)), 3);
        assert_eq!(decoder.nonzero().luma16_dc_context(addr(3)), 0);
    }

    #[test]
    fn records_luma4x4_counts_for_scan_neighbors() {
        let mut decoder = CabacResidualDecoder422::new_i_slice(grid(), 24);
        decoder.record_luma4x4(addr(3), 0, 2);
        decoder.record_luma4x4(addr(3), 1, 1);
        decoder.record_luma4x4(addr(3), 2, 1);

        assert_eq!(decoder.nonzero().luma4x4_context(addr(3), 1), 1);
        assert_eq!(decoder.nonzero().luma4x4_context(addr(3), 2), 2);
        assert_eq!(decoder.nonzero().luma4x4_context(addr(3), 3), 3);
    }

    #[test]
    fn records_chroma422_dc_and_ac_counts_separately() {
        let mut decoder = CabacResidualDecoder422::new_i_slice(grid(), 24);
        decoder.record_chroma422_dc(addr(0), 0, 4);
        decoder.record_chroma422_ac(addr(0), 1, 0, 2);

        assert_eq!(decoder.nonzero().chroma422_dc_context(addr(1), 0), 3);
        assert_eq!(decoder.nonzero().chroma422_dc_context(addr(1), 1), 2);
        assert_eq!(decoder.nonzero().chroma422_ac_context(addr(0), 1, 1), 3);
    }

    #[test]
    fn pcm_marks_all_residual_categories_nonzero() {
        let mut decoder = CabacResidualDecoder422::new_i_slice(grid(), 24);
        decoder.record_pcm(addr(0));

        assert_eq!(decoder.nonzero().luma16_dc_context(addr(1)), 3);
        assert_eq!(decoder.nonzero().luma4x4_context(addr(0), 1), 3);
        assert_eq!(decoder.nonzero().chroma422_dc_context(addr(1), 0), 3);
        assert_eq!(decoder.nonzero().chroma422_ac_context(addr(0), 0, 1), 3);
    }

    #[test]
    fn intra16x16_residual_combines_dc_transform_with_ac_blocks() {
        let mut dc = ResidualBlock4x4::zero();
        dc.coeffs[0][0] = 1;
        let mut ac = zero_blocks_16();
        ac[5].coeffs[1][2] = -3;
        let residual = Intra16x16LumaResidual422 { dc, ac };

        let blocks = residual.into_reconstruction_blocks(24, 16);

        assert!(blocks.iter().all(|block| block.coeffs[0][0] > 0));
        assert_eq!(blocks[5].coeffs[1][2], -3);
    }

    #[test]
    fn chroma422_residual_combines_dc_transform_with_ac_blocks() {
        let mut cb_dc = ResidualBlock4x4::zero();
        cb_dc.coeffs[0][0] = 1;
        let cr_dc = ResidualBlock4x4::zero();
        let mut ac = std::array::from_fn(|_| zero_blocks_8());
        ac[0][3].coeffs[2][1] = 7;
        let residual = Chroma422Residual {
            dc: [cb_dc, cr_dc],
            ac,
        };

        let blocks = residual.into_reconstruction_blocks(24, 24, 16, 16);

        assert!(blocks[0].iter().all(|block| block.coeffs[0][0] > 0));
        assert!(blocks[1].iter().all(|block| block.coeffs[0][0] == 0));
        assert_eq!(blocks[0][3].coeffs[2][1], 7);
    }
}
