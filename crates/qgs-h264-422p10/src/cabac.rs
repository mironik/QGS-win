use std::fmt;

#[derive(Clone, Debug)]
pub struct CabacDecoder<'a> {
    data: &'a [u8],
    next_byte: usize,
    bits_left: u8,
    bit_buffer: u64,
    range: u16,
    offset: u16,
    fault: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CabacContext {
    packed: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CabacInitValue {
    pub m: i8,
    pub n: i8,
}

impl CabacContext {
    pub fn new(state: u8, mps: bool) -> Self {
        Self {
            packed: state.min(63) | (u8::from(mps) << 6),
        }
    }
}

impl CabacInitValue {
    pub const fn new(m: i8, n: i8) -> Self {
        Self { m, n }
    }

    pub fn initialize(self, slice_qp_y: i16) -> CabacContext {
        let clipped_qp = slice_qp_y.clamp(0, 51);
        let pre_ctx_state =
            (((i16::from(self.m) * clipped_qp) >> 4) + i16::from(self.n)).clamp(1, 126);
        if pre_ctx_state <= 63 {
            CabacContext::new((63 - pre_ctx_state) as u8, false)
        } else {
            CabacContext::new((pre_ctx_state - 64) as u8, true)
        }
    }
}

impl<'a> CabacDecoder<'a> {
    pub fn new(data: &'a [u8]) -> Result<Self, CabacError> {
        let mut decoder = Self {
            data,
            next_byte: 0,
            bits_left: 0,
            bit_buffer: 0,
            range: 510,
            offset: 0,
            fault: false,
        };
        decoder.offset = decoder.read_bits(9)? as u16;
        Ok(decoder)
    }

    pub fn range(&self) -> u16 {
        self.range
    }

    pub(crate) fn fault(&self) -> bool {
        self.fault
    }

    pub fn offset(&self) -> u16 {
        self.offset
    }

    pub fn bit_position(&self) -> usize {
        self.next_byte
            .saturating_mul(8)
            .saturating_sub(usize::from(self.bits_left))
    }

    pub fn align_raw_byte(&mut self) -> Result<(), CabacError> {
        let discard = (8 - (self.bit_position() & 7)) & 7;
        for _ in 0..discard {
            self.read_bit()?;
        }
        Ok(())
    }

    pub fn read_raw_bits(&mut self, count: usize) -> Result<u32, CabacError> {
        self.read_bits(count)
    }

    pub fn reinitialize_from_current_position(&mut self) -> Result<(), CabacError> {
        self.range = 510;
        self.offset = self.read_bits(9)? as u16;
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn decode_bypass_bin(&mut self) -> bool {
        self.offset = (self.offset << 1) | self.read_renorm_bits(1);
        if self.offset >= self.range {
            self.offset -= self.range;
            true
        } else {
            false
        }
    }

    #[inline(always)]
    pub fn decode_bypass(&mut self) -> Result<bool, CabacError> {
        let bin = self.decode_bypass_bin();
        if self.fault {
            Err(CabacError::EndOfSlice)
        } else {
            Ok(bin)
        }
    }

    pub fn decode_bypass_ue(&mut self) -> Result<u32, CabacError> {
        self.decode_bypass_exp_golomb(0)
    }

    pub fn decode_bypass_exp_golomb(&mut self, order: u32) -> Result<u32, CabacError> {
        if order >= 31 {
            return Err(CabacError::SymbolValueOverflow);
        }
        let mut symbol = 0_u32;
        let mut suffix_bits = order;
        loop {
            if !self.decode_bypass()? {
                break;
            }
            let increment = 1_u32
                .checked_shl(suffix_bits)
                .ok_or(CabacError::SymbolValueOverflow)?;
            symbol = symbol
                .checked_add(increment)
                .ok_or(CabacError::SymbolValueOverflow)?;
            suffix_bits = suffix_bits
                .checked_add(1)
                .ok_or(CabacError::SymbolValueOverflow)?;
            if suffix_bits >= 32 {
                return Err(CabacError::SymbolValueOverflow);
            }
        }

        let mut suffix = 0_u32;
        for _ in 0..suffix_bits {
            suffix = (suffix << 1) | u32::from(self.decode_bypass()?);
        }
        symbol
            .checked_add(suffix)
            .ok_or(CabacError::SymbolValueOverflow)
    }

    pub fn decode_bypass_ue_legacy(&mut self) -> Result<u32, CabacError> {
        let mut leading_ones = 0_u32;
        while self.decode_bypass()? {
            leading_ones = leading_ones
                .checked_add(1)
                .ok_or(CabacError::SymbolValueOverflow)?;
            if leading_ones >= 32 {
                return Err(CabacError::SymbolValueOverflow);
            }
        }
        let mut suffix = 0_u32;
        for _ in 0..leading_ones {
            suffix = (suffix << 1) | u32::from(self.decode_bypass()?);
        }
        (1_u32 << leading_ones)
            .checked_sub(1)
            .and_then(|prefix| prefix.checked_add(suffix))
            .ok_or(CabacError::SymbolValueOverflow)
    }

    /// Arithmetic bin. The slice end is a sticky flag, checked by the caller
    /// once per block, so this stays a straight register update.
    #[inline(always)]
    pub(crate) fn decode_bin(&mut self, context: &mut CabacContext) -> bool {
        let packed = context.packed;
        let state = usize::from(packed & 63);
        let mps = packed & 64 != 0;
        let q = usize::from((self.range >> 6) & 0x03);
        let range_lps = u16::from(RANGE_LPS[state][q]);
        let range_mps = self.range - range_lps;
        if self.offset < range_mps {
            self.range = range_mps;
            context.packed = TRANS_PACKED_MPS[usize::from(packed)];
            if range_mps < 256 {
                let shift = range_mps.leading_zeros() - 7;
                self.range = range_mps << shift;
                self.offset = (self.offset << shift) | self.read_renorm_bits(shift);
            }
            mps
        } else {
            self.offset -= range_mps;
            self.range = range_lps;
            let bin = !mps;
            context.packed = TRANS_PACKED_LPS[usize::from(packed)];
            self.renorm();
            bin
        }
    }

    #[inline(always)]
    pub fn decode_decision(&mut self, context: &mut CabacContext) -> Result<bool, CabacError> {
        let bin = self.decode_bin(context);
        if self.fault {
            Err(CabacError::EndOfSlice)
        } else {
            Ok(bin)
        }
    }

    pub fn decode_unary_symbol(
        &mut self,
        contexts: &mut [CabacContext],
    ) -> Result<u32, CabacError> {
        if contexts.is_empty() {
            return Err(CabacError::MissingContext);
        }
        let mut value = 0_u32;
        loop {
            let context_index = usize::min(value as usize, contexts.len() - 1);
            if !self.decode_decision(&mut contexts[context_index])? {
                return Ok(value);
            }
            value = value
                .checked_add(1)
                .ok_or(CabacError::SymbolValueOverflow)?;
        }
    }

    pub fn decode_truncated_unary_symbol(
        &mut self,
        contexts: &mut [CabacContext],
        max_symbol: u32,
    ) -> Result<u32, CabacError> {
        if contexts.is_empty() {
            return Err(CabacError::MissingContext);
        }
        if max_symbol == 0 {
            return Ok(0);
        }
        let mut value = 0_u32;
        while value < max_symbol {
            let context_index = usize::min(value as usize, contexts.len() - 1);
            if !self.decode_decision(&mut contexts[context_index])? {
                return Ok(value);
            }
            value += 1;
        }
        Ok(max_symbol)
    }

    pub fn decode_terminate(&mut self) -> Result<bool, CabacError> {
        debug_assert!(self.range >= 2);
        self.range -= 2;
        if self.offset >= self.range {
            return if self.fault {
                Err(CabacError::EndOfSlice)
            } else {
                Ok(true)
            };
        }
        self.renorm();
        if self.fault {
            Err(CabacError::EndOfSlice)
        } else {
            Ok(false)
        }
    }

    #[inline(always)]
    fn renorm(&mut self) {
        if self.range < 256 {
            let shift = self.range.leading_zeros() - 7;
            self.range <<= shift;
            self.offset = (self.offset << shift) | self.read_renorm_bits(shift);
        }
    }

    #[inline(always)]
    fn read_renorm_bits(&mut self, shift: u32) -> u16 {
        let mut bits_left = self.bits_left;
        let mut bit_buffer = self.bit_buffer;
        let mut next_byte = self.next_byte;
        let len = self.data.len();
        while u32::from(bits_left) < shift {
            if next_byte >= len {
                self.fault = true;
                self.bits_left = bits_left;
                self.bit_buffer = bit_buffer;
                self.next_byte = next_byte;
                return 0;
            }
            bit_buffer = (bit_buffer << 8) | u64::from(self.data[next_byte]);
            next_byte += 1;
            bits_left += 8;
        }
        bits_left -= shift as u8;
        let bits = ((bit_buffer >> bits_left) as u16) & ((1_u16 << shift) - 1);
        self.bits_left = bits_left;
        self.bit_buffer = bit_buffer;
        self.next_byte = next_byte;
        bits
    }

    #[inline(always)]
    fn read_bit(&mut self) -> Result<bool, CabacError> {
        Ok(self.read_bits(1)? != 0)
    }

    fn read_bits(&mut self, count: usize) -> Result<u32, CabacError> {
        if count > 32 {
            return Err(CabacError::TooManyBits);
        }
        while usize::from(self.bits_left) < count {
            if self.next_byte >= self.data.len() {
                return Err(CabacError::EndOfSlice);
            }
            self.bit_buffer = (self.bit_buffer << 8) | u64::from(self.data[self.next_byte]);
            self.next_byte += 1;
            self.bits_left += 8;
        }
        self.bits_left -= count as u8;
        let mask = if count == 32 {
            u64::from(u32::MAX)
        } else {
            (1_u64 << count) - 1
        };
        Ok(((self.bit_buffer >> self.bits_left) & mask) as u32)
    }
}

pub const RANGE_LPS: [[u8; 4]; 64] = [
    [128, 176, 208, 240],
    [128, 167, 197, 227],
    [128, 158, 187, 216],
    [123, 150, 178, 205],
    [116, 142, 169, 195],
    [111, 135, 160, 185],
    [105, 128, 152, 175],
    [100, 122, 144, 166],
    [95, 116, 137, 158],
    [90, 110, 130, 150],
    [85, 104, 123, 142],
    [81, 99, 117, 135],
    [77, 94, 111, 128],
    [73, 89, 105, 122],
    [69, 85, 100, 116],
    [66, 80, 95, 110],
    [62, 76, 90, 104],
    [59, 72, 86, 99],
    [56, 69, 81, 94],
    [53, 65, 77, 89],
    [51, 62, 73, 85],
    [48, 59, 69, 80],
    [46, 56, 66, 76],
    [43, 53, 63, 72],
    [41, 50, 59, 69],
    [39, 48, 56, 65],
    [37, 45, 54, 62],
    [35, 43, 51, 59],
    [33, 41, 48, 56],
    [32, 39, 46, 53],
    [30, 37, 43, 50],
    [29, 35, 41, 48],
    [27, 33, 39, 45],
    [26, 31, 37, 43],
    [24, 30, 35, 41],
    [23, 28, 33, 39],
    [22, 27, 32, 37],
    [21, 26, 30, 35],
    [20, 24, 29, 33],
    [19, 23, 27, 31],
    [18, 22, 26, 30],
    [17, 21, 25, 28],
    [16, 20, 23, 27],
    [15, 19, 22, 25],
    [14, 18, 21, 24],
    [14, 17, 20, 23],
    [13, 16, 19, 22],
    [12, 15, 18, 21],
    [12, 14, 17, 20],
    [11, 14, 16, 19],
    [11, 13, 15, 18],
    [10, 12, 15, 17],
    [10, 12, 14, 16],
    [9, 11, 13, 15],
    [9, 11, 12, 14],
    [8, 10, 12, 14],
    [8, 9, 11, 13],
    [7, 9, 11, 12],
    [7, 9, 10, 12],
    [7, 8, 10, 11],
    [6, 8, 9, 11],
    [6, 7, 9, 10],
    [6, 7, 8, 9],
    [2, 2, 2, 2],
];

pub const TRANS_IDX_LPS: [u8; 64] = [
    0, 0, 1, 2, 2, 4, 4, 5, 6, 7, 8, 9, 9, 11, 11, 12, 13, 13, 15, 15, 16, 16, 18, 18, 19, 19, 21,
    21, 22, 22, 23, 24, 24, 25, 26, 26, 27, 27, 28, 29, 29, 30, 30, 30, 31, 32, 32, 33, 33, 33, 34,
    34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 63,
];

pub const TRANS_IDX_MPS: [u8; 64] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 62, 63,
];

const fn build_packed_transition_tables(lps: bool) -> [u8; 128] {
    let mut table = [0_u8; 128];
    let mut packed = 0_usize;
    while packed < table.len() {
        let state = packed & 63;
        let old_mps = (packed as u8) & 64;
        table[packed] = if lps {
            TRANS_IDX_LPS[state] | (old_mps ^ if state == 0 { 64 } else { 0 })
        } else {
            TRANS_IDX_MPS[state] | old_mps
        };
        packed += 1;
    }
    table
}

const TRANS_PACKED_MPS: [u8; 128] = build_packed_transition_tables(false);
const TRANS_PACKED_LPS: [u8; 128] = build_packed_transition_tables(true);

#[derive(Debug)]
pub enum CabacError {
    EndOfSlice,
    MissingContext,
    SymbolValueOverflow,
    TooManyBits,
}

impl fmt::Display for CabacError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EndOfSlice => write!(f, "CABAC reader reached end of slice"),
            Self::MissingContext => write!(f, "CABAC syntax decode is missing a context"),
            Self::SymbolValueOverflow => write!(f, "CABAC symbol value overflowed"),
            Self::TooManyBits => write!(f, "CABAC reader requested too many bits"),
        }
    }
}

impl std::error::Error for CabacError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cabac_initializes_range_and_offset_from_first_nine_bits() {
        let decoder = CabacDecoder::new(&[0b1010_1010, 0b1000_0000]).unwrap();

        assert_eq!(decoder.range(), 510);
        assert_eq!(decoder.offset(), 0b1010_1010_1);
        assert_eq!(decoder.bit_position(), 9);
    }

    #[test]
    fn cabac_bypass_decodes_raw_bins_against_range() {
        let mut decoder = CabacDecoder::new(&[0, 0b1100_0000]).unwrap();

        assert!(!decoder.decode_bypass().unwrap());
        assert!(!decoder.decode_bypass().unwrap());
        assert_eq!(decoder.bit_position(), 11);
    }

    #[test]
    fn cabac_bypass_ue_decodes_zero_value() {
        let mut decoder = CabacDecoder::new(&[0, 0b0000_0000]).unwrap();

        assert_eq!(decoder.decode_bypass_ue().unwrap(), 0);
    }

    #[test]
    fn cabac_terminate_returns_true_when_offset_enters_terminal_range() {
        let mut decoder = CabacDecoder::new(&[0xff, 0x80]).unwrap();

        assert!(decoder.decode_terminate().unwrap());
    }

    #[test]
    fn cabac_decision_mps_advances_context_state() {
        let mut decoder = CabacDecoder::new(&[0, 0]).unwrap();
        let mut context = CabacContext::new(0, false);

        assert!(!decoder.decode_decision(&mut context).unwrap());
        assert_eq!(context, CabacContext::new(1, false));
        assert!(decoder.range() >= 256);
    }

    #[test]
    fn cabac_decision_lps_flips_mps_at_state_zero() {
        let mut decoder = CabacDecoder::new(&[0xff, 0xff]).unwrap();
        let mut context = CabacContext::new(0, false);

        assert!(decoder.decode_decision(&mut context).unwrap());
        assert_eq!(context, CabacContext::new(0, true));
        assert!(decoder.range() >= 256);
    }

    #[test]
    fn cabac_unary_symbol_stops_on_first_zero_bin() {
        let mut decoder = CabacDecoder::new(&[0, 0, 0]).unwrap();
        let mut contexts = [CabacContext::new(0, false); 2];

        assert_eq!(decoder.decode_unary_symbol(&mut contexts).unwrap(), 0);
    }

    #[test]
    fn cabac_truncated_unary_symbol_respects_max_symbol() {
        let mut decoder = CabacDecoder::new(&[0, 0, 0, 0]).unwrap();
        let mut contexts = [CabacContext::new(0, true); 2];

        assert_eq!(
            decoder
                .decode_truncated_unary_symbol(&mut contexts, 3)
                .unwrap(),
            3
        );
    }

    #[test]
    fn cabac_unary_symbol_requires_at_least_one_context() {
        let mut decoder = CabacDecoder::new(&[0, 0]).unwrap();
        let err = decoder.decode_unary_symbol(&mut []).unwrap_err();

        assert!(matches!(err, CabacError::MissingContext));
    }

    #[test]
    fn cabac_tables_have_expected_edge_entries() {
        assert_eq!(RANGE_LPS[0], [128, 176, 208, 240]);
        assert_eq!(RANGE_LPS[63], [2, 2, 2, 2]);
        assert_eq!(TRANS_IDX_LPS[0], 0);
        assert_eq!(TRANS_IDX_MPS[0], 1);
        assert_eq!(TRANS_IDX_LPS[63], 63);
        assert_eq!(TRANS_IDX_MPS[63], 63);
    }

    #[test]
    fn cabac_context_initialization_uses_slice_qp_formula() {
        let init = CabacInitValue::new(20, -15);

        assert_eq!(init.initialize(26), CabacContext::new(46, false));
        assert_eq!(init.initialize(-8), CabacContext::new(62, false));
        assert_eq!(init.initialize(99), CabacContext::new(15, false));
    }
}
