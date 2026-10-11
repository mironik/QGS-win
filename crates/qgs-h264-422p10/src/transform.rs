pub fn clip10(value: i32) -> u16 {
    value.clamp(0, 1023) as u16
}

const NORM_ADJUST_4X4_DC: [i32; 6] = [10, 11, 13, 14, 16, 18];

pub fn inverse_transform_4x4(coeffs: [[i32; 4]; 4], pred: [[u16; 4]; 4]) -> [[u16; 4]; 4] {
    let mut tmp = [[0_i32; 4]; 4];
    for i in 0..4 {
        let a0 = coeffs[i][0].wrapping_add(coeffs[i][2]);
        let a1 = coeffs[i][0].wrapping_sub(coeffs[i][2]);
        let a2 = (coeffs[i][1] >> 1).wrapping_sub(coeffs[i][3]);
        let a3 = coeffs[i][1].wrapping_add(coeffs[i][3] >> 1);
        tmp[i][0] = a0.wrapping_add(a3);
        tmp[i][1] = a1.wrapping_add(a2);
        tmp[i][2] = a1.wrapping_sub(a2);
        tmp[i][3] = a0.wrapping_sub(a3);
    }

    let mut out = [[0_u16; 4]; 4];
    for i in 0..4 {
        let a0 = tmp[0][i].wrapping_add(tmp[2][i]);
        let a1 = tmp[0][i].wrapping_sub(tmp[2][i]);
        let a2 = (tmp[1][i] >> 1).wrapping_sub(tmp[3][i]);
        let a3 = tmp[1][i].wrapping_add(tmp[3][i] >> 1);
        let residuals = [
            a0.wrapping_add(a3),
            a1.wrapping_add(a2),
            a1.wrapping_sub(a2),
            a0.wrapping_sub(a3),
        ];
        for row in 0..4 {
            out[row][i] =
                clip10(i32::from(pred[row][i]).wrapping_add(residuals[row].wrapping_add(32) >> 6));
        }
    }
    out
}

pub fn inverse_transform_8x8(coeffs: [[i32; 8]; 8], pred: [[u16; 8]; 8]) -> [[u16; 8]; 8] {
    // Level scan stores the transpose of the block the inverse transform reads.
    let mut tmp = [[0_i32; 8]; 8];
    for column in 0..8 {
        let input = std::array::from_fn(|row| coeffs[column][row]);
        let transformed = inverse_transform_8x8_1d(input);
        for row in 0..8 {
            tmp[row][column] = transformed[row];
        }
    }

    let mut output = [[0_u16; 8]; 8];
    for column in 0..8 {
        let transformed = inverse_transform_8x8_1d(tmp[column]);
        for row in 0..8 {
            output[row][column] =
                clip10(i32::from(pred[row][column]).saturating_add((transformed[row] + 32) >> 6));
        }
    }
    output
}

fn inverse_transform_8x8_1d(input: [i32; 8]) -> [i32; 8] {
    let a0 = input[0] + input[4];
    let a2 = input[0] - input[4];
    let a4 = (input[2] >> 1) - input[6];
    let a6 = (input[6] >> 1) + input[2];
    let b0 = a0 + a6;
    let b2 = a2 + a4;
    let b4 = a2 - a4;
    let b6 = a0 - a6;
    let a1 = -input[3] + input[5] - input[7] - (input[7] >> 1);
    let a3 = input[1] + input[7] - input[3] - (input[3] >> 1);
    let a5 = -input[1] + input[7] + input[5] + (input[5] >> 1);
    let a7 = input[3] + input[5] + input[1] + (input[1] >> 1);
    let b1 = (a7 >> 2) + a1;
    let b3 = a3 + (a5 >> 2);
    let b5 = (a3 >> 2) - a5;
    let b7 = a7 - (a1 >> 2);
    [
        b0 + b7,
        b2 + b5,
        b4 + b3,
        b6 + b1,
        b6 - b1,
        b4 - b3,
        b2 - b5,
        b0 - b7,
    ]
}

pub fn inverse_intra16x16_dc_transform(levels: [[i32; 4]; 4], qp: u8) -> [[i32; 4]; 4] {
    inverse_intra16x16_dc_transform_weighted(levels, qp, 16)
}

pub fn inverse_intra16x16_dc_transform_weighted(
    levels: [[i32; 4]; 4],
    qp: u8,
    dc_weight: i32,
) -> [[i32; 4]; 4] {
    let transformed = hadamard_4x4(levels);
    let qp = i32::from(qp).clamp(0, 63);
    std::array::from_fn(|row| {
        std::array::from_fn(|column| scale_dc_coefficient(transformed[row][column], qp, dc_weight))
    })
}

fn scale_dc_coefficient(transformed: i32, qp: i32, dc_weight: i32) -> i32 {
    let qbits = qp / 6;
    let level_scale = i64::from(dc_weight) * i64::from(NORM_ADJUST_4X4_DC[(qp % 6) as usize]);
    let value = i64::from(transformed) * level_scale;
    let scaled = if qp >= 36 {
        value << (qbits - 6)
    } else {
        (value + (1_i64 << (5 - qbits))) >> (6 - qbits)
    };
    scaled.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

pub fn inverse_chroma422_dc_transform(levels: [[i32; 2]; 4], qp: u8) -> [[i32; 2]; 4] {
    inverse_chroma422_dc_transform_weighted(levels, qp, 16)
}

pub fn inverse_chroma422_dc_transform_weighted(
    levels: [[i32; 2]; 4],
    qp: u8,
    dc_weight: i32,
) -> [[i32; 2]; 4] {
    let mut horizontal = [[0_i32; 2]; 4];
    for row in 0..4 {
        horizontal[row][0] = levels[row][0] + levels[row][1];
        horizontal[row][1] = levels[row][0] - levels[row][1];
    }

    let mut transformed = [[0_i32; 2]; 4];
    for column in 0..2 {
        let a0 = horizontal[0][column] + horizontal[3][column];
        let a1 = horizontal[1][column] + horizontal[2][column];
        let a2 = horizontal[1][column] - horizontal[2][column];
        let a3 = horizontal[0][column] - horizontal[3][column];
        transformed[0][column] = a0 + a1;
        transformed[1][column] = a3 + a2;
        transformed[2][column] = a0 - a1;
        transformed[3][column] = a3 - a2;
    }

    let qp = i32::from(qp).clamp(0, 63);
    let qp_dc = qp + 3;
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            scale_dc_coefficient(transformed[row][column], qp_dc, dc_weight)
        })
    })
}

fn hadamard_4x4(input: [[i32; 4]; 4]) -> [[i32; 4]; 4] {
    let mut tmp = [[0_i32; 4]; 4];
    for row in 0..4 {
        let a0 = input[row][0] + input[row][3];
        let a1 = input[row][1] + input[row][2];
        let a2 = input[row][1] - input[row][2];
        let a3 = input[row][0] - input[row][3];
        tmp[row][0] = a0 + a1;
        tmp[row][1] = a3 + a2;
        tmp[row][2] = a0 - a1;
        tmp[row][3] = a3 - a2;
    }

    let mut transformed = [[0_i32; 4]; 4];
    for column in 0..4 {
        let a0 = tmp[0][column] + tmp[3][column];
        let a1 = tmp[1][column] + tmp[2][column];
        let a2 = tmp[1][column] - tmp[2][column];
        let a3 = tmp[0][column] - tmp[3][column];
        let rows = [a0 + a1, a3 + a2, a0 - a1, a3 - a2];
        for row in 0..4 {
            transformed[row][column] = rows[row];
        }
    }
    transformed
}

pub fn intra16x16_dc_prediction(
    top: Option<[u16; 16]>,
    left: Option<[u16; 16]>,
) -> [[u16; 16]; 16] {
    let value = match (top, left) {
        (Some(top), Some(left)) => {
            let total: u32 = top.into_iter().chain(left).map(u32::from).sum();
            ((total + 16) >> 5) as u16
        }
        (Some(top), None) => {
            let total: u32 = top.into_iter().map(u32::from).sum();
            ((total + 8) >> 4) as u16
        }
        (None, Some(left)) => {
            let total: u32 = left.into_iter().map(u32::from).sum();
            ((total + 8) >> 4) as u16
        }
        (None, None) => 512,
    };
    [[value; 16]; 16]
}

pub fn intra16x16_vertical_prediction(top: [u16; 16]) -> [[u16; 16]; 16] {
    [top; 16]
}

pub fn intra16x16_horizontal_prediction(left: [u16; 16]) -> [[u16; 16]; 16] {
    let mut out = [[0_u16; 16]; 16];
    for (row, value) in left.into_iter().enumerate() {
        out[row] = [value; 16];
    }
    out
}

pub fn intra16x16_plane_prediction(
    top: [u16; 16],
    left: [u16; 16],
    top_left: u16,
) -> [[u16; 16]; 16] {
    let mut h = 0_i32;
    let mut v = 0_i32;
    for i in 0..8 {
        let weight = i as i32 + 1;
        let top_mirror = if i < 7 { top[6 - i] } else { top_left };
        let left_mirror = if i < 7 { left[6 - i] } else { top_left };
        h += weight * (i32::from(top[8 + i]) - i32::from(top_mirror));
        v += weight * (i32::from(left[8 + i]) - i32::from(left_mirror));
    }
    let a = 16 * (i32::from(top[15]) + i32::from(left[15]));
    let b = (5 * h + 32) >> 6;
    let c = (5 * v + 32) >> 6;
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            clip10((a + b * (column as i32 - 7) + c * (row as i32 - 7) + 16) >> 5)
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_transform_adds_dc_residual_and_clips() {
        let mut coeffs = [[0_i32; 4]; 4];
        coeffs[0][0] = 64;
        let pred = [[100_u16; 4]; 4];

        let out = inverse_transform_4x4(coeffs, pred);

        assert_eq!(out, [[101_u16; 4]; 4]);
    }

    #[test]
    fn inverse_transform_8x8_adds_dc_residual_and_clips() {
        let mut coeffs = [[0_i32; 8]; 8];
        coeffs[0][0] = 64;
        let pred = [[100_u16; 8]; 8];

        let out = inverse_transform_8x8(coeffs, pred);

        assert!(out.iter().flatten().all(|sample| *sample >= 100));
        assert!(out.iter().flatten().all(|sample| *sample == out[0][0]));
    }

    #[test]
    fn inverse_intra16x16_dc_transform_spreads_dc_level_to_all_blocks() {
        let mut levels = [[0_i32; 4]; 4];
        levels[0][0] = 1;

        let dc = inverse_intra16x16_dc_transform(levels, 24);

        assert!(dc.iter().flatten().all(|value| *value > 0));
        assert!(dc.iter().flatten().all(|value| *value == dc[0][0]));
    }

    #[test]
    fn inverse_intra16x16_dc_transform_preserves_hadamard_signs() {
        let mut levels = [[0_i32; 4]; 4];
        levels[0][0] = 1;
        levels[0][1] = -1;

        let dc = inverse_intra16x16_dc_transform(levels, 24);
        let min = dc.iter().flatten().copied().min().unwrap();
        let max = dc.iter().flatten().copied().max().unwrap();

        assert!(dc.iter().flatten().any(|value| *value != 0));
        assert_ne!(min, max);
    }

    #[test]
    fn inverse_chroma422_dc_transform_spreads_dc_level_to_eight_blocks() {
        let mut levels = [[0_i32; 2]; 4];
        levels[0][0] = 1;

        let dc = inverse_chroma422_dc_transform(levels, 24);

        assert!(dc.iter().flatten().all(|value| *value > 0));
        assert!(dc.iter().flatten().all(|value| *value == dc[0][0]));
    }

    #[test]
    fn inverse_chroma422_dc_transform_preserves_nonuniform_pattern() {
        let mut levels = [[0_i32; 2]; 4];
        levels[0][0] = 1;
        levels[1][0] = -1;
        let dc = inverse_chroma422_dc_transform(levels, 24);
        let min = dc.iter().flatten().copied().min().unwrap();
        let max = dc.iter().flatten().copied().max().unwrap();

        assert_ne!(min, max);
    }

    #[test]
    fn chroma422_dc_scale_uses_qp_plus_three_below_the_high_threshold() {
        let mut levels = [[0_i32; 2]; 4];
        levels[0][0] = 1;

        let dc = inverse_chroma422_dc_transform(levels, 30);

        assert!(dc.iter().flatten().all(|value| *value == 112));
    }

    #[test]
    fn chroma422_dc_scale_keeps_qp_plus_three_at_the_high_threshold() {
        let mut levels = [[0_i32; 2]; 4];
        levels[0][0] = 1;

        let dc = inverse_chroma422_dc_transform(levels, 33);

        assert!(dc.iter().flatten().all(|value| *value == 160));
    }

    #[test]
    fn intra_dc_prediction_uses_midpoint_when_unavailable() {
        assert_eq!(intra16x16_dc_prediction(None, None), [[512_u16; 16]; 16]);
    }

    #[test]
    fn intra_horizontal_prediction_repeats_left_column() {
        let mut left = [0_u16; 16];
        left[3] = 77;

        let out = intra16x16_horizontal_prediction(left);

        assert_eq!(out[3], [77_u16; 16]);
    }

    #[test]
    fn intra_plane_prediction_slopes_from_top_and_left_edges() {
        let top = std::array::from_fn(|index| (400 + index as u16) as u16);
        let left = std::array::from_fn(|index| (500 + index as u16) as u16);

        let out = intra16x16_plane_prediction(top, left, 399);

        assert!(out[15][15] > out[0][0]);
    }

    #[test]
    fn intra_plane_prediction_uses_the_top_left_corner_in_the_last_term() {
        let mut top = [0_u16; 16];
        top[0] = 80;
        top[15] = 100;
        let left = [0_u16; 16];

        let predicted = intra16x16_plane_prediction(top, left, 0);

        assert_eq!(predicted[0][0], 46);
        assert_ne!(
            predicted[0][0],
            intra16x16_plane_prediction(top, left, 80)[0][0]
        );
    }
}
