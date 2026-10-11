#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Plane422P10 {
    pub width: usize,
    pub height: usize,
    #[cfg(target_endian = "little")]
    samples: Vec<u8>,
    #[cfg(target_endian = "big")]
    samples: Vec<u16>,
}

impl Plane422P10 {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            samples: vec![
                0;
                width
                    .saturating_mul(height)
                    .saturating_mul(if cfg!(target_endian = "little") { 2 } else { 1 })
            ],
        }
    }

    pub fn get(&self, x: usize, y: usize) -> Option<u16> {
        (x < self.width && y < self.height).then(|| self.samples()[y * self.width + x])
    }

    pub(crate) fn samples(&self) -> &[u16] {
        #[cfg(target_endian = "little")]
        return bytemuck::cast_slice(&self.samples);
        #[cfg(target_endian = "big")]
        return &self.samples;
    }

    pub(crate) fn samples_mut(&mut self) -> &mut [u16] {
        #[cfg(target_endian = "little")]
        return bytemuck::cast_slice_mut(&mut self.samples);
        #[cfg(target_endian = "big")]
        return &mut self.samples;
    }

    pub fn set(&mut self, x: usize, y: usize, value: u16) -> bool {
        if x >= self.width || y >= self.height {
            return false;
        }
        let index = y * self.width + x;
        self.samples_mut()[index] = value.min(1023);
        true
    }

    pub(crate) fn fill_zero(&mut self) {
        self.samples.fill(0);
    }

    pub fn write_block<const W: usize, const H: usize>(
        &mut self,
        x: usize,
        y: usize,
        block: &[[u16; W]; H],
    ) -> bool {
        if x + W > self.width || y + H > self.height {
            return false;
        }
        for (row, values) in block.iter().enumerate() {
            let start = (y + row) * self.width + x;
            debug_assert!(values.iter().all(|&value| value <= 1023));
            self.samples_mut()[start..start + W].copy_from_slice(values);
        }
        true
    }

    pub fn to_le_bytes(&self) -> Vec<u8> {
        #[cfg(target_endian = "little")]
        {
            return self.samples.clone();
        }

        #[cfg(target_endian = "big")]
        let mut bytes = Vec::with_capacity(self.samples.len() * 2);
        #[cfg(target_endian = "big")]
        for sample in &self.samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        #[cfg(target_endian = "big")]
        bytes
    }

    pub(crate) fn into_le_bytes(self) -> Vec<u8> {
        #[cfg(target_endian = "little")]
        {
            return self.samples;
        }

        #[cfg(target_endian = "big")]
        let mut bytes = Vec::with_capacity(self.samples.len() * 2);
        #[cfg(target_endian = "big")]
        for sample in self.samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        #[cfg(target_endian = "big")]
        bytes
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecodedFrame422P10 {
    pub coded_width: usize,
    pub coded_height: usize,
    pub visible_width: usize,
    pub visible_height: usize,
    pub y: Plane422P10,
    pub cb: Plane422P10,
    pub cr: Plane422P10,
    pub scaling: crate::residual::PictureScaling,
    pub slice_first_mb: u32,
}

impl DecodedFrame422P10 {
    pub fn new(
        coded_width: usize,
        coded_height: usize,
        visible_width: usize,
        visible_height: usize,
    ) -> Self {
        Self {
            coded_width,
            coded_height,
            visible_width,
            visible_height,
            y: Plane422P10::new(coded_width, coded_height),
            cb: Plane422P10::new(coded_width / 2, coded_height),
            cr: Plane422P10::new(coded_width / 2, coded_height),
            scaling: crate::residual::PictureScaling::flat(),
            slice_first_mb: 0,
        }
    }

    pub fn luma_in_slice(&self, x: usize, y: usize) -> bool {
        self.mb_in_slice(x / 16, y / 16)
    }

    pub fn chroma_in_slice(&self, x: usize, y: usize) -> bool {
        self.mb_in_slice(x / 8, y / 16)
    }

    pub fn mb_in_slice(&self, mb_x: usize, mb_y: usize) -> bool {
        let mb_width = (self.coded_width / 16).max(1);
        mb_y.saturating_mul(mb_width).saturating_add(mb_x) >= self.slice_first_mb as usize
    }

    pub(crate) fn clear_for_reuse(&mut self) {
        // Complete High 4:2:2 pictures overwrite every coded sample before
        // output/reference use, so reuse skips the 8 MB plane memset.
        self.slice_first_mb = 0;
    }

    pub(crate) fn matches_geometry(
        &self,
        coded_width: usize,
        coded_height: usize,
        visible_width: usize,
        visible_height: usize,
    ) -> bool {
        self.coded_width == coded_width
            && self.coded_height == coded_height
            && self.visible_width == visible_width
            && self.visible_height == visible_height
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_preserves_422_chroma_geometry() {
        let frame = DecodedFrame422P10::new(1920, 1088, 1920, 1080);

        assert_eq!(frame.y.width, 1920);
        assert_eq!(frame.y.height, 1088);
        assert_eq!(frame.cb.width, 960);
        assert_eq!(frame.cb.height, 1088);
        assert_eq!(frame.cr.width, 960);
        assert_eq!(frame.cr.height, 1088);
    }

    #[test]
    fn plane_writes_and_exports_little_endian_10bit_samples() {
        let mut plane = Plane422P10::new(2, 1);
        assert!(plane.set(0, 0, 1023));
        assert!(plane.set(1, 0, 2048));

        assert_eq!(plane.get(0, 0), Some(1023));
        assert_eq!(plane.get(1, 0), Some(1023));
        assert_eq!(plane.to_le_bytes(), vec![0xff, 0x03, 0xff, 0x03]);
    }

    #[cfg(target_endian = "little")]
    #[test]
    fn consuming_plane_reuses_the_output_allocation() {
        let mut plane = Plane422P10::new(4, 2);
        assert!(plane.set(1, 0, 513));
        let allocation = plane.samples.as_ptr();
        let bytes = plane.into_le_bytes();
        assert_eq!(bytes.as_ptr(), allocation);
        assert_eq!(&bytes[2..4], &513_u16.to_le_bytes());
    }
}
