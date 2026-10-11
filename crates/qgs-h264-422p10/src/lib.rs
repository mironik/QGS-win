#![deny(unsafe_op_in_unsafe_fn)]

pub mod bitstream;
pub mod bytestream;
pub mod cabac;
pub mod cabac_macroblock;
pub mod cabac_motion;
pub mod cabac_nonzero;
pub mod cabac_residual;
pub mod cabac_residual_422;
mod deblock;
pub mod decoder;
pub mod frame;
pub mod macroblock;
pub mod macroblock_type;
mod mc_interp;
pub mod motion;
pub mod residual;
mod simd;
pub mod slice;
mod sony_fast;
mod sony_plan;
pub mod transform;

use std::collections::BTreeMap;
use std::fmt;

use qgs_codec_h264::{H264DecoderState, H264PictureId, H264SliceKind, ParsedH264AccessUnit};
use qgs_protocol::{ChromaSubsampling, FieldOrder, H264Profile, ScanMode, VideoSurfaceFormat};

pub use decoder::{H264422P10Decoder, H264422P10PictureDecodeError};
pub use frame::{DecodedFrame422P10, Plane422P10};

pub const SONY_FX6_CODED_WIDTH: u32 = 1920;
pub const SONY_FX6_CODED_HEIGHT: u32 = 1088;
pub const SONY_FX6_VISIBLE_WIDTH: u32 = 1920;
pub const SONY_FX6_VISIBLE_HEIGHT: u32 = 1080;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H264422P10Profile {
    pub profile: H264Profile,
    pub bit_depth: u8,
    pub chroma: ChromaSubsampling,
    pub coded_width: u32,
    pub coded_height: u32,
    pub visible_width: u32,
    pub visible_height: u32,
    pub scan_mode: ScanMode,
}

impl H264422P10Profile {
    pub const fn sony_fx6_original() -> Self {
        Self {
            profile: H264Profile::High422,
            bit_depth: 10,
            chroma: ChromaSubsampling::Cs422,
            coded_width: SONY_FX6_CODED_WIDTH,
            coded_height: SONY_FX6_CODED_HEIGHT,
            visible_width: SONY_FX6_VISIBLE_WIDTH,
            visible_height: SONY_FX6_VISIBLE_HEIGHT,
            scan_mode: ScanMode::Progressive,
        }
    }
}

impl Default for H264422P10Profile {
    fn default() -> Self {
        Self::sony_fx6_original()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H264422P10Plane {
    pub width_samples: u32,
    pub height: u32,
    pub stride_bytes: usize,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H264422P10Frame {
    pub presentation_index: u64,
    pub coded_width: u32,
    pub coded_height: u32,
    pub visible_width: u32,
    pub visible_height: u32,
    pub y: H264422P10Plane,
    pub cb: H264422P10Plane,
    pub cr: H264422P10Plane,
}

impl H264422P10Frame {
    pub fn validate_layout(&self) -> Result<(), H264422P10Error> {
        validate_plane("Y", &self.y, self.coded_width, self.coded_height)?;
        validate_plane("Cb", &self.cb, self.coded_width / 2, self.coded_height)?;
        validate_plane("Cr", &self.cr, self.coded_width / 2, self.coded_height)?;
        if self.visible_width > self.coded_width || self.visible_height > self.coded_height {
            return Err(H264422P10Error::InvalidOutputLayout(
                "visible region exceeds coded dimensions",
            ));
        }
        Ok(())
    }

    pub fn owned_bytes(&self) -> usize {
        self.y.data.len() + self.cb.data.len() + self.cr.data.len()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H264422P10AccessUnitReport {
    pub presentation_index: u64,
    pub profile: H264422P10Profile,
    pub idr: bool,
    pub reference_picture: bool,
    pub slice_kinds: Vec<H264SliceKind>,
    pub reference_count: usize,
    pub max_dpb_frames: usize,
    pub max_num_reorder_frames: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H264422P10DecodeJob {
    pub decode_index: u64,
    pub picture_id: H264PictureId,
    pub idr: bool,
    pub reference_picture: bool,
    pub slice_kinds: Vec<H264SliceKind>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H264422P10OutputPicture {
    pub decode_index: u64,
    pub picture_id: H264PictureId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H264422P10DecodeStep {
    pub job: H264422P10DecodeJob,
    pub output_ready: Vec<H264422P10OutputPicture>,
    pub released: Vec<H264422P10OutputPicture>,
    pub max_dpb_occupancy: usize,
    pub max_output_pending: usize,
}

#[derive(Debug)]
pub struct H264422P10DecodePlanner {
    state: H264DecoderState,
    expected: H264422P10Profile,
    next_decode_index: u64,
    pictures: BTreeMap<(u16, i32), H264422P10DecodeJob>,
}

impl Default for H264422P10DecodePlanner {
    fn default() -> Self {
        Self::new(H264422P10Profile::default())
    }
}

impl H264422P10DecodePlanner {
    pub fn new(expected: H264422P10Profile) -> Self {
        Self {
            state: H264DecoderState::new(),
            expected,
            next_decode_index: 0,
            pictures: BTreeMap::new(),
        }
    }

    pub fn submit_access_unit(
        &mut self,
        access_unit: &[u8],
    ) -> Result<H264422P10DecodeStep, H264422P10Error> {
        let parsed = self.state.parse_access_unit(access_unit)?;
        validate_original_profile(&parsed, &self.expected)?;
        let job = decode_job(self.next_decode_index, &parsed);
        self.next_decode_index = self.next_decode_index.saturating_add(1);
        self.pictures
            .insert(picture_key(&job.picture_id), job.clone());
        let update = self.state.finish_picture(&parsed)?;
        let output_ready = self.resolve_output_pictures(update.output_ready)?;
        let released = self.resolve_and_release_pictures(update.released)?;
        Ok(H264422P10DecodeStep {
            job,
            output_ready,
            released,
            max_dpb_occupancy: update.max_dpb_occupancy,
            max_output_pending: update.max_output_pending,
        })
    }

    pub fn flush(&mut self) -> Result<Vec<H264422P10OutputPicture>, H264422P10Error> {
        let update = self.state.flush();
        let output_ready = self.resolve_output_pictures(update.output_ready)?;
        let _ = self.resolve_and_release_pictures(update.released)?;
        Ok(output_ready)
    }

    pub fn pending_picture_count(&self) -> usize {
        self.pictures.len()
    }

    fn resolve_output_pictures(
        &self,
        ids: Vec<H264PictureId>,
    ) -> Result<Vec<H264422P10OutputPicture>, H264422P10Error> {
        ids.into_iter()
            .map(|id| self.output_picture_for_id(id))
            .collect()
    }

    fn resolve_and_release_pictures(
        &mut self,
        ids: Vec<H264PictureId>,
    ) -> Result<Vec<H264422P10OutputPicture>, H264422P10Error> {
        ids.into_iter()
            .map(|id| {
                let output = self.output_picture_for_id(id.clone())?;
                self.pictures.remove(&picture_key(&id));
                Ok(output)
            })
            .collect()
    }

    fn output_picture_for_id(
        &self,
        id: H264PictureId,
    ) -> Result<H264422P10OutputPicture, H264422P10Error> {
        let job =
            self.pictures
                .get(&picture_key(&id))
                .ok_or(H264422P10Error::InvalidDecodeSchedule(
                    "output picture was not submitted",
                ))?;
        Ok(H264422P10OutputPicture {
            decode_index: job.decode_index,
            picture_id: id,
        })
    }
}

#[derive(Debug)]
pub struct H264422P10AccessUnitInspector {
    state: H264DecoderState,
    expected: H264422P10Profile,
    next_input_index: u64,
}

impl Default for H264422P10AccessUnitInspector {
    fn default() -> Self {
        Self::new(H264422P10Profile::default())
    }
}

impl H264422P10AccessUnitInspector {
    pub fn new(expected: H264422P10Profile) -> Self {
        Self {
            state: H264DecoderState::new(),
            expected,
            next_input_index: 0,
        }
    }

    pub fn inspect_access_unit(
        &mut self,
        access_unit: &[u8],
    ) -> Result<H264422P10AccessUnitReport, H264422P10Error> {
        let parsed = self.state.parse_access_unit(access_unit)?;
        validate_original_profile(&parsed, &self.expected)?;
        let presentation_index = self.next_input_index;
        self.next_input_index = self.next_input_index.saturating_add(1);
        Ok(access_unit_report(
            presentation_index,
            parsed,
            &self.expected,
        ))
    }
}

#[derive(Debug)]
pub enum H264422P10Error {
    H264(qgs_codec_h264::H264Error),
    UnsupportedProfile(&'static str),
    InvalidOutputLayout(&'static str),
    InvalidDecodeSchedule(&'static str),
}

impl fmt::Display for H264422P10Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::H264(error) => write!(f, "{error}"),
            Self::UnsupportedProfile(reason) => write!(
                f,
                "unsupported H.264 4:2:2 10-bit original profile: {reason}"
            ),
            Self::InvalidOutputLayout(reason) => {
                write!(f, "invalid yuv422p10le output layout: {reason}")
            }
            Self::InvalidDecodeSchedule(reason) => {
                write!(f, "invalid H.264 4:2:2 10-bit decode schedule: {reason}")
            }
        }
    }
}

impl std::error::Error for H264422P10Error {}

impl From<qgs_codec_h264::H264Error> for H264422P10Error {
    fn from(value: qgs_codec_h264::H264Error) -> Self {
        Self::H264(value)
    }
}

pub fn validate_original_profile(
    parsed: &ParsedH264AccessUnit,
    expected: &H264422P10Profile,
) -> Result<(), H264422P10Error> {
    let desc = &parsed.desc;
    if parsed.profile != expected.profile {
        return Err(H264422P10Error::UnsupportedProfile(
            "profile is not High 4:2:2",
        ));
    }
    if desc.bit_depth.get() != expected.bit_depth {
        return Err(H264422P10Error::UnsupportedProfile(
            "bit depth is not 10-bit",
        ));
    }
    if desc.chroma != expected.chroma {
        return Err(H264422P10Error::UnsupportedProfile("chroma is not 4:2:2"));
    }
    if desc.format != VideoSurfaceFormat::Yuv422_10 {
        return Err(H264422P10Error::UnsupportedProfile(
            "surface format is not YUV422 10-bit",
        ));
    }
    if desc.coded_width != expected.coded_width || desc.coded_height != expected.coded_height {
        return Err(H264422P10Error::UnsupportedProfile(
            "coded dimensions are not Sony FX6 original dimensions",
        ));
    }
    if desc.visible_region.width != expected.visible_width
        || desc.visible_region.height != expected.visible_height
    {
        return Err(H264422P10Error::UnsupportedProfile(
            "visible dimensions are not Sony FX6 original dimensions",
        ));
    }
    if desc.scan_mode != expected.scan_mode || desc.field_order != FieldOrder::Unknown {
        return Err(H264422P10Error::UnsupportedProfile(
            "stream is not progressive frame video",
        ));
    }
    Ok(())
}

pub fn expected_yuv422p10le_plane_shapes(coded_width: u32, coded_height: u32) -> [(u32, u32); 3] {
    [
        (coded_width, coded_height),
        (coded_width / 2, coded_height),
        (coded_width / 2, coded_height),
    ]
}

pub fn expected_yuv422p10le_owned_bytes(
    coded_width: u32,
    coded_height: u32,
) -> Result<usize, H264422P10Error> {
    expected_yuv422p10le_plane_shapes(coded_width, coded_height)
        .into_iter()
        .try_fold(0_usize, |total, (width, height)| {
            let stride = checked_stride_bytes(width)?;
            let bytes = checked_plane_bytes(stride, height)?;
            total
                .checked_add(bytes)
                .ok_or(H264422P10Error::InvalidOutputLayout(
                    "frame byte size overflow",
                ))
        })
}

fn access_unit_report(
    presentation_index: u64,
    parsed: ParsedH264AccessUnit,
    expected: &H264422P10Profile,
) -> H264422P10AccessUnitReport {
    H264422P10AccessUnitReport {
        presentation_index,
        profile: expected.clone(),
        idr: parsed.picture.idr_pic_flag,
        reference_picture: parsed.picture.reference_pic_flag,
        slice_kinds: parsed
            .slices
            .iter()
            .map(|slice| slice.kind.clone())
            .collect(),
        reference_count: parsed.reference_frames.len(),
        max_dpb_frames: parsed.max_dpb_frames,
        max_num_reorder_frames: parsed.max_num_reorder_frames,
    }
}

fn decode_job(decode_index: u64, parsed: &ParsedH264AccessUnit) -> H264422P10DecodeJob {
    H264422P10DecodeJob {
        decode_index,
        picture_id: parsed.picture.id(),
        idr: parsed.picture.idr_pic_flag,
        reference_picture: parsed.picture.reference_pic_flag,
        slice_kinds: parsed
            .slices
            .iter()
            .map(|slice| slice.kind.clone())
            .collect(),
    }
}

fn picture_key(id: &H264PictureId) -> (u16, i32) {
    (id.frame_num, id.poc)
}

fn validate_plane(
    label: &'static str,
    plane: &H264422P10Plane,
    width_samples: u32,
    height: u32,
) -> Result<(), H264422P10Error> {
    if plane.width_samples != width_samples || plane.height != height {
        return Err(H264422P10Error::InvalidOutputLayout(label));
    }
    let expected_stride = checked_stride_bytes(width_samples)?;
    if plane.stride_bytes != expected_stride {
        return Err(H264422P10Error::InvalidOutputLayout(label));
    }
    let expected_len = checked_plane_bytes(expected_stride, height)?;
    if plane.data.len() != expected_len {
        return Err(H264422P10Error::InvalidOutputLayout(label));
    }
    Ok(())
}

fn checked_stride_bytes(width_samples: u32) -> Result<usize, H264422P10Error> {
    let bytes = width_samples
        .checked_mul(2)
        .ok_or(H264422P10Error::InvalidOutputLayout("stride overflow"))?;
    usize::try_from(bytes).map_err(|_| H264422P10Error::InvalidOutputLayout("stride overflow"))
}

fn checked_plane_bytes(stride: usize, height: u32) -> Result<usize, H264422P10Error> {
    stride
        .checked_mul(
            usize::try_from(height)
                .map_err(|_| H264422P10Error::InvalidOutputLayout("plane height overflow"))?,
        )
        .ok_or(H264422P10Error::InvalidOutputLayout(
            "plane byte size overflow",
        ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use qgs_protocol::ScanMode;

    const PROFESSIONAL_LONG_GOP_FIXTURE: &[u8] =
        include_bytes!("../../../tests/fixtures/h264/professional-422-10bit-long-gop-128x72.h264");
    const PROFESSIONAL_IDR_FIXTURE: &[u8] =
        include_bytes!("../../../tests/fixtures/h264/professional-422-10bit-idr-128x72.h264");

    #[test]
    fn sony_fx6_original_profile_is_exact() {
        let profile = H264422P10Profile::sony_fx6_original();

        assert_eq!(profile.profile, H264Profile::High422);
        assert_eq!(profile.bit_depth, 10);
        assert_eq!(profile.chroma, ChromaSubsampling::Cs422);
        assert_eq!(profile.coded_width, 1920);
        assert_eq!(profile.coded_height, 1088);
        assert_eq!(profile.visible_width, 1920);
        assert_eq!(profile.visible_height, 1080);
    }

    #[test]
    fn yuv422p10le_plane_sizes_match_vulkan_upload_contract() {
        assert_eq!(
            expected_yuv422p10le_plane_shapes(1920, 1088),
            [(1920, 1088), (960, 1088), (960, 1088)]
        );
        assert_eq!(
            expected_yuv422p10le_owned_bytes(1920, 1088).unwrap(),
            8_355_840
        );
    }

    #[test]
    fn yuv422p10_frame_layout_accepts_exact_planes() {
        let y = H264422P10Plane {
            width_samples: 4,
            height: 2,
            stride_bytes: 8,
            data: vec![0; 16],
        };
        let chroma = H264422P10Plane {
            width_samples: 2,
            height: 2,
            stride_bytes: 4,
            data: vec![0; 8],
        };
        let frame = H264422P10Frame {
            presentation_index: 0,
            coded_width: 4,
            coded_height: 2,
            visible_width: 4,
            visible_height: 2,
            y,
            cb: chroma.clone(),
            cr: chroma,
        };

        frame.validate_layout().unwrap();
        assert_eq!(frame.owned_bytes(), 32);
    }

    #[test]
    fn yuv422p10_frame_layout_rejects_bad_chroma_stride() {
        let y = H264422P10Plane {
            width_samples: 4,
            height: 2,
            stride_bytes: 8,
            data: vec![0; 16],
        };
        let cb = H264422P10Plane {
            width_samples: 2,
            height: 2,
            stride_bytes: 6,
            data: vec![0; 12],
        };
        let cr = H264422P10Plane {
            width_samples: 2,
            height: 2,
            stride_bytes: 4,
            data: vec![0; 8],
        };
        let frame = H264422P10Frame {
            presentation_index: 0,
            coded_width: 4,
            coded_height: 2,
            visible_width: 4,
            visible_height: 2,
            y,
            cb,
            cr,
        };

        assert!(matches!(
            frame.validate_layout(),
            Err(H264422P10Error::InvalidOutputLayout("Cb"))
        ));
    }

    #[test]
    fn planner_tracks_long_gop_output_order_without_pixel_reconstruction() {
        let mut planner = H264422P10DecodePlanner::new(H264422P10Profile {
            coded_width: 128,
            coded_height: 80,
            visible_width: 128,
            visible_height: 72,
            scan_mode: ScanMode::Progressive,
            ..H264422P10Profile::sony_fx6_original()
        });
        let access_units = split_access_units_for_test(PROFESSIONAL_LONG_GOP_FIXTURE);
        assert_eq!(access_units.len(), 12);

        let mut decode_pocs = Vec::new();
        let mut output_pocs = Vec::new();
        let mut saw_b_slice = false;
        for access_unit in access_units {
            let step = planner
                .submit_access_unit(&access_unit)
                .expect("professional 4:2:2 10-bit AU scheduled");
            decode_pocs.push(step.job.picture_id.poc);
            saw_b_slice |= step
                .job
                .slice_kinds
                .iter()
                .any(|kind| *kind == H264SliceKind::B);
            output_pocs.extend(
                step.output_ready
                    .into_iter()
                    .map(|picture| picture.picture_id.poc),
            );
        }
        output_pocs.extend(
            planner
                .flush()
                .expect("professional 4:2:2 10-bit planner flush")
                .into_iter()
                .map(|picture| picture.picture_id.poc),
        );

        assert!(saw_b_slice);
        assert_eq!(decode_pocs.len(), 12);
        assert_eq!(output_pocs.len(), 12);
        assert_ne!(decode_pocs, output_pocs);
        assert!(output_pocs.windows(2).all(|pair| pair[0] <= pair[1]));
    }

    #[test]
    fn idr_fixture_reconstructs_yuv422p10le_planes() {
        let access_units = split_access_units_for_test(PROFESSIONAL_IDR_FIXTURE);
        assert_eq!(access_units.len(), 1);
        let profile = H264422P10Profile {
            profile: H264Profile::High422Intra,
            coded_width: 128,
            coded_height: 80,
            visible_width: 128,
            visible_height: 72,
            scan_mode: ScanMode::Progressive,
            ..H264422P10Profile::sony_fx6_original()
        };
        let mut decoder = H264422P10Decoder::new(profile);
        let frame = decoder
            .decode_access_unit(&access_units[0])
            .expect("IDR 4:2:2 10-bit access unit reconstructs one picture");
        frame.validate_layout().expect("yuv422p10le layout");
        assert_eq!(frame.coded_width, 128);
        assert_eq!(frame.coded_height, 80);
        assert_eq!(frame.visible_width, 128);
        assert_eq!(frame.visible_height, 72);
        assert_eq!(
            frame.owned_bytes(),
            expected_yuv422p10le_owned_bytes(128, 80).unwrap()
        );
        assert!(
            frame.y.data.iter().any(|byte| *byte != 0),
            "reconstructed luma plane is empty"
        );
    }

    #[test]
    fn long_gop_fixture_reconstructs_twelve_yuv422p10le_pictures() {
        let access_units = split_access_units_for_test(PROFESSIONAL_LONG_GOP_FIXTURE);
        assert_eq!(access_units.len(), 12);
        let profile = H264422P10Profile {
            coded_width: 128,
            coded_height: 80,
            visible_width: 128,
            visible_height: 72,
            scan_mode: ScanMode::Progressive,
            ..H264422P10Profile::sony_fx6_original()
        };
        let mut decoder = H264422P10Decoder::new(profile);
        let mut frames = Vec::new();
        for access_unit in &access_units {
            frames.extend(
                decoder
                    .submit_access_unit(access_unit)
                    .expect("access unit"),
            );
        }
        frames.extend(decoder.flush().expect("flush"));
        assert_eq!(frames.len(), 12);
        for frame in &frames {
            frame.validate_layout().expect("yuv422p10le layout");
            assert_eq!(
                frame.owned_bytes(),
                expected_yuv422p10le_owned_bytes(128, 80).unwrap()
            );
            assert!(frame.y.data.iter().any(|byte| *byte != 0));
        }
    }

    #[test]
    fn sony_profile_rejects_small_professional_fixture_dimensions() {
        let access_units = split_access_units_for_test(PROFESSIONAL_LONG_GOP_FIXTURE);
        let mut planner = H264422P10DecodePlanner::default();

        assert!(matches!(
            planner.submit_access_unit(&access_units[0]),
            Err(H264422P10Error::UnsupportedProfile(
                "coded dimensions are not Sony FX6 original dimensions"
            ))
        ));
    }

    fn split_access_units_for_test(data: &[u8]) -> Vec<Vec<u8>> {
        let nals = split_annex_b_nals_for_test(data);
        let mut access_units = Vec::new();
        let mut current = Vec::new();
        let mut seen_vcl = false;
        for nal in nals {
            let nal_type = nal[0] & 0x1f;
            let is_vcl = nal_type == 1 || nal_type == 5;
            if is_vcl && seen_vcl && !current.is_empty() {
                access_units.push(std::mem::take(&mut current));
            }
            current.extend_from_slice(&[0, 0, 0, 1]);
            current.extend_from_slice(nal);
            if is_vcl {
                seen_vcl = true;
            }
        }
        if !current.is_empty() {
            access_units.push(current);
        }
        access_units
    }

    fn split_annex_b_nals_for_test(data: &[u8]) -> Vec<&[u8]> {
        let mut starts = Vec::new();
        let mut index = 0_usize;
        while index + 3 <= data.len() {
            let start_code_len = if data[index..].starts_with(&[0, 0, 1]) {
                3
            } else if data[index..].starts_with(&[0, 0, 0, 1]) {
                4
            } else {
                index += 1;
                continue;
            };
            let nal_start = index + start_code_len;
            if nal_start < data.len() {
                starts.push(index);
            }
            index = nal_start;
        }
        starts.sort_unstable();
        starts.dedup();
        starts
            .iter()
            .enumerate()
            .map(|(position, start)| {
                let end = starts.get(position + 1).copied().unwrap_or(data.len());
                let nal_start = if data[*start..].starts_with(&[0, 0, 0, 1]) {
                    start + 4
                } else {
                    start + 3
                };
                data[nal_start..end].trim_ascii_end()
            })
            .filter(|nal| !nal.is_empty())
            .collect()
    }
}
