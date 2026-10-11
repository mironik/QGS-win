#![forbid(unsafe_code)]

use std::fmt;
use std::io;

use h264_reader::nal::pps::{PicParameterSet, SliceGroup};
use h264_reader::nal::slice::{
    DecRefPicMarking, FieldPic, MemoryManagementControlOperation, ModificationOfPicNums,
    NumRefIdxActive, PicOrderCountLsb, RefPicListModifications, SliceFamily, SliceHeader,
};
use h264_reader::nal::sps::{
    ChromaFormat, FrameMbsFlags, PicOrderCntType, Profile, ScalingLists8x8Resolved,
};
use h264_reader::nal::{Nal, RefNal, UnitType};
use h264_reader::rbsp::{BitRead, BitReaderError, Integer, Primitive};
use h264_reader::Context;
use qgs_protocol::{
    BitDepth, ChromaSubsampling, FieldOrder, H264Profile, ScanMode, VideoCodec, VideoProfile,
    VideoSurfaceDesc, VideoSurfaceFormat, VisibleRegion,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParsedH264AccessUnit {
    pub desc: VideoSurfaceDesc,
    pub profile: H264Profile,
    pub level_idc: u8,
    pub picture: ParsedH264Picture,
    pub slices: Vec<ParsedH264Slice>,
    pub reference_frames: Vec<ParsedH264Reference>,
    pub max_dpb_frames: usize,
    pub max_num_reorder_frames: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H264PictureId {
    pub frame_num: u16,
    pub poc: i32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParsedH264Reference {
    pub id: H264PictureId,
    pub frame_num: u16,
    pub top_field_order_cnt: i32,
    pub bottom_field_order_cnt: i32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H264DecodeUpdate {
    pub output_ready: Vec<H264PictureId>,
    pub released: Vec<H264PictureId>,
    pub max_dpb_occupancy: usize,
    pub max_output_pending: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum H264SliceKind {
    I,
    P,
    B,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParsedH264Picture {
    pub picture_width_in_mbs_minus1: u16,
    pub picture_height_in_mbs_minus1: u16,
    pub bit_depth_luma_minus8: u8,
    pub bit_depth_chroma_minus8: u8,
    pub num_ref_frames: u8,
    pub chroma_format_idc: u32,
    pub gaps_in_frame_num_value_allowed_flag: bool,
    pub frame_mbs_only_flag: bool,
    pub mb_adaptive_frame_field_flag: bool,
    pub direct_8x8_inference_flag: bool,
    pub log2_max_frame_num_minus4: u8,
    pub pic_order_cnt_type: u32,
    pub log2_max_pic_order_cnt_lsb_minus4: u32,
    pub delta_pic_order_always_zero_flag: bool,
    pub num_slice_groups_minus1: u8,
    pub slice_group_map_type: u8,
    pub slice_group_change_rate_minus1: u16,
    pub entropy_coding_mode_flag: bool,
    pub weighted_pred_flag: bool,
    pub weighted_bipred_idc: u8,
    pub transform_8x8_mode_flag: bool,
    pub field_pic_flag: bool,
    pub constrained_intra_pred_flag: bool,
    pub pic_order_present_flag: bool,
    pub deblocking_filter_control_present_flag: bool,
    pub redundant_pic_cnt_present_flag: bool,
    pub reference_pic_flag: bool,
    pub idr_pic_flag: bool,
    pub no_output_of_prior_pics_flag: bool,
    pub long_term_reference_flag: bool,
    dec_ref_pic_marking: Option<ParsedDecRefPicMarking>,
    pub pic_init_qp_minus26: i8,
    pub pic_init_qs_minus26: i8,
    pub chroma_qp_index_offset: i8,
    pub second_chroma_qp_index_offset: i8,
    pub frame_num: u16,
    pub top_field_order_cnt: i32,
    pub bottom_field_order_cnt: i32,
    pub scaling_list4x4: [[u8; 16]; 6],
    pub scaling_list8x8: [[u8; 64]; 2],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParsedH264Slice {
    pub nal_bytes: Vec<u8>,
    pub nal_ref_idc: u8,
    pub idr: bool,
    pub kind: H264SliceKind,
    pub slice_data_bit_offset: u16,
    pub first_mb_in_slice: u16,
    pub slice_type: u8,
    pub direct_spatial_mv_pred_flag: u8,
    pub num_ref_idx_l0_active_minus1: u8,
    pub num_ref_idx_l1_active_minus1: u8,
    pub cabac_init_idc: u8,
    pub slice_qp_delta: i8,
    pub disable_deblocking_filter_idc: u8,
    pub slice_alpha_c0_offset_div2: i8,
    pub slice_beta_offset_div2: i8,
    pub ref_pic_list0: Vec<H264PictureId>,
    pub ref_pic_list1: Vec<H264PictureId>,
}

#[derive(Clone, Debug)]
struct DpbPicture {
    id: H264PictureId,
    frame_num: u16,
    top_field_order_cnt: i32,
    bottom_field_order_cnt: i32,
    reference: bool,
    output_needed: bool,
}

#[derive(Clone, Debug)]
struct ParsedSliceHeaderForState {
    frame_num: u16,
    idr_pic_id: Option<u32>,
    pic_order_cnt_lsb: u32,
    delta_pic_order_cnt_bottom: i32,
    dec_ref_pic_marking: Option<ParsedDecRefPicMarking>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ParsedDecRefPicMarking {
    Idr {
        no_output_of_prior_pics_flag: bool,
        long_term_reference_flag: bool,
    },
    SlidingWindow,
    Adaptive(Vec<ParsedMemoryManagementControlOperation>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ParsedMemoryManagementControlOperation {
    ShortTermUnusedForRef { difference_of_pic_nums_minus1: u32 },
    Unsupported,
    AllRefPicturesUnused,
}

#[derive(Debug)]
pub struct H264DecoderState {
    context: Context,
    pic_order_cnt_type: u32,
    max_pic_order_cnt_lsb: i32,
    max_frame_num: u32,
    prev_pic_order_cnt_msb: i32,
    prev_pic_order_cnt_lsb: i32,
    max_num_ref_frames: usize,
    max_num_reorder_frames: usize,
    dpb: Vec<DpbPicture>,
    max_dpb_occupancy: usize,
    max_output_pending: usize,
}

impl Default for H264DecoderState {
    fn default() -> Self {
        Self::new()
    }
}

impl H264DecoderState {
    pub fn new() -> Self {
        Self {
            context: Context::new(),
            pic_order_cnt_type: 0,
            max_pic_order_cnt_lsb: 0,
            max_frame_num: 0,
            prev_pic_order_cnt_msb: 0,
            prev_pic_order_cnt_lsb: 0,
            max_num_ref_frames: 0,
            max_num_reorder_frames: 0,
            dpb: Vec::new(),
            max_dpb_occupancy: 0,
            max_output_pending: 0,
        }
    }

    pub fn parse_access_unit(&mut self, data: &[u8]) -> Result<ParsedH264AccessUnit, H264Error> {
        parse_annex_b_access_unit_with_state(data, self)
    }

    pub fn finish_picture(
        &mut self,
        parsed: &ParsedH264AccessUnit,
    ) -> Result<H264DecodeUpdate, H264Error> {
        let mut output_ready = Vec::new();
        let mut released = Vec::new();
        if parsed.picture.idr_pic_flag {
            if parsed.picture.no_output_of_prior_pics_flag {
                released.extend(self.dpb.iter().map(|picture| picture.id.clone()));
            } else {
                let (mut ready, mut prior_released) = self.drain_ready(true);
                output_ready.append(&mut ready);
                released.append(&mut prior_released);
                released.extend(self.dpb.iter().map(|picture| picture.id.clone()));
            }
            self.dpb.clear();
        }

        apply_reference_marking(self, parsed)?;
        self.dpb.push(DpbPicture {
            id: parsed.picture.id(),
            frame_num: parsed.picture.frame_num,
            top_field_order_cnt: parsed.picture.top_field_order_cnt,
            bottom_field_order_cnt: parsed.picture.bottom_field_order_cnt,
            reference: parsed.picture.reference_pic_flag,
            output_needed: true,
        });
        self.max_dpb_occupancy = self.max_dpb_occupancy.max(self.dpb.len());
        self.max_output_pending = self.max_output_pending.max(
            self.dpb
                .iter()
                .filter(|picture| picture.output_needed)
                .count(),
        );
        let (mut ready, mut newly_released) = self.drain_ready(false);
        output_ready.append(&mut ready);
        released.append(&mut newly_released);
        Ok(H264DecodeUpdate {
            output_ready,
            released,
            max_dpb_occupancy: self.max_dpb_occupancy,
            max_output_pending: self.max_output_pending,
        })
    }

    pub fn flush(&mut self) -> H264DecodeUpdate {
        let (output_ready, released) = self.drain_ready(true);
        H264DecodeUpdate {
            output_ready,
            released,
            max_dpb_occupancy: self.max_dpb_occupancy,
            max_output_pending: self.max_output_pending,
        }
    }

    pub fn max_dpb_occupancy(&self) -> usize {
        self.max_dpb_occupancy
    }

    pub fn max_output_pending(&self) -> usize {
        self.max_output_pending
    }

    fn drain_ready(&mut self, flush: bool) -> (Vec<H264PictureId>, Vec<H264PictureId>) {
        let mut output_ready = Vec::new();
        loop {
            let pending = self
                .dpb
                .iter()
                .filter(|picture| picture.output_needed)
                .count();
            if pending == 0 || (!flush && pending <= self.max_num_reorder_frames) {
                break;
            }
            let Some(index) = self
                .dpb
                .iter()
                .enumerate()
                .filter(|(_, picture)| picture.output_needed)
                .min_by_key(|(_, picture)| picture.id.poc)
                .map(|(index, _)| index)
            else {
                break;
            };
            self.dpb[index].output_needed = false;
            output_ready.push(self.dpb[index].id.clone());
        }

        let mut released = Vec::new();
        let mut index = 0;
        while index < self.dpb.len() {
            if !self.dpb[index].reference && !self.dpb[index].output_needed {
                released.push(self.dpb[index].id.clone());
                self.dpb.remove(index);
            } else {
                index += 1;
            }
        }
        (output_ready, released)
    }
}

impl ParsedH264Picture {
    pub fn id(&self) -> H264PictureId {
        H264PictureId {
            frame_num: self.frame_num,
            poc: self.top_field_order_cnt,
        }
    }
}

#[derive(Debug)]
pub enum H264Error {
    MalformedAnnexB,
    MissingSps,
    MissingPps,
    MissingSlice,
    UnsupportedFeature(&'static str),
    Parser(String),
}

impl fmt::Display for H264Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MalformedAnnexB => write!(f, "malformed Annex B H.264 data"),
            Self::MissingSps => write!(f, "H.264 access unit is missing SPS"),
            Self::MissingPps => write!(f, "H.264 access unit is missing PPS"),
            Self::MissingSlice => write!(f, "H.264 access unit is missing a slice"),
            Self::UnsupportedFeature(feature) => write!(f, "unsupported H.264 feature: {feature}"),
            Self::Parser(error) => write!(f, "H.264 parser error: {error}"),
        }
    }
}

impl std::error::Error for H264Error {}

pub fn parse_annex_b_access_unit(data: &[u8]) -> Result<ParsedH264AccessUnit, H264Error> {
    let mut state = H264DecoderState::new();
    state.parse_access_unit(data)
}

fn parse_annex_b_access_unit_with_state(
    data: &[u8],
    state: &mut H264DecoderState,
) -> Result<ParsedH264AccessUnit, H264Error> {
    let nals = split_annex_b(data)?;
    let mut parsed_sps = None;
    let mut parsed_pps = None;
    let mut slices = Vec::new();
    let mut first_slice_header = None;
    let mut first_slice_sps = None;
    let mut first_slice_pps = None;

    for nal_bytes in nals {
        let nal = RefNal::new(nal_bytes, &[], true);
        let header = nal
            .header()
            .map_err(|error| H264Error::Parser(format!("{error:?}")))?;
        match header.nal_unit_type() {
            UnitType::SeqParameterSet => {
                let sps = h264_reader::nal::sps::SeqParameterSet::from_bits(nal.rbsp_bits())
                    .map_err(|error| H264Error::Parser(format!("{error:?}")))?;
                state.context.put_seq_param_set(sps.clone());
                parsed_sps = Some(sps);
            }
            UnitType::PicParameterSet => {
                let pps = PicParameterSet::from_bits(&state.context, nal.rbsp_bits())
                    .map_err(|error| H264Error::Parser(format!("{error:?}")))?;
                state.context.put_pic_param_set(pps.clone());
                parsed_pps = Some(pps);
            }
            UnitType::SliceLayerWithoutPartitioningIdr
            | UnitType::SliceLayerWithoutPartitioningNonIdr => {
                let mut reader = CountingBitReader::new(nal.rbsp_bits());
                let (slice_header, sps, pps) =
                    SliceHeader::from_bits(&state.context, &mut reader, header, None)
                        .map_err(|error| H264Error::Parser(format!("{error:?}")))?;
                if first_slice_header.is_none() {
                    first_slice_header = Some(slice_header_for_state(&slice_header));
                    first_slice_sps = Some(sps.clone());
                    first_slice_pps = Some(pps.clone());
                }
                slices.push(parsed_slice(
                    nal_bytes,
                    header.nal_ref_idc(),
                    &slice_header,
                    &state.dpb,
                    1_u32 << u32::from(sps.log2_max_frame_num()),
                    reader.bits_read(),
                    pps.num_ref_idx_l0_default_active_minus1,
                    pps.num_ref_idx_l1_default_active_minus1,
                )?);
            }
            UnitType::AccessUnitDelimiter
            | UnitType::SEI
            | UnitType::EndOfSeq
            | UnitType::EndOfStream => {}
            _ => return Err(H264Error::UnsupportedFeature("unsupported NAL unit type")),
        }
    }

    let sps = parsed_sps
        .or(first_slice_sps)
        .ok_or(H264Error::MissingSps)?;
    let pps = parsed_pps
        .or(first_slice_pps)
        .ok_or(H264Error::MissingPps)?;
    let first_header = first_slice_header.ok_or(H264Error::MissingSlice)?;
    if slices.is_empty() {
        return Err(H264Error::MissingSlice);
    }

    validate_supported_sps_pps(&sps, &pps)?;
    let desc = surface_desc_from_sps(&sps)?;
    update_state_from_sps(state, &sps)?;
    let (top_field_order_cnt, bottom_field_order_cnt) =
        calculate_poc(state, &sps, &pps, &first_header, slices[0].nal_ref_idc)?;
    let picture = parsed_picture(
        &sps,
        &pps,
        &slices[0],
        &first_header,
        top_field_order_cnt,
        bottom_field_order_cnt,
    )?;
    let reference_frames = state
        .dpb
        .iter()
        .filter(|picture| picture.reference)
        .map(|picture| ParsedH264Reference {
            id: picture.id.clone(),
            frame_num: picture.frame_num,
            top_field_order_cnt: picture.top_field_order_cnt,
            bottom_field_order_cnt: picture.bottom_field_order_cnt,
        })
        .collect();

    Ok(ParsedH264AccessUnit {
        desc,
        profile: map_profile(sps.profile())?,
        level_idc: sps.level_idc,
        picture,
        slices,
        reference_frames,
        max_dpb_frames: state.max_num_ref_frames,
        max_num_reorder_frames: state.max_num_reorder_frames,
    })
}

fn parsed_slice(
    nal_bytes: &[u8],
    nal_ref_idc: u8,
    header: &SliceHeader,
    dpb: &[DpbPicture],
    max_frame_num: u32,
    bits_after_nal_header: u32,
    default_num_ref_idx_l0_active_minus1: u32,
    default_num_ref_idx_l1_active_minus1: u32,
) -> Result<ParsedH264Slice, H264Error> {
    if header.field_pic != FieldPic::Frame {
        return Err(H264Error::UnsupportedFeature("field pictures"));
    }
    let kind = match header.slice_type.family {
        SliceFamily::I => H264SliceKind::I,
        SliceFamily::P => H264SliceKind::P,
        SliceFamily::B => H264SliceKind::B,
        SliceFamily::SP | SliceFamily::SI => {
            return Err(H264Error::UnsupportedFeature("SP/SI slices"));
        }
    };
    let offset = 8_u32
        .checked_add(bits_after_nal_header)
        .ok_or(H264Error::UnsupportedFeature("oversized slice header"))?;
    let (num_ref_idx_l0_active_minus1, num_ref_idx_l1_active_minus1) = num_ref_indices(
        header,
        &kind,
        default_num_ref_idx_l0_active_minus1,
        default_num_ref_idx_l1_active_minus1,
    )?;
    let current_poc = match header.pic_order_cnt_lsb {
        Some(PicOrderCountLsb::Frame(value))
        | Some(PicOrderCountLsb::FieldsAbsolute {
            pic_order_cnt_lsb: value,
            ..
        }) => i32::try_from(value).unwrap_or(i32::MAX),
        Some(PicOrderCountLsb::FieldsDelta(_)) | None => 0,
    };
    let (ref_pic_list0, ref_pic_list1) =
        reference_lists(dpb, &kind, current_poc, header, max_frame_num)?;
    Ok(ParsedH264Slice {
        nal_bytes: nal_bytes.to_vec(),
        nal_ref_idc,
        idr: header.idr_pic_id.is_some(),
        kind,
        slice_data_bit_offset: u16::try_from(offset)
            .map_err(|_| H264Error::UnsupportedFeature("oversized slice header"))?,
        first_mb_in_slice: u16::try_from(header.first_mb_in_slice)
            .map_err(|_| H264Error::UnsupportedFeature("large first_mb_in_slice"))?,
        slice_type: va_slice_type(&header.slice_type.family)?,
        direct_spatial_mv_pred_flag: header.direct_spatial_mv_pred_flag.unwrap_or(false) as u8,
        num_ref_idx_l0_active_minus1,
        num_ref_idx_l1_active_minus1,
        cabac_init_idc: header.cabac_init_idc.unwrap_or(0) as u8,
        slice_qp_delta: i8::try_from(header.slice_qp_delta)
            .map_err(|_| H264Error::UnsupportedFeature("slice_qp_delta out of range"))?,
        disable_deblocking_filter_idc: header.disable_deblocking_filter_idc,
        slice_alpha_c0_offset_div2: i8::try_from(header.slice_alpha_c0_offset_div2.unwrap_or(0))
            .map_err(|_| H264Error::UnsupportedFeature("deblocking alpha out of range"))?,
        slice_beta_offset_div2: i8::try_from(header.slice_beta_offset_div2.unwrap_or(0))
            .map_err(|_| H264Error::UnsupportedFeature("deblocking beta out of range"))?,
        ref_pic_list0,
        ref_pic_list1,
    })
}

fn va_slice_type(family: &SliceFamily) -> Result<u8, H264Error> {
    match family {
        SliceFamily::P => Ok(0),
        SliceFamily::B => Ok(1),
        SliceFamily::I => Ok(2),
        SliceFamily::SP | SliceFamily::SI => Err(H264Error::UnsupportedFeature("SP/SI slices")),
    }
}

fn num_ref_indices(
    header: &SliceHeader,
    kind: &H264SliceKind,
    default_l0_active_minus1: u32,
    default_l1_active_minus1: u32,
) -> Result<(u8, u8), H264Error> {
    let (l0, l1) = match (&header.num_ref_idx_active, kind) {
        (
            Some(NumRefIdxActive::P {
                num_ref_idx_l0_active_minus1,
            }),
            H264SliceKind::P,
        ) => (*num_ref_idx_l0_active_minus1, 0),
        (
            Some(NumRefIdxActive::B {
                num_ref_idx_l0_active_minus1,
                num_ref_idx_l1_active_minus1,
            }),
            H264SliceKind::B,
        ) => (*num_ref_idx_l0_active_minus1, *num_ref_idx_l1_active_minus1),
        (_, H264SliceKind::I) => (0, 0),
        (None, H264SliceKind::P) => (default_l0_active_minus1, 0),
        (None, H264SliceKind::B) => (default_l0_active_minus1, default_l1_active_minus1),
        _ => return Err(H264Error::UnsupportedFeature("reference index syntax")),
    };
    Ok((
        u8::try_from(l0).map_err(|_| H264Error::UnsupportedFeature("too many L0 references"))?,
        u8::try_from(l1).map_err(|_| H264Error::UnsupportedFeature("too many L1 references"))?,
    ))
}

fn reference_lists(
    dpb: &[DpbPicture],
    kind: &H264SliceKind,
    current_poc_lsb_only: i32,
    header: &SliceHeader,
    max_frame_num: u32,
) -> Result<(Vec<H264PictureId>, Vec<H264PictureId>), H264Error> {
    let mut refs = dpb
        .iter()
        .filter(|picture| picture.reference)
        .cloned()
        .collect::<Vec<_>>();
    let lists = match kind {
        H264SliceKind::I => (Vec::new(), Vec::new()),
        H264SliceKind::P => {
            refs.sort_by_key(|picture| {
                std::cmp::Reverse(short_term_pic_num(
                    picture.frame_num,
                    header.frame_num,
                    max_frame_num,
                ))
            });
            let mut list0 = refs.into_iter().map(|picture| picture.id).collect();
            apply_ref_modifications_l0(&mut list0, dpb, header, max_frame_num, header.frame_num)?;
            (list0, Vec::new())
        }
        H264SliceKind::B => {
            let mut l0 = refs.clone();
            l0.sort_by(|left, right| {
                match (
                    left.id.poc < current_poc_lsb_only,
                    right.id.poc < current_poc_lsb_only,
                ) {
                    (true, true) => right.id.poc.cmp(&left.id.poc),
                    (false, false) => left.id.poc.cmp(&right.id.poc),
                    (true, false) => std::cmp::Ordering::Less,
                    (false, true) => std::cmp::Ordering::Greater,
                }
            });
            refs.sort_by(|left, right| {
                match (
                    left.id.poc > current_poc_lsb_only,
                    right.id.poc > current_poc_lsb_only,
                ) {
                    (true, true) => left.id.poc.cmp(&right.id.poc),
                    (false, false) => right.id.poc.cmp(&left.id.poc),
                    (true, false) => std::cmp::Ordering::Less,
                    (false, true) => std::cmp::Ordering::Greater,
                }
            });
            let mut list0 = l0.into_iter().map(|picture| picture.id).collect::<Vec<_>>();
            let mut list1 = refs
                .into_iter()
                .map(|picture| picture.id)
                .collect::<Vec<_>>();
            if list0 == list1 && list1.len() > 1 {
                list1.swap(0, 1);
            }
            apply_ref_modifications_l0(&mut list0, dpb, header, max_frame_num, header.frame_num)?;
            apply_ref_modifications_l1(&mut list1, dpb, header, max_frame_num, header.frame_num)?;
            (std::mem::take(&mut list0), list1)
        }
    };
    Ok(lists)
}

fn apply_ref_modifications_l0(
    list: &mut Vec<H264PictureId>,
    dpb: &[DpbPicture],
    header: &SliceHeader,
    max_frame_num: u32,
    curr_pic_num: u16,
) -> Result<(), H264Error> {
    match &header.ref_pic_list_modification {
        Some(RefPicListModifications::P {
            ref_pic_list_modification_l0,
        })
        | Some(RefPicListModifications::B {
            ref_pic_list_modification_l0,
            ..
        }) => apply_short_term_modifications(
            list,
            dpb,
            ref_pic_list_modification_l0,
            max_frame_num,
            curr_pic_num,
        ),
        Some(RefPicListModifications::I) | None => Ok(()),
    }
}

fn apply_ref_modifications_l1(
    list: &mut Vec<H264PictureId>,
    dpb: &[DpbPicture],
    header: &SliceHeader,
    max_frame_num: u32,
    curr_pic_num: u16,
) -> Result<(), H264Error> {
    match &header.ref_pic_list_modification {
        Some(RefPicListModifications::B {
            ref_pic_list_modification_l1,
            ..
        }) => apply_short_term_modifications(
            list,
            dpb,
            ref_pic_list_modification_l1,
            max_frame_num,
            curr_pic_num,
        ),
        Some(RefPicListModifications::P { .. }) | Some(RefPicListModifications::I) | None => Ok(()),
    }
}

fn apply_short_term_modifications(
    list: &mut Vec<H264PictureId>,
    dpb: &[DpbPicture],
    modifications: &[ModificationOfPicNums],
    max_frame_num: u32,
    curr_pic_num: u16,
) -> Result<(), H264Error> {
    let mut pic_num_pred = i32::from(curr_pic_num);
    let max_pic_num = i32::try_from(max_frame_num)
        .map_err(|_| H264Error::UnsupportedFeature("large MaxFrameNum"))?;
    for (ref_idx, modification) in modifications.iter().enumerate() {
        let pic_num_no_wrap = match modification {
            ModificationOfPicNums::Subtract(abs_diff_pic_num_minus1) => {
                let diff = i32::try_from(abs_diff_pic_num_minus1 + 1)
                    .map_err(|_| H264Error::UnsupportedFeature("large ref list modification"))?;
                let mut value = pic_num_pred - diff;
                if value < 0 {
                    value += max_pic_num;
                }
                value
            }
            ModificationOfPicNums::Add(abs_diff_pic_num_minus1) => {
                let diff = i32::try_from(abs_diff_pic_num_minus1 + 1)
                    .map_err(|_| H264Error::UnsupportedFeature("large ref list modification"))?;
                let mut value = pic_num_pred + diff;
                if value >= max_pic_num {
                    value -= max_pic_num;
                }
                value
            }
            ModificationOfPicNums::LongTermRef(_)
            | ModificationOfPicNums::SubtractViewIdx(_)
            | ModificationOfPicNums::AddViewIdx(_) => {
                return Err(H264Error::UnsupportedFeature(
                    "long-term or MVC reference list modification",
                ));
            }
        };
        pic_num_pred = pic_num_no_wrap;
        let pic_num = if pic_num_no_wrap > i32::from(curr_pic_num) {
            pic_num_no_wrap - max_pic_num
        } else {
            pic_num_no_wrap
        };
        let Some(reference) = dpb
            .iter()
            .filter(|picture| picture.reference)
            .find(|picture| {
                short_term_pic_num(picture.frame_num, curr_pic_num, max_frame_num) == pic_num
            })
            .map(|picture| picture.id.clone())
        else {
            return Err(H264Error::UnsupportedFeature(
                "reference list modification target not in DPB",
            ));
        };
        list.retain(|id| id != &reference);
        list.insert(ref_idx.min(list.len()), reference);
    }
    Ok(())
}

fn short_term_pic_num(frame_num: u16, curr_pic_num: u16, max_frame_num: u32) -> i32 {
    let frame_num = i32::from(frame_num);
    let curr_pic_num = i32::from(curr_pic_num);
    let max_frame_num = i32::try_from(max_frame_num).unwrap_or(i32::MAX);
    if frame_num > curr_pic_num {
        frame_num - max_frame_num
    } else {
        frame_num
    }
}

fn validate_supported_sps_pps(
    sps: &h264_reader::nal::sps::SeqParameterSet,
    pps: &PicParameterSet,
) -> Result<(), H264Error> {
    match sps.profile() {
        Profile::Baseline
        | Profile::ConstrainedBaseline
        | Profile::Main
        | Profile::High
        | Profile::ProgressiveHigh
        | Profile::ConstrainedHigh
        | Profile::High10
        | Profile::High10Intra
        | Profile::High422
        | Profile::High422Intra => {}
        _ => return Err(H264Error::UnsupportedFeature("H.264 profile")),
    }
    if sps.chroma_info.bit_depth_luma_minus8 != sps.chroma_info.bit_depth_chroma_minus8 {
        return Err(H264Error::UnsupportedFeature(
            "different luma/chroma bit depths",
        ));
    }
    if !matches!(sps.chroma_info.bit_depth_luma_minus8, 0 | 2) {
        return Err(H264Error::UnsupportedFeature("unsupported H.264 bit depth"));
    }
    if !matches!(
        sps.chroma_info.chroma_format,
        ChromaFormat::YUV420 | ChromaFormat::YUV422
    ) {
        return Err(H264Error::UnsupportedFeature(
            "unsupported H.264 chroma format",
        ));
    }
    if sps.chroma_info.separate_colour_plane_flag {
        return Err(H264Error::UnsupportedFeature("separate colour plane"));
    }
    if !matches!(sps.frame_mbs_flags, FrameMbsFlags::Frames) {
        return Err(H264Error::UnsupportedFeature("interlaced or MBAFF stream"));
    }
    if pps.slice_groups.is_some() {
        return Err(H264Error::UnsupportedFeature("slice groups"));
    }
    Ok(())
}

fn update_state_from_sps(
    state: &mut H264DecoderState,
    sps: &h264_reader::nal::sps::SeqParameterSet,
) -> Result<(), H264Error> {
    let log2_max_pic_order_cnt_lsb_minus4 = match sps.pic_order_cnt {
        PicOrderCntType::TypeZero {
            log2_max_pic_order_cnt_lsb_minus4,
        } => {
            state.pic_order_cnt_type = 0;
            log2_max_pic_order_cnt_lsb_minus4
        }
        PicOrderCntType::TypeOne { .. } => {
            return Err(H264Error::UnsupportedFeature("POC type 1"));
        }
        PicOrderCntType::TypeTwo => {
            state.pic_order_cnt_type = 2;
            0
        }
    };
    state.max_pic_order_cnt_lsb = 1_i32
        .checked_shl(u32::from(log2_max_pic_order_cnt_lsb_minus4) + 4)
        .ok_or(H264Error::UnsupportedFeature("POC LSB range"))?;
    state.max_num_ref_frames = usize::try_from(sps.max_num_ref_frames)
        .map_err(|_| H264Error::UnsupportedFeature("too many reference frames"))?;
    state.max_frame_num = 1_u32 << u32::from(sps.log2_max_frame_num());
    state.max_num_reorder_frames = sps
        .vui_parameters
        .as_ref()
        .and_then(|vui| vui.bitstream_restrictions.as_ref())
        .map(|restrictions| restrictions.max_num_reorder_frames as usize)
        .unwrap_or(0);
    Ok(())
}

fn slice_header_for_state(header: &SliceHeader) -> ParsedSliceHeaderForState {
    let (pic_order_cnt_lsb, delta_pic_order_cnt_bottom) = match header.pic_order_cnt_lsb {
        Some(PicOrderCountLsb::Frame(value)) => (value, 0),
        Some(PicOrderCountLsb::FieldsAbsolute {
            pic_order_cnt_lsb,
            delta_pic_order_cnt_bottom,
        }) => (pic_order_cnt_lsb, delta_pic_order_cnt_bottom),
        Some(PicOrderCountLsb::FieldsDelta(_)) | None => (0, 0),
    };
    ParsedSliceHeaderForState {
        frame_num: header.frame_num,
        idr_pic_id: header.idr_pic_id,
        pic_order_cnt_lsb,
        delta_pic_order_cnt_bottom,
        dec_ref_pic_marking: header
            .dec_ref_pic_marking
            .as_ref()
            .map(map_dec_ref_pic_marking),
    }
}

fn map_dec_ref_pic_marking(marking: &DecRefPicMarking) -> ParsedDecRefPicMarking {
    match marking {
        DecRefPicMarking::Idr {
            no_output_of_prior_pics_flag,
            long_term_reference_flag,
        } => ParsedDecRefPicMarking::Idr {
            no_output_of_prior_pics_flag: *no_output_of_prior_pics_flag,
            long_term_reference_flag: *long_term_reference_flag,
        },
        DecRefPicMarking::SlidingWindow => ParsedDecRefPicMarking::SlidingWindow,
        DecRefPicMarking::Adaptive(ops) => ParsedDecRefPicMarking::Adaptive(
            ops.iter()
                .map(|op| match op {
                    MemoryManagementControlOperation::ShortTermUnusedForRef {
                        difference_of_pic_nums_minus1,
                    } => ParsedMemoryManagementControlOperation::ShortTermUnusedForRef {
                        difference_of_pic_nums_minus1: *difference_of_pic_nums_minus1,
                    },
                    MemoryManagementControlOperation::AllRefPicturesUnused => {
                        ParsedMemoryManagementControlOperation::AllRefPicturesUnused
                    }
                    _ => ParsedMemoryManagementControlOperation::Unsupported,
                })
                .collect(),
        ),
    }
}

fn calculate_poc(
    state: &mut H264DecoderState,
    _sps: &h264_reader::nal::sps::SeqParameterSet,
    _pps: &PicParameterSet,
    header: &ParsedSliceHeaderForState,
    nal_ref_idc: u8,
) -> Result<(i32, i32), H264Error> {
    if state.pic_order_cnt_type == 2 {
        let poc = if header.idr_pic_id.is_some() {
            0
        } else if nal_ref_idc == 0 {
            2_i32
                .checked_mul(i32::from(header.frame_num))
                .and_then(|value| value.checked_sub(1))
                .ok_or(H264Error::UnsupportedFeature("POC type 2 overflow"))?
        } else {
            2_i32
                .checked_mul(i32::from(header.frame_num))
                .ok_or(H264Error::UnsupportedFeature("POC type 2 overflow"))?
        };
        return Ok((poc, poc));
    }
    if state.max_pic_order_cnt_lsb == 0 {
        return Err(H264Error::UnsupportedFeature("missing POC type 0 state"));
    }
    if header.idr_pic_id.is_some() {
        state.prev_pic_order_cnt_msb = 0;
        state.prev_pic_order_cnt_lsb = 0;
    }
    let pic_order_cnt_lsb = i32::try_from(header.pic_order_cnt_lsb)
        .map_err(|_| H264Error::UnsupportedFeature("large POC LSB"))?;
    let half = state.max_pic_order_cnt_lsb / 2;
    let pic_order_cnt_msb = if pic_order_cnt_lsb < state.prev_pic_order_cnt_lsb
        && state.prev_pic_order_cnt_lsb - pic_order_cnt_lsb >= half
    {
        state.prev_pic_order_cnt_msb + state.max_pic_order_cnt_lsb
    } else if pic_order_cnt_lsb > state.prev_pic_order_cnt_lsb
        && pic_order_cnt_lsb - state.prev_pic_order_cnt_lsb > half
    {
        state.prev_pic_order_cnt_msb - state.max_pic_order_cnt_lsb
    } else {
        state.prev_pic_order_cnt_msb
    };
    let top = pic_order_cnt_msb + pic_order_cnt_lsb;
    let bottom = top + header.delta_pic_order_cnt_bottom;
    if nal_ref_idc != 0 {
        state.prev_pic_order_cnt_msb = pic_order_cnt_msb;
        state.prev_pic_order_cnt_lsb = pic_order_cnt_lsb;
    }
    Ok((top, bottom))
}

fn apply_reference_marking(
    state: &mut H264DecoderState,
    parsed: &ParsedH264AccessUnit,
) -> Result<(), H264Error> {
    if parsed.picture.idr_pic_flag {
        if parsed.picture.long_term_reference_flag {
            return Err(H264Error::UnsupportedFeature("long-term IDR reference"));
        }
        return Ok(());
    }

    match &parsed.picture.dec_ref_pic_marking {
        Some(ParsedDecRefPicMarking::SlidingWindow) | None => {
            if parsed.picture.reference_pic_flag {
                let ref_count = state.dpb.iter().filter(|picture| picture.reference).count();
                if state.max_num_ref_frames > 0 && ref_count >= state.max_num_ref_frames {
                    if let Some(index) = state
                        .dpb
                        .iter()
                        .enumerate()
                        .filter(|(_, picture)| picture.reference)
                        .min_by_key(|(_, picture)| {
                            short_term_pic_num(
                                picture.frame_num,
                                parsed.picture.frame_num,
                                state.max_frame_num,
                            )
                        })
                        .map(|(index, _)| index)
                    {
                        state.dpb[index].reference = false;
                    }
                }
            }
        }
        Some(ParsedDecRefPicMarking::Adaptive(ops)) => {
            for op in ops {
                match op {
                    ParsedMemoryManagementControlOperation::ShortTermUnusedForRef {
                        difference_of_pic_nums_minus1,
                    } => {
                        let diff = i32::try_from(*difference_of_pic_nums_minus1 + 1)
                            .map_err(|_| H264Error::UnsupportedFeature("large MMCO diff"))?;
                        let max_frame_num = i32::try_from(state.max_frame_num)
                            .map_err(|_| H264Error::UnsupportedFeature("large MaxFrameNum"))?;
                        let mut pic_num = i32::from(parsed.picture.frame_num) - diff;
                        if pic_num < 0 {
                            pic_num += max_frame_num;
                        }
                        if let Some(picture) = state.dpb.iter_mut().find(|picture| {
                            short_term_pic_num(
                                picture.frame_num,
                                parsed.picture.frame_num,
                                state.max_frame_num,
                            ) == pic_num
                        }) {
                            picture.reference = false;
                        }
                    }
                    ParsedMemoryManagementControlOperation::AllRefPicturesUnused => {
                        for picture in &mut state.dpb {
                            picture.reference = false;
                        }
                    }
                    ParsedMemoryManagementControlOperation::Unsupported => {
                        return Err(H264Error::UnsupportedFeature("adaptive reference marking"));
                    }
                }
            }
        }
        Some(ParsedDecRefPicMarking::Idr { .. }) => {}
    }
    Ok(())
}

fn parsed_picture(
    sps: &h264_reader::nal::sps::SeqParameterSet,
    pps: &PicParameterSet,
    first_slice: &ParsedH264Slice,
    first_header: &ParsedSliceHeaderForState,
    top_field_order_cnt: i32,
    bottom_field_order_cnt: i32,
) -> Result<ParsedH264Picture, H264Error> {
    let (pic_order_cnt_type, log2_max_pic_order_cnt_lsb_minus4, delta_pic_order_always_zero_flag) =
        match sps.pic_order_cnt {
            PicOrderCntType::TypeZero {
                log2_max_pic_order_cnt_lsb_minus4,
            } => (0, u32::from(log2_max_pic_order_cnt_lsb_minus4), false),
            PicOrderCntType::TypeOne {
                delta_pic_order_always_zero_flag,
                ..
            } => (1, 0, delta_pic_order_always_zero_flag),
            PicOrderCntType::TypeTwo => (2, 0, false),
        };
    let transform_8x8_mode_flag = pps
        .extension
        .as_ref()
        .map(|extension| extension.transform_8x8_mode_flag)
        .unwrap_or(false);
    let second_chroma_qp_index_offset = pps
        .extension
        .as_ref()
        .map(|extension| extension.second_chroma_qp_index_offset)
        .unwrap_or(pps.chroma_qp_index_offset);

    Ok(ParsedH264Picture {
        picture_width_in_mbs_minus1: u16::try_from(sps.pic_width_in_mbs_minus1)
            .map_err(|_| H264Error::UnsupportedFeature("large width"))?,
        picture_height_in_mbs_minus1: u16::try_from(sps.pic_height_in_map_units_minus1)
            .map_err(|_| H264Error::UnsupportedFeature("large height"))?,
        bit_depth_luma_minus8: sps.chroma_info.bit_depth_luma_minus8,
        bit_depth_chroma_minus8: sps.chroma_info.bit_depth_chroma_minus8,
        num_ref_frames: u8::try_from(sps.max_num_ref_frames)
            .map_err(|_| H264Error::UnsupportedFeature("too many reference frames"))?,
        chroma_format_idc: sps.chroma_info.chroma_format.to_u32(),
        gaps_in_frame_num_value_allowed_flag: sps.gaps_in_frame_num_value_allowed_flag,
        frame_mbs_only_flag: true,
        mb_adaptive_frame_field_flag: false,
        direct_8x8_inference_flag: sps.direct_8x8_inference_flag,
        log2_max_frame_num_minus4: sps.log2_max_frame_num_minus4,
        pic_order_cnt_type,
        log2_max_pic_order_cnt_lsb_minus4,
        delta_pic_order_always_zero_flag,
        num_slice_groups_minus1: match &pps.slice_groups {
            None => 0,
            Some(SliceGroup::Interleaved { .. })
            | Some(SliceGroup::Dispersed { .. })
            | Some(SliceGroup::ForegroundAndLeftover { .. })
            | Some(SliceGroup::Changing { .. })
            | Some(SliceGroup::ExplicitAssignment { .. }) => {
                return Err(H264Error::UnsupportedFeature("slice groups"));
            }
        },
        slice_group_map_type: 0,
        slice_group_change_rate_minus1: 0,
        entropy_coding_mode_flag: pps.entropy_coding_mode_flag,
        weighted_pred_flag: pps.weighted_pred_flag,
        weighted_bipred_idc: pps.weighted_bipred_idc,
        transform_8x8_mode_flag,
        field_pic_flag: false,
        constrained_intra_pred_flag: pps.constrained_intra_pred_flag,
        pic_order_present_flag: pps.bottom_field_pic_order_in_frame_present_flag,
        deblocking_filter_control_present_flag: pps.deblocking_filter_control_present_flag,
        redundant_pic_cnt_present_flag: pps.redundant_pic_cnt_present_flag,
        reference_pic_flag: first_slice.nal_ref_idc != 0,
        idr_pic_flag: first_slice.idr,
        no_output_of_prior_pics_flag: matches!(
            &first_header.dec_ref_pic_marking,
            Some(ParsedDecRefPicMarking::Idr {
                no_output_of_prior_pics_flag: true,
                ..
            })
        ),
        long_term_reference_flag: matches!(
            &first_header.dec_ref_pic_marking,
            Some(ParsedDecRefPicMarking::Idr {
                long_term_reference_flag: true,
                ..
            })
        ),
        dec_ref_pic_marking: first_header.dec_ref_pic_marking.clone(),
        pic_init_qp_minus26: i8::try_from(pps.pic_init_qp_minus26)
            .map_err(|_| H264Error::UnsupportedFeature("pic_init_qp_minus26 out of range"))?,
        pic_init_qs_minus26: i8::try_from(pps.pic_init_qs_minus26)
            .map_err(|_| H264Error::UnsupportedFeature("pic_init_qs_minus26 out of range"))?,
        chroma_qp_index_offset: i8::try_from(pps.chroma_qp_index_offset)
            .map_err(|_| H264Error::UnsupportedFeature("chroma qp offset out of range"))?,
        second_chroma_qp_index_offset: i8::try_from(second_chroma_qp_index_offset)
            .map_err(|_| H264Error::UnsupportedFeature("second chroma qp offset out of range"))?,
        frame_num: first_header.frame_num,
        top_field_order_cnt,
        bottom_field_order_cnt,
        scaling_list4x4: scaling_list4x4(sps, pps),
        scaling_list8x8: scaling_list8x8(sps, pps),
    })
}

fn scaling_list4x4(
    sps: &h264_reader::nal::sps::SeqParameterSet,
    pps: &PicParameterSet,
) -> [[u8; 16]; 6] {
    pps.extension
        .as_ref()
        .and_then(|extension| extension.pic_scaling_matrix.as_ref())
        .map(|matrix| matrix.scaling_lists_4x4(&sps.chroma_info))
        .unwrap_or_else(|| sps.chroma_info.scaling_lists_4x4())
}

fn scaling_list8x8(
    sps: &h264_reader::nal::sps::SeqParameterSet,
    pps: &PicParameterSet,
) -> [[u8; 64]; 2] {
    let resolved = pps
        .extension
        .as_ref()
        .and_then(|extension| extension.pic_scaling_matrix.as_ref())
        .and_then(|matrix| matrix.scaling_lists_8x8(&sps.chroma_info))
        .unwrap_or_else(|| sps.chroma_info.scaling_lists_8x8());
    match resolved {
        ScalingLists8x8Resolved::Y(lists) => lists,
        ScalingLists8x8Resolved::YCbCr(lists) => [lists[0], lists[1]],
    }
}

fn surface_desc_from_sps(
    sps: &h264_reader::nal::sps::SeqParameterSet,
) -> Result<VideoSurfaceDesc, H264Error> {
    let (visible_width, visible_height) = sps
        .pixel_dimensions()
        .map_err(|error| H264Error::Parser(format!("{error:?}")))?;
    let (coded_width, coded_height) = coded_dimensions_from_sps(sps)?;
    let desc = VideoSurfaceDesc {
        coded_width,
        coded_height,
        visible_region: VisibleRegion {
            x: 0,
            y: 0,
            width: visible_width,
            height: visible_height,
        },
        format: surface_format_from_sps(sps)?,
        bit_depth: bit_depth_from_sps(sps)?,
        chroma: chroma_from_sps(sps)?,
        scan_mode: ScanMode::Progressive,
        field_order: FieldOrder::Unknown,
    };
    desc.validate()
        .map_err(|error| H264Error::Parser(error.to_string()))?;
    Ok(desc)
}

fn coded_dimensions_from_sps(
    sps: &h264_reader::nal::sps::SeqParameterSet,
) -> Result<(u32, u32), H264Error> {
    let coded_width = sps
        .pic_width_in_mbs_minus1
        .checked_add(1)
        .and_then(|width_in_mbs| width_in_mbs.checked_mul(16))
        .ok_or(H264Error::UnsupportedFeature("coded width overflow"))?;
    let height_multiplier = match sps.frame_mbs_flags {
        FrameMbsFlags::Frames => 1,
        FrameMbsFlags::Fields { .. } => {
            return Err(H264Error::UnsupportedFeature("interlaced or MBAFF stream"));
        }
    };
    let coded_height = sps
        .pic_height_in_map_units_minus1
        .checked_add(1)
        .and_then(|height_in_map_units| {
            height_in_map_units
                .checked_mul(height_multiplier)
                .and_then(|height_in_mbs| height_in_mbs.checked_mul(16))
        })
        .ok_or(H264Error::UnsupportedFeature("coded height overflow"))?;
    Ok((coded_width, coded_height))
}

fn bit_depth_from_sps(sps: &h264_reader::nal::sps::SeqParameterSet) -> Result<BitDepth, H264Error> {
    let value = sps
        .chroma_info
        .bit_depth_luma_minus8
        .checked_add(8)
        .ok_or(H264Error::UnsupportedFeature("H.264 bit depth overflow"))?;
    BitDepth::new(value).map_err(|error| H264Error::Parser(error.to_string()))
}

fn chroma_from_sps(
    sps: &h264_reader::nal::sps::SeqParameterSet,
) -> Result<ChromaSubsampling, H264Error> {
    match sps.chroma_info.chroma_format {
        ChromaFormat::YUV420 => Ok(ChromaSubsampling::Cs420),
        ChromaFormat::YUV422 => Ok(ChromaSubsampling::Cs422),
        ChromaFormat::YUV444 => Ok(ChromaSubsampling::Cs444),
        ChromaFormat::Monochrome | ChromaFormat::Invalid(_) => Err(H264Error::UnsupportedFeature(
            "unsupported H.264 chroma format",
        )),
    }
}

fn surface_format_from_sps(
    sps: &h264_reader::nal::sps::SeqParameterSet,
) -> Result<VideoSurfaceFormat, H264Error> {
    match (
        sps.chroma_info.chroma_format,
        sps.chroma_info.bit_depth_luma_minus8,
    ) {
        (ChromaFormat::YUV420, 0) => Ok(VideoSurfaceFormat::Nv12),
        (ChromaFormat::YUV420, 2) => Ok(VideoSurfaceFormat::P010),
        (ChromaFormat::YUV422, 0) => Ok(VideoSurfaceFormat::Yuv422_8),
        (ChromaFormat::YUV422, 2) => Ok(VideoSurfaceFormat::Yuv422_10),
        (ChromaFormat::YUV444, _) => Err(H264Error::UnsupportedFeature(
            "4:4:4 VideoSurface storage representation",
        )),
        _ => Err(H264Error::UnsupportedFeature(
            "unsupported H.264 surface representation",
        )),
    }
}

fn map_profile(profile: Profile) -> Result<H264Profile, H264Error> {
    match profile {
        Profile::Baseline | Profile::ConstrainedBaseline => Ok(H264Profile::Baseline),
        Profile::Main => Ok(H264Profile::Main),
        Profile::High | Profile::ProgressiveHigh | Profile::ConstrainedHigh => {
            Ok(H264Profile::High)
        }
        Profile::High10 => Ok(H264Profile::High10),
        Profile::High10Intra => Ok(H264Profile::High10Intra),
        Profile::High422 => Ok(H264Profile::High422),
        Profile::High422Intra => Ok(H264Profile::High422Intra),
        _ => Err(H264Error::UnsupportedFeature("H.264 profile")),
    }
}

pub fn decoder_profile(parsed: &ParsedH264AccessUnit) -> VideoProfile {
    VideoProfile::H264(parsed.profile)
}

pub const fn decoder_codec() -> VideoCodec {
    VideoCodec::H264
}

fn split_annex_b(data: &[u8]) -> Result<Vec<&[u8]>, H264Error> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 3 <= data.len() {
        if data[i..].starts_with(&[0, 0, 1]) {
            starts.push((i, 3));
            i += 3;
        } else if i + 4 <= data.len() && data[i..].starts_with(&[0, 0, 0, 1]) {
            starts.push((i, 4));
            i += 4;
        } else {
            i += 1;
        }
    }
    if starts.is_empty() {
        return Err(H264Error::MalformedAnnexB);
    }
    let mut nals = Vec::new();
    for (index, (start, prefix_len)) in starts.iter().copied().enumerate() {
        let payload_start = start + prefix_len;
        let payload_end = starts
            .get(index + 1)
            .map(|(next, _)| *next)
            .unwrap_or(data.len());
        if payload_start >= payload_end {
            return Err(H264Error::MalformedAnnexB);
        }
        nals.push(&data[payload_start..payload_end]);
    }
    Ok(nals)
}

struct CountingBitReader<R> {
    inner: R,
    bits_read: u32,
}

impl<R> CountingBitReader<R> {
    fn new(inner: R) -> Self {
        Self {
            inner,
            bits_read: 0,
        }
    }

    const fn bits_read(&self) -> u32 {
        self.bits_read
    }

    fn add_bits(&mut self, bits: u32) -> Result<(), BitReaderError> {
        self.bits_read = self
            .bits_read
            .checked_add(bits)
            .ok_or_else(|| io_error("bit counter overflow"))?;
        Ok(())
    }
}

impl<R: BitRead> BitRead for CountingBitReader<R> {
    fn read_ue(&mut self, name: &'static str) -> Result<u32, BitReaderError> {
        let mut zeros = 0_u32;
        while !self.read_bit(name)? {
            zeros += 1;
            if zeros > 31 {
                return Err(BitReaderError::ExpGolombTooLarge(name));
            }
        }
        if zeros == 0 {
            return Ok(0);
        }
        let suffix: u32 = self.read_var(zeros, name)?;
        Ok((1 << zeros) - 1 + suffix)
    }

    fn read_se(&mut self, name: &'static str) -> Result<i32, BitReaderError> {
        let code_num = self.read_ue(name)?;
        if code_num % 2 == 0 {
            Ok(-i32::try_from(code_num / 2).unwrap_or(i32::MAX))
        } else {
            Ok(i32::try_from(code_num.div_ceil(2)).unwrap_or(i32::MAX))
        }
    }

    fn read_bit(&mut self, name: &'static str) -> Result<bool, BitReaderError> {
        let value = self.inner.read_bit(name)?;
        self.add_bits(1)?;
        Ok(value)
    }

    fn read<const BITS: u32, I: Integer>(
        &mut self,
        name: &'static str,
    ) -> Result<I, BitReaderError> {
        let value = self.inner.read::<BITS, I>(name)?;
        self.add_bits(BITS)?;
        Ok(value)
    }

    fn read_var<I: Integer>(
        &mut self,
        bit_count: u32,
        name: &'static str,
    ) -> Result<I, BitReaderError> {
        let value = self.inner.read_var(bit_count, name)?;
        self.add_bits(bit_count)?;
        Ok(value)
    }

    fn read_to<V: Primitive>(&mut self, name: &'static str) -> Result<V, BitReaderError> {
        let value = self.inner.read_to(name)?;
        self.add_bits((std::mem::size_of::<V>() * 8) as u32)?;
        Ok(value)
    }

    fn skip(&mut self, bit_count: u32, name: &'static str) -> Result<(), BitReaderError> {
        self.inner.skip(bit_count, name)?;
        self.add_bits(bit_count)?;
        Ok(())
    }

    fn byte_aligned(&self) -> bool {
        self.inner.byte_aligned()
    }

    fn has_more_rbsp_data(&mut self, name: &'static str) -> Result<bool, BitReaderError> {
        self.inner.has_more_rbsp_data(name)
    }

    fn finish_rbsp(self) -> Result<(), BitReaderError> {
        self.inner.finish_rbsp()
    }

    fn finish_sei_payload(self) -> Result<(), BitReaderError> {
        self.inner.finish_sei_payload()
    }
}

fn io_error(message: &'static str) -> BitReaderError {
    BitReaderError::ReaderError(
        "qgs-codec-h264",
        io::Error::new(io::ErrorKind::InvalidData, message),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &[u8] = include_bytes!("../../../tests/fixtures/h264/idr-64x64-baseline.h264");
    const LONG_GOP_FIXTURE: &[u8] =
        include_bytes!("../../../tests/fixtures/h264/long-gop-128x72-main.h264");
    const PROFESSIONAL_INTRA_FIXTURE: &[u8] =
        include_bytes!("../../../tests/fixtures/h264/professional-422-10bit-idr-128x72.h264");
    const PROFESSIONAL_LONG_GOP_FIXTURE: &[u8] =
        include_bytes!("../../../tests/fixtures/h264/professional-422-10bit-long-gop-128x72.h264");

    #[test]
    fn parses_fixture_access_unit() {
        let parsed = parse_annex_b_access_unit(FIXTURE).expect("fixture parses");

        assert_eq!(parsed.profile, H264Profile::Baseline);
        assert_eq!(parsed.desc.coded_width, 64);
        assert_eq!(parsed.desc.coded_height, 64);
        assert_eq!(parsed.desc.format, VideoSurfaceFormat::Nv12);
        assert_eq!(parsed.desc.bit_depth.get(), 8);
        assert_eq!(parsed.desc.chroma, ChromaSubsampling::Cs420);
        assert_eq!(parsed.desc.scan_mode, ScanMode::Progressive);
        assert_eq!(parsed.slices.len(), 1);
        assert!(parsed.slices[0].idr);
        assert!(parsed.slices[0].slice_data_bit_offset > 8);
    }

    #[test]
    fn rejects_malformed_annex_b() {
        assert!(matches!(
            parse_annex_b_access_unit(&[1, 2, 3]),
            Err(H264Error::MalformedAnnexB)
        ));
    }

    #[test]
    fn parses_long_gop_p_and_b_slices() {
        let access_units = split_access_units_for_test(LONG_GOP_FIXTURE);
        assert_eq!(access_units.len(), 12);

        let mut state = H264DecoderState::new();
        let mut kinds = Vec::new();
        for access_unit in access_units {
            let parsed = state
                .parse_access_unit(&access_unit)
                .expect("Long-GOP AU parses");
            kinds.push(parsed.slices[0].kind.clone());
            state
                .finish_picture(&parsed)
                .expect("Long-GOP DPB update succeeds");
        }

        assert!(kinds.contains(&H264SliceKind::I));
        assert!(kinds.contains(&H264SliceKind::P));
        assert!(kinds.contains(&H264SliceKind::B));
    }

    #[test]
    fn parses_professional_high422_10bit_intra_fixture() {
        let parsed =
            parse_annex_b_access_unit(PROFESSIONAL_INTRA_FIXTURE).expect("professional AU parses");

        assert_eq!(parsed.profile, H264Profile::High422Intra);
        assert_eq!(parsed.desc.coded_width, 128);
        assert_eq!(parsed.desc.coded_height, 80);
        assert_eq!(parsed.desc.visible_region.width, 128);
        assert_eq!(parsed.desc.visible_region.height, 72);
        assert_eq!(parsed.picture.picture_width_in_mbs_minus1, 7);
        assert_eq!(parsed.picture.picture_height_in_mbs_minus1, 4);
        assert_eq!(parsed.desc.format, VideoSurfaceFormat::Yuv422_10);
        assert_eq!(parsed.desc.bit_depth.get(), 10);
        assert_eq!(parsed.desc.chroma, ChromaSubsampling::Cs422);
        assert_eq!(parsed.picture.bit_depth_luma_minus8, 2);
        assert_eq!(parsed.picture.bit_depth_chroma_minus8, 2);
        assert_eq!(parsed.picture.chroma_format_idc, 2);
        assert_eq!(parsed.slices[0].kind, H264SliceKind::I);
    }

    #[test]
    fn parses_professional_high422_10bit_long_gop_fixture() {
        let access_units = split_access_units_for_test(PROFESSIONAL_LONG_GOP_FIXTURE);
        assert_eq!(access_units.len(), 12);

        let mut state = H264DecoderState::new();
        let mut kinds = Vec::new();
        for access_unit in access_units {
            let parsed = state
                .parse_access_unit(&access_unit)
                .expect("professional Long-GOP AU parses");
            assert_eq!(parsed.profile, H264Profile::High422);
            assert_eq!(parsed.desc.format, VideoSurfaceFormat::Yuv422_10);
            assert_eq!(parsed.desc.bit_depth.get(), 10);
            assert_eq!(parsed.desc.chroma, ChromaSubsampling::Cs422);
            kinds.push(parsed.slices[0].kind.clone());
            state
                .finish_picture(&parsed)
                .expect("professional Long-GOP DPB update succeeds");
        }

        assert!(kinds.contains(&H264SliceKind::I));
        assert!(kinds.contains(&H264SliceKind::P));
        assert!(kinds.contains(&H264SliceKind::B));
        assert!(state.max_dpb_occupancy() > 1);
    }

    #[test]
    fn long_gop_decode_order_differs_from_presentation_order() {
        let access_units = split_access_units_for_test(LONG_GOP_FIXTURE);
        let mut state = H264DecoderState::new();
        let mut decode_pocs = Vec::new();
        let mut output_pocs = Vec::new();

        for access_unit in access_units {
            let parsed = state
                .parse_access_unit(&access_unit)
                .expect("Long-GOP AU parses");
            decode_pocs.push(parsed.picture.top_field_order_cnt);
            let update = state
                .finish_picture(&parsed)
                .expect("Long-GOP DPB update succeeds");
            output_pocs.extend(update.output_ready.into_iter().map(|id| id.poc));
        }
        output_pocs.extend(state.flush().output_ready.into_iter().map(|id| id.poc));

        assert_eq!(decode_pocs.len(), 12);
        assert_eq!(output_pocs.len(), 12);
        assert_ne!(decode_pocs, output_pocs);
        assert!(output_pocs.windows(2).all(|pair| pair[0] <= pair[1]));
        assert!(state.max_dpb_occupancy() > 1);
        assert!(state.max_output_pending() > 0);
    }

    #[test]
    fn short_term_pic_num_wraps_against_current_frame_num() {
        assert_eq!(short_term_pic_num(7, 8, 16), 7);
        assert_eq!(short_term_pic_num(8, 8, 16), 8);
        assert_eq!(short_term_pic_num(15, 8, 16), -1);
    }

    #[test]
    fn idr_reset_outputs_pending_pictures_when_allowed() {
        let mut state = H264DecoderState::new();
        state.max_num_reorder_frames = 2;
        state.dpb.push(dpb_picture_for_test(10, 10, true, true));
        state.dpb.push(dpb_picture_for_test(11, 11, true, true));

        let parsed = parsed_picture_for_finish_test(0, 0, true, false);
        let update = state
            .finish_picture(&parsed)
            .expect("IDR reset drains prior pictures");

        assert_eq!(
            update.output_ready,
            vec![
                H264PictureId {
                    frame_num: 10,
                    poc: 10
                },
                H264PictureId {
                    frame_num: 11,
                    poc: 11
                }
            ]
        );
        assert_eq!(
            update.released,
            vec![
                H264PictureId {
                    frame_num: 10,
                    poc: 10
                },
                H264PictureId {
                    frame_num: 11,
                    poc: 11
                }
            ]
        );
    }

    fn dpb_picture_for_test(
        frame_num: u16,
        poc: i32,
        reference: bool,
        output_needed: bool,
    ) -> DpbPicture {
        DpbPicture {
            id: H264PictureId { frame_num, poc },
            frame_num,
            top_field_order_cnt: poc,
            bottom_field_order_cnt: poc,
            reference,
            output_needed,
        }
    }

    fn parsed_picture_for_finish_test(
        frame_num: u16,
        poc: i32,
        idr: bool,
        no_output_of_prior_pics_flag: bool,
    ) -> ParsedH264AccessUnit {
        ParsedH264AccessUnit {
            desc: VideoSurfaceDesc {
                coded_width: 16,
                coded_height: 16,
                visible_region: VisibleRegion {
                    x: 0,
                    y: 0,
                    width: 16,
                    height: 16,
                },
                format: VideoSurfaceFormat::Nv12,
                bit_depth: BitDepth::new(8).expect("8-bit"),
                chroma: ChromaSubsampling::Cs420,
                scan_mode: ScanMode::Progressive,
                field_order: FieldOrder::Unknown,
            },
            profile: H264Profile::High,
            level_idc: 42,
            picture: ParsedH264Picture {
                picture_width_in_mbs_minus1: 0,
                picture_height_in_mbs_minus1: 0,
                bit_depth_luma_minus8: 0,
                bit_depth_chroma_minus8: 0,
                num_ref_frames: 2,
                chroma_format_idc: 1,
                gaps_in_frame_num_value_allowed_flag: false,
                frame_mbs_only_flag: true,
                mb_adaptive_frame_field_flag: false,
                direct_8x8_inference_flag: true,
                log2_max_frame_num_minus4: 0,
                pic_order_cnt_type: 0,
                log2_max_pic_order_cnt_lsb_minus4: 0,
                delta_pic_order_always_zero_flag: false,
                num_slice_groups_minus1: 0,
                slice_group_map_type: 0,
                slice_group_change_rate_minus1: 0,
                entropy_coding_mode_flag: true,
                weighted_pred_flag: false,
                weighted_bipred_idc: 0,
                transform_8x8_mode_flag: false,
                field_pic_flag: false,
                constrained_intra_pred_flag: false,
                pic_order_present_flag: false,
                deblocking_filter_control_present_flag: true,
                redundant_pic_cnt_present_flag: false,
                reference_pic_flag: true,
                idr_pic_flag: idr,
                no_output_of_prior_pics_flag,
                long_term_reference_flag: false,
                dec_ref_pic_marking: Some(ParsedDecRefPicMarking::Idr {
                    no_output_of_prior_pics_flag,
                    long_term_reference_flag: false,
                }),
                pic_init_qp_minus26: 0,
                pic_init_qs_minus26: 0,
                chroma_qp_index_offset: 0,
                second_chroma_qp_index_offset: 0,
                frame_num,
                top_field_order_cnt: poc,
                bottom_field_order_cnt: poc,
                scaling_list4x4: [[16; 16]; 6],
                scaling_list8x8: [[16; 64]; 2],
            },
            slices: Vec::new(),
            reference_frames: Vec::new(),
            max_dpb_frames: 2,
            max_num_reorder_frames: 2,
        }
    }

    fn split_access_units_for_test(data: &[u8]) -> Vec<Vec<u8>> {
        let nals = split_annex_b(data).expect("Annex B fixture");
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
}
