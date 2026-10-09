use std::any::Any;
use std::time::{Duration, Instant};

use qgs_core::{
    BackendDecodedSurface, BackendDecoder, BackendResource, DecoderBackend, DecoderError,
};
use qgs_protocol::{
    BitDepth, ChromaSubsampling, CreateDecoderRequest, DecoderConfig, FieldOrder, H264Profile,
    ScanMode, SubmitAccessUnitRequest, VideoCodec, VideoProfile, VideoSurfaceDesc,
    VideoSurfaceFormat, VisibleRegion, MAX_VIDEO_SURFACE_HEIGHT, MAX_VIDEO_SURFACE_WIDTH,
};
use rsmpeg::{
    avcodec::{AVCodec, AVCodecContext, AVCodecParserContext, AVPacket},
    avutil::{get_pix_fmt_name, AVFrame},
    error::RsmpegError,
    ffi,
};

pub const MAX_SOFTWARE_SURFACE_BYTES: usize = 128 * 1024 * 1024;
pub const MAX_LIVE_SOFTWARE_SURFACES: usize = 32;

#[derive(Debug, Default)]
pub struct SoftwareVideoBackend;

impl SoftwareVideoBackend {
    pub const fn new() -> Self {
        Self
    }

    pub fn supports_config(config: &DecoderConfig) -> bool {
        software_format_for_config(config).is_ok()
    }
}

impl DecoderBackend for SoftwareVideoBackend {
    fn create_decoder(
        &self,
        request: &CreateDecoderRequest,
    ) -> Result<Box<dyn BackendDecoder>, DecoderError> {
        Ok(Box::new(SoftwareH264Decoder::new(request.config.clone())?))
    }
}

pub struct SoftwareH264Decoder {
    context: AVCodecContext,
    parser: AVCodecParserContext,
    config: DecoderConfig,
    storage_format: SoftwarePixelFormat,
    max_live_surfaces: usize,
    output_count: u64,
    decode_start: Instant,
}

impl SoftwareH264Decoder {
    pub fn new(config: DecoderConfig) -> Result<Self, DecoderError> {
        config.validate()?;
        let storage_format = software_format_for_config(&config)?;
        let codec = AVCodec::find_decoder(ffi::AV_CODEC_ID_H264)
            .ok_or(DecoderError::UnsupportedDecodeConfiguration)?;
        let mut context = AVCodecContext::new(&codec);
        context
            .set_width(i32::try_from(config.coded_width).map_err(|_| DecoderError::DecodeFailed)?);
        context.set_height(
            i32::try_from(config.coded_height).map_err(|_| DecoderError::DecodeFailed)?,
        );
        context.set_time_base(ffi::AVRational { num: 1, den: 1 });
        context.set_pkt_timebase(ffi::AVRational { num: 1, den: 1 });
        context
            .open(None)
            .map_err(|_| DecoderError::UnsupportedDecodeConfiguration)?;
        let parser = AVCodecParserContext::init(ffi::AV_CODEC_ID_H264)
            .ok_or(DecoderError::UnsupportedDecodeConfiguration)?;
        Ok(Self {
            context,
            parser,
            config,
            storage_format,
            max_live_surfaces: 0,
            output_count: 0,
            decode_start: Instant::now(),
        })
    }

    pub fn decode_access_unit(
        &mut self,
        access_unit: &[u8],
    ) -> Result<Vec<SoftwareVideoSurface>, DecoderError> {
        self.send_access_unit_bytes(access_unit, None)?;
        self.receive_available_surfaces()
    }

    pub fn decode_access_unit_at(
        &mut self,
        access_unit: &[u8],
        presentation_position: u64,
    ) -> Result<Vec<SoftwareVideoSurface>, DecoderError> {
        self.send_access_unit_bytes(access_unit, Some(presentation_position))?;
        self.receive_available_surfaces()
    }

    pub fn flush_surfaces(&mut self) -> Result<Vec<SoftwareVideoSurface>, DecoderError> {
        self.drain_parser()?;
        match self.context.send_packet(None) {
            Ok(()) | Err(RsmpegError::DecoderFlushedError) => self.receive_available_surfaces(),
            Err(_) => Err(DecoderError::DecodeFailed),
        }
    }

    pub fn max_live_surfaces(&self) -> usize {
        self.max_live_surfaces
    }

    pub fn output_count(&self) -> u64 {
        self.output_count
    }

    pub fn elapsed(&self) -> Duration {
        self.decode_start.elapsed()
    }

    fn send_access_unit_bytes(
        &mut self,
        data: &[u8],
        presentation_position: Option<u64>,
    ) -> Result<(), DecoderError> {
        let mut offset = 0_usize;
        while offset < data.len() {
            let mut packet = AVPacket::new();
            if let Some(position) = presentation_position {
                let position = i64::try_from(position).map_err(|_| DecoderError::DecodeFailed)?;
                packet.set_pts(position);
                packet.set_dts(position);
            }
            let (ready, consumed) = self
                .parser
                .parse_packet(&mut self.context, &mut packet, &data[offset..])
                .map_err(|_| DecoderError::MalformedCompressedData)?;
            if consumed == 0 && !ready {
                break;
            }
            offset = offset
                .checked_add(consumed)
                .ok_or(DecoderError::DecodeFailed)?;
            if !ready {
                continue;
            }
            self.send_packet_to_decoder(&packet)?;
        }
        Ok(())
    }

    fn drain_parser(&mut self) -> Result<(), DecoderError> {
        loop {
            let mut packet = AVPacket::new();
            let (ready, consumed) = self
                .parser
                .parse_packet(&mut self.context, &mut packet, &[])
                .map_err(|_| DecoderError::MalformedCompressedData)?;
            if ready {
                self.send_packet_to_decoder(&packet)?;
            }
            if consumed == 0 || !ready {
                break;
            }
        }
        Ok(())
    }

    fn send_packet_to_decoder(&mut self, packet: &AVPacket) -> Result<(), DecoderError> {
        match self.context.send_packet(Some(packet)) {
            Ok(()) => Ok(()),
            Err(RsmpegError::DecoderFullError) => {
                let _ = self.receive_available_surfaces()?;
                self.context
                    .send_packet(Some(packet))
                    .map_err(|_| DecoderError::DecodeFailed)
            }
            Err(_) => Err(DecoderError::DecodeFailed),
        }
    }

    fn receive_available_surfaces(&mut self) -> Result<Vec<SoftwareVideoSurface>, DecoderError> {
        let mut surfaces = Vec::new();
        loop {
            match self.context.receive_frame() {
                Ok(frame) => {
                    let surface = SoftwareVideoSurface::from_frame(
                        frame,
                        self.config.clone(),
                        self.storage_format,
                        self.output_count,
                    )?;
                    self.output_count = self
                        .output_count
                        .checked_add(1)
                        .ok_or(DecoderError::DecodeFailed)?;
                    surfaces.push(surface);
                    self.max_live_surfaces = self.max_live_surfaces.max(surfaces.len());
                    if surfaces.len() > MAX_LIVE_SOFTWARE_SURFACES {
                        return Err(DecoderError::DecodeFailed);
                    }
                }
                Err(RsmpegError::DecoderDrainError | RsmpegError::DecoderFlushedError) => break,
                Err(_) => return Err(DecoderError::DecodeFailed),
            }
        }
        Ok(surfaces)
    }
}

impl BackendDecoder for SoftwareH264Decoder {
    fn submit_access_unit(
        &mut self,
        request: &SubmitAccessUnitRequest,
    ) -> Result<Vec<BackendDecodedSurface>, DecoderError> {
        let surfaces = self.decode_access_unit(&request.data)?;
        Ok(surfaces
            .into_iter()
            .map(BackendDecodedSurface::from)
            .collect())
    }

    fn flush(
        &mut self,
        _request: &qgs_protocol::FlushDecoderRequest,
    ) -> Result<Vec<BackendDecodedSurface>, DecoderError> {
        Ok(self
            .flush_surfaces()?
            .into_iter()
            .map(BackendDecodedSurface::from)
            .collect())
    }
}

impl From<SoftwareVideoSurface> for BackendDecodedSurface {
    fn from(surface: SoftwareVideoSurface) -> Self {
        Self {
            desc: surface.desc.clone(),
            resource: Box::new(surface),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SoftwarePixelFormat {
    Yuv420P8,
    Yuv422P10Le,
}

impl SoftwarePixelFormat {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Yuv420P8 => "yuv420p",
            Self::Yuv422P10Le => "yuv422p10le",
        }
    }

    const fn ffmpeg_format(self) -> ffi::AVPixelFormat {
        match self {
            Self::Yuv420P8 => ffi::AV_PIX_FMT_YUV420P,
            Self::Yuv422P10Le => ffi::AV_PIX_FMT_YUV422P10LE,
        }
    }

    pub const fn plane_shapes(self, width: u32, height: u32) -> [(u32, u32, u32); 3] {
        match self {
            Self::Yuv420P8 => [
                (width, height, 1),
                (width / 2, height / 2, 1),
                (width / 2, height / 2, 1),
            ],
            Self::Yuv422P10Le => [
                (width, height, 2),
                (width / 2, height, 2),
                (width / 2, height, 2),
            ],
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SoftwarePlane {
    pub width_samples: u32,
    pub height: u32,
    pub stride_bytes: usize,
    pub source_stride_bytes: usize,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SoftwareVideoSurface {
    pub desc: VideoSurfaceDesc,
    pub storage_format: SoftwarePixelFormat,
    pub decoder_pixel_format: String,
    pub presentation_index: u64,
    pub planes: Vec<SoftwarePlane>,
    pub checksum: u64,
}

impl SoftwareVideoSurface {
    pub fn from_frame(
        frame: AVFrame,
        config: DecoderConfig,
        storage_format: SoftwarePixelFormat,
        fallback_presentation_index: u64,
    ) -> Result<Self, DecoderError> {
        if frame.width <= 0 || frame.height <= 0 {
            return Err(DecoderError::DecodeFailed);
        }
        let width = u32::try_from(frame.width).map_err(|_| DecoderError::DecodeFailed)?;
        let height = u32::try_from(frame.height).map_err(|_| DecoderError::DecodeFailed)?;
        if width == 0 || height == 0 || width > config.coded_width || height > config.coded_height {
            return Err(DecoderError::DecodeFailed);
        }
        if frame.format != storage_format.ffmpeg_format() {
            return Err(DecoderError::UnsupportedDecodeConfiguration);
        }

        let decoder_pixel_format = get_pix_fmt_name(frame.format)
            .and_then(|name| name.to_str().ok())
            .unwrap_or(storage_format.name())
            .to_string();
        let mut packed = vec![
            0_u8;
            frame
                .image_get_buffer_size(1)
                .map_err(|_| DecoderError::DecodeFailed)?
        ];
        let copied = frame
            .image_copy_to_buffer(&mut packed, 1)
            .map_err(|_| DecoderError::DecodeFailed)?;
        packed.truncate(copied);

        let shapes = storage_format.plane_shapes(width, height);
        let mut offset = 0_usize;
        let mut planes = Vec::with_capacity(3);
        for (index, (plane_width, plane_height, bytes_per_sample)) in shapes.into_iter().enumerate()
        {
            let stride = checked_stride(plane_width, bytes_per_sample)?;
            let byte_len = checked_plane_len(stride, plane_height)?;
            let end = offset
                .checked_add(byte_len)
                .ok_or(DecoderError::DecodeFailed)?;
            let source_stride = usize::try_from(frame.linesize[index])
                .ok()
                .filter(|stride| *stride > 0)
                .ok_or(DecoderError::DecodeFailed)?;
            if end > packed.len() {
                return Err(DecoderError::DecodeFailed);
            }
            planes.push(SoftwarePlane {
                width_samples: plane_width,
                height: plane_height,
                stride_bytes: stride,
                source_stride_bytes: source_stride,
                data: packed[offset..end].to_vec(),
            });
            offset = end;
        }
        if offset != packed.len() {
            return Err(DecoderError::DecodeFailed);
        }
        if planes.iter().try_fold(0_usize, |total, plane| {
            total
                .checked_add(plane.data.len())
                .ok_or(DecoderError::DecodeFailed)
        })? > MAX_SOFTWARE_SURFACE_BYTES
        {
            return Err(DecoderError::DecodeFailed);
        }

        let desc = VideoSurfaceDesc {
            coded_width: width,
            coded_height: height,
            visible_region: VisibleRegion {
                x: 0,
                y: 0,
                width,
                height,
            },
            format: preferred_surface_format(&config)
                .ok_or(DecoderError::UnsupportedDecodeConfiguration)?,
            bit_depth: config.bit_depth,
            chroma: config.chroma,
            scan_mode: config.scan_mode,
            field_order: FieldOrder::Unknown,
        };
        desc.validate()?;
        let checksum = surface_checksum(&planes);
        let presentation_index = if frame.pts == ffi::AV_NOPTS_VALUE {
            fallback_presentation_index
        } else {
            u64::try_from(frame.pts).map_err(|_| DecoderError::DecodeFailed)?
        };
        Ok(Self {
            desc,
            storage_format,
            decoder_pixel_format,
            presentation_index,
            planes,
            checksum,
        })
    }

    pub fn owned_bytes(&self) -> usize {
        self.planes.iter().map(|plane| plane.data.len()).sum()
    }
}

impl BackendResource for SoftwareVideoSurface {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub fn software_format_for_config(
    config: &DecoderConfig,
) -> Result<SoftwarePixelFormat, DecoderError> {
    config.validate()?;
    if config.codec != VideoCodec::H264
        || config.scan_mode != ScanMode::Progressive
        || config.coded_width == 0
        || config.coded_height == 0
        || config.coded_width > MAX_VIDEO_SURFACE_WIDTH
        || config.coded_height > MAX_VIDEO_SURFACE_HEIGHT
    {
        return Err(DecoderError::UnsupportedDecodeConfiguration);
    }
    match (
        config.profile,
        config.bit_depth.get(),
        config.chroma,
        preferred_surface_format(config),
    ) {
        (
            VideoProfile::H264(H264Profile::Baseline | H264Profile::Main | H264Profile::High),
            8,
            ChromaSubsampling::Cs420,
            Some(VideoSurfaceFormat::Nv12),
        ) => Ok(SoftwarePixelFormat::Yuv420P8),
        (
            VideoProfile::H264(H264Profile::High422 | H264Profile::High422Intra),
            10,
            ChromaSubsampling::Cs422,
            Some(VideoSurfaceFormat::Yuv422_10),
        ) => Ok(SoftwarePixelFormat::Yuv422P10Le),
        _ => Err(DecoderError::UnsupportedDecodeConfiguration),
    }
}

pub fn preferred_surface_format(config: &DecoderConfig) -> Option<VideoSurfaceFormat> {
    match (config.bit_depth.get(), config.chroma) {
        (8, ChromaSubsampling::Cs420) => Some(VideoSurfaceFormat::Nv12),
        (10, ChromaSubsampling::Cs422) => Some(VideoSurfaceFormat::Yuv422_10),
        _ => None,
    }
}

pub fn checked_stride(width_samples: u32, bytes_per_sample: u32) -> Result<usize, DecoderError> {
    let stride = width_samples
        .checked_mul(bytes_per_sample)
        .ok_or(DecoderError::DecodeFailed)?;
    usize::try_from(stride).map_err(|_| DecoderError::DecodeFailed)
}

pub fn checked_plane_len(stride: usize, height: u32) -> Result<usize, DecoderError> {
    stride
        .checked_mul(usize::try_from(height).map_err(|_| DecoderError::DecodeFailed)?)
        .ok_or(DecoderError::DecodeFailed)
}

pub fn surface_checksum(planes: &[SoftwarePlane]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for plane in planes {
        for value in plane
            .width_samples
            .to_le_bytes()
            .into_iter()
            .chain(plane.height.to_le_bytes())
            .chain((plane.stride_bytes as u64).to_le_bytes())
            .chain(plane.data.iter().copied())
        {
            hash ^= u64::from(value);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

pub fn decoder_config_for_surface(
    device_id: qgs_protocol::DeviceId,
    profile: H264Profile,
    bit_depth: BitDepth,
    chroma: ChromaSubsampling,
    width: u32,
    height: u32,
) -> DecoderConfig {
    DecoderConfig {
        device_id,
        codec: VideoCodec::H264,
        profile: VideoProfile::H264(profile),
        bit_depth,
        chroma,
        coded_width: width,
        coded_height: height,
        scan_mode: ScanMode::Progressive,
    }
}

#[derive(Clone, Debug)]
pub struct DecodeRunStats {
    pub frames: Vec<SoftwareVideoSurface>,
    pub max_live_surfaces: usize,
    pub elapsed: Duration,
}

pub fn decode_access_units(
    config: DecoderConfig,
    access_units: &[Vec<u8>],
) -> Result<DecodeRunStats, DecoderError> {
    let mut decoder = SoftwareH264Decoder::new(config)?;
    let mut frames = Vec::new();
    for access_unit in access_units {
        frames.extend(decoder.decode_access_unit(access_unit)?);
    }
    frames.extend(decoder.flush_surfaces()?);
    Ok(DecodeRunStats {
        frames,
        max_live_surfaces: decoder.max_live_surfaces(),
        elapsed: decoder.elapsed(),
    })
}

pub fn decode_positioned_access_units(
    config: DecoderConfig,
    access_units: &[(u64, Vec<u8>)],
) -> Result<DecodeRunStats, DecoderError> {
    let mut decoder = SoftwareH264Decoder::new(config)?;
    let mut frames = Vec::new();
    for (position, access_unit) in access_units {
        frames.extend(decoder.decode_access_unit_at(access_unit, *position)?);
    }
    frames.extend(decoder.flush_surfaces()?);
    Ok(DecodeRunStats {
        frames,
        max_live_surfaces: decoder.max_live_surfaces(),
        elapsed: decoder.elapsed(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use qgs_protocol::DeviceId;

    fn config_10bit_422() -> DecoderConfig {
        decoder_config_for_surface(
            DeviceId::new(1).expect("device"),
            H264Profile::High422,
            BitDepth::new(10).expect("bit depth"),
            ChromaSubsampling::Cs422,
            1920,
            1080,
        )
    }

    #[test]
    fn supports_professional_10bit_422_config() {
        assert_eq!(
            software_format_for_config(&config_10bit_422()).expect("format"),
            SoftwarePixelFormat::Yuv422P10Le
        );
    }

    #[test]
    fn rejects_unsupported_software_config() {
        let mut config = config_10bit_422();
        config.chroma = ChromaSubsampling::Cs444;

        assert!(matches!(
            software_format_for_config(&config),
            Err(DecoderError::UnsupportedDecodeConfiguration)
        ));
    }

    #[test]
    fn yuv422p10_plane_model_uses_three_owned_planes() {
        let shapes = SoftwarePixelFormat::Yuv422P10Le.plane_shapes(1920, 1080);

        assert_eq!(shapes, [(1920, 1080, 2), (960, 1080, 2), (960, 1080, 2)]);
        assert_eq!(checked_stride(1920, 2).expect("stride"), 3840);
        assert_eq!(checked_plane_len(3840, 1080).expect("len"), 4_147_200);
    }

    #[test]
    fn decoded_size_arithmetic_is_checked() {
        assert!(checked_stride(u32::MAX, 4).is_err());
        assert!(checked_plane_len(usize::MAX, 2).is_err());
    }

    #[test]
    fn checksum_is_deterministic() {
        let planes = vec![SoftwarePlane {
            width_samples: 2,
            height: 1,
            stride_bytes: 4,
            source_stride_bytes: 8,
            data: vec![1, 0, 2, 0],
        }];

        assert_eq!(surface_checksum(&planes), surface_checksum(&planes));
    }
}
