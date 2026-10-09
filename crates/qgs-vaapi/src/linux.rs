use std::any::Any;
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::os::fd::OwnedFd;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Instant;

use libva::{
    BorrowedBufferType, BufferType, Context, Display, H264PicFields, H264SeqFields, IQMatrix,
    IQMatrixBufferH264, Image, Picture, PictureEnd, PictureH264, PictureParameter,
    PictureParameterBufferH264, SliceParameter, SliceParameterBufferH264, Surface, VAConfigAttrib,
    VAConfigAttribType, VAEntrypoint, VAProfile, VA_ATTRIB_NOT_SUPPORTED, VA_FOURCC_NV12,
    VA_FOURCC_P010, VA_FOURCC_UYVY, VA_FOURCC_Y210, VA_FOURCC_YUY2, VA_RT_FORMAT_YUV420,
    VA_RT_FORMAT_YUV420_10, VA_RT_FORMAT_YUV422, VA_RT_FORMAT_YUV422_10,
};
use qgs_codec_h264::{
    parse_annex_b_access_unit, H264DecoderState, H264Error, H264PictureId, ParsedH264AccessUnit,
    ParsedH264Reference,
};
use qgs_core::{
    BackendDecodedSurface, BackendDecoder, BackendResource, DecoderBackend, DecoderError,
    VideoCapabilityDiscovery, VideoCapabilityDiscoveryError,
};
use qgs_protocol::{
    BitDepth, ChromaSubsampling, CreateDecoderRequest, DecoderConfig, DecoderId, DeviceDesc,
    DeviceId, FlushDecoderRequest, H264Profile, Mpeg2Profile, SubmitAccessUnitRequest,
    VideoCapabilities, VideoCodec, VideoDecodeCapability, VideoProfile, VideoSurfaceDesc,
    VideoSurfaceFormat, VisibleRegion, MAX_VIDEO_SURFACE_HEIGHT, MAX_VIDEO_SURFACE_WIDTH,
};

const DEFAULT_MAX_WIDTH: u32 = MAX_VIDEO_SURFACE_WIDTH;
const DEFAULT_MAX_HEIGHT: u32 = MAX_VIDEO_SURFACE_HEIGHT;

#[derive(Debug)]
pub struct VaapiVideoDiscovery {
    devices: BTreeMap<DeviceId, VaapiDevice>,
    decode_mode: VaapiDecodeMode,
}

impl VaapiVideoDiscovery {
    pub fn new(qgs_devices: &[DeviceDesc]) -> Self {
        Self::with_decode_mode(qgs_devices, VaapiDecodeMode::Normal)
    }

    pub fn with_decode_mode(qgs_devices: &[DeviceDesc], decode_mode: VaapiDecodeMode) -> Self {
        let render_nodes = enumerate_render_nodes();
        let mut devices = BTreeMap::new();

        for device in qgs_devices {
            let Some(render_node) = render_nodes.iter().find(|node| {
                node.vendor_id == device.vendor_id && node.device_id == device.device_id
            }) else {
                continue;
            };

            devices.insert(
                device.id,
                VaapiDevice {
                    path: render_node.path.clone(),
                    vendor_id: render_node.vendor_id,
                    device_id: render_node.device_id,
                    driver: render_node.driver.clone(),
                },
            );
        }

        Self {
            devices,
            decode_mode,
        }
    }

    pub fn devices(&self) -> Vec<VaapiDeviceInfo> {
        self.devices
            .iter()
            .map(|(device_id, device)| VaapiDeviceInfo {
                device_id: *device_id,
                path: device.path.clone(),
                vendor_id: device.vendor_id,
                device_id_pci: device.device_id,
                driver: device.driver.clone(),
            })
            .collect()
    }
}

impl VideoCapabilityDiscovery for VaapiVideoDiscovery {
    fn query_video_capabilities(
        &self,
        device_id: DeviceId,
    ) -> Result<VideoCapabilities, VideoCapabilityDiscoveryError> {
        let device = self
            .devices
            .get(&device_id)
            .ok_or(VideoCapabilityDiscoveryError::UnknownDeviceId)?;

        let display = Display::open_drm_display(&device.path)
            .map_err(|_| VideoCapabilityDiscoveryError::BackendUnavailable)?;
        let capabilities = query_decode_capabilities(&display)
            .map_err(|_| VideoCapabilityDiscoveryError::BackendFailed)?;

        Ok(VideoCapabilities {
            device_id,
            decode: capabilities,
        })
    }
}

impl DecoderBackend for VaapiVideoDiscovery {
    fn create_decoder(
        &self,
        request: &CreateDecoderRequest,
    ) -> Result<Box<dyn BackendDecoder>, DecoderError> {
        Ok(Box::new(self.create_h264_decoder(
            &VaapiCreateDecoderRequest {
                config: request.config.clone(),
                surface_pool_size: 24,
            },
        )?))
    }
}

impl VaapiVideoDiscovery {
    fn create_h264_decoder(
        &self,
        request: &VaapiCreateDecoderRequest,
    ) -> Result<VaapiH264Decoder, DecoderError> {
        request.config.validate()?;
        if request.config.codec != VideoCodec::H264
            || request.config.bit_depth.get() != 8
            || request.config.chroma != ChromaSubsampling::Cs420
            || request.config.coded_width == 0
            || request.config.coded_height == 0
        {
            return Err(DecoderError::UnsupportedDecodeConfiguration);
        }
        match request.config.profile {
            VideoProfile::H264(H264Profile::Baseline | H264Profile::Main | H264Profile::High) => {}
            _ => return Err(DecoderError::UnsupportedDecodeConfiguration),
        }

        let device = self
            .devices
            .get(&request.config.device_id)
            .ok_or(DecoderError::UnknownDeviceId)?;
        let display = Display::open_drm_display(&device.path)
            .map_err(|_| DecoderError::UnsupportedDecodeConfiguration)?;
        let capabilities = query_decode_capabilities(&display)
            .map_err(|_| DecoderError::UnsupportedDecodeConfiguration)?;
        if !capabilities
            .iter()
            .any(|capability| request.config.is_satisfied_by(capability))
        {
            return Err(DecoderError::UnsupportedDecodeConfiguration);
        }
        let profile = va_profile_from_h264(request.config.profile)?;
        let entrypoints = display
            .query_config_entrypoints(profile)
            .map_err(|_| DecoderError::UnsupportedDecodeConfiguration)?;
        if !entrypoints.contains(&VAEntrypoint::VAEntrypointVLD) {
            return Err(DecoderError::UnsupportedDecodeConfiguration);
        }
        let attrs = vec![VAConfigAttrib {
            type_: VAConfigAttribType::VAConfigAttribRTFormat,
            value: VA_RT_FORMAT_YUV420,
        }];
        let config = display
            .create_config(attrs, profile, VAEntrypoint::VAEntrypointVLD)
            .map_err(|_| DecoderError::UnsupportedDecodeConfiguration)?;
        let surfaces = display
            .create_surfaces(
                VA_RT_FORMAT_YUV420,
                Some(VA_FOURCC_NV12),
                request.config.coded_width,
                request.config.coded_height,
                None,
                vec![(); request.surface_pool_size],
            )
            .map_err(|_| DecoderError::UnsupportedDecodeConfiguration)?;
        let context = display
            .create_context(
                &config,
                request.config.coded_width,
                request.config.coded_height,
                Some(&surfaces),
                true,
            )
            .map_err(|_| DecoderError::UnsupportedDecodeConfiguration)?;

        Ok(VaapiH264Decoder {
            context,
            pool: Rc::new(RefCell::new(VaapiSurfacePool::new(surfaces))),
            decoded_surfaces: BTreeMap::new(),
            frontend: H264DecoderState::new(),
            device_id: request.config.device_id,
            config: request.config.clone(),
            max_live_surfaces: 0,
            max_submitted_unsynced_surfaces: 0,
            timing: VaapiDecodeTiming::default(),
            mode: self.decode_mode,
            diagnostic_frames: 0,
            diagnostic_export_probes: 0,
        })
    }
}

struct VaapiCreateDecoderRequest {
    config: DecoderConfig,
    surface_pool_size: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VaapiDecodeMode {
    Normal,
    Diagnostic,
}

impl VaapiDecodeMode {
    const fn diagnostics_enabled(self) -> bool {
        matches!(self, Self::Diagnostic)
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct VaapiSurfacePoolStats {
    pub surfaces_allocated: usize,
    pub surface_reuse_count: usize,
    pub surfaces_recycled: usize,
    pub recycle_sync_count: usize,
    pub deferred_recycle_count: usize,
    pub peak_checked_out_surfaces: usize,
    pub peak_pending_recycle_surfaces: usize,
    pub minimum_free_surfaces: usize,
}

#[derive(Clone, Debug, Default)]
pub struct VaapiSurfaceDiagnostics {
    pub validation_checksum: Option<u32>,
    pub export_probe: Option<VaapiExportProbe>,
}

#[derive(Clone, Debug)]
pub struct VaapiDecodeObservation {
    pub frames: usize,
    pub selected_output_checksums: Vec<(usize, Option<u32>)>,
    pub pool_stats: VaapiSurfacePoolStats,
    pub timing: VaapiDecodeTiming,
    pub peak_dpb_occupancy: usize,
    pub peak_output_pending: usize,
    pub peak_live_surfaces: usize,
    pub peak_submitted_unsynced_surfaces: usize,
    pub max_client_held_outputs: usize,
    pub diagnostic_frames: usize,
    pub diagnostic_export_probes: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CpuNv12Plane {
    pub width_bytes: usize,
    pub height: usize,
    pub stride_bytes: usize,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CpuNv12Surface {
    pub desc: VideoSurfaceDesc,
    pub y: CpuNv12Plane,
    pub uv: CpuNv12Plane,
}

impl CpuNv12Surface {
    pub fn owned_bytes(&self) -> usize {
        self.y.data.len().saturating_add(self.uv.data.len())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CpuNv12FrameLayout {
    pub y_width_bytes: usize,
    pub y_height: usize,
    pub y_stride_bytes: usize,
    pub uv_width_bytes: usize,
    pub uv_height: usize,
    pub uv_stride_bytes: usize,
}

impl CpuNv12FrameLayout {
    pub fn tight_coded(coded_width: u32, coded_height: u32) -> Result<Self, DecoderError> {
        let width = usize::try_from(coded_width).map_err(|_| DecoderError::DecodeFailed)?;
        let height = usize::try_from(coded_height).map_err(|_| DecoderError::DecodeFailed)?;
        if width == 0 || height == 0 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            return Err(DecoderError::DecodeFailed);
        }
        Ok(Self {
            y_width_bytes: width,
            y_height: height,
            y_stride_bytes: width,
            uv_width_bytes: width,
            uv_height: height / 2,
            uv_stride_bytes: width,
        })
    }

    pub fn tight_visible(visible_region: VisibleRegion) -> Result<Self, DecoderError> {
        let width =
            usize::try_from(visible_region.width).map_err(|_| DecoderError::DecodeFailed)?;
        let height =
            usize::try_from(visible_region.height).map_err(|_| DecoderError::DecodeFailed)?;
        if width == 0 || height == 0 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            return Err(DecoderError::DecodeFailed);
        }
        Ok(Self {
            y_width_bytes: width,
            y_height: height,
            y_stride_bytes: width,
            uv_width_bytes: width,
            uv_height: height / 2,
            uv_stride_bytes: width,
        })
    }

    pub fn coded_with_destination_pitch(
        coded_width: u32,
        coded_height: u32,
        y_stride_bytes: usize,
        uv_stride_bytes: usize,
    ) -> Result<Self, DecoderError> {
        let tight = Self::tight_coded(coded_width, coded_height)?;
        if y_stride_bytes < tight.y_width_bytes || uv_stride_bytes < tight.uv_width_bytes {
            return Err(DecoderError::DecodeFailed);
        }
        Ok(Self {
            y_stride_bytes,
            uv_stride_bytes,
            ..tight
        })
    }

    pub fn full_source_rows(
        y_height: usize,
        y_stride_bytes: usize,
        uv_height: usize,
        uv_stride_bytes: usize,
    ) -> Result<Self, DecoderError> {
        if y_height == 0
            || uv_height == 0
            || y_stride_bytes == 0
            || uv_stride_bytes == 0
            || !y_stride_bytes.is_multiple_of(2)
            || !uv_stride_bytes.is_multiple_of(2)
        {
            return Err(DecoderError::DecodeFailed);
        }
        Ok(Self {
            y_width_bytes: y_stride_bytes,
            y_height,
            y_stride_bytes,
            uv_width_bytes: uv_stride_bytes,
            uv_height,
            uv_stride_bytes,
        })
    }

    fn bytes_per_frame(&self) -> Result<usize, DecoderError> {
        let y_len = self
            .y_stride_bytes
            .checked_mul(self.y_height)
            .ok_or(DecoderError::DecodeFailed)?;
        let uv_len = self
            .uv_stride_bytes
            .checked_mul(self.uv_height)
            .ok_or(DecoderError::DecodeFailed)?;
        y_len.checked_add(uv_len).ok_or(DecoderError::DecodeFailed)
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CpuNv12PoolStats {
    pub frames_created: usize,
    pub frames_reused: usize,
    pub peak_checked_out: usize,
    pub bytes_per_frame: usize,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CpuNv12TransferTiming {
    pub sync_ns: u128,
    pub image_create_ns: u128,
    pub copy_y_ns: u128,
    pub copy_uv_ns: u128,
    pub image_release_ns: u128,
    pub bytes_copied: usize,
    pub image_layout: Option<VaNv12ImageLayout>,
    pub total_ns: u128,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct VaNv12ImageLayout {
    pub fourcc: u32,
    pub width: u32,
    pub height: u32,
    pub num_planes: u32,
    pub offsets: [usize; 3],
    pub pitches: [usize; 3],
    pub data_size: usize,
    pub derived: bool,
}

impl VaNv12ImageLayout {
    pub fn fourcc_string(&self) -> String {
        let bytes = self.fourcc.to_le_bytes();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn y_uv_contiguous(&self) -> bool {
        self.offsets[0].checked_add(self.pitches[0].saturating_mul(self.height as usize))
            == Some(self.offsets[1])
    }
}

pub struct CpuNv12FramePool {
    coded_width: u32,
    coded_height: u32,
    visible_region: VisibleRegion,
    capacity: usize,
    available: Vec<CpuNv12Surface>,
    checked_out: usize,
    total_acquires: usize,
    stats: CpuNv12PoolStats,
}

impl CpuNv12FramePool {
    pub fn new(
        coded_width: u32,
        coded_height: u32,
        visible_region: VisibleRegion,
        capacity: usize,
    ) -> Result<Self, DecoderError> {
        Self::new_with_layout(
            coded_width,
            coded_height,
            visible_region,
            capacity,
            CpuNv12FrameLayout::tight_coded(coded_width, coded_height)?,
        )
    }

    pub fn new_with_layout(
        coded_width: u32,
        coded_height: u32,
        visible_region: VisibleRegion,
        capacity: usize,
        layout: CpuNv12FrameLayout,
    ) -> Result<Self, DecoderError> {
        if capacity == 0 || !coded_width.is_multiple_of(2) || !coded_height.is_multiple_of(2) {
            return Err(DecoderError::DecodeFailed);
        }
        let desc = nv12_surface_desc(coded_width, coded_height, visible_region)?;
        let y_len = layout
            .y_stride_bytes
            .checked_mul(layout.y_height)
            .ok_or(DecoderError::DecodeFailed)?;
        let uv_len = layout
            .uv_stride_bytes
            .checked_mul(layout.uv_height)
            .ok_or(DecoderError::DecodeFailed)?;
        let bytes_per_frame = layout.bytes_per_frame()?;
        let mut available = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            available.push(CpuNv12Surface {
                desc: desc.clone(),
                y: CpuNv12Plane {
                    width_bytes: layout.y_width_bytes,
                    height: layout.y_height,
                    stride_bytes: layout.y_stride_bytes,
                    data: vec![0; y_len],
                },
                uv: CpuNv12Plane {
                    width_bytes: layout.uv_width_bytes,
                    height: layout.uv_height,
                    stride_bytes: layout.uv_stride_bytes,
                    data: vec![0; uv_len],
                },
            });
        }
        Ok(Self {
            coded_width,
            coded_height,
            visible_region,
            capacity,
            available,
            checked_out: 0,
            total_acquires: 0,
            stats: CpuNv12PoolStats {
                frames_created: capacity,
                bytes_per_frame,
                ..CpuNv12PoolStats::default()
            },
        })
    }

    pub fn acquire(&mut self) -> Result<CpuNv12Surface, DecoderError> {
        let frame = self.available.pop().ok_or(DecoderError::DecodeFailed)?;
        if self.total_acquires >= self.capacity {
            self.stats.frames_reused = self.stats.frames_reused.saturating_add(1);
        }
        self.total_acquires = self.total_acquires.saturating_add(1);
        self.checked_out = self.checked_out.saturating_add(1);
        self.stats.peak_checked_out = self.stats.peak_checked_out.max(self.checked_out);
        Ok(frame)
    }

    pub fn release(&mut self, mut frame: CpuNv12Surface) -> Result<(), DecoderError> {
        if frame.desc.coded_width != self.coded_width
            || frame.desc.coded_height != self.coded_height
            || frame.desc.visible_region != self.visible_region
            || self.available.len() >= self.capacity
            || self.checked_out == 0
        {
            return Err(DecoderError::DecodeFailed);
        }
        frame.y.data.fill(0);
        frame.uv.data.fill(0);
        self.checked_out -= 1;
        self.available.push(frame);
        Ok(())
    }

    pub fn stats(&self) -> CpuNv12PoolStats {
        self.stats.clone()
    }
}

pub fn transfer_nv12_surface(
    resource: &dyn BackendResource,
    destination: CpuNv12Surface,
) -> Result<CpuNv12Surface, DecoderError> {
    transfer_nv12_surface_timed(resource, destination).map(|(surface, _timing)| surface)
}

pub fn transfer_nv12_surface_timed(
    resource: &dyn BackendResource,
    destination: CpuNv12Surface,
) -> Result<(CpuNv12Surface, CpuNv12TransferTiming), DecoderError> {
    let surface = resource
        .as_any()
        .downcast_ref::<VaapiVideoSurface>()
        .ok_or(DecoderError::UnsupportedDecodeConfiguration)?;
    surface.state.transfer_nv12(destination)
}

pub fn describe_nv12_surface_image(
    resource: &dyn BackendResource,
) -> Result<VaNv12ImageLayout, DecoderError> {
    let surface = resource
        .as_any()
        .downcast_ref::<VaapiVideoSurface>()
        .ok_or(DecoderError::UnsupportedDecodeConfiguration)?;
    surface.state.describe_nv12_image()
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct VaapiDecodeTiming {
    pub surface_acquire_ns: u128,
    pub h264_parse_ns: u128,
    pub parameter_build_ns: u128,
    pub buffer_create_ns: u128,
    pub begin_picture_ns: u128,
    pub render_picture_ns: u128,
    pub end_picture_ns: u128,
    pub inline_sync_ns: u128,
    pub reclaim_sync_ns: u128,
    pub diagnostics_ns: u128,
    pub finish_picture_ns: u128,
    pub output_mapping_ns: u128,
    pub release_ns: u128,
    pub flush_ns: u128,
    pub submit_total_ns: u128,
}

impl VaapiDecodeTiming {
    fn add_elapsed(bucket: &mut u128, start: Instant) {
        *bucket = bucket.saturating_add(start.elapsed().as_nanos());
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VaapiDeviceInfo {
    pub device_id: DeviceId,
    pub path: PathBuf,
    pub vendor_id: u32,
    pub device_id_pci: u32,
    pub driver: Option<String>,
}

pub struct VaapiDecodedPrimeSurfaceDiagnostic {
    pub fourcc: u32,
    pub width: u32,
    pub height: u32,
    pub objects: Vec<VaapiPrimeObjectDiagnostic>,
    pub layers: Vec<VaapiPrimeLayerDiagnostic>,
    pub validation_checksum: u32,
    _surface: Surface<()>,
}

#[derive(Debug)]
pub struct VaapiPrimeObjectDiagnostic {
    pub fd: OwnedFd,
    pub size: u32,
    pub drm_format_modifier: u64,
}

#[derive(Clone, Debug)]
pub struct VaapiPrimeLayerDiagnostic {
    pub drm_format: u32,
    pub num_planes: u32,
    pub object_index: [u8; 4],
    pub offset: [u32; 4],
    pub pitch: [u32; 4],
}

pub fn decode_h264_drm_prime_for_diagnostic(
    render_node: &Path,
    access_unit: &[u8],
) -> Result<VaapiDecodedPrimeSurfaceDiagnostic, DecoderError> {
    let parsed = parse_annex_b_access_unit(access_unit).map_err(decoder_error_from_h264)?;
    let display = Display::open_drm_display(render_node)
        .map_err(|_| DecoderError::UnsupportedDecodeConfiguration)?;
    let profile = va_profile_from_h264(decoder_profile_from_parsed(&parsed))?;
    let entrypoints = display
        .query_config_entrypoints(profile)
        .map_err(|_| DecoderError::UnsupportedDecodeConfiguration)?;
    if !entrypoints.contains(&VAEntrypoint::VAEntrypointVLD) {
        return Err(DecoderError::UnsupportedDecodeConfiguration);
    }

    let attrs = vec![VAConfigAttrib {
        type_: VAConfigAttribType::VAConfigAttribRTFormat,
        value: VA_RT_FORMAT_YUV420,
    }];
    let config = display
        .create_config(attrs, profile, VAEntrypoint::VAEntrypointVLD)
        .map_err(|_| DecoderError::UnsupportedDecodeConfiguration)?;
    let surfaces = display
        .create_surfaces(
            VA_RT_FORMAT_YUV420,
            Some(VA_FOURCC_NV12),
            parsed.desc.coded_width,
            parsed.desc.coded_height,
            None,
            vec![()],
        )
        .map_err(|_| DecoderError::UnsupportedDecodeConfiguration)?;
    let context = display
        .create_context(
            &config,
            parsed.desc.coded_width,
            parsed.desc.coded_height,
            Some(&surfaces),
            true,
        )
        .map_err(|_| DecoderError::UnsupportedDecodeConfiguration)?;
    let surface = surfaces
        .into_iter()
        .next()
        .ok_or(DecoderError::DecodeFailed)?;
    let mut timing = VaapiDecodeTiming::default();
    let surface = decode_h264_access_unit(
        context,
        surface,
        &parsed,
        &BTreeMap::new(),
        true,
        &mut timing,
    )?;
    let DecodedVaSurface::Ready(surface) = surface else {
        return Err(DecoderError::DecodeFailed);
    };
    let validation_checksum =
        validation_checksum(&surface, parsed.desc.coded_width, parsed.desc.coded_height)
            .map_err(|_| DecoderError::DecodeFailed)?;
    let descriptor = surface
        .export_prime()
        .map_err(|_| DecoderError::DecodeFailed)?;
    let objects = descriptor
        .objects
        .into_iter()
        .map(|object| VaapiPrimeObjectDiagnostic {
            fd: object.fd,
            size: object.size,
            drm_format_modifier: object.drm_format_modifier,
        })
        .collect();
    let layers = descriptor
        .layers
        .into_iter()
        .map(|layer| VaapiPrimeLayerDiagnostic {
            drm_format: layer.drm_format,
            num_planes: layer.num_planes,
            object_index: layer.object_index,
            offset: layer.offset,
            pitch: layer.pitch,
        })
        .collect();

    Ok(VaapiDecodedPrimeSurfaceDiagnostic {
        fourcc: descriptor.fourcc,
        width: descriptor.width,
        height: descriptor.height,
        objects,
        layers,
        validation_checksum,
        _surface: surface,
    })
}

pub fn decode_h264_access_units_for_observation(
    qgs_devices: &[DeviceDesc],
    device_id: DeviceId,
    config: &DecoderConfig,
    access_units: &[Vec<u8>],
    mode: VaapiDecodeMode,
    selected_output_ordinals: &[usize],
) -> Result<VaapiDecodeObservation, DecoderError> {
    decode_h264_access_units_for_observation_with_pool_size(
        qgs_devices,
        device_id,
        config,
        access_units,
        mode,
        selected_output_ordinals,
        24,
    )
}

pub fn decode_h264_access_units_for_observation_with_pool_size(
    qgs_devices: &[DeviceDesc],
    device_id: DeviceId,
    config: &DecoderConfig,
    access_units: &[Vec<u8>],
    mode: VaapiDecodeMode,
    selected_output_ordinals: &[usize],
    surface_pool_size: usize,
) -> Result<VaapiDecodeObservation, DecoderError> {
    let discovery = VaapiVideoDiscovery::with_decode_mode(qgs_devices, mode);
    let mut config = config.clone();
    config.device_id = device_id;
    let mut decoder = discovery.create_h264_decoder(&VaapiCreateDecoderRequest {
        config,
        surface_pool_size,
    })?;
    let decoder_id = DecoderId::new(1).map_err(DecoderError::from)?;
    let mut frames = 0_usize;
    let mut selected_output_checksums = Vec::new();
    let mut max_client_held_outputs = 0_usize;

    for access_unit in access_units {
        let outputs = decoder.submit_access_unit(&SubmitAccessUnitRequest {
            decoder_id,
            data: access_unit.clone(),
        })?;
        max_client_held_outputs = max_client_held_outputs.max(outputs.len());
        collect_observation_outputs(
            outputs,
            &mut frames,
            selected_output_ordinals,
            &mut selected_output_checksums,
        );
    }

    let outputs = decoder.flush(&FlushDecoderRequest { decoder_id })?;
    max_client_held_outputs = max_client_held_outputs.max(outputs.len());
    collect_observation_outputs(
        outputs,
        &mut frames,
        selected_output_ordinals,
        &mut selected_output_checksums,
    );

    let pool_stats = decoder.pool.borrow().stats.clone();
    Ok(VaapiDecodeObservation {
        frames,
        selected_output_checksums,
        pool_stats,
        timing: decoder.timing.clone(),
        peak_dpb_occupancy: decoder.frontend.max_dpb_occupancy(),
        peak_output_pending: decoder.frontend.max_output_pending(),
        peak_live_surfaces: decoder.max_live_surfaces,
        peak_submitted_unsynced_surfaces: decoder.max_submitted_unsynced_surfaces,
        max_client_held_outputs,
        diagnostic_frames: decoder.diagnostic_frames,
        diagnostic_export_probes: decoder.diagnostic_export_probes,
    })
}

fn collect_observation_outputs(
    outputs: Vec<BackendDecodedSurface>,
    frames: &mut usize,
    selected_output_ordinals: &[usize],
    selected_output_checksums: &mut Vec<(usize, Option<u32>)>,
) {
    for output in outputs {
        let ordinal = *frames;
        if selected_output_ordinals.contains(&ordinal) {
            let checksum = surface_diagnostics(output.resource.as_ref())
                .and_then(|diagnostics| diagnostics.validation_checksum);
            selected_output_checksums.push((ordinal, checksum));
        }
        *frames = frames.saturating_add(1);
    }
}

#[derive(Debug)]
struct VaapiDevice {
    path: PathBuf,
    vendor_id: u32,
    device_id: u32,
    driver: Option<String>,
}

fn decoder_profile_from_parsed(parsed: &ParsedH264AccessUnit) -> VideoProfile {
    qgs_codec_h264::decoder_profile(parsed)
}

fn parsed_access_unit_matches_config(
    parsed: &ParsedH264AccessUnit,
    config: &DecoderConfig,
) -> bool {
    config.codec == qgs_codec_h264::decoder_codec()
        && config.profile == qgs_codec_h264::decoder_profile(parsed)
        && config.bit_depth == parsed.desc.bit_depth
        && config.chroma == parsed.desc.chroma
        && config.coded_width == parsed.desc.coded_width
        && config.coded_height == parsed.desc.coded_height
        && config.scan_mode == parsed.desc.scan_mode
}

struct VaapiH264Decoder {
    context: Rc<Context>,
    pool: Rc<RefCell<VaapiSurfacePool>>,
    decoded_surfaces: BTreeMap<(u16, i32), Rc<DecodedSurfaceState>>,
    frontend: H264DecoderState,
    device_id: DeviceId,
    config: DecoderConfig,
    max_live_surfaces: usize,
    max_submitted_unsynced_surfaces: usize,
    timing: VaapiDecodeTiming,
    mode: VaapiDecodeMode,
    diagnostic_frames: usize,
    diagnostic_export_probes: usize,
}

impl BackendDecoder for VaapiH264Decoder {
    fn submit_access_unit(
        &mut self,
        request: &SubmitAccessUnitRequest,
    ) -> Result<Vec<BackendDecodedSurface>, DecoderError> {
        let submit_start = Instant::now();
        let parse_start = Instant::now();
        let parsed = self
            .frontend
            .parse_access_unit(&request.data)
            .map_err(decoder_error_from_h264)?;
        VaapiDecodeTiming::add_elapsed(&mut self.timing.h264_parse_ns, parse_start);
        if !parsed_access_unit_matches_config(&parsed, &self.config) {
            return Err(DecoderError::UnsupportedDecodeConfiguration);
        }
        let acquire_start = Instant::now();
        let surface = self
            .pool
            .borrow_mut()
            .acquire(&mut self.timing)
            .ok_or(DecoderError::DecodeFailed)?;
        VaapiDecodeTiming::add_elapsed(&mut self.timing.surface_acquire_ns, acquire_start);
        let surface = decode_h264_access_unit(
            Rc::clone(&self.context),
            surface,
            &parsed,
            &self.decoded_surfaces,
            self.mode.diagnostics_enabled(),
            &mut self.timing,
        )?;
        let diagnostics = if self.mode.diagnostics_enabled() {
            let diagnostics_start = Instant::now();
            let diagnostics = collect_surface_diagnostics(
                surface.as_surface(),
                parsed.desc.coded_width,
                parsed.desc.coded_height,
            )?;
            VaapiDecodeTiming::add_elapsed(&mut self.timing.diagnostics_ns, diagnostics_start);
            self.diagnostic_frames = self.diagnostic_frames.saturating_add(1);
            if diagnostics.export_probe.is_some() {
                self.diagnostic_export_probes = self.diagnostic_export_probes.saturating_add(1);
            }
            if let Some(validation_checksum) = diagnostics.validation_checksum {
                eprintln!(
                    "qgs-vaapi diagnostic: decoded H.264 frame on device {} validation checksum=0x{validation_checksum:08x}",
                    self.device_id.get()
                );
            }
            if let Some(probe) = &diagnostics.export_probe {
                eprintln!(
                    "qgs-vaapi diagnostic: VA surface DRM PRIME export probe succeeded: fourcc=0x{:08x} size={}x{} objects={} layers={} object_sizes={:?} modifiers={:?} layer_formats={:?} pitches={:?} offsets={:?}",
                    probe.fourcc,
                    probe.width,
                    probe.height,
                    probe.object_count,
                    probe.layer_count,
                    probe.object_sizes,
                    probe.modifiers,
                    probe.layer_formats,
                    probe.pitches,
                    probe.offsets
                );
            } else {
                eprintln!("qgs-vaapi diagnostic: VA surface DRM PRIME export probe failed");
            }
            diagnostics
        } else {
            VaapiSurfaceDiagnostics::default()
        };
        let id = parsed.picture.id();
        self.decoded_surfaces.insert(
            surface_key(&id),
            Rc::new(DecodedSurfaceState {
                pool: Rc::clone(&self.pool),
                codec_released: Cell::new(false),
                surface: RefCell::new(Some(surface)),
                desc: parsed.desc.clone(),
                diagnostics,
            }),
        );
        self.max_live_surfaces = self.max_live_surfaces.max(self.decoded_surfaces.len());
        self.update_submitted_unsynced_peak();
        let finish_start = Instant::now();
        let update = self
            .frontend
            .finish_picture(&parsed)
            .map_err(decoder_error_from_h264)?;
        VaapiDecodeTiming::add_elapsed(&mut self.timing.finish_picture_ns, finish_start);
        if self.mode.diagnostics_enabled() {
            eprintln!(
                "qgs-vaapi diagnostic: H.264 DPB occupancy={} output_pending={} live_va_surfaces={}",
                update.max_dpb_occupancy,
                update.max_output_pending,
                self.decoded_surfaces.len()
            );
        }
        let output_start = Instant::now();
        let outputs = self.outputs_from_ids(&update.output_ready)?;
        VaapiDecodeTiming::add_elapsed(&mut self.timing.output_mapping_ns, output_start);
        let release_start = Instant::now();
        self.release_decoded_surfaces(&update.released);
        VaapiDecodeTiming::add_elapsed(&mut self.timing.release_ns, release_start);
        self.update_submitted_unsynced_peak();
        VaapiDecodeTiming::add_elapsed(&mut self.timing.submit_total_ns, submit_start);
        Ok(outputs)
    }

    fn flush(
        &mut self,
        _request: &qgs_protocol::FlushDecoderRequest,
    ) -> Result<Vec<BackendDecodedSurface>, DecoderError> {
        let flush_start = Instant::now();
        let update = self.frontend.flush();
        if self.mode.diagnostics_enabled() {
            eprintln!(
                "qgs-vaapi diagnostic: H.264 flush DPB occupancy={} output_pending={} max_live_va_surfaces={}",
                update.max_dpb_occupancy, update.max_output_pending, self.max_live_surfaces
            );
        }
        let outputs = self.outputs_from_ids(&update.output_ready)?;
        self.release_decoded_surfaces(&update.released);
        self.update_submitted_unsynced_peak();
        VaapiDecodeTiming::add_elapsed(&mut self.timing.flush_ns, flush_start);
        Ok(outputs)
    }
}

impl VaapiH264Decoder {
    fn release_decoded_surfaces(&mut self, released: &[H264PictureId]) {
        for id in released {
            let Some(state) = self.decoded_surfaces.remove(&surface_key(id)) else {
                continue;
            };
            state.codec_released.set(true);
            if Rc::strong_count(&state) == 1 {
                state.recycle_if_released();
            } else {
                self.pool.borrow_mut().record_deferred_recycle();
            }
        }
    }

    fn outputs_from_ids(
        &self,
        output_ready: &[H264PictureId],
    ) -> Result<Vec<BackendDecodedSurface>, DecoderError> {
        output_ready
            .iter()
            .map(|id| {
                let surface = self
                    .decoded_surfaces
                    .get(&surface_key(id))
                    .ok_or(DecoderError::DecodeFailed)?;
                Ok(BackendDecodedSurface {
                    resource: Box::new(VaapiVideoSurface {
                        state: Rc::clone(surface),
                    }),
                    desc: surface.desc.clone(),
                })
            })
            .collect()
    }

    fn update_submitted_unsynced_peak(&mut self) {
        let live_submitted = self
            .decoded_surfaces
            .values()
            .filter(|surface| surface.is_submitted_unsynced())
            .count();
        let pending_submitted = self.pool.borrow().pending_submitted_unsynced();
        self.max_submitted_unsynced_surfaces = self
            .max_submitted_unsynced_surfaces
            .max(live_submitted.saturating_add(pending_submitted));
    }
}

struct DecodedSurfaceState {
    pool: Rc<RefCell<VaapiSurfacePool>>,
    codec_released: Cell<bool>,
    surface: RefCell<Option<DecodedVaSurface>>,
    desc: VideoSurfaceDesc,
    diagnostics: VaapiSurfaceDiagnostics,
}

impl DecodedSurfaceState {
    fn surface_id(&self) -> Result<u32, DecoderError> {
        self.surface
            .borrow()
            .as_ref()
            .map(DecodedVaSurface::surface_id)
            .ok_or(DecoderError::DecodeFailed)
    }

    fn is_submitted_unsynced(&self) -> bool {
        self.surface
            .borrow()
            .as_ref()
            .is_some_and(DecodedVaSurface::is_submitted)
    }

    fn transfer_nv12(
        &self,
        mut destination: CpuNv12Surface,
    ) -> Result<(CpuNv12Surface, CpuNv12TransferTiming), DecoderError> {
        let transfer_started = Instant::now();
        let mut timing = CpuNv12TransferTiming::default();
        if destination.desc.coded_width != self.desc.coded_width
            || destination.desc.coded_height != self.desc.coded_height
            || destination.desc.visible_region != self.desc.visible_region
            || destination.desc.format != VideoSurfaceFormat::Nv12
        {
            return Err(DecoderError::DecodeFailed);
        }
        let mut stored = self.surface.borrow_mut();
        let decoded = stored.take().ok_or(DecoderError::DecodeFailed)?;
        let sync_started = Instant::now();
        let ready = decoded.into_ready_for_cpu()?;
        timing.sync_ns = sync_started.elapsed().as_nanos();
        copy_nv12_surface_to_cpu(&ready, &mut destination, &mut timing)?;
        *stored = Some(DecodedVaSurface::Ready(ready));
        timing.total_ns = transfer_started.elapsed().as_nanos();
        Ok((destination, timing))
    }

    fn describe_nv12_image(&self) -> Result<VaNv12ImageLayout, DecoderError> {
        let mut stored = self.surface.borrow_mut();
        let decoded = stored.take().ok_or(DecoderError::DecodeFailed)?;
        let ready = decoded.into_ready_for_cpu()?;
        let layout =
            describe_nv12_surface_layout(&ready, self.desc.coded_width, self.desc.coded_height)?;
        *stored = Some(DecodedVaSurface::Ready(ready));
        Ok(layout)
    }

    fn recycle_if_released(&self) {
        if !self.codec_released.get() {
            return;
        }
        if let Some(surface) = self.surface.borrow_mut().take() {
            self.pool.borrow_mut().defer_recycle(surface);
        }
    }
}

fn nv12_surface_desc(
    coded_width: u32,
    coded_height: u32,
    visible_region: VisibleRegion,
) -> Result<VideoSurfaceDesc, DecoderError> {
    let desc = VideoSurfaceDesc {
        coded_width,
        coded_height,
        visible_region,
        format: VideoSurfaceFormat::Nv12,
        bit_depth: BitDepth::new(8).map_err(DecoderError::from)?,
        chroma: ChromaSubsampling::Cs420,
        scan_mode: qgs_protocol::ScanMode::Progressive,
        field_order: qgs_protocol::FieldOrder::Unknown,
    };
    desc.validate().map_err(DecoderError::from)?;
    Ok(desc)
}

fn copy_nv12_surface_to_cpu(
    surface: &Surface<()>,
    destination: &mut CpuNv12Surface,
    timing: &mut CpuNv12TransferTiming,
) -> Result<(), DecoderError> {
    let coded_width = destination.desc.coded_width;
    let coded_height = destination.desc.coded_height;
    let image_started = Instant::now();
    let image = Image::derive_from(surface, (coded_width, coded_height))
        .or_else(|_| {
            Image::create_from(
                surface,
                nv12_image_format(),
                (coded_width, coded_height),
                (coded_width, coded_height),
            )
        })
        .map_err(|_| DecoderError::DecodeFailed)?;
    timing.image_create_ns = image_started.elapsed().as_nanos();
    let va_image = image.image();
    if va_image.num_planes < 2 {
        return Err(DecoderError::DecodeFailed);
    }
    timing.image_layout = Some(va_image_layout(&image));
    let copy_y_started = Instant::now();
    timing.bytes_copied = timing.bytes_copied.saturating_add(copy_nv12_plane(
        image.as_ref(),
        va_image.offsets[0] as usize,
        va_image.pitches[0] as usize,
        &mut destination.y,
    )?);
    timing.copy_y_ns = copy_y_started.elapsed().as_nanos();
    let copy_uv_started = Instant::now();
    timing.bytes_copied = timing.bytes_copied.saturating_add(copy_nv12_plane(
        image.as_ref(),
        va_image.offsets[1] as usize,
        va_image.pitches[1] as usize,
        &mut destination.uv,
    )?);
    timing.copy_uv_ns = copy_uv_started.elapsed().as_nanos();
    let release_started = Instant::now();
    drop(image);
    timing.image_release_ns = release_started.elapsed().as_nanos();
    Ok(())
}

fn describe_nv12_surface_layout(
    surface: &Surface<()>,
    coded_width: u32,
    coded_height: u32,
) -> Result<VaNv12ImageLayout, DecoderError> {
    let image = Image::derive_from(surface, (coded_width, coded_height))
        .or_else(|_| {
            Image::create_from(
                surface,
                nv12_image_format(),
                (coded_width, coded_height),
                (coded_width, coded_height),
            )
        })
        .map_err(|_| DecoderError::DecodeFailed)?;
    let layout = va_image_layout(&image);
    if layout.num_planes < 2 {
        return Err(DecoderError::DecodeFailed);
    }
    Ok(layout)
}

fn va_image_layout(image: &Image<'_>) -> VaNv12ImageLayout {
    let va_image = image.image();
    VaNv12ImageLayout {
        fourcc: va_image.format.fourcc,
        width: u32::from(va_image.width),
        height: u32::from(va_image.height),
        num_planes: u32::from(va_image.num_planes),
        offsets: [
            va_image.offsets[0] as usize,
            va_image.offsets[1] as usize,
            va_image.offsets[2] as usize,
        ],
        pitches: [
            va_image.pitches[0] as usize,
            va_image.pitches[1] as usize,
            va_image.pitches[2] as usize,
        ],
        data_size: va_image.data_size as usize,
        derived: image.is_derived(),
    }
}

fn copy_nv12_plane(
    source: &[u8],
    source_offset: usize,
    source_pitch: usize,
    destination: &mut CpuNv12Plane,
) -> Result<usize, DecoderError> {
    if source_pitch < destination.width_bytes
        || destination.stride_bytes < destination.width_bytes
        || destination.height == 0
    {
        return Err(DecoderError::DecodeFailed);
    }
    let required_destination = destination
        .stride_bytes
        .checked_mul(destination.height)
        .ok_or(DecoderError::DecodeFailed)?;
    if destination.data.len() < required_destination {
        return Err(DecoderError::DecodeFailed);
    }
    if source_pitch == destination.stride_bytes
        && destination.width_bytes == destination.stride_bytes
    {
        let byte_count = destination
            .stride_bytes
            .checked_mul(destination.height)
            .ok_or(DecoderError::DecodeFailed)?;
        let source_end = source_offset
            .checked_add(byte_count)
            .ok_or(DecoderError::DecodeFailed)?;
        let source_slice = source
            .get(source_offset..source_end)
            .ok_or(DecoderError::DecodeFailed)?;
        let destination_slice = destination
            .data
            .get_mut(0..byte_count)
            .ok_or(DecoderError::DecodeFailed)?;
        destination_slice.copy_from_slice(source_slice);
        return Ok(byte_count);
    }
    let mut copied = 0_usize;
    for row in 0..destination.height {
        let source_start = source_offset
            .checked_add(
                row.checked_mul(source_pitch)
                    .ok_or(DecoderError::DecodeFailed)?,
            )
            .ok_or(DecoderError::DecodeFailed)?;
        let source_end = source_start
            .checked_add(destination.width_bytes)
            .ok_or(DecoderError::DecodeFailed)?;
        let destination_start = row
            .checked_mul(destination.stride_bytes)
            .ok_or(DecoderError::DecodeFailed)?;
        let destination_end = destination_start
            .checked_add(destination.width_bytes)
            .ok_or(DecoderError::DecodeFailed)?;
        let source_row = source
            .get(source_start..source_end)
            .ok_or(DecoderError::DecodeFailed)?;
        let destination_row = destination
            .data
            .get_mut(destination_start..destination_end)
            .ok_or(DecoderError::DecodeFailed)?;
        destination_row.copy_from_slice(source_row);
        copied = copied.saturating_add(destination.width_bytes);
    }
    Ok(copied)
}

impl Drop for DecodedSurfaceState {
    fn drop(&mut self) {
        if !self.codec_released.get() {
            return;
        }
        if let Some(surface) = self.surface.get_mut().take() {
            self.pool.borrow_mut().defer_recycle(surface);
        }
    }
}

struct VaapiVideoSurface {
    state: Rc<DecodedSurfaceState>,
}

impl BackendResource for VaapiVideoSurface {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub fn surface_diagnostics(resource: &dyn BackendResource) -> Option<VaapiSurfaceDiagnostics> {
    resource
        .as_any()
        .downcast_ref::<VaapiVideoSurface>()
        .map(|surface| surface.state.diagnostics.clone())
}

fn collect_surface_diagnostics(
    surface: &Surface<()>,
    width: u32,
    height: u32,
) -> Result<VaapiSurfaceDiagnostics, DecoderError> {
    let validation_checksum =
        validation_checksum(surface, width, height).map_err(|_| DecoderError::DecodeFailed)?;
    let export_probe = surface
        .export_prime()
        .ok()
        .map(|descriptor| VaapiExportProbe {
            object_count: descriptor.objects.len(),
            layer_count: descriptor.layers.len(),
            fourcc: descriptor.fourcc,
            width: descriptor.width,
            height: descriptor.height,
            object_sizes: descriptor
                .objects
                .iter()
                .map(|object| object.size)
                .collect(),
            modifiers: descriptor
                .objects
                .iter()
                .map(|object| object.drm_format_modifier)
                .collect(),
            layer_formats: descriptor
                .layers
                .iter()
                .map(|layer| layer.drm_format)
                .collect(),
            pitches: descriptor.layers.iter().map(|layer| layer.pitch).collect(),
            offsets: descriptor.layers.iter().map(|layer| layer.offset).collect(),
        });

    Ok(VaapiSurfaceDiagnostics {
        validation_checksum: Some(validation_checksum),
        export_probe,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VaapiExportProbe {
    pub object_count: usize,
    pub layer_count: usize,
    pub fourcc: u32,
    pub width: u32,
    pub height: u32,
    pub object_sizes: Vec<u32>,
    pub modifiers: Vec<u64>,
    pub layer_formats: Vec<u32>,
    pub pitches: Vec<[u32; 4]>,
    pub offsets: Vec<[u32; 4]>,
}

enum DecodedVaSurface {
    Submitted(Picture<PictureEnd, Surface<()>>),
    Ready(Surface<()>),
}

impl DecodedVaSurface {
    fn surface_id(&self) -> u32 {
        match self {
            Self::Submitted(picture) => picture.surface().id(),
            Self::Ready(surface) => surface.id(),
        }
    }

    fn as_surface(&self) -> &Surface<()> {
        match self {
            Self::Submitted(picture) => picture.surface(),
            Self::Ready(surface) => surface,
        }
    }

    fn is_submitted(&self) -> bool {
        matches!(self, Self::Submitted(_))
    }

    fn into_ready_for_cpu(self) -> Result<Surface<()>, DecoderError> {
        match self {
            Self::Submitted(picture) => picture
                .sync()
                .map_err(|_| DecoderError::DecodeFailed)?
                .take_surface()
                .map_err(|_| DecoderError::DecodeFailed),
            Self::Ready(surface) => {
                surface.sync().map_err(|_| DecoderError::DecodeFailed)?;
                Ok(surface)
            }
        }
    }

    fn reclaim_for_reuse(self, timing: Option<&mut VaapiDecodeTiming>) -> Option<Surface<()>> {
        match self {
            Self::Submitted(picture) => {
                let sync_start = Instant::now();
                let synced = picture.sync().ok()?;
                if let Some(timing) = timing {
                    VaapiDecodeTiming::add_elapsed(&mut timing.reclaim_sync_ns, sync_start);
                }
                synced.take_surface().ok()
            }
            Self::Ready(surface) => {
                let sync_start = Instant::now();
                surface.sync().ok()?;
                if let Some(timing) = timing {
                    VaapiDecodeTiming::add_elapsed(&mut timing.reclaim_sync_ns, sync_start);
                }
                Some(surface)
            }
        }
    }
}

struct VaapiSurfacePool {
    available: Vec<Surface<()>>,
    pending_recycle: Vec<DecodedVaSurface>,
    stats: VaapiSurfacePoolStats,
}

impl VaapiSurfacePool {
    fn new(surfaces: Vec<Surface<()>>) -> Self {
        let surfaces_allocated = surfaces.len();
        Self {
            available: surfaces,
            pending_recycle: Vec::new(),
            stats: VaapiSurfacePoolStats {
                surfaces_allocated,
                minimum_free_surfaces: surfaces_allocated,
                ..VaapiSurfacePoolStats::default()
            },
        }
    }

    fn acquire(&mut self, timing: &mut VaapiDecodeTiming) -> Option<Surface<()>> {
        let surface = match self.available.pop() {
            Some(surface) => surface,
            None => self.reclaim_one_pending(Some(timing))?,
        };
        let was_recycled = self.stats.surfaces_recycled > 0;
        if was_recycled {
            self.stats.surface_reuse_count = self.stats.surface_reuse_count.saturating_add(1);
        }
        let checked_out = self
            .stats
            .surfaces_allocated
            .saturating_sub(self.available.len());
        self.stats.peak_checked_out_surfaces =
            self.stats.peak_checked_out_surfaces.max(checked_out);
        self.stats.minimum_free_surfaces =
            self.stats.minimum_free_surfaces.min(self.available.len());
        Some(surface)
    }

    fn defer_recycle(&mut self, surface: DecodedVaSurface) {
        self.stats.deferred_recycle_count = self.stats.deferred_recycle_count.saturating_add(1);
        self.pending_recycle.push(surface);
        self.stats.peak_pending_recycle_surfaces = self
            .stats
            .peak_pending_recycle_surfaces
            .max(self.pending_recycle.len());
    }

    fn reclaim_one_pending(
        &mut self,
        timing: Option<&mut VaapiDecodeTiming>,
    ) -> Option<Surface<()>> {
        let surface = self.pending_recycle.pop()?;
        if let Some(surface) = surface.reclaim_for_reuse(timing) {
            self.stats.recycle_sync_count = self.stats.recycle_sync_count.saturating_add(1);
            self.stats.surfaces_recycled = self.stats.surfaces_recycled.saturating_add(1);
            Some(surface)
        } else {
            None
        }
    }

    fn record_deferred_recycle(&mut self) {
        self.stats.deferred_recycle_count = self.stats.deferred_recycle_count.saturating_add(1);
    }

    fn pending_submitted_unsynced(&self) -> usize {
        self.pending_recycle
            .iter()
            .filter(|surface| surface.is_submitted())
            .count()
    }
}

impl Drop for VaapiSurfacePool {
    fn drop(&mut self) {
        while self.reclaim_one_pending(None).is_some() {}
    }
}

fn decode_h264_access_unit(
    context: Rc<Context>,
    surface: Surface<()>,
    parsed: &ParsedH264AccessUnit,
    decoded_surfaces: &BTreeMap<(u16, i32), Rc<DecodedSurfaceState>>,
    sync_after_end: bool,
    timing: &mut VaapiDecodeTiming,
) -> Result<DecodedVaSurface, DecoderError> {
    let mut picture = Picture::new(0, Rc::clone(&context), surface);
    let parameter_start = Instant::now();
    let pic_param = picture_parameter(parsed, picture.surface().id(), decoded_surfaces)?;
    let slice_params = slice_parameters(parsed, decoded_surfaces)?;
    VaapiDecodeTiming::add_elapsed(&mut timing.parameter_build_ns, parameter_start);
    let buffer_start = Instant::now();
    picture.add_buffer(
        context
            .create_buffer(BufferType::PictureParameter(PictureParameter::H264(
                pic_param,
            )))
            .map_err(|_| DecoderError::DecodeFailed)?,
    );
    picture.add_buffer(
        context
            .create_buffer(BufferType::IQMatrix(IQMatrix::H264(
                IQMatrixBufferH264::new(
                    parsed.picture.scaling_list4x4,
                    parsed.picture.scaling_list8x8,
                ),
            )))
            .map_err(|_| DecoderError::DecodeFailed)?,
    );
    for (slice_param, slice) in slice_params.into_iter().zip(&parsed.slices) {
        picture.add_buffer(
            context
                .create_buffer(BufferType::SliceParameter(SliceParameter::H264(
                    slice_param,
                )))
                .map_err(|_| DecoderError::DecodeFailed)?,
        );
        picture.add_buffer(
            context
                .create_buffer_borrowed(BorrowedBufferType::SliceData(&slice.nal_bytes))
                .map_err(|_| DecoderError::DecodeFailed)?,
        );
    }
    VaapiDecodeTiming::add_elapsed(&mut timing.buffer_create_ns, buffer_start);

    let begin_start = Instant::now();
    let picture = picture.begin().map_err(|_| DecoderError::DecodeFailed)?;
    VaapiDecodeTiming::add_elapsed(&mut timing.begin_picture_ns, begin_start);
    let render_start = Instant::now();
    let picture = picture.render().map_err(|_| DecoderError::DecodeFailed)?;
    VaapiDecodeTiming::add_elapsed(&mut timing.render_picture_ns, render_start);
    let end_start = Instant::now();
    let picture = picture.end().map_err(|_| DecoderError::DecodeFailed)?;
    VaapiDecodeTiming::add_elapsed(&mut timing.end_picture_ns, end_start);
    if sync_after_end {
        let sync_start = Instant::now();
        let surface = picture
            .sync()
            .map_err(|_| DecoderError::DecodeFailed)?
            .take_surface()
            .map_err(|_| DecoderError::DecodeFailed)?;
        VaapiDecodeTiming::add_elapsed(&mut timing.inline_sync_ns, sync_start);
        Ok(DecodedVaSurface::Ready(surface))
    } else {
        Ok(DecodedVaSurface::Submitted(picture))
    }
}

fn picture_parameter(
    parsed: &ParsedH264AccessUnit,
    surface_id: u32,
    decoded_surfaces: &BTreeMap<(u16, i32), Rc<DecodedSurfaceState>>,
) -> Result<PictureParameterBufferH264, DecoderError> {
    let picture = &parsed.picture;
    let current = PictureH264::new(
        surface_id,
        u32::from(picture.frame_num),
        h264_picture_flags(picture.reference_pic_flag),
        picture.top_field_order_cnt,
        picture.bottom_field_order_cnt,
    );
    let references = reference_frames(parsed, decoded_surfaces)?;
    let seq_fields = H264SeqFields::new(
        picture.chroma_format_idc,
        0,
        picture.gaps_in_frame_num_value_allowed_flag as u32,
        picture.frame_mbs_only_flag as u32,
        picture.mb_adaptive_frame_field_flag as u32,
        picture.direct_8x8_inference_flag as u32,
        0,
        u32::from(picture.log2_max_frame_num_minus4),
        picture.pic_order_cnt_type,
        picture.log2_max_pic_order_cnt_lsb_minus4,
        picture.delta_pic_order_always_zero_flag as u32,
    );
    let pic_fields = H264PicFields::new(
        picture.entropy_coding_mode_flag as u32,
        picture.weighted_pred_flag as u32,
        u32::from(picture.weighted_bipred_idc),
        picture.transform_8x8_mode_flag as u32,
        picture.field_pic_flag as u32,
        picture.constrained_intra_pred_flag as u32,
        picture.pic_order_present_flag as u32,
        picture.deblocking_filter_control_present_flag as u32,
        picture.redundant_pic_cnt_present_flag as u32,
        picture.reference_pic_flag as u32,
    );
    Ok(PictureParameterBufferH264::new(
        current,
        references,
        picture.picture_width_in_mbs_minus1,
        picture.picture_height_in_mbs_minus1,
        picture.bit_depth_luma_minus8,
        picture.bit_depth_chroma_minus8,
        picture.num_ref_frames,
        &seq_fields,
        picture.num_slice_groups_minus1,
        picture.slice_group_map_type,
        picture.slice_group_change_rate_minus1,
        picture.pic_init_qp_minus26,
        picture.pic_init_qs_minus26,
        picture.chroma_qp_index_offset,
        picture.second_chroma_qp_index_offset,
        &pic_fields,
        picture.frame_num,
    ))
}

fn slice_parameters(
    parsed: &ParsedH264AccessUnit,
    decoded_surfaces: &BTreeMap<(u16, i32), Rc<DecodedSurfaceState>>,
) -> Result<Vec<SliceParameterBufferH264>, DecoderError> {
    let mut buffers = Vec::with_capacity(parsed.slices.len());
    for slice in &parsed.slices {
        let mut params = SliceParameterBufferH264::new_array();
        let ref_list0 = reference_list_array(&slice.ref_pic_list0, decoded_surfaces)?;
        let ref_list1 = reference_list_array(&slice.ref_pic_list1, decoded_surfaces)?;
        params.add_slice_parameter(
            slice.nal_bytes.len() as u32,
            0,
            0,
            slice.slice_data_bit_offset,
            slice.first_mb_in_slice,
            slice.slice_type,
            slice.direct_spatial_mv_pred_flag,
            slice.num_ref_idx_l0_active_minus1,
            slice.num_ref_idx_l1_active_minus1,
            slice.cabac_init_idc,
            slice.slice_qp_delta,
            slice.disable_deblocking_filter_idc,
            slice.slice_alpha_c0_offset_div2,
            slice.slice_beta_offset_div2,
            ref_list0,
            ref_list1,
            0,
            0,
            0,
            [0; 32],
            [0; 32],
            0,
            [[0; 2]; 32],
            [[0; 2]; 32],
            0,
            [0; 32],
            [0; 32],
            0,
            [[0; 2]; 32],
            [[0; 2]; 32],
        );
        buffers.push(params);
    }
    Ok(buffers)
}

fn reference_frames(
    parsed: &ParsedH264AccessUnit,
    decoded_surfaces: &BTreeMap<(u16, i32), Rc<DecodedSurfaceState>>,
) -> Result<[PictureH264; 16], DecoderError> {
    let mut references = invalid_picture_array_16();
    for (index, reference) in parsed.reference_frames.iter().take(16).enumerate() {
        references[index] = picture_from_reference(reference, decoded_surfaces)?;
    }
    Ok(references)
}

fn reference_list_array(
    refs: &[H264PictureId],
    decoded_surfaces: &BTreeMap<(u16, i32), Rc<DecodedSurfaceState>>,
) -> Result<[PictureH264; 32], DecoderError> {
    let mut pictures = invalid_picture_array_32();
    for (index, id) in refs.iter().take(32).enumerate() {
        let Some(surface) = decoded_surfaces.get(&surface_key(id)) else {
            return Err(DecoderError::DecodeFailed);
        };
        pictures[index] = PictureH264::new(
            surface.surface_id()?,
            u32::from(id.frame_num),
            h264_picture_flags(true),
            id.poc,
            id.poc,
        );
    }
    Ok(pictures)
}

fn picture_from_reference(
    reference: &ParsedH264Reference,
    decoded_surfaces: &BTreeMap<(u16, i32), Rc<DecodedSurfaceState>>,
) -> Result<PictureH264, DecoderError> {
    let Some(surface) = decoded_surfaces.get(&surface_key(&reference.id)) else {
        return Err(DecoderError::DecodeFailed);
    };
    Ok(PictureH264::new(
        surface.surface_id()?,
        u32::from(reference.frame_num),
        h264_picture_flags(true),
        reference.top_field_order_cnt,
        reference.bottom_field_order_cnt,
    ))
}

fn h264_picture_flags(short_term_reference: bool) -> u32 {
    if short_term_reference {
        libva::VA_PICTURE_H264_SHORT_TERM_REFERENCE
    } else {
        0
    }
}

fn surface_key(id: &H264PictureId) -> (u16, i32) {
    (id.frame_num, id.poc)
}

fn invalid_picture() -> PictureH264 {
    PictureH264::new(u32::MAX, 0, 1, 0, 0)
}

fn invalid_picture_array_16() -> [PictureH264; 16] {
    std::array::from_fn(|_| invalid_picture())
}

fn invalid_picture_array_32() -> [PictureH264; 32] {
    std::array::from_fn(|_| invalid_picture())
}

fn validation_checksum(
    surface: &Surface<()>,
    width: u32,
    height: u32,
) -> Result<u32, Box<dyn std::error::Error>> {
    let image = Image::derive_from(surface, (width, height)).or_else(|_| {
        Image::create_from(
            surface,
            nv12_image_format(),
            (width, height),
            (width, height),
        )
    })?;
    let data = image.as_ref();
    let va_image = image.image();
    if va_image.num_planes < 2 {
        return Err("decoded validation image did not expose NV12 planes".into());
    }
    let width = width as usize;
    let height = height as usize;
    let mut hash = 2166136261_u32;
    for plane in 0..2 {
        let rows = if plane == 0 {
            height
        } else {
            height.div_ceil(2)
        };
        let offset = va_image.offsets[plane] as usize;
        let pitch = va_image.pitches[plane] as usize;
        for row in 0..rows {
            let start = offset + row * pitch;
            let end = start + width;
            for byte in &data[start..end] {
                hash ^= u32::from(*byte);
                hash = hash.wrapping_mul(16777619);
            }
        }
    }
    if hash == 2166136261 {
        return Err("decoded validation checksum was empty".into());
    }
    Ok(hash)
}

fn nv12_image_format() -> libva::VAImageFormat {
    libva::VAImageFormat {
        fourcc: VA_FOURCC_NV12,
        byte_order: libva::VA_LSB_FIRST,
        bits_per_pixel: 12,
        depth: 12,
        red_mask: 0,
        green_mask: 0,
        blue_mask: 0,
        alpha_mask: 0,
        va_reserved: [0; 4],
    }
}

fn va_profile_from_h264(profile: VideoProfile) -> Result<VAProfile::Type, DecoderError> {
    match profile {
        VideoProfile::H264(H264Profile::Baseline) => {
            Ok(VAProfile::VAProfileH264ConstrainedBaseline)
        }
        VideoProfile::H264(H264Profile::Main) => Ok(VAProfile::VAProfileH264Main),
        VideoProfile::H264(H264Profile::High) => Ok(VAProfile::VAProfileH264High),
        _ => Err(DecoderError::UnsupportedDecodeConfiguration),
    }
}

fn decoder_error_from_h264(error: H264Error) -> DecoderError {
    match error {
        H264Error::MalformedAnnexB
        | H264Error::MissingSps
        | H264Error::MissingPps
        | H264Error::MissingSlice
        | H264Error::Parser(_) => DecoderError::MalformedCompressedData,
        H264Error::UnsupportedFeature(_) => DecoderError::UnsupportedH264StreamFeature,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RenderNode {
    path: PathBuf,
    vendor_id: u32,
    device_id: u32,
    driver: Option<String>,
}

fn enumerate_render_nodes() -> Vec<RenderNode> {
    let Ok(entries) = std::fs::read_dir("/dev/dri") else {
        return Vec::new();
    };

    let mut nodes = entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let file_name = entry.file_name();
            let file_name = file_name.to_string_lossy();
            if !file_name.starts_with("renderD") {
                return None;
            }
            render_node_from_path(entry.path())
        })
        .collect::<Vec<_>>();
    nodes.sort_by(|left, right| left.path.cmp(&right.path));
    nodes
}

fn render_node_from_path(path: PathBuf) -> Option<RenderNode> {
    let name = path.file_name()?.to_string_lossy();
    let sysfs_device = PathBuf::from("/sys/class/drm")
        .join(name.as_ref())
        .join("device");
    let vendor_id = read_hex_u32(&sysfs_device.join("vendor"))?;
    let device_id = read_hex_u32(&sysfs_device.join("device"))?;
    let driver = std::fs::read_link(sysfs_device.join("driver"))
        .ok()
        .and_then(|path| {
            path.file_name()
                .map(|name| name.to_string_lossy().into_owned())
        });

    Some(RenderNode {
        path,
        vendor_id,
        device_id,
        driver,
    })
}

fn read_hex_u32(path: &Path) -> Option<u32> {
    let value = std::fs::read_to_string(path).ok()?;
    u32::from_str_radix(value.trim().trim_start_matches("0x"), 16).ok()
}

fn query_decode_capabilities(
    display: &Display,
) -> Result<Vec<VideoDecodeCapability>, Box<dyn std::error::Error>> {
    let profiles = display.query_config_profiles()?;
    let image_formats = display.query_image_formats().unwrap_or_default();
    let fourccs = image_formats
        .iter()
        .map(|format| format.fourcc)
        .collect::<Vec<_>>();
    let mut capabilities = Vec::new();

    for profile in profiles {
        let Some(mapped_profile) = map_profile(profile) else {
            continue;
        };
        let entrypoints = display.query_config_entrypoints(profile)?;
        if !entrypoints.contains(&VAEntrypoint::VAEntrypointVLD) {
            continue;
        }

        let mut attrs = vec![
            VAConfigAttrib {
                type_: VAConfigAttribType::VAConfigAttribRTFormat,
                value: 0,
            },
            VAConfigAttrib {
                type_: VAConfigAttribType::VAConfigAttribMaxPictureWidth,
                value: 0,
            },
            VAConfigAttrib {
                type_: VAConfigAttribType::VAConfigAttribMaxPictureHeight,
                value: 0,
            },
        ];
        display.get_config_attributes(profile, VAEntrypoint::VAEntrypointVLD, &mut attrs)?;
        let rt_format = attr_value(&attrs, VAConfigAttribType::VAConfigAttribRTFormat)
            .filter(|value| *value != VA_ATTRIB_NOT_SUPPORTED)
            .unwrap_or(0);
        let mut output_formats = output_formats_from_rt_and_fourcc(rt_format, &fourccs);
        output_formats.sort_by_key(|format| format.wire_value());
        output_formats.dedup();
        if output_formats.is_empty() {
            continue;
        }

        let max_width = attr_value(&attrs, VAConfigAttribType::VAConfigAttribMaxPictureWidth)
            .filter(|value| *value != VA_ATTRIB_NOT_SUPPORTED && *value != 0)
            .unwrap_or(DEFAULT_MAX_WIDTH);
        let max_height = attr_value(&attrs, VAConfigAttribType::VAConfigAttribMaxPictureHeight)
            .filter(|value| *value != VA_ATTRIB_NOT_SUPPORTED && *value != 0)
            .unwrap_or(DEFAULT_MAX_HEIGHT);

        let (codec, profile, bit_depth, chroma) = match mapped_profile {
            MappedProfile::H264(profile) => (
                qgs_protocol::VideoCodec::H264,
                VideoProfile::H264(profile),
                BitDepth::new(8)?,
                ChromaSubsampling::Cs420,
            ),
            MappedProfile::Mpeg2(profile) => (
                qgs_protocol::VideoCodec::Mpeg2,
                VideoProfile::Mpeg2(profile),
                BitDepth::new(8)?,
                ChromaSubsampling::Cs420,
            ),
        };

        let capability = VideoDecodeCapability {
            codec,
            profile,
            bit_depth,
            chroma,
            max_width,
            max_height,
            progressive_supported: true,
            interlaced_supported: false,
            output_surface_formats: output_formats,
        };
        capability.validate()?;
        push_unique_capability(&mut capabilities, capability);
    }

    Ok(capabilities)
}

fn attr_value(attrs: &[VAConfigAttrib], attr_type: VAConfigAttribType::Type) -> Option<u32> {
    attrs
        .iter()
        .find(|attr| attr.type_ == attr_type)
        .map(|attr| attr.value)
}

fn output_formats_from_rt_and_fourcc(rt_format: u32, fourccs: &[u32]) -> Vec<VideoSurfaceFormat> {
    let mut formats = Vec::new();
    if rt_format & VA_RT_FORMAT_YUV420 != 0 && fourccs.contains(&VA_FOURCC_NV12) {
        formats.push(VideoSurfaceFormat::Nv12);
    }
    if rt_format & VA_RT_FORMAT_YUV420_10 != 0 && fourccs.contains(&VA_FOURCC_P010) {
        formats.push(VideoSurfaceFormat::P010);
    }
    if rt_format & VA_RT_FORMAT_YUV422 != 0
        && (fourccs.contains(&VA_FOURCC_YUY2) || fourccs.contains(&VA_FOURCC_UYVY))
    {
        formats.push(VideoSurfaceFormat::Yuv422_8);
    }
    if rt_format & VA_RT_FORMAT_YUV422_10 != 0 && fourccs.contains(&VA_FOURCC_Y210) {
        formats.push(VideoSurfaceFormat::Yuv422_10);
    }
    formats
}

fn push_unique_capability(
    capabilities: &mut Vec<VideoDecodeCapability>,
    capability: VideoDecodeCapability,
) {
    if !capabilities.iter().any(|existing| existing == &capability) {
        capabilities.push(capability);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MappedProfile {
    H264(H264Profile),
    Mpeg2(Mpeg2Profile),
}

fn map_profile(profile: VAProfile::Type) -> Option<MappedProfile> {
    match profile {
        VAProfile::VAProfileH264ConstrainedBaseline => {
            Some(MappedProfile::H264(H264Profile::Baseline))
        }
        VAProfile::VAProfileH264Main => Some(MappedProfile::H264(H264Profile::Main)),
        VAProfile::VAProfileH264High => Some(MappedProfile::H264(H264Profile::High)),
        VAProfile::VAProfileMPEG2Main => Some(MappedProfile::Mpeg2(Mpeg2Profile::Main)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROFESSIONAL_INTRA_FIXTURE: &[u8] =
        include_bytes!("../../../tests/fixtures/h264/professional-422-10bit-idr-128x72.h264");

    #[test]
    fn normal_decode_mode_disables_diagnostics() {
        assert!(!VaapiDecodeMode::Normal.diagnostics_enabled());
        assert!(VaapiDecodeMode::Diagnostic.diagnostics_enabled());
    }

    #[test]
    fn surface_pool_stats_start_empty() {
        let stats = VaapiSurfacePoolStats::default();
        assert_eq!(stats.surfaces_allocated, 0);
        assert_eq!(stats.surface_reuse_count, 0);
        assert_eq!(stats.surfaces_recycled, 0);
        assert_eq!(stats.recycle_sync_count, 0);
        assert_eq!(stats.deferred_recycle_count, 0);
        assert_eq!(stats.peak_checked_out_surfaces, 0);
        assert_eq!(stats.peak_pending_recycle_surfaces, 0);
        assert_eq!(stats.minimum_free_surfaces, 0);
    }

    #[test]
    fn maps_va_h264_profiles_to_qgs_profiles() {
        assert_eq!(
            map_profile(VAProfile::VAProfileH264ConstrainedBaseline),
            Some(MappedProfile::H264(H264Profile::Baseline))
        );
        assert_eq!(
            map_profile(VAProfile::VAProfileH264Main),
            Some(MappedProfile::H264(H264Profile::Main))
        );
        assert_eq!(
            map_profile(VAProfile::VAProfileH264High),
            Some(MappedProfile::H264(H264Profile::High))
        );
    }

    #[test]
    fn maps_va_mpeg2_profile_to_qgs_profile() {
        assert_eq!(
            map_profile(VAProfile::VAProfileMPEG2Main),
            Some(MappedProfile::Mpeg2(Mpeg2Profile::Main))
        );
    }

    #[test]
    fn unsupported_va_profiles_are_ignored() {
        assert_eq!(map_profile(VAProfile::VAProfileJPEGBaseline), None);
        assert_eq!(map_profile(VAProfile::VAProfileNone), None);
    }

    #[test]
    fn decode_entrypoint_is_distinct_from_encode_entrypoints() {
        assert_ne!(
            VAEntrypoint::VAEntrypointVLD,
            VAEntrypoint::VAEntrypointEncSlice
        );
    }

    #[test]
    fn translates_surface_formats_from_rt_format_and_fourcc() {
        let formats = output_formats_from_rt_and_fourcc(
            VA_RT_FORMAT_YUV420 | VA_RT_FORMAT_YUV422 | VA_RT_FORMAT_YUV420_10,
            &[VA_FOURCC_NV12, VA_FOURCC_YUY2, VA_FOURCC_P010],
        );

        assert_eq!(
            formats,
            vec![
                VideoSurfaceFormat::Nv12,
                VideoSurfaceFormat::P010,
                VideoSurfaceFormat::Yuv422_8,
            ]
        );
    }

    #[test]
    fn ignores_rt_format_without_matching_surface_fourcc() {
        assert!(output_formats_from_rt_and_fourcc(VA_RT_FORMAT_YUV422_10, &[]).is_empty());
    }

    #[test]
    fn duplicate_capabilities_are_not_inserted() {
        let capability = VideoDecodeCapability {
            codec: qgs_protocol::VideoCodec::H264,
            profile: VideoProfile::H264(H264Profile::High),
            bit_depth: BitDepth::new(8).expect("bit depth"),
            chroma: ChromaSubsampling::Cs420,
            max_width: 1920,
            max_height: 1080,
            progressive_supported: true,
            interlaced_supported: false,
            output_surface_formats: vec![VideoSurfaceFormat::Nv12],
        };
        let mut capabilities = Vec::new();

        push_unique_capability(&mut capabilities, capability.clone());
        push_unique_capability(&mut capabilities, capability);

        assert_eq!(capabilities.len(), 1);
    }

    #[test]
    fn cpu_nv12_pool_is_bounded_and_reuses_owned_frames() {
        let visible = VisibleRegion {
            x: 0,
            y: 0,
            width: 4,
            height: 2,
        };
        let mut pool = CpuNv12FramePool::new(4, 4, visible, 2).expect("pool");

        let first = pool.acquire().expect("first");
        let second = pool.acquire().expect("second");
        assert!(matches!(pool.acquire(), Err(DecoderError::DecodeFailed)));
        assert_eq!(pool.stats().peak_checked_out, 2);

        pool.release(first).expect("release");
        let reused = pool.acquire().expect("reused");
        assert_eq!(reused.desc.visible_region, visible);
        assert_eq!(reused.y.width_bytes, 4);
        assert_eq!(reused.y.height, 4);
        assert_eq!(reused.uv.width_bytes, 4);
        assert_eq!(reused.uv.height, 2);
        assert_eq!(reused.owned_bytes(), 24);
        assert_eq!(pool.stats().bytes_per_frame, 24);
        assert_eq!(pool.stats().frames_reused, 1);
        pool.release(second).expect("release second");
        pool.release(reused).expect("release reused");
    }

    #[test]
    fn copy_nv12_plane_skips_source_padding() {
        let source = [1_u8, 2, 3, 4, 0xaa, 0xbb, 5, 6, 7, 8, 0xcc, 0xdd];
        let mut plane = CpuNv12Plane {
            width_bytes: 4,
            height: 2,
            stride_bytes: 4,
            data: vec![0; 8],
        };

        let copied = copy_nv12_plane(&source, 0, 6, &mut plane).expect("copy");

        assert_eq!(copied, 8);
        assert_eq!(plane.data, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn professional_h264_stream_does_not_match_8bit_420_decoder_config() {
        let parsed = qgs_codec_h264::parse_annex_b_access_unit(PROFESSIONAL_INTRA_FIXTURE)
            .expect("professional H.264 stream parses");
        let config = DecoderConfig {
            device_id: DeviceId::new(1).expect("device id"),
            codec: VideoCodec::H264,
            profile: VideoProfile::H264(H264Profile::High),
            bit_depth: BitDepth::new(8).expect("8-bit"),
            chroma: ChromaSubsampling::Cs420,
            coded_width: 128,
            coded_height: 72,
            scan_mode: qgs_protocol::ScanMode::Progressive,
        };

        assert!(!parsed_access_unit_matches_config(&parsed, &config));
    }

    #[test]
    fn h264_slice_parameters_are_one_buffer_per_slice() {
        let mut parsed = qgs_codec_h264::parse_annex_b_access_unit(PROFESSIONAL_INTRA_FIXTURE)
            .expect("professional H.264 stream parses");
        let mut second_slice = parsed.slices[0].clone();
        parsed.slices[0].nal_bytes = vec![0x65, 0x88, 0x84];
        parsed.slices[0].slice_data_bit_offset = 9;
        second_slice.nal_bytes = vec![0x65, 0x99, 0xaa, 0xbb, 0xcc];
        second_slice.slice_data_bit_offset = 11;
        parsed.slices.push(second_slice);

        let params = slice_parameters(&parsed, &BTreeMap::new()).expect("slice params");

        assert_eq!(params.len(), 2);
        assert_eq!(params[0].inner().len(), 1);
        assert_eq!(params[0].inner()[0].slice_data_size, 3);
        assert_eq!(params[0].inner()[0].slice_data_offset, 0);
        assert_eq!(params[0].inner()[0].slice_data_bit_offset, 9);
        assert_eq!(params[1].inner().len(), 1);
        assert_eq!(params[1].inner()[0].slice_data_size, 5);
        assert_eq!(params[1].inner()[0].slice_data_offset, 0);
        assert_eq!(params[1].inner()[0].slice_data_bit_offset, 11);
    }

    #[test]
    fn render_node_matching_reads_known_sysfs_values_when_available() {
        let intel = render_node_from_path(PathBuf::from("/dev/dri/renderD128"));
        if let Some(node) = intel {
            assert_eq!(node.vendor_id, 0x8086);
        }
    }
}
