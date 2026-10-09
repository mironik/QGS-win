use std::path::PathBuf;

use qgs_core::{
    DecoderBackend, DecoderError, DeviceDiscovery, ResourceBackend, ResourceError, Session,
    SessionManager, SyncBackend, SyncError, VideoCapabilityDiscovery,
};
use qgs_linux::{
    bind_socket, default_socket_path, receive_message, remove_socket_file, send_message,
    send_message_with_attachments, TransportError,
};
use qgs_protocol::{
    DeviceCapabilities, DeviceCapabilitiesResponse, DeviceDesc, DeviceListResponse, ErrorResponse,
    ProtocolError, ProtocolErrorCode, ResourceExportedResponse, SyncExportedResponse,
    VideoCapabilities, VideoCapabilitiesResponse, WireMessage,
};
use qgs_software_video::SoftwareVideoBackend;
use qgs_vaapi::VaapiVideoDiscovery;
use qgs_vulkan::VulkanDeviceDiscovery;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let socket_path = socket_path_from_args();
    let listener = bind_socket(&socket_path)?;
    let sessions = SessionManager::new();
    let vulkan = VulkanDeviceDiscovery::new()?;
    let devices = vulkan.enumerate_devices()?;
    let vaapi = VaapiVideoDiscovery::new(&devices);
    let discovery = QgsBackends {
        vulkan,
        devices,
        vaapi,
        software_video: SoftwareVideoBackend::new(),
    };

    println!("qgsd listening on {}", socket_path.display());

    for stream in listener.incoming() {
        let mut stream = stream?;
        handle_client(&mut stream, &sessions, &discovery)?;
    }

    remove_socket_file(&socket_path)?;

    Ok(())
}

#[derive(Debug)]
struct QgsBackends {
    vulkan: VulkanDeviceDiscovery,
    devices: Vec<DeviceDesc>,
    vaapi: VaapiVideoDiscovery,
    software_video: SoftwareVideoBackend,
}

impl DeviceDiscovery for QgsBackends {
    fn enumerate_devices(&self) -> Result<Vec<DeviceDesc>, qgs_core::DeviceDiscoveryError> {
        Ok(self.devices.clone())
    }

    fn query_device_capabilities(
        &self,
        device_id: qgs_protocol::DeviceId,
    ) -> Result<DeviceCapabilities, qgs_core::DeviceDiscoveryError> {
        self.vulkan.query_device_capabilities(device_id)
    }
}

impl VideoCapabilityDiscovery for QgsBackends {
    fn query_video_capabilities(
        &self,
        device_id: qgs_protocol::DeviceId,
    ) -> Result<VideoCapabilities, qgs_core::VideoCapabilityDiscoveryError> {
        match self.vaapi.query_video_capabilities(device_id) {
            Ok(capabilities) => Ok(capabilities),
            Err(qgs_core::VideoCapabilityDiscoveryError::UnknownDeviceId)
                if self.devices.iter().any(|device| device.id == device_id) =>
            {
                Ok(VideoCapabilities {
                    device_id,
                    decode: Vec::new(),
                })
            }
            Err(err) => Err(err),
        }
    }
}

impl ResourceBackend for QgsBackends {
    fn create_buffer(
        &self,
        desc: &qgs_protocol::BufferDesc,
    ) -> Result<qgs_core::BackendBufferAllocation, ResourceError> {
        self.vulkan.create_buffer(desc)
    }

    fn create_image(
        &self,
        desc: &qgs_protocol::ImageDesc,
    ) -> Result<qgs_core::BackendImageAllocation, ResourceError> {
        self.vulkan.create_image(desc)
    }
}

impl SyncBackend for QgsBackends {
    fn create_sync(
        &self,
        request: &qgs_protocol::CreateSyncRequest,
    ) -> Result<Box<dyn qgs_core::BackendSync>, SyncError> {
        self.vulkan.create_sync(request)
    }
}

impl DecoderBackend for QgsBackends {
    fn create_decoder(
        &self,
        request: &qgs_protocol::CreateDecoderRequest,
    ) -> Result<Box<dyn qgs_core::BackendDecoder>, DecoderError> {
        match self.vaapi.create_decoder(request) {
            Ok(decoder) => {
                eprintln!("decoder backend selected: vaapi");
                Ok(decoder)
            }
            Err(DecoderError::UnsupportedDecodeConfiguration)
                if SoftwareVideoBackend::supports_config(&request.config) =>
            {
                eprintln!("decoder backend selected: software-video");
                self.software_video.create_decoder(request)
            }
            Err(err) => Err(err),
        }
    }
}

fn handle_client(
    stream: &mut std::os::unix::net::UnixStream,
    sessions: &SessionManager,
    discovery: &(impl DeviceDiscovery
          + ResourceBackend
          + SyncBackend
          + VideoCapabilityDiscovery
          + DecoderBackend),
) -> Result<(), Box<dyn std::error::Error>> {
    let mut session: Option<Session> = None;

    loop {
        let message = match receive_message(stream) {
            Ok(message) => message,
            Err(TransportError::Io(err)) if err.kind() == std::io::ErrorKind::UnexpectedEof => {
                if let Some(session) = &session {
                    if session.resource_count() > 0
                        || session.sync_count() > 0
                        || session.decoder_count() > 0
                    {
                        eprintln!(
                            "client disconnected; releasing {} resource(s), {} sync object(s), and {} decoder(s) for session {}",
                            session.resource_count(),
                            session.sync_count(),
                            session.decoder_count(),
                            session.id().get()
                        );
                    }
                }
                return Ok(());
            }
            Err(TransportError::Protocol(err)) => {
                send_message(
                    stream,
                    &WireMessage::Error {
                        request_id: 0,
                        response: ErrorResponse {
                            code: ProtocolErrorCode::from(&err),
                        },
                    },
                )?;
                continue;
            }
            Err(TransportError::Io(err)) => return Err(err.into()),
        };
        let request_id = message.request_id();

        let response = match message {
            WireMessage::Hello { request, .. } => match sessions.create_session() {
                Ok(created_session) => {
                    match qgs_protocol::handle_hello(&request, created_session.id()) {
                        Ok(response) => {
                            session = Some(created_session);
                            WireMessage::Welcome {
                                request_id,
                                response,
                            }
                        }
                        Err(err) => WireMessage::Error {
                            request_id,
                            response: ErrorResponse {
                                code: ProtocolErrorCode::from(&err),
                            },
                        },
                    }
                }
                Err(err) => WireMessage::Error {
                    request_id,
                    response: ErrorResponse {
                        code: ProtocolErrorCode::from(&err),
                    },
                },
            },
            WireMessage::EnumerateDevices { .. } => {
                if session.is_none() {
                    WireMessage::Error {
                        request_id,
                        response: ErrorResponse {
                            code: ProtocolErrorCode::SessionRequired,
                        },
                    }
                } else {
                    match discovery.enumerate_devices() {
                        Ok(devices) => WireMessage::DeviceList {
                            request_id,
                            response: DeviceListResponse { devices },
                        },
                        Err(err) => {
                            eprintln!("device discovery failed: {err}");
                            WireMessage::Error {
                                request_id,
                                response: ErrorResponse {
                                    code: ProtocolErrorCode::DiscoveryFailed,
                                },
                            }
                        }
                    }
                }
            }
            WireMessage::QueryDeviceCapabilities { request, .. } => {
                if session.is_none() {
                    WireMessage::Error {
                        request_id,
                        response: ErrorResponse {
                            code: ProtocolErrorCode::SessionRequired,
                        },
                    }
                } else {
                    match discovery.query_device_capabilities(request.device_id) {
                        Ok(capabilities) => WireMessage::DeviceCapabilities {
                            request_id,
                            response: DeviceCapabilitiesResponse { capabilities },
                        },
                        Err(qgs_core::DeviceDiscoveryError::UnknownDeviceId) => {
                            WireMessage::Error {
                                request_id,
                                response: ErrorResponse {
                                    code: ProtocolErrorCode::UnknownDeviceId,
                                },
                            }
                        }
                        Err(err) => {
                            eprintln!("device capability discovery failed: {err}");
                            WireMessage::Error {
                                request_id,
                                response: ErrorResponse {
                                    code: ProtocolErrorCode::DiscoveryFailed,
                                },
                            }
                        }
                    }
                }
            }
            WireMessage::QueryVideoCapabilities { request, .. } => {
                if session.is_none() {
                    WireMessage::Error {
                        request_id,
                        response: ErrorResponse {
                            code: ProtocolErrorCode::SessionRequired,
                        },
                    }
                } else {
                    match discovery.query_video_capabilities(request.device_id) {
                        Ok(capabilities) => WireMessage::VideoCapabilities {
                            request_id,
                            response: VideoCapabilitiesResponse { capabilities },
                        },
                        Err(qgs_core::VideoCapabilityDiscoveryError::UnknownDeviceId) => {
                            WireMessage::Error {
                                request_id,
                                response: ErrorResponse {
                                    code: ProtocolErrorCode::UnknownDeviceId,
                                },
                            }
                        }
                        Err(err) => {
                            eprintln!("video capability discovery failed: {err}");
                            WireMessage::Error {
                                request_id,
                                response: ErrorResponse {
                                    code: ProtocolErrorCode::DiscoveryFailed,
                                },
                            }
                        }
                    }
                }
            }
            WireMessage::CreateDecoder { request, .. } => {
                if let Some(session) = &mut session {
                    match session.create_decoder(discovery, &request) {
                        Ok(response) => WireMessage::DecoderCreated {
                            request_id,
                            response,
                        },
                        Err(err) => {
                            eprintln!("decoder creation failed: {err}");
                            WireMessage::Error {
                                request_id,
                                response: ErrorResponse {
                                    code: protocol_code_from_decoder_error(&err),
                                },
                            }
                        }
                    }
                } else {
                    WireMessage::Error {
                        request_id,
                        response: ErrorResponse {
                            code: ProtocolErrorCode::SessionRequired,
                        },
                    }
                }
            }
            WireMessage::SubmitAccessUnit { request, .. } => {
                if let Some(session) = &mut session {
                    match session.submit_access_unit(&request) {
                        Ok(response) => WireMessage::DecodeOutput {
                            request_id,
                            response,
                        },
                        Err(err) => {
                            eprintln!("decode submission failed: {err}");
                            WireMessage::Error {
                                request_id,
                                response: ErrorResponse {
                                    code: protocol_code_from_decoder_error(&err),
                                },
                            }
                        }
                    }
                } else {
                    WireMessage::Error {
                        request_id,
                        response: ErrorResponse {
                            code: ProtocolErrorCode::SessionRequired,
                        },
                    }
                }
            }
            WireMessage::FlushDecoder { request, .. } => {
                if let Some(session) = &mut session {
                    match session.flush_decoder(&request) {
                        Ok(response) => WireMessage::DecodeOutput {
                            request_id,
                            response,
                        },
                        Err(err) => {
                            eprintln!("decoder flush failed: {err}");
                            WireMessage::Error {
                                request_id,
                                response: ErrorResponse {
                                    code: protocol_code_from_decoder_error(&err),
                                },
                            }
                        }
                    }
                } else {
                    WireMessage::Error {
                        request_id,
                        response: ErrorResponse {
                            code: ProtocolErrorCode::SessionRequired,
                        },
                    }
                }
            }
            WireMessage::DestroyDecoder { request, .. } => {
                if let Some(session) = &mut session {
                    match session.destroy_decoder(&request) {
                        Ok(response) => WireMessage::DecoderDestroyed {
                            request_id,
                            response,
                        },
                        Err(err) => WireMessage::Error {
                            request_id,
                            response: ErrorResponse {
                                code: protocol_code_from_decoder_error(&err),
                            },
                        },
                    }
                } else {
                    WireMessage::Error {
                        request_id,
                        response: ErrorResponse {
                            code: ProtocolErrorCode::SessionRequired,
                        },
                    }
                }
            }
            WireMessage::CreateBuffer { request, .. } => {
                if let Some(session) = &mut session {
                    match session.create_buffer(discovery, &request.desc) {
                        Ok(response) => WireMessage::BufferCreated {
                            request_id,
                            response,
                        },
                        Err(err) => {
                            eprintln!("buffer creation failed: {err}");
                            WireMessage::Error {
                                request_id,
                                response: ErrorResponse {
                                    code: protocol_code_from_resource_error(&err),
                                },
                            }
                        }
                    }
                } else {
                    WireMessage::Error {
                        request_id,
                        response: ErrorResponse {
                            code: ProtocolErrorCode::SessionRequired,
                        },
                    }
                }
            }
            WireMessage::CreateImage { request, .. } => {
                if let Some(session) = &mut session {
                    match session.create_image(discovery, &request.desc) {
                        Ok(response) => WireMessage::ImageCreated {
                            request_id,
                            response,
                        },
                        Err(err) => {
                            eprintln!("image creation failed: {err}");
                            WireMessage::Error {
                                request_id,
                                response: ErrorResponse {
                                    code: protocol_code_from_resource_error(&err),
                                },
                            }
                        }
                    }
                } else {
                    WireMessage::Error {
                        request_id,
                        response: ErrorResponse {
                            code: ProtocolErrorCode::SessionRequired,
                        },
                    }
                }
            }
            WireMessage::DestroyResource { request, .. } => {
                if let Some(session) = &mut session {
                    match session.destroy_resource(request.resource_id) {
                        Ok(response) => WireMessage::ResourceDestroyed {
                            request_id,
                            response,
                        },
                        Err(err) => WireMessage::Error {
                            request_id,
                            response: ErrorResponse {
                                code: protocol_code_from_resource_error(&err),
                            },
                        },
                    }
                } else {
                    WireMessage::Error {
                        request_id,
                        response: ErrorResponse {
                            code: ProtocolErrorCode::SessionRequired,
                        },
                    }
                }
            }
            WireMessage::ExportResource { request, .. } => {
                if let Some(session) = &session {
                    match session.export_resource(&request) {
                        Ok(export) => {
                            let message = WireMessage::ResourceExported {
                                request_id,
                                response: ResourceExportedResponse {
                                    metadata: export.metadata,
                                },
                            };
                            send_message_with_attachments(stream, &message, &[&export.handle])?;
                            continue;
                        }
                        Err(err) => {
                            eprintln!("resource export failed: {err}");
                            WireMessage::Error {
                                request_id,
                                response: ErrorResponse {
                                    code: protocol_code_from_resource_error(&err),
                                },
                            }
                        }
                    }
                } else {
                    WireMessage::Error {
                        request_id,
                        response: ErrorResponse {
                            code: ProtocolErrorCode::SessionRequired,
                        },
                    }
                }
            }
            WireMessage::CreateSync { request, .. } => {
                if let Some(session) = &mut session {
                    match session.create_sync(discovery, &request) {
                        Ok(response) => WireMessage::SyncCreated {
                            request_id,
                            response,
                        },
                        Err(err) => {
                            eprintln!("sync creation failed: {err}");
                            WireMessage::Error {
                                request_id,
                                response: ErrorResponse {
                                    code: protocol_code_from_sync_error(&err),
                                },
                            }
                        }
                    }
                } else {
                    WireMessage::Error {
                        request_id,
                        response: ErrorResponse {
                            code: ProtocolErrorCode::SessionRequired,
                        },
                    }
                }
            }
            WireMessage::ExportSync { request, .. } => {
                if let Some(session) = &session {
                    match session.export_sync(&request) {
                        Ok(export) => {
                            let message = WireMessage::SyncExported {
                                request_id,
                                response: SyncExportedResponse {
                                    metadata: export.metadata,
                                },
                            };
                            send_message_with_attachments(stream, &message, &[&export.handle])?;
                            continue;
                        }
                        Err(err) => {
                            eprintln!("sync export failed: {err}");
                            WireMessage::Error {
                                request_id,
                                response: ErrorResponse {
                                    code: protocol_code_from_sync_error(&err),
                                },
                            }
                        }
                    }
                } else {
                    WireMessage::Error {
                        request_id,
                        response: ErrorResponse {
                            code: ProtocolErrorCode::SessionRequired,
                        },
                    }
                }
            }
            WireMessage::Welcome { .. }
            | WireMessage::Error { .. }
            | WireMessage::DeviceList { .. }
            | WireMessage::DeviceCapabilities { .. }
            | WireMessage::VideoCapabilities { .. }
            | WireMessage::DecoderCreated { .. }
            | WireMessage::DecodeOutput { .. }
            | WireMessage::DecoderDestroyed { .. }
            | WireMessage::BufferCreated { .. }
            | WireMessage::ImageCreated { .. }
            | WireMessage::ResourceDestroyed { .. }
            | WireMessage::ResourceExported { .. }
            | WireMessage::SyncCreated { .. }
            | WireMessage::SyncExported { .. } => WireMessage::Error {
                request_id,
                response: ErrorResponse {
                    code: ProtocolErrorCode::from(&ProtocolError::MalformedPayload),
                },
            },
        };

        send_message(stream, &response)?;
    }
}

fn protocol_code_from_sync_error(err: &SyncError) -> ProtocolErrorCode {
    match err {
        SyncError::UnknownSync => ProtocolErrorCode::UnknownSync,
        SyncError::UnknownResource => ProtocolErrorCode::UnknownResource,
        SyncError::UnknownDeviceId => ProtocolErrorCode::UnknownDeviceId,
        SyncError::UnsupportedSyncHandleType => ProtocolErrorCode::UnsupportedSyncHandleType,
        SyncError::SyncExportFailed => ProtocolErrorCode::SyncExportFailed,
        SyncError::Protocol(err) => ProtocolErrorCode::from(err),
    }
}

fn protocol_code_from_decoder_error(err: &DecoderError) -> ProtocolErrorCode {
    match err {
        DecoderError::UnknownDecoder => ProtocolErrorCode::UnknownDecoder,
        DecoderError::UnknownDeviceId => ProtocolErrorCode::UnknownDeviceId,
        DecoderError::UnsupportedDecodeConfiguration => {
            ProtocolErrorCode::UnsupportedDecodeConfiguration
        }
        DecoderError::MalformedCompressedData => ProtocolErrorCode::MalformedCompressedData,
        DecoderError::CompressedPacketTooLarge => ProtocolErrorCode::CompressedPacketTooLarge,
        DecoderError::UnsupportedH264StreamFeature => {
            ProtocolErrorCode::UnsupportedH264StreamFeature
        }
        DecoderError::DecodeFailed => ProtocolErrorCode::DecodeFailed,
        DecoderError::Protocol(err) => ProtocolErrorCode::from(err),
    }
}

fn protocol_code_from_resource_error(err: &ResourceError) -> ProtocolErrorCode {
    match err {
        ResourceError::UnknownResource => ProtocolErrorCode::UnknownResource,
        ResourceError::InvalidBufferSize => ProtocolErrorCode::InvalidBufferSize,
        ResourceError::InvalidImageDimensions => ProtocolErrorCode::InvalidImageDimensions,
        ResourceError::AllocationFailed => ProtocolErrorCode::AllocationFailed,
        ResourceError::UnsupportedMemoryRequirements => {
            ProtocolErrorCode::UnsupportedMemoryRequirements
        }
        ResourceError::UnsupportedPixelFormat => ProtocolErrorCode::UnsupportedPixelFormat,
        ResourceError::UnsupportedImageUsage => ProtocolErrorCode::UnsupportedImageUsage,
        ResourceError::UnsupportedImageExternalSharing => {
            ProtocolErrorCode::UnsupportedImageExternalSharing
        }
        ResourceError::ResourceNotExportable => ProtocolErrorCode::ResourceNotExportable,
        ResourceError::UnsupportedExternalHandleType => {
            ProtocolErrorCode::UnsupportedExternalHandleType
        }
        ResourceError::ExportFailed => ProtocolErrorCode::ExportFailed,
        ResourceError::UnknownDeviceId => ProtocolErrorCode::UnknownDeviceId,
        ResourceError::Protocol(err) => ProtocolErrorCode::from(err),
    }
}

fn socket_path_from_args() -> PathBuf {
    std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(default_socket_path)
}
