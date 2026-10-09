use std::collections::{BTreeMap, VecDeque};
use std::fs::File;
use std::io::{self, BufRead, IsTerminal, Write};
use std::os::fd::OwnedFd;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use qgs_audio_pipewire::{
    f32_interleaved_from_s24le_mono_tracks, submit_native_pipewire_buffer,
    submit_native_pipewire_buffers, MonoS24LeTrack, PipeWireAudioSampleFormat,
    PipeWireStreamFormat,
};
use qgs_broadcast_player::{
    AudioOutputMode, LiveOutputCoordinator, LiveOutputTickResult, LiveVideoOutputMode,
};
use qgs_core::{BackendDecodedSurface, DecoderBackend, DeviceDiscovery, VideoCapabilityDiscovery};
use qgs_h264_422p10::{
    H264422P10Decoder, H264422P10Frame, H264422P10PictureDecodeError, H264422P10Profile,
};
use qgs_linux::{
    connect_socket, default_socket_path, inspect_native_pipewire_stream_boundary,
    probe_linux_audio_device_boundary, receive_message, receive_message_with_attachments,
    send_message, submit_pipewire_audio_prototype, LinuxAudioConversionNeed,
    LinuxOriginalPcmAudioFormat, LinuxPipewirePrototypeBuffer, LinuxPipewirePrototypeSampleFormat,
};
use qgs_media_runtime::{
    audio_samples_for_duration, av_frame_audio_range, bind_broadcast_audio_payload,
    bind_broadcast_presentation_payload, bind_broadcast_video_payload_accounting,
    bind_broadcast_video_payload_ready, broadcast_player_events_from_audio_sink_evidence,
    broadcast_player_events_from_presentation_evidence, build_broadcast_audio_device_submission,
    build_broadcast_player_event_surface, build_broadcast_video_presenter_submission,
    classify_presentation, duration_abs_delta, duration_from_audio_samples,
    evaluate_broadcast_preroll, max_video_timestamp_outside_audio_range,
    project_qgs_monitor_update_from_file_presenter_result,
    project_qgs_monitor_update_from_presenter_result, project_qnc_events, qgs_frames_for_duration,
    qgs_qnc_projected_sequence_is_monotonic, simulate_broadcast_player_runtime_loop,
    submit_qgs_file_presenter_image, submit_qgs_file_presenter_manifest,
    submit_qgs_presenter_payload_to_test_boundary, summarize_broadcast_device_boundary,
    summarize_broadcast_payload_bindings, summarize_broadcast_player_runtime_events,
    summarize_broadcast_prepared_slots, summarize_broadcast_runtime_contract, AudioFormat,
    AudioSampleFormat, AudioTimeline, AudioTimingPacket, AvFrameAudioRange, BoundedQueue,
    BroadcastDevicePayloadStatus, BroadcastDeviceStatus, BroadcastMediaSourceRole,
    BroadcastPreparedAudioSlot, BroadcastPreparedPresentationSlot, BroadcastPreparedVideoSlot,
    BroadcastPreparedVideoSlotStatus, BroadcastPrerollConfig, BroadcastPrerollPlan,
    BroadcastPresentationPayloadBinding, BroadcastPresentationPayloadReadiness,
    BroadcastPreviewProfile, BroadcastRuntimeCapabilities, BroadcastRuntimePrepareFacts,
    BroadcastRuntimeQueueLimits, BroadcastRuntimeSessionDescription, BroadcastRuntimeStateMachine,
    BroadcastRuntimeVerificationMatrix, BroadcastTestAudioSink, BroadcastTestAudioSinkConfig,
    BroadcastTestVideoPresenter, BroadcastTestVideoPresenterConfig,
    BroadcastVideoPayloadBackendPath, BroadcastVideoPayloadBindingStatus,
    BroadcastVideoPayloadFormat, BroadcastVideoPayloadKind, BroadcastVideoPayloadReference,
    BroadcastVideoSourceMode, FrameIdentity as PlaybackFrameIdentity, OriginalAudioTrack,
    PcmAudioBlock, PcmAudioBlockLayout, PcmAudioPacket, PcmEndian, PcmSampleFormat, PlaybackClock,
    PlaybackConfig, PlaybackState, PresentationDecision, QgsActiveRangeTiming,
    QgsAudioOutputBackendKind, QgsAudioRepresentation, QgsBroadcastPlayerAssembly,
    QgsBroadcastPlayerCommand, QgsBroadcastPlayerCore, QgsBroadcastPlayerOperationalRuntime,
    QgsBroadcastPlayerSnapshot, QgsBroadcastPlayerStatus, QgsBufferDiscardReason,
    QgsDeviceBackendSelection, QgsDeviceBackendSelector, QgsDeviceSelectionPolicy,
    QgsFilePresenterSubmissionResult, QgsFrameClock, QgsFrameClockMode, QgsFrameClockRate,
    QgsInputPlan, QgsInputPlanQueueRequirements, QgsInputPlanSourceMode,
    QgsOperationalRuntimeConfig, QgsOperationalRuntimeSnapshot, QgsOriginalProxyAssociationStatus,
    QgsPlaybackRepresentation, QgsPlayoutBufferLimits, QgsPlayoutBufferState,
    QgsPreparedAudioChannel, QgsPreparedAudioLayout, QgsPreparedInputDescriptor,
    QgsPreparedMediaBinding, QgsPreparedSourceIdentity, QgsPreparedStreamLayout,
    QgsPreparedVideoTiming, QgsQncAudioChannelKindLike, QgsQncAudioChannelLike,
    QgsQncAudioRepresentationLike, QgsQncCommandEnvelope, QgsQncCommandKind, QgsQncCommandPayload,
    QgsQncCommandRequestEnvelope, QgsQncControlEventEnvelope, QgsQncControlEventKind,
    QgsQncControlSurface, QgsQncEventEnvelope, QgsQncFrameRangeLike, QgsQncMonitorProjection,
    QgsQncPassiveView, QgsQncPlaybackInputLike, QgsQncPlayerCommand, QgsQncPreparedInputLike,
    QgsQncPrivateBindingLike, QgsQncProjectAudioLike, QgsQncRepresentationLike,
    QgsQncRuntimeSnapshot, QgsQncSessionCommandExecutor, QgsQncSourceIdentityLike,
    QgsQncStreamLayoutLike, QgsQncTimelineProjection, QgsQncVideoInputLike,
    QgsRuntimeEventEnvelope, QgsRuntimeEventLog, QgsSessionCommandQueueLimits,
    QgsSessionDuplicateCommandPolicy, QgsSessionQueueRejection, QgsSessionRuntime,
    QgsSessionRuntimeConfig, QgsTickPreparationEvent, QgsTickPreparationInput, QgsTransportEngine,
    QgsTransportEvent, QgsVideoPresenterBackendKind, RationalRate, RealTimeClock, TestAudioSink,
    TestPresentationSink,
};
use qgs_mp4::{
    classify_video_track, nearest_random_access_before, MediaHealth, Mp4Source, Mp4TrackKind,
};
use qgs_mxf::{
    open_pcm_audio_index, read_pcm_audio_packet_at, IndexSource, MediaSource, PcmAudioIndex,
    RandomAccess, TrackKind, MAX_MXF_FILE_BYTES,
};
use qgs_protocol::{
    BitDepth, BufferDesc, BufferUsageFlags, ChromaSubsampling, CreateBufferRequest,
    CreateDecoderRequest, CreateImageRequest, CreateSyncRequest, DecoderConfig, DecoderId,
    DestroyDecoderRequest, DestroyResourceRequest, DeviceCapabilities, DeviceClass, DeviceDesc,
    ErrorResponse, ExportResourceRequest, ExportSyncRequest, ExternalHandleType, ExternalSharing,
    FlushDecoderRequest, H264Profile, HelloRequest, ImageDesc, ImageUsageFlags, MemoryPreference,
    PixelFormat, ProtocolErrorCode, QueryDeviceCapabilitiesRequest, QueryVideoCapabilitiesRequest,
    ResourceId, ScanMode, SelectedMemoryProperties, SubmitAccessUnitRequest, SyncExportHandleType,
    SyncId, SyncKind, VideoCapabilities, VideoCodec, VideoProfile, VisibleRegion, WireMessage,
    CURRENT_PROTOCOL_VERSION,
};
use qgs_software_video::{
    decoder_config_for_surface, DecodeRunStats, SoftwareH264Decoder, SoftwarePixelFormat,
    SoftwareVideoBackend,
};
use qgs_vulkan::{
    diagnose_haswell_video_import, yuv422p10_reference_rgba_u16, yuv422p10_rgba_u16_checksum,
    DiagnosticDrmLayer, DiagnosticDrmObject, DiagnosticDrmPlane, FrameIdentity,
    FrameProcessorError, GpuFrameProcessor, GpuFrameProcessorConfig, HaswellVideoDiagnosticInput,
    Nv12FrameProcessor, Nv12FrameProcessorConfig, Nv12FrameProcessorDiagnostics, Nv12Plane,
    Nv12Upload, VulkanDeviceDiscovery, YcbcrConversion, Yuv422P10Plane, Yuv422P10Upload,
};
use sha2::{Digest, Sha256};

const DEMO_BUFFER_SIZE: u64 = 1024 * 1024;
const IMAGE_PROOF_WIDTH: u32 = 64;
const IMAGE_PROOF_HEIGHT: u32 = 64;
const VIDEO_CAPABILITIES_ONLY_ARG: &str = "--video-capabilities-only";
const H264_DECODE_ONLY_ARG: &str = "--h264-decode-only";
const HASWELL_VIDEO_DIAGNOSTIC_ARG: &str = "--haswell-video-diagnostic";
const MXF_INSPECT_ARG: &str = "--mxf-inspect";
const SOFTWARE_DECODE_MXF_ARG: &str = "--software-decode-mxf";
const SOFTWARE_GPU_MXF_ARG: &str = "--software-gpu-mxf";
const NATIVE_ORIGINAL_VIDEO_DECODE_ARG: &str = "--native-original-video-decode";
const PROXY_PROOF_ARG: &str = "--proxy-proof";
const PROXY_THROUGHPUT_ARG: &str = "--proxy-throughput";
const PROXY_PLAYBACK_ARG: &str = "--proxy-playback";
const PROXY_PLAYBACK_PROFILE_ARG: &str = "--proxy-playback-profile";
const QNC_JOURNALIST_DEMO_ARG: &str = "--qnc-journalist-demo";
const ORIGINAL_AUDIO_EXTRACT_ARG: &str = "--original-audio-extract";
const BROADCAST_PLAYER_RUNTIME_CONTRACT_ARG: &str = "--broadcast-player-runtime-contract";
const BROADCAST_PLAYER_RUNTIME_STATE_MACHINE_ARG: &str = "--broadcast-player-runtime-state-machine";
const BROADCAST_PLAYER_RUNTIME_PREROLL_ARG: &str = "--broadcast-player-runtime-preroll";
const BROADCAST_PLAYER_RUNTIME_PREPARED_SLOTS_ARG: &str =
    "--broadcast-player-runtime-prepared-slots";
const LEGACY_BROADCAST_RUNTIME_CONTRACT_ARG: &str = "--broadcast-runtime-contract";
const LEGACY_BROADCAST_RUNTIME_STATE_MACHINE_ARG: &str = "--broadcast-runtime-state-machine";
const LEGACY_BROADCAST_RUNTIME_PREROLL_ARG: &str = "--broadcast-runtime-preroll";
const LEGACY_BROADCAST_RUNTIME_PREPARED_SLOTS_ARG: &str = "--broadcast-runtime-prepared-slots";
const BROADCAST_PLAYER_RUNTIME_EVENTS_ARG: &str = "--broadcast-player-runtime-events";
const BROADCAST_PLAYER_RUNTIME_PAYLOADS_ARG: &str = "--broadcast-player-runtime-payloads";
const BROADCAST_PLAYER_RUNTIME_VIDEO_PAYLOADS_ARG: &str =
    "--broadcast-player-runtime-video-payloads";
const BROADCAST_PLAYER_RUNTIME_DEVICE_BOUNDARY_ARG: &str =
    "--broadcast-player-runtime-device-boundary";
const BROADCAST_PLAYER_RUNTIME_TEST_PRESENTER_ARG: &str =
    "--broadcast-player-runtime-test-presenter";
const BROADCAST_PLAYER_RUNTIME_TEST_AUDIO_SINK_ARG: &str =
    "--broadcast-player-runtime-test-audio-sink";
const BROADCAST_PLAYER_RUNTIME_SIMULATE_ARG: &str = "--broadcast-player-runtime-simulate";
const BROADCAST_PLAYER_RUNTIME_ORIGINAL_VIDEO_PAYLOADS_ARG: &str =
    "--broadcast-player-runtime-original-video-payloads";
const BROADCAST_PLAYER_RUNTIME_VERIFICATION_ARG: &str = "--broadcast-player-runtime-verification";
const LINUX_AUDIO_DEVICE_PROBE_ARG: &str = "--linux-audio-device-probe";
const PIPEWIRE_AUDIO_PROTOTYPE_ARG: &str = "--pipewire-audio-prototype";
const PIPEWIRE_AUDIO_NATIVE_PROTOTYPE_ARG: &str = "--pipewire-audio-native-prototype";
const PIPEWIRE_AUDIO_AUDIBLE_SMOKE_TEST_ARG: &str = "--pipewire-audio-audible-smoke-test";
const PIPEWIRE_AUDIO_SEGMENT_PLAYBACK_ARG: &str = "--pipewire-audio-segment-playback";
const BROADCAST_RUNTIME_AUDIO_PIPEWIRE_ARG: &str = "--broadcast-runtime-audio-pipewire";
const BROADCAST_RUNTIME_AUDIO_PIPEWIRE_AUDIBLE_ARG: &str =
    "--broadcast-runtime-audio-pipewire-audible";
const PIPEWIRE_AUDIO_CONTENT_AUDIT_ARG: &str = "--pipewire-audio-content-audit";
const PIPEWIRE_AUDIO_MONO_MONITOR_ARG: &str = "--pipewire-audio-mono-monitor";
const PIPEWIRE_AUDIO_DISCRETE_4MONO_ARG: &str = "--pipewire-audio-discrete-4mono";
const QNC_PREPARED_INPUT_DESCRIPTOR_ARG: &str = "--qnc-prepared-input-descriptor";
const QGS_INPUT_PLAN_ARG: &str = "--qgs-input-plan";
const QGS_QNC_PREPARED_INPUT_MAPPING_ARG: &str = "--qgs-qnc-prepared-input-mapping";
const QGS_QNC_PREPARED_INPUT_FIXTURE_ARG: &str = "--qgs-qnc-prepared-input-fixture";
const QGS_TRANSPORT_ENGINE_PARITY_ARG: &str = "--qgs-transport-engine-parity";
const QGS_FRAME_CLOCK_PARITY_ARG: &str = "--qgs-frame-clock-parity";
const QGS_PLAYOUT_BUFFER_TICK_ARG: &str = "--qgs-playout-buffer-tick";
const QGS_RUNTIME_LIFECYCLE_EVENTS_ARG: &str = "--qgs-runtime-lifecycle-events";
const QGS_QNC_EVENT_PROJECTION_ARG: &str = "--qgs-qnc-event-projection";
const QGS_SESSION_COMMAND_BOUNDARY_ARG: &str = "--qgs-session-command-boundary";
const QGS_SESSION_RUNTIME_CONTROL_ARG: &str = "--qgs-session-runtime-control";
const QGS_PRESENTER_MONITOR_BOUNDARY_ARG: &str = "--qgs-presenter-monitor-boundary";
const QGS_FILE_PRESENTER_DIAGNOSTIC_ARG: &str = "--qgs-file-presenter-diagnostic";
const QGS_GPU_PAYLOAD_READBACK_DIAGNOSTIC_ARG: &str = "--qgs-gpu-payload-readback-diagnostic";
const QGS_RUNTIME_SURFACE_E2E_ARG: &str = "--qgs-runtime-surface-e2e";
const QGS_BROADCAST_PLAYER_CONTROL_CORE_ARG: &str = "--qgs-broadcast-player-control-core";
const QGS_DEVICE_BACKEND_SELECTION_ARG: &str = "--qgs-device-backend-selection";
const QGS_BROADCAST_PLAYER_OPERATIONAL_RUNTIME_ARG: &str =
    "--qgs-broadcast-player-operational-runtime";
const QGS_BROADCAST_PLAYER_FAULT_RECOVERY_ARG: &str = "--qgs-broadcast-player-fault-recovery";
const QGS_BROADCAST_PLAYER_RUN_ARG: &str = "--qgs-broadcast-player-run";
const QGS_BROADCAST_PLAYER_LIVE_ARG: &str = "--qgs-broadcast-player-live";
const QGS_BROADCAST_PLAYER_CONTROL_SESSION_ARG: &str = "--qgs-broadcast-player-control-session";
const QGS_BROADCAST_PLAYER_RUN_FRAMES_ARG: &str = "--frames";
const QGS_BROADCAST_PLAYER_LIVE_START_FRAME_ARG: &str = "--start-frame";
const QGS_BROADCAST_PLAYER_LIVE_STATUS_EVERY_ARG: &str = "--status-every";
const QGS_BROADCAST_PLAYER_LIVE_MAX_FRAMES_ARG: &str = "--max-frames";
const QGS_BROADCAST_PLAYER_LIVE_PACE_ARG: &str = "--pace";
const QGS_BROADCAST_PLAYER_LIVE_NO_INTERACTIVE_ARG: &str = "--no-interactive";
const QGS_BROADCAST_PLAYER_LIVE_VIDEO_OUTPUT_ARG: &str = "--video-output";
const QGS_BROADCAST_PLAYER_LIVE_AUDIO_OUTPUT_ARG: &str = "--audio-output";
const QGS_BROADCAST_PLAYER_LIVE_OUTPUT_DIR_ARG: &str = "--output-dir";
const QGS_BROADCAST_PLAYER_LIVE_PREVIEW_EVERY_ARG: &str = "--preview-every";
const QGS_BROADCAST_PLAYER_RUN_SEEK_FRAME_ARG: &str = "--seek-frame";
const QGS_BROADCAST_PLAYER_RUN_VIEW_ARG: &str = "--view";
const QGS_BROADCAST_PLAYER_CONTROL_SCRIPT_ARG: &str = "--script";
const WRITE_DIAGNOSTIC_WAV_ARG: &str = "--write-diagnostic-wav";
const AUDIO_AUDIT_START_MS_ARG: &str = "--start-ms";
const AUDIO_AUDIT_DURATION_MS_ARG: &str = "--duration-ms";
const AUDIO_AUDIT_OUTPUT_DIR_ARG: &str = "--output-dir";
const AUDIO_AUDIT_FIND_LOUDEST_RANGE_MS_ARG: &str = "--find-loudest-range-ms";
const AUDIO_MONITOR_TRACK_ARG: &str = "--track";
const EXPECTED_FX6_SAMPLE001_MXF_SHA256: &str =
    "6bb8d23f91be8812f0bf9c09b6ee680dce0560757b9d778333d3e996b5f69653";
const EXPECTED_FX6_SAMPLE001_PROXY_SHA256: &str =
    "9d0c64bed89303e6b8e1e32ef04984b62a174508528de42be87f82a985f039c8";
const EXPECTED_FX6_SAMPLE002_MXF_SHA256: &str =
    "52a82d3527717096892a78bfd62f62f864e891fc4321e09ad7ed0856a7b9f24e";
const EXPECTED_FX6_SAMPLE002_PROXY_SHA256: &str =
    "fa7b646f7dc84bee84744dbaee89924b7f94982405c2fc3ffea2936618f73007";
const H264_FIXTURE: &[u8] = include_bytes!("../../../tests/fixtures/h264/idr-64x64-baseline.h264");
const H264_LONG_GOP_FIXTURE: &[u8] =
    include_bytes!("../../../tests/fixtures/h264/long-gop-128x72-main.h264");
const DEFAULT_MXF_FIXTURE: &str = "tests/fixtures/mxf/h264-8bit-420-long-gop-128x72.mxf";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.haswell_video_diagnostic {
        return run_haswell_video_diagnostic();
    }
    if let Some(path) = args.mxf_inspect_path {
        return inspect_mxf(&path);
    }
    if let Some(path) = args.software_decode_mxf_path {
        return software_decode_mxf(&path);
    }
    if let Some(path) = args.software_gpu_mxf_path {
        return software_gpu_mxf(&path);
    }
    if let Some(path) = args.native_original_video_decode_path {
        return native_original_video_decode_mxf(&path);
    }
    if let Some((original, proxy)) = args.proxy_proof_paths {
        return proxy_proof(&args.socket_path, &original, &proxy);
    }
    if let Some((original, proxy)) = args.proxy_throughput_paths {
        return proxy_throughput(&original, &proxy);
    }
    if let Some((original, proxy)) = args.proxy_playback_paths {
        return proxy_playback(&original, &proxy, args.proxy_playback_profile);
    }
    if let Some((original, proxy)) = args.qnc_journalist_demo_paths {
        return qnc_journalist_demo(&original, &proxy);
    }
    if let Some(path) = args.original_audio_extract_path {
        return original_audio_extract(&path);
    }
    if let Some((original, proxy)) = args.broadcast_runtime_contract_paths {
        return broadcast_runtime_contract(&original, &proxy);
    }
    if let Some((original, proxy)) = args.broadcast_runtime_state_machine_paths {
        return broadcast_runtime_state_machine(&original, &proxy);
    }
    if let Some((original, proxy)) = args.broadcast_runtime_preroll_paths {
        return broadcast_runtime_preroll(&original, &proxy);
    }
    if let Some((original, proxy)) = args.broadcast_runtime_prepared_slots_paths {
        return broadcast_runtime_prepared_slots(
            &original,
            &proxy,
            BroadcastRuntimeReportFocus::PreparedSlots,
        );
    }
    if let Some((original, proxy)) = args.broadcast_player_runtime_events_paths {
        return broadcast_runtime_prepared_slots(
            &original,
            &proxy,
            BroadcastRuntimeReportFocus::Events,
        );
    }
    if let Some((original, proxy)) = args.broadcast_player_runtime_payloads_paths {
        return broadcast_runtime_prepared_slots(
            &original,
            &proxy,
            BroadcastRuntimeReportFocus::Payloads,
        );
    }
    if let Some((original, proxy)) = args.broadcast_player_runtime_video_payloads_paths {
        return broadcast_runtime_prepared_slots(
            &original,
            &proxy,
            BroadcastRuntimeReportFocus::VideoPayloads,
        );
    }
    if let Some((original, proxy)) = args.broadcast_player_runtime_device_boundary_paths {
        return broadcast_runtime_prepared_slots(
            &original,
            &proxy,
            BroadcastRuntimeReportFocus::DeviceBoundary,
        );
    }
    if let Some((original, proxy)) = args.broadcast_player_runtime_test_presenter_paths {
        return broadcast_runtime_prepared_slots(
            &original,
            &proxy,
            BroadcastRuntimeReportFocus::TestPresenter,
        );
    }
    if let Some((original, proxy)) = args.broadcast_player_runtime_test_audio_sink_paths {
        return broadcast_runtime_prepared_slots(
            &original,
            &proxy,
            BroadcastRuntimeReportFocus::TestAudioSink,
        );
    }
    if let Some((original, proxy)) = args.broadcast_player_runtime_simulate_paths {
        return broadcast_runtime_prepared_slots(
            &original,
            &proxy,
            BroadcastRuntimeReportFocus::SimulatedPlayback,
        );
    }
    if let Some((original, proxy)) = args.broadcast_player_runtime_original_video_payloads_paths {
        return broadcast_runtime_prepared_slots(
            &original,
            &proxy,
            BroadcastRuntimeReportFocus::OriginalVideoPayloads,
        );
    }
    if let Some((original, proxy)) = args.broadcast_player_runtime_verification_paths {
        return broadcast_player_runtime_verification(&original, &proxy);
    }
    if let Some((original, proxy)) = args.qnc_prepared_input_descriptor_paths {
        return qnc_prepared_input_descriptor_report(&original, &proxy);
    }
    if let Some((original, proxy)) = args.qgs_qnc_prepared_input_mapping_paths {
        return qgs_qnc_prepared_input_mapping_report(&original, &proxy);
    }
    if let Some(fixture) = args.qgs_qnc_prepared_input_fixture.as_deref() {
        return qgs_qnc_prepared_input_fixture_report(fixture);
    }
    if let Some((original, proxy)) = args.qgs_input_plan_paths {
        return qgs_input_plan_report(&original, &proxy);
    }
    if let Some((original, proxy)) = args.qgs_transport_engine_parity_paths {
        return qgs_transport_engine_parity_report(&original, &proxy);
    }
    if let Some((original, proxy)) = args.qgs_frame_clock_parity_paths {
        return qgs_frame_clock_parity_report(&original, &proxy);
    }
    if let Some((original, proxy)) = args.qgs_playout_buffer_tick_paths {
        return qgs_playout_buffer_tick_report(&original, &proxy);
    }
    if let Some((original, proxy)) = args.qgs_runtime_lifecycle_events_paths {
        return qgs_runtime_lifecycle_events_report(&original, &proxy);
    }
    if let Some((original, proxy)) = args.qgs_qnc_event_projection_paths {
        return qgs_qnc_event_projection_report(&original, &proxy);
    }
    if let Some((original, proxy)) = args.qgs_session_command_boundary_paths {
        return qgs_session_command_boundary_report(&original, &proxy);
    }
    if let Some((original, proxy)) = args.qgs_session_runtime_control_paths {
        return qgs_session_runtime_control_report(&original, &proxy);
    }
    if let Some((original, proxy)) = args.qgs_presenter_monitor_boundary_paths {
        return qgs_presenter_monitor_boundary_report(&original, &proxy);
    }
    if let Some((original, proxy)) = args.qgs_file_presenter_diagnostic_paths {
        return qgs_file_presenter_diagnostic_report(
            &original,
            &proxy,
            args.audio_content_audit_output_dir.as_deref(),
        );
    }
    if let Some((original, proxy)) = args.qgs_gpu_payload_readback_diagnostic_paths {
        return qgs_gpu_payload_readback_diagnostic_report(
            &original,
            &proxy,
            args.audio_content_audit_output_dir.as_deref(),
        );
    }
    if let Some((original, proxy)) = args.qgs_runtime_surface_e2e_paths {
        return qgs_runtime_surface_e2e_report(&original, &proxy);
    }
    if let Some((original, proxy)) = args.qgs_broadcast_player_control_core_paths {
        return qgs_broadcast_player_control_core_report(&original, &proxy);
    }
    if let Some((original, proxy)) = args.qgs_broadcast_player_operational_runtime_paths {
        return qgs_broadcast_player_operational_runtime_report(&original, &proxy);
    }
    if let Some((original, proxy)) = args.qgs_broadcast_player_fault_recovery_paths {
        return qgs_broadcast_player_fault_recovery_report(&original, &proxy);
    }
    if let Some((original, proxy)) = args.qgs_broadcast_player_live_paths {
        return qgs_broadcast_player_live_report(
            &original,
            &proxy,
            args.qgs_broadcast_player_live_options,
        );
    }
    if let Some((original, proxy)) = args.qgs_broadcast_player_run_paths {
        return qgs_broadcast_player_run_report(
            &original,
            &proxy,
            args.qgs_broadcast_player_run_options,
        );
    }
    if let Some((original, proxy)) = args.qgs_broadcast_player_control_session_paths {
        return qgs_broadcast_player_control_session_report(
            &original,
            &proxy,
            &args.qgs_broadcast_player_control_session_script,
        );
    }
    if let Some(policy) = args.qgs_device_backend_selection_policy {
        return qgs_device_backend_selection_report(policy);
    }
    if let Some(path) = args.linux_audio_device_probe_path {
        return linux_audio_device_probe(&path);
    }
    if let Some(path) = args.pipewire_audio_prototype_path {
        return pipewire_audio_prototype(&path);
    }
    if let Some(path) = args.pipewire_audio_native_prototype_path {
        return pipewire_audio_native_prototype(&path);
    }
    if let Some(path) = args.pipewire_audio_audible_smoke_test_path {
        return pipewire_audio_audible_smoke_test(&path);
    }
    if let Some(path) = args.pipewire_audio_segment_playback_path {
        return pipewire_audio_segment_playback(&path);
    }
    if let Some((original, proxy)) = args.broadcast_runtime_audio_pipewire_paths {
        return broadcast_runtime_audio_pipewire(&original, &proxy);
    }
    if let Some((original, proxy)) = args.broadcast_runtime_audio_pipewire_audible_paths {
        return broadcast_runtime_audio_pipewire_audible(&original, &proxy);
    }
    if let Some((original, proxy)) = args.pipewire_audio_content_audit_paths {
        return pipewire_audio_content_audit(
            &original,
            &proxy,
            AudioContentAuditOptions {
                start_ms: args.audio_content_audit_start_ms.unwrap_or(0),
                duration_ms: args.audio_content_audit_duration_ms.unwrap_or(1_000),
                output_dir: args
                    .audio_content_audit_output_dir
                    .unwrap_or_else(|| PathBuf::from("target/qgs-audio-audit")),
                find_loudest_range_ms: args.audio_content_audit_find_loudest_range_ms,
                diagnostic_wav_path: args.diagnostic_wav_path,
            },
        );
    }
    if let Some((original, proxy)) = args.pipewire_audio_mono_monitor_paths {
        return pipewire_audio_mono_monitor(
            &original,
            &proxy,
            AudioMonoMonitorOptions {
                start_ms: args.audio_content_audit_start_ms.unwrap_or(0),
                duration_ms: args.audio_content_audit_duration_ms.unwrap_or(1_000),
                source_channel_index: args.audio_monitor_track.unwrap_or(3),
            },
        );
    }
    if let Some((original, proxy)) = args.pipewire_audio_discrete_4mono_paths {
        return pipewire_audio_discrete_4mono(
            &original,
            &proxy,
            AudioDiscrete4MonoOptions {
                start_ms: args.audio_content_audit_start_ms.unwrap_or(0),
                duration_ms: args.audio_content_audit_duration_ms.unwrap_or(1_000),
            },
        );
    }

    let socket_path = args.socket_path;
    let mut stream = connect_socket(&socket_path)?;
    let mut request_id = 1;

    send_message(
        &mut stream,
        &WireMessage::Hello {
            request_id,
            request: HelloRequest::current(),
        },
    )?;

    let response = receive_message(&mut stream)?;
    let WireMessage::Welcome {
        request_id: response_request_id,
        response,
    } = response
    else {
        return Err("expected WELCOME response".into());
    };

    if response_request_id != request_id {
        return Err("response request_id did not match request".into());
    }

    if response.version != CURRENT_PROTOCOL_VERSION {
        return Err("negotiated unexpected protocol version".into());
    }

    println!(
        "QGS session established: version={} session_id={}",
        response.version,
        response.session_id.get()
    );

    request_id += 1;
    send_message(&mut stream, &WireMessage::EnumerateDevices { request_id })?;

    let response = receive_message(&mut stream)?;
    let WireMessage::DeviceList {
        request_id: response_request_id,
        response,
    } = response
    else {
        return Err("expected DEVICE_LIST response".into());
    };

    if response_request_id != request_id {
        return Err("device list request_id did not match request".into());
    }

    println!();
    println!("Devices:");
    for device in &response.devices {
        print_device(device);
        request_id += 1;
        send_message(
            &mut stream,
            &WireMessage::QueryDeviceCapabilities {
                request_id,
                request: QueryDeviceCapabilitiesRequest {
                    device_id: device.id,
                },
            },
        )?;

        let capability_response = receive_message(&mut stream)?;
        let WireMessage::DeviceCapabilities {
            request_id: response_request_id,
            response,
        } = capability_response
        else {
            return Err("expected DEVICE_CAPABILITIES response".into());
        };

        if response_request_id != request_id {
            return Err("capability response request_id did not match request".into());
        }

        if response.capabilities.device_id != device.id {
            return Err("capability response device_id did not match request".into());
        }

        print_capabilities(&response.capabilities);

        request_id += 1;
        send_message(
            &mut stream,
            &WireMessage::QueryVideoCapabilities {
                request_id,
                request: QueryVideoCapabilitiesRequest {
                    device_id: device.id,
                },
            },
        )?;

        let video_response = receive_message(&mut stream)?;
        let WireMessage::VideoCapabilities {
            request_id: response_request_id,
            response,
        } = video_response
        else {
            return Err("expected VIDEO_CAPABILITIES response".into());
        };

        if response_request_id != request_id {
            return Err("video capability response request_id did not match request".into());
        }
        if response.capabilities.device_id != device.id {
            return Err("video capability response device_id did not match request".into());
        }

        print_video_capabilities(&response.capabilities);
    }

    let physical_devices = response
        .devices
        .iter()
        .filter(|device| {
            matches!(
                device.class,
                DeviceClass::IntegratedGpu | DeviceClass::DiscreteGpu
            )
        })
        .collect::<Vec<_>>();

    if args.video_capabilities_only {
        return Ok(());
    }

    if args.h264_decode_only {
        println!();
        println!("H.264 VA-API decode proof:");
        for device in &physical_devices {
            request_id = test_h264_decode(&mut stream, request_id, device)?;
            request_id = test_h264_long_gop_decode(&mut stream, request_id, device)?;
            println!();
        }
        request_id = leave_h264_decoder_for_disconnect(&mut stream, request_id, &physical_devices)?;
        let _ = request_id;
        return Ok(());
    }

    println!();
    println!("Buffers:");

    for device in &physical_devices {
        request_id += 1;
        let resource_id = create_buffer(
            &mut stream,
            request_id,
            device,
            DEMO_BUFFER_SIZE,
            ExternalSharing::None,
        )?;

        request_id += 1;
        destroy_resource(&mut stream, request_id, resource_id)?;
        println!("Resource destroyed successfully.");
        println!();
    }

    println!("External memory sharing:");
    for device in &physical_devices {
        request_id = test_external_memory(&mut stream, request_id, device)?;
        println!();
    }

    println!("External GPU synchronization:");
    for device in &physical_devices {
        request_id = test_external_gpu_sync(&mut stream, request_id, device)?;
        println!();
    }

    println!("Compute proof:");
    for device in &physical_devices {
        request_id = test_compute_proof(&mut stream, request_id, device)?;
        println!();
    }

    println!("Image processing proof:");
    for device in &physical_devices {
        request_id = test_image_processing_proof(&mut stream, request_id, device)?;
        println!();
    }

    println!("Software YUV422P10 GPU upload proof:");
    for device in &physical_devices {
        test_synthetic_yuv422p10_gpu_proof(device)?;
        test_reusable_yuv422p10_gpu_processor(device)?;
        println!();
    }

    println!("H.264 VA-API decode proof:");
    for device in &physical_devices {
        request_id = test_h264_decode(&mut stream, request_id, device)?;
        println!();
    }

    if let Some(device) = physical_devices.first().copied() {
        println!("Creating transient resources for disconnect cleanup on:");
        println!("[{:?}] {}", device.class, device.name);
        for _ in 0..3 {
            request_id += 1;
            let _ = create_buffer(
                &mut stream,
                request_id,
                device,
                64 * 1024,
                ExternalSharing::None,
            )?;
        }
        request_id += 1;
        let _ = create_image(
            &mut stream,
            request_id,
            device,
            IMAGE_PROOF_WIDTH,
            IMAGE_PROOF_HEIGHT,
            ExternalSharing::None,
        )?;
        println!("Leaving 3 transient buffers and 1 transient image alive and disconnecting.");
    }

    Ok(())
}

fn run_haswell_video_diagnostic() -> Result<(), Box<dyn std::error::Error>> {
    println!("QGS M2 Step 4B Haswell imported-video diagnostic");
    println!("Validation layer: VK_LAYER_KHRONOS_validation requested");
    println!("Synchronization validation: requested through VkValidationFeaturesEXT");

    let render_node = PathBuf::from("/dev/dri/renderD128");
    let decoded = qgs_vaapi::decode_h264_drm_prime_for_diagnostic(&render_node, H264_FIXTURE)?;
    println!("VA decode/export:");
    println!("  render node: {}", render_node.display());
    println!("  checksum: 0x{:08x}", decoded.validation_checksum);
    println!("  fourcc: 0x{:08x}", decoded.fourcc);
    println!("  size: {} x {}", decoded.width, decoded.height);
    println!("  objects: {}", decoded.objects.len());
    println!("  layers: {}", decoded.layers.len());
    for (index, object) in decoded.objects.iter().enumerate() {
        println!(
            "  object {index}: size={} modifier={}",
            object.size, object.drm_format_modifier
        );
    }
    for (layer_index, layer) in decoded.layers.iter().enumerate() {
        println!(
            "  layer {layer_index}: drm_format=0x{:08x} planes={}",
            layer.drm_format, layer.num_planes
        );
        for plane_index in 0..layer.num_planes as usize {
            println!(
                "    plane {plane_index}: object={} pitch={} offset={}",
                layer.object_index[plane_index],
                layer.pitch[plane_index],
                layer.offset[plane_index]
            );
        }
    }

    let input = HaswellVideoDiagnosticInput {
        vendor_id: 0x8086,
        device_id: 0x0416,
        width: decoded.width,
        height: decoded.height,
        drm_fourcc: decoded.fourcc,
        objects: decoded
            .objects
            .into_iter()
            .map(|object| DiagnosticDrmObject {
                fd: object.fd,
                size: object.size,
                modifier: object.drm_format_modifier,
            })
            .collect(),
        layers: decoded
            .layers
            .into_iter()
            .map(|layer| DiagnosticDrmLayer {
                drm_format: layer.drm_format,
                planes: (0..layer.num_planes as usize)
                    .map(|index| DiagnosticDrmPlane {
                        object_index: u32::from(layer.object_index[index]),
                        offset: layer.offset[index],
                        pitch: layer.pitch[index],
                    })
                    .collect(),
            })
            .collect(),
    };

    let report = diagnose_haswell_video_import(input)?;
    print_haswell_diagnostic_report(&report);
    Ok(())
}

fn inspect_mxf(path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let source = MediaSource::parse(&bytes)?;

    println!("MXF:");
    println!("  file: {}", mxf_file_label(path));
    if let Some(duration) = source.duration {
        println!("  duration edit units: {duration}");
    }
    if let Some(rate) = source.edit_rate {
        println!("  edit rate: {}/{}", rate.numerator, rate.denominator);
    }
    println!("  KLV triplets: {}", source.klv_count);
    println!("  metadata sets: {}", source.metadata_set_count);
    if let Some(pattern) = source.operational_pattern {
        println!("  operational pattern: {pattern}");
    }
    println!("Partitions:");
    for partition in &source.partitions {
        println!(
            "  {:?}: offset={} body_sid={} index_sid={} header_bytes={} index_bytes={}",
            partition.kind,
            partition.offset,
            partition.body_sid,
            partition.index_sid,
            partition.header_byte_count,
            partition.index_byte_count
        );
    }
    println!("Packages:");
    for package in &source.packages {
        println!(
            "  {:?}: uid=<redacted> tracks={}",
            package.kind,
            package.track_refs.len()
        );
    }
    if let Some(timecode) = &source.timecode {
        println!("Timecode:");
        println!("  start frame: {}", timecode.start_frame);
        println!(
            "  rate: {}/{}",
            timecode.edit_rate.numerator, timecode.edit_rate.denominator
        );
        println!("  drop-frame: {}", yes_no(timecode.drop_frame));
    }

    println!("Tracks:");
    for track in &source.tracks {
        println!("  {:?} track {}", track.kind, track.id.0);
        if let Some(rate) = track.edit_rate {
            println!("    edit rate: {}/{}", rate.numerator, rate.denominator);
        }
        match track.kind {
            TrackKind::Video => {
                if let Some(video) = &track.video {
                    println!("    codec: {:?}", video.codec);
                    println!("    descriptor source: {:?}", video.source);
                    println!(
                        "    dimensions: {} x {}",
                        video.coded_width, video.coded_height
                    );
                    println!(
                        "    display: {} x {}",
                        video.display_width, video.display_height
                    );
                    println!("    bit depth: {}", video.bit_depth);
                    println!("    chroma: {:?}", video.chroma);
                    if let Some(aspect) = video.aspect_ratio {
                        println!(
                            "    aspect ratio: {}/{}",
                            aspect.numerator, aspect.denominator
                        );
                    }
                }
            }
            TrackKind::Audio => {
                if let Some(audio) = &track.audio {
                    println!("    descriptor source: {:?}", audio.source);
                    println!("    channels: {:?}", audio.channels);
                    if let Some(rate) = audio.sample_rate {
                        println!("    sample rate: {}/{}", rate.numerator, rate.denominator);
                    }
                    println!("    bit depth: {:?}", audio.bit_depth);
                }
            }
            TrackKind::Data => {
                if let Some(data) = &track.data {
                    println!("    descriptor source: {:?}", data.source);
                    if let Some(rate) = data.sample_rate {
                        println!("    sample rate: {}/{}", rate.numerator, rate.denominator);
                    }
                    if let Some(essence) = data.essence {
                        println!("    essence: {essence}");
                    }
                }
            }
            TrackKind::Timecode | TrackKind::Other => {}
        }
    }

    let random_access_points = source
        .index
        .video
        .iter()
        .filter(|entry| entry.random_access == RandomAccess::Yes)
        .count();
    let random_access_positions = source
        .index
        .video
        .iter()
        .filter(|entry| entry.random_access == RandomAccess::Yes)
        .map(|entry| entry.edit_unit)
        .collect::<Vec<_>>();
    println!("Index:");
    let index_source = if source
        .index
        .video
        .iter()
        .any(|entry| entry.source == IndexSource::MxfProvided)
    {
        "MXF provided"
    } else {
        "QGS derived"
    };
    println!("  source: {index_source}");
    println!("  video entries: {}", source.index.video.len());
    println!("  random access points: {random_access_points}");
    println!("  random access edit units: {:?}", random_access_positions);
    println!("  index segments: {}", source.index_segments.len());
    for segment in &source.index_segments {
        println!(
            "  segment: body_sid={} index_sid={} entries={} delta_entries={}",
            segment.body_sid,
            segment.index_sid,
            segment.entries.len(),
            segment.delta_entries.len()
        );
    }
    println!("  RIP entries: {}", source.rip.len());

    if !source.diagnostics.is_empty() {
        println!("Warnings:");
        for diagnostic in &source.diagnostics {
            println!("  {:?}: {}", diagnostic.kind, diagnostic.message);
        }
    }

    let target = source.index.video.len().saturating_div(2);
    let start = source
        .index
        .nearest_random_access_before(target as u64)
        .map(|entry| entry.edit_unit as usize)
        .unwrap_or(target);
    println!("Random access proof:");
    println!("  target edit unit: {target}");
    println!("  nearest prior random access: {start}");

    let mut state = qgs_codec_h264::H264DecoderState::new();
    let mut parsed_target = None;
    for index in start..=target {
        let access_unit = source.extract_video_access_unit(&bytes, index)?;
        let parsed = state.parse_access_unit(&access_unit)?;
        state.finish_picture(&parsed)?;
        if index == target {
            parsed_target = Some(parsed);
        }
    }
    let parsed = parsed_target.ok_or("no target access unit parsed")?;
    println!("H.264 classification:");
    println!("  profile: {:?}", parsed.profile);
    println!(
        "  dimensions: {} x {}",
        parsed.desc.coded_width, parsed.desc.coded_height
    );
    println!("  bit depth: {}", parsed.desc.bit_depth.get());
    println!("  chroma: {:?}", parsed.desc.chroma);
    println!(
        "  picture kind: {:?}",
        parsed.slices.first().map(|slice| &slice.kind)
    );
    print_h264_gop_summary(&source, &bytes)?;
    if let Some(sidecar) = SonyXmlSummary::from_mxf_path(path)? {
        print_sony_xml_comparison(&source, &parsed, &sidecar);
    }

    Ok(())
}

fn software_decode_mxf(path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let source = MediaSource::parse(&bytes)?;
    let first_access_unit = source.extract_video_access_unit(&bytes, 0)?;
    let parsed = qgs_codec_h264::parse_annex_b_access_unit(&first_access_unit)?;
    let config = decoder_config_for_surface(
        qgs_protocol::DeviceId::new(1)?,
        parsed.profile,
        parsed.desc.bit_depth,
        parsed.desc.chroma,
        parsed.desc.coded_width,
        parsed.desc.coded_height,
    );
    if !SoftwareVideoBackend::supports_config(&config) {
        return Err("software backend does not support parsed stream".into());
    }

    let positioned = source
        .index
        .video
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            Ok((
                entry.edit_unit,
                source.extract_video_access_unit(&bytes, index)?,
            ))
        })
        .collect::<Result<Vec<_>, qgs_mxf::MxfError>>()?;
    let sequential = decode_positioned_access_units_with_context(config.clone(), &positioned)?;

    let target = 53_u64.min(source.index.video.len().saturating_sub(1) as u64);
    let start = source
        .index
        .nearest_random_access_before(target)
        .map(|entry| entry.edit_unit)
        .unwrap_or(target);
    let random_positioned = (start..=target)
        .map(|edit_unit| {
            let index = usize::try_from(edit_unit)?;
            Ok((edit_unit, source.extract_video_access_unit(&bytes, index)?))
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    let random = decode_positioned_access_units_with_context(config, &random_positioned)?;

    let sequential_target = sequential
        .frames
        .iter()
        .find(|frame| frame.presentation_index == target)
        .ok_or("sequential target frame missing")?;
    let random_target = random
        .frames
        .iter()
        .find(|frame| frame.presentation_index == target)
        .ok_or("random-access target frame missing")?;
    if sequential_target.checksum != random_target.checksum {
        return Err("random-access target checksum did not match sequential decode".into());
    }

    let first = sequential.frames.first().ok_or("no decoded frames")?;
    let middle = sequential_target;
    let final_frame = sequential.frames.last().ok_or("no final decoded frame")?;
    let plane_count = first.planes.len();
    let owned_bytes = first.owned_bytes();
    let peak_owned = owned_bytes
        .checked_mul(sequential.max_live_surfaces.max(1))
        .ok_or("software surface memory overflow")?;
    let fps = if sequential.elapsed.as_secs_f64() > 0.0 {
        sequential.frames.len() as f64 / sequential.elapsed.as_secs_f64()
    } else {
        0.0
    };

    println!("Software H.264 decode proof:");
    println!("  file: {}", mxf_file_label(path));
    println!("  backend: rsmpeg/libavcodec");
    println!("  parsed profile: {:?}", parsed.profile);
    println!(
        "  parsed format: {} x {}, {}-bit {:?}",
        parsed.desc.coded_width,
        parsed.desc.coded_height,
        parsed.desc.bit_depth.get(),
        parsed.desc.chroma
    );
    println!("  software fallback selected: yes");
    println!("  decoded frames: {}", sequential.frames.len());
    println!("  decoder pixel format: {}", first.decoder_pixel_format);
    println!("  storage format: {:?}", first.storage_format);
    println!("  plane count: {plane_count}");
    for (index, plane) in first.planes.iter().enumerate() {
        println!(
            "  plane {index}: width_samples={} height={} stride={} source_stride={} bytes={}",
            plane.width_samples,
            plane.height,
            plane.stride_bytes,
            plane.source_stride_bytes,
            plane.data.len()
        );
    }
    println!("  first checksum: 0x{:016x}", first.checksum);
    println!(
        "  sequential target {} checksum: 0x{:016x}",
        target, middle.checksum
    );
    println!(
        "  random target {} checksum: 0x{:016x}",
        target, random_target.checksum
    );
    println!("  final checksum: 0x{:016x}", final_frame.checksum);
    println!("  random-access target: {target}");
    println!("  nearest prior random access: {start}");
    println!("  random-access checksum match: yes");
    println!("  bytes per software surface: {owned_bytes}");
    println!(
        "  max simultaneously live QGS software surfaces: {}",
        sequential.max_live_surfaces
    );
    println!("  approximate peak QGS-owned frame memory: {peak_owned} bytes");
    println!(
        "  DEVELOPMENT OBSERVATION - NOT A BENCHMARK: {:.3}s, {:.2} fps",
        sequential.elapsed.as_secs_f64(),
        fps
    );

    Ok(())
}

fn software_gpu_mxf(path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let source = MediaSource::parse(&bytes)?;
    let first_access_unit = source.extract_video_access_unit(&bytes, 0)?;
    let parsed = qgs_codec_h264::parse_annex_b_access_unit(&first_access_unit)?;
    let config = decoder_config_for_surface(
        qgs_protocol::DeviceId::new(1)?,
        parsed.profile,
        parsed.desc.bit_depth,
        parsed.desc.chroma,
        parsed.desc.coded_width,
        parsed.desc.coded_height,
    );
    if !SoftwareVideoBackend::supports_config(&config) {
        return Err("software backend does not support parsed stream".into());
    }

    let positioned = source
        .index
        .video
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            Ok((
                entry.edit_unit,
                source.extract_video_access_unit(&bytes, index)?,
            ))
        })
        .collect::<Result<Vec<_>, qgs_mxf::MxfError>>()?;
    let sequential_start = Instant::now();
    let sequential = decode_positioned_access_units_with_context(config.clone(), &positioned)?;
    let sequential_decode_elapsed = sequential_start.elapsed();

    let target = 53_u64.min(source.index.video.len().saturating_sub(1) as u64);
    let start = source
        .index
        .nearest_random_access_before(target)
        .map(|entry| entry.edit_unit)
        .unwrap_or(target);
    let random_positioned = (start..=target)
        .map(|edit_unit| {
            let index = usize::try_from(edit_unit)?;
            Ok((edit_unit, source.extract_video_access_unit(&bytes, index)?))
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    let random = decode_positioned_access_units_with_context(config, &random_positioned)?;

    let first = sequential.frames.first().ok_or("no decoded frames")?;
    let middle = sequential
        .frames
        .iter()
        .find(|frame| frame.presentation_index == target)
        .ok_or("sequential target frame missing")?;
    let random_target = random
        .frames
        .iter()
        .find(|frame| frame.presentation_index == target)
        .ok_or("random-access target frame missing")?;
    let final_frame = sequential.frames.last().ok_or("no final decoded frame")?;

    let discovery = VulkanDeviceDiscovery::new()?;
    let devices = discovery.enumerate_devices()?;
    let physical_devices = devices
        .iter()
        .filter(|device| {
            matches!(
                device.class,
                DeviceClass::IntegratedGpu | DeviceClass::DiscreteGpu
            )
        })
        .collect::<Vec<_>>();
    if physical_devices.is_empty() {
        return Err("no physical Vulkan devices for software GPU proof".into());
    }

    println!("Software YUV422P10 -> GPU processing proof:");
    println!("  file: {}", mxf_file_label(path));
    println!("  parsed profile: {:?}", parsed.profile);
    println!(
        "  parsed format: {} x {}, {}-bit {:?}",
        parsed.desc.coded_width,
        parsed.desc.coded_height,
        parsed.desc.bit_depth.get(),
        parsed.desc.chroma
    );
    println!(
        "  software decoder pixel format: {}",
        first.decoder_pixel_format
    );
    println!("  sequential decoded frames: {}", sequential.frames.len());
    println!("  random-access target: {target}");
    println!("  nearest prior random access: {start}");
    println!(
        "  software decode observation: {:.3}s for {} frames",
        sequential_decode_elapsed.as_secs_f64(),
        sequential.frames.len()
    );

    for device in physical_devices {
        println!();
        println!("[{:?}] {}", device.class, device.name);
        let first_six = sequential
            .frames
            .iter()
            .take(6)
            .enumerate()
            .map(|(index, frame)| (format!("seq-{index}"), frame))
            .collect::<Vec<_>>();
        run_software_gpu_sequence_proof(
            &discovery,
            device,
            "first-six",
            &first_six
                .iter()
                .map(|(label, frame)| (label.as_str(), *frame))
                .collect::<Vec<_>>(),
        )?;
        let middle_outputs = run_software_gpu_sequence_proof(
            &discovery,
            device,
            "frame-53-sequential-random",
            &[("sequential-53", middle), ("random-53", random_target)],
        )?;
        let sequential_middle = *middle_outputs
            .get("sequential-53")
            .ok_or("missing sequential frame 53 GPU output")?;
        let random_middle = *middle_outputs
            .get("random-53")
            .ok_or("missing random frame 53 GPU output")?;
        if sequential_middle != random_middle {
            return Err("sequential/random frame 53 GPU checksum mismatch".into());
        }
        run_software_gpu_sequence_proof(&discovery, device, "final", &[("final", final_frame)])?;
        println!("  sequential/random frame 53 GPU match: yes");
    }

    Ok(())
}

fn proxy_proof(
    socket_path: &Path,
    original_path: &Path,
    proxy_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let original_sha256 = sha256_hex(original_path)?;
    let proxy_sha256 = sha256_hex(proxy_path)?;
    let case = identify_camera_case(&original_sha256, &proxy_sha256)?;
    if case.damaged_proxy {
        return diagnose_damaged_proxy_case(case, proxy_path);
    }

    let original_bytes = std::fs::read(original_path)?;
    let original = MediaSource::parse(&original_bytes)?;
    let proxy = Mp4Source::open(proxy_path)?;
    let proxy_video = proxy
        .video
        .as_ref()
        .ok_or("proxy has no H.264 video track")?;
    let proxy_h264 = classify_video_track(proxy_video)?;
    let presentation_ordinals = proxy_presentation_ordinals(proxy_video)?;

    println!("Camera original/proxy proof:");
    println!("  original: {}", case.original_label);
    println!("  proxy: {}", case.proxy_label);
    println!("  hashes: verified");
    println!("  association: filename ignored for proof metadata");
    println!("Proxy MP4:");
    println!("  major brand: {}", proxy.major_brand);
    println!("  compatible brands: {:?}", proxy.compatible_brands);
    println!("  movie timescale: {}", proxy.movie_timescale);
    println!("Tracks:");
    for track in &proxy.tracks {
        println!(
            "  track {}: kind={:?} handler={} codec={} timescale={} duration={} samples={}",
            track.track_id,
            track.kind,
            track.handler,
            track.codec,
            track.timescale,
            track.duration_units,
            track.sample_count
        );
        if track.kind == Mp4TrackKind::Video {
            println!("    dimensions: {:?} x {:?}", track.width, track.height);
        }
        if track.kind == Mp4TrackKind::Audio {
            println!(
                "    sample_rate={:?} channels={:?}",
                track.sample_rate, track.channels
            );
        }
    }
    println!("Proxy video:");
    println!(
        "  H.264 {:?}, {}-bit {:?}, {} x {}",
        proxy_h264.profile,
        proxy_h264.bit_depth,
        proxy_h264.chroma,
        proxy_h264.width,
        proxy_h264.height
    );
    println!(
        "  rate: {}/{}",
        proxy_video.frame_rate.numerator, proxy_video.frame_rate.denominator
    );
    println!("  samples: {}", proxy_video.samples.len());
    println!("  duration units: {}", proxy_video.duration_units);
    println!("  timescale: {}", proxy_video.timescale);
    println!("  nal length size: {}", proxy_video.nal_length_size);
    println!(
        "  avcC parameter sets: SPS={} PPS={}",
        proxy_video.sps_count, proxy_video.pps_count
    );
    println!("  picture counts: {:?}", proxy_h264.picture_counts);
    println!("  random access positions: {:?}", proxy_h264.idr_positions);

    let original_video_entries = original.index.video.len();
    println!("Original/proxy timing:");
    println!("  original edit units: {original_video_entries}");
    println!(
        "  proxy presentation samples: {}",
        proxy_video.samples.len()
    );
    println!("  original duration: {:?}", original.duration);
    println!(
        "  original edit rate: {:?}",
        original
            .edit_rate
            .map(|rate| format!("{}/{}", rate.numerator, rate.denominator))
    );
    println!(
        "  proxy duration/rate: {}/{} units at {}/{} fps",
        proxy_video.duration_units,
        proxy_video.timescale,
        proxy_video.frame_rate.numerator,
        proxy_video.frame_rate.denominator
    );
    if original_video_entries == proxy_video.samples.len() {
        println!("  one proxy presentation frame per original edit unit: yes");
    } else {
        println!("  one proxy presentation frame per original edit unit: no");
    }

    let sidecar = SonyXmlSummary::from_mxf_path(original_path)?;
    if let Some(sidecar) = &sidecar {
        println!("Sony XML sidecar diagnostic:");
        println!("  XML duration edit units: {:?}", sidecar.duration);
        println!(
            "  XML reports proxy/substream metadata: {}",
            yes_no(sidecar.has_proxy_metadata)
        );
        println!("  XML video codec: {:?}", sidecar.video_codec);
        println!("  XML fps: {:?}", sidecar.format_fps);
        println!("  XML layout: {:?} x {:?}", sidecar.width, sidecar.height);
    }
    let strong_metadata = sidecar
        .as_ref()
        .map(|xml| {
            xml.has_proxy_metadata
                && xml.duration == Some(proxy_video.samples.len() as u64)
                && xml.width == Some(proxy_h264.width)
                && xml.height == Some(proxy_h264.height)
        })
        .unwrap_or(false);
    println!(
        "  association confidence: {}",
        if strong_metadata {
            "strong metadata + timing evidence"
        } else {
            "supporting timing evidence only"
        }
    );

    let proxy_config = decoder_config_for_surface(
        qgs_protocol::DeviceId::new(1)?,
        proxy_h264.profile,
        BitDepth::new(proxy_h264.bit_depth)?,
        proxy_h264.chroma,
        proxy_h264.coded_width,
        proxy_h264.coded_height,
    );
    if !SoftwareVideoBackend::supports_config(&proxy_config) {
        return Err("software backend does not support proxy H.264 stream".into());
    }
    let proxy_positioned = proxy_video
        .samples
        .iter()
        .map(|sample| {
            Ok((
                *presentation_ordinals
                    .get(&sample.sample_index)
                    .ok_or("missing proxy presentation ordinal")?,
                sample.annex_b.clone(),
            ))
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    let proxy_decode_start = Instant::now();
    let proxy_sequential =
        decode_positioned_access_units_with_context(proxy_config.clone(), &proxy_positioned)?;
    let proxy_decode_elapsed = proxy_decode_start.elapsed();
    if proxy_sequential.frames.len() != proxy_video.samples.len() {
        return Err(format!(
            "proxy software reference expected {} frames, got {}",
            proxy_video.samples.len(),
            proxy_sequential.frames.len()
        )
        .into());
    }

    let target = 53_u64.min(proxy_video.samples.len().saturating_sub(1) as u64);
    let target_decode_index = presentation_ordinals
        .iter()
        .find_map(|(sample_index, ordinal)| (*ordinal == target).then_some(*sample_index))
        .ok_or("target proxy presentation frame not found")?;
    let start_decode_index = nearest_random_access_before(proxy_video, target_decode_index)
        .ok_or("proxy random-access point not found")?;
    let proxy_random_positioned = proxy_video
        .samples
        .iter()
        .filter(|sample| {
            sample.sample_index >= start_decode_index && sample.sample_index <= target_decode_index
        })
        .map(|sample| {
            Ok((
                *presentation_ordinals
                    .get(&sample.sample_index)
                    .ok_or("missing random proxy presentation ordinal")?,
                sample.annex_b.clone(),
            ))
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    let proxy_random = decode_positioned_access_units_with_context(
        proxy_config.clone(),
        &proxy_random_positioned,
    )?;
    let proxy_seq_53 = proxy_sequential
        .frames
        .iter()
        .find(|frame| frame.presentation_index == target)
        .ok_or("proxy sequential target frame missing")?;
    let proxy_rand_53 = proxy_random
        .frames
        .iter()
        .find(|frame| frame.presentation_index == target)
        .ok_or("proxy random target frame missing")?;
    if proxy_seq_53.checksum != proxy_rand_53.checksum {
        return Err("proxy sequential/random frame 53 software checksum mismatch".into());
    }

    println!("Proxy software reference decode:");
    println!("  decoded frames: {}", proxy_sequential.frames.len());
    println!(
        "  decoder pixel format: {}",
        proxy_sequential
            .frames
            .first()
            .map(|frame| frame.decoder_pixel_format.as_str())
            .unwrap_or("n/a")
    );
    println!(
        "  random access target: {} start decode sample: {}",
        target, start_decode_index
    );
    println!("  frame 53 sequential/random checksum match: yes");
    println!(
        "  DEVELOPMENT OBSERVATION - NOT A BENCHMARK: {:.3}s, {:.2} fps",
        proxy_decode_elapsed.as_secs_f64(),
        proxy_sequential.frames.len() as f64 / proxy_decode_elapsed.as_secs_f64().max(0.000_001)
    );

    let original_positioned = original
        .index
        .video
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            Ok((
                entry.edit_unit,
                original.extract_video_access_unit(&original_bytes, index)?,
            ))
        })
        .collect::<Result<Vec<_>, qgs_mxf::MxfError>>()?;
    let first_original_au = original.extract_video_access_unit(&original_bytes, 0)?;
    let original_parsed = qgs_codec_h264::parse_annex_b_access_unit(&first_original_au)?;
    let original_config = decoder_config_for_surface(
        qgs_protocol::DeviceId::new(1)?,
        original_parsed.profile,
        original_parsed.desc.bit_depth,
        original_parsed.desc.chroma,
        original_parsed.desc.coded_width,
        original_parsed.desc.coded_height,
    );
    let original_decode_start = Instant::now();
    let original_decoded =
        decode_positioned_access_units_with_context(original_config, &original_positioned)?;
    let original_decode_elapsed = original_decode_start.elapsed();
    let original_53 = original_decoded
        .frames
        .iter()
        .find(|frame| frame.presentation_index == target)
        .ok_or("original target frame missing")?;
    let correspondence = luma_signature_delta(original_53, proxy_seq_53)?;
    println!("Original/proxy frame correspondence:");
    println!("  compared frame: {target}");
    println!(
        "  original luma signature: 0x{:016x}",
        correspondence.original_hash
    );
    println!(
        "  proxy luma signature: 0x{:016x}",
        correspondence.proxy_hash
    );
    println!(
        "  mean absolute luma signature delta: {:.2}",
        correspondence.mean_abs_delta
    );
    println!(
        "  original software decode observation: {:.3}s, {:.2} fps",
        original_decode_elapsed.as_secs_f64(),
        original_decoded.frames.len() as f64 / original_decode_elapsed.as_secs_f64().max(0.000_001)
    );

    run_proxy_hardware_decode(socket_path, &proxy_config, proxy_video)?;

    let discovery = VulkanDeviceDiscovery::new()?;
    let devices = discovery.enumerate_devices()?;
    let physical_devices = devices
        .iter()
        .filter(|device| {
            matches!(
                device.class,
                DeviceClass::IntegratedGpu | DeviceClass::DiscreteGpu
            )
        })
        .collect::<Vec<_>>();
    println!("Proxy -> GPU proof:");
    for device in physical_devices {
        println!("  [{:?}] {}", device.class, device.name);
        let first = proxy_sequential
            .frames
            .first()
            .ok_or("missing proxy first frame")?;
        let final_frame = proxy_sequential
            .frames
            .last()
            .ok_or("missing proxy final frame")?;
        run_proxy_gpu_frame(&discovery, device, "proxy-first", first)?;
        run_proxy_gpu_frame(&discovery, device, "proxy-53-sequential", proxy_seq_53)?;
        run_proxy_gpu_frame(&discovery, device, "proxy-53-random", proxy_rand_53)?;
        run_proxy_gpu_frame(&discovery, device, "proxy-final", final_frame)?;
    }
    println!("VA->Vulkan zero-copy path: frozen / not used");

    Ok(())
}

#[derive(Clone, Copy)]
struct CameraCase {
    original_label: &'static str,
    proxy_label: &'static str,
    damaged_proxy: bool,
}

fn identify_camera_case(
    original_sha256: &str,
    proxy_sha256: &str,
) -> Result<CameraCase, Box<dyn std::error::Error>> {
    match (original_sha256, proxy_sha256) {
        (EXPECTED_FX6_SAMPLE001_MXF_SHA256, EXPECTED_FX6_SAMPLE001_PROXY_SHA256) => {
            Ok(CameraCase {
                original_label: "Sony FX6 sample 001",
                proxy_label: "Sony FX6 sample 001 proxy",
                damaged_proxy: true,
            })
        }
        (EXPECTED_FX6_SAMPLE002_MXF_SHA256, EXPECTED_FX6_SAMPLE002_PROXY_SHA256) => {
            Ok(CameraCase {
                original_label: "Sony FX6 sample 002",
                proxy_label: "Sony FX6 sample 002 proxy",
                damaged_proxy: false,
            })
        }
        _ => Err("external camera case hashes do not match known Step 12 corpus".into()),
    }
}

fn diagnose_damaged_proxy_case(
    case: CameraCase,
    proxy_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Camera original/proxy proof:");
    println!("  original: {}", case.original_label);
    println!("  proxy: {}", case.proxy_label);
    println!("  hashes: verified");
    match Mp4Source::open(proxy_path) {
        Ok(_) => Err("damaged proxy unexpectedly parsed as valid MP4/H.264".into()),
        Err(error) => {
            let health = MediaHealth::from_open_error(&error);
            println!("Damaged proxy diagnostic:");
            println!("  health: {:?}", health);
            println!("  strict parser result: {error}");
            if health != MediaHealth::DamagedUnrecoverable {
                return Err(format!("expected damaged/unrecoverable proxy, got {health:?}").into());
            }
            println!("  complete proxy playback acceptance: not attempted");
            Ok(())
        }
    }
}

fn qnc_journalist_demo(
    original_path: &Path,
    proxy_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let original_sha256 = sha256_hex(original_path)?;
    let proxy_sha256 = sha256_hex(proxy_path)?;
    let case = identify_camera_case(&original_sha256, &proxy_sha256)?;
    if case.damaged_proxy {
        return diagnose_damaged_proxy_case(case, proxy_path);
    }

    let original_bytes = std::fs::read(original_path)?;
    let original = MediaSource::parse(&original_bytes)?;
    let proxy = Mp4Source::open(proxy_path)?;
    let proxy_video = proxy
        .video
        .as_ref()
        .ok_or("proxy has no H.264 video track")?;
    let proxy_h264 = classify_video_track(proxy_video)?;
    let source_frames = proxy_video.samples.len();
    let audio_report = build_original_audio_report(&original, proxy_video)?;
    let preview_frames =
        ProxyPlaybackProfile::Journalist50iPreview.selected_frame_count(source_frames);
    let intentional_skips = source_frames.saturating_sub(preview_frames);
    let preview_rate = RationalRate::new(25, 1)?;
    let cut_start_preview = 10_u64;
    let cut_end_preview_exclusive = 40_u64;
    let cut_start_source = cut_start_preview
        .checked_mul(2)
        .ok_or("cut source start overflow")?;
    let cut_end_source_exclusive = cut_end_preview_exclusive
        .checked_mul(2)
        .ok_or("cut source end overflow")?;
    let cut_preview_frames = cut_end_preview_exclusive
        .checked_sub(cut_start_preview)
        .ok_or("invalid cut range")?;
    let cut_duration = preview_rate.duration_for_frames(cut_preview_frames)?;
    let cut_start_time = preview_rate.frame_offset(cut_start_preview)?;
    let cut_end_time = preview_rate.frame_offset(cut_end_preview_exclusive)?;

    println!("QNC Journalist Demo");
    println!("-------------------");
    println!("Original: {}", case.original_label);
    println!("Proxy: {}", case.proxy_label);
    println!("Association: strong metadata + timing evidence from existing Step 12 corpus");
    println!(
        "Source: {}x{} visible, {}x{} coded, H.264 {:?}, {}-bit {:?}, {}/{} fps proxy",
        proxy_h264.width,
        proxy_h264.height,
        proxy_h264.coded_width,
        proxy_h264.coded_height,
        proxy_h264.profile,
        proxy_h264.bit_depth,
        proxy_h264.chroma,
        proxy_video.frame_rate.numerator,
        proxy_video.frame_rate.denominator
    );
    println!("Original edit units: {}", original.index.video.len());
    print_original_audio_report(&audio_report);
    println!("Preview profile: journalist-50i-preview");
    println!("Broadcast target: 1080i50-compatible news preview");
    println!("Processing workload: 25 frame periods/s");
    println!("True interlaced output: not implemented in this milestone");
    println!("Source frames: {source_frames}");
    println!("Preview frames: {preview_frames}");
    println!("Intentional profile skips: {intentional_skips}");
    println!("Running preview acceptance path...");

    proxy_playback(
        original_path,
        proxy_path,
        ProxyPlaybackProfile::Journalist50iPreview,
    )?;

    println!("QNC Journalist Demo Summary");
    println!("---------------------------");
    println!("Preview profile: journalist-50i-preview");
    println!("Broadcast target: 1080i50-compatible news preview");
    println!("Processing workload: 25 frame periods/s");
    println!("True interlaced output: not implemented in this milestone");
    println!("Source frames: {source_frames}");
    println!("Preview frames: {preview_frames}");
    println!("Intentional profile skips: {intentional_skips}");
    println!("Lateness drops: 0");
    println!("Duplicated frames: 0");
    println!("GPU processing: completed for all selected preview frames");
    println!("Audio source: original MXF");
    println!("Video source: proxy MP4");
    println!(
        "A/V timing: original_audio={:.3}s proxy_video={:.3}s delta={:.3} ms",
        audio_report.audio_duration.as_secs_f64(),
        audio_report.proxy_video_duration.as_secs_f64(),
        audio_report.duration_delta.as_secs_f64() * 1000.0
    );
    println!(
        "Audio timeline: tracks={} total_channels={} sample_rate={} bit_depth={:?} bounded_queue_peak={} payload_extraction={} indexed_payload_packets={}",
        audio_report.timeline.tracks.len(),
        audio_report.total_channels,
        audio_report
            .uniform_sample_rate
            .map(|value| value.to_string())
            .unwrap_or_else(|| "mixed".to_string()),
        audio_report.uniform_bit_depth,
        audio_report.audio_queue_peak,
        if audio_report.audio_payload_packets_indexed == 0 {
            "not available"
        } else {
            "available"
        },
        audio_report.audio_payload_packets_indexed
    );
    println!(
        "A/V clock foundation: max selected video timestamp outside audio range {:.3} ms, usable_as_future_master_clock={}",
        audio_report.max_selected_video_outside_audio.as_secs_f64() * 1000.0,
        yes_no(audio_report.usable_as_master_clock)
    );
    println!("News cut:");
    println!(
        "  start: preview frame {} / source frame {} / {:.3}s",
        cut_start_preview,
        cut_start_source,
        cut_start_time.as_secs_f64()
    );
    println!(
        "  end: preview frame {} / source frame {} / {:.3}s",
        cut_end_preview_exclusive,
        cut_end_source_exclusive,
        cut_end_time.as_secs_f64()
    );
    println!("  preview frames in cut: {cut_preview_frames}");
    println!("  selected source frames in cut: {cut_preview_frames}");
    println!("  source frames skipped by profile in cut: {cut_preview_frames}");
    println!("  estimated duration: {:.3}s", cut_duration.as_secs_f64());
    println!("QNC export plan:");
    println!("  status: planned only, not rendered");
    println!("  profile: journalist-50i-preview");
    println!("  source: Sony FX6 original/proxy pair");
    println!("  preview media: proxy MP4");
    println!("  audio source: original MXF");
    println!("  finishing media: original MXF available");
    println!(
        "  selected range: {:.3}s..{:.3}s",
        cut_start_time.as_secs_f64(),
        cut_end_time.as_secs_f64()
    );
    println!("  target delivery: future milestone");

    Ok(())
}

fn qnc_prepared_input_descriptor_report(
    original_path: &Path,
    proxy_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let descriptor = build_qnc_prepared_input_descriptor(original_path, proxy_path)?;
    let validation = descriptor.validate();

    println!("QNC Prepared Input Descriptor Compatibility");
    println!("-------------------------------------------");
    println!(
        "Public original URI: {}",
        descriptor.binding.original_media_uri
    );
    println!(
        "Public proxy URI: {}",
        descriptor
            .binding
            .proxy_media_uri
            .as_deref()
            .unwrap_or("<none>")
    );
    println!(
        "Private original path binding present: {}",
        yes_no(descriptor.binding.private_original_path_bound)
    );
    println!(
        "Private proxy path binding present: {}",
        yes_no(descriptor.binding.private_proxy_path_bound)
    );
    println!(
        "Selected picture representation: {:?}",
        descriptor.selected_picture
    );
    println!(
        "Authoritative audio representation: {:?}",
        descriptor.authoritative_audio
    );
    println!("Proxy AAC authoritative: no");
    println!(
        "Original/proxy association: {:?}",
        descriptor.binding.association_status
    );
    println!(
        "Audio channel count: {}",
        descriptor.audio_layout.channels.len()
    );
    println!("Mono channel identities:");
    for channel in &descriptor.audio_layout.channels {
        println!(
            "  lane {}: track_id={} channel_index={}",
            channel.lane_index + 1,
            channel.track_id,
            channel.channel_index
        );
    }
    println!(
        "Source duration: {:.3}s",
        descriptor.layout.original_video.duration.as_secs_f64()
    );
    println!(
        "Video timebase: {}/{} fps",
        descriptor.layout.original_video.timebase.numerator(),
        descriptor.layout.original_video.timebase.denominator()
    );
    println!(
        "Source duration frames: {}",
        descriptor.layout.original_video.duration_frames
    );
    println!(
        "Proxy duration frames: {}",
        descriptor
            .layout
            .proxy_video
            .map(|video| video.duration_frames)
            .unwrap_or(0)
    );
    println!(
        "Audio sample rate: {} Hz",
        descriptor.audio_layout.sample_rate
    );
    println!("Audio bit depth: {} bit", descriptor.audio_layout.bit_depth);
    println!(
        "URI identity separate from private path binding: {}",
        yes_no(
            descriptor.identity.public_uris_are_valid()
                && descriptor.binding.public_uris_are_valid()
        )
    );
    println!("Validation result: {}", result_label(&validation));

    validation?;
    Ok(())
}

fn qgs_input_plan_report(
    original_path: &Path,
    proxy_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let descriptor = build_qnc_prepared_input_descriptor(original_path, proxy_path)?;
    let descriptor_validation = descriptor.validate();
    let queue_requirements = default_qgs_input_plan_queue_requirements();
    let plan = QgsInputPlan::from_descriptor(&descriptor, queue_requirements)?;

    println!("QGS InputPlan Compatibility");
    println!("---------------------------");
    println!(
        "Descriptor validation result: {}",
        result_label(&descriptor_validation)
    );
    println!("Source mode: {:?}", plan.source_mode);
    println!(
        "Video source plan: representation={:?} uri={} frames={} timebase={}/{}",
        plan.video_source.representation,
        plan.video_source.media_uri,
        plan.video_source.duration_frames,
        plan.video_source.timebase.numerator(),
        plan.video_source.timebase.denominator()
    );
    println!(
        "Audio source plan: representation={:?} uri={} sample_rate={}Hz bit_depth={} lanes={}",
        plan.audio_source.representation,
        plan.audio_source.media_uri,
        plan.audio_source.sample_rate,
        plan.audio_source.bit_depth,
        plan.audio_source.lanes.len()
    );
    println!(
        "Original/proxy timing compatible: {}",
        yes_no(descriptor.layout.proxy_original_timing_compatible())
    );
    println!("Audio lane layout:");
    for lane in &plan.audio_source.lanes {
        println!(
            "  lane {}: original track {} channel {}",
            lane.lane_index + 1,
            lane.track_id,
            lane.channel_index
        );
    }
    println!(
        "Queue requirements: min_video_frames={} min_audio_ranges={} max_video_queue={} max_audio_queue={}",
        plan.queue_requirements.min_video_frames,
        plan.queue_requirements.min_audio_ranges,
        plan.queue_requirements.max_video_queue,
        plan.queue_requirements.max_audio_queue
    );
    println!(
        "Capability requirements: original_audio={} discrete_mono_lanes={} proxy_aac_diagnostic_only={} uri_identity={}",
        yes_no(plan.capability_requirements.requires_original_audio),
        yes_no(plan.capability_requirements.requires_discrete_mono_lanes),
        yes_no(plan.capability_requirements.requires_proxy_aac_diagnostic_only),
        yes_no(plan.capability_requirements.requires_uri_identity)
    );
    println!(
        "1000 ms maps to audio samples: {}",
        plan.samples_for_duration(Duration::from_millis(1_000))?
    );
    println!("Not implemented in Phase 22:");
    println!("  transport engine: no");
    println!("  frame clock: no");
    println!("  playout buffer: no");
    println!("  presenter: no");
    println!("  realtime: no");
    println!("  A/V sync: no");

    descriptor_validation?;
    Ok(())
}

fn qgs_qnc_prepared_input_fixture_report(name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let input = match name {
        "proxy-preview" => QgsQncPreparedInputLike::frozen_block_x_proxy_preview(),
        "original-media" => QgsQncPreparedInputLike::frozen_block_x_original_media(),
        _ => {
            return Err("fixture expects proxy-preview or original-media, not a media path".into());
        }
    };
    let mapping = input.map_to_qgs_descriptor()?;
    let plan = QgsInputPlan::from_descriptor(
        &mapping.descriptor,
        default_qgs_input_plan_queue_requirements(),
    )?;
    let public_text = format!(
        "{} {} {} {}",
        mapping.descriptor.identity.source_record_uri,
        mapping.descriptor.identity.workspace_db_uri,
        mapping.descriptor.binding.original_media_uri,
        mapping
            .descriptor
            .binding
            .proxy_media_uri
            .as_deref()
            .unwrap_or("")
    );
    println!("QGS QNC PreparedInput Fixture");
    println!("-----------------------------");
    println!("Fixture: {name}");
    println!(
        "Shared contract: {} {}",
        qgs_media_runtime::qnc_qgs_contract::CONTRACT_NAME,
        qgs_media_runtime::qnc_qgs_contract::CONTRACT_VERSION
    );
    println!("Derived from Phase 22 descriptor: no");
    println!("Playback input: {:?}", input.playback_input);
    println!("Source mode: {:?}", plan.source_mode);
    println!(
        "Selected picture: {:?}",
        mapping.descriptor.selected_picture
    );
    println!(
        "Authoritative audio: {:?}",
        mapping.descriptor.authoritative_audio
    );
    println!(
        "Proxy AAC authoritative: {}",
        yes_no(mapping.descriptor.audio_layout.proxy_aac_authoritative)
    );
    println!(
        "Original audio lanes: {}",
        mapping.descriptor.audio_layout.channels.len()
    );
    println!(
        "Active range: {}",
        mapping
            .active_range
            .map(|range| format!("{}..{}", range.start_frame, range.end_frame_exclusive))
            .unwrap_or_else(|| "none".to_string())
    );
    println!(
        "Private binding ref in public output: {}",
        yes_no(public_text.contains(&input.private_binding.original_binding_ref))
    );
    println!(
        "Private path exposed: {}",
        yes_no(
            mapping.private_path_exposed
                || public_text.starts_with('/')
                || public_text.contains("file:")
        )
    );
    println!("Realtime: no");
    println!("A/V sync: no");
    println!("Real display: no");
    println!("Production audio: no");
    Ok(())
}

fn qgs_qnc_prepared_input_mapping_report(
    original_path: &Path,
    proxy_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let qnc_like = build_qgs_qnc_prepared_input_like(original_path, proxy_path)?;
    let validation = qnc_like.validate_for_qgs_mapping();
    let mapping = qnc_like.map_to_qgs_descriptor();
    let (descriptor_created, plan_created, assembly_created, descriptor, plan) = match mapping {
        Ok(mapping) => {
            let descriptor = mapping.descriptor;
            let plan = QgsInputPlan::from_descriptor(
                &descriptor,
                default_qgs_input_plan_queue_requirements(),
            );
            let assembly = plan
                .as_ref()
                .ok()
                .and_then(|plan| QgsBroadcastPlayerAssembly::from_input_plan(plan.clone()).ok());
            (
                true,
                plan.is_ok(),
                assembly.is_some(),
                Some(descriptor),
                plan.ok(),
            )
        }
        Err(error) => {
            println!("Mapping error: {error}");
            (false, false, false, None, None)
        }
    };

    println!("QGS QNC PreparedInput Mapping");
    println!("-----------------------------");
    println!(
        "QNC-like source URI: {}",
        qnc_like.source_identity.public_source_uri
    );
    println!("Source id: {}", qnc_like.source_identity.source_id);
    println!("Source mode: {:?}", qnc_like.source_mode);
    println!(
        "Selected picture representation: {:?}",
        qnc_like.selected_picture_representation
    );
    println!(
        "Authoritative audio representation: {:?}",
        qnc_like.authoritative_audio_representation
    );
    println!(
        "Original media id: {}",
        qnc_like
            .source_identity
            .original_media_id
            .as_deref()
            .unwrap_or("<none>")
    );
    println!(
        "Proxy media id: {}",
        qnc_like
            .source_identity
            .proxy_media_id
            .as_deref()
            .unwrap_or("<none>")
    );
    println!(
        "Private original binding present: {}",
        yes_no(qnc_like.private_binding.original_bound)
    );
    println!(
        "Private proxy binding present: {}",
        yes_no(qnc_like.private_binding.proxy_bound)
    );
    println!(
        "Private path exposed: {}",
        yes_no(qnc_like.private_binding.private_path_exposed)
    );
    println!(
        "Proxy AAC authoritative: {}",
        yes_no(qnc_like.proxy_aac_authoritative)
    );
    println!(
        "Original audio lane count: {}",
        qnc_like.stream_layout.audio_channels.len()
    );
    println!("Audio lane mapping:");
    for lane in &qnc_like.stream_layout.audio_channels {
        println!(
            "  lane {}: source_track={} source_channel={} kind={:?} authoritative={}",
            lane.lane_index + 1,
            lane.source_track_index,
            lane.source_channel_index,
            lane.channel_kind,
            yes_no(lane.authoritative)
        );
    }
    println!(
        "Project audio: channels={} sample_rate={}Hz",
        qnc_like.project_audio.channels, qnc_like.project_audio.sample_rate_hz
    );
    println!(
        "Original/proxy timing compatible: {}",
        qnc_like
            .original_proxy_timing_compatible
            .map(yes_no)
            .unwrap_or("unknown")
    );
    println!("Descriptor revision: {}", qnc_like.descriptor_revision);
    println!("Validation result: {}", result_label(&validation));
    println!("QGS descriptor created: {}", yes_no(descriptor_created));
    println!("QGS input plan created: {}", yes_no(plan_created));
    println!(
        "Assembly construction from mapped descriptor: {}",
        yes_no(assembly_created)
    );
    if let Some(descriptor) = descriptor.as_ref() {
        println!(
            "Mapped descriptor picture/audio: {:?} / {:?}",
            descriptor.selected_picture, descriptor.authoritative_audio
        );
    }
    if let Some(plan) = plan.as_ref() {
        println!(
            "Mapped input plan: source_mode={:?} video_uri={} audio_uri={} lanes={}",
            plan.source_mode,
            plan.video_source.media_uri,
            plan.audio_source.media_uri,
            plan.audio_source.lanes.len()
        );
    }
    println!("Realtime playback claimed: no");
    println!("A/V sync verified: no");
    println!("Device output claimed: no");
    validation?;
    Ok(())
}

fn qgs_transport_engine_parity_report(
    original_path: &Path,
    proxy_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let descriptor = build_qnc_prepared_input_descriptor(original_path, proxy_path)?;
    descriptor.validate()?;
    let plan =
        QgsInputPlan::from_descriptor(&descriptor, default_qgs_input_plan_queue_requirements())?;
    let active_duration = Duration::from_millis(1_000);
    let active_frames = qgs_frames_for_duration(active_duration, plan.video_source.timebase)?;

    let mut engine = QgsTransportEngine::new();
    let handle = engine.load_source(&plan)?;
    engine.preload_source(&handle)?;
    engine.set_active_source(&handle)?;
    let active_range = engine.set_active_range_frames(0, active_frames)?;
    let cue = engine.cue_frame(0)?;
    let play_before_anchor = engine.play().is_err();
    let play_ready_before_anchor = engine.snapshot().play_ready;
    let anchor = engine.prepare_anchor()?;
    let play_ready_after_anchor = engine.evaluate_play_ready();
    let play_result = engine.play();
    let snapshot = engine.snapshot();

    println!("QGS Transport Engine Parity");
    println!("---------------------------");
    println!(
        "Public original URI: {}",
        descriptor.binding.original_media_uri
    );
    println!(
        "Public proxy URI: {}",
        descriptor
            .binding
            .proxy_media_uri
            .as_deref()
            .unwrap_or("<none>")
    );
    println!("Source handle: {}", handle.source_id);
    println!("Source revision: {}", handle.revision.0);
    println!(
        "Source identity exposes private path: {}",
        yes_no(handle.exposes_private_path())
    );
    println!(
        "Active range: [{}..{}) frames, samples [{}..{})",
        active_range.start_frame,
        active_range.end_frame,
        active_range.start_sample,
        active_range.end_sample
    );
    println!("Active duration: {:.3}s", active_duration.as_secs_f64());
    println!("Cue point: frame={} sample={}", cue.frame, cue.sample);
    println!(
        "Prepared anchor: frame={} sample={} revision={}",
        anchor.frame, anchor.sample, anchor.revision.0
    );
    println!("Play before Ready rejected: {}", yes_no(play_before_anchor));
    println!(
        "play_ready before anchor: {}",
        yes_no(play_ready_before_anchor)
    );
    println!(
        "play_ready after anchor: {}",
        yes_no(play_ready_after_anchor)
    );
    println!("Play result: {}", result_label(&play_result));
    println!("Final status: {:?}", snapshot.status);
    println!("No work on Play:");
    println!(
        "  source_open_on_play: {}",
        snapshot.no_work_on_play.source_open_on_play
    );
    println!(
        "  decode_on_play: {}",
        snapshot.no_work_on_play.decode_on_play
    );
    println!(
        "  queue_fill_on_play: {}",
        snapshot.no_work_on_play.queue_fill_on_play
    );
    println!(
        "  preroll_on_play: {}",
        snapshot.no_work_on_play.preroll_on_play
    );
    println!(
        "  anchor_prepare_on_play: {}",
        snapshot.no_work_on_play.anchor_prepare_on_play
    );
    println!("Event count: {}", snapshot.events.len());
    println!("Events:");
    for (index, event) in snapshot.events.iter().enumerate() {
        println!("  {:02}: {}", index, qgs_transport_event_label(event));
    }
    println!("UI involved: no");
    println!("Export/render: no");
    println!("Realtime: no");
    println!("A/V sync: no");
    println!("Device verification: no");

    play_result?;
    Ok(())
}

fn qgs_transport_event_label(event: &QgsTransportEvent) -> String {
    match event {
        QgsTransportEvent::TransportEngineCreated => "TransportEngineCreated".to_string(),
        QgsTransportEvent::SourceLoaded { source_id, revision } => {
            format!("SourceLoaded source={source_id} revision={}", revision.0)
        }
        QgsTransportEvent::SourcePreloaded { source_id, revision } => {
            format!("SourcePreloaded source={source_id} revision={}", revision.0)
        }
        QgsTransportEvent::ActiveSourceChanged {
            source_id,
            revision,
        } => format!(
            "ActiveSourceChanged source={} revision={}",
            source_id.as_deref().unwrap_or("<none>"),
            revision.map(|value| value.0).unwrap_or(0)
        ),
        QgsTransportEvent::ActiveRangeSet {
            start_frame,
            end_frame,
            start_sample,
            end_sample,
        } => format!(
            "ActiveRangeSet frames=[{start_frame}..{end_frame}) samples=[{start_sample}..{end_sample})"
        ),
        QgsTransportEvent::CueCompleted { frame, sample } => {
            format!("CueCompleted frame={frame} sample={sample}")
        }
        QgsTransportEvent::PreparedAnchorReady { frame, revision } => {
            format!("PreparedAnchorReady frame={frame} revision={}", revision.0)
        }
        QgsTransportEvent::PlayReadinessChanged { ready } => {
            format!("PlayReadinessChanged ready={}", yes_no(*ready))
        }
        QgsTransportEvent::TransportPlayRejected { reason } => {
            format!("TransportPlayRejected reason={reason}")
        }
        QgsTransportEvent::TransportStarted { frame } => {
            format!("TransportStarted frame={frame}")
        }
        QgsTransportEvent::TransportPaused => "TransportPaused".to_string(),
        QgsTransportEvent::TransportStopped => "TransportStopped".to_string(),
        QgsTransportEvent::ActiveSourceCleared {
            source_id,
            revision,
        } => format!(
            "ActiveSourceCleared source={source_id} revision={}",
            revision.0
        ),
        QgsTransportEvent::SourceRevisionInvalidated {
            source_id,
            revision,
        } => format!(
            "SourceRevisionInvalidated source={source_id} revision={}",
            revision.0
        ),
        QgsTransportEvent::SourceUnloaded {
            source_id,
            revision,
        } => format!("SourceUnloaded source={source_id} revision={}", revision.0),
        QgsTransportEvent::PreparedStateDiscarded { frames } => {
            format!("PreparedStateDiscarded frames={frames}")
        }
        QgsTransportEvent::TransportValidationFailed { reason } => {
            format!("TransportValidationFailed reason={reason}")
        }
    }
}

fn qgs_frame_clock_parity_report(
    original_path: &Path,
    proxy_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let descriptor = build_qnc_prepared_input_descriptor(original_path, proxy_path)?;
    descriptor.validate()?;
    let plan =
        QgsInputPlan::from_descriptor(&descriptor, default_qgs_input_plan_queue_requirements())?;
    let active_duration = Duration::from_millis(1_000);
    let active_frames = qgs_frames_for_duration(active_duration, plan.video_source.timebase)?;
    let timing = QgsActiveRangeTiming::new(
        plan.video_source.duration_frames,
        plan.video_source.timebase,
        plan.audio_source.sample_rate,
        0,
        active_frames,
    )?;
    let cue = timing.validate_cue(0)?;
    let cue_at_end_rejected = timing.validate_cue(timing.end_frame).is_err();
    let frame0 = timing.frame_audio_sample_range(0)?;
    let frame1 = timing.frame_audio_sample_range(1)?;
    let forward_clock = QgsFrameClock::forward(timing)?;
    let still_clock = QgsFrameClock::still(timing, 0)?;
    let reverse_clock = QgsFrameClock::reverse(timing)?;
    let double_rate_clock = QgsFrameClock::new(
        timing,
        QgsFrameClockMode::Forward,
        QgsFrameClockRate::two(),
        0,
    )?;
    let latest_0 = forward_clock.latest_due_frame(Duration::ZERO)?;
    let latest_60 = forward_clock.latest_due_frame(Duration::from_millis(60))?;
    let latest_1000 = forward_clock.latest_due_frame(active_duration)?;
    let forward_drain = forward_clock.drain_due_frames(None, Duration::from_millis(60), 16)?;
    let forward_next = forward_clock.drain_due_frames(Some(3), Duration::from_millis(120), 16)?;
    let still_drain = still_clock.drain_due_frames(None, Duration::from_millis(500), 16)?;
    let reverse_drain = reverse_clock.drain_due_frames(None, Duration::from_millis(80), 16)?;
    let double_latest = double_rate_clock.latest_due_frame(Duration::from_millis(40))?;

    println!("QGS FrameClock / ActiveRange / Cue Parity");
    println!("-----------------------------------------");
    println!("Public source URI: {}", plan.video_source.media_uri);
    println!(
        "Source frame rate rational: {}/{}",
        plan.video_source.timebase.numerator(),
        plan.video_source.timebase.denominator()
    );
    println!("Audio sample rate: {}Hz", plan.audio_source.sample_rate);
    println!("Active range convention: half-open [start_frame..end_frame)");
    println!(
        "Active range: [{}..{}) frames, samples [{}..{})",
        timing.start_frame, timing.end_frame, timing.start_sample, timing.end_sample
    );
    println!("Active duration: {:.3}s", timing.duration()?.as_secs_f64());
    println!("Cue frame 0 valid: {}", yes_no(cue.valid));
    println!("Cue frame 0 sample: {}", cue.sample);
    println!("Cue at end_frame rejected: {}", yes_no(cue_at_end_rejected));
    println!(
        "Frame 0 sample range: [{}..{}) samples, duration {:.3}ms",
        frame0.start_sample,
        frame0.end_sample,
        frame0.duration.as_secs_f64() * 1000.0
    );
    println!(
        "Frame 1 sample range: [{}..{}) samples, duration {:.3}ms",
        frame1.start_sample,
        frame1.end_sample,
        frame1.duration.as_secs_f64() * 1000.0
    );
    println!(
        "1000 ms sample range: [{}..{}) samples",
        timing.start_sample, timing.end_sample
    );
    println!(
        "latest_due_frame examples: 0ms={} 60ms={} 1000ms={}",
        optional_frame(latest_0),
        optional_frame(latest_60),
        optional_frame(latest_1000)
    );
    println!(
        "drain_due_frames forward 0..60ms: {}",
        frame_list_label(&forward_drain.frames)
    );
    println!(
        "drain_due_frames forward after frame 3 at 120ms: {}",
        frame_list_label(&forward_next.frames)
    );
    println!(
        "drain_due_frames still at 500ms: {}",
        frame_list_label(&still_drain.frames)
    );
    println!(
        "drain_due_frames reverse at 80ms: {}",
        frame_list_label(&reverse_drain.frames)
    );
    println!(
        "Mode summaries: forward_latest_60ms={} still_advances={} reverse_first={} double_rate_latest_40ms={}",
        optional_frame(latest_60),
        yes_no(!still_drain.frames.is_empty()),
        reverse_drain
            .frames
            .first()
            .map(|frame| frame.to_string())
            .unwrap_or_else(|| "<none>".to_string()),
        optional_frame(double_latest)
    );
    println!("Realtime playback: no");
    println!("Playout buffer: no");
    println!("Device output: no");
    println!("A/V sync: no");

    Ok(())
}

fn optional_frame(frame: Option<u64>) -> String {
    frame
        .map(|frame| frame.to_string())
        .unwrap_or_else(|| "<none>".to_string())
}

fn frame_list_label(frames: &[u64]) -> String {
    if frames.is_empty() {
        "[]".to_string()
    } else {
        format!("{frames:?}")
    }
}

fn qgs_playout_buffer_tick_report(
    original_path: &Path,
    proxy_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let descriptor = build_qnc_prepared_input_descriptor(original_path, proxy_path)?;
    descriptor.validate()?;
    let plan =
        QgsInputPlan::from_descriptor(&descriptor, default_qgs_input_plan_queue_requirements())?;
    let active_duration = Duration::from_millis(1_000);
    let active_frames = qgs_frames_for_duration(active_duration, plan.video_source.timebase)?;

    let mut engine = QgsTransportEngine::new();
    let handle = engine.load_source(&plan)?;
    engine.preload_source(&handle)?;
    engine.set_active_source(&handle)?;
    let transport_range = engine.set_active_range_frames(0, active_frames)?;
    engine.cue_frame(0)?;
    engine.prepare_anchor()?;
    let play_ready = engine.evaluate_play_ready();
    let before_play_snapshot = engine.snapshot();

    let active_range = QgsActiveRangeTiming::new(
        handle.duration_frames,
        handle.timebase,
        handle.audio_sample_rate,
        transport_range.start_frame,
        transport_range.end_frame,
    )?;
    let limits = QgsPlayoutBufferLimits::default_transport_window();
    let clock = QgsFrameClock::forward(active_range)?;
    let mut buffer = QgsPlayoutBufferState::new(
        active_range,
        handle.revision,
        plan.source_mode,
        plan.audio_source.lanes.clone(),
        limits,
    )?;
    let carriers = [0_u64, 3, 6];
    let mut results = Vec::new();
    for carrier_frame in carriers {
        let elapsed = handle.timebase.frame_offset(carrier_frame)?;
        results.push(buffer.tick_prepare(
            clock,
            QgsTickPreparationInput {
                carrier_frame,
                elapsed,
                max_due_frames: limits.max_prepared_frames,
            },
        )?);
    }

    println!("QGS Prepared/Playout Buffer Tick Preparation");
    println!("--------------------------------------------");
    println!("Public source URI: {}", handle.source_id);
    println!(
        "Active range: [{}..{}) frames, samples [{}..{})",
        transport_range.start_frame,
        transport_range.end_frame,
        transport_range.start_sample,
        transport_range.end_sample
    );
    println!(
        "Buffer limits: backward_keep_frames={} forward_prepare_frames={} max_prepared_frames={}",
        limits.backward_keep_frames, limits.forward_prepare_frames, limits.max_prepared_frames
    );
    println!("Transport play_ready before ticks: {}", yes_no(play_ready));
    println!("Tick count: {}", results.len());
    println!("Carrier frames: {}", frame_list_label(&carriers));
    for (index, result) in results.iter().enumerate() {
        println!("Tick {}:", index + 1);
        println!("  carrier frame: {}", result.carrier_frame);
        println!(
            "  prepared window: [{}..{})",
            result.window.start_frame, result.window.end_frame
        );
        println!(
            "  due frames drained: {}",
            frame_list_label(&result.due_frames)
        );
        println!(
            "  frames prepared this tick: {}",
            frame_list_label(&result.prepared_frames)
        );
        println!(
            "  frames discarded this tick: {}",
            frame_list_label(&result.discarded_frames)
        );
        println!(
            "  buffered frames after tick: {}",
            frame_list_label(
                &result
                    .slots
                    .iter()
                    .map(|slot| slot.key.frame)
                    .collect::<Vec<_>>()
            )
        );
        for slot in result.slots.iter().take(3) {
            println!(
                "    frame {}: audio [{}..{}) lanes={} audio_status={:?} video_status={:?} video_payload={:?}",
                slot.key.frame,
                slot.audio.sample_range.start_sample,
                slot.audio.sample_range.end_sample,
                slot.audio.lanes.len(),
                slot.audio.status,
                slot.video.status,
                slot.video.payload_kind
            );
        }
        println!("  event count: {}", result.events.len());
        for event in &result.events {
            println!("    {}", qgs_tick_event_label(event));
        }
    }
    println!("No work on Play counters before tick preparation:");
    println!(
        "  source_open_on_play: {}",
        before_play_snapshot.no_work_on_play.source_open_on_play
    );
    println!(
        "  decode_on_play: {}",
        before_play_snapshot.no_work_on_play.decode_on_play
    );
    println!(
        "  queue_fill_on_play: {}",
        before_play_snapshot.no_work_on_play.queue_fill_on_play
    );
    println!(
        "  preroll_on_play: {}",
        before_play_snapshot.no_work_on_play.preroll_on_play
    );
    println!(
        "  anchor_prepare_on_play: {}",
        before_play_snapshot.no_work_on_play.anchor_prepare_on_play
    );
    println!("Proxy AAC used: no");
    println!(
        "Original mono audio lanes preserved: {}",
        plan.audio_source.lanes.len()
    );
    println!("Realtime playback: no");
    println!("Device output: no");
    println!("Frame presented: no");
    println!("A/V sync: no");

    Ok(())
}

fn qgs_tick_event_label(event: &QgsTickPreparationEvent) -> String {
    match event {
        QgsTickPreparationEvent::PlayoutBufferCreated {
            max_prepared_frames,
        } => format!("PlayoutBufferCreated max_prepared_frames={max_prepared_frames}"),
        QgsTickPreparationEvent::TickPreparationStarted { carrier_frame } => {
            format!("TickPreparationStarted carrier_frame={carrier_frame}")
        }
        QgsTickPreparationEvent::DueFramesDrained { frames } => {
            format!("DueFramesDrained frames={}", frame_list_label(frames))
        }
        QgsTickPreparationEvent::FramePrepared { frame } => {
            format!("FramePrepared frame={frame}")
        }
        QgsTickPreparationEvent::AudioRangePrepared {
            frame,
            start_sample,
            end_sample,
            lanes,
        } => format!(
            "AudioRangePrepared frame={frame} samples=[{start_sample}..{end_sample}) lanes={lanes}"
        ),
        QgsTickPreparationEvent::VideoPayloadMarkedReady {
            frame,
            payload_kind,
        } => format!("VideoPayloadMarkedReady frame={frame} payload={payload_kind:?}"),
        QgsTickPreparationEvent::PreparedWindowAdvanced {
            start_frame,
            end_frame,
        } => format!("PreparedWindowAdvanced frames=[{start_frame}..{end_frame})"),
        QgsTickPreparationEvent::OldFrameDiscarded { frame, reason } => {
            format!("OldFrameDiscarded frame={frame} reason={reason:?}")
        }
        QgsTickPreparationEvent::PreparedStateDiscarded { frames, reason } => {
            format!(
                "PreparedStateDiscarded frames={} reason={reason:?}",
                frame_list_label(frames)
            )
        }
        QgsTickPreparationEvent::BufferLimitReached {
            max_prepared_frames,
        } => format!("BufferLimitReached max_prepared_frames={max_prepared_frames}"),
        QgsTickPreparationEvent::TickPreparationCompleted { prepared_frames } => {
            format!("TickPreparationCompleted prepared_frames={prepared_frames}")
        }
        QgsTickPreparationEvent::TickValidationFailed { reason } => {
            format!("TickValidationFailed reason={reason}")
        }
    }
}

fn qgs_runtime_lifecycle_events_report(
    original_path: &Path,
    proxy_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let descriptor = build_qnc_prepared_input_descriptor(original_path, proxy_path)?;
    descriptor.validate()?;
    let plan =
        QgsInputPlan::from_descriptor(&descriptor, default_qgs_input_plan_queue_requirements())?;
    let active_duration = Duration::from_millis(1_000);
    let active_frames = qgs_frames_for_duration(active_duration, plan.video_source.timebase)?;

    let mut engine = QgsTransportEngine::new();
    let handle = engine.load_source(&plan)?;
    engine.preload_source(&handle)?;
    engine.set_active_source(&handle)?;
    let transport_range = engine.set_active_range_frames(0, active_frames)?;
    engine.cue_frame(0)?;
    engine.prepare_anchor()?;
    let initial_play_ready = engine.evaluate_play_ready();
    let ready_snapshot = engine.snapshot();

    let active_range = QgsActiveRangeTiming::new(
        handle.duration_frames,
        handle.timebase,
        handle.audio_sample_rate,
        transport_range.start_frame,
        transport_range.end_frame,
    )?;
    let limits = QgsPlayoutBufferLimits::default_transport_window();
    let mut buffer = QgsPlayoutBufferState::new(
        active_range,
        handle.revision,
        plan.source_mode,
        plan.audio_source.lanes.clone(),
        limits,
    )?;
    let tick_result = buffer.tick_prepare(
        QgsFrameClock::forward(active_range)?,
        QgsTickPreparationInput {
            carrier_frame: 0,
            elapsed: Duration::ZERO,
            max_due_frames: limits.max_prepared_frames,
        },
    )?;
    let buffer_frames_before_unload = buffer.prepared_frame_count();

    let mut event_log = QgsRuntimeEventLog::new();
    for event in &ready_snapshot.events {
        event_log.push_transport_event(event);
    }
    for event in &tick_result.events {
        event_log.push_tick_event(event);
    }

    let previous_transport_event_count = ready_snapshot.events.len();
    let discarded_frames = buffer.discard_all(QgsBufferDiscardReason::SourceUnloaded);
    engine.record_prepared_state_discarded(discarded_frames.len());
    event_log.increment_generation();
    let unload_result = engine.unload_source(&handle);
    let stale_command_rejected = engine.set_active_source(&handle).is_err();
    let final_snapshot = engine.snapshot();
    for event in final_snapshot
        .events
        .iter()
        .skip(previous_transport_event_count)
    {
        event_log.push_transport_event(event);
    }
    for event in buffer.events().iter().skip(tick_result.events.len() + 1) {
        event_log.push_tick_event(event);
    }

    println!("QGS Source Lifecycle / Runtime Event Envelope");
    println!("---------------------------------------------");
    println!("Public source URI: {}", handle.source_id);
    println!(
        "Private path exposed in events: {}",
        yes_no(
            event_log
                .envelopes()
                .iter()
                .any(|event| event.exposes_private_path())
        )
    );
    println!(
        "Initial play_ready before unload: {}",
        yes_no(initial_play_ready)
    );
    println!(
        "Buffer frames before unload: {}",
        buffer_frames_before_unload
    );
    println!("Unload result: unloaded={} was_active={} revision_invalidated={} already_missing_or_invalid={}",
        yes_no(unload_result.unloaded),
        yes_no(unload_result.was_active),
        yes_no(unload_result.revision_invalidated),
        yes_no(unload_result.already_missing_or_invalid)
    );
    println!(
        "Final active source: {}",
        if final_snapshot.active_source.is_some() {
            "present"
        } else {
            "none"
        }
    );
    println!("Final play_ready: {}", yes_no(final_snapshot.play_ready));
    println!(
        "Prepared buffer after unload: {}",
        buffer.prepared_frame_count()
    );
    println!(
        "Prepared frames discarded: {}",
        frame_list_label(&discarded_frames)
    );
    println!("Stale command rejected: {}", yes_no(stale_command_rejected));
    println!("Event count: {}", event_log.envelopes().len());
    println!(
        "Event sequence monotonic: {}",
        yes_no(event_log.sequence_is_monotonic())
    );
    println!(
        "Generation behavior: initial=0 lifecycle={}",
        event_log.generation.0
    );
    println!("Event envelope transcript:");
    for event in event_log.envelopes() {
        println!("  {}", qgs_runtime_event_label(event));
    }
    println!("No work on Play counters:");
    println!(
        "  source_open_on_play: {}",
        final_snapshot.no_work_on_play.source_open_on_play
    );
    println!(
        "  decode_on_play: {}",
        final_snapshot.no_work_on_play.decode_on_play
    );
    println!(
        "  queue_fill_on_play: {}",
        final_snapshot.no_work_on_play.queue_fill_on_play
    );
    println!(
        "  preroll_on_play: {}",
        final_snapshot.no_work_on_play.preroll_on_play
    );
    println!(
        "  anchor_prepare_on_play: {}",
        final_snapshot.no_work_on_play.anchor_prepare_on_play
    );
    println!("Realtime playback: no");
    println!("Device output: no");
    println!("Frame presented: no");
    println!("A/V sync: no");

    Ok(())
}

fn qgs_runtime_event_label(event: &QgsRuntimeEventEnvelope) -> String {
    format!(
        "#{:02} gen={} kind={:?} source={} summary={}",
        event.sequence.0,
        event.generation.0,
        event.kind,
        event
            .source
            .as_ref()
            .map(|source| format!("{}@{}", source.source_id, source.revision.0))
            .unwrap_or_else(|| "<none>".to_string()),
        event.payload.summary
    )
}

fn qgs_qnc_event_projection_report(
    original_path: &Path,
    proxy_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let descriptor = build_qnc_prepared_input_descriptor(original_path, proxy_path)?;
    descriptor.validate()?;
    let plan =
        QgsInputPlan::from_descriptor(&descriptor, default_qgs_input_plan_queue_requirements())?;
    let active_duration = Duration::from_millis(1_000);
    let active_frames = qgs_frames_for_duration(active_duration, plan.video_source.timebase)?;

    let mut engine = QgsTransportEngine::new();
    let handle = engine.load_source(&plan)?;
    let command_examples = qgs_qnc_command_examples(&handle.source_id, active_frames);
    engine.preload_source(&handle)?;
    engine.set_active_source(&handle)?;
    let transport_range = engine.set_active_range_frames(0, active_frames)?;
    engine.cue_frame(0)?;
    engine.prepare_anchor()?;
    engine.evaluate_play_ready();
    let ready_snapshot = engine.snapshot();

    let active_range = QgsActiveRangeTiming::new(
        handle.duration_frames,
        handle.timebase,
        handle.audio_sample_rate,
        transport_range.start_frame,
        transport_range.end_frame,
    )?;
    let limits = QgsPlayoutBufferLimits::default_transport_window();
    let mut buffer = QgsPlayoutBufferState::new(
        active_range,
        handle.revision,
        plan.source_mode,
        plan.audio_source.lanes.clone(),
        limits,
    )?;
    let tick_result = buffer.tick_prepare(
        QgsFrameClock::forward(active_range)?,
        QgsTickPreparationInput {
            carrier_frame: 0,
            elapsed: Duration::ZERO,
            max_due_frames: limits.max_prepared_frames,
        },
    )?;
    let prepared_slots = buffer.slots();

    let mut event_log = QgsRuntimeEventLog::new();
    for event in &ready_snapshot.events {
        event_log.push_transport_event(event);
    }
    for event in &tick_result.events {
        event_log.push_tick_event(event);
    }
    let ready_passive_view =
        QgsQncPassiveView::from_parts(event_log.generation, &ready_snapshot, &prepared_slots, 0);
    let timeline_projection =
        QgsQncTimelineProjection::from_parts(&ready_snapshot, &prepared_slots);
    let monitor_projection = QgsQncMonitorProjection::from_slots(&prepared_slots);

    let previous_transport_event_count = ready_snapshot.events.len();
    let discarded_frames = buffer.discard_all(QgsBufferDiscardReason::SourceUnloaded);
    engine.record_prepared_state_discarded(discarded_frames.len());
    event_log.increment_generation();
    engine.unload_source(&handle);
    let stale_command_rejected = engine.set_active_source(&handle).is_err();
    let final_snapshot = engine.snapshot();
    for event in final_snapshot
        .events
        .iter()
        .skip(previous_transport_event_count)
    {
        event_log.push_transport_event(event);
    }
    for event in buffer.events().iter().skip(tick_result.events.len() + 1) {
        event_log.push_tick_event(event);
    }

    let projected_events = project_qnc_events(event_log.envelopes());
    let final_passive_view = QgsQncPassiveView::from_parts(
        event_log.generation,
        &final_snapshot,
        &buffer.slots(),
        discarded_frames.len(),
    );
    let private_path_exposed = command_examples
        .iter()
        .any(QgsQncCommandEnvelope::exposes_private_path)
        || projected_events
            .iter()
            .any(QgsQncEventEnvelope::exposes_private_path)
        || ready_passive_view.private_path_exposed
        || final_passive_view.private_path_exposed;

    println!("QGS QNC-Compatible Command/Event Projection");
    println!("-------------------------------------------");
    println!(
        "Public original URI: {}",
        descriptor.binding.original_media_uri
    );
    println!(
        "Public proxy URI: {}",
        descriptor
            .binding
            .proxy_media_uri
            .as_deref()
            .unwrap_or("<none>")
    );
    println!("Private path exposed: {}", yes_no(private_path_exposed));
    println!("Projection only: yes");
    println!("IPC implemented: no");
    println!("QNC UI implemented: no");
    println!("Command envelope examples:");
    for command in &command_examples {
        println!("  {}", qgs_qnc_command_label(command));
    }
    println!("Projected event count: {}", projected_events.len());
    println!(
        "Event sequence monotonic: {}",
        yes_no(qgs_qnc_projected_sequence_is_monotonic(&projected_events))
    );
    println!("Projected event transcript:");
    for event in &projected_events {
        println!("  {}", qgs_qnc_projected_event_label(event));
    }
    println!("Passive view before unload:");
    print_qgs_qnc_passive_view(&ready_passive_view);
    println!("Passive view after unload:");
    print_qgs_qnc_passive_view(&final_passive_view);
    println!("Timeline projection:");
    println!(
        "  active_range: {}",
        ready_passive_view
            .transport
            .active_range
            .map(qgs_transport_range_label)
            .unwrap_or_else(|| "<none>".to_string())
    );
    println!(
        "  prepared_frame_window: {}",
        qgs_optional_frame_window_label(
            timeline_projection.prepared_start_frame,
            timeline_projection.prepared_end_frame_exclusive
        )
    );
    println!(
        "  cue_frame: {}",
        qgs_optional_u64_label(timeline_projection.cue_frame)
    );
    println!(
        "  carrier_frame: {}",
        qgs_optional_u64_label(timeline_projection.carrier_frame)
    );
    println!(
        "  frame_rate: {}",
        timeline_projection
            .frame_rate
            .map(qgs_rate_label)
            .unwrap_or_else(|| "<none>".to_string())
    );
    println!(
        "  audio_sample_rate: {}",
        timeline_projection
            .audio_sample_rate
            .map(|rate| format!("{rate} Hz"))
            .unwrap_or_else(|| "<none>".to_string())
    );
    println!(
        "  presented_frame_claimed: {}",
        yes_no(timeline_projection.presented_frame_claimed)
    );
    println!("Monitor projection:");
    println!(
        "  prepared_descriptor_present: {}",
        yes_no(monitor_projection.prepared_descriptor_present)
    );
    println!(
        "  source_frame: {}",
        qgs_optional_u64_label(monitor_projection.source_frame)
    );
    println!("  payload_status: {:?}", monitor_projection.payload_status);
    println!("  presented: {}", yes_no(monitor_projection.presented));
    println!(
        "  real_display_evidence: {}",
        monitor_projection.real_display_evidence.unwrap_or("none")
    );
    println!("Evidence summary:");
    println!("  prepared: {:?}", ready_passive_view.evidence.prepared);
    println!(
        "  submitted_to_device: {:?}",
        ready_passive_view.evidence.submitted_to_device
    );
    println!("  presented: {:?}", ready_passive_view.evidence.presented);
    println!("  verified: {:?}", ready_passive_view.evidence.verified);
    println!(
        "  stale command rejected: {}",
        yes_no(stale_command_rejected)
    );
    println!("  realtime: no");
    println!("  A/V sync: no");
    println!("  FramePresented: no");
    println!("  AudioDeviceVerified: no");

    Ok(())
}

fn qgs_qnc_command_examples(source_uri: &str, active_frames: u64) -> Vec<QgsQncCommandEnvelope> {
    vec![
        QgsQncCommandEnvelope::new(
            0,
            Some(qgs_media_runtime::QgsRuntimeEventGeneration(0)),
            Some(source_uri.to_string()),
            QgsQncPlayerCommand::LoadPreparedInput,
            "load prepared input by public qnc uri",
        ),
        QgsQncCommandEnvelope::new(
            1,
            Some(qgs_media_runtime::QgsRuntimeEventGeneration(0)),
            Some(source_uri.to_string()),
            QgsQncPlayerCommand::PreloadSource,
            "preload source handle",
        ),
        QgsQncCommandEnvelope::new(
            2,
            Some(qgs_media_runtime::QgsRuntimeEventGeneration(0)),
            Some(source_uri.to_string()),
            QgsQncPlayerCommand::SetActiveSource,
            "set active source",
        ),
        QgsQncCommandEnvelope::new(
            3,
            Some(qgs_media_runtime::QgsRuntimeEventGeneration(0)),
            Some(source_uri.to_string()),
            QgsQncPlayerCommand::SetActiveRange {
                start_frame: 0,
                end_frame: active_frames,
            },
            format!("set active range frames=[0..{active_frames})"),
        ),
        QgsQncCommandEnvelope::new(
            4,
            Some(qgs_media_runtime::QgsRuntimeEventGeneration(0)),
            Some(source_uri.to_string()),
            QgsQncPlayerCommand::Cue { frame: 0 },
            "cue frame 0",
        ),
        QgsQncCommandEnvelope::new(
            5,
            Some(qgs_media_runtime::QgsRuntimeEventGeneration(0)),
            Some(source_uri.to_string()),
            QgsQncPlayerCommand::PrepareAnchor,
            "prepare anchor before play",
        ),
        QgsQncCommandEnvelope::new(
            6,
            Some(qgs_media_runtime::QgsRuntimeEventGeneration(0)),
            Some(source_uri.to_string()),
            QgsQncPlayerCommand::TickPrepare { carrier_frame: 0 },
            "prepare bounded buffer around carrier frame",
        ),
        QgsQncCommandEnvelope::new(
            7,
            Some(qgs_media_runtime::QgsRuntimeEventGeneration(1)),
            Some(source_uri.to_string()),
            QgsQncPlayerCommand::UnloadSource,
            "unload source and invalidate revision",
        ),
    ]
}

fn qgs_session_command_boundary_report(
    original_path: &Path,
    proxy_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let descriptor = build_qnc_prepared_input_descriptor(original_path, proxy_path)?;
    descriptor.validate()?;
    let plan =
        QgsInputPlan::from_descriptor(&descriptor, default_qgs_input_plan_queue_requirements())?;
    let active_duration = Duration::from_millis(1_000);
    let active_frames = qgs_frames_for_duration(active_duration, plan.video_source.timebase)?;
    let source_uri = plan.video_source.media_uri.clone();
    let mut session = QgsQncSessionCommandExecutor::new(plan)?;
    let mut commands = qgs_session_command_sequence(&source_uri, active_frames);
    commands.push(QgsQncCommandEnvelope::new(
        10,
        Some(qgs_media_runtime::QgsRuntimeEventGeneration(0)),
        Some(source_uri.clone()),
        QgsQncPlayerCommand::Stop,
        "wrong-generation stop after source close",
    ));

    let mut accepted_count = 0usize;
    let mut rejected_count = 0usize;
    let mut projected_event_count = 0usize;
    let mut private_path_exposed = false;
    let mut results = Vec::new();
    for command in &commands {
        private_path_exposed |= command.exposes_private_path();
        let result = session.execute(command);
        accepted_count += usize::from(result.outcome.accepted);
        rejected_count += usize::from(!result.outcome.accepted);
        projected_event_count += result.projected_events.len();
        private_path_exposed |= result
            .projected_events
            .iter()
            .any(QgsQncEventEnvelope::exposes_private_path)
            || result.passive_view.private_path_exposed;
        results.push((command.clone(), result));
    }
    let final_view = session.passive_view();
    private_path_exposed |= final_view.private_path_exposed;
    let wrong_generation_rejected = results.last().is_some_and(|(_, result)| {
        !result.outcome.accepted && result.outcome.reason == Some("runtime generation mismatch")
    });

    println!("QGS Session Command Execution Boundary");
    println!("--------------------------------------");
    println!(
        "Public original URI: {}",
        descriptor.binding.original_media_uri
    );
    println!(
        "Public proxy URI: {}",
        descriptor
            .binding
            .proxy_media_uri
            .as_deref()
            .unwrap_or("<none>")
    );
    println!("Private path exposed: {}", yes_no(private_path_exposed));
    println!("IPC implemented: no");
    println!("QNC UI implemented: no");
    println!("Command count: {}", commands.len());
    println!("Accepted count: {accepted_count}");
    println!("Rejected count: {rejected_count}");
    println!(
        "Generation behavior: initial=0 final={}",
        session.current_generation().0
    );
    println!(
        "Wrong-generation command rejected: {}",
        yes_no(wrong_generation_rejected)
    );
    println!("Projected event count: {projected_event_count}");
    println!("Command transcript:");
    for (command, result) in &results {
        println!(
            "  id={} command={:?} expected_gen={} accepted={} reason={} events={} view_state={:?}",
            command.command_id,
            command.command,
            command
                .expected_generation
                .map(|generation| generation.0.to_string())
                .unwrap_or_else(|| "<none>".to_string()),
            yes_no(result.outcome.accepted),
            result.outcome.reason.unwrap_or("none"),
            result.projected_events.len(),
            result.passive_view.transport.state
        );
    }
    println!("Final passive view:");
    print_qgs_qnc_passive_view(&final_view);
    println!("Evidence summary:");
    println!("  prepared: {:?}", final_view.evidence.prepared);
    println!(
        "  submitted_to_device: {:?}",
        final_view.evidence.submitted_to_device
    );
    println!("  presented: {:?}", final_view.evidence.presented);
    println!("  verified: {:?}", final_view.evidence.verified);
    println!("  realtime: no");
    println!("  A/V sync: no");
    println!("  device output: no");
    println!("  FramePresented: no");
    println!("  AudioDeviceVerified: no");

    Ok(())
}

fn qgs_session_command_sequence(
    source_uri: &str,
    active_frames: u64,
) -> Vec<QgsQncCommandEnvelope> {
    vec![
        QgsQncCommandEnvelope::new(
            0,
            Some(qgs_media_runtime::QgsRuntimeEventGeneration(0)),
            Some(source_uri.to_string()),
            QgsQncPlayerCommand::LoadPreparedInput,
            "load prepared input",
        ),
        QgsQncCommandEnvelope::new(
            1,
            Some(qgs_media_runtime::QgsRuntimeEventGeneration(0)),
            Some(source_uri.to_string()),
            QgsQncPlayerCommand::PreloadSource,
            "preload source",
        ),
        QgsQncCommandEnvelope::new(
            2,
            Some(qgs_media_runtime::QgsRuntimeEventGeneration(0)),
            Some(source_uri.to_string()),
            QgsQncPlayerCommand::SetActiveSource,
            "set active source",
        ),
        QgsQncCommandEnvelope::new(
            3,
            Some(qgs_media_runtime::QgsRuntimeEventGeneration(0)),
            Some(source_uri.to_string()),
            QgsQncPlayerCommand::SetActiveRange {
                start_frame: 0,
                end_frame: active_frames,
            },
            format!("set active range [0..{active_frames})"),
        ),
        QgsQncCommandEnvelope::new(
            4,
            Some(qgs_media_runtime::QgsRuntimeEventGeneration(0)),
            Some(source_uri.to_string()),
            QgsQncPlayerCommand::Cue { frame: 0 },
            "cue frame 0",
        ),
        QgsQncCommandEnvelope::new(
            5,
            Some(qgs_media_runtime::QgsRuntimeEventGeneration(0)),
            Some(source_uri.to_string()),
            QgsQncPlayerCommand::PrepareAnchor,
            "prepare anchor",
        ),
        QgsQncCommandEnvelope::new(
            6,
            Some(qgs_media_runtime::QgsRuntimeEventGeneration(0)),
            Some(source_uri.to_string()),
            QgsQncPlayerCommand::Play,
            "play after ready",
        ),
        QgsQncCommandEnvelope::new(
            7,
            Some(qgs_media_runtime::QgsRuntimeEventGeneration(0)),
            Some(source_uri.to_string()),
            QgsQncPlayerCommand::TickPrepare { carrier_frame: 0 },
            "explicit tick preparation",
        ),
        QgsQncCommandEnvelope::new(
            8,
            Some(qgs_media_runtime::QgsRuntimeEventGeneration(0)),
            Some(source_uri.to_string()),
            QgsQncPlayerCommand::Pause,
            "pause transport",
        ),
        QgsQncCommandEnvelope::new(
            9,
            Some(qgs_media_runtime::QgsRuntimeEventGeneration(0)),
            Some(source_uri.to_string()),
            QgsQncPlayerCommand::CloseActiveSource,
            "close active source",
        ),
    ]
}

fn qgs_session_runtime_control_report(
    original_path: &Path,
    proxy_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let descriptor = build_qnc_prepared_input_descriptor(original_path, proxy_path)?;
    descriptor.validate()?;
    let plan =
        QgsInputPlan::from_descriptor(&descriptor, default_qgs_input_plan_queue_requirements())?;
    let active_duration = Duration::from_millis(1_000);
    let active_frames = qgs_frames_for_duration(active_duration, plan.video_source.timebase)?;
    let source_uri = plan.video_source.media_uri.clone();
    let config = QgsSessionRuntimeConfig::default();
    let queue_limit = config.queue_limits.max_queued_commands;
    let mut runtime = QgsSessionRuntime::new(plan.clone(), config)?;
    let commands = qgs_session_command_sequence(&source_uri, active_frames);

    let mut commands_enqueued = 0usize;
    for command in &commands {
        runtime.enqueue(command.clone()).map_err(|err| {
            format!(
                "session runtime scenario enqueue failed for id {}: {err:?}",
                command.command_id
            )
        })?;
        commands_enqueued = commands_enqueued.saturating_add(1);
    }
    let duplicate_rejected = runtime.enqueue(qgs_test_public_command(
        commands[0].command_id,
        0,
        &source_uri,
        QgsQncPlayerCommand::Stop,
        "duplicate command id probe",
    )) == Err(QgsSessionQueueRejection::DuplicateCommandId);
    let queue_full_rejected = qgs_session_runtime_queue_full_probe(&plan, &source_uri)?;

    let mut command_results = Vec::new();
    let mut projected_events = Vec::new();
    while let Some(result) = runtime.execute_next() {
        projected_events.extend(result.projected_events.clone());
        command_results.push(result);
    }

    let stale_command = qgs_test_public_command(
        10_000,
        0,
        &source_uri,
        QgsQncPlayerCommand::Stop,
        "stale generation probe",
    );
    runtime
        .enqueue(stale_command)
        .map_err(|err| format!("stale generation probe enqueue failed: {err:?}"))?;
    let stale_result = runtime
        .execute_next()
        .ok_or("stale generation command was not queued")?;
    let stale_generation_rejected = !stale_result
        .outcome
        .as_ref()
        .is_some_and(|outcome| outcome.accepted)
        && stale_result
            .outcome
            .as_ref()
            .and_then(|outcome| outcome.reason)
            == Some("runtime generation mismatch");
    projected_events.extend(stale_result.projected_events.clone());
    command_results.push(stale_result);

    let snapshot = runtime.snapshot();
    let transcript = runtime.transcript();
    let accepted_count = command_results
        .iter()
        .filter(|result| {
            result
                .outcome
                .as_ref()
                .is_some_and(|outcome| outcome.accepted)
        })
        .count();
    let rejected_execution_count = command_results
        .iter()
        .filter(|result| {
            result
                .outcome
                .as_ref()
                .is_some_and(|outcome| !outcome.accepted)
        })
        .count();
    let rejected_count = rejected_execution_count
        + usize::from(duplicate_rejected)
        + usize::from(queue_full_rejected);
    let private_path_exposed = snapshot.private_path_exposed
        || transcript.exposes_private_path()
        || projected_events
            .iter()
            .any(QgsQncEventEnvelope::exposes_private_path);
    let event_sequence_monotonic = qgs_qnc_projected_sequence_is_monotonic(&projected_events);

    println!("QGS Session Runtime Control Surface");
    println!("-----------------------------------");
    println!(
        "Public original URI: {}",
        descriptor.binding.original_media_uri
    );
    println!(
        "Public proxy URI: {}",
        descriptor
            .binding
            .proxy_media_uri
            .as_deref()
            .unwrap_or("<none>")
    );
    println!("Private path exposed: {}", yes_no(private_path_exposed));
    println!("Queue limit: {queue_limit}");
    println!("Commands enqueued: {commands_enqueued}");
    println!("Commands executed: {}", command_results.len());
    println!("Duplicate rejected: {}", yes_no(duplicate_rejected));
    println!("Queue-full rejected: {}", yes_no(queue_full_rejected));
    println!(
        "Stale generation rejected: {}",
        yes_no(stale_generation_rejected)
    );
    println!("Accepted count: {accepted_count}");
    println!("Rejected count: {rejected_count}");
    println!("Final runtime generation: {}", snapshot.generation.0);
    println!("Projected event count: {}", projected_events.len());
    println!(
        "Event sequence monotonic: {}",
        yes_no(event_sequence_monotonic)
    );
    println!("Command transcript:");
    for entry in transcript.entries() {
        println!(
            "  id={} command={:?} status={:?} gen={}->{} events={} view_state={:?} reason={}",
            entry.command_id.0,
            entry.command,
            entry.status,
            entry.generation_before.0,
            entry.generation_after.0,
            entry.projected_event_count,
            entry.passive_view.transport.state,
            entry.rejection.unwrap_or("none")
        );
    }
    println!("Final passive view:");
    print_qgs_qnc_passive_view(&snapshot.passive_view);
    println!("Runtime surface non-claims:");
    println!("  realtime playback: no");
    println!("  A/V sync: no");
    println!("  device output: no");
    println!("  FramePresented: no");
    println!("  AudioDeviceVerified: no");
    println!("  RealtimeVerified: no");

    Ok(())
}

fn qgs_session_runtime_queue_full_probe(
    plan: &QgsInputPlan,
    source_uri: &str,
) -> Result<bool, Box<dyn std::error::Error>> {
    let mut runtime = QgsSessionRuntime::new(
        plan.clone(),
        QgsSessionRuntimeConfig {
            queue_limits: QgsSessionCommandQueueLimits {
                max_queued_commands: 1,
            },
            duplicate_policy: QgsSessionDuplicateCommandPolicy::RejectStrict,
        },
    )?;
    runtime
        .enqueue(qgs_test_public_command(
            1,
            0,
            source_uri,
            QgsQncPlayerCommand::LoadPreparedInput,
            "queue full probe first command",
        ))
        .map_err(|err| format!("queue full probe first enqueue failed: {err:?}"))?;
    Ok(runtime.enqueue(qgs_test_public_command(
        2,
        0,
        source_uri,
        QgsQncPlayerCommand::PreloadSource,
        "queue full probe second command",
    )) == Err(QgsSessionQueueRejection::QueueFull))
}

fn qgs_test_public_command(
    command_id: u64,
    generation: u64,
    source_uri: &str,
    command: QgsQncPlayerCommand,
    summary: impl Into<String>,
) -> QgsQncCommandEnvelope {
    QgsQncCommandEnvelope::new(
        command_id,
        Some(qgs_media_runtime::QgsRuntimeEventGeneration(generation)),
        Some(source_uri.to_string()),
        command,
        summary,
    )
}

fn qgs_presenter_monitor_boundary_report(
    original_path: &Path,
    proxy_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let descriptor = build_qnc_prepared_input_descriptor(original_path, proxy_path)?;
    descriptor.validate()?;
    let input_plan =
        QgsInputPlan::from_descriptor(&descriptor, default_qgs_input_plan_queue_requirements())?;
    let active_duration = Duration::from_millis(1_000);
    let active_frames = qgs_frames_for_duration(active_duration, input_plan.video_source.timebase)?;
    let source_uri = input_plan.video_source.media_uri.clone();

    let mut runtime = QgsSessionRuntime::new(input_plan, QgsSessionRuntimeConfig::default())?;
    let setup_commands = qgs_session_command_sequence(&source_uri, active_frames);
    for command in setup_commands.iter().take(8) {
        runtime.enqueue(command.clone()).map_err(|err| {
            format!(
                "presenter monitor session setup enqueue failed for id {}: {err:?}",
                command.command_id
            )
        })?;
    }
    let setup_report = runtime.drain_queue();
    let setup_snapshot = runtime.snapshot();

    let proxy = Mp4Source::open(proxy_path)?;
    let proxy_video = proxy
        .video
        .as_ref()
        .ok_or("proxy has no H.264 video track")?;
    let proxy_h264 = classify_video_track(proxy_video)?;
    let selected = selected_journalist_preview_frames(proxy_video)?;
    let selected_frame = selected
        .first()
        .ok_or("proxy has no selected journalist preview frame")?;
    let frame_duration = ProxyPlaybackProfile::Journalist50iPreview
        .presentation_rate(RationalRate::new(
            u64::from(proxy_video.frame_rate.numerator),
            u64::from(proxy_video.frame_rate.denominator),
        )?)?
        .frame_duration()?;
    let video_slot = BroadcastPreparedVideoSlot {
        slot_index: 0,
        source_mode: BroadcastVideoSourceMode::ProxyPreview,
        video_source_role: BroadcastMediaSourceRole::ProxyPreviewVideo,
        source_frame_index: Some(selected_frame.source_presentation_index),
        selected_preview_frame_index: Some(selected_frame.preview_index),
        presentation_time: selected_frame.start_time,
        duration: frame_duration,
        status: BroadcastPreparedVideoSlotStatus::Prepared,
    };
    let proxy_payload_proof = bind_proxy_preview_video_payloads(
        proxy_video,
        &proxy_h264,
        std::slice::from_ref(&video_slot),
    )?;
    let video_binding = proxy_payload_proof
        .bindings
        .first()
        .cloned()
        .ok_or("proxy payload proof did not produce a binding")?;
    let presentation_binding = BroadcastPresentationPayloadBinding {
        presentation_slot_index: 0,
        video_binding_index: video_binding.video_slot_index,
        audio_binding_index: 0,
        presentation_time: selected_frame.start_time,
        duration: frame_duration,
        ready: true,
        readiness: BroadcastPresentationPayloadReadiness::PayloadReady,
    };
    let payload = video_binding
        .payload
        .as_ref()
        .ok_or("presenter boundary missing proxy video payload")?;
    let mut presenter = BroadcastTestVideoPresenter::new(BroadcastTestVideoPresenterConfig {
        accepted_kind: BroadcastVideoPayloadKind::ProcessedGpuFrame,
        accepted_format: BroadcastVideoPayloadFormat::RgbaU16,
        visible_width: proxy_h264.width,
        visible_height: proxy_h264.height,
        coded_width: proxy_h264.coded_width,
        coded_height: proxy_h264.coded_height,
    })?;
    let presenter_result = submit_qgs_presenter_payload_to_test_boundary(
        descriptor
            .binding
            .proxy_media_uri
            .as_deref()
            .unwrap_or(&source_uri),
        &presentation_binding,
        &video_binding,
        &mut presenter,
    )?;
    let monitor_update = project_qgs_monitor_update_from_presenter_result(&presenter_result);
    let monitor_projection = QgsQncMonitorProjection::from_presenter_update(&monitor_update);
    let private_path_exposed = monitor_projection.private_path_exposed
        || setup_snapshot.private_path_exposed
        || setup_report.transcript.exposes_private_path();
    let presenter_evidence_kind = monitor_update
        .presenter_evidence_kind
        .map(|kind| format!("{kind:?}"))
        .unwrap_or_else(|| "none".to_string());
    let frame_presented_claim = if monitor_update.frame_presented_test_boundary {
        "test-boundary evidence only; no real display claim"
    } else {
        "no"
    };

    println!("QGS Presenter / Monitor Boundary");
    println!("--------------------------------");
    println!(
        "Public source URI: {}",
        descriptor
            .binding
            .proxy_media_uri
            .as_deref()
            .unwrap_or(&source_uri)
    );
    println!(
        "Selected frame: {}",
        selected_frame.source_presentation_index
    );
    println!("Payload kind: {:?}", payload.kind);
    println!("Payload format: {:?}", payload.format);
    println!("Payload backend: {:?}", payload.backend_path);
    println!(
        "Payload dimensions: visible={}x{} coded={}x{}",
        payload.visible_width, payload.visible_height, payload.coded_width, payload.coded_height
    );
    println!("Presenter kind: TestPresenter");
    println!(
        "Submitted to presenter: {}",
        yes_no(monitor_update.submitted_to_presenter)
    );
    println!(
        "Test presenter accepted: {}",
        yes_no(presenter.accepted_count() == 1 && presenter.rejected_count() == 0)
    );
    println!("Presenter evidence kind: {presenter_evidence_kind}");
    println!("Evidence level: {:?}", presenter_result.evidence_level);
    println!("FramePresented claim: {frame_presented_claim}");
    println!("Monitor projection:");
    println!(
        "  prepared descriptor present: {}",
        yes_no(monitor_projection.prepared_descriptor_present)
    );
    println!(
        "  submitted to presenter: {}",
        yes_no(monitor_projection.submitted_to_presenter)
    );
    println!(
        "  source frame: {}",
        qgs_optional_u64_label(monitor_projection.source_frame)
    );
    println!("  payload status: {:?}", monitor_projection.payload_status);
    println!(
        "  test-boundary presented: {}",
        yes_no(monitor_projection.frame_presented_test_boundary)
    );
    println!(
        "  presented as real display: {}",
        yes_no(monitor_projection.frame_presented_real_backend)
    );
    println!(
        "  real display evidence: {}",
        if monitor_projection.real_display_evidence_present {
            "present"
        } else {
            "none"
        }
    );
    println!(
        "  visual verified: {}",
        yes_no(monitor_projection.visual_verified)
    );
    println!(
        "Session setup: commands_executed={} accepted={} rejected={} final_state={:?}",
        setup_report.executed,
        setup_report.accepted,
        setup_report.rejected,
        setup_snapshot.passive_view.transport.state
    );
    println!(
        "Presenter accepted={} rejected={}",
        presenter.accepted_count(),
        presenter.rejected_count()
    );
    println!("Realtime playback: no");
    println!("A/V sync: no");
    println!("Device verified: no");
    println!("FramePresented real display claim: no");
    println!(
        "FramePresented test boundary evidence: {}",
        yes_no(monitor_projection.frame_presented_test_boundary)
    );
    println!("VisualVerified: no");
    println!("Private path exposed: {}", yes_no(private_path_exposed));

    Ok(())
}

fn qgs_file_presenter_diagnostic_report(
    original_path: &Path,
    proxy_path: &Path,
    output_dir: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let descriptor = build_qnc_prepared_input_descriptor(original_path, proxy_path)?;
    descriptor.validate()?;
    let input_plan =
        QgsInputPlan::from_descriptor(&descriptor, default_qgs_input_plan_queue_requirements())?;
    let active_duration = Duration::from_millis(1_000);
    let active_frames = qgs_frames_for_duration(active_duration, input_plan.video_source.timebase)?;
    let source_uri = input_plan.video_source.media_uri.clone();
    let public_source_uri = descriptor
        .binding
        .proxy_media_uri
        .as_deref()
        .unwrap_or(&source_uri)
        .to_string();

    let mut runtime = QgsSessionRuntime::new(input_plan, QgsSessionRuntimeConfig::default())?;
    let setup_commands = qgs_session_command_sequence(&source_uri, active_frames);
    for command in setup_commands.iter().take(8) {
        runtime.enqueue(command.clone()).map_err(|err| {
            format!(
                "file presenter diagnostic setup enqueue failed for id {}: {err:?}",
                command.command_id
            )
        })?;
    }
    let setup_report = runtime.drain_queue();
    let setup_snapshot = runtime.snapshot();

    let proxy = Mp4Source::open(proxy_path)?;
    let proxy_video = proxy
        .video
        .as_ref()
        .ok_or("proxy has no H.264 video track")?;
    let proxy_h264 = classify_video_track(proxy_video)?;
    let selected = selected_journalist_preview_frames(proxy_video)?;
    let selected_frame = selected
        .first()
        .ok_or("proxy has no selected journalist preview frame")?;
    let frame_duration = ProxyPlaybackProfile::Journalist50iPreview
        .presentation_rate(RationalRate::new(
            u64::from(proxy_video.frame_rate.numerator),
            u64::from(proxy_video.frame_rate.denominator),
        )?)?
        .frame_duration()?;
    let video_slot = BroadcastPreparedVideoSlot {
        slot_index: 0,
        source_mode: BroadcastVideoSourceMode::ProxyPreview,
        video_source_role: BroadcastMediaSourceRole::ProxyPreviewVideo,
        source_frame_index: Some(selected_frame.source_presentation_index),
        selected_preview_frame_index: Some(selected_frame.preview_index),
        presentation_time: selected_frame.start_time,
        duration: frame_duration,
        status: BroadcastPreparedVideoSlotStatus::Prepared,
    };
    let proxy_payload_proof = bind_proxy_preview_video_payloads(
        proxy_video,
        &proxy_h264,
        std::slice::from_ref(&video_slot),
    )?;
    let video_binding = proxy_payload_proof
        .bindings
        .first()
        .cloned()
        .ok_or("proxy payload proof did not produce a binding")?;
    let payload = video_binding
        .payload
        .as_ref()
        .ok_or("file presenter diagnostic missing proxy video payload")?;

    let output_dir = output_dir.unwrap_or_else(|| Path::new("target/qgs-visual-diagnostics"));
    std::fs::create_dir_all(output_dir)?;
    let artifact_file_name = file_presenter_manifest_file_name(
        &public_source_uri,
        selected_frame.source_presentation_index,
    );
    let artifact_path = output_dir.join(&artifact_file_name);
    let public_artifact_path = format!("qgs-visual-diagnostics/{artifact_file_name}");

    let file_result = submit_qgs_file_presenter_manifest(
        public_source_uri.clone(),
        &video_binding,
        public_artifact_path,
    )?;
    let monitor_update = project_qgs_monitor_update_from_file_presenter_result(&file_result);
    let monitor_projection = QgsQncMonitorProjection::from_presenter_update(&monitor_update);
    let manifest_json = qgs_file_presenter_manifest_json(
        &file_result,
        &monitor_projection,
        setup_report.executed,
        setup_report.accepted,
        setup_report.rejected,
        &format!("{:?}", setup_snapshot.passive_view.transport.state),
    );
    File::create(&artifact_path)?.write_all(manifest_json.as_bytes())?;

    let private_path_exposed = monitor_projection.private_path_exposed
        || setup_snapshot.private_path_exposed
        || setup_report.transcript.exposes_private_path()
        || manifest_json_contains_private_path(&manifest_json);
    let artifact_kind = format!("{:?}", file_result.artifact.kind);
    let evidence_level = format!("{:?}", file_result.evidence.evidence_level);
    let reason = file_result.artifact.reason.unwrap_or("n/a");

    println!("QGS File Presenter Visual Diagnostic");
    println!("------------------------------------");
    println!("Audio source: original MXF");
    println!("Video source: proxy MP4");
    println!("Proxy AAC: not used");
    println!("Presenter kind: FilePresenter");
    println!("Preview profile: journalist-50i-preview");
    println!("Public source URI: {public_source_uri}");
    println!(
        "Selected frame: {}",
        selected_frame.source_presentation_index
    );
    println!("Selected preview frame: {}", selected_frame.preview_index);
    println!("Payload kind: {:?}", payload.kind);
    println!("Payload format: {:?}", payload.format);
    println!("Payload backend: {:?}", payload.backend_path);
    println!(
        "Payload dimensions: visible={}x{} coded={}x{}",
        payload.visible_width, payload.visible_height, payload.coded_width, payload.coded_height
    );
    println!("Artifact kind: {artifact_kind}");
    println!("Artifact path: {}", artifact_path.display());
    println!(
        "Artifact public path: {}",
        file_result.artifact.public_artifact_path
    );
    println!(
        "Image artifact available: {}",
        yes_no(file_result.artifact.image_written)
    );
    println!(
        "Manifest written: {}",
        yes_no(file_result.artifact.manifest_written)
    );
    println!("Image unavailable reason: {reason}");
    println!("Evidence level: {evidence_level}");
    println!("Monitor projection:");
    println!(
        "  prepared descriptor present: {}",
        yes_no(monitor_projection.prepared_descriptor_present)
    );
    println!(
        "  submitted to file presenter: {}",
        yes_no(monitor_projection.submitted_to_presenter)
    );
    println!(
        "  source frame: {}",
        qgs_optional_u64_label(monitor_projection.source_frame)
    );
    println!("  payload status: {:?}", monitor_projection.payload_status);
    println!(
        "  frame presented test boundary: {}",
        yes_no(monitor_projection.frame_presented_test_boundary)
    );
    println!(
        "  frame presented real display: {}",
        yes_no(monitor_projection.frame_presented_real_backend)
    );
    println!(
        "  real display evidence: {}",
        if monitor_projection.real_display_evidence_present {
            "present"
        } else {
            "none"
        }
    );
    println!(
        "  visual verified: {}",
        yes_no(monitor_projection.visual_verified)
    );
    println!(
        "Session setup: commands_executed={} accepted={} rejected={} final_state={:?}",
        setup_report.executed,
        setup_report.accepted,
        setup_report.rejected,
        setup_snapshot.passive_view.transport.state
    );
    println!("Real display output: no");
    println!("VisualVerified: no");
    println!("Realtime playback: no");
    println!("A/V sync: no");
    println!("Device output: no");
    println!("FramePresented real display claim: no");
    println!("Private path exposed: {}", yes_no(private_path_exposed));

    Ok(())
}

fn qgs_gpu_payload_readback_diagnostic_report(
    original_path: &Path,
    proxy_path: &Path,
    output_dir: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let descriptor = build_qnc_prepared_input_descriptor(original_path, proxy_path)?;
    descriptor.validate()?;
    let input_plan =
        QgsInputPlan::from_descriptor(&descriptor, default_qgs_input_plan_queue_requirements())?;
    let active_duration = Duration::from_millis(1_000);
    let active_frames = qgs_frames_for_duration(active_duration, input_plan.video_source.timebase)?;
    let source_uri = input_plan.video_source.media_uri.clone();
    let public_source_uri = descriptor
        .binding
        .proxy_media_uri
        .as_deref()
        .unwrap_or(&source_uri)
        .to_string();

    let mut runtime = QgsSessionRuntime::new(input_plan, QgsSessionRuntimeConfig::default())?;
    let setup_commands = qgs_session_command_sequence(&source_uri, active_frames);
    for command in setup_commands.iter().take(8) {
        runtime.enqueue(command.clone()).map_err(|err| {
            format!(
                "GPU payload readback diagnostic setup enqueue failed for id {}: {err:?}",
                command.command_id
            )
        })?;
    }
    let setup_report = runtime.drain_queue();
    let setup_snapshot = runtime.snapshot();

    let proxy = Mp4Source::open(proxy_path)?;
    let proxy_video = proxy
        .video
        .as_ref()
        .ok_or("proxy has no H.264 video track")?;
    let proxy_h264 = classify_video_track(proxy_video)?;
    let selected = selected_journalist_preview_frames(proxy_video)?;
    let selected_frame = selected
        .first()
        .ok_or("proxy has no selected journalist preview frame")?;
    let frame_duration = ProxyPlaybackProfile::Journalist50iPreview
        .presentation_rate(RationalRate::new(
            u64::from(proxy_video.frame_rate.numerator),
            u64::from(proxy_video.frame_rate.denominator),
        )?)?
        .frame_duration()?;
    let video_slot = BroadcastPreparedVideoSlot {
        slot_index: 0,
        source_mode: BroadcastVideoSourceMode::ProxyPreview,
        video_source_role: BroadcastMediaSourceRole::ProxyPreviewVideo,
        source_frame_index: Some(selected_frame.source_presentation_index),
        selected_preview_frame_index: Some(selected_frame.preview_index),
        presentation_time: selected_frame.start_time,
        duration: frame_duration,
        status: BroadcastPreparedVideoSlotStatus::Prepared,
    };
    let proxy_payload_proof = bind_proxy_preview_video_payloads_with_readback(
        proxy_video,
        &proxy_h264,
        std::slice::from_ref(&video_slot),
        true,
    )?;
    let video_binding = proxy_payload_proof
        .bindings
        .first()
        .cloned()
        .ok_or("proxy payload proof did not produce a binding")?;
    let payload = video_binding
        .payload
        .as_ref()
        .ok_or("GPU payload readback diagnostic missing proxy video payload")?;
    let readback = proxy_payload_proof
        .readback_outputs
        .get(&selected_frame.source_presentation_index)
        .ok_or("GPU payload readback diagnostic did not receive CPU-readable pixels")?;

    let output_dir = output_dir.unwrap_or_else(|| Path::new("target/qgs-visual-diagnostics"));
    std::fs::create_dir_all(output_dir)?;
    let image_file_name = gpu_payload_readback_image_file_name(
        &public_source_uri,
        selected_frame.source_presentation_index,
    );
    let manifest_file_name = gpu_payload_readback_manifest_file_name(
        &public_source_uri,
        selected_frame.source_presentation_index,
    );
    let image_path = output_dir.join(&image_file_name);
    let manifest_path = output_dir.join(&manifest_file_name);
    let public_image_path = format!("qgs-visual-diagnostics/{image_file_name}");
    let public_manifest_path = format!("qgs-visual-diagnostics/{manifest_file_name}");

    let ppm_bytes = rgba_u16_to_ppm_p6_rgb8(&readback.rgba_u16, readback.width, readback.height)?;
    File::create(&image_path)?.write_all(&ppm_bytes)?;

    let file_result = submit_qgs_file_presenter_image(
        public_source_uri.clone(),
        &video_binding,
        public_image_path,
    )?;
    let monitor_update = project_qgs_monitor_update_from_file_presenter_result(&file_result);
    let monitor_projection = QgsQncMonitorProjection::from_presenter_update(&monitor_update);
    let manifest_json = qgs_gpu_payload_readback_manifest_json(
        &file_result,
        &monitor_projection,
        &public_manifest_path,
        ppm_bytes.len(),
        readback.checksum,
        readback.output_bytes,
        setup_report.executed,
        setup_report.accepted,
        setup_report.rejected,
        &format!("{:?}", setup_snapshot.passive_view.transport.state),
    );
    File::create(&manifest_path)?.write_all(manifest_json.as_bytes())?;

    let private_path_exposed = monitor_projection.private_path_exposed
        || setup_snapshot.private_path_exposed
        || setup_report.transcript.exposes_private_path()
        || manifest_json_contains_private_path(&manifest_json);

    println!("QGS GPU Payload Readback Visual Diagnostic");
    println!("------------------------------------------");
    println!("Audio source: original MXF");
    println!("Video source: proxy MP4");
    println!("Proxy AAC: not used");
    println!("Presenter kind: FilePresenter");
    println!("Preview profile: journalist-50i-preview");
    println!("Public source URI: {public_source_uri}");
    println!(
        "Selected frame: {}",
        selected_frame.source_presentation_index
    );
    println!("Selected preview frame: {}", selected_frame.preview_index);
    println!("Payload kind: {:?}", payload.kind);
    println!("Payload format: {:?}", payload.format);
    println!("Payload backend: {:?}", payload.backend_path);
    println!(
        "Payload dimensions: visible={}x{} coded={}x{}",
        payload.visible_width, payload.visible_height, payload.coded_width, payload.coded_height
    );
    println!("Readback attempted: yes");
    println!("Readback available: yes");
    println!("Readback format: RgbaU16");
    println!("Output artifact format: PPM P6 RGB8");
    println!("Conversion: RgbaU16 high byte -> RGB8; alpha discarded");
    println!("Image width: {}", readback.width);
    println!("Image height: {}", readback.height);
    println!("Artifact kind: image");
    println!("Image artifact path: {}", image_path.display());
    println!("Manifest artifact path: {}", manifest_path.display());
    println!("Image bytes written: {}", ppm_bytes.len());
    println!("GPU readback payload bytes: {}", readback.output_bytes);
    println!("GPU readback checksum: 0x{:016x}", readback.checksum);
    println!("Evidence level: {:?}", file_result.evidence.evidence_level);
    println!("Monitor projection:");
    println!(
        "  prepared descriptor present: {}",
        yes_no(monitor_projection.prepared_descriptor_present)
    );
    println!(
        "  submitted to file presenter: {}",
        yes_no(monitor_projection.submitted_to_presenter)
    );
    println!(
        "  source frame: {}",
        qgs_optional_u64_label(monitor_projection.source_frame)
    );
    println!("  payload status: {:?}", monitor_projection.payload_status);
    println!(
        "  frame presented test boundary: {}",
        yes_no(monitor_projection.frame_presented_test_boundary)
    );
    println!(
        "  frame presented real display: {}",
        yes_no(monitor_projection.frame_presented_real_backend)
    );
    println!(
        "  real display evidence: {}",
        if monitor_projection.real_display_evidence_present {
            "present"
        } else {
            "none"
        }
    );
    println!(
        "  visual verified: {}",
        yes_no(monitor_projection.visual_verified)
    );
    println!("Real display output: no");
    println!("VisualVerified: no");
    println!("Realtime playback: no");
    println!("A/V sync: no");
    println!("Device output: no");
    println!("FramePresented real display claim: no");
    println!("Private path exposed: {}", yes_no(private_path_exposed));

    Ok(())
}

fn qgs_runtime_surface_e2e_report(
    original_path: &Path,
    proxy_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let descriptor = build_qnc_prepared_input_descriptor(original_path, proxy_path)?;
    descriptor.validate()?;
    let input_plan =
        QgsInputPlan::from_descriptor(&descriptor, default_qgs_input_plan_queue_requirements())?;
    let active_duration = Duration::from_millis(1_000);
    let active_frames = qgs_frames_for_duration(active_duration, input_plan.video_source.timebase)?;
    let source_uri = input_plan.video_source.media_uri.clone();
    let proxy_uri = descriptor
        .binding
        .proxy_media_uri
        .as_deref()
        .unwrap_or(&source_uri)
        .to_string();

    let mut runtime = QgsSessionRuntime::new(input_plan, QgsSessionRuntimeConfig::default())?;
    let commands = qgs_session_command_sequence(&source_uri, active_frames);
    for command in commands.iter().take(8) {
        runtime.enqueue(command.clone()).map_err(|err| {
            format!(
                "runtime surface e2e enqueue failed for id {}: {err:?}",
                command.command_id
            )
        })?;
    }

    let mut command_results = Vec::new();
    let mut projected_events = Vec::new();
    let mut play_ready_before_play = false;
    let mut prepared_buffer_count = 0usize;
    let mut selected_prepared_frame = None;
    while let Some(result) = runtime.execute_next() {
        if result.command_id.0 == 5 {
            play_ready_before_play = result.passive_view.readiness.play_ready;
        }
        if result.command_id.0 == 7 {
            prepared_buffer_count = result.passive_view.prepared_buffer.prepared_frame_count;
            selected_prepared_frame = result.passive_view.prepared_buffer.prepared_start_frame;
        }
        projected_events.extend(result.projected_events.clone());
        command_results.push(result);
    }
    let tick_snapshot = runtime.snapshot();

    let proxy = Mp4Source::open(proxy_path)?;
    let proxy_video = proxy
        .video
        .as_ref()
        .ok_or("proxy has no H.264 video track")?;
    let proxy_h264 = classify_video_track(proxy_video)?;
    let selected = selected_journalist_preview_frames(proxy_video)?;
    let selected_frame = selected
        .first()
        .ok_or("proxy has no selected journalist preview frame")?;
    let frame_duration = ProxyPlaybackProfile::Journalist50iPreview
        .presentation_rate(RationalRate::new(
            u64::from(proxy_video.frame_rate.numerator),
            u64::from(proxy_video.frame_rate.denominator),
        )?)?
        .frame_duration()?;
    let video_slot = BroadcastPreparedVideoSlot {
        slot_index: 0,
        source_mode: BroadcastVideoSourceMode::ProxyPreview,
        video_source_role: BroadcastMediaSourceRole::ProxyPreviewVideo,
        source_frame_index: Some(selected_frame.source_presentation_index),
        selected_preview_frame_index: Some(selected_frame.preview_index),
        presentation_time: selected_frame.start_time,
        duration: frame_duration,
        status: BroadcastPreparedVideoSlotStatus::Prepared,
    };
    let proxy_payload_proof = bind_proxy_preview_video_payloads(
        proxy_video,
        &proxy_h264,
        std::slice::from_ref(&video_slot),
    )?;
    let video_binding = proxy_payload_proof
        .bindings
        .first()
        .cloned()
        .ok_or("proxy payload proof did not produce a binding")?;
    let presentation_binding = BroadcastPresentationPayloadBinding {
        presentation_slot_index: 0,
        video_binding_index: video_binding.video_slot_index,
        audio_binding_index: 0,
        presentation_time: selected_frame.start_time,
        duration: frame_duration,
        ready: true,
        readiness: BroadcastPresentationPayloadReadiness::PayloadReady,
    };
    let payload = video_binding
        .payload
        .as_ref()
        .ok_or("runtime surface e2e missing proxy video payload")?;
    let mut presenter = BroadcastTestVideoPresenter::new(BroadcastTestVideoPresenterConfig {
        accepted_kind: BroadcastVideoPayloadKind::ProcessedGpuFrame,
        accepted_format: BroadcastVideoPayloadFormat::RgbaU16,
        visible_width: proxy_h264.width,
        visible_height: proxy_h264.height,
        coded_width: proxy_h264.coded_width,
        coded_height: proxy_h264.coded_height,
    })?;
    let presenter_result = submit_qgs_presenter_payload_to_test_boundary(
        proxy_uri.clone(),
        &presentation_binding,
        &video_binding,
        &mut presenter,
    )?;
    let monitor_update = project_qgs_monitor_update_from_presenter_result(&presenter_result);
    let monitor_projection = QgsQncMonitorProjection::from_presenter_update(&monitor_update);

    for command in commands.iter().skip(8).take(2) {
        runtime.enqueue(command.clone()).map_err(|err| {
            format!(
                "runtime surface e2e cleanup enqueue failed for id {}: {err:?}",
                command.command_id
            )
        })?;
    }
    while let Some(result) = runtime.execute_next() {
        projected_events.extend(result.projected_events.clone());
        command_results.push(result);
    }
    let final_snapshot = runtime.snapshot();
    let transcript = runtime.transcript();
    let accepted_count = command_results
        .iter()
        .filter(|result| {
            result
                .outcome
                .as_ref()
                .is_some_and(|outcome| outcome.accepted)
        })
        .count();
    let rejected_count = command_results
        .iter()
        .filter(|result| {
            result
                .outcome
                .as_ref()
                .is_some_and(|outcome| !outcome.accepted)
        })
        .count();
    let event_sequence_monotonic = qgs_qnc_projected_sequence_is_monotonic(&projected_events);
    let private_path_exposed = final_snapshot.private_path_exposed
        || tick_snapshot.private_path_exposed
        || monitor_projection.private_path_exposed
        || transcript.exposes_private_path()
        || projected_events
            .iter()
            .any(QgsQncEventEnvelope::exposes_private_path);
    let presenter_evidence_kind = monitor_update
        .presenter_evidence_kind
        .map(|kind| format!("{kind:?}"))
        .unwrap_or_else(|| "none".to_string());

    println!("QGS Runtime Surface End-to-End Acceptance");
    println!("-----------------------------------------");
    println!(
        "Public original URI: {}",
        descriptor.binding.original_media_uri
    );
    println!("Public proxy URI: {proxy_uri}");
    println!("Private path exposed: {}", yes_no(private_path_exposed));
    println!("Command queue accepted count: {accepted_count}");
    println!("Command queue rejected count: {rejected_count}");
    println!(
        "Transport play_ready before Play: {}",
        yes_no(play_ready_before_play)
    );
    println!(
        "Final transport state: {:?}",
        final_snapshot.passive_view.transport.state
    );
    println!("Prepared buffer count: {prepared_buffer_count}");
    println!(
        "Selected prepared frame: {}",
        qgs_optional_u64_label(
            selected_prepared_frame.or(Some(selected_frame.source_presentation_index))
        )
    );
    println!(
        "Presenter boundary submitted: {}",
        yes_no(monitor_update.submitted_to_presenter)
    );
    println!("Presenter kind: TestPresenter");
    println!("Presenter evidence kind: {presenter_evidence_kind}");
    println!(
        "Test presenter accepted: {}",
        yes_no(presenter.accepted_count() == 1 && presenter.rejected_count() == 0)
    );
    println!("Payload kind: {:?}", payload.kind);
    println!("Payload format: {:?}", payload.format);
    println!("Payload backend: {:?}", payload.backend_path);
    println!("Monitor projection:");
    println!(
        "  prepared descriptor present: {}",
        yes_no(monitor_projection.prepared_descriptor_present)
    );
    println!(
        "  submitted to presenter: {}",
        yes_no(monitor_projection.submitted_to_presenter)
    );
    println!(
        "  real display evidence: {}",
        if monitor_projection.real_display_evidence_present {
            "present"
        } else {
            "none"
        }
    );
    println!(
        "  visual verified: {}",
        yes_no(monitor_projection.visual_verified)
    );
    println!(
        "  real presented: {}",
        yes_no(monitor_projection.frame_presented_real_backend)
    );
    println!("Projected event count: {}", projected_events.len());
    println!(
        "Event sequence monotonic: {}",
        yes_no(event_sequence_monotonic)
    );
    println!("Final passive view summary:");
    print_qgs_qnc_passive_view(&final_snapshot.passive_view);
    println!("Runtime surface non-claims:");
    println!("  realtime playback: no");
    println!("  A/V sync: no");
    println!("  real display output: no");
    println!("  FramePresented real display: no");
    println!("  AudioDeviceVerified: no");
    println!("  RealtimeVerified: no");

    Ok(())
}

fn qgs_broadcast_player_control_core_report(
    original_path: &Path,
    proxy_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let descriptor = build_qnc_prepared_input_descriptor(original_path, proxy_path)?;
    descriptor.validate()?;
    let plan =
        QgsInputPlan::from_descriptor(&descriptor, default_qgs_input_plan_queue_requirements())?;
    let active_duration = Duration::from_millis(1_000);
    let active_frames = qgs_frames_for_duration(active_duration, plan.video_source.timebase)?;
    let safe_seek_frame = if active_frames > 2 { 2 } else { 0 };
    let mut core = QgsBroadcastPlayerCore::new(plan)?;
    let commands = vec![
        QgsBroadcastPlayerCommand::LoadPreparedInput,
        QgsBroadcastPlayerCommand::Prepare {
            active_range: Some((0, active_frames)),
        },
        QgsBroadcastPlayerCommand::Cue { frame: 0 },
        QgsBroadcastPlayerCommand::Tick { carrier_frame: 0 },
        QgsBroadcastPlayerCommand::Play,
        QgsBroadcastPlayerCommand::Tick { carrier_frame: 1 },
        QgsBroadcastPlayerCommand::Pause,
        QgsBroadcastPlayerCommand::Seek {
            frame: safe_seek_frame,
        },
        QgsBroadcastPlayerCommand::Prepare {
            active_range: Some((0, active_frames)),
        },
        QgsBroadcastPlayerCommand::Play,
        QgsBroadcastPlayerCommand::Stop,
        QgsBroadcastPlayerCommand::Unload,
    ];

    let mut results = Vec::new();
    for command in commands {
        results.push(core.execute(command));
    }
    let snapshot = core.snapshot();
    let accepted_count = results.iter().filter(|result| result.accepted).count();
    let rejected_count = results.len().saturating_sub(accepted_count);
    let lower_level_event_count = results
        .iter()
        .map(|result| {
            result
                .lower_level_results
                .iter()
                .map(|lower| lower.projected_events.len())
                .sum::<usize>()
        })
        .sum::<usize>();
    let private_path_exposed = snapshot.exposes_private_path()
        || results.iter().any(|result| {
            result
                .product_events
                .iter()
                .any(qgs_media_runtime::QgsBroadcastPlayerEvent::exposes_private_path)
                || result
                    .lower_level_results
                    .iter()
                    .any(|lower| lower.exposes_private_path())
        });

    println!("QGS Broadcast Player Control Core");
    println!("---------------------------------");
    println!("Player surface: production-shaped backend control facade");
    println!("Video source mode: ProxyPreview");
    println!("Video source: proxy MP4");
    println!("Audio source: original MXF");
    println!("Original MXF audio authoritative: yes");
    println!("Proxy AAC: not used / not authoritative");
    println!("Broadcast audio model: discrete original mono lanes");
    println!("Real display output: no");
    println!("Realtime playback: no");
    println!("A/V sync: no");
    println!("Command sequence:");
    for (index, result) in results.iter().enumerate() {
        println!(
            "  {:02}. {:?}: accepted={} status={:?} reason={}",
            index + 1,
            result.command,
            yes_no(result.accepted),
            result.snapshot.status,
            result.reason.unwrap_or("n/a")
        );
    }
    println!("Command count: {}", results.len());
    println!("Accepted commands: {accepted_count}");
    println!("Rejected commands: {rejected_count}");
    println!("Product event count: {}", core.events().len());
    println!("Lower-level event count: {lower_level_event_count}");
    println!("Final player snapshot:");
    print_broadcast_player_snapshot(&snapshot);
    println!("Product events:");
    for event in core.events() {
        println!(
            "  #{:02} {:?} status={:?} source={} summary={}",
            event.sequence,
            event.kind,
            event.status,
            event.public_source_uri.as_deref().unwrap_or("none"),
            event.summary
        );
    }
    println!("Private path exposed: {}", yes_no(private_path_exposed));
    println!(
        "Real display: {}",
        snapshot.device_status.real_display_backend
    );
    println!(
        "QNC OS display target: {}",
        snapshot.device_status.qnc_os_display_target
    );
    println!("X11 target: {}", snapshot.device_status.x11_target);
    println!(
        "Visual verified: {}",
        yes_no(snapshot.device_status.visual_verified)
    );
    println!(
        "Realtime verified: {}",
        yes_no(snapshot.device_status.realtime_verified)
    );
    println!(
        "Audio device production verified: {}",
        yes_no(snapshot.device_status.audio_device_production_verified)
    );
    println!(
        "A/V sync verified: {}",
        yes_no(snapshot.device_status.av_sync_verified)
    );
    println!("QNC UI integration: no");
    println!("Realtime playback: no");
    println!("Device output: no");
    println!("FramePresented real display claim: no");

    Ok(())
}

fn qgs_broadcast_player_operational_runtime_report(
    original_path: &Path,
    proxy_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let descriptor = build_qnc_prepared_input_descriptor(original_path, proxy_path)?;
    descriptor.validate()?;
    let plan =
        QgsInputPlan::from_descriptor(&descriptor, default_qgs_input_plan_queue_requirements())?;
    let active_duration = Duration::from_millis(1_000);
    let active_frames = qgs_frames_for_duration(active_duration, plan.video_source.timebase)?;
    let safe_seek_frame = if active_frames > 4 { 4 } else { 0 };
    let mut runtime =
        QgsBroadcastPlayerOperationalRuntime::new(plan, QgsOperationalRuntimeConfig::default())?;
    let commands = vec![
        QgsBroadcastPlayerCommand::Play,
        QgsBroadcastPlayerCommand::LoadPreparedInput,
        QgsBroadcastPlayerCommand::Prepare {
            active_range: Some((0, active_frames)),
        },
        QgsBroadcastPlayerCommand::Cue { frame: 0 },
        QgsBroadcastPlayerCommand::Tick { carrier_frame: 0 },
        QgsBroadcastPlayerCommand::Play,
        QgsBroadcastPlayerCommand::Tick { carrier_frame: 0 },
        QgsBroadcastPlayerCommand::Tick { carrier_frame: 0 },
        QgsBroadcastPlayerCommand::Tick { carrier_frame: 0 },
        QgsBroadcastPlayerCommand::Pause,
        QgsBroadcastPlayerCommand::Tick { carrier_frame: 0 },
        QgsBroadcastPlayerCommand::Seek {
            frame: safe_seek_frame,
        },
        QgsBroadcastPlayerCommand::Tick { carrier_frame: 0 },
        QgsBroadcastPlayerCommand::Prepare {
            active_range: Some((0, active_frames)),
        },
        QgsBroadcastPlayerCommand::Play,
        QgsBroadcastPlayerCommand::Tick { carrier_frame: 0 },
        QgsBroadcastPlayerCommand::Tick { carrier_frame: 0 },
        QgsBroadcastPlayerCommand::Tick { carrier_frame: 0 },
        QgsBroadcastPlayerCommand::Stop,
        QgsBroadcastPlayerCommand::Unload,
    ];

    let mut results = Vec::new();
    for command in commands {
        results.push(runtime.execute(command));
    }
    let snapshot = runtime.snapshot();
    let accepted_count = results.iter().filter(|result| result.accepted).count();
    let rejected_count = results.len().saturating_sub(accepted_count);
    let status_transitions = results
        .iter()
        .map(|result| format!("{:?}", result.snapshot.operational_status))
        .collect::<Vec<_>>()
        .join(" -> ");
    let private_path_exposed = snapshot.exposes_private_path()
        || results
            .iter()
            .any(|result| result.snapshot.exposes_private_path());
    let source_loaded_after_stop = results
        .iter()
        .find(|result| matches!(result.command, QgsBroadcastPlayerCommand::Stop))
        .is_some_and(|result| result.snapshot.player.readiness.source_loaded);
    let source_unloaded_after_unload = results
        .iter()
        .rev()
        .find(|result| matches!(result.command, QgsBroadcastPlayerCommand::Unload))
        .is_some_and(|result| !result.snapshot.player.readiness.source_loaded);

    println!("QGS Broadcast Player Operational Runtime");
    println!("----------------------------------------");
    println!("Runtime surface: operational deterministic backend player");
    println!("Video source mode: ProxyPreview");
    println!("Video source: proxy MP4");
    println!("Audio source: original MXF");
    println!("Original MXF audio authoritative: yes");
    println!("Proxy AAC: not used / not authoritative");
    println!("Broadcast audio model: discrete original mono lanes");
    println!("Real display output: no");
    println!("Realtime playback: no");
    println!("A/V sync: no");
    println!("Command sequence:");
    for (index, result) in results.iter().enumerate() {
        println!(
            "  {:02}. {:?}: accepted={} status={:?} frame={} audio={} reason={}",
            index + 1,
            result.command,
            yes_no(result.accepted),
            result.snapshot.operational_status,
            qgs_optional_u64_label(result.snapshot.current_frame),
            qgs_optional_sample_range_label(result.snapshot.current_audio_sample_range),
            result.reason.unwrap_or("n/a")
        );
        for event in &result.events {
            println!(
                "      event #{:02} {:?} status={:?} frame={} summary={}",
                event.sequence,
                event.kind,
                event.status,
                qgs_optional_u64_label(event.frame),
                event.summary
            );
        }
    }
    println!("Command count: {}", results.len());
    println!("Accepted commands: {accepted_count}");
    println!("Rejected commands: {rejected_count}");
    println!("Status transitions: {status_transitions}");
    println!("Final operational snapshot:");
    print_operational_runtime_snapshot(&snapshot);
    println!(
        "Source remains loaded after Stop: {}",
        yes_no(source_loaded_after_stop)
    );
    println!(
        "Source unloaded after Unload: {}",
        yes_no(source_unloaded_after_unload)
    );
    println!("Private path exposed: {}", yes_no(private_path_exposed));
    println!(
        "Real display: {}",
        snapshot.player.device_status.real_display_backend
    );
    println!(
        "QNC OS display target: {}",
        snapshot.player.device_status.qnc_os_display_target
    );
    println!("X11 target: {}", snapshot.player.device_status.x11_target);
    println!(
        "Visual verified: {}",
        yes_no(snapshot.player.device_status.visual_verified)
    );
    println!(
        "Realtime verified: {}",
        yes_no(snapshot.player.device_status.realtime_verified)
    );
    println!(
        "Audio device production verified: {}",
        yes_no(
            snapshot
                .player
                .device_status
                .audio_device_production_verified
        )
    );
    println!(
        "A/V sync verified: {}",
        yes_no(snapshot.player.device_status.av_sync_verified)
    );
    println!("QNC UI integration: no");
    println!("Device output: no");
    println!("FramePresented real display claim: no");

    Ok(())
}

fn qgs_broadcast_player_fault_recovery_report(
    original_path: &Path,
    proxy_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let descriptor = build_qnc_prepared_input_descriptor(original_path, proxy_path)?;
    descriptor.validate()?;
    let plan =
        QgsInputPlan::from_descriptor(&descriptor, default_qgs_input_plan_queue_requirements())?;
    let active_duration = Duration::from_millis(1_000);
    let active_frames = qgs_frames_for_duration(active_duration, plan.video_source.timebase)?;
    let outside_frame = active_frames.saturating_add(50);
    let mut runtime =
        QgsBroadcastPlayerOperationalRuntime::new(plan, QgsOperationalRuntimeConfig::default())?;
    let commands = vec![
        QgsBroadcastPlayerCommand::Play,
        QgsBroadcastPlayerCommand::Seek { frame: 0 },
        QgsBroadcastPlayerCommand::LoadPreparedInput,
        QgsBroadcastPlayerCommand::Cue {
            frame: outside_frame,
        },
        QgsBroadcastPlayerCommand::Prepare {
            active_range: Some((0, active_frames)),
        },
        QgsBroadcastPlayerCommand::Play,
        QgsBroadcastPlayerCommand::Cue { frame: 0 },
        QgsBroadcastPlayerCommand::Tick { carrier_frame: 0 },
        QgsBroadcastPlayerCommand::Play,
        QgsBroadcastPlayerCommand::Seek {
            frame: outside_frame,
        },
        QgsBroadcastPlayerCommand::Tick { carrier_frame: 0 },
        QgsBroadcastPlayerCommand::Tick { carrier_frame: 0 },
        QgsBroadcastPlayerCommand::Stop,
        QgsBroadcastPlayerCommand::Unload,
        QgsBroadcastPlayerCommand::Seek { frame: 0 },
    ];

    let mut results = Vec::new();
    let mut rejected_state_mutations = 0usize;
    let mut rejected_source_loaded_after = Vec::new();
    for command in commands {
        let before = runtime.snapshot();
        let result = runtime.execute(command);
        if !result.accepted {
            let state_mutated = before.operational_status != result.snapshot.operational_status
                || before.current_frame != result.snapshot.current_frame
                || before.player.readiness.source_loaded
                    != result.snapshot.player.readiness.source_loaded
                || before.player.position.active_range
                    != result.snapshot.player.position.active_range;
            if state_mutated {
                rejected_state_mutations = rejected_state_mutations.saturating_add(1);
            }
            rejected_source_loaded_after.push(result.snapshot.player.readiness.source_loaded);
        }
        results.push(result);
    }
    let snapshot = runtime.snapshot();
    let accepted_count = results.iter().filter(|result| result.accepted).count();
    let rejected_count = results.len().saturating_sub(accepted_count);
    let private_path_exposed = snapshot.exposes_private_path()
        || results
            .iter()
            .any(|result| result.snapshot.exposes_private_path());
    let status_transitions = results
        .iter()
        .map(|result| format!("{:?}", result.snapshot.operational_status))
        .collect::<Vec<_>>()
        .join(" -> ");

    println!("QGS Broadcast Player Fault And Recovery Rules");
    println!("---------------------------------------------");
    println!("Runtime surface: production-shaped fault/recovery policy");
    println!("Video source mode: ProxyPreview");
    println!("Video source: proxy MP4");
    println!("Audio source: original MXF");
    println!("Original MXF audio authoritative: yes");
    println!("Proxy AAC: not used / not authoritative");
    println!("Broadcast audio model: discrete original mono lanes");
    println!("Real display output: no");
    println!("Realtime playback: no");
    println!("A/V sync: no");
    println!("Command sequence:");
    for (index, result) in results.iter().enumerate() {
        let last_fault = result.snapshot.fault_snapshot.last_fault.as_ref();
        println!(
            "  {:02}. {:?}: accepted={} status={:?} frame={} reason={} last_fault={} recovery={}",
            index + 1,
            result.command,
            yes_no(result.accepted),
            result.snapshot.operational_status,
            qgs_optional_u64_label(result.snapshot.current_frame),
            result.reason.unwrap_or("n/a"),
            last_fault.map(|fault| fault.kind.label()).unwrap_or("none"),
            last_fault
                .map(|fault| fault.recovery_action.label())
                .unwrap_or("none")
        );
        for event in &result.events {
            println!(
                "      event #{:02} {:?} status={:?} frame={} summary={}",
                event.sequence,
                event.kind,
                event.status,
                qgs_optional_u64_label(event.frame),
                event.summary
            );
        }
    }
    println!("Command count: {}", results.len());
    println!("Accepted commands: {accepted_count}");
    println!("Rejected commands: {rejected_count}");
    println!(
        "Faults recorded: {}",
        snapshot.fault_snapshot.active_faults.len()
    );
    println!(
        "Recoverable fault count: {}",
        snapshot.fault_snapshot.recoverable_fault_count
    );
    println!(
        "Fatal fault count: {}",
        snapshot.fault_snapshot.fatal_fault_count
    );
    println!(
        "Last fault kind: {}",
        snapshot
            .fault_snapshot
            .last_fault
            .as_ref()
            .map(|fault| fault.kind.label())
            .unwrap_or("none")
    );
    println!(
        "Last fault severity: {}",
        snapshot
            .fault_snapshot
            .last_fault
            .as_ref()
            .map(|fault| fault.severity.label())
            .unwrap_or("none")
    );
    println!(
        "Last fault scope: {}",
        snapshot
            .fault_snapshot
            .last_fault
            .as_ref()
            .map(|fault| fault.scope.label())
            .unwrap_or("none")
    );
    println!(
        "Last fault recoverable: {}",
        snapshot
            .fault_snapshot
            .last_fault
            .as_ref()
            .map(|fault| yes_no(fault.recoverable))
            .unwrap_or("n/a")
    );
    println!(
        "Recommended recovery action: {}",
        snapshot
            .fault_snapshot
            .recommended_recovery_action
            .map(|action| action.label())
            .unwrap_or("none")
    );
    println!(
        "State mutated by rejected command: {}",
        yes_no(rejected_state_mutations > 0)
    );
    println!(
        "Rejected command state mutation count: {}",
        rejected_state_mutations
    );
    println!(
        "Source loaded after rejected commands: {}",
        rejected_source_loaded_after
            .iter()
            .map(|value| yes_no(*value))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("Status transitions: {status_transitions}");
    println!("Final operational snapshot:");
    print_operational_runtime_snapshot(&snapshot);
    println!("Private path exposed: {}", yes_no(private_path_exposed));
    println!(
        "Real display: {}",
        snapshot.player.device_status.real_display_backend
    );
    println!(
        "Visual verified: {}",
        yes_no(snapshot.player.device_status.visual_verified)
    );
    println!(
        "Realtime verified: {}",
        yes_no(snapshot.player.device_status.realtime_verified)
    );
    println!(
        "Audio device production verified: {}",
        yes_no(
            snapshot
                .player
                .device_status
                .audio_device_production_verified
        )
    );
    println!(
        "A/V sync verified: {}",
        yes_no(snapshot.player.device_status.av_sync_verified)
    );
    println!(
        "QNC OS display target: {}",
        snapshot.player.device_status.qnc_os_display_target
    );
    println!("X11 target: {}", snapshot.player.device_status.x11_target);
    println!("QNC UI integration: no");
    println!("Device output: no");
    println!("FramePresented real display claim: no");

    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum QgsBroadcastPlayerRunView {
    Compact,
    Detailed,
}

impl QgsBroadcastPlayerRunView {
    fn parse(value: &std::ffi::OsStr) -> Result<Self, &'static str> {
        match value.to_string_lossy().as_ref() {
            "compact" => Ok(Self::Compact),
            "detailed" => Ok(Self::Detailed),
            _ => Err("expects compact or detailed"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct QgsBroadcastPlayerRunOptions {
    frames: u64,
    seek_frame: Option<u64>,
    view: QgsBroadcastPlayerRunView,
}

impl Default for QgsBroadcastPlayerRunOptions {
    fn default() -> Self {
        Self {
            frames: 25,
            seek_frame: None,
            view: QgsBroadcastPlayerRunView::Compact,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum QgsBroadcastPlayerLivePace {
    Logical,
    Wall,
}

impl QgsBroadcastPlayerLivePace {
    fn parse(value: &std::ffi::OsStr) -> Result<Self, &'static str> {
        match value.to_string_lossy().as_ref() {
            "logical" => Ok(Self::Logical),
            "wall" => Ok(Self::Wall),
            _ => Err("expects logical or wall"),
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Logical => "logical",
            Self::Wall => "wall",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum QgsBroadcastPlayerLiveVideoOutputMode {
    None,
    PreviewFiles,
    PreviewWindow,
}

impl QgsBroadcastPlayerLiveVideoOutputMode {
    fn parse(value: &std::ffi::OsStr) -> Result<Self, &'static str> {
        match value.to_string_lossy().as_ref() {
            "none" => Ok(Self::None),
            "preview-files" => Ok(Self::PreviewFiles),
            "preview-window" => Ok(Self::PreviewWindow),
            _ => Err("expects none, preview-files, or preview-window"),
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::PreviewFiles => "preview-files",
            Self::PreviewWindow => "preview-window",
        }
    }

    const fn to_brick(self) -> LiveVideoOutputMode {
        match self {
            Self::None => LiveVideoOutputMode::None,
            Self::PreviewFiles => LiveVideoOutputMode::PreviewFiles,
            Self::PreviewWindow => LiveVideoOutputMode::PreviewWindow,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum QgsBroadcastPlayerLiveAudioOutputMode {
    None,
    PipeWireDesktopMonitor,
    PipeWireMonitor,
    PipeWire4Mono,
}

impl QgsBroadcastPlayerLiveAudioOutputMode {
    fn parse(value: &std::ffi::OsStr) -> Result<Self, &'static str> {
        match value.to_string_lossy().as_ref() {
            "none" => Ok(Self::None),
            "pipewire-desktop-monitor" => Ok(Self::PipeWireDesktopMonitor),
            "pipewire-monitor" => Ok(Self::PipeWireMonitor),
            "pipewire-4mono" => Ok(Self::PipeWire4Mono),
            _ => Err("expects none, pipewire-desktop-monitor, pipewire-monitor, or pipewire-4mono"),
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::PipeWireDesktopMonitor => "pipewire-desktop-monitor",
            Self::PipeWireMonitor => "pipewire-desktop-monitor",
            Self::PipeWire4Mono => "pipewire-4mono",
        }
    }

    const fn to_brick(self) -> AudioOutputMode {
        match self {
            Self::None => AudioOutputMode::None,
            Self::PipeWireDesktopMonitor => AudioOutputMode::PipeWireDesktopMonitor,
            Self::PipeWireMonitor => AudioOutputMode::PipeWireMonitor,
            Self::PipeWire4Mono => AudioOutputMode::PipeWire4Mono,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct QgsBroadcastPlayerLiveOptions {
    start_frame: u64,
    status_every: u64,
    max_frames: Option<u64>,
    seek_frame: Option<u64>,
    view: QgsBroadcastPlayerRunView,
    pace: QgsBroadcastPlayerLivePace,
    no_interactive: bool,
    video_output: QgsBroadcastPlayerLiveVideoOutputMode,
    audio_output: QgsBroadcastPlayerLiveAudioOutputMode,
    output_dir: PathBuf,
    preview_every: u64,
}

impl Default for QgsBroadcastPlayerLiveOptions {
    fn default() -> Self {
        Self {
            start_frame: 0,
            status_every: 10,
            max_frames: Some(500),
            seek_frame: None,
            view: QgsBroadcastPlayerRunView::Compact,
            pace: QgsBroadcastPlayerLivePace::Logical,
            no_interactive: false,
            video_output: QgsBroadcastPlayerLiveVideoOutputMode::None,
            audio_output: QgsBroadcastPlayerLiveAudioOutputMode::None,
            output_dir: PathBuf::from("target/qgs-live-preview"),
            preview_every: 100,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum QgsBroadcastPlayerLiveInputCommand {
    Pause,
    Resume,
    Seek(u64),
    Status,
    Quit,
    Help,
    Stop,
    Unload,
    Invalid(String),
    Noop,
}

impl QgsBroadcastPlayerLiveInputCommand {
    fn parse(line: &str) -> Self {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return Self::Noop;
        }
        let mut parts = trimmed.split_whitespace();
        let Some(command) = parts.next() else {
            return Self::Noop;
        };
        let command = command.to_ascii_lowercase();
        match command.as_str() {
            "p" | "pause" => Self::Pause,
            "r" | "resume" | "play" => Self::Resume,
            "status" => Self::Status,
            "q" | "quit" => Self::Quit,
            "help" | "?" => Self::Help,
            "stop" => Self::Stop,
            "unload" => Self::Unload,
            "s" | "seek" => {
                let Some(frame) = parts.next() else {
                    return Self::Invalid("seek requires a frame number".to_string());
                };
                if parts.next().is_some() {
                    return Self::Invalid("seek accepts one frame number".to_string());
                }
                match frame.parse::<u64>() {
                    Ok(frame) => Self::Seek(frame),
                    Err(_) => Self::Invalid(format!("invalid seek frame: {frame}")),
                }
            }
            _ => Self::Invalid(format!("unknown live control: {trimmed}")),
        }
    }
}

fn qgs_live_output_coordinator(
    original_path: &Path,
    proxy_path: &Path,
    options: &QgsBroadcastPlayerLiveOptions,
) -> Result<LiveOutputCoordinator, Box<dyn std::error::Error>> {
    LiveOutputCoordinator::new(
        original_path,
        proxy_path,
        options.video_output.to_brick(),
        options.audio_output.to_brick(),
        &options.output_dir,
        options.preview_every,
    )
}

fn qgs_live_engine_picture_status(snapshot: &QgsQncRuntimeSnapshot) -> &'static str {
    match snapshot.picture_representation {
        Some(QgsPlaybackRepresentation::Original) => "original-picture",
        Some(QgsPlaybackRepresentation::Proxy) => "proxy-picture",
        None => "missing-picture",
    }
}

fn qgs_live_source_mode_label(snapshot: &QgsQncRuntimeSnapshot) -> &'static str {
    match snapshot.source_mode {
        Some(QgsInputPlanSourceMode::ProxyPreview) => "ProxyPreview",
        Some(QgsInputPlanSourceMode::OriginalMedia) => "OriginalMedia",
        None => "none",
    }
}

fn qgs_live_picture_label(snapshot: &QgsQncRuntimeSnapshot) -> &'static str {
    match snapshot.picture_representation {
        Some(QgsPlaybackRepresentation::Proxy) => "proxy MP4",
        Some(QgsPlaybackRepresentation::Original) => "original picture",
        None => "none",
    }
}

fn qgs_live_preview_frame_file_name(output_index: u64) -> String {
    format!("frame_{output_index:06}.ppm")
}

fn qgs_live_preview_latest_json(
    frame: u64,
    snapshot: &QgsQncRuntimeSnapshot,
    source_timebase: RationalRate,
    frame_file_name: &str,
    width: u32,
    height: u32,
    image_bytes: usize,
    checksum: u64,
) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"qgs.live-preview.latest.v1\",\n",
            "  \"current_frame\": {},\n",
            "  \"media_time\": \"{}\",\n",
            "  \"audio_sample_range\": \"{}\",\n",
            "  \"prepared_window\": \"{}\",\n",
            "  \"source_mode\": \"{}\",\n",
            "  \"video_source\": \"{}\",\n",
            "  \"audio_source\": \"original MXF\",\n",
            "  \"proxy_aac_authoritative\": false,\n",
            "  \"real_display\": \"no\",\n",
            "  \"visual_verified\": false,\n",
            "  \"private_path_exposed\": false,\n",
            "  \"artifact\": \"{}\",\n",
            "  \"width\": {},\n",
            "  \"height\": {},\n",
            "  \"image_bytes\": {},\n",
            "  \"checksum\": \"0x{:016x}\"\n",
            "}}\n"
        ),
        frame,
        qgs_live_snapshot_media_time_label(snapshot, source_timebase),
        qgs_optional_sample_range_label(snapshot.current_audio_sample_range),
        qgs_qnc_optional_frame_window_label(snapshot.prepared_window.as_ref()),
        qgs_live_source_mode_label(snapshot),
        qgs_live_picture_label(snapshot),
        json_escape(frame_file_name),
        width,
        height,
        image_bytes,
        checksum
    )
}

fn qgs_broadcast_player_run_report(
    original_path: &Path,
    proxy_path: &Path,
    options: QgsBroadcastPlayerRunOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let assembly = build_qgs_broadcast_player_assembly(original_path, proxy_path)?;
    let active_frames = assembly.active_frame_count();
    let requested_run_frames = options.frames.max(1);
    let first_run_frames = requested_run_frames.min(active_frames.saturating_sub(1));
    let seek_frame = qgs_resolve_run_seek_frame(options.seek_frame, active_frames);
    let second_run_frames = requested_run_frames.min(active_frames.saturating_sub(seek_frame + 1));
    let mut runtime = assembly.new_operational_runtime()?;

    match options.view {
        QgsBroadcastPlayerRunView::Compact => print_running_runtime_compact_header(&assembly),
        QgsBroadcastPlayerRunView::Detailed => print_running_runtime_detailed_header(&assembly),
    }

    let load = runtime.execute(QgsBroadcastPlayerCommand::LoadPreparedInput);
    match options.view {
        QgsBroadcastPlayerRunView::Compact => {
            print_running_runtime_compact_command("LOAD", &load.snapshot)
        }
        QgsBroadcastPlayerRunView::Detailed => {
            print_running_runtime_command("[LOAD]", &load.snapshot, load.accepted, load.reason)
        }
    }
    let prepare = runtime.execute(QgsBroadcastPlayerCommand::Prepare {
        active_range: Some((0, active_frames)),
    });
    match options.view {
        QgsBroadcastPlayerRunView::Compact => {
            println!(
                "PREPARE  status={:?} phase=input-ready",
                prepare.snapshot.operational_status
            );
        }
        QgsBroadcastPlayerRunView::Detailed => print_running_runtime_command_with_context(
            "[PREPARE INPUT]",
            &prepare.snapshot,
            prepare.accepted,
            prepare.reason,
            &[("meaning", qgs_running_prepare_input_meaning())],
        ),
    }
    let cue = runtime.execute(QgsBroadcastPlayerCommand::Cue { frame: 0 });
    match options.view {
        QgsBroadcastPlayerRunView::Compact => println!(
            "CUE      frame={} audio={}",
            qgs_optional_u64_label(cue.snapshot.current_frame),
            qgs_optional_sample_range_label(cue.snapshot.current_audio_sample_range)
        ),
        QgsBroadcastPlayerRunView::Detailed => print_running_runtime_command_with_context(
            "[CUE]",
            &cue.snapshot,
            cue.accepted,
            cue.reason,
            &[("phase", "Cued")],
        ),
    }
    let tick_prepare = runtime.execute(QgsBroadcastPlayerCommand::Tick { carrier_frame: 0 });
    match options.view {
        QgsBroadcastPlayerRunView::Compact => {
            print_running_runtime_compact_preroll("PREROLL", &tick_prepare.snapshot)
        }
        QgsBroadcastPlayerRunView::Detailed => print_running_runtime_command_with_context(
            "[PREROLL / PREPARE WINDOW]",
            &tick_prepare.snapshot,
            tick_prepare.accepted,
            tick_prepare.reason,
            &[("meaning", qgs_running_preroll_meaning())],
        ),
    }
    let play = runtime.execute(QgsBroadcastPlayerCommand::Play);
    match options.view {
        QgsBroadcastPlayerRunView::Compact => println!("\nPLAY"),
        QgsBroadcastPlayerRunView::Detailed => {
            print_running_runtime_command("[PLAY]", &play.snapshot, play.accepted, play.reason)
        }
    }

    for index in 0..first_run_frames {
        let snapshot = runtime.snapshot();
        match options.view {
            QgsBroadcastPlayerRunView::Compact => {
                print_running_runtime_compact_tick(index, &snapshot)
            }
            QgsBroadcastPlayerRunView::Detailed => {
                print_running_runtime_tick("[TICK", index, &snapshot)
            }
        }
        io::stdout().flush()?;
        runtime.execute(QgsBroadcastPlayerCommand::Tick { carrier_frame: 0 });
    }

    let pause = runtime.execute(QgsBroadcastPlayerCommand::Pause);
    match options.view {
        QgsBroadcastPlayerRunView::Compact => println!(
            "\nPAUSE    frame={} audio={}",
            qgs_optional_u64_label(pause.snapshot.current_frame),
            qgs_optional_sample_range_label(pause.snapshot.current_audio_sample_range)
        ),
        QgsBroadcastPlayerRunView::Detailed => {
            print_running_runtime_command("[PAUSE]", &pause.snapshot, pause.accepted, pause.reason)
        }
    }
    for index in 0..2 {
        let before = runtime.snapshot();
        let paused_tick = runtime.execute(QgsBroadcastPlayerCommand::Tick { carrier_frame: 0 });
        match options.view {
            QgsBroadcastPlayerRunView::Compact => {
                println!(
                    "PAUSED   position_changed={}",
                    yes_no(before.current_frame != paused_tick.snapshot.current_frame)
                );
            }
            QgsBroadcastPlayerRunView::Detailed => {
                println!("[TICK PAUSED {index:03}]");
                println!("status: {:?}", paused_tick.snapshot.operational_status);
                println!(
                    "frame: {}",
                    qgs_optional_u64_label(paused_tick.snapshot.current_frame)
                );
                println!(
                    "audio samples: {}",
                    qgs_optional_sample_range_label(
                        paused_tick.snapshot.current_audio_sample_range
                    )
                );
                println!(
                    "position changed: {}",
                    yes_no(before.current_frame != paused_tick.snapshot.current_frame)
                );
                println!();
            }
        }
    }

    let before_seek = runtime.snapshot();
    let seek = runtime.execute(QgsBroadcastPlayerCommand::Seek { frame: seek_frame });
    match options.view {
        QgsBroadcastPlayerRunView::Compact => println!(
            "\nSEEK     {} -> {} audio={}",
            qgs_optional_u64_label(before_seek.current_frame),
            qgs_optional_u64_label(seek.snapshot.current_frame),
            qgs_optional_sample_range_label(seek.snapshot.current_audio_sample_range)
        ),
        QgsBroadcastPlayerRunView::Detailed => {
            println!("[SEEK]");
            println!("accepted: {}", yes_no(seek.accepted));
            println!(
                "from frame: {}",
                qgs_optional_u64_label(before_seek.current_frame)
            );
            println!(
                "to frame: {}",
                qgs_optional_u64_label(seek.snapshot.current_frame)
            );
            println!(
                "audio samples: {}",
                qgs_optional_sample_range_label(seek.snapshot.current_audio_sample_range)
            );
            println!(
                "prepared window before refresh: {}",
                qgs_optional_frame_window_label(
                    before_seek.buffer_health.prepared_start_frame,
                    before_seek.buffer_health.prepared_end_frame_exclusive
                )
            );
            println!("reason: {}", seek.reason.unwrap_or("n/a"));
            println!();
        }
    }

    let prepare_after_seek = runtime.execute(QgsBroadcastPlayerCommand::Prepare {
        active_range: Some((0, active_frames)),
    });
    if options.view == QgsBroadcastPlayerRunView::Detailed {
        print_running_runtime_command_with_context(
            "[PREPARE INPUT AFTER SEEK]",
            &prepare_after_seek.snapshot,
            prepare_after_seek.accepted,
            prepare_after_seek.reason,
            &[("meaning", qgs_running_prepare_input_meaning())],
        );
    }
    let tick_prepare_after_seek =
        runtime.execute(QgsBroadcastPlayerCommand::Tick { carrier_frame: 0 });
    match options.view {
        QgsBroadcastPlayerRunView::Compact => {
            print_running_runtime_compact_preroll("PREROLL", &tick_prepare_after_seek.snapshot)
        }
        QgsBroadcastPlayerRunView::Detailed => print_running_runtime_command_with_context(
            "[PREROLL / PREPARE WINDOW AFTER SEEK]",
            &tick_prepare_after_seek.snapshot,
            tick_prepare_after_seek.accepted,
            tick_prepare_after_seek.reason,
            &[("meaning", qgs_running_preroll_meaning())],
        ),
    }
    let replay = runtime.execute(QgsBroadcastPlayerCommand::Play);
    match options.view {
        QgsBroadcastPlayerRunView::Compact => println!("\nPLAY"),
        QgsBroadcastPlayerRunView::Detailed => print_running_runtime_command(
            "[PLAY]",
            &replay.snapshot,
            replay.accepted,
            replay.reason,
        ),
    }

    for index in 0..second_run_frames {
        let snapshot = runtime.snapshot();
        match options.view {
            QgsBroadcastPlayerRunView::Compact => {
                print_running_runtime_compact_tick(index, &snapshot)
            }
            QgsBroadcastPlayerRunView::Detailed => {
                print_running_runtime_tick("[TICK", index, &snapshot)
            }
        }
        io::stdout().flush()?;
        runtime.execute(QgsBroadcastPlayerCommand::Tick { carrier_frame: 0 });
    }

    let stop = runtime.execute(QgsBroadcastPlayerCommand::Stop);
    match options.view {
        QgsBroadcastPlayerRunView::Compact => {
            print_running_runtime_compact_command("STOP", &stop.snapshot)
        }
        QgsBroadcastPlayerRunView::Detailed => {
            print_running_runtime_command("[STOP]", &stop.snapshot, stop.accepted, stop.reason);
            println!(
                "source loaded: {}",
                yes_no(stop.snapshot.player.readiness.source_loaded)
            );
            println!();
        }
    }

    let unload = runtime.execute(QgsBroadcastPlayerCommand::Unload);
    match options.view {
        QgsBroadcastPlayerRunView::Compact => {
            print_running_runtime_compact_command("UNLOAD", &unload.snapshot)
        }
        QgsBroadcastPlayerRunView::Detailed => {
            print_running_runtime_command(
                "[UNLOAD]",
                &unload.snapshot,
                unload.accepted,
                unload.reason,
            );
            println!(
                "source loaded: {}",
                yes_no(unload.snapshot.player.readiness.source_loaded)
            );
            println!();
        }
    }

    let final_snapshot = runtime.snapshot();
    println!("Final:");
    println!("run result: completed");
    println!("runtime behavior evidence: yes");
    println!(
        "private path exposed: {}",
        yes_no(final_snapshot.exposes_private_path())
    );
    println!(
        "real display: {}",
        final_snapshot.player.device_status.real_display_backend
    );
    println!(
        "visual verified: {}",
        yes_no(final_snapshot.player.device_status.visual_verified)
    );
    println!(
        "realtime verified: {}",
        yes_no(final_snapshot.player.device_status.realtime_verified)
    );
    println!(
        "audio production verified: {}",
        yes_no(
            final_snapshot
                .player
                .device_status
                .audio_device_production_verified
        )
    );
    println!(
        "A/V sync verified: {}",
        yes_no(final_snapshot.player.device_status.av_sync_verified)
    );
    println!("FramePresented real display claim: no");

    Ok(())
}

fn qgs_broadcast_player_live_report(
    original_path: &Path,
    proxy_path: &Path,
    options: QgsBroadcastPlayerLiveOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    qgs_live_apply_quiet_backend_environment(&options);
    let assembly = build_qgs_broadcast_player_assembly(original_path, proxy_path)?;
    let active_frames = assembly.active_frame_count();
    if active_frames == 0 {
        return Err("live runtime requires at least one active frame".into());
    }
    let start_frame = options.start_frame.min(active_frames - 1);
    let status_every = options.status_every.max(1);
    let source_timebase = assembly.input_plan().video_source.timebase;
    let frame_duration = source_timebase.frame_duration()?;
    let remaining_frames = active_frames.saturating_sub(start_frame).max(1);
    let run_limit = options
        .max_frames
        .unwrap_or(remaining_frames)
        .max(1)
        .min(remaining_frames);
    let mut live_outputs = qgs_live_output_coordinator(original_path, proxy_path, &options)?;
    let mut surface = QgsQncControlSurface::from_assembly(&assembly, "qgs-test-live-runtime-loop")?;
    let mut accepted = 0_u64;
    let mut rejected = 0_u64;
    let mut projected_events = 0_u64;
    let mut command_ordinal = 0_u64;
    let input_rx = qgs_live_spawn_input_thread(!options.no_interactive);
    let interactive_enabled = input_rx.is_some();

    match options.view {
        QgsBroadcastPlayerRunView::Compact => {
            print_broadcast_player_live_compact_header(&assembly, &options, &live_outputs)
        }
        QgsBroadcastPlayerRunView::Detailed => print_broadcast_player_live_detailed_header(
            &assembly,
            &options,
            &live_outputs,
            run_limit,
        ),
    }
    if interactive_enabled {
        print_broadcast_player_live_controls();
    }

    let load = qgs_live_send(
        &mut surface,
        &mut command_ordinal,
        QgsQncCommandKind::LoadPreparedInput,
        QgsQncCommandPayload::Empty,
    );
    qgs_live_count_reply(&load, &mut accepted, &mut rejected, &mut projected_events);
    match options.view {
        QgsBroadcastPlayerRunView::Compact => {
            print_broadcast_player_live_compact_command("LOAD", &load.snapshot)
        }
        QgsBroadcastPlayerRunView::Detailed => {
            print_qgs_live_reply("[LOAD]", &load);
        }
    }

    let prepare = qgs_live_send(
        &mut surface,
        &mut command_ordinal,
        QgsQncCommandKind::Prepare,
        QgsQncCommandPayload::Empty,
    );
    qgs_live_count_reply(
        &prepare,
        &mut accepted,
        &mut rejected,
        &mut projected_events,
    );
    match options.view {
        QgsBroadcastPlayerRunView::Compact => {
            println!(
                "PREPARE  status={:?} phase=input-ready",
                prepare.snapshot.status
            );
        }
        QgsBroadcastPlayerRunView::Detailed => print_qgs_live_reply("[PREPARE]", &prepare),
    }

    let cue = qgs_live_send(
        &mut surface,
        &mut command_ordinal,
        QgsQncCommandKind::Cue,
        QgsQncCommandPayload::Frame { frame: start_frame },
    );
    qgs_live_count_reply(&cue, &mut accepted, &mut rejected, &mut projected_events);
    match options.view {
        QgsBroadcastPlayerRunView::Compact => println!(
            "CUE      frame={} audio={}",
            qgs_optional_u64_label(cue.snapshot.current_frame),
            qgs_optional_sample_range_label(cue.snapshot.current_audio_sample_range)
        ),
        QgsBroadcastPlayerRunView::Detailed => print_qgs_live_reply("[CUE]", &cue),
    }

    let preroll = qgs_live_send(
        &mut surface,
        &mut command_ordinal,
        QgsQncCommandKind::Preroll,
        QgsQncCommandPayload::Preroll {
            target_frame: Some(start_frame),
        },
    );
    qgs_live_count_reply(
        &preroll,
        &mut accepted,
        &mut rejected,
        &mut projected_events,
    );
    match options.view {
        QgsBroadcastPlayerRunView::Compact => {
            print_broadcast_player_live_compact_preroll("PREROLL", &preroll.snapshot)
        }
        QgsBroadcastPlayerRunView::Detailed => print_qgs_live_reply("[PREROLL]", &preroll),
    }

    let play = qgs_live_send(
        &mut surface,
        &mut command_ordinal,
        QgsQncCommandKind::Play,
        QgsQncCommandPayload::Play { frame_count: None },
    );
    qgs_live_count_reply(&play, &mut accepted, &mut rejected, &mut projected_events);
    let mut live_ticking = play.accepted;
    match options.view {
        QgsBroadcastPlayerRunView::Compact => {
            println!("PLAY     status={:?}", play.snapshot.status)
        }
        QgsBroadcastPlayerRunView::Detailed => print_qgs_live_reply("[PLAY]", &play),
    }

    println!();
    println!("LIVE");
    let mut processed = 0_u64;
    let mut stop_reason = "end-of-range";
    let mut scripted_seek_done = false;
    while processed < run_limit {
        if let Some(rx) = &input_rx {
            let mut should_exit = None;
            while let Ok(line) = rx.try_recv() {
                if let Some(reason) = qgs_live_process_input_command(
                    &line,
                    &mut surface,
                    &mut live_outputs,
                    source_timebase,
                    active_frames,
                    options.view,
                    &mut live_ticking,
                    &mut command_ordinal,
                    &mut accepted,
                    &mut rejected,
                    &mut projected_events,
                ) {
                    should_exit = Some(reason);
                    break;
                }
            }
            if let Some(reason) = should_exit {
                stop_reason = reason;
                break;
            }
        }

        let snapshot = surface.snapshot();
        let output_result = live_outputs.emit_for_snapshot(
            processed,
            &snapshot,
            |parts, frame_name, image_bytes| {
                qgs_live_preview_latest_json(
                    parts.frame,
                    &snapshot,
                    source_timebase,
                    frame_name,
                    parts.width,
                    parts.height,
                    image_bytes,
                    parts.checksum,
                )
            },
        );
        if output_result.video_status == "window-closed" || live_outputs.preview_window_closed() {
            stop_reason = "preview-window-closed";
            break;
        }
        if processed == 0 || processed % status_every == 0 {
            match options.view {
                QgsBroadcastPlayerRunView::Compact => print_broadcast_player_live_compact_tick(
                    processed,
                    &snapshot,
                    source_timebase,
                    &output_result,
                ),
                QgsBroadcastPlayerRunView::Detailed => print_broadcast_player_live_detailed_tick(
                    processed,
                    &snapshot,
                    source_timebase,
                    &output_result,
                ),
            }
            io::stdout().flush()?;
        }

        if let Some(seek_frame) = options.seek_frame {
            if live_ticking && !scripted_seek_done && processed >= status_every {
                scripted_seek_done = true;
                let target = seek_frame.min(active_frames - 1);
                let pause = qgs_live_send(
                    &mut surface,
                    &mut command_ordinal,
                    QgsQncCommandKind::Pause,
                    QgsQncCommandPayload::Empty,
                );
                qgs_live_count_reply(&pause, &mut accepted, &mut rejected, &mut projected_events);
                let seek = qgs_live_send(
                    &mut surface,
                    &mut command_ordinal,
                    QgsQncCommandKind::Seek,
                    QgsQncCommandPayload::Frame { frame: target },
                );
                qgs_live_count_reply(&seek, &mut accepted, &mut rejected, &mut projected_events);
                if seek.accepted {
                    live_outputs.reset_audio_coverage();
                }
                let prepare_after_seek = qgs_live_send(
                    &mut surface,
                    &mut command_ordinal,
                    QgsQncCommandKind::Prepare,
                    QgsQncCommandPayload::Empty,
                );
                qgs_live_count_reply(
                    &prepare_after_seek,
                    &mut accepted,
                    &mut rejected,
                    &mut projected_events,
                );
                let preroll_after_seek = qgs_live_send(
                    &mut surface,
                    &mut command_ordinal,
                    QgsQncCommandKind::Preroll,
                    QgsQncCommandPayload::Preroll {
                        target_frame: Some(target),
                    },
                );
                qgs_live_count_reply(
                    &preroll_after_seek,
                    &mut accepted,
                    &mut rejected,
                    &mut projected_events,
                );
                let resume = qgs_live_send(
                    &mut surface,
                    &mut command_ordinal,
                    QgsQncCommandKind::Play,
                    QgsQncCommandPayload::Play { frame_count: None },
                );
                qgs_live_count_reply(&resume, &mut accepted, &mut rejected, &mut projected_events);
                live_ticking = resume.accepted;
                match options.view {
                    QgsBroadcastPlayerRunView::Compact => {
                        println!(
                            "SEEK     {} -> {} audio={}",
                            qgs_optional_u64_label(snapshot.current_frame),
                            qgs_optional_u64_label(seek.snapshot.current_frame),
                            qgs_optional_sample_range_label(
                                seek.snapshot.current_audio_sample_range
                            )
                        );
                        print_broadcast_player_live_compact_preroll(
                            "PREROLL",
                            &preroll_after_seek.snapshot,
                        );
                        println!("PLAY     status={:?}", resume.snapshot.status);
                    }
                    QgsBroadcastPlayerRunView::Detailed => {
                        print_qgs_live_reply("[PAUSE FOR SCRIPTED SEEK]", &pause);
                        print_qgs_live_reply("[SEEK]", &seek);
                        print_qgs_live_reply("[PREPARE AFTER SCRIPTED SEEK]", &prepare_after_seek);
                        print_qgs_live_reply("[PREROLL AFTER SEEK]", &preroll_after_seek);
                        print_qgs_live_reply("[RESUME]", &resume);
                    }
                }
            }
        }

        if !live_ticking {
            qgs_live_sleep_for_input(&options, frame_duration, interactive_enabled);
            continue;
        }

        let tick = qgs_live_send(
            &mut surface,
            &mut command_ordinal,
            QgsQncCommandKind::Play,
            QgsQncCommandPayload::Play {
                frame_count: Some(1),
            },
        );
        qgs_live_count_reply(&tick, &mut accepted, &mut rejected, &mut projected_events);
        if !tick.accepted {
            stop_reason = "rejected-command";
            break;
        }
        processed = processed.saturating_add(1);
        if tick
            .snapshot
            .current_frame
            .is_some_and(|frame| frame >= active_frames.saturating_sub(1))
        {
            stop_reason = "end-of-range";
            break;
        }
        if processed >= run_limit {
            stop_reason = "max-frames";
            break;
        }
        qgs_live_sleep_for_input(&options, frame_duration, interactive_enabled);
    }

    let stop = qgs_live_send(
        &mut surface,
        &mut command_ordinal,
        QgsQncCommandKind::Stop,
        QgsQncCommandPayload::Empty,
    );
    qgs_live_count_reply(&stop, &mut accepted, &mut rejected, &mut projected_events);
    if stop.accepted {
        live_outputs.reset_audio_coverage();
    }
    match options.view {
        QgsBroadcastPlayerRunView::Compact => {
            println!();
            println!("STOP     reason={stop_reason}");
            print_broadcast_player_live_compact_command("STOP", &stop.snapshot);
        }
        QgsBroadcastPlayerRunView::Detailed => {
            println!("[STOP] reason={stop_reason}");
            print_qgs_live_reply("[STOP RESULT]", &stop);
        }
    }

    let unload = qgs_live_send(
        &mut surface,
        &mut command_ordinal,
        QgsQncCommandKind::Unload,
        QgsQncCommandPayload::Empty,
    );
    qgs_live_count_reply(&unload, &mut accepted, &mut rejected, &mut projected_events);
    match options.view {
        QgsBroadcastPlayerRunView::Compact => {
            print_broadcast_player_live_compact_command("UNLOAD", &unload.snapshot)
        }
        QgsBroadcastPlayerRunView::Detailed => print_qgs_live_reply("[UNLOAD]", &unload),
    }

    let final_snapshot = surface.snapshot();
    println!();
    println!("Final:");
    println!("frames processed: {processed}");
    println!(
        "last frame: {}",
        qgs_optional_u64_label(stop.snapshot.current_frame)
    );
    println!("run result: {}", qgs_live_run_result_label(stop_reason));
    println!("stop reason: {stop_reason}");
    println!("video output: {}", live_outputs.video_label());
    if let Some(output_dir) = live_outputs.video_output_dir() {
        println!("video output dir: {}", output_dir.display());
    }
    println!("audio output: {}", live_outputs.audio_label());
    println!(
        "audio submitted: {}",
        yes_no(live_outputs.audio_submitted())
    );
    println!("audio submissions: {}", live_outputs.audio_submissions());
    println!("audio bytes copied: {}", live_outputs.audio_bytes_copied());
    println!("commands accepted: {accepted}");
    println!("commands rejected: {rejected}");
    println!("projected events: {projected_events}");
    println!(
        "private path exposed: {}",
        yes_no(final_snapshot.exposes_private_path())
    );
    println!(
        "real display: {}",
        final_snapshot.device_status.real_display_status
    );
    println!(
        "visual verified: {}",
        yes_no(final_snapshot.device_status.visual_verified)
    );
    println!(
        "realtime verified: {}",
        yes_no(final_snapshot.device_status.realtime_verified)
    );
    println!(
        "audio production verified: {}",
        yes_no(final_snapshot.device_status.audio_device_verified)
    );
    println!(
        "A/V sync verified: {}",
        yes_no(final_snapshot.device_status.av_sync_verified)
    );
    println!("FramePresented real display claim: no");

    Ok(())
}

fn qgs_live_apply_quiet_backend_environment(options: &QgsBroadcastPlayerLiveOptions) {
    if options.video_output != QgsBroadcastPlayerLiveVideoOutputMode::PreviewFiles {
        return;
    }
    if std::env::var_os("LIBVA_MESSAGING_LEVEL").is_none() {
        std::env::set_var("LIBVA_MESSAGING_LEVEL", "1");
    }
    if std::env::var_os("MESA_DEBUG").is_none() {
        std::env::set_var("MESA_DEBUG", "silent");
    }
}

fn qgs_live_spawn_input_thread(enabled: bool) -> Option<mpsc::Receiver<String>> {
    if !enabled {
        return None;
    }
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        for line in io::stdin().lock().lines() {
            let Ok(line) = line else {
                break;
            };
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    Some(rx)
}

fn qgs_live_process_input_command(
    line: &str,
    surface: &mut QgsQncControlSurface,
    live_outputs: &mut LiveOutputCoordinator,
    source_timebase: RationalRate,
    active_frames: u64,
    view: QgsBroadcastPlayerRunView,
    live_ticking: &mut bool,
    command_ordinal: &mut u64,
    accepted: &mut u64,
    rejected: &mut u64,
    projected_events: &mut u64,
) -> Option<&'static str> {
    let parsed = QgsBroadcastPlayerLiveInputCommand::parse(line);
    if !matches!(parsed, QgsBroadcastPlayerLiveInputCommand::Noop) {
        println!("> {}", line.trim());
    }
    match parsed {
        QgsBroadcastPlayerLiveInputCommand::Pause => {
            let reply = qgs_live_send(
                surface,
                command_ordinal,
                QgsQncCommandKind::Pause,
                QgsQncCommandPayload::Empty,
            );
            qgs_live_count_reply(&reply, accepted, rejected, projected_events);
            if reply.accepted {
                *live_ticking = false;
            }
            qgs_live_print_control_reply("PAUSE", &reply, view, source_timebase);
        }
        QgsBroadcastPlayerLiveInputCommand::Resume => {
            let reply = qgs_live_send(
                surface,
                command_ordinal,
                QgsQncCommandKind::Play,
                QgsQncCommandPayload::Play { frame_count: None },
            );
            qgs_live_count_reply(&reply, accepted, rejected, projected_events);
            if reply.accepted {
                *live_ticking = true;
            }
            qgs_live_print_control_reply("PLAY", &reply, view, source_timebase);
        }
        QgsBroadcastPlayerLiveInputCommand::Seek(frame) => {
            let target = frame.min(active_frames.saturating_sub(1));
            let was_ticking = *live_ticking;
            let before = surface.snapshot();
            if was_ticking {
                let pause = qgs_live_send(
                    surface,
                    command_ordinal,
                    QgsQncCommandKind::Pause,
                    QgsQncCommandPayload::Empty,
                );
                qgs_live_count_reply(&pause, accepted, rejected, projected_events);
                if view == QgsBroadcastPlayerRunView::Detailed {
                    print_qgs_live_reply("[PAUSE FOR INTERACTIVE SEEK]", &pause);
                }
            }
            let seek = qgs_live_send(
                surface,
                command_ordinal,
                QgsQncCommandKind::Seek,
                QgsQncCommandPayload::Frame { frame: target },
            );
            qgs_live_count_reply(&seek, accepted, rejected, projected_events);
            if seek.accepted {
                live_outputs.reset_audio_coverage();
            }
            let prepare = qgs_live_send(
                surface,
                command_ordinal,
                QgsQncCommandKind::Prepare,
                QgsQncCommandPayload::Empty,
            );
            qgs_live_count_reply(&prepare, accepted, rejected, projected_events);
            let preroll = qgs_live_send(
                surface,
                command_ordinal,
                QgsQncCommandKind::Preroll,
                QgsQncCommandPayload::Preroll {
                    target_frame: Some(target),
                },
            );
            qgs_live_count_reply(&preroll, accepted, rejected, projected_events);
            let resume = if was_ticking {
                let reply = qgs_live_send(
                    surface,
                    command_ordinal,
                    QgsQncCommandKind::Play,
                    QgsQncCommandPayload::Play { frame_count: None },
                );
                qgs_live_count_reply(&reply, accepted, rejected, projected_events);
                *live_ticking = reply.accepted;
                Some(reply)
            } else {
                *live_ticking = false;
                None
            };
            match view {
                QgsBroadcastPlayerRunView::Compact => {
                    println!(
                        "SEEK     {} -> {} audio={}",
                        qgs_optional_u64_label(before.current_frame),
                        qgs_optional_u64_label(seek.snapshot.current_frame),
                        qgs_optional_sample_range_label(seek.snapshot.current_audio_sample_range)
                    );
                    print_broadcast_player_live_compact_preroll("PREROLL", &preroll.snapshot);
                    if let Some(resume) = &resume {
                        println!("PLAY     status={:?}", resume.snapshot.status);
                    } else {
                        println!("PAUSED   awaiting play/resume");
                    }
                }
                QgsBroadcastPlayerRunView::Detailed => {
                    print_qgs_live_reply("[SEEK]", &seek);
                    print_qgs_live_reply("[PREPARE AFTER INTERACTIVE SEEK]", &prepare);
                    print_qgs_live_reply("[PREROLL AFTER INTERACTIVE SEEK]", &preroll);
                    if let Some(resume) = &resume {
                        print_qgs_live_reply("[RESUME AFTER INTERACTIVE SEEK]", resume);
                    }
                }
            }
        }
        QgsBroadcastPlayerLiveInputCommand::Status => {
            let reply = qgs_live_send(
                surface,
                command_ordinal,
                QgsQncCommandKind::Snapshot,
                QgsQncCommandPayload::Empty,
            );
            qgs_live_count_reply(&reply, accepted, rejected, projected_events);
            qgs_live_print_control_reply("STATUS", &reply, view, source_timebase);
        }
        QgsBroadcastPlayerLiveInputCommand::Quit => {
            println!("QUIT     requested");
            return Some("user-quit");
        }
        QgsBroadcastPlayerLiveInputCommand::Help => print_broadcast_player_live_controls(),
        QgsBroadcastPlayerLiveInputCommand::Stop => {
            let reply = qgs_live_send(
                surface,
                command_ordinal,
                QgsQncCommandKind::Stop,
                QgsQncCommandPayload::Empty,
            );
            qgs_live_count_reply(&reply, accepted, rejected, projected_events);
            if reply.accepted {
                *live_ticking = false;
                live_outputs.reset_audio_coverage();
            }
            qgs_live_print_control_reply("STOP", &reply, view, source_timebase);
        }
        QgsBroadcastPlayerLiveInputCommand::Unload => {
            println!("UNLOAD   requested; exiting for stop/unload cleanup");
            return Some("user-unload");
        }
        QgsBroadcastPlayerLiveInputCommand::Invalid(reason) => {
            println!("INVALID  {reason}");
        }
        QgsBroadcastPlayerLiveInputCommand::Noop => {}
    }
    None
}

fn qgs_live_print_control_reply(
    label: &str,
    reply: &qgs_media_runtime::QgsQncCommandReplyEnvelope,
    view: QgsBroadcastPlayerRunView,
    source_timebase: RationalRate,
) {
    match view {
        QgsBroadcastPlayerRunView::Compact => {
            println!(
                "{label:<8} accepted={} status={:?} frame={} t={} audio={} window={} buffer={}",
                yes_no(reply.accepted),
                reply.snapshot.status,
                qgs_optional_u64_label(reply.snapshot.current_frame),
                qgs_live_snapshot_media_time_label(&reply.snapshot, source_timebase),
                qgs_optional_sample_range_label(reply.snapshot.current_audio_sample_range),
                qgs_qnc_optional_frame_window_label(reply.snapshot.prepared_window.as_ref()),
                qgs_live_buffer_label(&reply.snapshot)
            );
            if let Some(reason) = reply.rejection_reason {
                println!("{label:<8} reason={reason}");
            }
        }
        QgsBroadcastPlayerRunView::Detailed => print_qgs_live_reply(&format!("[{label}]"), reply),
    }
}

fn qgs_live_preroll_ready(snapshot: &QgsQncRuntimeSnapshot) -> bool {
    snapshot.video_payload_ready
        && snapshot.audio_payload_ready
        && snapshot.buffer_status == "ready"
        && qgs_qnc_prepared_window_covers_current_frame(snapshot)
}

fn qgs_live_buffer_label(snapshot: &QgsQncRuntimeSnapshot) -> &'static str {
    if snapshot.buffer_status == "ready" && !qgs_qnc_prepared_window_covers_current_frame(snapshot)
    {
        "stale"
    } else {
        snapshot.buffer_status
    }
}

fn qgs_qnc_prepared_window_covers_current_frame(snapshot: &QgsQncRuntimeSnapshot) -> bool {
    let Some(frame) = snapshot.current_frame else {
        return snapshot.prepared_window.is_none();
    };
    let Some(window) = snapshot.prepared_window.as_ref() else {
        return false;
    };
    window
        .start_frame
        .zip(window.end_frame_exclusive)
        .is_some_and(|(start, end)| start <= frame && frame < end)
}

fn qgs_live_sleep_for_input(
    options: &QgsBroadcastPlayerLiveOptions,
    frame_duration: Duration,
    interactive_enabled: bool,
) {
    if options.pace == QgsBroadcastPlayerLivePace::Wall {
        std::thread::sleep(frame_duration);
    } else if interactive_enabled {
        std::thread::sleep(frame_duration.min(Duration::from_millis(20)));
    }
}

fn qgs_live_run_result_label(stop_reason: &str) -> &'static str {
    match stop_reason {
        "user-quit" | "user-unload" => "user-stopped",
        "rejected-command" => "stopped",
        _ => "completed",
    }
}

fn qgs_resolve_run_seek_frame(requested: Option<u64>, active_frames: u64) -> u64 {
    if active_frames <= 1 {
        return 0;
    }
    if let Some(frame) = requested {
        return frame.min(active_frames - 1);
    }
    if active_frames > 50 {
        50
    } else {
        active_frames
            .saturating_div(2)
            .max(1)
            .min(active_frames - 1)
    }
}

fn qgs_live_send(
    surface: &mut QgsQncControlSurface,
    ordinal: &mut u64,
    command: QgsQncCommandKind,
    payload: QgsQncCommandPayload,
) -> qgs_media_runtime::QgsQncCommandReplyEnvelope {
    *ordinal = ordinal.saturating_add(1);
    let request = QgsQncCommandRequestEnvelope::new(format!("live-{ordinal:06}"), command)
        .with_expected_generation(surface.generation())
        .with_payload(payload);
    surface.handle_command(request)
}

fn qgs_live_count_reply(
    reply: &qgs_media_runtime::QgsQncCommandReplyEnvelope,
    accepted: &mut u64,
    rejected: &mut u64,
    projected_events: &mut u64,
) {
    if reply.accepted {
        *accepted = accepted.saturating_add(1);
    } else {
        *rejected = rejected.saturating_add(1);
    }
    *projected_events = projected_events.saturating_add(reply.events.len() as u64);
}

fn print_broadcast_player_live_compact_header(
    assembly: &QgsBroadcastPlayerAssembly,
    options: &QgsBroadcastPlayerLiveOptions,
    outputs: &LiveOutputCoordinator,
) {
    println!("QGS Broadcast Player Live");
    println!("mode: ProxyPreview");
    let video_target = outputs
        .video_output_dir()
        .map(|path| {
            format!(
                "diagnostic preview files -> {} / latest.ppm",
                path.display()
            )
        })
        .unwrap_or_else(|| outputs.video_label().to_string());
    println!("video: {video_target}");
    println!(
        "audio: original MXF mono lanes -> {}",
        outputs.audio_label()
    );
    println!("device: preview-qnc-os");
    println!("display: NotImplemented / target Wayland+Vulkan");
    println!("video output mode: {}", options.video_output.label());
    println!("audio output mode: {}", options.audio_output.label());
    println!("preview every: {}", options.preview_every.max(1));
    println!("pace: {}", options.pace.label());
    println!("interactive stdin: {}", yes_no(!options.no_interactive));
    println!(
        "interactive control implemented: {}",
        yes_no(!options.no_interactive)
    );
    println!("private source path: hidden");
    println!("proxy AAC authoritative: no");
    println!("realtime verified: no");
    println!("A/V sync verified: no");
    println!(
        "source frames: {} @ {} fps",
        assembly.active_frame_count(),
        qgs_rate_label(assembly.input_plan().video_source.timebase)
    );
    println!();
}

fn print_broadcast_player_live_detailed_header(
    assembly: &QgsBroadcastPlayerAssembly,
    options: &QgsBroadcastPlayerLiveOptions,
    outputs: &LiveOutputCoordinator,
    run_limit: u64,
) {
    println!("QGS Broadcast Player Live");
    println!("-------------------------");
    println!("Runtime surface: live logical backend loop");
    println!("control layer: QgsQncControlSurface");
    println!("assembly: modular broadcast player runtime");
    println!("modules: {}", assembly.module_labels().join(", "));
    println!("mode: ProxyPreview");
    println!("video: proxy MP4");
    println!(
        "video output meaning: {}",
        match options.video_output {
            QgsBroadcastPlayerLiveVideoOutputMode::PreviewWindow => {
                "Wayland monitor window of engine pixels; not a display presenter and not VisualVerified"
            }
            QgsBroadcastPlayerLiveVideoOutputMode::PreviewFiles => "diagnostic preview files",
            QgsBroadcastPlayerLiveVideoOutputMode::None => "off",
        }
    );
    println!("audio: original MXF discrete mono lanes");
    println!("video output mode: {}", options.video_output.label());
    if let Some(output_dir) = outputs.video_output_dir() {
        println!("video output dir: {}", output_dir.display());
    }
    println!("audio output mode: {}", options.audio_output.label());
    println!("audio output label: {}", outputs.audio_label());
    println!("preview every: {}", options.preview_every.max(1));
    println!("original MXF audio authoritative: yes");
    println!("proxy AAC authoritative: no");
    println!("device policy: preview-qnc-os");
    println!("real display: NotImplemented");
    println!("display target: Wayland + Vulkan");
    println!("x11: legacy/non-target");
    println!("pace: {}", options.pace.label());
    println!(
        "wall-paced preview loop: {}",
        yes_no(options.pace == QgsBroadcastPlayerLivePace::Wall)
    );
    println!("run frame cap: {run_limit}");
    println!("start frame: {}", options.start_frame);
    println!("status every: {}", options.status_every.max(1));
    println!("interactive stdin: {}", yes_no(!options.no_interactive));
    println!(
        "interactive control implemented: {}",
        yes_no(!options.no_interactive)
    );
    println!("visual verified: no");
    println!("realtime verified: no");
    println!("audio production verified: no");
    println!("A/V sync verified: no");
    print_running_runtime_persistent_warnings();
    println!();
}

fn print_broadcast_player_live_controls() {
    println!(
        "Controls: p/pause, r/resume/play, s <frame>/seek <frame>, status, stop, q/quit, help"
    );
    println!(
        "Interactive control is stdin-only; no IPC, QNC UI, realtime, or device output claim."
    );
    println!();
}

fn print_broadcast_player_live_compact_command(label: &str, snapshot: &QgsQncRuntimeSnapshot) {
    println!(
        "{label:<8} status={:?} source={}",
        snapshot.status,
        yes_no(snapshot.source_loaded)
    );
}

fn print_broadcast_player_live_compact_preroll(label: &str, snapshot: &QgsQncRuntimeSnapshot) {
    println!(
        "{label:<8} ready={} window={} covers_current={}",
        yes_no(qgs_live_preroll_ready(snapshot)),
        qgs_qnc_optional_frame_window_label(snapshot.prepared_window.as_ref()),
        yes_no(qgs_qnc_prepared_window_covers_current_frame(snapshot))
    );
}

fn print_broadcast_player_live_compact_tick(
    tick: u64,
    snapshot: &QgsQncRuntimeSnapshot,
    source_timebase: RationalRate,
    output: &LiveOutputTickResult,
) {
    println!(
        "tick={tick:06} frame={:<4} t={} audio={:<18} window={:<9} buffer={} video={} audio={}",
        qgs_optional_u64_label(snapshot.current_frame),
        qgs_live_snapshot_media_time_label(snapshot, source_timebase),
        qgs_optional_sample_range_label(snapshot.current_audio_sample_range),
        qgs_qnc_optional_frame_window_label(snapshot.prepared_window.as_ref()),
        qgs_live_buffer_label(snapshot),
        output.video_status,
        output.audio_status
    );
}

fn print_broadcast_player_live_detailed_tick(
    tick: u64,
    snapshot: &QgsQncRuntimeSnapshot,
    source_timebase: RationalRate,
    output: &LiveOutputTickResult,
) {
    println!("[LIVE TICK {tick:06}]");
    println!("status: {:?}", snapshot.status);
    println!("frame: {}", qgs_optional_u64_label(snapshot.current_frame));
    println!(
        "media time: {}",
        qgs_live_snapshot_media_time_label(snapshot, source_timebase)
    );
    println!(
        "audio samples: {}",
        qgs_optional_sample_range_label(snapshot.current_audio_sample_range)
    );
    println!(
        "prepared window: {}",
        qgs_qnc_optional_frame_window_label(snapshot.prepared_window.as_ref())
    );
    println!("buffer: {}", snapshot.buffer_status);
    println!("video output: {}", output.video_status);
    println!("audio output: {}", output.audio_status);
    println!(
        "video payload ready: {}",
        yes_no(snapshot.video_payload_ready)
    );
    println!(
        "audio payload ready: {}",
        yes_no(snapshot.audio_payload_ready)
    );
    println!(
        "private path exposed: {}",
        yes_no(snapshot.exposes_private_path())
    );
    println!();
}

fn print_qgs_live_reply(label: &str, reply: &qgs_media_runtime::QgsQncCommandReplyEnvelope) {
    println!("{label}");
    println!("accepted: {}", yes_no(reply.accepted));
    println!(
        "generation: {} -> {}",
        reply.generation_before.0, reply.generation_after.0
    );
    println!("status: {:?}", reply.snapshot.status);
    println!(
        "frame: {}",
        qgs_optional_u64_label(reply.snapshot.current_frame)
    );
    println!(
        "media time: {}",
        qgs_media_time_label(reply.snapshot.media_time)
    );
    println!(
        "audio samples: {}",
        qgs_optional_sample_range_label(reply.snapshot.current_audio_sample_range)
    );
    println!(
        "prepared window: {}",
        qgs_qnc_optional_frame_window_label(reply.snapshot.prepared_window.as_ref())
    );
    println!("buffer: {}", reply.snapshot.buffer_status);
    println!("events: {}", reply.events.len());
    println!(
        "private path exposed: {}",
        yes_no(reply.exposes_private_path())
    );
    if let Some(reason) = reply.rejection_reason {
        println!("reason: {reason}");
    }
    println!();
}

fn qgs_media_time_label(value: Option<Duration>) -> String {
    value
        .map(qgs_format_media_time)
        .unwrap_or_else(|| "n/a".to_string())
}

fn qgs_live_snapshot_media_time_label(
    snapshot: &QgsQncRuntimeSnapshot,
    source_timebase: RationalRate,
) -> String {
    snapshot
        .media_time
        .or_else(|| {
            snapshot
                .current_frame
                .and_then(|frame| source_timebase.frame_offset(frame).ok())
        })
        .map(qgs_format_media_time)
        .unwrap_or_else(|| "n/a".to_string())
}

fn qgs_format_media_time(duration: Duration) -> String {
    let total_millis = duration.as_millis();
    let hours = total_millis / 3_600_000;
    let minutes = (total_millis / 60_000) % 60;
    let seconds = (total_millis / 1_000) % 60;
    let millis = total_millis % 1_000;
    format!("{hours:02}:{minutes:02}:{seconds:02}.{millis:03}")
}

fn print_running_runtime_compact_header(assembly: &QgsBroadcastPlayerAssembly) {
    println!("QGS Broadcast Player Runtime");
    println!("assembly: modular broadcast player runtime");
    println!("modules: {}", assembly.module_labels().join(", "));
    println!("mode: ProxyPreview");
    println!("video: proxy MP4");
    println!("audio: original MXF mono lanes");
    println!("device: preview-qnc-os");
    println!("display: NotImplemented / target Wayland+Vulkan");
    println!("x11: legacy/non-target");
    println!("private source path: hidden");
    println!("proxy AAC authoritative: no");
    println!();
}

fn print_running_runtime_detailed_header(assembly: &QgsBroadcastPlayerAssembly) {
    println!("QGS Broadcast Player Runtime");
    println!("----------------------------");
    println!("Runtime surface: running operational backend player");
    println!("assembly: modular broadcast player runtime");
    println!("modules: {}", assembly.module_labels().join(", "));
    println!("source: qnc://local media identity (private path hidden)");
    println!("mode: ProxyPreview");
    println!("video: proxy MP4");
    println!("audio: original MXF discrete mono lanes");
    println!("original MXF audio authoritative: yes");
    println!("proxy AAC authoritative: no");
    println!("device policy: preview-qnc-os");
    println!("real display: NotImplemented");
    println!("display target: Wayland + Vulkan");
    println!("x11: legacy/non-target");
    println!("visual verified: no");
    println!("realtime verified: no");
    println!("audio production verified: no");
    println!("A/V sync verified: no");
    print_running_runtime_persistent_warnings();
    println!();
}

fn print_running_runtime_compact_command(label: &str, snapshot: &QgsOperationalRuntimeSnapshot) {
    println!(
        "{label:<8} status={:?} source={}",
        snapshot.operational_status,
        yes_no(snapshot.player.readiness.source_loaded)
    );
}

fn print_running_runtime_compact_preroll(label: &str, snapshot: &QgsOperationalRuntimeSnapshot) {
    println!(
        "{label:<8} ready={} window={}",
        yes_no(
            snapshot.player.readiness.video_payload_ready
                && snapshot.player.readiness.audio_payload_ready
                && !snapshot.buffer_health.underrun
        ),
        qgs_optional_frame_window_label(
            snapshot.buffer_health.prepared_start_frame,
            snapshot.buffer_health.prepared_end_frame_exclusive
        )
    );
}

fn print_running_runtime_compact_tick(index: u64, snapshot: &QgsOperationalRuntimeSnapshot) {
    println!(
        "{index:03}  frame={:<4} audio={:<16} window={:<9} buffer={}",
        qgs_optional_u64_label(snapshot.current_frame),
        qgs_optional_sample_range_label(snapshot.current_audio_sample_range),
        qgs_optional_frame_window_label(
            snapshot.buffer_health.prepared_start_frame,
            snapshot.buffer_health.prepared_end_frame_exclusive
        ),
        qgs_running_buffer_label(snapshot)
    );
}

fn qgs_running_buffer_label(snapshot: &QgsOperationalRuntimeSnapshot) -> &'static str {
    if snapshot.buffer_health.underrun {
        "underrun"
    } else if snapshot.buffer_health.prepared_frame_count > 0 {
        "ready"
    } else {
        "empty"
    }
}

fn print_running_runtime_command(
    label: &str,
    snapshot: &QgsOperationalRuntimeSnapshot,
    accepted: bool,
    reason: Option<&'static str>,
) {
    print_running_runtime_command_with_context(label, snapshot, accepted, reason, &[]);
}

fn print_running_runtime_command_with_context(
    label: &str,
    snapshot: &QgsOperationalRuntimeSnapshot,
    accepted: bool,
    reason: Option<&'static str>,
    context: &[(&str, &'static str)],
) {
    println!("{label}");
    println!("accepted: {}", yes_no(accepted));
    for (key, value) in context {
        println!("{key}: {value}");
    }
    println!("status: {:?}", snapshot.operational_status);
    println!(
        "source loaded: {}",
        yes_no(snapshot.player.readiness.source_loaded)
    );
    println!("frame: {}", qgs_optional_u64_label(snapshot.current_frame));
    println!(
        "audio samples: {}",
        qgs_optional_sample_range_label(snapshot.current_audio_sample_range)
    );
    println!(
        "prepared window: {}",
        qgs_optional_frame_window_label(
            snapshot.buffer_health.prepared_start_frame,
            snapshot.buffer_health.prepared_end_frame_exclusive
        )
    );
    println!(
        "video payload ready: {}",
        yes_no(snapshot.player.readiness.video_payload_ready)
    );
    println!(
        "audio payload ready: {}",
        yes_no(snapshot.player.readiness.audio_payload_ready)
    );
    println!("reason: {}", reason.unwrap_or("n/a"));
    println!();
}

fn print_running_runtime_tick(
    label_prefix: &str,
    index: u64,
    snapshot: &QgsOperationalRuntimeSnapshot,
) {
    println!("{label_prefix} {index:03}]");
    println!("status: {:?}", snapshot.operational_status);
    println!("frame: {}", qgs_optional_u64_label(snapshot.current_frame));
    println!(
        "audio samples: {}",
        qgs_optional_sample_range_label(snapshot.current_audio_sample_range)
    );
    println!(
        "prepared window: {}",
        qgs_optional_frame_window_label(
            snapshot.buffer_health.prepared_start_frame,
            snapshot.buffer_health.prepared_end_frame_exclusive
        )
    );
    println!("buffer: {}", qgs_running_buffer_label(snapshot));
    println!("underrun: {}", yes_no(snapshot.buffer_health.underrun));
    println!("warnings: {}", qgs_running_warning_change_label(snapshot));
    println!();
}

fn print_running_runtime_persistent_warnings() {
    println!("Persistent warnings:");
    for line in qgs_running_runtime_persistent_warning_lines() {
        println!("- {line}");
    }
}

fn qgs_running_runtime_persistent_warning_lines() -> &'static [&'static str] {
    &[
        "RealDisplayUnavailable: expected until Wayland+Vulkan backend exists",
        "AudioOutputNotProductionVerified: PipeWire prototype not production-certified",
        "VisualVerificationUnavailable: no visual verification claim",
        "RealtimeVerificationUnavailable: no realtime certification",
        "AvSyncNotVerified: no A/V sync certification",
        "BackendNotImplemented: selected real backend is not implemented",
    ]
}

fn qgs_running_prepare_input_meaning() -> &'static str {
    "input validated / plan available / waiting for cue and preroll"
}

fn qgs_running_preroll_meaning() -> &'static str {
    "bounded prepared window ready for logical runtime ticks"
}

fn qgs_running_warning_change_label(snapshot: &QgsOperationalRuntimeSnapshot) -> String {
    let mut labels = snapshot
        .fault_snapshot
        .active_faults
        .iter()
        .filter(|fault| !qgs_running_is_persistent_warning_kind(fault.kind))
        .map(|fault| fault.kind.label())
        .collect::<Vec<_>>();
    labels.sort_unstable();
    labels.dedup();
    if labels.is_empty() {
        "unchanged".to_string()
    } else {
        labels.join(", ")
    }
}

fn qgs_running_is_persistent_warning_kind(
    kind: qgs_media_runtime::QgsOperationalFaultKind,
) -> bool {
    matches!(
        kind,
        qgs_media_runtime::QgsOperationalFaultKind::RealDisplayUnavailable
            | qgs_media_runtime::QgsOperationalFaultKind::AudioOutputNotProductionVerified
            | qgs_media_runtime::QgsOperationalFaultKind::BackendNotImplemented
            | qgs_media_runtime::QgsOperationalFaultKind::VisualVerificationUnavailable
            | qgs_media_runtime::QgsOperationalFaultKind::RealtimeVerificationUnavailable
            | qgs_media_runtime::QgsOperationalFaultKind::AvSyncNotVerified
    )
}

const QGS_BROADCAST_PLAYER_CONTROL_SESSION_DEFAULT_SCRIPT: &str =
    "load,prepare,cue:0,preroll,play:10,pause,seek:50,preroll,play:10,stop,unload";

#[derive(Clone, Debug, Eq, PartialEq)]
enum QgsControlSessionScriptCommand {
    Load,
    Prepare,
    Cue(u64),
    Preroll,
    Play(u64),
    Pause,
    Seek(u64),
    Stop,
    Unload,
    Snapshot,
    Tick(u64),
    Status,
}

impl QgsControlSessionScriptCommand {
    fn parse_script(script: &str) -> Result<Vec<Self>, String> {
        script
            .split(',')
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .map(Self::parse_one)
            .collect()
    }

    fn parse_one(part: &str) -> Result<Self, String> {
        if part == "load" {
            Ok(Self::Load)
        } else if part == "prepare" {
            Ok(Self::Prepare)
        } else if part == "preroll" {
            Ok(Self::Preroll)
        } else if part == "pause" {
            Ok(Self::Pause)
        } else if part == "stop" {
            Ok(Self::Stop)
        } else if part == "unload" {
            Ok(Self::Unload)
        } else if part == "snapshot" {
            Ok(Self::Snapshot)
        } else if part == "status" {
            Ok(Self::Status)
        } else if let Some(value) = part.strip_prefix("cue:") {
            Ok(Self::Cue(qgs_parse_control_script_u64(part, value)?))
        } else if let Some(value) = part.strip_prefix("seek:") {
            Ok(Self::Seek(qgs_parse_control_script_u64(part, value)?))
        } else if let Some(value) = part.strip_prefix("play:") {
            Ok(Self::Play(
                qgs_parse_control_script_u64(part, value)?.max(1),
            ))
        } else if let Some(value) = part.strip_prefix("tick:") {
            Ok(Self::Tick(
                qgs_parse_control_script_u64(part, value)?.max(1),
            ))
        } else {
            Err(format!("unsupported control script command: {part}"))
        }
    }

    fn label(&self) -> String {
        match self {
            Self::Load => "load".to_string(),
            Self::Prepare => "prepare".to_string(),
            Self::Cue(frame) => format!("cue:{frame}"),
            Self::Preroll => "preroll".to_string(),
            Self::Play(frames) => format!("play:{frames}"),
            Self::Pause => "pause".to_string(),
            Self::Seek(frame) => format!("seek:{frame}"),
            Self::Stop => "stop".to_string(),
            Self::Unload => "unload".to_string(),
            Self::Snapshot => "snapshot".to_string(),
            Self::Tick(frames) => format!("tick:{frames}"),
            Self::Status => "status".to_string(),
        }
    }
}

fn qgs_parse_control_script_u64(part: &str, value: &str) -> Result<u64, String> {
    value
        .parse::<u64>()
        .map_err(|_| format!("{part} expects an unsigned integer argument"))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct QgsControlSessionStateKey {
    status: QgsBroadcastPlayerStatus,
    source_loaded: bool,
    frame: Option<u64>,
    audio_range: Option<(u64, u64)>,
    prepared_start: Option<u64>,
    prepared_end: Option<u64>,
}

impl QgsControlSessionStateKey {
    fn from_snapshot(snapshot: &QgsQncRuntimeSnapshot) -> Self {
        Self {
            status: snapshot.status,
            source_loaded: snapshot.source_loaded,
            frame: snapshot.current_frame,
            audio_range: snapshot.current_audio_sample_range,
            prepared_start: snapshot
                .prepared_window
                .as_ref()
                .and_then(|window| window.start_frame),
            prepared_end: snapshot
                .prepared_window
                .as_ref()
                .and_then(|window| window.end_frame_exclusive),
        }
    }
}

fn qgs_broadcast_player_control_session_report(
    original_path: &Path,
    proxy_path: &Path,
    script: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let assembly = build_qgs_broadcast_player_assembly(original_path, proxy_path)?;
    let commands = QgsControlSessionScriptCommand::parse_script(script)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidInput, err))?;
    let mut surface = QgsQncControlSurface::from_assembly(&assembly, "qgs-test-control-session")?;
    let mut accepted_count = 0_usize;
    let mut rejected_count = 0_usize;
    let mut projected_event_count = 0_usize;

    println!("QGS QNC-Compatible Broadcast Player Control Session");
    println!("assembly: modular broadcast player runtime");
    println!("control surface: QgsQncControlSurface");
    println!("session id: {}", surface.session_id().0);
    println!("modules: {}", assembly.module_labels().join(", "));
    println!("mode: ProxyPreview");
    println!("video: proxy MP4");
    println!("audio: original MXF mono lanes");
    println!("device: preview-qnc-os");
    println!("display: NotImplemented / target Wayland+Vulkan");
    println!("x11: legacy/non-target");
    println!("private source path: hidden");
    println!("proxy AAC authoritative: no");
    println!();
    println!("SCRIPT");
    println!("{script}");
    println!();

    for (index, command) in commands.iter().enumerate() {
        let ordinal = index + 1;
        let label = command.label();
        let request = qgs_control_request_for_script_command(
            format!("cmd-{ordinal:04}"),
            surface.generation(),
            command,
        );
        let reply = surface.handle_command(request);
        println!(
            "[{ordinal:03}] id={} gen={}->{} {label}",
            reply.command_id.0, reply.generation_before.0, reply.generation_after.0
        );
        qgs_print_control_command_result(&reply.snapshot, reply.accepted, reply.rejection_reason);
        if !reply.accepted {
            qgs_print_rejection_detail_if_needed(&reply);
        }
        for (tick, event) in reply
            .events
            .iter()
            .filter(|event| event.event_kind == QgsQncControlEventKind::Ticked)
            .enumerate()
        {
            qgs_print_control_tick(tick as u64, event);
        }
        projected_event_count = projected_event_count.saturating_add(reply.events.len());
        qgs_count_control_result(reply.accepted, &mut accepted_count, &mut rejected_count);
        println!();
    }

    let stale_before = surface.snapshot();
    let stale_reply = surface.handle_command(
        QgsQncCommandRequestEnvelope::new("cmd-stale", QgsQncCommandKind::Pause)
            .with_expected_generation(qgs_media_runtime::QgsQncGeneration(
                surface.generation().0.saturating_add(999),
            )),
    );
    println!(
        "STALE id={} gen={}->{} accepted={} reason={} state_mutated={}",
        stale_reply.command_id.0,
        stale_reply.generation_before.0,
        stale_reply.generation_after.0,
        yes_no(stale_reply.accepted),
        stale_reply.rejection_reason.unwrap_or("<none>"),
        yes_no(stale_reply.state_mutated)
    );
    println!(
        "stale snapshot unchanged: {}",
        yes_no(
            QgsControlSessionStateKey::from_snapshot(&stale_before)
                == QgsControlSessionStateKey::from_snapshot(&stale_reply.snapshot)
        )
    );
    projected_event_count = projected_event_count.saturating_add(stale_reply.events.len());
    qgs_count_control_result(
        stale_reply.accepted,
        &mut accepted_count,
        &mut rejected_count,
    );
    println!();

    let final_snapshot = surface.snapshot();
    println!("Final:");
    println!("commands accepted: {accepted_count}");
    println!("commands rejected: {rejected_count}");
    println!("projected events: {projected_event_count}");
    println!("generation: {}", surface.generation().0);
    println!("final status: {:?}", final_snapshot.status);
    println!("source loaded: {}", yes_no(final_snapshot.source_loaded));
    println!(
        "prepared window: {}",
        qgs_qnc_optional_frame_window_label(final_snapshot.prepared_window.as_ref())
    );
    println!("run result: completed");
    println!(
        "private path exposed: {}",
        yes_no(final_snapshot.exposes_private_path())
    );
    println!(
        "real display: {}",
        final_snapshot.device_status.real_display_status
    );
    println!(
        "visual verified: {}",
        yes_no(final_snapshot.device_status.visual_verified)
    );
    println!(
        "realtime verified: {}",
        yes_no(final_snapshot.device_status.realtime_verified)
    );
    println!(
        "audio production verified: {}",
        yes_no(final_snapshot.device_status.audio_device_verified)
    );
    println!(
        "A/V sync verified: {}",
        yes_no(final_snapshot.device_status.av_sync_verified)
    );
    println!("FramePresented real display claim: no");

    Ok(())
}

fn qgs_control_request_for_script_command(
    command_id: String,
    generation: qgs_media_runtime::QgsQncGeneration,
    command: &QgsControlSessionScriptCommand,
) -> QgsQncCommandRequestEnvelope {
    let (kind, payload) = match *command {
        QgsControlSessionScriptCommand::Load => (
            QgsQncCommandKind::LoadPreparedInput,
            QgsQncCommandPayload::Empty,
        ),
        QgsControlSessionScriptCommand::Prepare => {
            (QgsQncCommandKind::Prepare, QgsQncCommandPayload::Empty)
        }
        QgsControlSessionScriptCommand::Cue(frame) => (
            QgsQncCommandKind::Cue,
            QgsQncCommandPayload::Frame { frame },
        ),
        QgsControlSessionScriptCommand::Preroll => (
            QgsQncCommandKind::Preroll,
            QgsQncCommandPayload::Preroll { target_frame: None },
        ),
        QgsControlSessionScriptCommand::Play(frame_count) => (
            QgsQncCommandKind::Play,
            QgsQncCommandPayload::Play {
                frame_count: Some(frame_count),
            },
        ),
        QgsControlSessionScriptCommand::Pause => {
            (QgsQncCommandKind::Pause, QgsQncCommandPayload::Empty)
        }
        QgsControlSessionScriptCommand::Seek(frame) => (
            QgsQncCommandKind::Seek,
            QgsQncCommandPayload::Frame { frame },
        ),
        QgsControlSessionScriptCommand::Stop => {
            (QgsQncCommandKind::Stop, QgsQncCommandPayload::Empty)
        }
        QgsControlSessionScriptCommand::Unload => {
            (QgsQncCommandKind::Unload, QgsQncCommandPayload::Empty)
        }
        QgsControlSessionScriptCommand::Snapshot | QgsControlSessionScriptCommand::Status => {
            (QgsQncCommandKind::Snapshot, QgsQncCommandPayload::Empty)
        }
        QgsControlSessionScriptCommand::Tick(frames) => (
            QgsQncCommandKind::Play,
            QgsQncCommandPayload::Play {
                frame_count: Some(frames),
            },
        ),
    };
    QgsQncCommandRequestEnvelope::new(command_id, kind)
        .with_expected_generation(generation)
        .with_payload(payload)
}

fn qgs_count_control_result(
    accepted: bool,
    accepted_count: &mut usize,
    rejected_count: &mut usize,
) {
    if accepted {
        *accepted_count = accepted_count.saturating_add(1);
    } else {
        *rejected_count = rejected_count.saturating_add(1);
    }
}

fn qgs_print_control_command_result(
    snapshot: &QgsQncRuntimeSnapshot,
    accepted: bool,
    reason: Option<&'static str>,
) {
    println!(
        "accepted={} status={:?} source={} frame={} audio={} window={} video_ready={} audio_ready={}",
        yes_no(accepted),
        snapshot.status,
        yes_no(snapshot.source_loaded),
        qgs_optional_u64_label(snapshot.current_frame),
        qgs_optional_sample_range_label(snapshot.current_audio_sample_range),
        qgs_qnc_optional_frame_window_label(snapshot.prepared_window.as_ref()),
        yes_no(snapshot.video_payload_ready),
        yes_no(snapshot.audio_payload_ready)
    );
    if let Some(reason) = reason {
        println!("reason={reason}");
    }
}

fn qgs_print_rejection_detail_if_needed(reply: &qgs_media_runtime::QgsQncCommandReplyEnvelope) {
    if let Some(fault) = &reply.fault {
        println!(
            "fault={} severity={} scope={} recovery={}",
            fault.fault_kind, fault.severity, fault.scope, fault.recovery_action
        );
    }
    println!("state_mutated={}", yes_no(reply.state_mutated));
    println!(
        "private path exposed: {}",
        yes_no(reply.exposes_private_path())
    );
}

fn qgs_print_control_tick(index: u64, event: &QgsQncControlEventEnvelope) {
    println!(
        "  {index:03} frame={} audio={} window={} buffer={}",
        qgs_optional_u64_label(event.frame),
        qgs_optional_sample_range_label(event.audio_sample_range),
        qgs_qnc_optional_frame_window_label(event.prepared_window.as_ref()),
        if event
            .prepared_window
            .as_ref()
            .is_some_and(|window| window.prepared_frame_count > 0)
        {
            "ready"
        } else {
            "empty"
        }
    );
}

fn qgs_qnc_optional_frame_window_label(
    window: Option<&qgs_media_runtime::QgsQncPreparedWindowView>,
) -> String {
    window.map_or_else(
        || "<none>".to_string(),
        |window| qgs_optional_frame_window_label(window.start_frame, window.end_frame_exclusive),
    )
}

fn qgs_device_backend_selection_report(
    policy: QgsDeviceSelectionPolicy,
) -> Result<(), Box<dyn std::error::Error>> {
    let selection = QgsDeviceBackendSelector::select(policy);
    println!("QGS Broadcast Player Device Backend Selection");
    println!("---------------------------------------------");
    print_device_backend_selection(&selection);
    println!("Production claims:");
    println!(
        "  real display ready: {}",
        yes_no(selection.real_display_ready)
    );
    println!("  visual verified: {}", yes_no(selection.visual_verified));
    println!(
        "  realtime verified: {}",
        yes_no(selection.realtime_verified)
    );
    println!(
        "  audio production verified: {}",
        yes_no(selection.audio_device_production_verified)
    );
    println!(
        "  A/V sync verified: {}",
        yes_no(selection.av_sync_verified)
    );
    println!("  FramePresented real backend: no");
    println!("  proxy AAC authoritative: no");
    Ok(())
}

fn qgs_qnc_command_label(command: &QgsQncCommandEnvelope) -> String {
    format!(
        "id={} gen={} source={} command={:?} summary={}",
        command.command_id,
        command
            .expected_generation
            .map(|generation| generation.0.to_string())
            .unwrap_or_else(|| "<none>".to_string()),
        command.public_source_uri.as_deref().unwrap_or("<none>"),
        command.command,
        command.payload_summary
    )
}

fn qgs_qnc_projected_event_label(event: &QgsQncEventEnvelope) -> String {
    format!(
        "#{:02} gen={} kind={:?} source={} evidence={:?} summary={}",
        event.sequence,
        event.generation,
        event.event.kind,
        event.public_source_uri.as_deref().unwrap_or("<none>"),
        event.event.evidence_status,
        event.event.public_payload_summary
    )
}

fn print_qgs_qnc_passive_view(view: &QgsQncPassiveView) {
    println!("  generation: {}", view.generation.0);
    println!("  transport_state: {:?}", view.transport.state);
    println!(
        "  current_public_source: {}",
        view.source.public_source_uri.as_deref().unwrap_or("<none>")
    );
    println!(
        "  active_range: {}",
        view.transport
            .active_range
            .map(qgs_transport_range_label)
            .unwrap_or_else(|| "<none>".to_string())
    );
    println!(
        "  cue_frame: {}",
        view.transport
            .cue
            .map(|cue| cue.frame.to_string())
            .unwrap_or_else(|| "<none>".to_string())
    );
    println!(
        "  prepared_anchor: {}",
        view.transport
            .prepared_anchor
            .map(|anchor| format!("frame={} revision={}", anchor.frame, anchor.revision.0))
            .unwrap_or_else(|| "<none>".to_string())
    );
    println!("  play_ready: {}", yes_no(view.readiness.play_ready));
    println!("  loaded_source_count: {}", view.loaded_source_count);
    println!(
        "  prepared_buffer: count={} window={} latest_discarded={}",
        view.prepared_buffer.prepared_frame_count,
        qgs_optional_frame_window_label(
            view.prepared_buffer.prepared_start_frame,
            view.prepared_buffer.prepared_end_frame_exclusive
        ),
        view.prepared_buffer.latest_discarded_frame_count
    );
    println!(
        "  private_path_exposed: {}",
        yes_no(view.private_path_exposed)
    );
}

fn print_broadcast_player_snapshot(snapshot: &QgsBroadcastPlayerSnapshot) {
    println!("  status: {:?}", snapshot.status);
    println!(
        "  current source URI: {}",
        snapshot.public_source_uri.as_deref().unwrap_or("none")
    );
    println!("  source mode: {:?}", snapshot.source_mode);
    println!(
        "  selected representation: {:?}",
        snapshot.selected_representation
    );
    println!(
        "  active range: {}",
        snapshot
            .position
            .active_range
            .map(|range| format!(
                "[{}..{}) frames / [{}..{}) samples",
                range.start_frame, range.end_frame, range.start_sample, range.end_sample
            ))
            .unwrap_or_else(|| "none".to_string())
    );
    println!(
        "  cue frame: {}",
        qgs_optional_u64_label(snapshot.position.cue_frame)
    );
    println!(
        "  current frame: {}",
        qgs_optional_u64_label(snapshot.position.current_frame)
    );
    println!(
        "  current audio sample range: {}",
        snapshot
            .position
            .current_audio_sample_range
            .map(|(start, end)| format!("[{start}..{end})"))
            .unwrap_or_else(|| "none".to_string())
    );
    println!(
        "  frame rate: {}",
        qgs_rate_label(snapshot.position.frame_rate)
    );
    println!(
        "  audio sample rate: {}",
        snapshot.position.audio_sample_rate
    );
    println!("  readiness:");
    println!(
        "    input valid: {}",
        yes_no(snapshot.readiness.input_valid)
    );
    println!(
        "    source loaded: {}",
        yes_no(snapshot.readiness.source_loaded)
    );
    println!(
        "    transport ready: {}",
        yes_no(snapshot.readiness.transport_ready)
    );
    println!("    cue ready: {}", yes_no(snapshot.readiness.cue_ready));
    println!(
        "    prepared window ready: {}",
        yes_no(snapshot.readiness.prepared_window_ready)
    );
    println!(
        "    video payload ready: {}",
        yes_no(snapshot.readiness.video_payload_ready)
    );
    println!(
        "    audio payload ready: {}",
        yes_no(snapshot.readiness.audio_payload_ready)
    );
    println!(
        "    presenter backend ready: {}",
        yes_no(snapshot.readiness.presenter_backend_ready)
    );
    println!(
        "    audio backend ready: {}",
        yes_no(snapshot.readiness.audio_backend_ready)
    );
    println!(
        "    real display ready: {}",
        yes_no(snapshot.readiness.real_display_ready)
    );
    println!(
        "    audio device verified: {}",
        yes_no(snapshot.readiness.audio_device_verified)
    );
    println!(
        "    visual verified: {}",
        yes_no(snapshot.readiness.visual_verified)
    );
    println!(
        "    realtime verified: {}",
        yes_no(snapshot.readiness.realtime_verified)
    );
    println!(
        "    A/V sync verified: {}",
        yes_no(snapshot.readiness.av_sync_verified)
    );
    println!(
        "    proxy AAC authoritative: {}",
        yes_no(snapshot.readiness.proxy_aac_authoritative)
    );
    println!(
        "    discrete mono audio: {}",
        yes_no(snapshot.readiness.discrete_mono_audio)
    );
    println!("  prepared window:");
    println!(
        "    frame start: {}",
        qgs_optional_u64_label(snapshot.prepared_window.prepared_start_frame)
    );
    println!(
        "    frame end exclusive: {}",
        qgs_optional_u64_label(snapshot.prepared_window.prepared_end_frame_exclusive)
    );
    println!(
        "    selected prepared frame: {}",
        qgs_optional_u64_label(snapshot.prepared_window.selected_prepared_frame)
    );
    println!(
        "    buffer count: {}",
        snapshot.prepared_window.prepared_frame_count
    );
    println!(
        "    discard count: {}",
        snapshot.prepared_window.latest_discarded_frame_count
    );
    println!(
        "    window policy: backward_keep={} forward_prepare={} max_prepared={}",
        snapshot.prepared_window.window_policy.backward_keep_frames,
        snapshot
            .prepared_window
            .window_policy
            .forward_prepare_frames,
        snapshot.prepared_window.window_policy.max_prepared_frames
    );
    println!("  device status:");
    print_device_backend_selection(&snapshot.device_status.selection);
    println!(
        "    test presenter boundary available: {}",
        yes_no(snapshot.device_status.test_presenter_boundary_available)
    );
    println!(
        "    file presenter diagnostic available: {}",
        yes_no(snapshot.device_status.file_presenter_diagnostic_available)
    );
    println!(
        "    GPU readback diagnostic available: {}",
        yes_no(snapshot.device_status.gpu_readback_diagnostic_available)
    );
    println!(
        "    real display backend: {}",
        snapshot.device_status.real_display_backend
    );
    println!(
        "    real audio backend: {}",
        snapshot.device_status.real_audio_backend
    );
    println!(
        "    QNC OS display target: {}",
        snapshot.device_status.qnc_os_display_target
    );
    println!("    X11 target: {}", snapshot.device_status.x11_target);
    println!(
        "    visual verified: {}",
        yes_no(snapshot.device_status.visual_verified)
    );
    println!(
        "    realtime verified: {}",
        yes_no(snapshot.device_status.realtime_verified)
    );
    println!(
        "    audio device production verified: {}",
        yes_no(snapshot.device_status.audio_device_production_verified)
    );
    println!(
        "    A/V sync verified: {}",
        yes_no(snapshot.device_status.av_sync_verified)
    );
    println!("  event counters:");
    println!(
        "    product events: {}",
        snapshot.event_counters.product_events
    );
    println!(
        "    lower-level events: {}",
        snapshot.event_counters.lower_level_events
    );
    println!(
        "    rejected commands: {}",
        snapshot.event_counters.rejected_commands
    );
}

fn print_device_backend_selection(selection: &QgsDeviceBackendSelection) {
    println!("  requested policy: {}", selection.policy.label());
    println!(
        "  selected video backend: {} ({})",
        selection.selected_video_backend.label(),
        selection.selected_video_availability.label()
    );
    println!(
        "  selected audio backend: {} ({})",
        selection.selected_audio_backend.label(),
        selection.selected_audio_availability.label()
    );
    println!(
        "  diagnostic video fallback: {}",
        qgs_optional_video_backend_label(selection.diagnostic_video_fallback)
    );
    println!(
        "  diagnostic audio fallback: {}",
        qgs_optional_audio_backend_label(selection.diagnostic_audio_fallback)
    );
    println!("  video candidates:");
    for candidate in &selection.video_candidates {
        println!(
            "    - {}: {} ({})",
            candidate.kind.label(),
            candidate.availability.label(),
            candidate.reason
        );
    }
    println!("  audio candidates:");
    for candidate in &selection.audio_candidates {
        println!(
            "    - {}: {} ({})",
            candidate.kind.label(),
            candidate.availability.label(),
            candidate.reason
        );
    }
    println!("  unavailable reasons:");
    for reason in &selection.unavailable_reasons {
        println!("    - {reason}");
    }
    println!(
        "  QNC OS display target: {}",
        selection.qnc_os_display_target
    );
    println!("  X11 target: {}", selection.x11_target);
    println!(
        "  real display ready: {}",
        yes_no(selection.real_display_ready)
    );
    println!(
        "  real audio backend ready: {}",
        yes_no(selection.real_audio_backend_ready)
    );
    println!("  visual verified: {}", yes_no(selection.visual_verified));
    println!(
        "  realtime verified: {}",
        yes_no(selection.realtime_verified)
    );
    println!(
        "  audio production verified: {}",
        yes_no(selection.audio_device_production_verified)
    );
    println!(
        "  A/V sync verified: {}",
        yes_no(selection.av_sync_verified)
    );
    println!(
        "  private path exposed: {}",
        yes_no(selection.exposes_private_path())
    );
}

fn print_operational_runtime_snapshot(snapshot: &QgsOperationalRuntimeSnapshot) {
    println!("  operational status: {:?}", snapshot.operational_status);
    println!(
        "  current frame: {}",
        qgs_optional_u64_label(snapshot.current_frame)
    );
    println!(
        "  current audio sample range: {}",
        qgs_optional_sample_range_label(snapshot.current_audio_sample_range)
    );
    println!("  completed: {}", yes_no(snapshot.completed));
    println!("  accepted commands: {}", snapshot.accepted_commands);
    println!("  rejected commands: {}", snapshot.rejected_commands);
    println!("  player status: {:?}", snapshot.player.status);
    println!(
        "  source URI: {}",
        snapshot
            .player
            .public_source_uri
            .as_deref()
            .unwrap_or("none")
    );
    println!("  source mode: {:?}", snapshot.player.source_mode);
    println!(
        "  active range: {}",
        snapshot
            .player
            .position
            .active_range
            .map(|range| format!(
                "[{}..{}) frames / [{}..{}) samples",
                range.start_frame, range.end_frame, range.start_sample, range.end_sample
            ))
            .unwrap_or_else(|| "none".to_string())
    );
    println!("  buffer health:");
    println!(
        "    prepared range: {}",
        qgs_optional_frame_window_label(
            snapshot.buffer_health.prepared_start_frame,
            snapshot.buffer_health.prepared_end_frame_exclusive
        )
    );
    println!(
        "    selected prepared frame: {}",
        qgs_optional_u64_label(snapshot.buffer_health.selected_prepared_frame)
    );
    println!(
        "    prepared count: {}",
        snapshot.buffer_health.prepared_frame_count
    );
    println!(
        "    low water: {}",
        yes_no(snapshot.buffer_health.low_water)
    );
    println!("    underrun: {}", yes_no(snapshot.buffer_health.underrun));
    println!(
        "    discarded count: {}",
        snapshot.buffer_health.discarded_count
    );
    println!(
        "    last prepared frame: {}",
        qgs_optional_u64_label(snapshot.buffer_health.last_prepared_frame)
    );
    println!(
        "    preparation pending: {}",
        yes_no(snapshot.buffer_health.preparation_pending)
    );
    println!("  faults/warnings:");
    if snapshot.faults.is_empty() {
        println!("    none");
    } else {
        for fault in &snapshot.faults {
            println!(
                "    - {} severity={} scope={} recoverable={} recovery={} status={} code={} reason={}",
                fault.kind.label(),
                fault.severity.label(),
                fault.scope.label(),
                yes_no(fault.recoverable),
                fault.recovery_action.label(),
                fault.recovery_status.label(),
                fault.qnc_safe_code,
                fault.reason
            );
        }
    }
    println!("  fault snapshot:");
    println!(
        "    active faults: {}",
        snapshot.fault_snapshot.active_faults.len()
    );
    println!(
        "    recoverable faults: {}",
        snapshot.fault_snapshot.recoverable_fault_count
    );
    println!(
        "    fatal faults: {}",
        snapshot.fault_snapshot.fatal_fault_count
    );
    println!(
        "    last fault: {}",
        snapshot
            .fault_snapshot
            .last_fault
            .as_ref()
            .map(|fault| fault.kind.label())
            .unwrap_or("none")
    );
    println!(
        "    recommended recovery: {}",
        snapshot
            .fault_snapshot
            .recommended_recovery_action
            .map(|action| action.label())
            .unwrap_or("none")
    );
    println!("    fault counters:");
    if snapshot.fault_snapshot.fault_counters.is_empty() {
        println!("      none");
    } else {
        for counter in &snapshot.fault_snapshot.fault_counters {
            println!("      {}: {}", counter.kind.label(), counter.count);
        }
    }
    println!(
        "    backend/device warnings: {}",
        snapshot.fault_snapshot.backend_device_warnings.len()
    );
    println!("  device selection:");
    print_device_backend_selection(&snapshot.player.device_status.selection);
    println!(
        "  private path exposed: {}",
        yes_no(snapshot.private_path_exposed)
    );
}

fn qgs_optional_sample_range_label(value: Option<(u64, u64)>) -> String {
    value
        .map(|(start, end)| format!("[{start}..{end})"))
        .unwrap_or_else(|| "none".to_string())
}

fn qgs_optional_video_backend_label(value: Option<QgsVideoPresenterBackendKind>) -> &'static str {
    value
        .map(QgsVideoPresenterBackendKind::label)
        .unwrap_or("none")
}

fn qgs_optional_audio_backend_label(value: Option<QgsAudioOutputBackendKind>) -> &'static str {
    value
        .map(QgsAudioOutputBackendKind::label)
        .unwrap_or("none")
}

fn qgs_optional_u64_label(value: Option<u64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "<none>".to_string())
}

fn qgs_optional_frame_window_label(start: Option<u64>, end: Option<u64>) -> String {
    match (start, end) {
        (Some(start), Some(end)) => format!("[{start}..{end})"),
        _ => "<none>".to_string(),
    }
}

fn qgs_transport_range_label(range: qgs_media_runtime::QgsTransportActiveRange) -> String {
    format!(
        "frames=[{}..{}) samples=[{}..{})",
        range.start_frame, range.end_frame, range.start_sample, range.end_sample
    )
}

fn qgs_rate_label(rate: RationalRate) -> String {
    format!("{}/{}", rate.numerator(), rate.denominator())
}

fn result_label<T, E>(result: &std::result::Result<T, E>) -> &'static str {
    if result.is_ok() {
        "ok"
    } else {
        "error"
    }
}

fn default_qgs_input_plan_queue_requirements() -> QgsInputPlanQueueRequirements {
    QgsInputPlanQueueRequirements {
        min_video_frames: 3,
        min_audio_ranges: 3,
        max_video_queue: 8,
        max_audio_queue: 8,
    }
}

fn build_qgs_broadcast_player_assembly(
    original_path: &Path,
    proxy_path: &Path,
) -> Result<QgsBroadcastPlayerAssembly, Box<dyn std::error::Error>> {
    let descriptor = build_qnc_prepared_input_descriptor(original_path, proxy_path)?;
    descriptor.validate()?;
    Ok(QgsBroadcastPlayerAssembly::from_prepared_descriptor(
        &descriptor,
        default_qgs_input_plan_queue_requirements(),
    )?)
}

fn build_qgs_qnc_prepared_input_like(
    original_path: &Path,
    proxy_path: &Path,
) -> Result<QgsQncPreparedInputLike, Box<dyn std::error::Error>> {
    let descriptor = build_qnc_prepared_input_descriptor(original_path, proxy_path)?;
    Ok(QgsQncPreparedInputLike {
        contract_version: "qnc-qgs-contract.m2-v1-draft".to_string(),
        descriptor_revision: 1,
        source_identity: QgsQncSourceIdentityLike {
            public_source_uri: descriptor.identity.source_record_uri.clone(),
            source_id: descriptor.identity.clip_id.clone(),
            workspace_db_uri: descriptor.identity.workspace_db_uri.clone(),
            original_media_id: Some(descriptor.binding.original_media_uri.clone()),
            proxy_media_id: descriptor.binding.proxy_media_uri.clone(),
        },
        private_binding: QgsQncPrivateBindingLike {
            original_binding_ref: format!("binding://original/{}", descriptor.identity.clip_id),
            proxy_binding_ref: Some(format!("binding://proxy/{}", descriptor.identity.clip_id)),
            original_bound: descriptor.binding.private_original_path_bound,
            proxy_bound: descriptor.binding.private_proxy_path_bound,
            private_path_exposed: false,
        },
        playback_input: QgsQncPlaybackInputLike::ProxyIfAvailable,
        selected_picture_representation: QgsQncRepresentationLike::Proxy,
        authoritative_audio_representation: QgsQncAudioRepresentationLike::Original,
        source_mode: QgsInputPlanSourceMode::ProxyPreview,
        active_range: Some(QgsQncFrameRangeLike {
            start_frame: 0,
            end_frame_exclusive: descriptor.layout.original_video.duration_frames,
        }),
        project_audio: QgsQncProjectAudioLike {
            channels: descriptor.project_audio_channels,
            sample_rate_hz: descriptor.project_audio_sample_rate,
        },
        stream_layout: QgsQncStreamLayoutLike {
            original_video: QgsQncVideoInputLike {
                timebase: descriptor.layout.original_video.timebase,
                duration_frames: descriptor.layout.original_video.duration_frames,
                duration: descriptor.layout.original_video.duration,
            },
            proxy_video: descriptor
                .layout
                .proxy_video
                .map(|video| QgsQncVideoInputLike {
                    timebase: video.timebase,
                    duration_frames: video.duration_frames,
                    duration: video.duration,
                }),
            audio_sample_rate: descriptor.audio_layout.sample_rate,
            audio_bit_depth: descriptor.audio_layout.bit_depth,
            audio_channels: descriptor
                .audio_layout
                .channels
                .iter()
                .map(|lane| QgsQncAudioChannelLike {
                    lane_index: lane.lane_index,
                    source_track_index: lane.track_id,
                    source_channel_index: lane.channel_index,
                    channel_kind: QgsQncAudioChannelKindLike::Mono,
                    authoritative: true,
                })
                .collect(),
        },
        original_proxy_timing_compatible: Some(
            descriptor.layout.proxy_original_timing_compatible(),
        ),
        proxy_aac_authoritative: false,
    })
}

fn build_qnc_prepared_input_descriptor(
    original_path: &Path,
    proxy_path: &Path,
) -> Result<QgsPreparedInputDescriptor, Box<dyn std::error::Error>> {
    let audio_index = open_pcm_audio_index(original_path)?;
    let proxy = Mp4Source::open(proxy_path)?;
    let proxy_video = proxy
        .video
        .as_ref()
        .ok_or("proxy has no H.264 video track")?;
    let audio_tracks = sorted_audio_tracks(&audio_index);
    if audio_tracks.is_empty() {
        return Err("original MXF has no audio tracks".into());
    }
    let sample_rate = uniform_original_audio_sample_rate(&audio_tracks)?;
    let bit_depth = uniform_original_audio_bit_depth(&audio_tracks)?;
    let original_duration_samples = original_audio_duration_samples(&audio_index);
    if original_duration_samples == 0 {
        return Err("original MXF audio duration is missing".into());
    }
    let source_duration = duration_from_audio_samples(original_duration_samples, sample_rate)?;
    let original_rate =
        original_video_timebase_from_index(&audio_index).unwrap_or(RationalRate::new(
            u64::from(proxy_video.frame_rate.numerator),
            u64::from(proxy_video.frame_rate.denominator),
        )?);
    let original_duration_frames =
        duration_samples_to_frames(original_duration_samples, sample_rate, original_rate)?;
    let proxy_duration = duration_from_units(
        proxy_video.duration_units,
        u64::from(proxy_video.timescale),
        1,
    )?;
    let proxy_timing = QgsPreparedVideoTiming {
        timebase: RationalRate::new(
            u64::from(proxy_video.frame_rate.numerator),
            u64::from(proxy_video.frame_rate.denominator),
        )?,
        duration_frames: proxy_video.samples.len() as u64,
        duration: proxy_duration,
    };
    let original_timing = QgsPreparedVideoTiming {
        timebase: original_rate,
        duration_frames: original_duration_frames,
        duration: source_duration,
    };
    let association_status = if proxy_timing.timebase == original_timing.timebase
        && proxy_timing.duration_frames == original_timing.duration_frames
    {
        QgsOriginalProxyAssociationStatus::TimingCompatible
    } else {
        QgsOriginalProxyAssociationStatus::TimingMismatch
    };
    let clip_id = sanitized_source_stem(original_path);
    let channels = audio_tracks
        .iter()
        .enumerate()
        .map(|(lane, track)| {
            let audio = track.audio.as_ref().ok_or("audio descriptor missing")?;
            if audio.channels != Some(1) {
                return Err("Phase 22 expects discrete mono original audio tracks".into());
            }
            Ok(QgsPreparedAudioChannel {
                track_id: track.id.0,
                lane_index: u16::try_from(lane)?,
                channel_index: 0,
            })
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;

    Ok(QgsPreparedInputDescriptor {
        contract_version: "qgs.phase22.qnc-prepared-input.v1".to_string(),
        identity: QgsPreparedSourceIdentity {
            clip_id: clip_id.clone(),
            workspace_db_uri: format!("qnc://local/db/project_workspace/{clip_id}"),
            source_record_uri: format!("qnc://local/source/{clip_id}"),
        },
        binding: QgsPreparedMediaBinding {
            original_media_uri: format!("qnc://local/media/original/{clip_id}"),
            proxy_media_uri: Some(format!("qnc://local/media/proxy/{clip_id}")),
            private_original_path_bound: original_path.is_file(),
            private_proxy_path_bound: proxy_path.is_file(),
            association_status,
        },
        selected_picture: QgsPlaybackRepresentation::Proxy,
        authoritative_audio: QgsAudioRepresentation::Original,
        project_audio_channels: u16::try_from(channels.len())?,
        project_audio_sample_rate: sample_rate,
        layout: QgsPreparedStreamLayout {
            original_video: original_timing,
            proxy_video: Some(proxy_timing),
            audio_sample_rate: sample_rate,
        },
        audio_layout: QgsPreparedAudioLayout {
            representation: QgsAudioRepresentation::Original,
            sample_rate,
            bit_depth,
            channels,
            proxy_aac_authoritative: false,
        },
    })
}

fn sorted_audio_tracks(index: &PcmAudioIndex) -> Vec<&qgs_mxf::MxfTrack> {
    let mut tracks = index
        .tracks
        .iter()
        .filter(|track| track.kind == TrackKind::Audio)
        .collect::<Vec<_>>();
    tracks.sort_by_key(|track| track.id.0);
    tracks
}

fn uniform_original_audio_sample_rate(
    tracks: &[&qgs_mxf::MxfTrack],
) -> Result<u32, Box<dyn std::error::Error>> {
    let rates = tracks
        .iter()
        .map(|track| {
            let rate = track
                .audio
                .as_ref()
                .and_then(|audio| audio.sample_rate)
                .ok_or("audio sample rate missing")?;
            rational_to_u32(rate)
        })
        .collect::<Result<Vec<_>, _>>()?;
    uniform_value(rates.as_slice()).ok_or_else(|| "original audio sample rates differ".into())
}

fn uniform_original_audio_bit_depth(
    tracks: &[&qgs_mxf::MxfTrack],
) -> Result<u8, Box<dyn std::error::Error>> {
    let depths = tracks
        .iter()
        .map(|track| {
            track
                .audio
                .as_ref()
                .and_then(|audio| audio.bit_depth)
                .ok_or("audio bit depth missing")
        })
        .collect::<Result<Vec<_>, _>>()?;
    uniform_value(depths.as_slice()).ok_or_else(|| "original audio bit depths differ".into())
}

fn original_video_timebase_from_index(index: &PcmAudioIndex) -> Option<RationalRate> {
    index
        .tracks
        .iter()
        .find(|track| track.kind == TrackKind::Video)
        .and_then(|track| track.edit_rate)
        .and_then(|rate| {
            RationalRate::new(u64::from(rate.numerator), u64::from(rate.denominator)).ok()
        })
}

fn original_audio_duration_samples(index: &PcmAudioIndex) -> u64 {
    index
        .audio
        .iter()
        .map(|entry| {
            entry
                .start_sample
                .saturating_add(u64::from(entry.sample_count))
        })
        .max()
        .unwrap_or(0)
}

fn duration_samples_to_frames(
    samples: u64,
    sample_rate: u32,
    frame_rate: RationalRate,
) -> Result<u64, Box<dyn std::error::Error>> {
    if sample_rate == 0 {
        return Err("invalid audio sample rate".into());
    }
    let numerator = u128::from(samples)
        .checked_mul(u128::from(frame_rate.numerator()))
        .ok_or("duration frame overflow")?;
    let denominator = u128::from(sample_rate)
        .checked_mul(u128::from(frame_rate.denominator()))
        .ok_or("duration frame overflow")?;
    if denominator == 0 {
        return Err("invalid frame denominator".into());
    }
    let rounded = numerator
        .checked_add(denominator / 2)
        .ok_or("duration frame overflow")?
        / denominator;
    Ok(u64::try_from(rounded)?)
}

#[derive(Clone, Debug)]
struct OriginalAudioReport {
    timeline: AudioTimeline,
    total_channels: u16,
    uniform_sample_rate: Option<u32>,
    uniform_bit_depth: Option<u8>,
    audio_duration: Duration,
    proxy_video_duration: Duration,
    duration_delta: Duration,
    audio_queue_capacity: usize,
    audio_queue_peak: usize,
    audio_queue_backpressure: u64,
    audio_packets_recorded: usize,
    audio_payload_packets_indexed: usize,
    audio_timestamps_monotonic: bool,
    max_selected_video_outside_audio: Duration,
    usable_as_master_clock: bool,
}

fn build_original_audio_report(
    original: &MediaSource,
    proxy_video: &qgs_mp4::Mp4VideoTrack,
) -> Result<OriginalAudioReport, Box<dyn std::error::Error>> {
    let edit_units = original
        .duration
        .ok_or("original MXF duration is missing")?;
    let edit_rate = original
        .edit_rate
        .ok_or("original MXF edit rate is missing")?;
    let audio_duration = duration_from_units(
        edit_units,
        u64::from(edit_rate.numerator),
        u64::from(edit_rate.denominator),
    )?;
    let proxy_video_duration = duration_from_units(
        proxy_video.duration_units,
        u64::from(proxy_video.timescale),
        1,
    )?;
    let duration_delta = duration_abs_delta(audio_duration, proxy_video_duration);

    let audio_tracks = original
        .tracks
        .iter()
        .filter(|track| track.kind == TrackKind::Audio)
        .collect::<Vec<_>>();
    if audio_tracks.is_empty() {
        return Err("original MXF has no audio tracks".into());
    }

    let mut tracks = Vec::with_capacity(audio_tracks.len());
    let mut total_channels = 0_u16;
    for (index, track) in audio_tracks.iter().enumerate() {
        let audio = track
            .audio
            .as_ref()
            .ok_or("original MXF audio track missing descriptor")?;
        let sample_rate = rational_to_u32(audio.sample_rate.ok_or("audio sample rate missing")?)?;
        let channels = audio.channels.ok_or("audio channel count missing")?;
        let bit_depth = audio.bit_depth;
        let sample_count = audio_samples_for_duration(audio_duration, sample_rate)?;
        let format = AudioFormat {
            sample_rate,
            channels,
            sample_format: bit_depth
                .map(|bits_per_sample| AudioSampleFormat::PcmSignedInt { bits_per_sample })
                .unwrap_or(AudioSampleFormat::Unknown),
        }
        .validate()?;
        total_channels = total_channels
            .checked_add(channels)
            .ok_or("audio channel count overflow")?;
        tracks.push(OriginalAudioTrack {
            track_id: track.id.0,
            channel_index: u16::try_from(index).map_err(|_| "audio channel index overflow")?,
            format,
            sample_count: Some(sample_count),
            duration: audio_duration,
        });
    }
    let timeline = AudioTimeline::new(tracks)?;
    let uniform_sample_rate = uniform_value(
        timeline
            .tracks
            .iter()
            .map(|track| track.format.sample_rate)
            .collect::<Vec<_>>()
            .as_slice(),
    );
    let uniform_bit_depth = {
        let depths = timeline
            .tracks
            .iter()
            .map(|track| match track.format.sample_format {
                AudioSampleFormat::PcmSignedInt { bits_per_sample } => Some(bits_per_sample),
                AudioSampleFormat::Unknown => None,
            })
            .collect::<Vec<_>>();
        uniform_value(depths.as_slice()).flatten()
    };

    let audio_queue_capacity = timeline.tracks.len().max(1);
    let mut queue = BoundedQueue::new(audio_queue_capacity)?;
    for track in &timeline.tracks {
        let sample_count = track
            .sample_count
            .and_then(|value| u32::try_from(value).ok())
            .ok_or("audio timing packet sample count overflow")?;
        queue
            .try_push(AudioTimingPacket {
                track_id: track.track_id,
                start: Duration::ZERO,
                duration: track.duration,
                sample_count,
                has_payload: false,
            })
            .map_err(|_| "bounded audio timing queue unexpectedly full")?;
    }
    let stats = queue.stats();
    let mut sink = TestAudioSink::new();
    while let Some(packet) = queue.pop_front() {
        sink.record(packet);
    }

    let preview_rate = RationalRate::new(25, 1)?;
    let selected_frame_count =
        ProxyPlaybackProfile::Journalist50iPreview.selected_frame_count(proxy_video.samples.len());
    let selected_timestamps = (0..selected_frame_count)
        .map(|position| preview_rate.frame_offset(position as u64))
        .collect::<Result<Vec<_>, _>>()?;
    let max_selected_video_outside_audio =
        max_video_timestamp_outside_audio_range(&selected_timestamps, timeline.duration);

    Ok(OriginalAudioReport {
        timeline,
        total_channels,
        uniform_sample_rate,
        uniform_bit_depth,
        audio_duration,
        proxy_video_duration,
        duration_delta,
        audio_queue_capacity,
        audio_queue_peak: stats.peak_depth,
        audio_queue_backpressure: stats.backpressure_events,
        audio_packets_recorded: sink.packets().len(),
        audio_payload_packets_indexed: original.index.audio.len(),
        audio_timestamps_monotonic: sink.monotonic(),
        max_selected_video_outside_audio,
        usable_as_master_clock: max_selected_video_outside_audio == Duration::ZERO,
    })
}

fn print_original_audio_report(report: &OriginalAudioReport) {
    println!("Audio source: original MXF");
    println!("Video preview source: proxy MP4");
    println!("Proxy audio: diagnostic/fallback only");
    println!(
        "Original audio: tracks={} total_channels={} sample_rate={} bit_depth={:?}",
        report.timeline.tracks.len(),
        report.total_channels,
        report
            .uniform_sample_rate
            .map(|value| value.to_string())
            .unwrap_or_else(|| "mixed".to_string()),
        report.uniform_bit_depth
    );
    println!(
        "Original audio timeline: duration={:.3}s packets={} queue_capacity={} queue_peak={} backpressure={} monotonic={}",
        report.audio_duration.as_secs_f64(),
        report.audio_packets_recorded,
        report.audio_queue_capacity,
        report.audio_queue_peak,
        report.audio_queue_backpressure,
        yes_no(report.audio_timestamps_monotonic)
    );
    println!(
        "Original/proxy A/V duration: original_audio={:.3}s proxy_video={:.3}s delta={:.3} ms",
        report.audio_duration.as_secs_f64(),
        report.proxy_video_duration.as_secs_f64(),
        report.duration_delta.as_secs_f64() * 1000.0
    );
}

#[derive(Clone, Debug, Default)]
struct PcmTrackPayloadStats {
    channel_index: u16,
    packets: u64,
    samples: u64,
    payload_bytes: u64,
    first_start: Option<Duration>,
    last_end: Option<Duration>,
    last_start: Option<Duration>,
    monotonic: bool,
}

#[derive(Clone, Debug)]
struct PcmPayloadStats {
    total_packets: u64,
    total_payload_bytes: u64,
    tracks: BTreeMap<u32, PcmTrackPayloadStats>,
}

#[derive(Clone, Debug, Default)]
struct PcmTrackBlockStats {
    channel_index: u16,
    blocks: u64,
    samples: u64,
    payload_bytes: u64,
    first_start: Option<Duration>,
    last_end: Option<Duration>,
    last_start: Option<Duration>,
    monotonic: bool,
    gaps: u64,
    overlaps: u64,
}

#[derive(Clone, Debug)]
struct PcmBlockStats {
    queue_capacity: usize,
    queue_peak: usize,
    queue_backpressure: u64,
    total_blocks: u64,
    total_payload_bytes: u64,
    tracks: BTreeMap<u32, PcmTrackBlockStats>,
}

fn original_audio_extract(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let source = MediaSource::parse(&bytes)?;
    let audio_tracks = source
        .tracks
        .iter()
        .filter(|track| track.kind == TrackKind::Audio)
        .collect::<Vec<_>>();
    if audio_tracks.is_empty() {
        return Err("original MXF has no audio tracks".into());
    }
    if source.index.audio.is_empty() {
        return Err("original MXF has no indexed PCM audio payload packets".into());
    }

    let block_queue_capacity = 8_usize;
    let mut block_queue = BoundedQueue::new(block_queue_capacity)?;
    let mut stats = PcmPayloadStats {
        total_packets: 0,
        total_payload_bytes: 0,
        tracks: BTreeMap::new(),
    };
    let mut block_stats = PcmBlockStats {
        queue_capacity: block_queue_capacity,
        queue_peak: 0,
        queue_backpressure: 0,
        total_blocks: 0,
        total_payload_bytes: 0,
        tracks: BTreeMap::new(),
    };

    for entry_index in 0..source.index.audio.len() {
        let extracted = source.extract_pcm_audio_packet(&bytes, entry_index)?;
        let track = source
            .tracks
            .iter()
            .find(|track| track.id == extracted.entry.track_id)
            .ok_or("PCM packet references missing audio track")?;
        let audio = track
            .audio
            .as_ref()
            .ok_or("PCM packet track has no audio descriptor")?;
        let sample_rate = rational_to_u32(audio.sample_rate.ok_or("audio sample rate missing")?)?;
        let bits_per_sample = audio.bit_depth.ok_or("audio bit depth missing")?;
        let format = PcmSampleFormat::SignedInteger {
            bits_per_sample,
            endian: PcmEndian::Little,
        };
        let start = duration_from_audio_samples(extracted.entry.start_sample, sample_rate)?;
        let duration =
            duration_from_audio_samples(u64::from(extracted.entry.sample_count), sample_rate)?;
        let packet = PcmAudioPacket::new(
            extracted.entry.track_id.0,
            extracted.entry.channel_index,
            start,
            duration,
            extracted.entry.sample_count,
            format,
            extracted.payload,
        )?;
        consume_pcm_payload_packet(&packet, &mut stats)?;
        let block = PcmAudioBlock::from_mono_packet(packet, sample_rate)?;
        if let Err(block) = block_queue.try_push(block) {
            let drained = block_queue
                .pop_front()
                .ok_or("bounded PCM block queue was full but empty")?;
            consume_pcm_audio_block(drained, &mut block_stats)?;
            block_queue
                .try_push(block)
                .map_err(|_| "bounded PCM block queue remained full after draining")?;
        }
    }
    let queue_stats = block_queue.stats();
    while let Some(block) = block_queue.pop_front() {
        consume_pcm_audio_block(block, &mut block_stats)?;
    }
    block_stats.queue_peak = queue_stats.peak_depth;
    block_stats.queue_backpressure = queue_stats.backpressure_events;

    println!("Original MXF PCM Extraction");
    println!("---------------------------");
    println!("Audio source: original MXF");
    println!("Proxy AAC: not used");
    println!("Audio tracks: {}", audio_tracks.len());
    for track in audio_tracks {
        let audio = track.audio.as_ref().ok_or("audio descriptor missing")?;
        let sample_rate = audio
            .sample_rate
            .map(rational_to_u32)
            .transpose()?
            .map(|value| value.to_string())
            .unwrap_or_else(|| "unknown".to_string());
        println!(
            "  track_id={} channels={} sample_rate={} bit_depth={:?} block_align={:?}",
            track.id.0,
            audio.channels.unwrap_or(0),
            sample_rate,
            audio.bit_depth,
            audio.block_align
        );
    }
    println!(
        "PCM payload: packets={} bytes={}",
        stats.total_packets, stats.total_payload_bytes
    );
    for (track_id, track_stats) in &stats.tracks {
        let duration = track_stats.last_end.unwrap_or(Duration::ZERO);
        println!(
            "  track_id={} channel={} packets={} samples={} payload_bytes={} first={:.3}s duration={:.3}s monotonic={}",
            track_id,
            track_stats.channel_index,
            track_stats.packets,
            track_stats.samples,
            track_stats.payload_bytes,
            track_stats
                .first_start
                .unwrap_or(Duration::ZERO)
                .as_secs_f64(),
            duration.as_secs_f64(),
            yes_no(track_stats.monotonic)
        );
    }
    println!("Runtime block model: mono-track PCM blocks");
    println!(
        "Runtime block queue: capacity={} peak={} backpressure={}",
        block_stats.queue_capacity, block_stats.queue_peak, block_stats.queue_backpressure
    );
    println!(
        "Runtime blocks: blocks={} bytes={}",
        block_stats.total_blocks, block_stats.total_payload_bytes
    );
    for (track_id, track_stats) in &block_stats.tracks {
        let duration = track_stats.last_end.unwrap_or(Duration::ZERO);
        println!(
            "  track_id={} channel={} blocks={} samples={} payload_bytes={} first={:.3}s duration={:.3}s monotonic={} gaps={} overlaps={}",
            track_id,
            track_stats.channel_index,
            track_stats.blocks,
            track_stats.samples,
            track_stats.payload_bytes,
            track_stats
                .first_start
                .unwrap_or(Duration::ZERO)
                .as_secs_f64(),
            duration.as_secs_f64(),
            yes_no(track_stats.monotonic),
            track_stats.gaps,
            track_stats.overlaps
        );
    }
    let clock_ready = block_stats.total_blocks == stats.total_packets
        && block_stats
            .tracks
            .values()
            .all(|track| track.monotonic && track.gaps == 0 && track.overlaps == 0);
    println!("Suitable for future audio clock: {}", yes_no(clock_ready));

    Ok(())
}

fn linux_audio_device_probe(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let source = MediaSource::parse(&bytes)?;
    let audio_tracks = source
        .tracks
        .iter()
        .filter(|track| track.kind == TrackKind::Audio)
        .collect::<Vec<_>>();
    if audio_tracks.is_empty() {
        return Err("original MXF has no audio tracks".into());
    }

    let source_format = original_linux_pcm_audio_format(&audio_tracks)?;
    let report = probe_linux_audio_device_boundary(source_format)?;

    println!("Linux Audio Device Boundary Spike");
    println!("----------------------------------");
    println!("Audio source: original MXF");
    println!("Proxy AAC: not used");
    println!("Full playback: no");
    println!(
        "AudioDeviceVerified: {}",
        yes_no(report.audio_device_verified)
    );
    println!(
        "Original PCM: tracks={} channels_per_track={} sample_rate={}Hz bit_depth={}bit",
        report.source_format.track_count,
        report.source_format.channels_per_track,
        report.source_format.sample_rate,
        report.source_format.bits_per_sample
    );
    println!(
        "PipeWire runtime socket: {}",
        yes_no(report.pipewire_runtime_socket_available)
    );
    println!(
        "pw-cli available: {}",
        yes_no(report.pipewire_cli_available)
    );
    println!(
        "PipeWire server reachable: {}",
        optional_yes_no(report.pipewire_server_reachable)
    );
    println!(
        "wpctl available: {}",
        yes_no(report.wireplumber_cli_available)
    );
    println!(
        "Default output device visible: {}",
        optional_yes_no(report.default_output_device_available)
    );
    println!(
        "Direct 48kHz 24-bit original PCM acceptance known: {}",
        optional_yes_no(report.direct_original_pcm_acceptance_known)
    );
    println!(
        "Device-boundary conversion needed: {}",
        linux_audio_conversion_need_label(report.conversion_needed)
    );
    println!(
        "Timing/playback-position evidence available: {}",
        optional_yes_no(report.timing_evidence_available)
    );
    println!("Probe outcome: {:?}", report.outcome);
    println!("Notes:");
    for note in report.notes {
        println!("  - {note}");
    }

    Ok(())
}

fn pipewire_audio_prototype(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let source = MediaSource::parse(&bytes)?;
    let audio_tracks = source
        .tracks
        .iter()
        .filter(|track| track.kind == TrackKind::Audio)
        .collect::<Vec<_>>();
    if audio_tracks.is_empty() {
        return Err("original MXF has no audio tracks".into());
    }

    let source_format = original_linux_pcm_audio_format(&audio_tracks)?;
    let blocks = build_original_pcm_blocks(&source, &bytes)?;
    let prototype_sample_count = 960_u32;
    let buffer = build_pipewire_f32_interleaved_prototype_buffer(
        &blocks,
        source_format.sample_rate,
        prototype_sample_count,
    )?;
    let report = submit_pipewire_audio_prototype(&buffer)?;

    println!("PipeWire Audio Device Prototype");
    println!("-------------------------------");
    println!("Audio source: original MXF");
    println!("Proxy AAC: not used");
    println!("Full playback: no");
    println!("Realtime Broadcast Player playback: no");
    println!(
        "Input PCM: tracks={} channels_per_track={} sample_rate={}Hz bit_depth={}bit",
        source_format.track_count,
        source_format.channels_per_track,
        source_format.sample_rate,
        source_format.bits_per_sample
    );
    println!("Device-boundary conversion: original 24-bit mono tracks -> f32 interleaved");
    println!(
        "Prototype buffer: channels={} sample_rate={}Hz samples={} bytes={}",
        buffer.channels,
        buffer.sample_rate,
        buffer.sample_count,
        buffer.bytes.len()
    );
    println!(
        "PipeWire available: {}",
        yes_no(report.pipewire_cli_available && report.pipewire_server_reachable)
    );
    println!(
        "Stream open attempted: {}",
        yes_no(report.stream_open_attempted)
    );
    println!("Stream opened: {}", yes_no(report.stream_opened));
    println!("Buffer submitted: {}", yes_no(report.buffer_submitted));
    println!("Bytes submitted: {}", report.bytes_submitted);
    println!("Evidence level: {:?}", report.evidence_level);
    println!(
        "AudioDeviceVerified: {}{}",
        yes_no(report.audio_device_verified),
        if report.audio_device_verified {
            " (tiny bounded PipeWire prototype buffer only)"
        } else {
            ""
        }
    );
    println!("Status: {}", report.status_message);

    Ok(())
}

fn pipewire_audio_native_prototype(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let source = MediaSource::parse(&bytes)?;
    let audio_tracks = source
        .tracks
        .iter()
        .filter(|track| track.kind == TrackKind::Audio)
        .collect::<Vec<_>>();
    if audio_tracks.is_empty() {
        return Err("original MXF has no audio tracks".into());
    }

    let source_format = original_linux_pcm_audio_format(&audio_tracks)?;
    let blocks = build_original_pcm_blocks(&source, &bytes)?;
    let prototype_sample_count = 960_u32;
    let pipewire_tracks = first_pipewire_mono_tracks(&blocks, prototype_sample_count)?;
    let converted =
        f32_interleaved_from_s24le_mono_tracks(&pipewire_tracks, prototype_sample_count)?;
    let buffer = LinuxPipewirePrototypeBuffer {
        sample_rate: source_format.sample_rate,
        channels: u16::try_from(pipewire_tracks.len())?,
        sample_count: prototype_sample_count,
        sample_format: LinuxPipewirePrototypeSampleFormat::F32Interleaved,
        bytes: converted.clone(),
    };
    let report = inspect_native_pipewire_stream_boundary(&buffer)?;

    println!("Native PipeWire Stream Prototype");
    println!("--------------------------------");
    println!("Native PipeWire post-submit evidence: enabled");
    println!("Audio source: original MXF");
    println!("Proxy AAC: not used");
    println!("Full playback: no");
    println!("Realtime Broadcast Player playback: no");
    println!(
        "Input PCM: tracks={} channels_per_track={} sample_rate={}Hz bit_depth={}bit",
        source_format.track_count,
        source_format.channels_per_track,
        source_format.sample_rate,
        source_format.bits_per_sample
    );
    println!("Planned device-boundary conversion: original 24-bit mono tracks -> f32 interleaved");
    println!(
        "Prototype buffer: channels={} sample_rate={}Hz samples={} bytes={}",
        buffer.channels,
        buffer.sample_rate,
        buffer.sample_count,
        buffer.bytes.len()
    );
    println!(
        "PipeWire runtime library available: {}",
        yes_no(report.runtime_library_available)
    );
    println!(
        "PipeWire server reachable: {}",
        yes_no(report.pipewire_server_reachable)
    );
    println!(
        "PipeWire pkg-config entry available: {}",
        yes_no(report.pkg_config_entry_available)
    );
    println!(
        "PipeWire headers available: {}",
        yes_no(report.headers_available)
    );
    if !report.pkg_config_entry_available || !report.headers_available {
        println!(
            "Stream create attempted: {}",
            yes_no(report.stream_create_attempted)
        );
        println!("Buffer dequeued: {}", yes_no(report.buffer_dequeued));
        println!("Buffer submitted: {}", yes_no(report.buffer_submitted));
        println!("Post-submit callbacks observed: 0");
        println!("Stream states after submission: []");
        println!("Drain requested: no");
        println!("Drain completed: no");
        println!("Stream error after submit: no");
        println!("Post-submit timeout: no");
        println!("Evidence level: {:?}", report.evidence_level);
        println!(
            "AudioDeviceVerified: {}",
            yes_no(report.audio_device_verified)
        );
        println!("Status: {}", report.status_message);
        return Ok(());
    }

    let stream_format = PipeWireStreamFormat {
        sample_rate: source_format.sample_rate,
        channels: u32::from(buffer.channels),
        sample_format: PipeWireAudioSampleFormat::F32Interleaved,
    };
    match submit_native_pipewire_buffer(
        stream_format,
        converted,
        prototype_sample_count,
        Duration::from_secs(2),
    ) {
        Ok(submission_report) => {
            let stream_report = &submission_report.stream_report;
            println!(
                "Stream create attempted: {}",
                yes_no(stream_report.stream_create_attempted)
            );
            println!("Stream created: {}", yes_no(stream_report.stream_created));
            println!(
                "Stream configured: {}",
                yes_no(stream_report.stream_configured)
            );
            println!(
                "Selected stream format: {:?} {}Hz channels={}",
                stream_report.selected_format.sample_format,
                stream_report.selected_format.sample_rate,
                stream_report.selected_format.channels
            );
            println!(
                "Observed stream states: {:?}",
                stream_report.observed_states
            );
            println!("Final stream state: {:?}", stream_report.final_state);
            println!(
                "Process callback reached: {}",
                yes_no(submission_report.process_callback_reached)
            );
            println!(
                "Buffer dequeued: {}",
                yes_no(submission_report.buffer_dequeued)
            );
            println!("Buffer capacity: {}", submission_report.buffer_capacity);
            println!("Buffers planned: {}", submission_report.buffers_planned);
            println!("Buffers submitted: {}", submission_report.buffers_submitted);
            println!(
                "Samples converted per track: {}",
                submission_report.samples_converted
            );
            println!("Output channels: {}", submission_report.output_channels);
            println!(
                "F32 samples written: {}",
                submission_report.f32_samples_written
            );
            println!("Bytes copied: {}", submission_report.bytes_copied);
            println!(
                "Buffer submitted: {}",
                yes_no(submission_report.buffer_submitted)
            );
            println!(
                "Post-submit callbacks observed: {}",
                submission_report.post_submit_process_callbacks
            );
            println!(
                "Stream states after submission: {:?}",
                submission_report.stream_states_after_submit
            );
            println!(
                "Drain requested: {}",
                yes_no(submission_report.drain_requested)
            );
            println!(
                "Drain completed: {}",
                yes_no(submission_report.drain_completed)
            );
            println!(
                "Stream error after submit: {}",
                yes_no(submission_report.stream_error_after_submit)
            );
            println!(
                "Post-submit timeout: {}",
                yes_no(submission_report.post_submit_timeout)
            );
            println!("Evidence level: {:?}", submission_report.evidence_level);
            println!(
                "AudioDeviceVerified: {}",
                yes_no(submission_report.audio_device_verified)
            );
            println!("Audible output claimed: no");
            println!("Status: {}", submission_report.status_message);
        }
        Err(err) => {
            println!("Stream create attempted: yes");
            println!("Stream created: no");
            println!("Stream configured: no");
            println!(
                "Selected stream format: F32Interleaved {}Hz channels={}",
                stream_format.sample_rate, stream_format.channels
            );
            println!("Buffer dequeued: no");
            println!("Buffer submitted: no");
            println!("Post-submit callbacks observed: 0");
            println!("Stream states after submission: []");
            println!("Drain requested: no");
            println!("Drain completed: no");
            println!("Stream error after submit: no");
            println!("Post-submit timeout: no");
            println!("Evidence level: NativeStreamCreateFailed");
            println!("AudioDeviceVerified: no");
            println!("Audible output claimed: no");
            println!("Status: {err}");
        }
    }

    Ok(())
}

fn pipewire_audio_audible_smoke_test(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let source = MediaSource::parse(&bytes)?;
    let audio_tracks = source
        .tracks
        .iter()
        .filter(|track| track.kind == TrackKind::Audio)
        .collect::<Vec<_>>();
    if audio_tracks.is_empty() {
        return Err("original MXF has no audio tracks".into());
    }

    let source_format = original_linux_pcm_audio_format(&audio_tracks)?;
    let blocks = build_original_pcm_blocks(&source, &bytes)?;
    let samples_per_buffer = 960_u32;
    let target_duration = Duration::from_millis(500);
    let buffers = smoke_test_buffer_count(
        target_duration,
        source_format.sample_rate,
        samples_per_buffer,
    )?;
    let pipewire_tracks = first_pipewire_mono_tracks(&blocks, samples_per_buffer)?;
    let segment = f32_interleaved_from_s24le_mono_tracks(&pipewire_tracks, samples_per_buffer)?;
    let smoke_buffers = repeated_smoke_test_buffers(&segment, buffers)?;
    let total_bytes = smoke_buffers.iter().map(Vec::len).sum::<usize>();
    let total_samples = samples_per_buffer.saturating_mul(buffers);
    let actual_duration =
        duration_from_audio_samples(u64::from(total_samples), source_format.sample_rate)?;
    let buffer = LinuxPipewirePrototypeBuffer {
        sample_rate: source_format.sample_rate,
        channels: u16::try_from(pipewire_tracks.len())?,
        sample_count: samples_per_buffer,
        sample_format: LinuxPipewirePrototypeSampleFormat::F32Interleaved,
        bytes: segment,
    };
    let report = inspect_native_pipewire_stream_boundary(&buffer)?;

    println!("Native PipeWire Audible Smoke Test");
    println!("----------------------------------");
    println!("Audio source: original MXF");
    println!("Proxy AAC: not used");
    println!("Full playback: no");
    println!("Realtime Broadcast Player playback: no");
    println!("A/V sync: no");
    println!("Generated tone primary evidence: no");
    println!(
        "Input PCM: tracks={} channels_per_track={} sample_rate={}Hz bit_depth={}bit",
        source_format.track_count,
        source_format.channels_per_track,
        source_format.sample_rate,
        source_format.bits_per_sample
    );
    println!("Device-boundary conversion: original 24-bit mono tracks -> f32 interleaved");
    println!("Routing: track 1 -> FL, track 2 -> FR, track 3 -> RL, track 4 -> RR");
    println!(
        "Smoke-test source range per buffer: {} samples/track",
        samples_per_buffer
    );
    println!(
        "Smoke-test duration: {:.3} ms",
        actual_duration.as_secs_f64() * 1000.0
    );
    println!("Smoke-test buffers planned: {}", buffers);
    println!("Smoke-test total bytes planned: {}", total_bytes);
    println!("Manual audible confirmation required: yes");
    println!(
        "PipeWire runtime library available: {}",
        yes_no(report.runtime_library_available)
    );
    println!(
        "PipeWire server reachable: {}",
        yes_no(report.pipewire_server_reachable)
    );
    println!(
        "PipeWire pkg-config entry available: {}",
        yes_no(report.pkg_config_entry_available)
    );
    println!(
        "PipeWire headers available: {}",
        yes_no(report.headers_available)
    );
    if !report.pkg_config_entry_available || !report.headers_available {
        println!(
            "Stream create attempted: {}",
            yes_no(report.stream_create_attempted)
        );
        println!("Buffers submitted: 0");
        println!("Drain requested: no");
        println!("Drain completed: no");
        println!("Manual confirmation status: ManualAudibleConfirmationRequired");
        println!("Evidence level: {:?}", report.evidence_level);
        println!("AudioDeviceVerified: no");
        println!("Audible output claimed: no");
        println!("Status: {}", report.status_message);
        return Ok(());
    }

    let stream_format = PipeWireStreamFormat {
        sample_rate: source_format.sample_rate,
        channels: u32::from(buffer.channels),
        sample_format: PipeWireAudioSampleFormat::F32Interleaved,
    };
    let submission_report = submit_native_pipewire_buffers(
        stream_format,
        smoke_buffers,
        samples_per_buffer,
        Duration::from_secs(4),
    )?;
    let confirmation = read_manual_audible_confirmation()?;
    let evidence = audible_smoke_evidence_label(
        submission_report.buffer_submitted,
        submission_report.drain_completed,
        confirmation,
    );

    println!(
        "Stream configured: {}",
        yes_no(submission_report.stream_report.stream_configured)
    );
    println!(
        "Observed stream states: {:?}",
        submission_report.stream_report.observed_states
    );
    println!(
        "Final stream state: {:?}",
        submission_report.stream_report.final_state
    );
    println!(
        "Process callback reached: {}",
        yes_no(submission_report.process_callback_reached)
    );
    println!("Buffer capacity: {}", submission_report.buffer_capacity);
    println!("Buffers planned: {}", submission_report.buffers_planned);
    println!("Buffers submitted: {}", submission_report.buffers_submitted);
    println!(
        "Samples submitted per track: {}",
        submission_report.samples_converted
    );
    println!("Output channels: {}", submission_report.output_channels);
    println!(
        "F32 samples written: {}",
        submission_report.f32_samples_written
    );
    println!("Bytes copied: {}", submission_report.bytes_copied);
    println!(
        "Drain requested: {}",
        yes_no(submission_report.drain_requested)
    );
    println!(
        "Drain completed: {}",
        yes_no(submission_report.drain_completed)
    );
    println!(
        "Post-submit callbacks observed: {}",
        submission_report.post_submit_process_callbacks
    );
    println!(
        "Post-submit timeout: {}",
        yes_no(submission_report.post_submit_timeout)
    );
    println!(
        "PipeWire evidence level: {:?}",
        submission_report.evidence_level
    );
    println!("Manual confirmation status: {}", confirmation.label());
    println!("Smoke-test evidence level: {}", evidence);
    println!("AudioDeviceVerified: no");
    println!("AudioDeviceVerified scope: not upgraded by smoke test");
    println!(
        "Audible output claimed: {}",
        yes_no(confirmation == ManualAudibleConfirmation::Yes)
    );
    println!(
        "Status: bounded original-audio-derived smoke test completed; full playback is not claimed"
    );
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ManualAudibleConfirmation {
    Yes,
    No,
    Required,
}

impl ManualAudibleConfirmation {
    const fn label(self) -> &'static str {
        match self {
            Self::Yes => "ManualContentAudibilityPartiallyObserved",
            Self::No => "ManualAudibleSmokeTestNotHeard",
            Self::Required => "ManualAudibleConfirmationRequired",
        }
    }

    const fn original_segment_label(self) -> &'static str {
        match self {
            Self::Yes => "ManualContentAudibilityPartiallyObserved",
            Self::No => "ManualOriginalSegmentNotHeard",
            Self::Required => "ManualOriginalSegmentConfirmationRequired",
        }
    }

    const fn runtime_payload_label(self) -> &'static str {
        match self {
            Self::Yes => "RuntimeAudioPayloadAudibleConfirmed",
            Self::No => "RuntimeAudioPayloadNotHeard",
            Self::Required => "RuntimeAudioPayloadConfirmationRequired",
        }
    }

    const fn mono_monitor_label(self) -> &'static str {
        match self {
            Self::Yes => "ManualDesktopMonoListeningHelperHeard",
            Self::No => "ManualDesktopMonoListeningHelperNotHeard",
            Self::Required => "ManualDesktopMonoListeningHelperConfirmationRequired",
        }
    }
}

fn read_manual_audible_confirmation() -> io::Result<ManualAudibleConfirmation> {
    if !io::stdin().is_terminal() {
        return Ok(ManualAudibleConfirmation::Required);
    }
    print!("Did you hear audio from the default PipeWire output? yes/no: ");
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(parse_manual_audible_confirmation(&input))
}

fn read_manual_original_segment_confirmation() -> io::Result<ManualAudibleConfirmation> {
    if !io::stdin().is_terminal() {
        return Ok(ManualAudibleConfirmation::Required);
    }
    print!("Did you hear the bounded original-audio segment from the default PipeWire output? yes/no: ");
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(parse_manual_audible_confirmation(&input))
}

fn parse_manual_audible_confirmation(input: &str) -> ManualAudibleConfirmation {
    match input.trim().to_ascii_lowercase().as_str() {
        "y" | "yes" => ManualAudibleConfirmation::Yes,
        "n" | "no" => ManualAudibleConfirmation::No,
        _ => ManualAudibleConfirmation::Required,
    }
}

fn audible_smoke_evidence_label(
    submitted: bool,
    drain_completed: bool,
    confirmation: ManualAudibleConfirmation,
) -> &'static str {
    match (submitted, drain_completed, confirmation) {
        (true, _, ManualAudibleConfirmation::Yes) => "ManualContentAudibilityPartiallyObserved",
        (true, _, ManualAudibleConfirmation::No) => "ManualAudibleSmokeTestNotHeard",
        (true, true, ManualAudibleConfirmation::Required) => "ManualAudibleConfirmationRequired",
        (true, false, ManualAudibleConfirmation::Required) => "AudibleSmokeTestSubmitted",
        _ => "AudibleSmokeTestNotSubmitted",
    }
}

fn original_segment_evidence_label(
    submitted: bool,
    drain_completed: bool,
    confirmation: ManualAudibleConfirmation,
) -> &'static str {
    match (submitted, drain_completed, confirmation) {
        (true, _, ManualAudibleConfirmation::Yes) => "ManualContentAudibilityPartiallyObserved",
        (true, _, ManualAudibleConfirmation::No) => "ManualOriginalSegmentNotHeard",
        (true, true, ManualAudibleConfirmation::Required) => {
            "ManualOriginalSegmentConfirmationRequired"
        }
        (true, false, ManualAudibleConfirmation::Required) => "OriginalSegmentSubmitted",
        _ => "OriginalSegmentNotSubmitted",
    }
}

fn read_manual_runtime_payload_confirmation() -> io::Result<ManualAudibleConfirmation> {
    if !io::stdin().is_terminal() {
        return Ok(ManualAudibleConfirmation::Required);
    }
    print!(
        "Did you hear the runtime-prepared original-audio payload from the default PipeWire output? yes/no: "
    );
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(parse_manual_audible_confirmation(&input))
}

fn read_manual_mono_monitor_confirmation() -> io::Result<ManualAudibleConfirmation> {
    if !io::stdin().is_terminal() {
        return Ok(ManualAudibleConfirmation::Required);
    }
    print!(
        "Did you hear the desktop mono listening helper from the default PipeWire output? yes/no: "
    );
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(parse_manual_audible_confirmation(&input))
}

fn runtime_audio_payload_evidence_label(
    submitted: bool,
    drain_completed: bool,
    confirmation: ManualAudibleConfirmation,
) -> &'static str {
    match (submitted, drain_completed, confirmation) {
        (true, _, ManualAudibleConfirmation::Yes) => "RuntimeAudioPayloadAudibleConfirmed",
        (true, _, ManualAudibleConfirmation::No) => "RuntimeAudioPayloadNotHeard",
        (true, true, ManualAudibleConfirmation::Required) => "RuntimeAudioPayloadDrainCompleted",
        (true, false, ManualAudibleConfirmation::Required) => {
            "RuntimeAudioPayloadSubmittedToPipeWire"
        }
        _ => "RuntimeAudioPayloadNotSubmitted",
    }
}

fn mono_monitor_diagnostic_evidence_label(
    submitted: bool,
    drain_completed: bool,
    confirmation: ManualAudibleConfirmation,
) -> &'static str {
    match (submitted, drain_completed, confirmation) {
        (true, _, ManualAudibleConfirmation::Yes) => "ManualDesktopMonoListeningHelperHeard",
        (true, _, ManualAudibleConfirmation::No) => "ManualDesktopMonoListeningHelperNotHeard",
        (true, true, ManualAudibleConfirmation::Required) => {
            "DesktopMonoListeningHelperDrainCompleted"
        }
        (true, false, ManualAudibleConfirmation::Required) => "DesktopMonoListeningHelperSubmitted",
        _ => "DesktopMonoListeningHelperNotSubmitted",
    }
}

fn discrete_4mono_evidence_label(submitted: bool, drain_completed: bool) -> &'static str {
    match (submitted, drain_completed) {
        (true, true) => "Discrete4MonoOutputDrainCompleted",
        (true, false) => "Discrete4MonoOutputSubmitted",
        _ => "Discrete4MonoOutputNotSubmitted",
    }
}

fn smoke_test_buffer_count(
    target_duration: Duration,
    sample_rate: u32,
    samples_per_buffer: u32,
) -> Result<u32, Box<dyn std::error::Error>> {
    if sample_rate == 0 || samples_per_buffer == 0 {
        return Err("smoke-test sample rate and samples per buffer must be non-zero".into());
    }
    let target_samples = audio_samples_for_duration(target_duration, sample_rate)?;
    let samples_per_buffer = u64::from(samples_per_buffer);
    let buffers = target_samples.div_ceil(samples_per_buffer);
    Ok(u32::try_from(buffers.max(1))?)
}

fn repeated_smoke_test_buffers(
    segment: &[u8],
    count: u32,
) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
    if segment.is_empty() || count == 0 {
        return Err("smoke-test segment and count must be non-zero".into());
    }
    Ok((0..count).map(|_| segment.to_vec()).collect())
}

fn pipewire_audio_segment_playback(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let source = MediaSource::parse(&bytes)?;
    let audio_tracks = source
        .tracks
        .iter()
        .filter(|track| track.kind == TrackKind::Audio)
        .collect::<Vec<_>>();
    if audio_tracks.is_empty() {
        return Err("original MXF has no audio tracks".into());
    }

    let source_format = original_linux_pcm_audio_format(&audio_tracks)?;
    let blocks = build_original_pcm_blocks(&source, &bytes)?;
    let segment = build_sequential_original_audio_segment(
        &blocks,
        source_format.sample_rate,
        Duration::from_secs(1),
        Duration::from_millis(500),
    )?;
    let report_buffer = LinuxPipewirePrototypeBuffer {
        sample_rate: source_format.sample_rate,
        channels: u16::try_from(segment.channels)?,
        sample_count: segment.samples_per_buffer,
        sample_format: LinuxPipewirePrototypeSampleFormat::F32Interleaved,
        bytes: segment
            .buffers
            .first()
            .cloned()
            .ok_or("empty segment buffer plan")?,
    };
    let report = inspect_native_pipewire_stream_boundary(&report_buffer)?;

    println!("Native PipeWire Bounded Original-Audio Segment Playback");
    println!("------------------------------------------------------");
    println!("Audio source: original MXF");
    println!("Proxy AAC: not used");
    println!("Sequential original blocks: yes");
    println!("Full playback: no");
    println!("Realtime Broadcast Player playback: no");
    println!("A/V sync: no");
    println!(
        "Input PCM: tracks={} channels_per_track={} sample_rate={}Hz bit_depth={}bit",
        source_format.track_count,
        source_format.channels_per_track,
        source_format.sample_rate,
        source_format.bits_per_sample
    );
    println!("Device-boundary conversion: original 24-bit mono tracks -> f32 interleaved");
    println!("Routing: track 1 -> FL, track 2 -> FR, track 3 -> RL, track 4 -> RR");
    println!("Selected segment start sample: {}", segment.start_sample);
    println!(
        "Selected segment duration: {:.3} ms",
        segment.duration.as_secs_f64() * 1000.0
    );
    println!("Source blocks per track: {}", segment.blocks_per_track);
    println!("Samples per track: {}", segment.samples_per_track);
    println!("Continuity status: {}", segment.continuity_status);
    println!("Output buffers planned: {}", segment.buffers.len());
    println!("Output bytes planned: {}", segment.total_bytes);
    println!("Manual audible confirmation required: yes");
    println!(
        "PipeWire runtime library available: {}",
        yes_no(report.runtime_library_available)
    );
    println!(
        "PipeWire server reachable: {}",
        yes_no(report.pipewire_server_reachable)
    );
    println!(
        "PipeWire pkg-config entry available: {}",
        yes_no(report.pkg_config_entry_available)
    );
    println!(
        "PipeWire headers available: {}",
        yes_no(report.headers_available)
    );
    if !report.pkg_config_entry_available || !report.headers_available {
        println!(
            "Stream create attempted: {}",
            yes_no(report.stream_create_attempted)
        );
        println!("Buffers submitted: 0");
        println!("Drain requested: no");
        println!("Drain completed: no");
        println!("Manual confirmation status: ManualOriginalSegmentConfirmationRequired");
        println!("Evidence level: {:?}", report.evidence_level);
        println!("AudioDeviceVerified: no");
        println!("Audible output claimed: no");
        println!("Status: {}", report.status_message);
        return Ok(());
    }

    let stream_format = PipeWireStreamFormat {
        sample_rate: source_format.sample_rate,
        channels: u32::try_from(segment.channels)?,
        sample_format: PipeWireAudioSampleFormat::F32Interleaved,
    };
    let submission_report = submit_native_pipewire_buffers(
        stream_format,
        segment.buffers,
        segment.samples_per_buffer,
        Duration::from_secs(5),
    )?;
    let confirmation = read_manual_original_segment_confirmation()?;
    let evidence = original_segment_evidence_label(
        submission_report.buffer_submitted,
        submission_report.drain_completed,
        confirmation,
    );

    println!(
        "Stream configured: {}",
        yes_no(submission_report.stream_report.stream_configured)
    );
    println!(
        "Observed stream states: {:?}",
        submission_report.stream_report.observed_states
    );
    println!(
        "Final stream state: {:?}",
        submission_report.stream_report.final_state
    );
    println!(
        "Process callback reached: {}",
        yes_no(submission_report.process_callback_reached)
    );
    println!("Buffer capacity: {}", submission_report.buffer_capacity);
    println!("Buffers planned: {}", submission_report.buffers_planned);
    println!("Buffers submitted: {}", submission_report.buffers_submitted);
    println!(
        "Samples submitted per track: {}",
        submission_report.samples_converted
    );
    println!("Output channels: {}", submission_report.output_channels);
    println!(
        "F32 samples written: {}",
        submission_report.f32_samples_written
    );
    println!("Bytes copied: {}", submission_report.bytes_copied);
    println!(
        "Drain requested: {}",
        yes_no(submission_report.drain_requested)
    );
    println!(
        "Drain completed: {}",
        yes_no(submission_report.drain_completed)
    );
    println!(
        "Post-submit callbacks observed: {}",
        submission_report.post_submit_process_callbacks
    );
    println!(
        "Post-submit timeout: {}",
        yes_no(submission_report.post_submit_timeout)
    );
    println!(
        "PipeWire evidence level: {:?}",
        submission_report.evidence_level
    );
    println!(
        "Manual confirmation status: {}",
        confirmation.original_segment_label()
    );
    println!("Segment evidence level: {}", evidence);
    println!("AudioDeviceVerified: no");
    println!("AudioDeviceVerified scope: not upgraded by bounded segment playback");
    println!(
        "Audible output claimed: {}",
        yes_no(confirmation == ManualAudibleConfirmation::Yes)
    );
    println!(
        "Status: bounded sequential original-audio segment completed; full playback is not claimed"
    );
    Ok(())
}

fn broadcast_runtime_audio_pipewire(
    original_path: &Path,
    proxy_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let plan = prepare_runtime_audio_pipewire_payload(original_path, proxy_path)?;
    let payload_buffer = &plan.payload_buffer;
    let audio_slot = &plan.audio_slot;
    let audio_binding = &plan.audio_binding;
    let report_buffer = LinuxPipewirePrototypeBuffer {
        sample_rate: payload_buffer.sample_rate,
        channels: u16::try_from(payload_buffer.track_count)?,
        sample_count: payload_buffer.output_frames,
        sample_format: LinuxPipewirePrototypeSampleFormat::F32Interleaved,
        bytes: payload_buffer.output_bytes.clone(),
    };
    let report = inspect_native_pipewire_stream_boundary(&report_buffer)?;

    println!("QGS Broadcast Player Runtime Audio Payload to PipeWire");
    println!("------------------------------------------------------");
    println!("Source mode: ProxyPreview");
    println!("Audio source: original MXF");
    println!("Video source: proxy MP4");
    println!("Original MXF audio authoritative: yes");
    println!("Proxy AAC: not used");
    println!("Full playback: no");
    println!("Realtime Broadcast Player playback: no");
    println!("A/V sync: no");
    println!("QNC UI integration: no");
    print_runtime_audio_payload_geometry(audio_slot, audio_binding, payload_buffer);
    println!("Manual audible confirmation required: yes");
    println!(
        "PipeWire runtime library available: {}",
        yes_no(report.runtime_library_available)
    );
    println!(
        "PipeWire server reachable: {}",
        yes_no(report.pipewire_server_reachable)
    );
    println!(
        "PipeWire pkg-config entry available: {}",
        yes_no(report.pkg_config_entry_available)
    );
    println!(
        "PipeWire headers available: {}",
        yes_no(report.headers_available)
    );
    if !report.pkg_config_entry_available || !report.headers_available {
        println!(
            "Stream create attempted: {}",
            yes_no(report.stream_create_attempted)
        );
        println!("Buffers submitted: 0");
        println!("Drain requested: no");
        println!("Drain completed: no");
        println!("Manual confirmation status: RuntimeAudioPayloadConfirmationRequired");
        println!("Evidence level: {:?}", report.evidence_level);
        println!("AudioDeviceVerified: no");
        println!("AudioDeviceVerified scope: not upgraded by runtime payload boundary");
        println!("Audible output claimed: no");
        println!("Status: {}", report.status_message);
        return Ok(());
    }

    let stream_format = PipeWireStreamFormat {
        sample_rate: payload_buffer.sample_rate,
        channels: u32::try_from(payload_buffer.track_count)?,
        sample_format: PipeWireAudioSampleFormat::F32Interleaved,
    };
    let submission_report = submit_native_pipewire_buffer(
        stream_format,
        payload_buffer.output_bytes.clone(),
        payload_buffer.output_frames,
        Duration::from_secs(3),
    )?;
    let confirmation = read_manual_runtime_payload_confirmation()?;
    let evidence = runtime_audio_payload_evidence_label(
        submission_report.buffer_submitted,
        submission_report.drain_completed,
        confirmation,
    );

    print_runtime_audio_pipewire_submission(&submission_report);
    println!(
        "Manual confirmation status: {}",
        confirmation.runtime_payload_label()
    );
    println!("Runtime payload evidence level: {}", evidence);
    println!("AudioDeviceVerified: no");
    println!("AudioDeviceVerified scope: not upgraded by runtime payload boundary");
    println!(
        "Audible output claimed: {}",
        yes_no(confirmation == ManualAudibleConfirmation::Yes)
    );
    println!(
        "Status: runtime-prepared original-audio payload submitted to PipeWire; full playback is not claimed"
    );
    Ok(())
}

fn broadcast_runtime_audio_pipewire_audible(
    original_path: &Path,
    proxy_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let plan = prepare_runtime_audio_pipewire_payload(original_path, proxy_path)?;
    let payload_buffer = &plan.payload_buffer;
    let audio_slot = &plan.audio_slot;
    let audio_binding = &plan.audio_binding;
    let helper_repeats = smoke_test_buffer_count(
        Duration::from_millis(500),
        payload_buffer.sample_rate,
        payload_buffer.output_frames,
    )?;
    let helper_buffers = repeated_smoke_test_buffers(&payload_buffer.output_bytes, helper_repeats)?;
    let total_frames_submitted = payload_buffer
        .output_frames
        .checked_mul(helper_repeats)
        .ok_or("runtime payload audible helper frame count overflow")?;
    let helper_duration = duration_from_audio_samples(
        u64::from(total_frames_submitted),
        payload_buffer.sample_rate,
    )?;
    let total_bytes_planned = helper_buffers.iter().map(Vec::len).sum::<usize>();
    let report_buffer = LinuxPipewirePrototypeBuffer {
        sample_rate: payload_buffer.sample_rate,
        channels: u16::try_from(payload_buffer.track_count)?,
        sample_count: payload_buffer.output_frames,
        sample_format: LinuxPipewirePrototypeSampleFormat::F32Interleaved,
        bytes: payload_buffer.output_bytes.clone(),
    };
    let report = inspect_native_pipewire_stream_boundary(&report_buffer)?;

    println!("QGS Broadcast Player Runtime Audio Payload Audible Confirmation");
    println!("---------------------------------------------------------------");
    println!("Source mode: ProxyPreview");
    println!("Audio source: original MXF");
    println!("Video source: proxy MP4");
    println!("Original MXF audio authoritative: yes");
    println!("Proxy AAC: not used");
    println!("Full playback: no");
    println!("Realtime Broadcast Player playback: no");
    println!("A/V sync: no");
    println!("QNC UI integration: no");
    print_runtime_audio_payload_geometry(audio_slot, audio_binding, payload_buffer);
    println!("Helper repeat enabled: yes");
    println!("Helper repeat count: {}", helper_repeats);
    println!(
        "Helper repeat duration: {:.3} ms",
        helper_duration.as_secs_f64() * 1000.0
    );
    println!("Total frames submitted planned: {}", total_frames_submitted);
    println!("Total bytes planned: {}", total_bytes_planned);
    println!("Manual audible confirmation required: yes");
    println!(
        "PipeWire runtime library available: {}",
        yes_no(report.runtime_library_available)
    );
    println!(
        "PipeWire server reachable: {}",
        yes_no(report.pipewire_server_reachable)
    );
    println!(
        "PipeWire pkg-config entry available: {}",
        yes_no(report.pkg_config_entry_available)
    );
    println!(
        "PipeWire headers available: {}",
        yes_no(report.headers_available)
    );
    if !report.pkg_config_entry_available || !report.headers_available {
        println!(
            "Stream create attempted: {}",
            yes_no(report.stream_create_attempted)
        );
        println!("Buffers submitted: 0");
        println!("Drain requested: no");
        println!("Drain completed: no");
        println!("Manual confirmation status: RuntimeAudioPayloadConfirmationRequired");
        println!("Evidence level: {:?}", report.evidence_level);
        println!("AudioDeviceVerified: no");
        println!("AudioDeviceVerified scope: not upgraded by runtime payload audible helper");
        println!("Audible output claimed: no");
        println!("Status: {}", report.status_message);
        return Ok(());
    }

    let stream_format = PipeWireStreamFormat {
        sample_rate: payload_buffer.sample_rate,
        channels: u32::try_from(payload_buffer.track_count)?,
        sample_format: PipeWireAudioSampleFormat::F32Interleaved,
    };
    let submission_report = submit_native_pipewire_buffers(
        stream_format,
        helper_buffers,
        payload_buffer.output_frames,
        Duration::from_secs(4),
    )?;
    let confirmation = read_manual_runtime_payload_confirmation()?;
    let evidence = runtime_audio_payload_evidence_label(
        submission_report.buffer_submitted,
        submission_report.drain_completed,
        confirmation,
    );

    print_runtime_audio_pipewire_submission(&submission_report);
    println!("Helper repeat enabled: yes");
    println!("Helper repeat count: {}", helper_repeats);
    println!("Total frames submitted: {}", total_frames_submitted);
    println!("Total bytes copied planned: {}", total_bytes_planned);
    println!(
        "Manual confirmation status: {}",
        confirmation.runtime_payload_label()
    );
    println!("Runtime payload evidence level: {}", evidence);
    println!("AudioDeviceBoundaryEvidence: {}", evidence);
    println!("AudioDeviceVerified: no");
    println!("AudioDeviceVerified scope: not upgraded by runtime payload audible helper");
    println!(
        "Audible output claimed: {}",
        yes_no(confirmation == ManualAudibleConfirmation::Yes)
    );
    println!(
        "Status: repeated runtime-prepared audio payload audible helper completed; full playback is not claimed"
    );
    Ok(())
}

fn prepare_runtime_audio_pipewire_payload(
    original_path: &Path,
    proxy_path: &Path,
) -> Result<RuntimeAudioPayloadPipeWirePlan, Box<dyn std::error::Error>> {
    let original_bytes = std::fs::read(original_path)?;
    let original = MediaSource::parse(&original_bytes)?;
    let audio_tracks = original
        .tracks
        .iter()
        .filter(|track| track.kind == TrackKind::Audio)
        .collect::<Vec<_>>();
    if audio_tracks.is_empty() {
        return Err("original MXF has no audio tracks".into());
    }
    let source_format = original_linux_pcm_audio_format(&audio_tracks)?;
    let blocks = build_original_pcm_blocks(&original, &original_bytes)?;

    let proxy = Mp4Source::open(proxy_path)?;
    let proxy_video = proxy
        .video
        .as_ref()
        .ok_or("proxy has no H.264 video track")?;
    let selected = selected_journalist_preview_frames(proxy_video)?;
    let first_selected = selected
        .first()
        .ok_or("journalist-50i-preview selected no proxy frames")?;
    let presentation_rate =
        ProxyPlaybackProfile::Journalist50iPreview.presentation_rate(RationalRate::new(
            u64::from(proxy_video.frame_rate.numerator),
            u64::from(proxy_video.frame_rate.denominator),
        )?)?;
    let frame_duration = presentation_rate.frame_duration()?;
    let range = av_frame_audio_range(
        first_selected.preview_index,
        first_selected.start_time,
        frame_duration,
        source_format.sample_rate,
        &blocks,
        audio_tracks.len(),
    )?;
    let audio_slot = BroadcastPreparedAudioSlot::from_audio_range(
        0,
        BroadcastVideoSourceMode::ProxyPreview,
        &range,
    );
    let audio_binding = bind_broadcast_audio_payload(&audio_slot, &blocks)?;
    let payload_buffer = build_runtime_audio_payload_pipewire_buffer(&audio_binding, &blocks)?;
    Ok(RuntimeAudioPayloadPipeWirePlan {
        audio_slot,
        audio_binding,
        payload_buffer,
    })
}

fn print_runtime_audio_payload_geometry(
    audio_slot: &BroadcastPreparedAudioSlot,
    audio_binding: &qgs_media_runtime::BroadcastAudioPayloadBinding,
    payload_buffer: &RuntimeAudioPayloadPipeWireBuffer,
) {
    println!("Runtime source mode: {:?}", audio_binding.source_mode);
    println!(
        "Prepared audio slot index: {}",
        audio_binding.audio_slot_index
    );
    println!(
        "Presentation range duration: {:.3} ms",
        audio_slot.duration.as_secs_f64() * 1000.0
    );
    println!(
        "Presentation start sample: {} sample_count={}",
        audio_binding.start_sample, audio_binding.sample_count
    );
    println!(
        "Source blocks per track: {}",
        payload_buffer.source_blocks_per_track
    );
    println!(
        "Source blocks total: {}",
        payload_buffer.source_blocks_total
    );
    println!("Source bytes: {}", payload_buffer.source_bytes);
    println!("Samples per track: {}", payload_buffer.output_frames);
    println!("Device-boundary conversion: runtime original 24-bit mono tracks -> f32 interleaved");
    println!("Routing: track 1 -> FL, track 2 -> FR, track 3 -> RL, track 4 -> RR");
    println!(
        "Output format: F32Interleaved {}Hz channels={}",
        payload_buffer.sample_rate, payload_buffer.track_count
    );
    println!("Output frames: {}", payload_buffer.output_frames);
    println!("F32 samples planned: {}", payload_buffer.f32_sample_count());
    println!(
        "Output bytes planned: {}",
        payload_buffer.output_bytes.len()
    );
}

fn print_runtime_audio_pipewire_submission(
    submission_report: &qgs_audio_pipewire::PipeWireBufferSubmissionReport,
) {
    println!(
        "Stream configured: {}",
        yes_no(submission_report.stream_report.stream_configured)
    );
    println!(
        "Observed stream states: {:?}",
        submission_report.stream_report.observed_states
    );
    println!(
        "Final stream state: {:?}",
        submission_report.stream_report.final_state
    );
    println!(
        "Process callback reached: {}",
        yes_no(submission_report.process_callback_reached)
    );
    println!(
        "Buffer dequeued: {}",
        yes_no(submission_report.buffer_dequeued)
    );
    println!("Buffer capacity: {}", submission_report.buffer_capacity);
    println!("Buffers planned: {}", submission_report.buffers_planned);
    println!("Buffers submitted: {}", submission_report.buffers_submitted);
    println!(
        "Samples submitted per track: {}",
        submission_report.samples_converted
    );
    println!("Output channels: {}", submission_report.output_channels);
    println!(
        "F32 samples written: {}",
        submission_report.f32_samples_written
    );
    println!("Bytes copied: {}", submission_report.bytes_copied);
    println!(
        "Drain requested: {}",
        yes_no(submission_report.drain_requested)
    );
    println!(
        "Drain completed: {}",
        yes_no(submission_report.drain_completed)
    );
    println!(
        "Post-submit callbacks observed: {}",
        submission_report.post_submit_process_callbacks
    );
    println!(
        "Post-submit timeout: {}",
        yes_no(submission_report.post_submit_timeout)
    );
    println!(
        "PipeWire evidence level: {:?}",
        submission_report.evidence_level
    );
}

struct RuntimeAudioPayloadPipeWirePlan {
    audio_slot: BroadcastPreparedAudioSlot,
    audio_binding: qgs_media_runtime::BroadcastAudioPayloadBinding,
    payload_buffer: RuntimeAudioPayloadPipeWireBuffer,
}

struct RuntimeAudioPayloadPipeWireBuffer {
    output_bytes: Vec<u8>,
    output_frames: u32,
    sample_rate: u32,
    track_count: usize,
    source_blocks_per_track: usize,
    source_blocks_total: usize,
    source_bytes: usize,
}

impl RuntimeAudioPayloadPipeWireBuffer {
    fn f32_sample_count(&self) -> usize {
        usize::try_from(self.output_frames).unwrap_or(0) * self.track_count
    }
}

fn build_runtime_audio_payload_pipewire_buffer(
    binding: &qgs_media_runtime::BroadcastAudioPayloadBinding,
    blocks: &[PcmAudioBlock],
) -> Result<RuntimeAudioPayloadPipeWireBuffer, Box<dyn std::error::Error>> {
    if binding.source_mode != BroadcastVideoSourceMode::ProxyPreview || !binding.complete {
        return Err(
            "runtime PipeWire payload requires a complete ProxyPreview audio binding".into(),
        );
    }
    let output_frames = u32::try_from(binding.sample_count)?;
    let mut coverage_by_channel: BTreeMap<
        u16,
        Vec<&qgs_media_runtime::BroadcastAudioBlockCoverage>,
    > = BTreeMap::new();
    for coverage in &binding.block_coverage {
        coverage_by_channel
            .entry(coverage.channel_index)
            .or_default()
            .push(coverage);
    }
    if coverage_by_channel.len() != binding.track_count {
        return Err("runtime audio payload binding does not cover every track".into());
    }

    let mut track_payloads = Vec::with_capacity(coverage_by_channel.len());
    let mut source_bytes = 0_usize;
    let mut source_blocks_per_track = None;
    for (channel_index, coverages) in coverage_by_channel {
        let mut coverages = coverages;
        coverages.sort_by_key(|coverage| coverage.block_start_sample);
        if let Some(expected) = source_blocks_per_track {
            if expected != coverages.len() {
                return Err("runtime audio payload has uneven block coverage per track".into());
            }
        } else {
            source_blocks_per_track = Some(coverages.len());
        }
        let mut payload = Vec::new();
        for coverage in coverages {
            let block = blocks
                .get(coverage.block_index)
                .ok_or("runtime audio payload coverage references missing block")?;
            let PcmAudioBlockLayout::MonoTrack {
                track_id,
                channel_index: block_channel_index,
            } = block.layout
            else {
                return Err("runtime audio payload requires mono-track PCM blocks".into());
            };
            if track_id != coverage.track_id || block_channel_index != coverage.channel_index {
                return Err("runtime audio payload coverage does not match block identity".into());
            }
            let PcmSampleFormat::SignedInteger {
                bits_per_sample,
                endian,
            } = block.format;
            if bits_per_sample != 24 || endian != PcmEndian::Little {
                return Err(
                    "runtime PipeWire boundary requires original 24-bit little-endian PCM".into(),
                );
            }
            if block.sample_rate != binding.sample_rate {
                return Err("runtime audio payload contains mixed sample rates".into());
            }
            let end = coverage
                .byte_offset_within_block
                .checked_add(coverage.byte_count)
                .ok_or("runtime audio payload byte range overflow")?;
            let slice = block
                .payload
                .get(coverage.byte_offset_within_block..end)
                .ok_or("runtime audio payload byte range is outside block payload")?;
            payload.extend_from_slice(slice);
            source_bytes = source_bytes
                .checked_add(slice.len())
                .ok_or("runtime audio payload source byte count overflow")?;
        }
        let expected_bytes = usize::try_from(binding.sample_count)?
            .checked_mul(3)
            .ok_or("runtime audio payload expected byte count overflow")?;
        if payload.len() != expected_bytes {
            return Err(
                "runtime audio payload track byte count does not match sample range".into(),
            );
        }
        track_payloads.push((channel_index, payload));
    }

    let mono_tracks = track_payloads
        .iter()
        .map(|(channel_index, payload)| MonoS24LeTrack {
            channel_index: *channel_index,
            payload,
        })
        .collect::<Vec<_>>();
    let output_bytes = f32_interleaved_from_s24le_mono_tracks(&mono_tracks, output_frames)?;
    Ok(RuntimeAudioPayloadPipeWireBuffer {
        output_bytes,
        output_frames,
        sample_rate: binding.sample_rate,
        track_count: binding.track_count,
        source_blocks_per_track: source_blocks_per_track.unwrap_or(0),
        source_blocks_total: binding.block_coverage.len(),
        source_bytes,
    })
}

fn pipewire_audio_content_audit(
    original_path: &Path,
    proxy_path: &Path,
    options: AudioContentAuditOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let duration_ms = bounded_audio_audit_duration_ms(options.duration_ms)?;
    let mut start_ms = options.start_ms;
    let audio_index = open_pcm_audio_index(original_path)?;
    let original_file_size = audio_index.file_len;
    let audio_tracks = audio_index
        .tracks
        .iter()
        .filter(|track| track.kind == TrackKind::Audio)
        .collect::<Vec<_>>();
    if audio_tracks.is_empty() {
        return Err("original MXF has no audio tracks".into());
    }
    let source_format = original_linux_pcm_audio_format(&audio_tracks)?;
    let source_total_samples = audio_audit_total_samples_from_index(&audio_index)?;
    let source_duration =
        duration_from_audio_samples(source_total_samples, source_format.sample_rate)?;
    if let Some(loudest_duration_ms) = options.find_loudest_range_ms {
        if original_file_size > MAX_MXF_FILE_BYTES {
            return Err("loudest-range scan for large MXF is not implemented yet; use --start-ms with --duration-ms for bounded extraction".into());
        }
        let loudest_duration_ms = bounded_audio_audit_duration_ms(loudest_duration_ms)?;
        let loudest_sample_count = u32::try_from(audio_samples_for_duration(
            Duration::from_millis(loudest_duration_ms),
            source_format.sample_rate,
        )?)?;
        let all_blocks = build_original_pcm_blocks_from_index_range(
            original_path,
            &audio_index,
            0,
            source_total_samples,
        )?;
        let all_track_groups =
            original_pcm_blocks_by_channel(&all_blocks, source_format.sample_rate)?;
        let loudest = find_loudest_audio_audit_range(
            &all_track_groups,
            source_format.sample_rate,
            loudest_sample_count,
        )?;
        start_ms = audio_sample_to_ms(loudest.start_sample, source_format.sample_rate)?;
        println!("Loudest-range scan:");
        println!("  requested duration: {loudest_duration_ms} ms");
        println!("  selected start sample: {}", loudest.start_sample);
        println!("  selected start time: {start_ms} ms");
        println!("  combined RMS: {:.6}", loudest.combined_rms);
        println!("  RMS per track: {}", f64_list(&loudest.rms_per_channel));
        println!("  peak per track: {}", f32_list(&loudest.peak_per_channel));
    }
    let requested_sample_count = u32::try_from(audio_samples_for_duration(
        Duration::from_millis(duration_ms),
        source_format.sample_rate,
    )?)?;
    let start_sample =
        audio_samples_for_duration(Duration::from_millis(start_ms), source_format.sample_rate)?;
    let end_sample = start_sample
        .checked_add(u64::from(requested_sample_count))
        .ok_or("audio audit sample range overflow")?;
    if end_sample > source_total_samples {
        return Err(format!(
            "audio audit range {}..{} exceeds source duration {} samples",
            start_sample, end_sample, source_total_samples
        )
        .into());
    }
    let blocks = build_original_pcm_blocks_from_index_range(
        original_path,
        &audio_index,
        start_sample,
        end_sample,
    )?;
    let track_groups = original_pcm_blocks_by_channel(&blocks, source_format.sample_rate)?;
    let selected_duration =
        duration_from_audio_samples(u64::from(requested_sample_count), source_format.sample_rate)?;
    let all_channels = track_groups.keys().copied().collect::<Vec<_>>();
    let four_channel_bytes = build_audio_audit_f32_interleaved_range(
        &track_groups,
        &all_channels,
        source_format.sample_rate,
        start_sample,
        requested_sample_count,
    )?;
    let four_channel_stats = f32_buffer_stats(
        &four_channel_bytes,
        requested_sample_count,
        u32::try_from(all_channels.len())?,
    )?;
    let source_stem = sanitized_source_stem(original_path);
    let output_dir = options.output_dir;
    let range_label = audio_audit_range_label(start_ms, duration_ms);
    let four_channel_path = options.diagnostic_wav_path.unwrap_or_else(|| {
        output_dir.join(format!(
            "{source_stem}-original-{}ch-f32-{range_label}.wav",
            all_channels.len()
        ))
    });
    write_f32_wav(
        &four_channel_path,
        source_format.sample_rate,
        u16::try_from(all_channels.len())?,
        &four_channel_bytes,
    )?;
    let mut diagnostic_wav_paths = vec![("original 4ch", four_channel_path.clone())];

    if all_channels.len() >= 2 {
        let track12 = all_channels.iter().take(2).copied().collect::<Vec<_>>();
        let track12_path = output_dir.join(format!(
            "{source_stem}-stereo-track12-f32-{range_label}.wav"
        ));
        let track12_bytes = build_audio_audit_f32_interleaved_range(
            &track_groups,
            &track12,
            source_format.sample_rate,
            start_sample,
            requested_sample_count,
        )?;
        write_f32_wav(&track12_path, source_format.sample_rate, 2, &track12_bytes)?;
        diagnostic_wav_paths.push(("stereo track 1/2 monitor", track12_path));
    }

    let track_summaries = audio_audit_track_summaries(
        &track_groups,
        source_format.sample_rate,
        start_sample,
        requested_sample_count,
    )?;
    if all_channels.len() >= 4 {
        let track34 = all_channels
            .iter()
            .skip(2)
            .take(2)
            .copied()
            .collect::<Vec<_>>();
        if track34.iter().any(|channel| {
            track_summaries
                .iter()
                .find(|summary| summary.channel_index == *channel)
                .is_some_and(|summary| !summary.stats.likely_silent)
        }) {
            let track34_path = output_dir.join(format!(
                "{source_stem}-stereo-track34-f32-{range_label}.wav"
            ));
            let track34_bytes = build_audio_audit_f32_interleaved_range(
                &track_groups,
                &track34,
                source_format.sample_rate,
                start_sample,
                requested_sample_count,
            )?;
            write_f32_wav(&track34_path, source_format.sample_rate, 2, &track34_bytes)?;
            diagnostic_wav_paths.push(("stereo track 3/4 monitor", track34_path));
        }
    }
    let loudest_pair = loudest_monitor_pair(&track_summaries);
    if loudest_pair.len() == 2 {
        let loudest_pair_path = output_dir.join(format!(
            "{source_stem}-stereo-loudest-pair-f32-{range_label}.wav"
        ));
        let loudest_pair_bytes = build_audio_audit_f32_interleaved_range(
            &track_groups,
            &loudest_pair,
            source_format.sample_rate,
            start_sample,
            requested_sample_count,
        )?;
        write_f32_wav(
            &loudest_pair_path,
            source_format.sample_rate,
            2,
            &loudest_pair_bytes,
        )?;
        diagnostic_wav_paths.push(("stereo loudest pair monitor", loudest_pair_path));
    }

    let runtime_comparison = if original_file_size <= MAX_MXF_FILE_BYTES {
        runtime_audio_audit_comparison(
            original_path,
            proxy_path,
            &track_groups,
            source_format.sample_rate,
        )
    } else {
        Err("runtime prepared payload comparison skipped for large MXF because that legacy path still uses full-file parsing".into())
    };

    println!("QGS PipeWire Audio Content Path Audit");
    println!("-------------------------------------");
    println!("Selected original path: {}", original_path.display());
    println!("Original MXF file size: {}", original_file_size);
    println!("Selected proxy path: {}", proxy_path.display());
    println!("Audio source: original MXF");
    println!("Video source for runtime timing: proxy MP4");
    println!("Proxy AAC: not used");
    println!("Bounded extraction: yes");
    println!("Full MXF loaded into memory: no");
    println!("PipeWire playback required: no");
    println!("AudioDeviceVerified: no");
    println!("Original audio content audibility confirmed: partial manual observation only");
    println!("Full playback: no");
    println!("Realtime playback: no");
    println!("A/V sync: no");
    println!("Selected start: {start_ms} ms");
    println!("Selected duration: {duration_ms} ms");
    println!(
        "Selected duration exact: {:.3} ms",
        selected_duration.as_secs_f64() * 1000.0
    );
    println!(
        "Source duration: {:.3} ms",
        source_duration.as_secs_f64() * 1000.0
    );
    println!("Analyzed sample range: {start_sample}..{end_sample}");
    println!(
        "Input PCM: tracks={} channels_per_track={} sample_rate={}Hz bit_depth={}bit",
        source_format.track_count,
        source_format.channels_per_track,
        source_format.sample_rate,
        source_format.bits_per_sample
    );
    println!("MXF declared byte order: not explicitly modeled by qgs-mxf descriptor");
    println!("QGS selected interpretation: signed 24-bit little-endian");
    println!("Selection reason: Step 20B/20C extraction stores Sony FX6 PCM payload as 24-bit little-endian runtime blocks");
    println!("Audio packet index entries: {}", audio_index.audio.len());
    println!("Blocks extracted for selected range: {}", blocks.len());
    println!(
        "Bytes extracted for selected range: {}",
        blocks
            .iter()
            .map(|block| block.payload.len())
            .sum::<usize>()
    );
    println!("Blocks per track:");
    for (channel_index, channel_blocks) in &track_groups {
        println!("  channel {channel_index}: {}", channel_blocks.len());
    }
    for summary in &track_summaries {
        let stats = &summary.stats;
        println!("Track/channel audit:");
        println!(
            "  track_id={} channel_index={}",
            summary.track_id, summary.channel_index
        );
        println!("  blocks={}", summary.block_count);
        println!("  block_byte_size={}", summary.block_byte_size);
        println!("  samples_per_block={}", summary.samples_per_block);
        println!("  raw first 12 bytes: {}", hex_bytes(&stats.raw_first_12));
        println!("  first samples signed24 LE: {:?}", stats.first_s24le);
        println!("  first samples signed24 BE: {:?}", stats.first_s24be);
        println!("  first samples unsigned24 LE: {:?}", stats.first_u24le);
        println!("  first 20 f32 samples: {}", f32_list(&stats.first_f32));
        println!("  samples analyzed: {}", stats.samples_analyzed);
        println!(
            "  min={} max={} mean={:.3} rms={:.3}",
            stats.min_sample, stats.max_sample, stats.mean, stats.rms
        );
        println!(
            "  peak_f32={:.6} rms_f32={:.6} dc_offset={:.6}",
            stats.peak_f32, stats.rms_f32, stats.dc_offset
        );
        println!("  relative level: {:.2} dBFS", stats.rms_dbfs);
        println!(
            "  zero_samples={:.3}% clipping={:.3}%",
            stats.zero_ratio * 100.0,
            stats.clipping_ratio * 100.0
        );
        println!("  likely silent: {}", yes_no(stats.likely_silent));
        println!(
            "  useful for monitoring: {}",
            yes_no(!stats.likely_silent && stats.rms_f32 >= 0.0001)
        );
        println!(
            "  likely DC/buzz risk: {}",
            yes_no(stats.likely_dc_buzz_risk)
        );
    }

    println!("Suggested diagnostic monitor pairs:");
    println!("  track 1/2: {}", monitor_pair_label(&all_channels, 0, 2));
    println!("  track 3/4: {}", monitor_pair_label(&all_channels, 2, 2));
    println!(
        "  loudest two tracks: {}",
        channel_pair_label(&loudest_pair)
    );
    println!(
        "Selected 4ch f32 min={:.6} max={:.6} rms={:.6} nonzero={}",
        four_channel_stats.min,
        four_channel_stats.max,
        four_channel_stats.rms,
        yes_no(four_channel_stats.nonzero)
    );
    match runtime_comparison {
        Ok(comparison) => {
            println!("Standalone segment path vs runtime prepared payload:");
            println!(
                "  segment source range start_sample: {}",
                comparison.segment_start_sample
            );
            println!(
                "  runtime source range start_sample: {}",
                comparison.runtime_start_sample
            );
            println!(
                "  segment output frames: {} bytes: {}",
                comparison.segment_frames, comparison.segment_bytes
            );
            println!(
                "  runtime output frames: {} bytes: {}",
                comparison.runtime_frames, comparison.runtime_bytes
            );
            println!(
                "  same source range f32 bytes identical: {}",
                yes_no(comparison.same_range_outputs_match)
            );
            println!(
                "  segment f32 min={:.6} max={:.6} rms={:.6} nonzero={}",
                comparison.segment_stats.min,
                comparison.segment_stats.max,
                comparison.segment_stats.rms,
                yes_no(comparison.segment_stats.nonzero)
            );
            println!(
                "  runtime f32 min={:.6} max={:.6} rms={:.6} nonzero={}",
                comparison.runtime_stats.min,
                comparison.runtime_stats.max,
                comparison.runtime_stats.rms,
                yes_no(comparison.runtime_stats.nonzero)
            );
            println!("Interleaved first frames:");
            println!(
                "  runtime first 5 frames: {}",
                comparison.runtime_first_frames
            );
        }
        Err(err) => {
            println!("Standalone segment path vs runtime prepared payload:");
            println!("  skipped: {err}");
        }
    }
    println!("PipeWire buffer geometry review:");
    println!("  format: F32Interleaved 48000Hz 4ch");
    println!("  bytes_per_sample: 4");
    println!("  bytes_per_frame: 16");
    println!("  960 frames expected bytes: 15360");
    println!("  1920 frames expected bytes: 30720");
    println!("  48000 frames expected bytes: 768000");
    println!("  qgs-audio-pipewire writes data plane 0, chunk offset 0, chunk stride channels*4, chunk size copied bytes");
    println!("Diagnostic WAV paths:");
    for (label, path) in diagnostic_wav_paths {
        println!("  {label}: {}", path.display());
    }
    println!("Likely root cause: source range/routing/gain remains the diagnostic focus; this command does not upgrade AudioDeviceVerified");
    Ok(())
}

fn pipewire_audio_mono_monitor(
    original_path: &Path,
    proxy_path: &Path,
    options: AudioMonoMonitorOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let plan = build_mono_monitor_diagnostic_plan(original_path, options)?;
    let first_buffer = plan
        .buffers
        .first()
        .cloned()
        .ok_or("mono monitor diagnostic produced no buffers")?;
    let report_buffer = LinuxPipewirePrototypeBuffer {
        sample_rate: plan.sample_rate,
        channels: 2,
        sample_count: plan.samples_per_buffer,
        sample_format: LinuxPipewirePrototypeSampleFormat::F32Interleaved,
        bytes: first_buffer,
    };
    let report = inspect_native_pipewire_stream_boundary(&report_buffer)?;

    println!("QGS PipeWire Desktop Mono Listening Helper");
    println!("------------------------------------------");
    println!("Audio source: original MXF");
    println!("Source original path: {}", original_path.display());
    println!("Source proxy path: {}", proxy_path.display());
    println!("Proxy AAC: not used");
    println!("Original MXF audio authoritative: yes");
    println!("Original audio model: separate mono tracks remain runtime truth");
    println!(
        "Selected source: original mono track {}",
        plan.source_track_number()
    );
    println!("Monitor output: duplicated mono to L/R for desktop listening only");
    println!("Desktop listening helper only: yes");
    println!("Discrete 4-mono broadcast output verified: no");
    println!("Production routing: no");
    println!("Channel certification: no");
    println!("Full playback: no");
    println!("Realtime playback: no");
    println!("A/V sync: no");
    println!("AudioDeviceVerified: no");
    println!("Bounded extraction: yes");
    println!("Full MXF loaded into memory: no");
    println!(
        "Source duration: {:.3} ms",
        plan.source_duration.as_secs_f64() * 1000.0
    );
    println!("Selected start: {} ms", plan.start_ms);
    println!("Selected duration: {} ms", plan.duration_ms);
    println!(
        "Selected duration exact: {:.3} ms",
        plan.selected_duration.as_secs_f64() * 1000.0
    );
    println!(
        "Selected sample range: {}..{}",
        plan.start_sample, plan.end_sample
    );
    println!(
        "Selected source track: track {}",
        plan.source_track_number()
    );
    println!("Selected source track is mono source: yes");
    println!(
        "Output routing: track {} mono duplicated -> FL/FR",
        plan.source_track_number()
    );
    println!("Output format: F32Interleaved 48000Hz 2ch");
    println!("Output frames: {}", plan.output_frames);
    println!("Output bytes planned: {}", plan.total_bytes);
    println!("Output buffers planned: {}", plan.buffers.len());
    println!("Samples per selected source track: {}", plan.output_frames);
    println!("Manual confirmation required: yes");
    for summary in &plan.selected_track_summaries {
        println!("Selected mono track audit:");
        println!(
            "  track {} channel_index={}",
            u32::from(summary.channel_index) + 1,
            summary.channel_index
        );
        println!("  peak_f32={:.6}", summary.stats.peak_f32);
        println!("  rms_f32={:.6}", summary.stats.rms_f32);
        println!("  relative level: {:.2} dBFS", summary.stats.rms_dbfs);
        println!("  likely silent: {}", yes_no(summary.stats.likely_silent));
    }
    println!(
        "PipeWire runtime library available: {}",
        yes_no(report.runtime_library_available)
    );
    println!(
        "PipeWire server reachable: {}",
        yes_no(report.pipewire_server_reachable)
    );
    println!(
        "PipeWire pkg-config entry available: {}",
        yes_no(report.pkg_config_entry_available)
    );
    println!(
        "PipeWire headers available: {}",
        yes_no(report.headers_available)
    );
    if !report.pkg_config_entry_available || !report.headers_available {
        println!(
            "Stream create attempted: {}",
            yes_no(report.stream_create_attempted)
        );
        println!("Buffers submitted: 0");
        println!("Drain requested: no");
        println!("Drain completed: no");
        println!(
            "Manual confirmation status: ManualDesktopMonoListeningHelperConfirmationRequired"
        );
        println!("Evidence level: {:?}", report.evidence_level);
        println!("AudioDeviceBoundaryEvidence: DesktopMonoListeningHelperNotSubmitted");
        println!("AudioDeviceVerified: no");
        println!("AudioDeviceVerified scope: not upgraded by desktop mono listening helper");
        println!("Audible output claimed: no");
        println!("Status: {}", report.status_message);
        return Ok(());
    }

    let stream_format = PipeWireStreamFormat {
        sample_rate: plan.sample_rate,
        channels: 2,
        sample_format: PipeWireAudioSampleFormat::F32Interleaved,
    };
    let submission_report = submit_native_pipewire_buffers(
        stream_format,
        plan.buffers,
        plan.samples_per_buffer,
        Duration::from_secs(5),
    )?;
    let confirmation = read_manual_mono_monitor_confirmation()?;
    let evidence = mono_monitor_diagnostic_evidence_label(
        submission_report.buffer_submitted,
        submission_report.drain_completed,
        confirmation,
    );

    println!(
        "Stream configured: {}",
        yes_no(submission_report.stream_report.stream_configured)
    );
    println!(
        "Observed stream states: {:?}",
        submission_report.stream_report.observed_states
    );
    println!(
        "Final stream state: {:?}",
        submission_report.stream_report.final_state
    );
    println!(
        "Process callback reached: {}",
        yes_no(submission_report.process_callback_reached)
    );
    println!("Buffer capacity: {}", submission_report.buffer_capacity);
    println!("Buffers planned: {}", submission_report.buffers_planned);
    println!("Buffers submitted: {}", submission_report.buffers_submitted);
    println!(
        "Samples submitted for selected mono source track: {}",
        submission_report.samples_converted
    );
    println!("Output channels: {}", submission_report.output_channels);
    println!(
        "F32 samples written: {}",
        submission_report.f32_samples_written
    );
    println!("Bytes copied: {}", submission_report.bytes_copied);
    println!(
        "Drain requested: {}",
        yes_no(submission_report.drain_requested)
    );
    println!(
        "Drain completed: {}",
        yes_no(submission_report.drain_completed)
    );
    println!(
        "Post-submit callbacks observed: {}",
        submission_report.post_submit_process_callbacks
    );
    println!(
        "Post-submit timeout: {}",
        yes_no(submission_report.post_submit_timeout)
    );
    println!(
        "PipeWire evidence level: {:?}",
        submission_report.evidence_level
    );
    println!(
        "Manual confirmation status: {}",
        confirmation.mono_monitor_label()
    );
    println!("Desktop mono listening helper evidence level: {}", evidence);
    println!("AudioDeviceBoundaryEvidence: {}", evidence);
    println!("AudioDeviceVerified: no");
    println!("AudioDeviceVerified scope: not upgraded by desktop mono listening helper");
    println!(
        "Audible output claimed: {}",
        yes_no(confirmation == ManualAudibleConfirmation::Yes)
    );
    println!(
        "Status: desktop mono listening helper completed; discrete 4-mono output and production routing are not verified"
    );
    Ok(())
}

fn pipewire_audio_discrete_4mono(
    original_path: &Path,
    proxy_path: &Path,
    options: AudioDiscrete4MonoOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let plan = build_discrete_4mono_output_plan(original_path, options)?;
    let first_buffer = plan
        .buffers
        .first()
        .cloned()
        .ok_or("discrete 4-mono output produced no buffers")?;
    let report_buffer = LinuxPipewirePrototypeBuffer {
        sample_rate: plan.sample_rate,
        channels: 4,
        sample_count: plan.samples_per_buffer,
        sample_format: LinuxPipewirePrototypeSampleFormat::F32Interleaved,
        bytes: first_buffer,
    };
    let report = inspect_native_pipewire_stream_boundary(&report_buffer)?;

    println!("Native PipeWire Discrete 4-Mono Output Boundary");
    println!("-----------------------------------------------");
    println!("Audio source: original MXF");
    println!("Source original path: {}", original_path.display());
    println!("Source proxy path: {}", proxy_path.display());
    println!("Proxy AAC: not used");
    println!("Original MXF audio authoritative: yes");
    println!("Source mapping:");
    println!("  track 1 -> output channel 1");
    println!("  track 2 -> output channel 2");
    println!("  track 3 -> output channel 3");
    println!("  track 4 -> output channel 4");
    println!("Stereo fold: no");
    println!("Duplicated mono: no");
    println!("Production routing: no");
    println!("Channel certification: no");
    println!("Full playback: no");
    println!("Realtime playback: no");
    println!("A/V sync: no");
    println!("AudioDeviceVerified: no");
    println!("Bounded extraction: yes");
    println!("Full MXF loaded into memory: no");
    println!(
        "Source duration: {:.3} ms",
        plan.source_duration.as_secs_f64() * 1000.0
    );
    println!("Selected start: {} ms", plan.start_ms);
    println!("Selected duration: {} ms", plan.duration_ms);
    println!(
        "Selected duration exact: {:.3} ms",
        plan.selected_duration.as_secs_f64() * 1000.0
    );
    println!(
        "Selected sample range: {}..{}",
        plan.start_sample, plan.end_sample
    );
    println!("Output format: F32Interleaved 48000Hz 4ch");
    println!("Output channels: {}", plan.output_channels);
    println!("Output frames: {}", plan.output_frames);
    println!("Output bytes planned: {}", plan.total_bytes);
    println!("Output buffers planned: {}", plan.buffers.len());
    println!("Samples per source track: {}", plan.output_frames);
    println!("PipeWire API position labels: FL/FR/RL/RR if accepted by the default sink");
    println!("QGS source identity: track 1/2/3/4 remains authoritative");
    for summary in &plan.track_summaries {
        println!("Source mono track audit:");
        println!(
            "  track {} -> output channel {}",
            u32::from(summary.channel_index) + 1,
            u32::from(summary.channel_index) + 1
        );
        println!("  peak_f32={:.6}", summary.stats.peak_f32);
        println!("  rms_f32={:.6}", summary.stats.rms_f32);
        println!("  relative level: {:.2} dBFS", summary.stats.rms_dbfs);
        println!("  likely silent: {}", yes_no(summary.stats.likely_silent));
    }
    println!(
        "PipeWire runtime library available: {}",
        yes_no(report.runtime_library_available)
    );
    println!(
        "PipeWire server reachable: {}",
        yes_no(report.pipewire_server_reachable)
    );
    println!(
        "PipeWire pkg-config entry available: {}",
        yes_no(report.pkg_config_entry_available)
    );
    println!(
        "PipeWire headers available: {}",
        yes_no(report.headers_available)
    );
    if !report.pkg_config_entry_available || !report.headers_available {
        println!(
            "Stream create attempted: {}",
            yes_no(report.stream_create_attempted)
        );
        println!("Buffers submitted: 0");
        println!("Drain requested: no");
        println!("Drain completed: no");
        println!("Evidence level: {:?}", report.evidence_level);
        println!("AudioDeviceBoundaryEvidence: Discrete4MonoOutputNotSubmitted");
        println!("AudioDeviceVerified: no");
        println!("AudioDeviceVerified scope: not upgraded by discrete 4-mono boundary");
        println!("Status: {}", report.status_message);
        return Ok(());
    }

    let stream_format = PipeWireStreamFormat {
        sample_rate: plan.sample_rate,
        channels: 4,
        sample_format: PipeWireAudioSampleFormat::F32Interleaved,
    };
    let submission_report = submit_native_pipewire_buffers(
        stream_format,
        plan.buffers,
        plan.samples_per_buffer,
        Duration::from_secs(5),
    )?;
    let evidence = discrete_4mono_evidence_label(
        submission_report.buffer_submitted,
        submission_report.drain_completed,
    );

    println!(
        "Stream configured: {}",
        yes_no(submission_report.stream_report.stream_configured)
    );
    println!(
        "Observed stream states: {:?}",
        submission_report.stream_report.observed_states
    );
    println!(
        "Final stream state: {:?}",
        submission_report.stream_report.final_state
    );
    println!(
        "Process callback reached: {}",
        yes_no(submission_report.process_callback_reached)
    );
    println!("Buffer capacity: {}", submission_report.buffer_capacity);
    println!("Buffers planned: {}", submission_report.buffers_planned);
    println!("Buffers submitted: {}", submission_report.buffers_submitted);
    println!(
        "Samples submitted per source track: {}",
        submission_report.samples_converted
    );
    println!("Output channels: {}", submission_report.output_channels);
    println!(
        "F32 samples written: {}",
        submission_report.f32_samples_written
    );
    println!("Bytes copied: {}", submission_report.bytes_copied);
    println!(
        "Drain requested: {}",
        yes_no(submission_report.drain_requested)
    );
    println!(
        "Drain completed: {}",
        yes_no(submission_report.drain_completed)
    );
    println!(
        "Post-submit callbacks observed: {}",
        submission_report.post_submit_process_callbacks
    );
    println!(
        "Post-submit timeout: {}",
        yes_no(submission_report.post_submit_timeout)
    );
    println!(
        "PipeWire evidence level: {:?}",
        submission_report.evidence_level
    );
    println!("Discrete 4-mono evidence level: {}", evidence);
    println!("AudioDeviceBoundaryEvidence: {}", evidence);
    println!("AudioDeviceVerified: no");
    println!("AudioDeviceVerified scope: not upgraded by discrete 4-mono boundary");
    println!("Physical channel mapping certified: no");
    println!(
        "Status: discrete 4-mono boundary completed; production routing and physical channel mapping are not certified"
    );
    Ok(())
}

struct AudioContentAuditOptions {
    start_ms: u64,
    duration_ms: u64,
    output_dir: PathBuf,
    find_loudest_range_ms: Option<u64>,
    diagnostic_wav_path: Option<PathBuf>,
}

#[derive(Clone, Copy)]
struct AudioMonoMonitorOptions {
    start_ms: u64,
    duration_ms: u64,
    source_channel_index: u16,
}

#[derive(Clone, Copy)]
struct AudioDiscrete4MonoOptions {
    start_ms: u64,
    duration_ms: u64,
}

struct MonoMonitorDiagnosticPlan {
    sample_rate: u32,
    start_ms: u64,
    duration_ms: u64,
    start_sample: u64,
    end_sample: u64,
    source_duration: Duration,
    selected_duration: Duration,
    source_channel_index: u16,
    samples_per_buffer: u32,
    output_frames: u32,
    total_bytes: usize,
    buffers: Vec<Vec<u8>>,
    selected_track_summaries: Vec<AudioAuditTrackSummary>,
}

struct Discrete4MonoOutputPlan {
    sample_rate: u32,
    start_ms: u64,
    duration_ms: u64,
    start_sample: u64,
    end_sample: u64,
    source_duration: Duration,
    selected_duration: Duration,
    samples_per_buffer: u32,
    output_channels: u16,
    output_frames: u32,
    total_bytes: usize,
    buffers: Vec<Vec<u8>>,
    track_summaries: Vec<AudioAuditTrackSummary>,
}

impl MonoMonitorDiagnosticPlan {
    const fn source_track_number(&self) -> u16 {
        self.source_channel_index + 1
    }
}

fn build_mono_monitor_diagnostic_plan(
    original_path: &Path,
    options: AudioMonoMonitorOptions,
) -> Result<MonoMonitorDiagnosticPlan, Box<dyn std::error::Error>> {
    let duration_ms = bounded_audio_audit_duration_ms(options.duration_ms)?;
    let audio_index = open_pcm_audio_index(original_path)?;
    let audio_tracks = audio_index
        .tracks
        .iter()
        .filter(|track| track.kind == TrackKind::Audio)
        .collect::<Vec<_>>();
    if audio_tracks.is_empty() {
        return Err("original MXF has no audio tracks".into());
    }
    let source_format = original_linux_pcm_audio_format(&audio_tracks)?;
    if source_format.sample_rate != 48_000 {
        return Err("mono monitor diagnostic currently expects 48 kHz original PCM".into());
    }
    let source_total_samples = audio_audit_total_samples_from_index(&audio_index)?;
    let source_duration =
        duration_from_audio_samples(source_total_samples, source_format.sample_rate)?;
    let requested_sample_count = u32::try_from(audio_samples_for_duration(
        Duration::from_millis(duration_ms),
        source_format.sample_rate,
    )?)?;
    let start_sample = audio_samples_for_duration(
        Duration::from_millis(options.start_ms),
        source_format.sample_rate,
    )?;
    let end_sample = start_sample
        .checked_add(u64::from(requested_sample_count))
        .ok_or("mono monitor sample range overflow")?;
    if end_sample > source_total_samples {
        return Err(format!(
            "mono monitor range {}..{} exceeds source duration {} samples",
            start_sample, end_sample, source_total_samples
        )
        .into());
    }
    let blocks = build_original_pcm_blocks_from_index_range(
        original_path,
        &audio_index,
        start_sample,
        end_sample,
    )?;
    let track_groups = original_pcm_blocks_by_channel(&blocks, source_format.sample_rate)?;
    if !track_groups.contains_key(&options.source_channel_index) {
        return Err(format!(
            "requested mono monitor track {} is missing",
            u32::from(options.source_channel_index) + 1
        )
        .into());
    }
    let source_blocks = track_groups
        .get(&options.source_channel_index)
        .ok_or("requested mono monitor track is missing")?;
    let samples_per_buffer = source_blocks
        .first()
        .ok_or("requested mono monitor track has no blocks")?
        .sample_count;
    if samples_per_buffer == 0 {
        return Err("mono monitor source block has zero samples".into());
    }
    if requested_sample_count % samples_per_buffer != 0 {
        return Err("mono monitor duration must align to source PCM block size".into());
    }
    let buffers_needed = usize::try_from(requested_sample_count / samples_per_buffer)?;
    if source_blocks.len() != buffers_needed {
        return Err("mono monitor selected range is not fully covered by source blocks".into());
    }
    let mut buffers = Vec::with_capacity(buffers_needed);
    for block_index in 0..buffers_needed {
        let block_start = start_sample
            .checked_add(
                u64::try_from(block_index)?
                    .checked_mul(u64::from(samples_per_buffer))
                    .ok_or("mono monitor block start overflow")?,
            )
            .ok_or("mono monitor block start overflow")?;
        let channels = [options.source_channel_index, options.source_channel_index];
        buffers.push(build_audio_audit_f32_interleaved_range(
            &track_groups,
            &channels,
            source_format.sample_rate,
            block_start,
            samples_per_buffer,
        )?);
    }
    let selected_duration =
        duration_from_audio_samples(u64::from(requested_sample_count), source_format.sample_rate)?;
    let selected_track_summaries = audio_audit_track_summaries(
        &track_groups,
        source_format.sample_rate,
        start_sample,
        requested_sample_count,
    )?
    .into_iter()
    .filter(|summary| summary.channel_index == options.source_channel_index)
    .collect::<Vec<_>>();
    if selected_track_summaries.len() != 1 {
        return Err("mono monitor selected track summary is missing".into());
    }
    let total_bytes = buffers.iter().map(Vec::len).sum();
    Ok(MonoMonitorDiagnosticPlan {
        sample_rate: source_format.sample_rate,
        start_ms: options.start_ms,
        duration_ms,
        start_sample,
        end_sample,
        source_duration,
        selected_duration,
        source_channel_index: options.source_channel_index,
        samples_per_buffer,
        output_frames: requested_sample_count,
        total_bytes,
        buffers,
        selected_track_summaries,
    })
}

fn build_discrete_4mono_output_plan(
    original_path: &Path,
    options: AudioDiscrete4MonoOptions,
) -> Result<Discrete4MonoOutputPlan, Box<dyn std::error::Error>> {
    let duration_ms = bounded_audio_audit_duration_ms(options.duration_ms)?;
    let audio_index = open_pcm_audio_index(original_path)?;
    let audio_tracks = audio_index
        .tracks
        .iter()
        .filter(|track| track.kind == TrackKind::Audio)
        .collect::<Vec<_>>();
    if audio_tracks.is_empty() {
        return Err("original MXF has no audio tracks".into());
    }
    let source_format = original_linux_pcm_audio_format(&audio_tracks)?;
    if source_format.sample_rate != 48_000 {
        return Err("discrete 4-mono output currently expects 48 kHz original PCM".into());
    }
    let source_total_samples = audio_audit_total_samples_from_index(&audio_index)?;
    let source_duration =
        duration_from_audio_samples(source_total_samples, source_format.sample_rate)?;
    let requested_sample_count = u32::try_from(audio_samples_for_duration(
        Duration::from_millis(duration_ms),
        source_format.sample_rate,
    )?)?;
    let start_sample = audio_samples_for_duration(
        Duration::from_millis(options.start_ms),
        source_format.sample_rate,
    )?;
    let end_sample = start_sample
        .checked_add(u64::from(requested_sample_count))
        .ok_or("discrete 4-mono sample range overflow")?;
    if end_sample > source_total_samples {
        return Err(format!(
            "discrete 4-mono range {}..{} exceeds source duration {} samples",
            start_sample, end_sample, source_total_samples
        )
        .into());
    }
    let blocks = build_original_pcm_blocks_from_index_range(
        original_path,
        &audio_index,
        start_sample,
        end_sample,
    )?;
    let track_groups = original_pcm_blocks_by_channel(&blocks, source_format.sample_rate)?;
    let output_channels = [0_u16, 1, 2, 3];
    for channel in output_channels {
        if !track_groups.contains_key(&channel) {
            return Err(format!(
                "required original mono track {} is missing",
                u32::from(channel) + 1
            )
            .into());
        }
    }
    let first_channel_blocks = track_groups
        .get(&0)
        .ok_or("required original mono track 1 is missing")?;
    let samples_per_buffer = first_channel_blocks
        .first()
        .ok_or("discrete 4-mono source has no blocks")?
        .sample_count;
    if samples_per_buffer == 0 {
        return Err("discrete 4-mono source block has zero samples".into());
    }
    if requested_sample_count % samples_per_buffer != 0 {
        return Err("discrete 4-mono duration must align to source PCM block size".into());
    }
    let buffers_needed = usize::try_from(requested_sample_count / samples_per_buffer)?;
    for channel in output_channels {
        let blocks = track_groups
            .get(&channel)
            .ok_or("required original mono track is missing")?;
        if blocks.len() != buffers_needed {
            return Err(
                "discrete 4-mono selected range is not fully covered by every track".into(),
            );
        }
    }

    let mut buffers = Vec::with_capacity(buffers_needed);
    for block_index in 0..buffers_needed {
        let block_start = start_sample
            .checked_add(
                u64::try_from(block_index)?
                    .checked_mul(u64::from(samples_per_buffer))
                    .ok_or("discrete 4-mono block start overflow")?,
            )
            .ok_or("discrete 4-mono block start overflow")?;
        buffers.push(build_audio_audit_f32_interleaved_range(
            &track_groups,
            &output_channels,
            source_format.sample_rate,
            block_start,
            samples_per_buffer,
        )?);
    }
    let selected_duration =
        duration_from_audio_samples(u64::from(requested_sample_count), source_format.sample_rate)?;
    let track_summaries = audio_audit_track_summaries(
        &track_groups,
        source_format.sample_rate,
        start_sample,
        requested_sample_count,
    )?
    .into_iter()
    .filter(|summary| output_channels.contains(&summary.channel_index))
    .collect::<Vec<_>>();
    if track_summaries.len() != 4 {
        return Err("discrete 4-mono track summaries are incomplete".into());
    }
    let total_bytes = buffers.iter().map(Vec::len).sum();
    Ok(Discrete4MonoOutputPlan {
        sample_rate: source_format.sample_rate,
        start_ms: options.start_ms,
        duration_ms,
        start_sample,
        end_sample,
        source_duration,
        selected_duration,
        samples_per_buffer,
        output_channels: 4,
        output_frames: requested_sample_count,
        total_bytes,
        buffers,
        track_summaries,
    })
}

struct TrackContentStats {
    raw_first_12: Vec<u8>,
    first_s24le: Vec<i32>,
    first_s24be: Vec<i32>,
    first_u24le: Vec<u32>,
    first_f32: Vec<f32>,
    samples_analyzed: usize,
    min_sample: i32,
    max_sample: i32,
    mean: f64,
    rms: f64,
    peak_f32: f32,
    rms_f32: f64,
    rms_dbfs: f64,
    dc_offset: f64,
    zero_ratio: f64,
    clipping_ratio: f64,
    likely_silent: bool,
    likely_dc_buzz_risk: bool,
}

struct AudioAuditTrackSummary {
    track_id: u32,
    channel_index: u16,
    block_count: usize,
    block_byte_size: usize,
    samples_per_block: u32,
    stats: TrackContentStats,
}

struct AudioAuditRuntimeComparison {
    segment_start_sample: u64,
    runtime_start_sample: u64,
    segment_frames: u32,
    runtime_frames: u32,
    segment_bytes: usize,
    runtime_bytes: usize,
    same_range_outputs_match: bool,
    segment_stats: F32BufferStats,
    runtime_stats: F32BufferStats,
    runtime_first_frames: String,
}

struct LoudestAudioRange {
    start_sample: u64,
    combined_rms: f64,
    rms_per_channel: Vec<f64>,
    peak_per_channel: Vec<f32>,
}

struct F32BufferStats {
    min: f32,
    max: f32,
    rms: f64,
    nonzero: bool,
}

fn original_pcm_blocks_by_channel(
    blocks: &[PcmAudioBlock],
    sample_rate: u32,
) -> Result<BTreeMap<u16, Vec<&PcmAudioBlock>>, Box<dyn std::error::Error>> {
    let mut by_channel: BTreeMap<u16, Vec<&PcmAudioBlock>> = BTreeMap::new();
    for block in blocks {
        if block.sample_rate != sample_rate {
            return Err("audio content audit found mixed sample rates".into());
        }
        let PcmAudioBlockLayout::MonoTrack { channel_index, .. } = block.layout else {
            return Err("audio content audit expects mono-track PCM blocks".into());
        };
        by_channel.entry(channel_index).or_default().push(block);
    }
    for channel_blocks in by_channel.values_mut() {
        channel_blocks.sort_by_key(|block| block.start_time);
    }
    Ok(by_channel)
}

fn bounded_audio_audit_duration_ms(duration_ms: u64) -> Result<u64, Box<dyn std::error::Error>> {
    if duration_ms == 0 {
        return Err("audio audit duration must be greater than zero".into());
    }
    if duration_ms > 5_000 {
        return Err("audio audit duration is bounded to 5000 ms".into());
    }
    Ok(duration_ms)
}

fn audio_sample_to_ms(sample: u64, sample_rate: u32) -> Result<u64, Box<dyn std::error::Error>> {
    sample
        .checked_mul(1_000)
        .map(|value| value / u64::from(sample_rate))
        .ok_or_else(|| "audio sample to ms overflow".into())
}

fn audio_audit_total_samples_from_index(
    index: &PcmAudioIndex,
) -> Result<u64, Box<dyn std::error::Error>> {
    let mut totals: BTreeMap<u16, u64> = BTreeMap::new();
    for entry in &index.audio {
        let end = entry
            .start_sample
            .checked_add(u64::from(entry.sample_count))
            .ok_or("audio index sample range overflow")?;
        totals
            .entry(entry.channel_index)
            .and_modify(|current| *current = (*current).max(end))
            .or_insert(end);
    }
    totals
        .values()
        .copied()
        .min()
        .ok_or_else(|| "audio index has no PCM entries".into())
}

fn audio_audit_track_summaries(
    track_groups: &BTreeMap<u16, Vec<&PcmAudioBlock>>,
    sample_rate: u32,
    start_sample: u64,
    sample_count: u32,
) -> Result<Vec<AudioAuditTrackSummary>, Box<dyn std::error::Error>> {
    let mut summaries = Vec::new();
    for (channel_index, channel_blocks) in track_groups {
        let first = channel_blocks
            .first()
            .ok_or("channel group unexpectedly empty")?;
        if first.sample_rate != sample_rate {
            return Err("audio audit summary found mixed sample rates".into());
        }
        let PcmAudioBlockLayout::MonoTrack { track_id, .. } = first.layout else {
            return Err("audit expects mono-track original PCM blocks".into());
        };
        summaries.push(AudioAuditTrackSummary {
            track_id,
            channel_index: *channel_index,
            block_count: channel_blocks.len(),
            block_byte_size: first.payload.len(),
            samples_per_block: first.sample_count,
            stats: audit_track_stats_range(channel_blocks, start_sample, sample_count)?,
        });
    }
    Ok(summaries)
}

fn build_audio_audit_f32_interleaved_range(
    track_groups: &BTreeMap<u16, Vec<&PcmAudioBlock>>,
    channels: &[u16],
    sample_rate: u32,
    start_sample: u64,
    sample_count: u32,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    if channels.is_empty() {
        return Err("audio audit output requires at least one channel".into());
    }
    let mut output = Vec::with_capacity(
        usize::try_from(sample_count)?
            .checked_mul(channels.len())
            .and_then(|samples| samples.checked_mul(4))
            .ok_or("audio audit f32 output size overflow")?,
    );
    let end_sample = start_sample
        .checked_add(u64::from(sample_count))
        .ok_or("audio audit output range overflow")?;
    for absolute_sample in start_sample..end_sample {
        for channel in channels {
            let blocks = track_groups
                .get(channel)
                .ok_or("requested audio audit channel is missing")?;
            let sample = sample_at_absolute(blocks, absolute_sample)?;
            let value = (sample as f32 / 8_388_608.0).clamp(-1.0, 1.0);
            output.extend_from_slice(&value.to_le_bytes());
        }
    }
    let expected = usize::try_from(sample_count)?
        .checked_mul(channels.len())
        .and_then(|samples| samples.checked_mul(4))
        .ok_or("audio audit expected output size overflow")?;
    if output.len() != expected {
        return Err("audio audit f32 output size mismatch".into());
    }
    let _ = sample_rate;
    Ok(output)
}

fn find_loudest_audio_audit_range(
    track_groups: &BTreeMap<u16, Vec<&PcmAudioBlock>>,
    sample_rate: u32,
    sample_count: u32,
) -> Result<LoudestAudioRange, Box<dyn std::error::Error>> {
    let first_channel = track_groups.values().next().ok_or("no audio tracks")?;
    let first_block = first_channel.first().ok_or("no audio blocks")?;
    let samples_per_block = first_block.sample_count;
    if samples_per_block == 0 {
        return Err("audio blocks have zero samples".into());
    }
    let blocks_per_window = usize::try_from(sample_count.div_ceil(samples_per_block))?;
    if blocks_per_window == 0 {
        return Err("loudest range selected zero blocks".into());
    }
    let min_blocks = track_groups
        .values()
        .map(Vec::len)
        .min()
        .ok_or("no audio track groups")?;
    if min_blocks < blocks_per_window {
        return Err("not enough audio blocks for loudest range duration".into());
    }

    let mut combined_prefix = vec![0.0_f64; min_blocks + 1];
    for block_index in 0..min_blocks {
        let mut block_sum_sq = 0_f64;
        for blocks in track_groups.values() {
            let block = blocks[block_index];
            for sample_index in 0..usize::try_from(block.sample_count)? {
                let sample =
                    f64::from(decode_s24le_i32(&block.payload, sample_index)) / 8_388_608.0;
                block_sum_sq += sample * sample;
            }
        }
        combined_prefix[block_index + 1] = combined_prefix[block_index] + block_sum_sq;
    }

    let mut best_start_block = 0_usize;
    let mut best_sum_sq = -1.0_f64;
    for start_block in 0..=min_blocks - blocks_per_window {
        let end_block = start_block + blocks_per_window;
        let sum_sq = combined_prefix[end_block] - combined_prefix[start_block];
        if sum_sq > best_sum_sq {
            best_sum_sq = sum_sq;
            best_start_block = start_block;
        }
    }
    let start_sample = u64::try_from(best_start_block)?
        .checked_mul(u64::from(samples_per_block))
        .ok_or("loudest range start sample overflow")?;
    let mut rms_per_channel = Vec::new();
    let mut peak_per_channel = Vec::new();
    for blocks in track_groups.values() {
        let stats = audit_track_stats_range(blocks, start_sample, sample_count)?;
        rms_per_channel.push(stats.rms_f32);
        peak_per_channel.push(stats.peak_f32);
    }
    let combined_rms = (best_sum_sq
        / (blocks_per_window as f64 * f64::from(samples_per_block) * track_groups.len() as f64))
        .sqrt();
    let _ = sample_rate;
    Ok(LoudestAudioRange {
        start_sample,
        combined_rms,
        rms_per_channel,
        peak_per_channel,
    })
}

fn runtime_audio_audit_comparison(
    original_path: &Path,
    proxy_path: &Path,
    track_groups: &BTreeMap<u16, Vec<&PcmAudioBlock>>,
    sample_rate: u32,
) -> Result<AudioAuditRuntimeComparison, Box<dyn std::error::Error>> {
    let runtime_plan = prepare_runtime_audio_pipewire_payload(original_path, proxy_path)?;
    let runtime_buffer = &runtime_plan.payload_buffer;
    let channels = track_groups.keys().copied().collect::<Vec<_>>();
    let segment_bytes = build_audio_audit_f32_interleaved_range(
        track_groups,
        &channels,
        sample_rate,
        runtime_plan.audio_binding.start_sample,
        runtime_buffer.output_frames,
    )?;
    let same_range_outputs_match = segment_bytes == runtime_buffer.output_bytes;
    let segment_stats = f32_buffer_stats(
        &segment_bytes,
        runtime_buffer.output_frames,
        u32::try_from(channels.len())?,
    )?;
    let runtime_stats = f32_buffer_stats(
        &runtime_buffer.output_bytes,
        runtime_buffer.output_frames,
        u32::try_from(runtime_buffer.track_count)?,
    )?;
    Ok(AudioAuditRuntimeComparison {
        segment_start_sample: runtime_plan.audio_binding.start_sample,
        runtime_start_sample: runtime_plan.audio_binding.start_sample,
        segment_frames: runtime_buffer.output_frames,
        runtime_frames: runtime_buffer.output_frames,
        segment_bytes: segment_bytes.len(),
        runtime_bytes: runtime_buffer.output_bytes.len(),
        same_range_outputs_match,
        segment_stats,
        runtime_stats,
        runtime_first_frames: interleaved_frame_list(
            &runtime_buffer.output_bytes,
            runtime_buffer.track_count,
            5,
        )?,
    })
}

fn sanitized_source_stem(path: &Path) -> String {
    let raw = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("original-audio");
    let mut out = String::new();
    let mut last_dash = false;
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
            last_dash = false;
        } else if !last_dash {
            out.push('-');
            last_dash = true;
        }
    }
    let out = out.trim_matches('-').to_string();
    if out.is_empty() {
        "original-audio".to_string()
    } else {
        out
    }
}

fn public_source_uri_stem(public_source_uri: &str) -> String {
    let raw = public_source_uri
        .rsplit('/')
        .next()
        .filter(|stem| !stem.is_empty())
        .unwrap_or("qgs-video");
    let mut out = String::new();
    let mut last_dash = false;
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
            last_dash = false;
        } else if !last_dash {
            out.push('-');
            last_dash = true;
        }
    }
    let out = out.trim_matches('-').to_string();
    if out.is_empty() {
        "qgs-video".to_string()
    } else {
        out
    }
}

fn file_presenter_manifest_file_name(public_source_uri: &str, source_frame_index: u64) -> String {
    format!(
        "{}-frame{source_frame_index:06}-proxy-preview-manifest.json",
        public_source_uri_stem(public_source_uri)
    )
}

fn gpu_payload_readback_image_file_name(
    public_source_uri: &str,
    source_frame_index: u64,
) -> String {
    format!(
        "{}-frame{source_frame_index:06}-proxy-preview-readback.ppm",
        public_source_uri_stem(public_source_uri)
    )
}

fn gpu_payload_readback_manifest_file_name(
    public_source_uri: &str,
    source_frame_index: u64,
) -> String {
    format!(
        "{}-frame{source_frame_index:06}-proxy-preview-readback-manifest.json",
        public_source_uri_stem(public_source_uri)
    )
}

fn rgba_u16_to_rgb8_bytes(
    rgba_u16: &[u16],
    width: u32,
    height: u32,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let pixel_count = usize::try_from(width)?
        .checked_mul(usize::try_from(height)?)
        .ok_or("readback image dimensions overflow")?;
    let expected_samples = pixel_count
        .checked_mul(4)
        .ok_or("readback RGBA sample count overflow")?;
    if rgba_u16.len() != expected_samples {
        return Err(format!(
            "readback RGBA sample count mismatch: expected {expected_samples}, got {}",
            rgba_u16.len()
        )
        .into());
    }
    let mut rgb = Vec::with_capacity(
        pixel_count
            .checked_mul(3)
            .ok_or("readback RGB byte count overflow")?,
    );
    for pixel in rgba_u16.chunks_exact(4) {
        rgb.push((pixel[0] >> 8) as u8);
        rgb.push((pixel[1] >> 8) as u8);
        rgb.push((pixel[2] >> 8) as u8);
    }
    Ok(rgb)
}

fn rgba_u16_to_ppm_p6_rgb8(
    rgba_u16: &[u16],
    width: u32,
    height: u32,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let rgb = rgba_u16_to_rgb8_bytes(rgba_u16, width, height)?;
    let mut ppm = format!("P6\n{width} {height}\n255\n").into_bytes();
    ppm.extend_from_slice(&rgb);
    Ok(ppm)
}

fn json_escape(value: &str) -> String {
    let mut escaped = String::new();
    for ch in value.chars() {
        match ch {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            ch if ch.is_control() => escaped.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => escaped.push(ch),
        }
    }
    escaped
}

fn manifest_json_contains_private_path(manifest_json: &str) -> bool {
    manifest_json.contains("/home/")
        || manifest_json.contains("/tmp/")
        || manifest_json.contains("/mnt/")
        || manifest_json.contains("file://")
}

fn qgs_file_presenter_manifest_json(
    result: &QgsFilePresenterSubmissionResult,
    monitor_projection: &QgsQncMonitorProjection,
    setup_commands_executed: usize,
    setup_commands_accepted: usize,
    setup_commands_rejected: usize,
    final_setup_state: &str,
) -> String {
    let payload = &result.submission.payload;
    let reason = result.artifact.reason.unwrap_or("n/a");
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"qgs.file-presenter-diagnostic.v1\",\n",
            "  \"presenter_kind\": \"FilePresenter\",\n",
            "  \"source_mode\": \"{:?}\",\n",
            "  \"video_source\": \"proxy MP4\",\n",
            "  \"audio_source\": \"original MXF\",\n",
            "  \"proxy_aac\": \"not used\",\n",
            "  \"public_source_uri\": \"{}\",\n",
            "  \"private_path_exposed\": {},\n",
            "  \"source_frame_index\": {},\n",
            "  \"selected_preview_frame_index\": {},\n",
            "  \"presentation_time_ms\": {:.3},\n",
            "  \"duration_ms\": {:.3},\n",
            "  \"payload_kind\": \"{:?}\",\n",
            "  \"payload_format\": \"{:?}\",\n",
            "  \"backend_path\": \"{:?}\",\n",
            "  \"payload_id\": {},\n",
            "  \"visible_width\": {},\n",
            "  \"visible_height\": {},\n",
            "  \"coded_width\": {},\n",
            "  \"coded_height\": {},\n",
            "  \"artifact_kind\": \"{:?}\",\n",
            "  \"artifact_public_path\": \"{}\",\n",
            "  \"image_artifact_available\": {},\n",
            "  \"manifest_written\": {},\n",
            "  \"image_unavailable_reason\": \"{}\",\n",
            "  \"evidence_kind\": \"{:?}\",\n",
            "  \"evidence_level\": \"{:?}\",\n",
            "  \"monitor_projection\": {{\n",
            "    \"prepared_descriptor_present\": {},\n",
            "    \"submitted_to_presenter\": {},\n",
            "    \"frame_presented_test_boundary\": {},\n",
            "    \"frame_presented_real_backend\": {},\n",
            "    \"real_display_evidence_present\": {},\n",
            "    \"visual_verified\": {}\n",
            "  }},\n",
            "  \"session_setup\": {{\n",
            "    \"commands_executed\": {},\n",
            "    \"commands_accepted\": {},\n",
            "    \"commands_rejected\": {},\n",
            "    \"final_state\": \"{}\"\n",
            "  }},\n",
            "  \"real_display_output\": false,\n",
            "  \"visual_verified\": false,\n",
            "  \"realtime_playback\": false,\n",
            "  \"av_sync\": false,\n",
            "  \"device_output\": false,\n",
            "  \"frame_presented_real_display_claim\": false\n",
            "}}\n"
        ),
        payload.source_mode,
        json_escape(&payload.public_source_uri),
        payload.private_path_exposed || result.artifact.exposes_private_path(),
        payload.source_frame,
        payload
            .selected_preview_frame
            .map(|frame| frame.to_string())
            .unwrap_or_else(|| "null".to_string()),
        duration_ms(payload.presentation_time),
        duration_ms(payload.duration),
        payload.payload_kind,
        payload.payload_format,
        payload.backend_path,
        payload.payload_id,
        payload.visible_width,
        payload.visible_height,
        payload.coded_width,
        payload.coded_height,
        result.artifact.kind,
        json_escape(&result.artifact.public_artifact_path),
        result.artifact.image_written,
        result.artifact.manifest_written,
        json_escape(reason),
        result.evidence.evidence_kind,
        result.evidence.evidence_level,
        monitor_projection.prepared_descriptor_present,
        monitor_projection.submitted_to_presenter,
        monitor_projection.frame_presented_test_boundary,
        monitor_projection.frame_presented_real_backend,
        monitor_projection.real_display_evidence_present,
        monitor_projection.visual_verified,
        setup_commands_executed,
        setup_commands_accepted,
        setup_commands_rejected,
        json_escape(final_setup_state),
    )
}

#[allow(clippy::too_many_arguments)]
fn qgs_gpu_payload_readback_manifest_json(
    result: &QgsFilePresenterSubmissionResult,
    monitor_projection: &QgsQncMonitorProjection,
    public_manifest_path: &str,
    image_bytes_written: usize,
    readback_checksum: u64,
    readback_payload_bytes: usize,
    setup_commands_executed: usize,
    setup_commands_accepted: usize,
    setup_commands_rejected: usize,
    final_setup_state: &str,
) -> String {
    let payload = &result.submission.payload;
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"qgs.gpu-payload-readback-diagnostic.v1\",\n",
            "  \"presenter_kind\": \"FilePresenter\",\n",
            "  \"source_mode\": \"{:?}\",\n",
            "  \"video_source\": \"proxy MP4\",\n",
            "  \"audio_source\": \"original MXF\",\n",
            "  \"proxy_aac\": \"not used\",\n",
            "  \"public_source_uri\": \"{}\",\n",
            "  \"private_path_exposed\": {},\n",
            "  \"source_frame_index\": {},\n",
            "  \"selected_preview_frame_index\": {},\n",
            "  \"presentation_time_ms\": {:.3},\n",
            "  \"duration_ms\": {:.3},\n",
            "  \"payload_kind\": \"{:?}\",\n",
            "  \"payload_format\": \"{:?}\",\n",
            "  \"backend_path\": \"{:?}\",\n",
            "  \"payload_id\": {},\n",
            "  \"visible_width\": {},\n",
            "  \"visible_height\": {},\n",
            "  \"coded_width\": {},\n",
            "  \"coded_height\": {},\n",
            "  \"readback_attempted\": true,\n",
            "  \"readback_available\": true,\n",
            "  \"readback_format\": \"RgbaU16\",\n",
            "  \"output_artifact_format\": \"PPM P6 RGB8\",\n",
            "  \"conversion\": \"RgbaU16 high byte to RGB8; alpha discarded\",\n",
            "  \"artifact_kind\": \"{:?}\",\n",
            "  \"artifact_public_path\": \"{}\",\n",
            "  \"manifest_public_path\": \"{}\",\n",
            "  \"image_artifact_available\": {},\n",
            "  \"image_bytes_written\": {},\n",
            "  \"readback_payload_bytes\": {},\n",
            "  \"readback_checksum\": \"0x{:016x}\",\n",
            "  \"evidence_kind\": \"{:?}\",\n",
            "  \"evidence_level\": \"{:?}\",\n",
            "  \"monitor_projection\": {{\n",
            "    \"prepared_descriptor_present\": {},\n",
            "    \"submitted_to_presenter\": {},\n",
            "    \"frame_presented_test_boundary\": {},\n",
            "    \"frame_presented_real_backend\": {},\n",
            "    \"real_display_evidence_present\": {},\n",
            "    \"visual_verified\": {}\n",
            "  }},\n",
            "  \"session_setup\": {{\n",
            "    \"commands_executed\": {},\n",
            "    \"commands_accepted\": {},\n",
            "    \"commands_rejected\": {},\n",
            "    \"final_state\": \"{}\"\n",
            "  }},\n",
            "  \"real_display_output\": false,\n",
            "  \"visual_verified\": false,\n",
            "  \"realtime_playback\": false,\n",
            "  \"av_sync\": false,\n",
            "  \"device_output\": false,\n",
            "  \"frame_presented_real_display_claim\": false\n",
            "}}\n"
        ),
        payload.source_mode,
        json_escape(&payload.public_source_uri),
        payload.private_path_exposed || result.artifact.exposes_private_path(),
        payload.source_frame,
        payload
            .selected_preview_frame
            .map(|frame| frame.to_string())
            .unwrap_or_else(|| "null".to_string()),
        duration_ms(payload.presentation_time),
        duration_ms(payload.duration),
        payload.payload_kind,
        payload.payload_format,
        payload.backend_path,
        payload.payload_id,
        payload.visible_width,
        payload.visible_height,
        payload.coded_width,
        payload.coded_height,
        result.artifact.kind,
        json_escape(&result.artifact.public_artifact_path),
        json_escape(public_manifest_path),
        result.artifact.image_written,
        image_bytes_written,
        readback_payload_bytes,
        readback_checksum,
        result.evidence.evidence_kind,
        result.evidence.evidence_level,
        monitor_projection.prepared_descriptor_present,
        monitor_projection.submitted_to_presenter,
        monitor_projection.frame_presented_test_boundary,
        monitor_projection.frame_presented_real_backend,
        monitor_projection.real_display_evidence_present,
        monitor_projection.visual_verified,
        setup_commands_executed,
        setup_commands_accepted,
        setup_commands_rejected,
        json_escape(final_setup_state),
    )
}

fn audio_audit_range_label(start_ms: u64, duration_ms: u64) -> String {
    format!("start{start_ms:06}ms-dur{duration_ms:06}ms")
}

fn loudest_monitor_pair(summaries: &[AudioAuditTrackSummary]) -> Vec<u16> {
    let mut ranked = summaries
        .iter()
        .map(|summary| (summary.channel_index, summary.stats.rms_f32))
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked
        .into_iter()
        .take(2)
        .map(|(channel, _)| channel)
        .collect()
}

fn channel_pair_label(channels: &[u16]) -> String {
    if channels.is_empty() {
        "unavailable".to_string()
    } else {
        channels
            .iter()
            .map(|channel| format!("track {}", u32::from(*channel) + 1))
            .collect::<Vec<_>>()
            .join("/")
    }
}

fn monitor_pair_label(channels: &[u16], start: usize, count: usize) -> String {
    let pair = channels
        .iter()
        .skip(start)
        .take(count)
        .copied()
        .collect::<Vec<_>>();
    channel_pair_label(&pair)
}

fn audit_track_stats_range(
    blocks: &[&PcmAudioBlock],
    start_sample: u64,
    sample_count: u32,
) -> Result<TrackContentStats, Box<dyn std::error::Error>> {
    let first = blocks.first().ok_or("no blocks for track stats")?;
    let raw_first_12 = first.payload.iter().take(12).copied().collect::<Vec<_>>();
    let first_s24le = decode_first_samples(&first.payload, 4, decode_s24le_i32);
    let first_s24be = decode_first_samples(&first.payload, 4, decode_s24be_i32);
    let first_u24le = decode_first_samples(&first.payload, 4, decode_u24le);
    let mut values = Vec::new();
    let end_sample = start_sample
        .checked_add(u64::from(sample_count))
        .ok_or("track stats range overflow")?;
    for absolute_sample in start_sample..end_sample {
        values.push(sample_at_absolute(blocks, absolute_sample)?);
    }
    if values.is_empty() {
        return Err("no samples available for track stats".into());
    }
    stats_from_i32_values(raw_first_12, first_s24le, first_s24be, first_u24le, values)
}

fn stats_from_i32_values(
    raw_first_12: Vec<u8>,
    first_s24le: Vec<i32>,
    first_s24be: Vec<i32>,
    first_u24le: Vec<u32>,
    values: Vec<i32>,
) -> Result<TrackContentStats, Box<dyn std::error::Error>> {
    if values.is_empty() {
        return Err("no samples available for track stats".into());
    }
    let mut min_sample = i32::MAX;
    let mut max_sample = i32::MIN;
    let mut sum = 0_f64;
    let mut sum_sq = 0_f64;
    let mut zero_count = 0_usize;
    let mut clipping_count = 0_usize;
    let mut peak_f32 = 0_f32;
    let mut first_f32 = Vec::new();
    for (index, sample) in values.iter().copied().enumerate() {
        min_sample = min_sample.min(sample);
        max_sample = max_sample.max(sample);
        let sample_f64 = f64::from(sample);
        sum += sample_f64;
        sum_sq += sample_f64 * sample_f64;
        if sample == 0 {
            zero_count += 1;
        }
        if sample == -8_388_608 || sample == 8_388_607 {
            clipping_count += 1;
        }
        let f32_value = (sample as f32 / 8_388_608.0).clamp(-1.0, 1.0);
        peak_f32 = peak_f32.max(f32_value.abs());
        if index < 20 {
            first_f32.push(f32_value);
        }
    }
    let samples_analyzed = values.len();
    let mean = sum / samples_analyzed as f64;
    let rms = (sum_sq / samples_analyzed as f64).sqrt();
    let rms_f32 = rms / 8_388_608.0;
    let rms_dbfs = if rms_f32 > 0.0 {
        20.0 * rms_f32.log10()
    } else {
        f64::NEG_INFINITY
    };
    let dc_offset = mean / 8_388_608.0;
    let zero_ratio = zero_count as f64 / samples_analyzed as f64;
    let clipping_ratio = clipping_count as f64 / samples_analyzed as f64;
    Ok(TrackContentStats {
        raw_first_12,
        first_s24le,
        first_s24be,
        first_u24le,
        first_f32,
        samples_analyzed,
        min_sample,
        max_sample,
        mean,
        rms,
        peak_f32,
        rms_f32,
        rms_dbfs,
        dc_offset,
        zero_ratio,
        clipping_ratio,
        likely_silent: rms_f32 < 0.0001,
        likely_dc_buzz_risk: dc_offset.abs() > 0.05
            || (rms_f32 > 0.0 && dc_offset.abs() > rms_f32 * 0.5),
    })
}

fn sample_at_absolute(
    blocks: &[&PcmAudioBlock],
    absolute_sample: u64,
) -> Result<i32, Box<dyn std::error::Error>> {
    let first = blocks.first().ok_or("no blocks for sample lookup")?;
    let first_start = audio_samples_for_duration(first.start_time, first.sample_rate)?;
    if absolute_sample < first_start {
        return Err("audio sample is before first block".into());
    }
    let samples_per_block = first.sample_count;
    if samples_per_block == 0 {
        return Err("audio block has zero samples".into());
    }
    let relative_sample = absolute_sample - first_start;
    let block_index = usize::try_from(relative_sample / u64::from(samples_per_block))?;
    let local_sample = usize::try_from(relative_sample % u64::from(samples_per_block))?;
    let block = blocks
        .get(block_index)
        .ok_or("audio sample range is not covered by original PCM blocks")?;
    let PcmSampleFormat::SignedInteger {
        bits_per_sample,
        endian,
    } = block.format;
    if bits_per_sample != 24 || endian != PcmEndian::Little {
        return Err("audio content audit expects 24-bit little-endian PCM blocks".into());
    }
    if block.sample_count != samples_per_block {
        return Err("audio content audit expects stable sample counts per block".into());
    }
    let block_start = audio_samples_for_duration(block.start_time, block.sample_rate)?;
    if block_start != first_start + u64::try_from(block_index)? * u64::from(samples_per_block) {
        return Err("audio content audit sample lookup found a gap or overlap".into());
    }
    Ok(decode_s24le_i32(&block.payload, local_sample))
}

fn decode_first_samples<T>(payload: &[u8], count: usize, decode: fn(&[u8], usize) -> T) -> Vec<T> {
    let available = payload.len() / 3;
    (0..count.min(available))
        .map(|index| decode(payload, index))
        .collect()
}

fn decode_s24le_i32(payload: &[u8], sample_index: usize) -> i32 {
    let offset = sample_index * 3;
    let bytes = &payload[offset..offset + 3];
    let mut value = i32::from(bytes[0]) | (i32::from(bytes[1]) << 8) | (i32::from(bytes[2]) << 16);
    if value & 0x0080_0000 != 0 {
        value |= !0x00ff_ffff;
    }
    value
}

fn decode_s24be_i32(payload: &[u8], sample_index: usize) -> i32 {
    let offset = sample_index * 3;
    let bytes = &payload[offset..offset + 3];
    let mut value = (i32::from(bytes[0]) << 16) | (i32::from(bytes[1]) << 8) | i32::from(bytes[2]);
    if value & 0x0080_0000 != 0 {
        value |= !0x00ff_ffff;
    }
    value
}

fn decode_u24le(payload: &[u8], sample_index: usize) -> u32 {
    let offset = sample_index * 3;
    let bytes = &payload[offset..offset + 3];
    u32::from(bytes[0]) | (u32::from(bytes[1]) << 8) | (u32::from(bytes[2]) << 16)
}

fn f32_buffer_stats(
    bytes: &[u8],
    frame_count: u32,
    channels: u32,
) -> Result<F32BufferStats, Box<dyn std::error::Error>> {
    let expected = usize::try_from(frame_count)?
        .checked_mul(usize::try_from(channels)?)
        .and_then(|samples| samples.checked_mul(4))
        .ok_or("f32 audit buffer size overflow")?;
    if bytes.len() != expected {
        return Err("f32 audit buffer size mismatch".into());
    }
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum_sq = 0_f64;
    let mut nonzero = false;
    let mut count = 0_usize;
    for chunk in bytes.chunks_exact(4) {
        let value = f32::from_le_bytes(chunk.try_into()?);
        min = min.min(value);
        max = max.max(value);
        sum_sq += f64::from(value) * f64::from(value);
        nonzero |= value != 0.0;
        count += 1;
    }
    Ok(F32BufferStats {
        min,
        max,
        rms: (sum_sq / count as f64).sqrt(),
        nonzero,
    })
}

fn hex_bytes(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn f32_list(values: &[f32]) -> String {
    values
        .iter()
        .map(|value| format!("{value:.6}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn f64_list(values: &[f64]) -> String {
    values
        .iter()
        .map(|value| format!("{value:.6}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn interleaved_frame_list(
    bytes: &[u8],
    channels: usize,
    frames: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    let samples = bytes
        .chunks_exact(4)
        .map(|chunk| Ok(f32::from_le_bytes(chunk.try_into()?)))
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    let mut rows = Vec::new();
    for frame in 0..frames {
        let start = frame
            .checked_mul(channels)
            .ok_or("interleaved frame offset overflow")?;
        let end = start
            .checked_add(channels)
            .ok_or("interleaved frame end overflow")?;
        if end > samples.len() {
            break;
        }
        rows.push(format!("[{}]", f32_list(&samples[start..end])));
    }
    Ok(rows.join(" "))
}

fn write_f32_wav(
    path: &Path,
    sample_rate: u32,
    channels: u16,
    f32_interleaved_le: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let data_len = u32::try_from(f32_interleaved_le.len())?;
    let fmt_len = 16_u32;
    let audio_format = 3_u16; // IEEE float
    let bits_per_sample = 32_u16;
    let block_align = channels
        .checked_mul(bits_per_sample / 8)
        .ok_or("WAV block align overflow")?;
    let byte_rate = sample_rate
        .checked_mul(u32::from(block_align))
        .ok_or("WAV byte rate overflow")?;
    let riff_size = 4_u32
        .checked_add(8 + fmt_len)
        .and_then(|value| value.checked_add(8 + data_len))
        .ok_or("WAV RIFF size overflow")?;
    let mut file = File::create(path)?;
    file.write_all(b"RIFF")?;
    file.write_all(&riff_size.to_le_bytes())?;
    file.write_all(b"WAVE")?;
    file.write_all(b"fmt ")?;
    file.write_all(&fmt_len.to_le_bytes())?;
    file.write_all(&audio_format.to_le_bytes())?;
    file.write_all(&channels.to_le_bytes())?;
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&byte_rate.to_le_bytes())?;
    file.write_all(&block_align.to_le_bytes())?;
    file.write_all(&bits_per_sample.to_le_bytes())?;
    file.write_all(b"data")?;
    file.write_all(&data_len.to_le_bytes())?;
    file.write_all(f32_interleaved_le)?;
    Ok(())
}

struct OriginalAudioSegmentPlan {
    buffers: Vec<Vec<u8>>,
    start_sample: u64,
    duration: Duration,
    blocks_per_track: usize,
    samples_per_buffer: u32,
    samples_per_track: u32,
    channels: usize,
    total_bytes: usize,
    continuity_status: &'static str,
}

fn build_sequential_original_audio_segment(
    blocks: &[PcmAudioBlock],
    sample_rate: u32,
    preferred_duration: Duration,
    fallback_duration: Duration,
) -> Result<OriginalAudioSegmentPlan, Box<dyn std::error::Error>> {
    build_sequential_original_audio_segment_for_duration(blocks, sample_rate, preferred_duration)
        .or_else(|_| {
            build_sequential_original_audio_segment_for_duration(
                blocks,
                sample_rate,
                fallback_duration,
            )
        })
}

fn build_sequential_original_audio_segment_for_duration(
    blocks: &[PcmAudioBlock],
    sample_rate: u32,
    duration: Duration,
) -> Result<OriginalAudioSegmentPlan, Box<dyn std::error::Error>> {
    let mut by_channel: BTreeMap<u16, Vec<&PcmAudioBlock>> = BTreeMap::new();
    for block in blocks {
        let PcmAudioBlockLayout::MonoTrack { channel_index, .. } = block.layout else {
            continue;
        };
        if block.sample_rate != sample_rate {
            return Err("segment contains mixed sample rates".into());
        }
        let PcmSampleFormat::SignedInteger {
            bits_per_sample,
            endian,
        } = block.format;
        if bits_per_sample != 24 || endian != PcmEndian::Little {
            return Err("segment playback requires 24-bit little-endian original PCM".into());
        }
        by_channel.entry(channel_index).or_default().push(block);
    }
    if by_channel.len() != 4 {
        return Err("segment playback requires four original mono tracks".into());
    }
    for channel_blocks in by_channel.values_mut() {
        channel_blocks.sort_by_key(|block| block.start_time);
    }

    let first_channel = by_channel.values().next().ok_or("no original PCM blocks")?;
    let first_block = first_channel.first().ok_or("no original PCM blocks")?;
    let samples_per_buffer = first_block.sample_count;
    if samples_per_buffer == 0 {
        return Err("original PCM block has zero samples".into());
    }
    let target_samples = audio_samples_for_duration(duration, sample_rate)?;
    let blocks_needed = usize::try_from(target_samples.div_ceil(u64::from(samples_per_buffer)))?;
    if blocks_needed == 0 {
        return Err("segment duration selected zero blocks".into());
    }

    let mut output_buffers = Vec::with_capacity(blocks_needed);
    for block_index in 0..blocks_needed {
        let mut tracks = Vec::with_capacity(by_channel.len());
        let expected_start = u64::from(samples_per_buffer)
            .checked_mul(u64::try_from(block_index)?)
            .ok_or("segment sample offset overflow")?;
        for (channel_index, channel_blocks) in &by_channel {
            let block = channel_blocks
                .get(block_index)
                .ok_or("not enough sequential original PCM blocks for selected segment")?;
            if block.sample_count != samples_per_buffer {
                return Err("original PCM block sample count changed within segment".into());
            }
            let block_start = audio_samples_for_duration(block.start_time, sample_rate)?;
            if block_start != expected_start {
                return Err("original PCM block continuity check failed".into());
            }
            tracks.push(MonoS24LeTrack {
                channel_index: *channel_index,
                payload: &block.payload,
            });
        }
        output_buffers.push(f32_interleaved_from_s24le_mono_tracks(
            &tracks,
            samples_per_buffer,
        )?);
    }

    let samples_per_track = samples_per_buffer
        .checked_mul(u32::try_from(blocks_needed)?)
        .ok_or("segment sample count overflow")?;
    let duration = duration_from_audio_samples(u64::from(samples_per_track), sample_rate)?;
    let total_bytes = output_buffers.iter().map(Vec::len).sum();
    Ok(OriginalAudioSegmentPlan {
        buffers: output_buffers,
        start_sample: 0,
        duration,
        blocks_per_track: blocks_needed,
        samples_per_buffer,
        samples_per_track,
        channels: by_channel.len(),
        total_bytes,
        continuity_status: "contiguous/no gaps/no overlaps",
    })
}

fn build_pipewire_f32_interleaved_prototype_buffer(
    blocks: &[PcmAudioBlock],
    sample_rate: u32,
    sample_count: u32,
) -> Result<LinuxPipewirePrototypeBuffer, Box<dyn std::error::Error>> {
    let mut mono_blocks = blocks
        .iter()
        .filter_map(|block| {
            let PcmAudioBlockLayout::MonoTrack {
                track_id,
                channel_index,
            } = block.layout
            else {
                return None;
            };
            Some((track_id, channel_index, block))
        })
        .filter(|(_, _, block)| block.start_time == Duration::ZERO)
        .collect::<Vec<_>>();
    mono_blocks.sort_by_key(|(track_id, channel_index, _)| (*channel_index, *track_id));
    mono_blocks.dedup_by_key(|(track_id, channel_index, _)| (*track_id, *channel_index));

    if mono_blocks.is_empty() {
        return Err("no mono PCM blocks starting at zero for PipeWire prototype".into());
    }
    for (_, _, block) in &mono_blocks {
        let PcmSampleFormat::SignedInteger {
            bits_per_sample,
            endian,
        } = block.format;
        if bits_per_sample != 24 || endian != PcmEndian::Little || block.sample_rate != sample_rate
        {
            return Err("PipeWire prototype currently expects 24-bit little-endian PCM blocks at the source sample rate".into());
        }
        if block.sample_count < sample_count {
            return Err("PCM block is shorter than requested prototype sample count".into());
        }
    }

    let channel_count = u16::try_from(mono_blocks.len())?;
    let mut bytes = Vec::with_capacity(
        usize::try_from(sample_count)?
            .checked_mul(usize::from(channel_count))
            .and_then(|values| values.checked_mul(4))
            .ok_or("prototype output size overflow")?,
    );
    for sample_index in 0..usize::try_from(sample_count)? {
        for (_, _, block) in &mono_blocks {
            let sample = pcm_s24le_sample_to_f32(&block.payload, sample_index)?;
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
    }

    Ok(LinuxPipewirePrototypeBuffer {
        sample_rate,
        channels: channel_count,
        sample_count,
        sample_format: LinuxPipewirePrototypeSampleFormat::F32Interleaved,
        bytes,
    })
}

fn pcm_s24le_sample_to_f32(payload: &[u8], sample_index: usize) -> Result<f32, &'static str> {
    let offset = sample_index
        .checked_mul(3)
        .ok_or("24-bit sample offset overflow")?;
    let bytes = payload
        .get(offset..offset + 3)
        .ok_or("24-bit sample index out of range")?;
    let mut value = i32::from(bytes[0]) | (i32::from(bytes[1]) << 8) | (i32::from(bytes[2]) << 16);
    if value & 0x0080_0000 != 0 {
        value |= !0x00ff_ffff;
    }
    Ok((value as f32 / 8_388_608.0).clamp(-1.0, 1.0))
}

fn first_pipewire_mono_tracks<'a>(
    blocks: &'a [PcmAudioBlock],
    sample_count: u32,
) -> Result<Vec<MonoS24LeTrack<'a>>, Box<dyn std::error::Error>> {
    let mut tracks = blocks
        .iter()
        .filter_map(|block| {
            let PcmAudioBlockLayout::MonoTrack {
                track_id,
                channel_index,
            } = block.layout
            else {
                return None;
            };
            Some((track_id, channel_index, block))
        })
        .filter(|(_, _, block)| block.start_time == Duration::ZERO)
        .collect::<Vec<_>>();
    tracks.sort_by_key(|(track_id, channel_index, _)| (*channel_index, *track_id));
    tracks.dedup_by_key(|(track_id, channel_index, _)| (*track_id, *channel_index));
    if tracks.is_empty() {
        return Err("no mono PCM blocks starting at zero for PipeWire prototype".into());
    }
    for (_, _, block) in &tracks {
        let PcmSampleFormat::SignedInteger {
            bits_per_sample,
            endian,
        } = block.format;
        if bits_per_sample != 24 || endian != PcmEndian::Little {
            return Err("PipeWire prototype currently expects 24-bit little-endian PCM".into());
        }
        if block.sample_count < sample_count {
            return Err("PCM block is shorter than requested prototype sample count".into());
        }
    }
    Ok(tracks
        .into_iter()
        .map(|(_, channel_index, block)| MonoS24LeTrack {
            channel_index,
            payload: &block.payload,
        })
        .collect())
}

fn original_linux_pcm_audio_format(
    audio_tracks: &[&qgs_mxf::MxfTrack],
) -> Result<LinuxOriginalPcmAudioFormat, Box<dyn std::error::Error>> {
    let mut sample_rate = None;
    let mut bits_per_sample = None;
    let mut channels_per_track = None;
    for track in audio_tracks {
        let audio = track.audio.as_ref().ok_or("audio descriptor missing")?;
        let track_rate = rational_to_u32(audio.sample_rate.ok_or("audio sample rate missing")?)?;
        let track_depth = audio.bit_depth.ok_or("audio bit depth missing")?;
        let track_channels = audio.channels.ok_or("audio channel count missing")?;
        if let Some(sample_rate) = sample_rate {
            if sample_rate != track_rate {
                return Err("mixed original audio sample rates are not supported yet".into());
            }
        } else {
            sample_rate = Some(track_rate);
        }
        if let Some(bits_per_sample) = bits_per_sample {
            if bits_per_sample != track_depth {
                return Err("mixed original audio bit depths are not supported yet".into());
            }
        } else {
            bits_per_sample = Some(track_depth);
        }
        if let Some(channels_per_track) = channels_per_track {
            if channels_per_track != track_channels {
                return Err("mixed original audio channel layouts are not supported yet".into());
            }
        } else {
            channels_per_track = Some(track_channels);
        }
    }

    Ok(LinuxOriginalPcmAudioFormat {
        sample_rate: sample_rate.ok_or("original audio sample rate unavailable")?,
        bits_per_sample: bits_per_sample.ok_or("original audio bit depth unavailable")?,
        track_count: audio_tracks.len(),
        channels_per_track: channels_per_track.ok_or("original audio channel count unavailable")?,
    })
}

fn optional_yes_no(value: Option<bool>) -> &'static str {
    match value {
        Some(true) => "yes",
        Some(false) => "no",
        None => "unknown",
    }
}

fn linux_audio_conversion_need_label(value: LinuxAudioConversionNeed) -> &'static str {
    match value {
        LinuxAudioConversionNeed::No => "no",
        LinuxAudioConversionNeed::Yes => "yes",
        LinuxAudioConversionNeed::Unknown => "unknown",
    }
}

fn consume_pcm_payload_packet(
    packet: &PcmAudioPacket,
    stats: &mut PcmPayloadStats,
) -> Result<(), Box<dyn std::error::Error>> {
    stats.total_packets = stats
        .total_packets
        .checked_add(1)
        .ok_or("PCM packet count overflow")?;
    stats.total_payload_bytes = stats
        .total_payload_bytes
        .checked_add(u64::try_from(packet.payload_bytes()).map_err(|_| "payload size overflow")?)
        .ok_or("PCM payload byte count overflow")?;
    let end = packet
        .start
        .checked_add(packet.duration)
        .ok_or("PCM packet timestamp overflow")?;
    let track = stats
        .tracks
        .entry(packet.track_id)
        .or_insert_with(|| PcmTrackPayloadStats {
            channel_index: packet.channel_index,
            monotonic: true,
            ..PcmTrackPayloadStats::default()
        });
    if let Some(last_start) = track.last_start {
        if packet.start < last_start {
            track.monotonic = false;
        }
    }
    track.first_start.get_or_insert(packet.start);
    track.last_start = Some(packet.start);
    track.last_end = Some(end);
    track.packets = track
        .packets
        .checked_add(1)
        .ok_or("PCM track packet count overflow")?;
    track.samples = track
        .samples
        .checked_add(u64::from(packet.sample_count))
        .ok_or("PCM track sample count overflow")?;
    track.payload_bytes = track
        .payload_bytes
        .checked_add(u64::try_from(packet.payload_bytes()).map_err(|_| "payload size overflow")?)
        .ok_or("PCM track payload byte count overflow")?;
    Ok(())
}

fn consume_pcm_audio_block(
    block: PcmAudioBlock,
    stats: &mut PcmBlockStats,
) -> Result<(), Box<dyn std::error::Error>> {
    let (track_id, channel_index) = match block.layout {
        PcmAudioBlockLayout::MonoTrack {
            track_id,
            channel_index,
        } => (track_id, channel_index),
        PcmAudioBlockLayout::InterleavedChannels { .. } => {
            return Err("Step 20C qgs-test expects mono-track PCM blocks".into());
        }
    };
    stats.total_blocks = stats
        .total_blocks
        .checked_add(1)
        .ok_or("PCM block count overflow")?;
    stats.total_payload_bytes = stats
        .total_payload_bytes
        .checked_add(u64::try_from(block.payload_bytes()).map_err(|_| "payload size overflow")?)
        .ok_or("PCM block payload byte count overflow")?;
    let end = block.end_time()?;
    let track = stats
        .tracks
        .entry(track_id)
        .or_insert_with(|| PcmTrackBlockStats {
            channel_index,
            monotonic: true,
            ..PcmTrackBlockStats::default()
        });
    if let Some(last_start) = track.last_start {
        if block.start_time < last_start {
            track.monotonic = false;
        }
    }
    if let Some(last_end) = track.last_end {
        if block.start_time > last_end {
            track.gaps = track
                .gaps
                .checked_add(1)
                .ok_or("PCM block gap count overflow")?;
        } else if block.start_time < last_end {
            track.overlaps = track
                .overlaps
                .checked_add(1)
                .ok_or("PCM block overlap count overflow")?;
        }
    }
    track.first_start.get_or_insert(block.start_time);
    track.last_start = Some(block.start_time);
    track.last_end = Some(end);
    track.blocks = track
        .blocks
        .checked_add(1)
        .ok_or("PCM track block count overflow")?;
    track.samples = track
        .samples
        .checked_add(u64::from(block.sample_count))
        .ok_or("PCM track block sample count overflow")?;
    track.payload_bytes = track
        .payload_bytes
        .checked_add(u64::try_from(block.payload_bytes()).map_err(|_| "payload size overflow")?)
        .ok_or("PCM track block payload byte count overflow")?;
    Ok(())
}

fn broadcast_runtime_contract(
    original_path: &Path,
    proxy_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let original_bytes = std::fs::read(original_path)?;
    let original = MediaSource::parse(&original_bytes)?;
    let proxy = Mp4Source::open(proxy_path)?;
    let proxy_video = proxy
        .video
        .as_ref()
        .ok_or("proxy has no H.264 video track")?;
    let proxy_h264 = classify_video_track(proxy_video)?;
    let audio_tracks = original
        .tracks
        .iter()
        .filter(|track| track.kind == TrackKind::Audio)
        .collect::<Vec<_>>();
    if audio_tracks.is_empty() {
        return Err("original MXF has no audio tracks".into());
    }

    let sample_rate = original_audio_sample_rate(&audio_tracks)?;
    let blocks = build_original_pcm_blocks(&original, &original_bytes)?;
    let audio_duration = blocks
        .iter()
        .map(|block| block.end_time())
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .ok_or("no original PCM blocks available")?;
    let queue_limits = BroadcastRuntimeQueueLimits {
        audio_block_capacity: 8,
        video_frame_capacity: 6,
        processed_frame_capacity: 3,
    };
    let session = BroadcastRuntimeSessionDescription {
        audio_source: BroadcastMediaSourceRole::OriginalAuthoritativeAudio,
        video_source: BroadcastMediaSourceRole::ProxyPreviewVideo,
        video_source_mode: BroadcastVideoSourceMode::ProxyPreview,
        preview_profile: BroadcastPreviewProfile::Journalist50iPreview,
        audio_sample_rate: sample_rate,
        audio_track_count: audio_tracks.len(),
        queue_limits,
        capabilities: BroadcastRuntimeCapabilities {
            sample_clock_aware: true,
            preserves_original_pcm_format: true,
            preserves_track_channel_identity: true,
            proxy_video_preview: true,
            original_media_video_source: !original.index.video.is_empty(),
            original_media_realtime_supported: false,
            proxy_audio_primary: false,
            ui_dependent: false,
        },
    }
    .validate()?;

    let selected = selected_journalist_preview_frames(proxy_video)?;
    let frame_duration = ProxyPlaybackProfile::Journalist50iPreview
        .presentation_rate(RationalRate::new(
            u64::from(proxy_video.frame_rate.numerator),
            u64::from(proxy_video.frame_rate.denominator),
        )?)?
        .frame_duration()?;
    let ranges = selected
        .iter()
        .map(|frame| {
            av_frame_audio_range(
                frame.preview_index,
                frame.start_time,
                frame_duration,
                sample_rate,
                &blocks,
                audio_tracks.len(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let summary = summarize_broadcast_runtime_contract(&ranges, audio_duration);

    println!("QGS Broadcast Player Runtime Contract Draft");
    println!("-------------------------------------------");
    println!("Audio source: original MXF");
    println!("Video source mode: proxy-preview");
    println!("Video source: proxy MP4");
    println!("Proxy AAC: not used");
    println!("Preview profile: journalist-50i-preview");
    println!(
        "Clock policy: future QGS Broadcast Player Runtime/device backend policy; QNC applications observe and control it through session and transport commands."
    );
    println!("Contract owner: QGS backend-neutral Broadcast Player Runtime");
    println!(
        "Session: audio_role={:?} video_role={:?} sample_clock_aware={} ui_dependent={}",
        session.audio_source,
        session.video_source,
        yes_no(session.capabilities.sample_clock_aware),
        yes_no(session.capabilities.ui_dependent)
    );
    println!(
        "Proxy video: {}x{} H.264 {:?} {}-bit {:?} source_frames={} selected_preview_frames={}",
        proxy_h264.width,
        proxy_h264.height,
        proxy_h264.profile,
        proxy_h264.bit_depth,
        proxy_h264.chroma,
        proxy_video.samples.len(),
        selected.len()
    );
    println!(
        "Original audio: tracks={} sample_rate={}Hz blocks={} duration={:.3}s",
        audio_tracks.len(),
        sample_rate,
        blocks.len(),
        audio_duration.as_secs_f64()
    );
    println!(
        "Queue limits: audio_blocks={} video_frames={} processed_frames={}",
        queue_limits.audio_block_capacity,
        queue_limits.video_frame_capacity,
        queue_limits.processed_frame_capacity
    );
    println!(
        "Frames checked: {} complete={} incomplete={} outside_audio_range={} max_av_delta_ms={:.3}",
        summary.frames_checked,
        summary.complete_frames,
        summary.incomplete_frames,
        summary.frames_outside_audio_range,
        summary.max_audio_video_delta.as_secs_f64() * 1000.0
    );
    println!(
        "Suitable for Broadcast Player Runtime contract: {}",
        yes_no(summary.suitable_for_broadcast_runtime_contract)
    );
    println!("Representative frame-to-sample ranges:");
    for index in representative_range_indices(ranges.len()) {
        print_av_range(&selected[index], &ranges[index]);
    }

    Ok(())
}

fn broadcast_runtime_state_machine(
    original_path: &Path,
    proxy_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let original_bytes = std::fs::read(original_path)?;
    let original = MediaSource::parse(&original_bytes)?;
    let proxy = Mp4Source::open(proxy_path)?;
    let proxy_video = proxy
        .video
        .as_ref()
        .ok_or("proxy has no H.264 video track")?;
    let audio_tracks = original
        .tracks
        .iter()
        .filter(|track| track.kind == TrackKind::Audio)
        .collect::<Vec<_>>();
    if audio_tracks.is_empty() {
        return Err("original MXF has no audio tracks".into());
    }

    let sample_rate = original_audio_sample_rate(&audio_tracks)?;
    let blocks = build_original_pcm_blocks(&original, &original_bytes)?;
    let audio_duration = blocks
        .iter()
        .map(|block| block.end_time())
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .ok_or("no original PCM blocks available")?;
    let queue_limits = BroadcastRuntimeQueueLimits {
        audio_block_capacity: 8,
        video_frame_capacity: 6,
        processed_frame_capacity: 3,
    };
    let session = BroadcastRuntimeSessionDescription {
        audio_source: BroadcastMediaSourceRole::OriginalAuthoritativeAudio,
        video_source: BroadcastMediaSourceRole::ProxyPreviewVideo,
        video_source_mode: BroadcastVideoSourceMode::ProxyPreview,
        preview_profile: BroadcastPreviewProfile::Journalist50iPreview,
        audio_sample_rate: sample_rate,
        audio_track_count: audio_tracks.len(),
        queue_limits,
        capabilities: BroadcastRuntimeCapabilities {
            sample_clock_aware: true,
            preserves_original_pcm_format: true,
            preserves_track_channel_identity: true,
            proxy_video_preview: true,
            original_media_video_source: !original.index.video.is_empty(),
            original_media_realtime_supported: false,
            proxy_audio_primary: false,
            ui_dependent: false,
        },
    }
    .validate()?;
    let selected = selected_journalist_preview_frames(proxy_video)?;
    let selected_frame_count = selected.len();
    let intentional_skips = proxy_video
        .samples
        .len()
        .saturating_sub(selected_frame_count);
    let frame_duration = ProxyPlaybackProfile::Journalist50iPreview
        .presentation_rate(RationalRate::new(
            u64::from(proxy_video.frame_rate.numerator),
            u64::from(proxy_video.frame_rate.denominator),
        )?)?
        .frame_duration()?;
    let ranges = selected
        .iter()
        .map(|frame| {
            av_frame_audio_range(
                frame.preview_index,
                frame.start_time,
                frame_duration,
                sample_rate,
                &blocks,
                audio_tracks.len(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let summary = summarize_broadcast_runtime_contract(&ranges, audio_duration);
    let mut invalid_checks = Vec::new();
    let mut invalid = BroadcastRuntimeStateMachine::create(session)?;
    invalid_checks.push(("play_from_idle", invalid.play().is_err()));
    invalid_checks.push(("pause_from_idle", invalid.pause().is_err()));

    let mut runtime = BroadcastRuntimeStateMachine::create(session)?;
    let mut state_sequence = vec![format!("{:?}", runtime.state())];
    runtime.prepare(BroadcastRuntimePrepareFacts {
        selected_frame_count,
        intentional_profile_skips: intentional_skips,
        audio_ranges_complete: summary.incomplete_frames == 0,
        frames_outside_audio_range: summary.frames_outside_audio_range,
    })?;
    state_sequence.push("Preparing".to_string());
    state_sequence.push(format!("{:?}", runtime.state()));
    runtime.play()?;
    state_sequence.push(format!("{:?}", runtime.state()));
    runtime.pause()?;
    state_sequence.push(format!("{:?}", runtime.state()));
    runtime.play()?;
    state_sequence.push(format!("{:?}", runtime.state()));
    runtime.seek(Duration::from_millis(400))?;
    runtime.account_happy_path()?;
    runtime.drain()?;
    state_sequence.push(format!("{:?}", runtime.state()));
    runtime.complete()?;
    state_sequence.push(format!("{:?}", runtime.state()));
    invalid_checks.push((
        "seek_after_completed",
        runtime.seek(Duration::ZERO).is_err(),
    ));

    let accounting = runtime.accounting();
    let lateness_drop_events = runtime
        .events()
        .iter()
        .filter(|event| {
            matches!(
                event,
                qgs_media_runtime::BroadcastRuntimeEvent::LatenessDrop { .. }
            )
        })
        .count();

    println!("QGS Broadcast Player Runtime State Machine Skeleton");
    println!("---------------------------------------------------");
    println!("Not real playback: no speaker output, no display output, no real-time Broadcast Player loop");
    println!("Audio source: original MXF");
    println!("Video source mode: proxy-preview");
    println!("Video source: proxy MP4");
    println!("Proxy AAC: not used");
    println!("Preview profile: journalist-50i-preview");
    println!("State sequence: {}", state_sequence.join(" -> "));
    println!("Events emitted: {}", runtime.events().len());
    println!("Selected frames: {selected_frame_count}");
    println!("Intentional profile skips: {intentional_skips}");
    println!(
        "Audio ranges accounted: {}",
        accounting.audio_ranges_accounted
    );
    println!(
        "Selected frames accounted: {}",
        accounting.selected_frames_accounted
    );
    println!("Lateness drops: {}", accounting.lateness_drops);
    println!("Lateness drop events: {lateness_drop_events}");
    println!(
        "Queue limits: audio_blocks={} video_frames={} processed_frames={}",
        queue_limits.audio_block_capacity,
        queue_limits.video_frame_capacity,
        queue_limits.processed_frame_capacity
    );
    println!("Invalid transition checks:");
    for (name, passed) in invalid_checks {
        println!("  {name}: {}", if passed { "passed" } else { "failed" });
    }
    println!("Final state: {:?}", runtime.state());
    println!(
        "Happy path completed: {}",
        yes_no(
            runtime.state() == qgs_media_runtime::BroadcastRuntimeState::Completed
                && accounting.selected_frames_accounted == selected_frame_count
                && accounting.audio_ranges_accounted == selected_frame_count
                && accounting.intentional_profile_skips == intentional_skips
                && accounting.lateness_drops == 0
        )
    );

    Ok(())
}

fn broadcast_runtime_preroll(
    original_path: &Path,
    proxy_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let original_bytes = std::fs::read(original_path)?;
    let original = MediaSource::parse(&original_bytes)?;
    let proxy = Mp4Source::open(proxy_path)?;
    let proxy_video = proxy
        .video
        .as_ref()
        .ok_or("proxy has no H.264 video track")?;
    let audio_tracks = original
        .tracks
        .iter()
        .filter(|track| track.kind == TrackKind::Audio)
        .collect::<Vec<_>>();
    if audio_tracks.is_empty() {
        return Err("original MXF has no audio tracks".into());
    }

    let sample_rate = original_audio_sample_rate(&audio_tracks)?;
    let blocks = build_original_pcm_blocks(&original, &original_bytes)?;
    let audio_duration = blocks
        .iter()
        .map(|block| block.end_time())
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .ok_or("no original PCM blocks available")?;
    let queue_limits = BroadcastRuntimeQueueLimits {
        audio_block_capacity: 8,
        video_frame_capacity: 6,
        processed_frame_capacity: 3,
    };
    let session = BroadcastRuntimeSessionDescription {
        audio_source: BroadcastMediaSourceRole::OriginalAuthoritativeAudio,
        video_source: BroadcastMediaSourceRole::ProxyPreviewVideo,
        video_source_mode: BroadcastVideoSourceMode::ProxyPreview,
        preview_profile: BroadcastPreviewProfile::Journalist50iPreview,
        audio_sample_rate: sample_rate,
        audio_track_count: audio_tracks.len(),
        queue_limits,
        capabilities: BroadcastRuntimeCapabilities {
            sample_clock_aware: true,
            preserves_original_pcm_format: true,
            preserves_track_channel_identity: true,
            proxy_video_preview: true,
            original_media_video_source: !original.index.video.is_empty(),
            original_media_realtime_supported: false,
            proxy_audio_primary: false,
            ui_dependent: false,
        },
    }
    .validate()?;
    let selected = selected_journalist_preview_frames(proxy_video)?;
    let selected_frame_count = selected.len();
    let intentional_skips = proxy_video
        .samples
        .len()
        .saturating_sub(selected_frame_count);
    let frame_duration = ProxyPlaybackProfile::Journalist50iPreview
        .presentation_rate(RationalRate::new(
            u64::from(proxy_video.frame_rate.numerator),
            u64::from(proxy_video.frame_rate.denominator),
        )?)?
        .frame_duration()?;
    let ranges = selected
        .iter()
        .map(|frame| {
            av_frame_audio_range(
                frame.preview_index,
                frame.start_time,
                frame_duration,
                sample_rate,
                &blocks,
                audio_tracks.len(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let summary = summarize_broadcast_runtime_contract(&ranges, audio_duration);
    let config = BroadcastPrerollConfig {
        video_source_mode: BroadcastVideoSourceMode::ProxyPreview,
        video_frames_required: 3,
        audio_ranges_required: 3,
        max_video_queue: 6,
        max_audio_queue: 8,
        max_presentation_queue: 3,
    }
    .validate()?;
    let plan = BroadcastPrerollPlan {
        video_source_mode: BroadcastVideoSourceMode::ProxyPreview,
        selected_video_frames_planned: selected_frame_count,
        audio_ranges_planned: ranges.len(),
        intentional_skips_planned: intentional_skips,
        duration_covered: audio_duration,
        finite_queue_limits: queue_limits,
        video_source_available: true,
        video_runtime_supported: true,
    };
    let ready_status = evaluate_broadcast_preroll(
        config,
        plan,
        config.video_frames_required,
        config.audio_ranges_required,
    );
    let not_ready_status = evaluate_broadcast_preroll(
        config,
        plan,
        config.video_frames_required.saturating_sub(1),
        config.audio_ranges_required,
    );
    let original_media_config = BroadcastPrerollConfig {
        video_source_mode: BroadcastVideoSourceMode::OriginalMedia,
        ..config
    };
    let original_media_plan = BroadcastPrerollPlan {
        video_source_mode: BroadcastVideoSourceMode::OriginalMedia,
        selected_video_frames_planned: original.index.video.len(),
        audio_ranges_planned: original.index.video.len(),
        intentional_skips_planned: 0,
        duration_covered: audio_duration,
        finite_queue_limits: queue_limits,
        video_source_available: !original.index.video.is_empty(),
        video_runtime_supported: false,
    };
    let original_media_status = evaluate_broadcast_preroll(
        original_media_config,
        original_media_plan,
        config.video_frames_required,
        config.audio_ranges_required,
    );

    let mut not_ready_runtime = BroadcastRuntimeStateMachine::create(session)?;
    not_ready_runtime.prepare_with_preroll(
        BroadcastRuntimePrepareFacts {
            selected_frame_count,
            intentional_profile_skips: intentional_skips,
            audio_ranges_complete: summary.incomplete_frames == 0,
            frames_outside_audio_range: summary.frames_outside_audio_range,
        },
        not_ready_status,
    )?;
    let play_before_ready_rejected = not_ready_runtime.play().is_err();

    let mut runtime = BroadcastRuntimeStateMachine::create(session)?;
    runtime.prepare_with_preroll(
        BroadcastRuntimePrepareFacts {
            selected_frame_count,
            intentional_profile_skips: intentional_skips,
            audio_ranges_complete: summary.incomplete_frames == 0,
            frames_outside_audio_range: summary.frames_outside_audio_range,
        },
        ready_status,
    )?;
    let ready_reached = runtime.state() == qgs_media_runtime::BroadcastRuntimeState::Ready;
    let play_from_ready_succeeds = runtime.play().is_ok();

    println!("QGS Broadcast Player Runtime Preroll Plan");
    println!("------------------------------------------");
    println!("Not real playback: no speaker output, no display output, no real-time Broadcast Player loop");
    println!("Audio source: original MXF");
    println!("Video source mode: proxy-preview");
    println!("Video source: proxy MP4");
    println!("Proxy AAC: not used");
    println!("Original media mode: available in contract, realtime support not claimed in this milestone");
    println!("Preview profile: journalist-50i-preview");
    println!(
        "Preroll config: source_mode={:?} video_required={} audio_required={} max_video_queue={} max_audio_queue={} max_presentation_queue={}",
        config.video_source_mode,
        config.video_frames_required,
        config.audio_ranges_required,
        config.max_video_queue,
        config.max_audio_queue,
        config.max_presentation_queue
    );
    println!(
        "Preroll plan: selected_frames={} intentional_skips={} audio_ranges={} duration={:.3}s",
        plan.selected_video_frames_planned,
        plan.intentional_skips_planned,
        plan.audio_ranges_planned,
        plan.duration_covered.as_secs_f64()
    );
    println!(
        "Ready status: ready={} prepared_video={} prepared_audio={} missing_video={} missing_audio={} queue_limits_ok={}",
        yes_no(ready_status.ready),
        ready_status.prepared_video_frames,
        ready_status.prepared_audio_ranges,
        ready_status.missing_video_frames,
        ready_status.missing_audio_ranges,
        yes_no(ready_status.queue_limits_ok)
    );
    println!(
        "Original media mode probe: ready={} reason={:?} original_video_frames={}",
        yes_no(original_media_status.ready),
        original_media_status.reason,
        original_media_plan.selected_video_frames_planned
    );
    println!(
        "Not-ready probe: ready={} reason={:?} play_before_ready_rejected={}",
        yes_no(not_ready_status.ready),
        not_ready_status.reason,
        yes_no(play_before_ready_rejected)
    );
    println!(
        "Queue peaks: video={} audio={} presentation={}",
        config.video_frames_required, config.audio_ranges_required, 0
    );
    println!("Ready reached after preroll: {}", yes_no(ready_reached));
    println!(
        "Play from Ready succeeds: {}",
        yes_no(play_from_ready_succeeds)
    );

    Ok(())
}

fn broadcast_player_runtime_verification(
    original_path: &Path,
    proxy_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let original_bytes = std::fs::read(original_path)?;
    let original = MediaSource::parse(&original_bytes)?;
    let proxy = Mp4Source::open(proxy_path)?;
    let proxy_video = proxy
        .video
        .as_ref()
        .ok_or("proxy has no H.264 video track")?;
    let proxy_h264 = classify_video_track(proxy_video)?;
    let audio_tracks = original
        .tracks
        .iter()
        .filter(|track| track.kind == TrackKind::Audio)
        .count();
    let matrix = BroadcastRuntimeVerificationMatrix::sony_fx6_sample_002_current();
    matrix.validate_truth_rules()?;

    println!("QGS Broadcast Player Runtime Acceptance Verification Matrix");
    println!("-----------------------------------------------------------");
    println!("This is a truthfulness/validation report, not a new media feature.");
    println!("Audio source rule: original MXF is authoritative");
    println!("Proxy AAC: not authoritative / not used as primary audio");
    println!("Proxy video role: responsive ProxyPreview source");
    println!("Original video role: OriginalMedia source; realtime support not claimed");
    println!(
        "Media inspected: proxy_video_frames={} proxy={}x{} coded={}x{} original_audio_tracks={} original_video_index_entries={}",
        proxy_video.samples.len(),
        proxy_h264.width,
        proxy_h264.height,
        proxy_h264.coded_width,
        proxy_h264.coded_height,
        audio_tracks,
        original.index.video.len()
    );
    println!();
    println!("{:<48} | {:<24} | Evidence", "Subsystem", "Level");
    println!("{:-<48}-+-{:-<24}-+-{:-<1}", "", "", "");
    for entry in &matrix.entries {
        println!(
            "{:<48} | {:<24} | {}",
            entry.subsystem.label(),
            entry.level.label(),
            entry.summary
        );
    }
    println!();
    println!("Truth rules:");
    println!("  Test video presenter evidence is not real display output");
    println!("  Test presenter evidence is not real FramePresented from a display backend");
    println!("  Test audio sink evidence is not real speaker output");
    println!("  VisualVerified: no");
    println!("  AudioDeviceVerified: no");
    println!("  RealtimeVerified: no");
    println!("  A/V sync verified: no");
    println!("  Production audio output: no");
    println!("  Completed QNC integration: no");
    println!("  Haswell CPU-bridge 50p is not marked realtime verified");
    println!("  Modern-hardware zero-copy remains frozen/not verified");
    println!("  Matrix truth-rule validation: passed");

    Ok(())
}

struct ProxyVideoPayloadBindingProof {
    bindings: Vec<qgs_media_runtime::BroadcastVideoPayloadBinding>,
    gpu_processor: Nv12FrameProcessor,
    readback_outputs: BTreeMap<u64, qgs_vulkan::ProcessedFrameOutput>,
    gpu_submissions: usize,
    gpu_completions: usize,
    device_name: String,
}

impl ProxyVideoPayloadBindingProof {
    fn bounded_gpu_slots(&self) -> usize {
        usize::try_from(self.gpu_processor.counters().command_buffer_count).unwrap_or(0)
    }
}

struct OriginalVideoPayloadBindingProof {
    bindings: Vec<qgs_media_runtime::BroadcastVideoPayloadBinding>,
    gpu_processor: GpuFrameProcessor,
    target_frames: Vec<u64>,
    random_access_start: u64,
    decode_end: u64,
    decoded_frames: usize,
    gpu_submissions: usize,
    gpu_completions: usize,
    device_name: String,
    source_format: String,
    wall_elapsed: Duration,
}

impl OriginalVideoPayloadBindingProof {
    fn bounded_gpu_slots(&self) -> usize {
        usize::try_from(self.gpu_processor.counters().command_buffer_count).unwrap_or(0)
    }
}

fn finish_proxy_video_payload_bindings(
    proxy_video_slots: &[BroadcastPreparedVideoSlot],
    bindings_by_source: &mut BTreeMap<u64, qgs_media_runtime::BroadcastVideoPayloadBinding>,
    readback_by_source: &mut BTreeMap<u64, qgs_vulkan::ProcessedFrameOutput>,
    gpu_processor: Nv12FrameProcessor,
    gpu_submissions: usize,
    gpu_completions: usize,
    device_name: String,
) -> Result<ProxyVideoPayloadBindingProof, Box<dyn std::error::Error>> {
    let bindings = proxy_video_slots
        .iter()
        .map(|slot| {
            let source_frame_index = slot
                .source_frame_index
                .ok_or("proxy video payload slot missing source frame index")?;
            bindings_by_source
                .remove(&source_frame_index)
                .ok_or_else(|| {
                    format!(
                        "missing proxy video payload binding for source frame {source_frame_index}"
                    )
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let readback_outputs = proxy_video_slots
        .iter()
        .filter_map(|slot| {
            let source_frame = slot.source_frame_index?;
            readback_by_source
                .remove(&source_frame)
                .map(|output| (source_frame, output))
        })
        .collect::<BTreeMap<_, _>>();
    Ok(ProxyVideoPayloadBindingProof {
        bindings,
        gpu_processor,
        readback_outputs,
        gpu_submissions,
        gpu_completions,
        device_name,
    })
}

fn bind_proxy_preview_video_payloads(
    proxy_video: &qgs_mp4::Mp4VideoTrack,
    proxy_h264: &qgs_mp4::Mp4H264Summary,
    proxy_video_slots: &[BroadcastPreparedVideoSlot],
) -> Result<ProxyVideoPayloadBindingProof, Box<dyn std::error::Error>> {
    bind_proxy_preview_video_payloads_with_readback(
        proxy_video,
        proxy_h264,
        proxy_video_slots,
        false,
    )
}

fn bind_proxy_preview_video_payloads_with_readback(
    proxy_video: &qgs_mp4::Mp4VideoTrack,
    proxy_h264: &qgs_mp4::Mp4H264Summary,
    proxy_video_slots: &[BroadcastPreparedVideoSlot],
    validation_readback: bool,
) -> Result<ProxyVideoPayloadBindingProof, Box<dyn std::error::Error>> {
    let discovery = VulkanDeviceDiscovery::new()?;
    let devices = discovery.enumerate_devices()?;
    let Some(device) = devices
        .iter()
        .find(|device| {
            device.vendor_id == 0x8086 && matches!(device.class, DeviceClass::IntegratedGpu)
        })
        .or_else(|| {
            devices
                .iter()
                .find(|device| matches!(device.class, DeviceClass::IntegratedGpu))
        })
    else {
        return Err("Broadcast Player proxy payload binding: no integrated GPU advertised".into());
    };
    let proxy_config = decoder_config_for_surface(
        device.id,
        proxy_h264.profile,
        BitDepth::new(proxy_h264.bit_depth)?,
        proxy_h264.chroma,
        proxy_h264.coded_width,
        proxy_h264.coded_height,
    );
    let vaapi = qgs_vaapi::VaapiVideoDiscovery::new(&devices);
    let capabilities = vaapi.query_video_capabilities(device.id)?;
    let supports_proxy = capabilities
        .decode
        .iter()
        .any(|capability| proxy_config.is_satisfied_by(capability));
    if !supports_proxy {
        return Err("Broadcast Player proxy payload binding: Intel VA backend does not support proxy configuration".into());
    }

    let mut targets = BTreeMap::new();
    for slot in proxy_video_slots {
        let source_frame_index = slot
            .source_frame_index
            .ok_or("proxy video payload slot missing source frame index")?;
        targets.insert(source_frame_index, slot);
    }

    let mut decoder = vaapi.create_decoder(&CreateDecoderRequest {
        config: proxy_config.clone(),
    })?;
    let decoder_id = DecoderId::new(20)?;
    let visible_region = VisibleRegion {
        x: 0,
        y: 0,
        width: proxy_h264.width,
        height: proxy_h264.height,
    };
    let mut cpu_pool = qgs_vaapi::CpuNv12FramePool::new(
        proxy_h264.coded_width,
        proxy_h264.coded_height,
        visible_region,
        proxy_video_slots.len().max(1),
    )?;
    let mut gpu_processor = Nv12FrameProcessor::new(
        &discovery,
        Nv12FrameProcessorConfig {
            device_id: device.id,
            coded_width: proxy_h264.coded_width,
            coded_height: proxy_h264.coded_height,
            visible_width: proxy_h264.width,
            visible_height: proxy_h264.height,
            slot_count: proxy_video_slots.len().max(1),
            conversion: YcbcrConversion::Rec709Limited,
            validation_readback,
        },
    )?;

    let mut bindings_by_source = BTreeMap::new();
    let mut readback_by_source = BTreeMap::new();
    let mut output_index = 0_u64;
    let mut gpu_submissions = 0_usize;
    let mut gpu_completions = 0_usize;

    for sample in &proxy_video.samples {
        let outputs = decoder.submit_access_unit(&SubmitAccessUnitRequest {
            decoder_id,
            data: sample.annex_b.clone(),
        })?;
        for output in outputs {
            if let Some(slot) = targets.get(&output_index) {
                let identity = FrameIdentity {
                    presentation_position: slot
                        .selected_preview_frame_index
                        .ok_or("proxy video payload slot missing selected preview index")?,
                };
                let cpu_frame = cpu_pool.acquire()?;
                let (cpu_frame, _) =
                    qgs_vaapi::transfer_nv12_surface_timed(output.resource.as_ref(), cpu_frame)?;
                let upload = nv12_upload_for_cpu_surface(device.id, &cpu_frame)?;
                let token = gpu_processor.submit_frame(&upload, identity)?;
                gpu_submissions += 1;
                cpu_pool.release(cpu_frame)?;
                if validation_readback {
                    let readback = gpu_processor.wait_for_frame(token)?;
                    readback_by_source.insert(output_index, readback);
                } else {
                    gpu_processor.wait_for_completion(token)?;
                }
                gpu_completions += 1;
                let payload = BroadcastVideoPayloadReference {
                    payload_id: token.get(),
                    kind: BroadcastVideoPayloadKind::ProcessedGpuFrame,
                    format: BroadcastVideoPayloadFormat::RgbaU16,
                    backend_path: BroadcastVideoPayloadBackendPath::VaapiCpuNv12Vulkan,
                    source_frame_index: output_index,
                    selected_preview_frame_index: slot.selected_preview_frame_index,
                    presentation_time: slot.presentation_time,
                    duration: slot.duration,
                    coded_width: proxy_h264.coded_width,
                    coded_height: proxy_h264.coded_height,
                    visible_width: proxy_h264.width,
                    visible_height: proxy_h264.height,
                    bounded_slot_index: slot.slot_index,
                    session_index: 0,
                };
                let binding = bind_broadcast_video_payload_ready(slot, payload)?;
                bindings_by_source.insert(output_index, binding);
                if bindings_by_source.len() == targets.len() {
                    return finish_proxy_video_payload_bindings(
                        proxy_video_slots,
                        &mut bindings_by_source,
                        &mut readback_by_source,
                        gpu_processor,
                        gpu_submissions,
                        gpu_completions,
                        device.name.clone(),
                    );
                }
            }
            output_index = output_index
                .checked_add(1)
                .ok_or("proxy video output index overflow")?;
        }
    }

    let outputs = decoder.flush(&FlushDecoderRequest { decoder_id })?;
    for output in outputs {
        if let Some(slot) = targets.get(&output_index) {
            let identity = FrameIdentity {
                presentation_position: slot
                    .selected_preview_frame_index
                    .ok_or("proxy video payload slot missing selected preview index")?,
            };
            let cpu_frame = cpu_pool.acquire()?;
            let (cpu_frame, _) =
                qgs_vaapi::transfer_nv12_surface_timed(output.resource.as_ref(), cpu_frame)?;
            let upload = nv12_upload_for_cpu_surface(device.id, &cpu_frame)?;
            let token = gpu_processor.submit_frame(&upload, identity)?;
            gpu_submissions += 1;
            cpu_pool.release(cpu_frame)?;
            if validation_readback {
                let readback = gpu_processor.wait_for_frame(token)?;
                readback_by_source.insert(output_index, readback);
            } else {
                gpu_processor.wait_for_completion(token)?;
            }
            gpu_completions += 1;
            let payload = BroadcastVideoPayloadReference {
                payload_id: token.get(),
                kind: BroadcastVideoPayloadKind::ProcessedGpuFrame,
                format: BroadcastVideoPayloadFormat::RgbaU16,
                backend_path: BroadcastVideoPayloadBackendPath::VaapiCpuNv12Vulkan,
                source_frame_index: output_index,
                selected_preview_frame_index: slot.selected_preview_frame_index,
                presentation_time: slot.presentation_time,
                duration: slot.duration,
                coded_width: proxy_h264.coded_width,
                coded_height: proxy_h264.coded_height,
                visible_width: proxy_h264.width,
                visible_height: proxy_h264.height,
                bounded_slot_index: slot.slot_index,
                session_index: 0,
            };
            let binding = bind_broadcast_video_payload_ready(slot, payload)?;
            bindings_by_source.insert(output_index, binding);
            if bindings_by_source.len() == targets.len() {
                return finish_proxy_video_payload_bindings(
                    proxy_video_slots,
                    &mut bindings_by_source,
                    &mut readback_by_source,
                    gpu_processor,
                    gpu_submissions,
                    gpu_completions,
                    device.name.clone(),
                );
            }
        }
        output_index = output_index
            .checked_add(1)
            .ok_or("proxy video output index overflow")?;
    }

    let missing = proxy_video_slots
        .iter()
        .filter_map(|slot| slot.source_frame_index)
        .filter(|source| !bindings_by_source.contains_key(source))
        .map(|source| source.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    Err(format!("missing proxy video payload bindings for source frames: {missing}").into())
}

fn bind_original_media_video_payloads(
    original: &MediaSource,
    original_bytes: &[u8],
    original_video_slots: &[BroadcastPreparedVideoSlot],
) -> Result<OriginalVideoPayloadBindingProof, Box<dyn std::error::Error>> {
    let binding_started = Instant::now();
    if original_video_slots.is_empty() {
        return Err("Broadcast Player original payload binding: no prepared video slots".into());
    }
    let first_access_unit = original.extract_video_access_unit(original_bytes, 0)?;
    let parsed = qgs_codec_h264::parse_annex_b_access_unit(&first_access_unit)?;
    let discovery = VulkanDeviceDiscovery::new()?;
    let devices = discovery.enumerate_devices()?;
    let Some(device) = devices
        .iter()
        .find(|device| {
            device.vendor_id == 0x8086 && matches!(device.class, DeviceClass::IntegratedGpu)
        })
        .or_else(|| {
            devices
                .iter()
                .find(|device| matches!(device.class, DeviceClass::IntegratedGpu))
        })
        .or_else(|| {
            devices
                .iter()
                .find(|device| matches!(device.class, DeviceClass::DiscreteGpu))
        })
    else {
        return Err("Broadcast Player original payload binding: no Vulkan GPU advertised".into());
    };
    let config = decoder_config_for_surface(
        device.id,
        parsed.profile,
        parsed.desc.bit_depth,
        parsed.desc.chroma,
        parsed.desc.coded_width,
        parsed.desc.coded_height,
    );
    if !SoftwareVideoBackend::supports_config(&config) {
        return Err(
            "Broadcast Player original payload binding: software backend does not support original stream"
                .into(),
        );
    }
    let target_frames = original_video_slots
        .iter()
        .map(|slot| {
            slot.source_frame_index
                .ok_or("original video payload slot missing source frame index")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let min_target = *target_frames
        .iter()
        .min()
        .ok_or("original video payload binding has no targets")?;
    let max_target = *target_frames
        .iter()
        .max()
        .ok_or("original video payload binding has no targets")?;
    let random_access_start = original
        .index
        .nearest_random_access_before(min_target)
        .map(|entry| entry.edit_unit)
        .unwrap_or(min_target);
    let last_video_index = original.index.video.len().saturating_sub(1);
    let decode_end = max_target
        .saturating_add(16)
        .min(u64::try_from(last_video_index)?);
    let positioned = (random_access_start..=decode_end)
        .map(|edit_unit| {
            let index = usize::try_from(edit_unit)?;
            Ok((
                edit_unit,
                original.extract_video_access_unit(original_bytes, index)?,
            ))
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    let decode_start = Instant::now();
    let decoded = decode_positioned_access_units_with_context(config, &positioned)?;
    let _decode_elapsed = decode_start.elapsed();
    let frames_by_presentation = decoded
        .frames
        .iter()
        .map(|frame| (frame.presentation_index, frame))
        .collect::<BTreeMap<_, _>>();
    let first_frame = original_video_slots
        .iter()
        .filter_map(|slot| {
            slot.source_frame_index
                .and_then(|index| frames_by_presentation.get(&index).copied())
        })
        .next()
        .ok_or("Broadcast Player original payload binding: no target decoded frames")?;
    let first_upload = yuv422p10_upload_for_frame(device.id, first_frame)?;
    let mut gpu_processor = GpuFrameProcessor::new(
        &discovery,
        GpuFrameProcessorConfig {
            device_id: device.id,
            width: first_upload.width,
            height: first_upload.height,
            slot_count: original_video_slots.len().max(1),
            conversion: first_upload.conversion,
        },
    )?;
    let mut bindings = Vec::new();
    let mut gpu_submissions = 0_usize;
    let mut gpu_completions = 0_usize;
    for slot in original_video_slots {
        let source_frame_index = slot
            .source_frame_index
            .ok_or("original video payload slot missing source frame index")?;
        let frame = frames_by_presentation
            .get(&source_frame_index)
            .copied()
            .or_else(|| {
                source_frame_index
                    .checked_sub(random_access_start)
                    .and_then(|relative| usize::try_from(relative).ok())
                    .and_then(|relative| decoded.frames.get(relative))
            })
            .ok_or_else(|| {
                format!("missing decoded original frame for source frame {source_frame_index}")
            })?;
        let upload = yuv422p10_upload_for_frame(device.id, frame)?;
        let identity = FrameIdentity {
            presentation_position: source_frame_index,
        };
        let token = gpu_processor.submit_frame(&upload, identity)?;
        gpu_submissions += 1;
        gpu_processor.wait_for_frame(token)?;
        gpu_completions += 1;
        let payload = BroadcastVideoPayloadReference {
            payload_id: token.get(),
            kind: BroadcastVideoPayloadKind::ProcessedGpuFrame,
            format: BroadcastVideoPayloadFormat::RgbaU16,
            backend_path: BroadcastVideoPayloadBackendPath::SoftwareH264Yuv422P10Vulkan,
            source_frame_index,
            selected_preview_frame_index: slot.selected_preview_frame_index,
            presentation_time: slot.presentation_time,
            duration: slot.duration,
            coded_width: parsed.desc.coded_width,
            coded_height: parsed.desc.coded_height,
            visible_width: parsed.desc.visible_region.width,
            visible_height: parsed.desc.visible_region.height,
            bounded_slot_index: slot.slot_index,
            session_index: 1,
        };
        bindings.push(bind_broadcast_video_payload_ready(slot, payload)?);
    }
    Ok(OriginalVideoPayloadBindingProof {
        bindings,
        gpu_processor,
        target_frames,
        random_access_start,
        decode_end,
        decoded_frames: decoded.frames.len(),
        gpu_submissions,
        gpu_completions,
        device_name: device.name.clone(),
        source_format: format!(
            "H.264 {:?}, {}-bit {:?}",
            parsed.profile,
            parsed.desc.bit_depth.get(),
            parsed.desc.chroma
        ),
        wall_elapsed: binding_started.elapsed(),
    })
}

fn broadcast_runtime_prepared_slots(
    original_path: &Path,
    proxy_path: &Path,
    report_focus: BroadcastRuntimeReportFocus,
) -> Result<(), Box<dyn std::error::Error>> {
    let original_bytes = std::fs::read(original_path)?;
    let original = MediaSource::parse(&original_bytes)?;
    let proxy = Mp4Source::open(proxy_path)?;
    let proxy_video = proxy
        .video
        .as_ref()
        .ok_or("proxy has no H.264 video track")?;
    let proxy_h264 = classify_video_track(proxy_video)?;
    let audio_tracks = original
        .tracks
        .iter()
        .filter(|track| track.kind == TrackKind::Audio)
        .collect::<Vec<_>>();
    if audio_tracks.is_empty() {
        return Err("original MXF has no audio tracks".into());
    }

    let sample_rate = original_audio_sample_rate(&audio_tracks)?;
    let blocks = build_original_pcm_blocks(&original, &original_bytes)?;
    let audio_duration = blocks
        .iter()
        .map(|block| block.end_time())
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .ok_or("no original PCM blocks available")?;
    let queue_limits = BroadcastRuntimeQueueLimits {
        audio_block_capacity: 8,
        video_frame_capacity: 6,
        processed_frame_capacity: 3,
    };
    let session = BroadcastRuntimeSessionDescription {
        audio_source: BroadcastMediaSourceRole::OriginalAuthoritativeAudio,
        video_source: BroadcastMediaSourceRole::ProxyPreviewVideo,
        video_source_mode: BroadcastVideoSourceMode::ProxyPreview,
        preview_profile: BroadcastPreviewProfile::Journalist50iPreview,
        audio_sample_rate: sample_rate,
        audio_track_count: audio_tracks.len(),
        queue_limits,
        capabilities: BroadcastRuntimeCapabilities {
            sample_clock_aware: true,
            preserves_original_pcm_format: true,
            preserves_track_channel_identity: true,
            proxy_video_preview: true,
            original_media_video_source: !original.index.video.is_empty(),
            original_media_realtime_supported: false,
            proxy_audio_primary: false,
            ui_dependent: false,
        },
    }
    .validate()?;
    let selected = selected_journalist_preview_frames(proxy_video)?;
    let selected_frame_count = selected.len();
    let intentional_skips = proxy_video
        .samples
        .len()
        .saturating_sub(selected_frame_count);
    let frame_duration = ProxyPlaybackProfile::Journalist50iPreview
        .presentation_rate(RationalRate::new(
            u64::from(proxy_video.frame_rate.numerator),
            u64::from(proxy_video.frame_rate.denominator),
        )?)?
        .frame_duration()?;
    let ranges = selected
        .iter()
        .map(|frame| {
            av_frame_audio_range(
                frame.preview_index,
                frame.start_time,
                frame_duration,
                sample_rate,
                &blocks,
                audio_tracks.len(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let runtime_summary = summarize_broadcast_runtime_contract(&ranges, audio_duration);
    let config = BroadcastPrerollConfig {
        video_source_mode: BroadcastVideoSourceMode::ProxyPreview,
        video_frames_required: 3,
        audio_ranges_required: 3,
        max_video_queue: 6,
        max_audio_queue: 8,
        max_presentation_queue: 3,
    }
    .validate()?;
    let plan = BroadcastPrerollPlan {
        video_source_mode: BroadcastVideoSourceMode::ProxyPreview,
        selected_video_frames_planned: selected_frame_count,
        audio_ranges_planned: ranges.len(),
        intentional_skips_planned: intentional_skips,
        duration_covered: audio_duration,
        finite_queue_limits: queue_limits,
        video_source_available: true,
        video_runtime_supported: true,
    };
    let prepared_len = config.video_frames_required;
    let proxy_video_slots = selected
        .iter()
        .take(prepared_len)
        .enumerate()
        .map(|(slot_index, frame)| BroadcastPreparedVideoSlot {
            slot_index,
            source_mode: BroadcastVideoSourceMode::ProxyPreview,
            video_source_role: BroadcastMediaSourceRole::ProxyPreviewVideo,
            source_frame_index: Some(frame.source_presentation_index),
            selected_preview_frame_index: Some(frame.preview_index),
            presentation_time: frame.start_time,
            duration: frame_duration,
            status: BroadcastPreparedVideoSlotStatus::Prepared,
        })
        .collect::<Vec<_>>();
    let proxy_audio_slots = ranges
        .iter()
        .take(prepared_len)
        .enumerate()
        .map(|(slot_index, range)| {
            BroadcastPreparedAudioSlot::from_audio_range(
                slot_index,
                BroadcastVideoSourceMode::ProxyPreview,
                range,
            )
        })
        .collect::<Vec<_>>();
    let proxy_presentation_slots = selected
        .iter()
        .take(prepared_len)
        .enumerate()
        .map(|(slot_index, frame)| BroadcastPreparedPresentationSlot {
            presentation_index: slot_index,
            source_mode: BroadcastVideoSourceMode::ProxyPreview,
            selected_source_frame: Some(frame.source_presentation_index),
            video_slot_index: slot_index,
            audio_slot_index: slot_index,
            presentation_time: frame.start_time,
            duration: frame_duration,
            ready: proxy_video_slots
                .get(slot_index)
                .map(|slot| slot.is_ready())
                .unwrap_or(false)
                && proxy_audio_slots
                    .get(slot_index)
                    .map(|slot| slot.complete)
                    .unwrap_or(false),
        })
        .collect::<Vec<_>>();
    let proxy_slot_summary = summarize_broadcast_prepared_slots(
        config,
        plan,
        &proxy_video_slots,
        &proxy_audio_slots,
        &proxy_presentation_slots,
    );
    let proxy_audio_payload_bindings = proxy_audio_slots
        .iter()
        .map(|slot| bind_broadcast_audio_payload(slot, &blocks))
        .collect::<Result<Vec<_>, _>>()?;
    let proxy_video_payload_proof = if matches!(
        report_focus,
        BroadcastRuntimeReportFocus::VideoPayloads
            | BroadcastRuntimeReportFocus::DeviceBoundary
            | BroadcastRuntimeReportFocus::TestPresenter
            | BroadcastRuntimeReportFocus::TestAudioSink
            | BroadcastRuntimeReportFocus::SimulatedPlayback
            | BroadcastRuntimeReportFocus::OriginalVideoPayloads
    ) {
        Some(bind_proxy_preview_video_payloads(
            proxy_video,
            &proxy_h264,
            &proxy_video_slots,
        )?)
    } else {
        None
    };
    let proxy_video_payload_bindings = proxy_video_payload_proof
        .as_ref()
        .map(|proof| proof.bindings.clone())
        .unwrap_or_else(|| {
            proxy_video_slots
                .iter()
                .map(bind_broadcast_video_payload_accounting)
                .collect::<Vec<_>>()
        });
    let proxy_presentation_payload_bindings = proxy_presentation_slots
        .iter()
        .enumerate()
        .map(|(slot_index, slot)| {
            bind_broadcast_presentation_payload(
                slot,
                &proxy_video_payload_bindings[slot_index],
                &proxy_audio_payload_bindings[slot_index],
            )
        })
        .collect::<Vec<_>>();
    let proxy_payload_summary = summarize_broadcast_payload_bindings(
        BroadcastVideoSourceMode::ProxyPreview,
        config,
        &proxy_audio_payload_bindings,
        &proxy_video_payload_bindings,
        &proxy_presentation_payload_bindings,
    );

    let original_config = BroadcastPrerollConfig {
        video_source_mode: BroadcastVideoSourceMode::OriginalMedia,
        ..config
    };
    let original_payload_attempt =
        report_focus == BroadcastRuntimeReportFocus::OriginalVideoPayloads;
    let original_plan = BroadcastPrerollPlan {
        video_source_mode: BroadcastVideoSourceMode::OriginalMedia,
        selected_video_frames_planned: original.index.video.len(),
        audio_ranges_planned: ranges.len(),
        intentional_skips_planned: 0,
        duration_covered: audio_duration,
        finite_queue_limits: queue_limits,
        video_source_available: !original.index.video.is_empty(),
        video_runtime_supported: original_payload_attempt,
    };
    let original_video_slots = (0..prepared_len)
        .map(|slot_index| {
            let selected_frame = selected
                .get(slot_index)
                .ok_or("missing selected frame for original media slot")?;
            Ok::<_, Box<dyn std::error::Error>>(BroadcastPreparedVideoSlot {
                slot_index,
                source_mode: BroadcastVideoSourceMode::OriginalMedia,
                video_source_role: BroadcastMediaSourceRole::OriginalFinishingMedia,
                source_frame_index: Some(selected_frame.source_presentation_index),
                selected_preview_frame_index: None,
                presentation_time: selected_frame.start_time,
                duration: frame_duration,
                status: if original_payload_attempt {
                    BroadcastPreparedVideoSlotStatus::Prepared
                } else {
                    BroadcastPreparedVideoSlotStatus::CapabilityMissing
                },
            })
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    let original_audio_slots = ranges
        .iter()
        .take(prepared_len)
        .enumerate()
        .map(|(slot_index, range)| {
            BroadcastPreparedAudioSlot::from_audio_range(
                slot_index,
                BroadcastVideoSourceMode::OriginalMedia,
                range,
            )
        })
        .collect::<Vec<_>>();
    let original_presentation_slots = (0..prepared_len)
        .map(|slot_index| {
            let selected_frame = selected
                .get(slot_index)
                .ok_or("missing selected frame for original media presentation slot")?;
            Ok::<_, Box<dyn std::error::Error>>(BroadcastPreparedPresentationSlot {
                presentation_index: slot_index,
                source_mode: BroadcastVideoSourceMode::OriginalMedia,
                selected_source_frame: Some(selected_frame.source_presentation_index),
                video_slot_index: slot_index,
                audio_slot_index: slot_index,
                presentation_time: selected_frame.start_time,
                duration: frame_duration,
                ready: original_payload_attempt,
            })
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    let original_slot_summary = summarize_broadcast_prepared_slots(
        original_config,
        original_plan,
        &original_video_slots,
        &original_audio_slots,
        &original_presentation_slots,
    );
    let original_audio_payload_bindings = original_audio_slots
        .iter()
        .map(|slot| bind_broadcast_audio_payload(slot, &blocks))
        .collect::<Result<Vec<_>, _>>()?;
    let original_video_payload_proof = if original_payload_attempt {
        Some(bind_original_media_video_payloads(
            &original,
            &original_bytes,
            &original_video_slots,
        )?)
    } else {
        None
    };
    let original_video_payload_bindings = original_video_payload_proof
        .as_ref()
        .map(|proof| proof.bindings.clone())
        .unwrap_or_else(|| {
            original_video_slots
                .iter()
                .map(bind_broadcast_video_payload_accounting)
                .collect::<Vec<_>>()
        });
    let original_presentation_payload_bindings = original_presentation_slots
        .iter()
        .enumerate()
        .map(|(slot_index, slot)| {
            bind_broadcast_presentation_payload(
                slot,
                &original_video_payload_bindings[slot_index],
                &original_audio_payload_bindings[slot_index],
            )
        })
        .collect::<Vec<_>>();
    let original_payload_summary = summarize_broadcast_payload_bindings(
        BroadcastVideoSourceMode::OriginalMedia,
        original_config,
        &original_audio_payload_bindings,
        &original_video_payload_bindings,
        &original_presentation_payload_bindings,
    );

    let mut runtime = BroadcastRuntimeStateMachine::create(session)?;
    runtime.prepare_with_preroll(
        BroadcastRuntimePrepareFacts {
            selected_frame_count,
            intentional_profile_skips: intentional_skips,
            audio_ranges_complete: runtime_summary.incomplete_frames == 0,
            frames_outside_audio_range: runtime_summary.frames_outside_audio_range,
        },
        proxy_slot_summary.preroll_status,
    )?;
    let ready_reached = runtime.state() == qgs_media_runtime::BroadcastRuntimeState::Ready;
    let play_from_ready = runtime.play().is_ok();
    if play_from_ready {
        runtime.account_happy_path()?;
        runtime.drain()?;
        runtime.complete()?;
    }
    let proxy_events = build_broadcast_player_event_surface(
        0,
        session,
        config,
        plan,
        runtime.state(),
        runtime.accounting(),
        runtime.events(),
        &proxy_slot_summary,
        &proxy_video_slots,
        &proxy_audio_slots,
        &proxy_presentation_slots,
    );
    let proxy_event_summary =
        summarize_broadcast_player_runtime_events(&proxy_events, runtime.accounting());
    let original_events = build_broadcast_player_event_surface(
        1,
        BroadcastRuntimeSessionDescription {
            video_source: BroadcastMediaSourceRole::OriginalFinishingMedia,
            video_source_mode: BroadcastVideoSourceMode::OriginalMedia,
            ..session
        },
        original_config,
        original_plan,
        qgs_media_runtime::BroadcastRuntimeState::Preparing,
        qgs_media_runtime::BroadcastRuntimeAccounting {
            selected_frames_accounted: 0,
            audio_ranges_accounted: 0,
            intentional_profile_skips: 0,
            lateness_drops: 0,
        },
        &[
            qgs_media_runtime::BroadcastRuntimeEvent::SessionCreated,
            qgs_media_runtime::BroadcastRuntimeEvent::PreparingStarted,
        ],
        &original_slot_summary,
        &original_video_slots,
        &original_audio_slots,
        &original_presentation_slots,
    );
    let original_event_summary = summarize_broadcast_player_runtime_events(
        &original_events,
        qgs_media_runtime::BroadcastRuntimeAccounting {
            selected_frames_accounted: 0,
            audio_ranges_accounted: 0,
            intentional_profile_skips: 0,
            lateness_drops: 0,
        },
    );

    match report_focus {
        BroadcastRuntimeReportFocus::PreparedSlots => {
            println!("QGS Broadcast Player Runtime Prepared Payload Slots");
            println!("---------------------------------------------------");
        }
        BroadcastRuntimeReportFocus::Events => {
            println!("QGS Broadcast Player Runtime Event Surface");
            println!("------------------------------------------");
        }
        BroadcastRuntimeReportFocus::Payloads => {
            println!("QGS Broadcast Player Runtime Payload Binding");
            println!("--------------------------------------------");
        }
        BroadcastRuntimeReportFocus::VideoPayloads => {
            println!("QGS Broadcast Player Runtime Proxy Video Payload Binding");
            println!("-------------------------------------------------------");
        }
        BroadcastRuntimeReportFocus::DeviceBoundary => {
            println!("QGS Broadcast Player Runtime Device Boundary Contract");
            println!("-----------------------------------------------------");
        }
        BroadcastRuntimeReportFocus::TestPresenter => {
            println!("QGS Broadcast Player Runtime Test Video Presenter Evidence");
            println!("----------------------------------------------------------");
        }
        BroadcastRuntimeReportFocus::TestAudioSink => {
            println!("QGS Broadcast Player Runtime Test Audio Sink Evidence");
            println!("-----------------------------------------------------");
        }
        BroadcastRuntimeReportFocus::SimulatedPlayback => {
            println!("QGS Broadcast Player Runtime Simulated Playback Loop");
            println!("----------------------------------------------------");
        }
        BroadcastRuntimeReportFocus::OriginalVideoPayloads => {
            println!("QGS Broadcast Player Runtime OriginalMedia Video Payload Binding");
            println!("---------------------------------------------------------------");
        }
    }
    println!(
        "Not real playback: no speaker output, no display output, no real-time Broadcast Player loop"
    );
    if report_focus == BroadcastRuntimeReportFocus::TestPresenter {
        println!("Test presenter only: no real display output");
    } else if report_focus == BroadcastRuntimeReportFocus::TestAudioSink {
        println!("Test audio sink only: no real speaker output");
    } else if report_focus == BroadcastRuntimeReportFocus::SimulatedPlayback {
        println!("Test boundaries only: no real speaker output or display output");
        println!("No realtime scheduler");
    } else if report_focus == BroadcastRuntimeReportFocus::OriginalVideoPayloads {
        println!("Not realtime playback: no display output, no speaker output");
        println!("No FramePresented event");
    } else {
        println!("No FramePresented event");
    }
    println!("No QNC UI integration");
    println!("Audio source: original MXF");
    println!("Proxy AAC: not used");
    println!("Preview profile: journalist-50i-preview");
    println!("ProxyPreview:");
    println!("  video source: proxy MP4");
    println!(
        "  slots: video={}/{} audio={}/{} presentation={}/{}",
        proxy_slot_summary.video_slot_count,
        proxy_slot_summary.video_slot_capacity,
        proxy_slot_summary.audio_slot_count,
        proxy_slot_summary.audio_slot_capacity,
        proxy_slot_summary.presentation_slot_count,
        proxy_slot_summary.presentation_slot_capacity
    );
    println!(
        "  ready slots: video={} audio={} presentation={}",
        proxy_slot_summary.video_slots_prepared,
        proxy_slot_summary.audio_slots_complete,
        proxy_slot_summary.presentation_slots_ready
    );
    println!("  tracks covered: {}", proxy_slot_summary.tracks_covered);
    println!(
        "  source frames referenced: {}",
        proxy_video_slots
            .iter()
            .filter_map(|slot| slot.source_frame_index)
            .map(|frame| frame.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "  first audio range: start_sample={} sample_count={}",
        proxy_audio_slots
            .first()
            .map(|slot| slot.start_sample)
            .unwrap_or(0),
        proxy_audio_slots
            .first()
            .map(|slot| slot.sample_count)
            .unwrap_or(0)
    );
    println!(
        "  Ready: {} Play from Ready: {}",
        yes_no(ready_reached),
        yes_no(play_from_ready)
    );
    println!("  Intentional source-frame skips: {intentional_skips}");
    println!(
        "  Broadcast Player runtime events: {}",
        proxy_event_summary.event_count
    );
    println!(
        "  Event surface: runtime_ready={} prepared_slot_available={} frame_accounted={} audio_range_accounted={} intentional_skip={} lateness_drops={} runtime_completed={} frame_presented={}",
        yes_no(proxy_event_summary.runtime_ready),
        proxy_event_summary.prepared_slot_events,
        proxy_event_summary.frame_accounted_events,
        proxy_event_summary.audio_range_accounted_events,
        proxy_event_summary.intentional_skip_events,
        proxy_event_summary.lateness_drops,
        yes_no(proxy_event_summary.runtime_completed),
        proxy_event_summary.frame_presented_events
    );
    if report_focus == BroadcastRuntimeReportFocus::Events {
        println!(
            "  Event sequence summary: SessionCreated -> PrepareStarted -> PrerollPlanned -> PreparedSlotAvailable x{} -> RuntimeReady -> TransportStarted -> FrameAccounted/AudioRangeAccounted x{} -> IntentionalProfileSkip x{} -> RuntimeCompleted",
            proxy_event_summary.prepared_slot_events,
            proxy_event_summary.frame_accounted_events,
            proxy_event_summary.intentional_skip_events
        );
    }
    if matches!(
        report_focus,
        BroadcastRuntimeReportFocus::Payloads
            | BroadcastRuntimeReportFocus::VideoPayloads
            | BroadcastRuntimeReportFocus::DeviceBoundary
            | BroadcastRuntimeReportFocus::TestPresenter
            | BroadcastRuntimeReportFocus::TestAudioSink
            | BroadcastRuntimeReportFocus::SimulatedPlayback
            | BroadcastRuntimeReportFocus::OriginalVideoPayloads
    ) {
        let blocks_per_presentation = proxy_audio_payload_bindings
            .first()
            .map(|binding| binding.block_coverage.len())
            .unwrap_or(0);
        let bytes_per_presentation = proxy_audio_payload_bindings
            .first()
            .map(|binding| binding.total_referenced_payload_bytes)
            .unwrap_or(0);
        let video_binding_status = proxy_video_payload_bindings
            .first()
            .map(|binding| binding.status)
            .unwrap_or(BroadcastVideoPayloadBindingStatus::Missing);
        let presentation_readiness = proxy_presentation_payload_bindings
            .first()
            .map(|binding| binding.readiness);
        let presentation_readiness_label = presentation_readiness
            .map(|readiness| format!("{readiness:?}"))
            .unwrap_or_else(|| "Missing".to_string());
        println!(
            "  Payload bindings: audio={}/{} video={}/{} presentation={}/{}",
            proxy_payload_summary.audio_binding_count,
            proxy_payload_summary.audio_binding_capacity,
            proxy_payload_summary.video_binding_count,
            proxy_payload_summary.video_binding_capacity,
            proxy_payload_summary.presentation_binding_count,
            proxy_payload_summary.presentation_binding_capacity
        );
        println!(
            "  Audio blocks referenced per presentation: {}",
            blocks_per_presentation
        );
        println!(
            "  Audio bytes referenced per presentation: {}",
            bytes_per_presentation
        );
        println!(
            "  Total referenced audio bytes: {}",
            proxy_payload_summary.total_referenced_audio_bytes
        );
        println!("  Video binding status: {:?}", video_binding_status);
        println!(
            "  Presentation binding readiness: {}",
            presentation_readiness_label
        );
        println!(
            "  Runtime-accounting-ready presentations: {}",
            proxy_payload_summary.runtime_accounting_ready_presentations
        );
        println!(
            "  Payload-ready presentations: {}",
            proxy_payload_summary.payload_ready_presentations
        );
        println!(
            "  Device-payload-ready presentations: {}",
            proxy_payload_summary.device_payload_ready_presentations
        );
        if let Some(proof) = &proxy_video_payload_proof {
            let first_payload = proxy_video_payload_bindings
                .first()
                .and_then(|binding| binding.payload.as_ref());
            let source_frames = proxy_video_payload_bindings
                .iter()
                .filter_map(|binding| binding.source_frame_index)
                .map(|frame| frame.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            println!(
                "  Proxy video payloads: bound={} submissions={} completions={} bounded_gpu_slots={} device={}",
                proof.bindings.len(),
                proof.gpu_submissions,
                proof.gpu_completions,
                proof.bounded_gpu_slots(),
                proof.device_name
            );
            println!("  Bound source frame indices: {source_frames}");
            if let Some(payload) = first_payload {
                println!(
                    "  Payload kind: {:?} format={:?} backend={:?}",
                    payload.kind, payload.format, payload.backend_path
                );
                println!(
                    "  Payload dimensions: visible={}x{} coded={}x{}",
                    payload.visible_width,
                    payload.visible_height,
                    payload.coded_width,
                    payload.coded_height
                );
            }
        }
        if report_focus == BroadcastRuntimeReportFocus::DeviceBoundary {
            let audio_format = blocks
                .first()
                .map(|block| block.format)
                .ok_or("no PCM blocks available for device boundary report")?;
            let audio_binding = proxy_audio_payload_bindings
                .first()
                .ok_or("missing proxy audio payload binding")?;
            let video_binding = proxy_video_payload_bindings
                .first()
                .ok_or("missing proxy video payload binding")?;
            let audio_submission = build_broadcast_audio_device_submission(
                0,
                audio_binding,
                audio_format,
                BroadcastDeviceStatus::NotConfigured,
                &[],
            );
            let video_submission = build_broadcast_video_presenter_submission(
                0,
                video_binding,
                BroadcastDeviceStatus::NotConfigured,
                &[],
            );
            let boundary = summarize_broadcast_device_boundary(
                &audio_submission,
                &video_submission,
                &[],
                BroadcastDeviceStatus::NotConfigured,
                BroadcastDeviceStatus::NotConfigured,
            );
            println!(
                "  Device boundary: audio_payload_ready={} video_payload_ready={} device_payload_ready={} frame_presented={}",
                yes_no(boundary.audio_payload_ready),
                yes_no(boundary.video_payload_ready),
                yes_no(boundary.device_payload_ready),
                boundary.frame_presented_count
            );
            println!(
                "  Audio sink status: {:?} submission={:?}",
                boundary.audio_device_status, boundary.audio_submission_status
            );
            println!(
                "  Video presenter status: {:?} submission={:?}",
                boundary.video_presenter_status, boundary.video_submission_status
            );
            println!(
                "  Device-ready reason: {}",
                boundary.reason.unwrap_or("ready")
            );
        }
        if report_focus == BroadcastRuntimeReportFocus::TestPresenter {
            let audio_format = blocks
                .first()
                .map(|block| block.format)
                .ok_or("no PCM blocks available for test presenter report")?;
            let audio_binding = proxy_audio_payload_bindings
                .first()
                .ok_or("missing proxy audio payload binding")?;
            let audio_submission = build_broadcast_audio_device_submission(
                0,
                audio_binding,
                audio_format,
                BroadcastDeviceStatus::NotConfigured,
                &[],
            );
            let mut presenter =
                BroadcastTestVideoPresenter::new(BroadcastTestVideoPresenterConfig {
                    accepted_kind: BroadcastVideoPayloadKind::ProcessedGpuFrame,
                    accepted_format: BroadcastVideoPayloadFormat::RgbaU16,
                    visible_width: proxy_h264.width,
                    visible_height: proxy_h264.height,
                    coded_width: proxy_h264.coded_width,
                    coded_height: proxy_h264.coded_height,
                })?;
            let presenter_capabilities = presenter.capabilities();
            let video_submission = build_broadcast_video_presenter_submission(
                0,
                proxy_video_payload_bindings
                    .first()
                    .ok_or("missing proxy video payload binding")?,
                BroadcastDeviceStatus::Ready,
                &presenter_capabilities,
            );
            let mut evidence = Vec::new();
            for (presentation_binding, video_binding) in proxy_presentation_payload_bindings
                .iter()
                .zip(proxy_video_payload_bindings.iter())
            {
                evidence.push(presenter.submit(presentation_binding, video_binding)?);
            }
            let presenter_events = broadcast_player_events_from_presentation_evidence(&evidence);
            let presenter_event_summary = summarize_broadcast_player_runtime_events(
                &presenter_events,
                qgs_media_runtime::BroadcastRuntimeAccounting {
                    selected_frames_accounted: 0,
                    audio_ranges_accounted: 0,
                    intentional_profile_skips: 0,
                    lateness_drops: 0,
                },
            );
            let boundary = summarize_broadcast_device_boundary(
                &audio_submission,
                &video_submission,
                &evidence,
                BroadcastDeviceStatus::NotConfigured,
                BroadcastDeviceStatus::Ready,
            );
            let evidence_kind = evidence
                .first()
                .map(|item| format!("{:?}", item.evidence_kind))
                .unwrap_or_else(|| "None".to_string());
            println!(
                "  Test presenter: submitted={} accepted={} rejected={} evidence_records={}",
                proxy_video_payload_bindings.len(),
                presenter.accepted_count(),
                presenter.rejected_count(),
                evidence.len()
            );
            println!(
                "  Test presenter video submission status: {:?}",
                video_submission.status
            );
            println!(
                "  A/V DevicePayloadReady: {}",
                yes_no(boundary.device_payload_ready)
            );
            println!(
                "  Video presenter DevicePayloadReady: {}",
                yes_no(video_submission.status == BroadcastDevicePayloadStatus::DevicePayloadReady)
            );
            println!(
                "  FramePresented count: {}",
                presenter_event_summary.frame_presented_events
            );
            println!("  Evidence kind: {evidence_kind}");
            println!("  Evidence source: test video presenter, not real display output");
        }
        if report_focus == BroadcastRuntimeReportFocus::TestAudioSink {
            let audio_format = blocks
                .first()
                .map(|block| block.format)
                .ok_or("no PCM blocks available for test audio sink report")?;
            let PcmSampleFormat::SignedInteger {
                bits_per_sample, ..
            } = audio_format;
            let track_count = proxy_audio_payload_bindings
                .first()
                .map(|binding| binding.track_count)
                .unwrap_or(0);
            let mut sink = BroadcastTestAudioSink::new(BroadcastTestAudioSinkConfig {
                sample_rate,
                bits_per_sample,
                track_count,
            })?;
            let mut evidence = Vec::new();
            for (presentation_binding, audio_binding) in proxy_presentation_payload_bindings
                .iter()
                .zip(proxy_audio_payload_bindings.iter())
            {
                evidence.push(sink.submit(presentation_binding, audio_binding, audio_format)?);
            }
            let audio_events = broadcast_player_events_from_audio_sink_evidence(&evidence);
            let audio_event_summary = summarize_broadcast_player_runtime_events(
                &audio_events,
                qgs_media_runtime::BroadcastRuntimeAccounting {
                    selected_frames_accounted: 0,
                    audio_ranges_accounted: 0,
                    intentional_profile_skips: 0,
                    lateness_drops: 0,
                },
            );
            let evidence_kind = evidence
                .first()
                .map(|item| format!("{:?}", item.evidence_kind))
                .unwrap_or_else(|| "None".to_string());
            let bytes_per_presentation = proxy_audio_payload_bindings
                .first()
                .map(|binding| binding.total_referenced_payload_bytes)
                .unwrap_or(0);
            let samples_per_track = proxy_audio_payload_bindings
                .first()
                .map(|binding| binding.sample_count)
                .unwrap_or(0);
            println!(
                "  Test audio sink: submitted={} accepted={} rejected={} evidence_records={}",
                proxy_audio_payload_bindings.len(),
                sink.accepted_count(),
                sink.rejected_count(),
                evidence.len()
            );
            println!(
                "  Audio format accepted: signed integer PCM {}-bit {} Hz tracks={}",
                bits_per_sample, sample_rate, track_count
            );
            println!(
                "  Samples accepted per presentation range: {} per track",
                samples_per_track
            );
            println!(
                "  Audio bytes accepted per presentation range: {}",
                bytes_per_presentation
            );
            println!("  Total audio bytes accepted: {}", sink.bytes_accepted());
            println!(
                "  Total sample ranges accepted: {}",
                sink.samples_accepted()
            );
            println!("  Evidence kind: {evidence_kind}");
            println!(
                "  Audio evidence events: {} frame_presented={}",
                audio_events.len(),
                audio_event_summary.frame_presented_events
            );
            println!("  Evidence source: test audio sink, not real speaker output");
        }
        if report_focus == BroadcastRuntimeReportFocus::SimulatedPlayback {
            let audio_format = blocks
                .first()
                .map(|block| block.format)
                .ok_or("no PCM blocks available for simulated playback report")?;
            let PcmSampleFormat::SignedInteger {
                bits_per_sample, ..
            } = audio_format;
            let track_count = proxy_audio_payload_bindings
                .first()
                .map(|binding| binding.track_count)
                .unwrap_or(0);
            let mut sink = BroadcastTestAudioSink::new(BroadcastTestAudioSinkConfig {
                sample_rate,
                bits_per_sample,
                track_count,
            })?;
            let mut presenter =
                BroadcastTestVideoPresenter::new(BroadcastTestVideoPresenterConfig {
                    accepted_kind: BroadcastVideoPayloadKind::ProcessedGpuFrame,
                    accepted_format: BroadcastVideoPayloadFormat::RgbaU16,
                    visible_width: proxy_h264.width,
                    visible_height: proxy_h264.height,
                    coded_width: proxy_h264.coded_width,
                    coded_height: proxy_h264.coded_height,
                })?;
            let simulation = simulate_broadcast_player_runtime_loop(
                BroadcastVideoSourceMode::ProxyPreview,
                &proxy_presentation_payload_bindings,
                &proxy_audio_payload_bindings,
                &proxy_video_payload_bindings,
                audio_format,
                &mut sink,
                &mut presenter,
            );
            println!(
                "  Simulated playback: slots={} audio_submitted={} audio_accepted={} video_submitted={} video_accepted={}",
                simulation.summary.presentation_slots_attempted,
                simulation.summary.audio_submissions,
                simulation.summary.audio_accepted,
                simulation.summary.video_submissions,
                simulation.summary.video_accepted
            );
            println!(
                "  Test evidence: audio={} video={} total={}",
                simulation.summary.audio_evidence_count,
                simulation.summary.video_evidence_count,
                simulation.summary.test_evidence_count
            );
            println!(
                "  FramePresented count: {} (test presenter evidence only)",
                simulation.summary.frame_presented_count
            );
            println!("  Lateness drops: {}", simulation.summary.lateness_drops);
            println!("  Failed slots: {}", simulation.summary.failed_slots);
            println!("  Final state: {:?}", simulation.summary.final_state);
            println!("  Completed: {}", yes_no(simulation.summary.completed));
            println!("  Audio evidence source: test audio sink, not real speaker output");
            println!("  Video evidence source: test video presenter, not real display output");
        }
    }
    println!("OriginalMedia:");
    println!("  video source: original MXF");
    println!("  audio source: original MXF");
    println!(
        "  original video source present: {}",
        yes_no(!original.index.video.is_empty())
    );
    println!(
        "  slots: video={}/{} audio={}/{} presentation={}/{}",
        original_slot_summary.video_slot_count,
        original_slot_summary.video_slot_capacity,
        original_slot_summary.audio_slot_count,
        original_slot_summary.audio_slot_capacity,
        original_slot_summary.presentation_slot_count,
        original_slot_summary.presentation_slot_capacity
    );
    println!(
        "  ready slots: video={} audio={} presentation={}",
        original_slot_summary.video_slots_prepared,
        original_slot_summary.audio_slots_complete,
        original_slot_summary.presentation_slots_ready
    );
    println!(
        "  Ready: {} reason={:?}",
        yes_no(original_slot_summary.preroll_status.ready),
        original_slot_summary.preroll_status.reason
    );
    println!(
        "  Event surface: capability_missing={} broadcast_player_ready={}",
        yes_no(original_event_summary.capability_missing),
        yes_no(original_event_summary.runtime_ready)
    );
    if report_focus == BroadcastRuntimeReportFocus::Events {
        println!(
            "  Event sequence summary: SessionCreated -> PrepareStarted -> PrerollPlanned -> CapabilityMissing"
        );
    }
    if matches!(
        report_focus,
        BroadcastRuntimeReportFocus::Payloads
            | BroadcastRuntimeReportFocus::VideoPayloads
            | BroadcastRuntimeReportFocus::DeviceBoundary
            | BroadcastRuntimeReportFocus::TestPresenter
            | BroadcastRuntimeReportFocus::TestAudioSink
            | BroadcastRuntimeReportFocus::SimulatedPlayback
            | BroadcastRuntimeReportFocus::OriginalVideoPayloads
    ) {
        let video_binding_status = original_video_payload_bindings
            .first()
            .map(|binding| binding.status)
            .unwrap_or(BroadcastVideoPayloadBindingStatus::Missing);
        let presentation_readiness = original_presentation_payload_bindings
            .first()
            .map(|binding| binding.readiness);
        let presentation_readiness_label = presentation_readiness
            .map(|readiness| format!("{readiness:?}"))
            .unwrap_or_else(|| "Missing".to_string());
        println!(
            "  Payload bindings: audio={}/{} video={}/{} presentation={}/{}",
            original_payload_summary.audio_binding_count,
            original_payload_summary.audio_binding_capacity,
            original_payload_summary.video_binding_count,
            original_payload_summary.video_binding_capacity,
            original_payload_summary.presentation_binding_count,
            original_payload_summary.presentation_binding_capacity
        );
        println!(
            "  Audio binding possible: {}",
            yes_no(original_payload_summary.complete_audio_bindings > 0)
        );
        println!("  Video binding status: {:?}", video_binding_status);
        println!(
            "  Presentation binding readiness: {}",
            presentation_readiness_label
        );
        println!(
            "  Capability-missing presentations: {}",
            original_payload_summary.capability_missing_presentations
        );
        if report_focus == BroadcastRuntimeReportFocus::DeviceBoundary {
            println!("  Device boundary: not attempted for OriginalMedia");
            println!("  DevicePayloadReady: no");
            println!("  FramePresented: 0");
        }
        if report_focus == BroadcastRuntimeReportFocus::TestPresenter {
            println!("  Test presenter: not attempted for OriginalMedia");
            println!("  Original video payload: CapabilityMissing");
            println!("  FramePresented count: 0");
        }
        if report_focus == BroadcastRuntimeReportFocus::TestAudioSink {
            let audio_format = blocks
                .first()
                .map(|block| block.format)
                .ok_or("no PCM blocks available for OriginalMedia test audio sink report")?;
            let PcmSampleFormat::SignedInteger {
                bits_per_sample, ..
            } = audio_format;
            let track_count = original_audio_payload_bindings
                .first()
                .map(|binding| binding.track_count)
                .unwrap_or(0);
            let mut sink = BroadcastTestAudioSink::new(BroadcastTestAudioSinkConfig {
                sample_rate,
                bits_per_sample,
                track_count,
            })?;
            let mut accepted = 0_usize;
            for (presentation_binding, audio_binding) in original_presentation_payload_bindings
                .iter()
                .zip(original_audio_payload_bindings.iter())
            {
                if sink
                    .submit(presentation_binding, audio_binding, audio_format)
                    .is_ok()
                {
                    accepted += 1;
                }
            }
            println!(
                "  Test audio sink: accepted={} rejected={} total_bytes={}",
                accepted,
                sink.rejected_count(),
                sink.bytes_accepted()
            );
            println!("  Original video payload: CapabilityMissing");
            println!("  Full OriginalMedia presentation ready: no");
        }
        if report_focus == BroadcastRuntimeReportFocus::SimulatedPlayback {
            let audio_format = blocks
                .first()
                .map(|block| block.format)
                .ok_or("no PCM blocks available for OriginalMedia simulated playback report")?;
            let PcmSampleFormat::SignedInteger {
                bits_per_sample, ..
            } = audio_format;
            let track_count = original_audio_payload_bindings
                .first()
                .map(|binding| binding.track_count)
                .unwrap_or(0);
            let mut sink = BroadcastTestAudioSink::new(BroadcastTestAudioSinkConfig {
                sample_rate,
                bits_per_sample,
                track_count,
            })?;
            let mut presenter =
                BroadcastTestVideoPresenter::new(BroadcastTestVideoPresenterConfig {
                    accepted_kind: BroadcastVideoPayloadKind::ProcessedGpuFrame,
                    accepted_format: BroadcastVideoPayloadFormat::RgbaU16,
                    visible_width: proxy_h264.width,
                    visible_height: proxy_h264.height,
                    coded_width: proxy_h264.coded_width,
                    coded_height: proxy_h264.coded_height,
                })?;
            let simulation = simulate_broadcast_player_runtime_loop(
                BroadcastVideoSourceMode::OriginalMedia,
                &original_presentation_payload_bindings,
                &original_audio_payload_bindings,
                &original_video_payload_bindings,
                audio_format,
                &mut sink,
                &mut presenter,
            );
            println!(
                "  Simulated playback: completed={} final_state={:?} failed_slots={}",
                yes_no(simulation.summary.completed),
                simulation.summary.final_state,
                simulation.summary.failed_slots
            );
            println!(
                "  FramePresented count: {}",
                simulation.summary.frame_presented_count
            );
            println!("  Original video payload: CapabilityMissing");
            println!("  Simulation result: not ready / capability missing");
        }
        if report_focus == BroadcastRuntimeReportFocus::OriginalVideoPayloads {
            if let Some(proof) = &original_video_payload_proof {
                let target_frames = proof
                    .target_frames
                    .iter()
                    .map(|frame| frame.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                let first_payload = original_video_payload_bindings
                    .first()
                    .and_then(|binding| binding.payload.as_ref());
                println!(
                    "  Original video payloads: bound={} submissions={} completions={} bounded_gpu_slots={} device={}",
                    proof.bindings.len(),
                    proof.gpu_submissions,
                    proof.gpu_completions,
                    proof.bounded_gpu_slots(),
                    proof.device_name
                );
                println!("  Target source frames: {target_frames}");
                println!(
                    "  Random access start: {} decode_end={} decoded_frames={}",
                    proof.random_access_start, proof.decode_end, proof.decoded_frames
                );
                println!(
                    "  DEVELOPMENT OBSERVATION - NOT A BENCHMARK: {:.3}s for bounded original payload binding",
                    proof.wall_elapsed.as_secs_f64()
                );
                println!("  Source format: {}", proof.source_format);
                if let Some(payload) = first_payload {
                    println!(
                        "  Payload kind: {:?} format={:?} backend={:?}",
                        payload.kind, payload.format, payload.backend_path
                    );
                    println!(
                        "  Payload dimensions: visible={}x{} coded={}x{}",
                        payload.visible_width,
                        payload.visible_height,
                        payload.coded_width,
                        payload.coded_height
                    );
                }
                println!(
                    "  Presentation payload-ready: {}",
                    original_payload_summary.payload_ready_presentations
                );
                println!("  DevicePayloadReady: no");
                println!("  FramePresented: 0");
                println!("  Realtime support claimed: no");
                println!("  Proxy video used for OriginalMedia payload: no");
            }
        }
    }
    if report_focus == BroadcastRuntimeReportFocus::OriginalVideoPayloads {
        println!(
            "  Original video runtime backend: bounded payload binding only; realtime not claimed"
        );
    } else {
        println!("  Original video runtime backend: capability missing in this milestone");
    }

    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BroadcastRuntimeReportFocus {
    PreparedSlots,
    Events,
    Payloads,
    VideoPayloads,
    DeviceBoundary,
    TestPresenter,
    TestAudioSink,
    SimulatedPlayback,
    OriginalVideoPayloads,
}

#[derive(Clone, Debug)]
struct SelectedPreviewFrame {
    preview_index: u64,
    source_presentation_index: u64,
    sample_index: u32,
    start_time: Duration,
}

fn selected_journalist_preview_frames(
    proxy_video: &qgs_mp4::Mp4VideoTrack,
) -> Result<Vec<SelectedPreviewFrame>, Box<dyn std::error::Error>> {
    let mut ordered = proxy_video
        .samples
        .iter()
        .map(|sample| (sample.pts, sample.sample_index, sample))
        .collect::<Vec<_>>();
    ordered.sort_by_key(|(pts, sample_index, _)| (*pts, *sample_index));
    let media_origin_pts = ordered
        .first()
        .map(|(pts, _, _)| *pts)
        .ok_or("proxy video has no samples")?;
    let mut selected = Vec::new();
    for (source_presentation_index, (_, _, sample)) in ordered.into_iter().enumerate() {
        let source_presentation_index = u64::try_from(source_presentation_index)?;
        let Some(preview_index) = ProxyPlaybackProfile::Journalist50iPreview
            .presentation_position(source_presentation_index)
        else {
            continue;
        };
        let relative_pts = sample
            .pts
            .checked_sub(media_origin_pts)
            .ok_or("proxy PTS precedes media origin")?;
        let pts = u64::try_from(relative_pts).map_err(|_| "negative proxy PTS is unsupported")?;
        selected.push(SelectedPreviewFrame {
            preview_index,
            source_presentation_index,
            sample_index: sample.sample_index,
            start_time: duration_from_units(pts, u64::from(proxy_video.timescale), 1)?,
        });
    }
    Ok(selected)
}

fn build_original_pcm_blocks(
    source: &MediaSource,
    bytes: &[u8],
) -> Result<Vec<PcmAudioBlock>, Box<dyn std::error::Error>> {
    let mut blocks = Vec::with_capacity(source.index.audio.len());
    for entry_index in 0..source.index.audio.len() {
        let extracted = source.extract_pcm_audio_packet(bytes, entry_index)?;
        let track = source
            .tracks
            .iter()
            .find(|track| track.id == extracted.entry.track_id)
            .ok_or("PCM packet references missing audio track")?;
        let audio = track
            .audio
            .as_ref()
            .ok_or("PCM packet track has no audio descriptor")?;
        let sample_rate = rational_to_u32(audio.sample_rate.ok_or("audio sample rate missing")?)?;
        let bits_per_sample = audio.bit_depth.ok_or("audio bit depth missing")?;
        let format = PcmSampleFormat::SignedInteger {
            bits_per_sample,
            endian: PcmEndian::Little,
        };
        let packet = PcmAudioPacket::new(
            extracted.entry.track_id.0,
            extracted.entry.channel_index,
            duration_from_audio_samples(extracted.entry.start_sample, sample_rate)?,
            duration_from_audio_samples(u64::from(extracted.entry.sample_count), sample_rate)?,
            extracted.entry.sample_count,
            format,
            extracted.payload,
        )?;
        blocks.push(PcmAudioBlock::from_mono_packet(packet, sample_rate)?);
    }
    Ok(blocks)
}

fn build_original_pcm_blocks_from_index_range(
    path: &Path,
    index: &PcmAudioIndex,
    start_sample: u64,
    end_sample: u64,
) -> Result<Vec<PcmAudioBlock>, Box<dyn std::error::Error>> {
    let mut blocks = Vec::new();
    for entry in &index.audio {
        let entry_end = entry
            .start_sample
            .checked_add(u64::from(entry.sample_count))
            .ok_or("audio index entry sample range overflow")?;
        if entry_end <= start_sample || entry.start_sample >= end_sample {
            continue;
        }
        let extracted = read_pcm_audio_packet_at(path, entry)?;
        let track = index
            .tracks
            .iter()
            .find(|track| track.id == extracted.entry.track_id)
            .ok_or("PCM packet references missing audio track")?;
        let audio = track
            .audio
            .as_ref()
            .ok_or("PCM packet track has no audio descriptor")?;
        let sample_rate = rational_to_u32(audio.sample_rate.ok_or("audio sample rate missing")?)?;
        let bits_per_sample = audio.bit_depth.ok_or("audio bit depth missing")?;
        let format = PcmSampleFormat::SignedInteger {
            bits_per_sample,
            endian: PcmEndian::Little,
        };
        let packet = PcmAudioPacket::new(
            extracted.entry.track_id.0,
            extracted.entry.channel_index,
            duration_from_audio_samples(extracted.entry.start_sample, sample_rate)?,
            duration_from_audio_samples(u64::from(extracted.entry.sample_count), sample_rate)?,
            extracted.entry.sample_count,
            format,
            extracted.payload,
        )?;
        blocks.push(PcmAudioBlock::from_mono_packet(packet, sample_rate)?);
    }
    if blocks.is_empty() {
        return Err("bounded audio audit range extracted no PCM blocks".into());
    }
    Ok(blocks)
}

fn original_audio_sample_rate(
    audio_tracks: &[&qgs_mxf::MxfTrack],
) -> Result<u32, Box<dyn std::error::Error>> {
    let mut sample_rate = None;
    for track in audio_tracks {
        let audio = track.audio.as_ref().ok_or("audio descriptor missing")?;
        let track_rate = rational_to_u32(audio.sample_rate.ok_or("audio sample rate missing")?)?;
        if let Some(sample_rate) = sample_rate {
            if sample_rate != track_rate {
                return Err("mixed original audio sample rates are not supported yet".into());
            }
        } else {
            sample_rate = Some(track_rate);
        }
    }
    sample_rate.ok_or_else(|| "no original audio sample rate available".into())
}

fn representative_range_indices(len: usize) -> Vec<usize> {
    if len == 0 {
        return Vec::new();
    }
    let mut indices = vec![0, len / 2, len - 1];
    indices.sort_unstable();
    indices.dedup();
    indices
}

fn print_av_range(frame: &SelectedPreviewFrame, range: &AvFrameAudioRange) {
    let track_ids = range
        .covered_tracks
        .iter()
        .map(|coverage| coverage.track_id.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let total_gaps = range
        .covered_tracks
        .iter()
        .map(|coverage| u64::from(coverage.gaps))
        .sum::<u64>();
    let total_overlaps = range
        .covered_tracks
        .iter()
        .map(|coverage| u64::from(coverage.overlaps))
        .sum::<u64>();
    println!(
        "  preview_frame={} source_presentation={} sample_index={} video_start={:.3}s audio_samples={}..{} sample_count={} tracks=[{}] complete={} gaps={} overlaps={}",
        frame.preview_index,
        frame.source_presentation_index,
        frame.sample_index,
        frame.start_time.as_secs_f64(),
        range.audio_start_sample,
        range.audio_start_sample + range.audio_sample_count,
        range.audio_sample_count,
        track_ids,
        yes_no(range.complete),
        total_gaps,
        total_overlaps
    );
}

fn duration_from_units(
    units: u64,
    units_per_second: u64,
    second_scale: u64,
) -> Result<Duration, Box<dyn std::error::Error>> {
    if units_per_second == 0 || second_scale == 0 {
        return Err("invalid duration rate".into());
    }
    let nanos = u128::from(units)
        .checked_mul(u128::from(second_scale))
        .and_then(|value| value.checked_mul(1_000_000_000))
        .ok_or("duration overflow")?
        / u128::from(units_per_second);
    Ok(Duration::new(
        u64::try_from(nanos / 1_000_000_000).map_err(|_| "duration seconds overflow")?,
        u32::try_from(nanos % 1_000_000_000).map_err(|_| "duration nanos overflow")?,
    ))
}

fn rational_to_u32(value: qgs_mxf::Rational) -> Result<u32, Box<dyn std::error::Error>> {
    if value.denominator == 0 || value.numerator % value.denominator != 0 {
        return Err("non-integer audio sample rate is not supported in Step 20A".into());
    }
    Ok(value.numerator / value.denominator)
}

fn uniform_value<T: Copy + Eq>(values: &[T]) -> Option<T> {
    let first = *values.first()?;
    values.iter().all(|value| *value == first).then_some(first)
}

fn proxy_throughput(
    original_path: &Path,
    proxy_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let original_sha256 = sha256_hex(original_path)?;
    let proxy_sha256 = sha256_hex(proxy_path)?;
    let case = identify_camera_case(&original_sha256, &proxy_sha256)?;
    if case.damaged_proxy {
        return diagnose_damaged_proxy_case(case, proxy_path);
    }

    let original_bytes = std::fs::read(original_path)?;
    let original = MediaSource::parse(&original_bytes)?;
    let proxy = Mp4Source::open(proxy_path)?;
    let proxy_video = proxy
        .video
        .as_ref()
        .ok_or("proxy has no H.264 video track")?;
    let proxy_h264 = classify_video_track(proxy_video)?;
    let source_seconds = proxy_video.samples.len() as f64
        / (proxy_video.frame_rate.numerator as f64 / proxy_video.frame_rate.denominator as f64);
    let access_units = proxy_video
        .samples
        .iter()
        .map(|sample| sample.annex_b.clone())
        .collect::<Vec<_>>();

    println!("QGS proxy VA throughput audit:");
    println!("  original: {}", case.original_label);
    println!("  proxy: {}", case.proxy_label);
    println!("  hashes: verified");
    println!("  original edit units: {}", original.index.video.len());
    println!("  proxy samples: {}", proxy_video.samples.len());
    println!(
        "  proxy: H.264 {:?}, {}-bit {:?}, {} x {}, {}/{} fps",
        proxy_h264.profile,
        proxy_h264.bit_depth,
        proxy_h264.chroma,
        proxy_h264.width,
        proxy_h264.height,
        proxy_video.frame_rate.numerator,
        proxy_video.frame_rate.denominator
    );

    let frontend_start = Instant::now();
    let frontend = run_proxy_frontend_only(proxy_video)?;
    let frontend_elapsed = frontend_start.elapsed();
    print_decode_timing(
        "Container + H.264 frontend only",
        frontend.frames,
        source_seconds,
        frontend_elapsed,
    );
    println!(
        "  frontend peak DPB={} peak output_pending={}",
        frontend.peak_dpb_occupancy, frontend.peak_output_pending
    );

    let discovery = VulkanDeviceDiscovery::new()?;
    let devices = discovery.enumerate_devices()?;
    let Some(device) = devices
        .iter()
        .find(|device| {
            device.vendor_id == 0x8086 && matches!(device.class, DeviceClass::IntegratedGpu)
        })
        .or_else(|| {
            devices
                .iter()
                .find(|device| matches!(device.class, DeviceClass::IntegratedGpu))
        })
    else {
        println!("Intel VA proxy decode: no integrated GPU advertised");
        return Ok(());
    };
    let proxy_config = decoder_config_for_surface(
        device.id,
        proxy_h264.profile,
        BitDepth::new(proxy_h264.bit_depth)?,
        proxy_h264.chroma,
        proxy_h264.coded_width,
        proxy_h264.coded_height,
    );
    let vaapi = qgs_vaapi::VaapiVideoDiscovery::new(&devices);
    let capabilities = vaapi.query_video_capabilities(device.id)?;
    let supports_proxy = capabilities
        .decode
        .iter()
        .any(|capability| proxy_config.is_satisfied_by(capability));
    println!("Intel proxy capability:");
    println!("  device: {}", device.name);
    println!(
        "  H.264 {:?} {}-bit {:?}: {}",
        proxy_h264.profile,
        proxy_h264.bit_depth,
        proxy_h264.chroma,
        yes_no(supports_proxy)
    );
    if !supports_proxy {
        return Ok(());
    }

    let selected = selected_proxy_ordinals(proxy_video.samples.len());
    let normal_start = Instant::now();
    let normal = qgs_vaapi::decode_h264_access_units_for_observation(
        &devices,
        device.id,
        &proxy_config,
        &access_units,
        qgs_vaapi::VaapiDecodeMode::Normal,
        &selected,
    )?;
    let normal_elapsed = normal_start.elapsed();
    print_decode_timing(
        "Clean VA hardware decode",
        normal.frames,
        source_seconds,
        normal_elapsed,
    );
    print_va_observation(&normal);
    print_va_timing(&normal.timing);
    if normal.frames != proxy_video.samples.len() {
        return Err(format!(
            "clean VA decode expected {} frames, got {}",
            proxy_video.samples.len(),
            normal.frames
        )
        .into());
    }

    println!("Clean VA pool-size experiment:");
    for pool_size in [8_usize, 12, 16, 24, 32] {
        let pool_start = Instant::now();
        let observation = qgs_vaapi::decode_h264_access_units_for_observation_with_pool_size(
            &devices,
            device.id,
            &proxy_config,
            &access_units,
            qgs_vaapi::VaapiDecodeMode::Normal,
            &[],
            pool_size,
        )?;
        let elapsed = pool_start.elapsed();
        let seconds = elapsed.as_secs_f64().max(0.000_001);
        println!(
            "  pool={pool_size}: frames={} {:.3}s {:.2} fps recycle_sync={} peak_pending_recycle={} peak_unsynced={} min_free={} client_held={}",
            observation.frames,
            seconds,
            observation.frames as f64 / seconds,
            observation.pool_stats.recycle_sync_count,
            observation.pool_stats.peak_pending_recycle_surfaces,
            observation.peak_submitted_unsynced_surfaces,
            observation.pool_stats.minimum_free_surfaces,
            observation.max_client_held_outputs
        );
        if observation.frames != proxy_video.samples.len() {
            return Err(format!(
                "pool-size experiment {pool_size} expected {} frames, got {}",
                proxy_video.samples.len(),
                observation.frames
            )
            .into());
        }
    }

    let diagnostic_start = Instant::now();
    let diagnostic = qgs_vaapi::decode_h264_access_units_for_observation(
        &devices,
        device.id,
        &proxy_config,
        &access_units,
        qgs_vaapi::VaapiDecodeMode::Diagnostic,
        &selected,
    )?;
    let diagnostic_elapsed = diagnostic_start.elapsed();
    print_decode_timing(
        "Diagnostic VA hardware decode",
        diagnostic.frames,
        source_seconds,
        diagnostic_elapsed,
    );
    print_va_observation(&diagnostic);
    print_va_timing(&diagnostic.timing);
    println!(
        "  selected diagnostic checksums: {:?}",
        diagnostic.selected_output_checksums
    );
    if diagnostic.frames != proxy_video.samples.len() {
        return Err(format!(
            "diagnostic VA decode expected {} frames, got {}",
            proxy_video.samples.len(),
            diagnostic.frames
        )
        .into());
    }

    Ok(())
}

fn proxy_playback(
    original_path: &Path,
    proxy_path: &Path,
    profile: ProxyPlaybackProfile,
) -> Result<(), Box<dyn std::error::Error>> {
    let original_sha256 = sha256_hex(original_path)?;
    let proxy_sha256 = sha256_hex(proxy_path)?;
    let case = identify_camera_case(&original_sha256, &proxy_sha256)?;
    if case.damaged_proxy {
        return diagnose_damaged_proxy_case(case, proxy_path);
    }

    let original_bytes = std::fs::read(original_path)?;
    let original = MediaSource::parse(&original_bytes)?;
    let source_open_start = Instant::now();
    let proxy = Mp4Source::open(proxy_path)?;
    let source_open_elapsed = source_open_start.elapsed();
    let proxy_video = proxy
        .video
        .as_ref()
        .ok_or("proxy has no H.264 video track")?;
    let proxy_h264 = classify_video_track(proxy_video)?;
    let access_units = proxy_video
        .samples
        .iter()
        .map(|sample| sample.annex_b.clone())
        .collect::<Vec<_>>();
    let frame_count = access_units.len();
    let rate = RationalRate::new(
        u64::from(proxy_video.frame_rate.numerator),
        u64::from(proxy_video.frame_rate.denominator),
    )?;
    let source_duration = rate.duration_for_frames(u64::try_from(frame_count)?)?;
    let presentation_rate = profile.presentation_rate(rate)?;
    let selected_frame_count = profile.selected_frame_count(frame_count);
    let target_duration =
        presentation_rate.duration_for_frames(u64::try_from(selected_frame_count)?)?;
    let config = PlaybackConfig::default().validate()?;

    println!("QGS bounded realtime proxy playback:");
    println!("  profile: {} ({})", profile.label(), profile.description());
    println!("  original: {}", case.original_label);
    println!("  proxy: {}", case.proxy_label);
    println!("  hashes: verified");
    println!("  original edit units: {}", original.index.video.len());
    println!(
        "  proxy: H.264 {:?}, {}-bit {:?}, visible {} x {}, coded {} x {}, {}/{} fps",
        proxy_h264.profile,
        proxy_h264.bit_depth,
        proxy_h264.chroma,
        proxy_h264.width,
        proxy_h264.height,
        proxy_h264.coded_width,
        proxy_h264.coded_height,
        rate.numerator(),
        rate.denominator()
    );
    println!("  source presentation frames: {frame_count}");
    println!("  source duration: {:.3}s", source_duration.as_secs_f64());
    println!(
        "  selected presentation frames: {selected_frame_count} at {}/{} fps target={:.3}s",
        presentation_rate.numerator(),
        presentation_rate.denominator(),
        target_duration.as_secs_f64()
    );
    println!(
        "  queue capacities: compressed={} decoded={} gpu={} presentation={} preroll={}",
        config.compressed_capacity,
        config.decoded_capacity,
        config.gpu_capacity,
        config.presentation_capacity,
        config.preroll_frames
    );

    let discovery = VulkanDeviceDiscovery::new()?;
    let devices = discovery.enumerate_devices()?;
    let Some(device) = devices
        .iter()
        .find(|device| {
            device.vendor_id == 0x8086 && matches!(device.class, DeviceClass::IntegratedGpu)
        })
        .or_else(|| {
            devices
                .iter()
                .find(|device| matches!(device.class, DeviceClass::IntegratedGpu))
        })
    else {
        return Err("Intel VA proxy playback: no integrated GPU advertised".into());
    };

    let proxy_config = decoder_config_for_surface(
        device.id,
        proxy_h264.profile,
        BitDepth::new(proxy_h264.bit_depth)?,
        proxy_h264.chroma,
        proxy_h264.coded_width,
        proxy_h264.coded_height,
    );
    let vaapi = qgs_vaapi::VaapiVideoDiscovery::new(&devices);
    let capabilities = vaapi.query_video_capabilities(device.id)?;
    let supports_proxy = capabilities
        .decode
        .iter()
        .any(|capability| proxy_config.is_satisfied_by(capability));
    println!("Intel proxy capability:");
    println!("  device: {}", device.name);
    println!(
        "  H.264 {:?} {}-bit {:?}: {}",
        proxy_h264.profile,
        proxy_h264.bit_depth,
        proxy_h264.chroma,
        yes_no(supports_proxy)
    );
    if !supports_proxy {
        return Err("Intel VA backend does not support proxy configuration".into());
    }

    let decoder_create_start = Instant::now();
    let mut decoder = vaapi.create_decoder(&CreateDecoderRequest {
        config: proxy_config.clone(),
    })?;
    let decoder_create_elapsed = decoder_create_start.elapsed();
    let decoder_id = DecoderId::new(1)?;
    let visible_region = VisibleRegion {
        x: 0,
        y: 0,
        width: proxy_h264.width,
        height: proxy_h264.height,
    };
    let cpu_pool_capacity = 6_usize;
    let mut cpu_pool = qgs_vaapi::CpuNv12FramePool::new(
        proxy_h264.coded_width,
        proxy_h264.coded_height,
        visible_region,
        cpu_pool_capacity,
    )?;
    let mut gpu_processor = Nv12FrameProcessor::new(
        &discovery,
        Nv12FrameProcessorConfig {
            device_id: device.id,
            coded_width: proxy_h264.coded_width,
            coded_height: proxy_h264.coded_height,
            visible_width: proxy_h264.width,
            visible_height: proxy_h264.height,
            slot_count: config.gpu_capacity.max(1),
            conversion: YcbcrConversion::Rec709Limited,
            validation_readback: false,
        },
    )?;

    let playback_start = Instant::now();
    let mut clock = RealTimeClock::start_now();
    let mut sink = TestPresentationSink::new();
    let mut state = PlaybackState::Prerolling;
    let mut compressed_queue = BoundedQueue::new(config.compressed_capacity)?;
    let mut decoded_queue = BoundedQueue::new(config.decoded_capacity)?;
    let mut gpu_queue = BoundedQueue::new(config.gpu_capacity)?;
    let mut presentation_queue = BoundedQueue::new(config.presentation_capacity)?;
    let mut next_input = 0_usize;
    let mut next_output = 0_usize;
    let mut flushed = false;
    let mut access_units_submitted = 0_usize;
    let mut decoder_outputs = 0_usize;
    let mut intentionally_skipped_source_frames = 0_usize;
    let mut gpu_submissions = 0_usize;
    let mut gpu_completions = 0_usize;
    let mut decode_submit_elapsed = Duration::ZERO;
    let mut va_cpu_sync_copy_elapsed = Duration::ZERO;
    let mut gpu_submit_elapsed = Duration::ZERO;
    let mut gpu_completion_elapsed = Duration::ZERO;
    let mut presentation_wait_elapsed = Duration::ZERO;
    let mut max_lateness = Duration::ZERO;
    let mut lateness_values = Vec::new();
    let mut transfer_totals = Vec::new();
    let mut transfer_sync_times = Vec::new();
    let mut transfer_image_times = Vec::new();
    let mut transfer_copy_y_times = Vec::new();
    let mut transfer_copy_uv_times = Vec::new();
    let mut transfer_release_times = Vec::new();
    let mut transfer_bytes_copied = 0_usize;
    let mut first_va_image_layout = None;
    let mut pool_acquire_elapsed = Duration::ZERO;
    let mut upload_view_elapsed = Duration::ZERO;
    let mut pool_release_elapsed = Duration::ZERO;
    let mut gpu_submit_times = Vec::new();
    let mut gpu_completion_times = Vec::new();
    let progress_trace = std::env::var_os("QGS_STEP17_PROGRESS").is_some();
    let step17_diag = std::env::var_os("QGS_STEP17_DIAG").is_some();

    loop {
        while next_input < access_units.len() && !compressed_queue.is_full() {
            compressed_queue
                .try_push(ScheduledAccessUnit {
                    access_unit: access_units[next_input].clone(),
                })
                .map_err(|_| "compressed queue unexpectedly full")?;
            next_input += 1;
        }

        while !compressed_queue.is_empty() && !decoded_queue.is_full() {
            let scheduled = compressed_queue
                .pop_front()
                .ok_or("compressed queue unexpectedly empty")?;
            let started = Instant::now();
            let outputs = decoder.submit_access_unit(&SubmitAccessUnitRequest {
                decoder_id,
                data: scheduled.access_unit,
            })?;
            decode_submit_elapsed += started.elapsed();
            access_units_submitted += 1;
            for output in outputs {
                let source_position = u64::try_from(next_output)?;
                let presentation_position = profile.presentation_position(source_position);
                next_output += 1;
                decoded_queue
                    .try_push(DecodedPlaybackFrame {
                        source_position,
                        presentation_position,
                        surface: output,
                    })
                    .map_err(|_| "decoded queue unexpectedly full")?;
                decoder_outputs += 1;
            }
        }

        if next_input == access_units.len()
            && compressed_queue.is_empty()
            && !flushed
            && decoded_queue.len() < decoded_queue.capacity()
        {
            let started = Instant::now();
            let outputs = decoder.flush(&FlushDecoderRequest { decoder_id })?;
            decode_submit_elapsed += started.elapsed();
            flushed = true;
            for output in outputs {
                let source_position = u64::try_from(next_output)?;
                let presentation_position = profile.presentation_position(source_position);
                next_output += 1;
                decoded_queue
                    .try_push(DecodedPlaybackFrame {
                        source_position,
                        presentation_position,
                        surface: output,
                    })
                    .map_err(|_| "decoded queue unexpectedly full during flush")?;
                decoder_outputs += 1;
            }
        }

        while !decoded_queue.is_empty() && !gpu_queue.is_full() {
            let started = Instant::now();
            let decoded = decoded_queue
                .pop_front()
                .ok_or("decoded queue unexpectedly empty")?;
            let Some(presentation_position) = decoded.presentation_position else {
                intentionally_skipped_source_frames =
                    intentionally_skipped_source_frames.saturating_add(1);
                continue;
            };
            let identity = PlaybackFrameIdentity::from_position(presentation_position);
            let pool_started = Instant::now();
            let cpu_frame = cpu_pool.acquire()?;
            pool_acquire_elapsed += pool_started.elapsed();
            let transfer_started = Instant::now();
            if progress_trace {
                eprintln!(
                    "step17 progress: transfer start source={} presentation={}",
                    decoded.source_position, identity.presentation_position
                );
            }
            let (cpu_frame, transfer_timing) = qgs_vaapi::transfer_nv12_surface_timed(
                decoded.surface.resource.as_ref(),
                cpu_frame,
            )?;
            let transfer_elapsed = transfer_started.elapsed();
            va_cpu_sync_copy_elapsed += transfer_elapsed;
            transfer_totals.push(transfer_elapsed);
            transfer_sync_times.push(duration_from_ns(transfer_timing.sync_ns));
            transfer_image_times.push(duration_from_ns(transfer_timing.image_create_ns));
            transfer_copy_y_times.push(duration_from_ns(transfer_timing.copy_y_ns));
            transfer_copy_uv_times.push(duration_from_ns(transfer_timing.copy_uv_ns));
            transfer_release_times.push(duration_from_ns(transfer_timing.image_release_ns));
            transfer_bytes_copied =
                transfer_bytes_copied.saturating_add(transfer_timing.bytes_copied);
            if first_va_image_layout.is_none() {
                first_va_image_layout = transfer_timing.image_layout.clone();
            }
            if progress_trace {
                eprintln!(
                    "step17 progress: transfer done source={} presentation={}",
                    decoded.source_position, identity.presentation_position
                );
            }
            let upload_started = Instant::now();
            let upload = nv12_upload_for_cpu_surface(device.id, &cpu_frame)?;
            upload_view_elapsed += upload_started.elapsed();
            let token = match gpu_processor.submit_frame(
                &upload,
                FrameIdentity {
                    presentation_position: identity.presentation_position,
                },
            ) {
                Ok(token) => token,
                Err(FrameProcessorError::NoFrameSlotAvailable) => {
                    break;
                }
                Err(err) => return Err(Box::new(err)),
            };
            let release_started = Instant::now();
            cpu_pool.release(cpu_frame)?;
            pool_release_elapsed += release_started.elapsed();
            if progress_trace {
                eprintln!(
                    "step17 progress: gpu submit presentation={} token={}",
                    identity.presentation_position,
                    token.get()
                );
            }
            gpu_queue
                .try_push(GpuPendingPlaybackFrame { token, identity })
                .map_err(|_| "GPU queue unexpectedly full")?;
            gpu_submissions += 1;
            let submit_elapsed = started.elapsed();
            gpu_submit_elapsed += submit_elapsed;
            gpu_submit_times.push(submit_elapsed);
        }

        while !gpu_queue.is_empty() && !presentation_queue.is_full() {
            let started = Instant::now();
            let Some(front) = gpu_queue.front() else {
                break;
            };
            let completed = gpu_processor.poll_completed()?;
            if !completed.contains(&front.token) {
                let input_drained = flushed
                    && next_input == access_units.len()
                    && compressed_queue.is_empty()
                    && decoded_queue.is_empty();
                if gpu_queue.is_full() || input_drained {
                    gpu_processor.wait_for_completion(front.token)?;
                } else {
                    break;
                }
            }
            let processed = gpu_queue
                .pop_front()
                .ok_or("GPU queue unexpectedly empty")?;
            gpu_processor.retire_completed_token(processed.token)?;
            presentation_queue
                .try_push(ReadyPlaybackFrame {
                    identity: processed.identity,
                })
                .map_err(|_| "presentation queue unexpectedly full")?;
            if progress_trace {
                eprintln!(
                    "step17 progress: gpu complete presentation={} token={}",
                    processed.identity.presentation_position,
                    processed.token.get()
                );
            }
            gpu_completions += 1;
            let completion_elapsed = started.elapsed();
            gpu_completion_elapsed += completion_elapsed;
            gpu_completion_times.push(completion_elapsed);
        }

        if state == PlaybackState::Prerolling && presentation_queue.len() >= config.preroll_frames {
            state = PlaybackState::Playing;
            clock = RealTimeClock::start_now();
        }
        if state == PlaybackState::Prerolling && flushed && !presentation_queue.is_empty() {
            state = PlaybackState::Playing;
            clock = RealTimeClock::start_now();
        }

        if state == PlaybackState::Playing {
            if let Some(frame) = presentation_queue.front() {
                let expected =
                    presentation_rate.frame_offset(frame.identity.presentation_position)?;
                let wait_started = Instant::now();
                clock.sleep_until(expected);
                presentation_wait_elapsed += wait_started.elapsed();
                let actual = clock.now();
                let lateness = actual.saturating_sub(expected);
                max_lateness = max_lateness.max(lateness);
                lateness_values.push(lateness);
                let status = classify_presentation(lateness, config);
                let frame = presentation_queue
                    .pop_front()
                    .ok_or("presentation queue unexpectedly empty")?;
                sink.record(PresentationDecision {
                    identity: frame.identity,
                    expected,
                    actual,
                    lateness,
                    status,
                });
            }
        }

        if sink.decisions().len() == selected_frame_count {
            break;
        }

        if flushed
            && presentation_queue.is_empty()
            && decoded_queue.is_empty()
            && gpu_queue.is_empty()
        {
            state = PlaybackState::Completed;
            break;
        }
    }

    if state == PlaybackState::Playing {
        clock.sleep_until(target_duration);
        state = PlaybackState::Completed;
    }
    let playback_elapsed = playback_start.elapsed();

    if sink.decisions().len() != selected_frame_count {
        return Err(format!(
            "playback expected {selected_frame_count} presentation decisions, got {}",
            sink.decisions().len()
        )
        .into());
    }
    if decoder_outputs != frame_count {
        return Err(format!(
            "playback expected {frame_count} decoder outputs, got {decoder_outputs}"
        )
        .into());
    }

    let counts = sink.counts();
    let presented_total = counts.presented + counts.late;
    let mean_lateness = mean_duration(&lateness_values);
    let median_lateness = median_duration(&mut lateness_values);

    println!("Realtime playback result:");
    println!("  state: {:?}", state);
    println!("  access units submitted: {access_units_submitted}");
    println!("  decoder outputs: {decoder_outputs}");
    println!("  selected presentation frames: {selected_frame_count}");
    println!("  intentionally skipped source frames: {intentionally_skipped_source_frames}");
    println!("  GPU queue submissions: {gpu_submissions}");
    println!("  GPU queue completions: {gpu_completions}");
    println!(
        "  presentation decisions: {} presented={} on_time={} late={} dropped={} duplicated={}",
        sink.decisions().len(),
        presented_total,
        counts.presented,
        counts.late,
        counts.dropped,
        counts.duplicated
    );
    println!(
        "  playback wall-clock: {:.3}s target={:.3}s",
        playback_elapsed.as_secs_f64(),
        target_duration.as_secs_f64()
    );
    println!(
        "  lateness: max={:.3} ms mean={:.3} ms median={:.3} ms",
        duration_ms(max_lateness),
        duration_ms(mean_lateness),
        duration_ms(median_lateness)
    );
    println!(
        "  queue peaks: compressed={} decoded={} gpu={} presentation={}",
        compressed_queue.stats().peak_depth,
        decoded_queue.stats().peak_depth,
        gpu_queue.stats().peak_depth,
        presentation_queue.stats().peak_depth
    );
    println!(
        "  backpressure events: compressed={} decoded={} gpu={} presentation={}",
        compressed_queue.stats().backpressure_events,
        decoded_queue.stats().backpressure_events,
        gpu_queue.stats().backpressure_events,
        presentation_queue.stats().backpressure_events
    );
    let cpu_pool_stats = cpu_pool.stats();
    let gpu_counters = gpu_processor.counters();
    let gpu_diagnostics = gpu_processor.diagnostics();
    println!("  VA surfaces: transferred to bounded CPU NV12 pool before GPU upload; VA->Vulkan zero-copy remains frozen");
    println!(
        "  CPU NV12 pool: capacity={} bytes_per_frame={} peak_checked_out={} reused={}",
        cpu_pool_capacity,
        cpu_pool_stats.bytes_per_frame,
        cpu_pool_stats.peak_checked_out,
        cpu_pool_stats.frames_reused
    );
    println!(
        "  GPU processing: submissions={} completions={} slot_reuses={} staging_allocations={} gpu_plane_allocations={} output_allocations={}",
        gpu_submissions,
        gpu_completions,
        gpu_counters.slot_reuses,
        gpu_counters.staging_allocations,
        gpu_counters.gpu_plane_allocations,
        gpu_counters.output_allocations
    );
    println!("Stage timing observations:");
    println!(
        "  MP4 open/sample extraction: {:.3} ms",
        duration_ms(source_open_elapsed)
    );
    println!(
        "  VA decoder creation: {:.3} ms",
        duration_ms(decoder_create_elapsed)
    );
    println!(
        "  VA decode submit/flush aggregate: {:.3} ms",
        duration_ms(decode_submit_elapsed)
    );
    println!(
        "  VA sync + CPU transfer aggregate: {:.3} ms",
        duration_ms(va_cpu_sync_copy_elapsed)
    );
    print_duration_distribution("  VA transfer total/frame", &mut transfer_totals);
    print_duration_distribution("  VA transfer sync/frame", &mut transfer_sync_times);
    print_duration_distribution("  VA image derive/get/frame", &mut transfer_image_times);
    print_duration_distribution("  VA copy Y/frame", &mut transfer_copy_y_times);
    print_duration_distribution("  VA copy UV/frame", &mut transfer_copy_uv_times);
    print_duration_distribution("  VA image release/frame", &mut transfer_release_times);
    if let Some(layout) = &first_va_image_layout {
        print_va_nv12_image_layout("  VA NV12 image", layout);
    }
    println!(
        "  VA transfer bytes copied: total={} per_frame={}",
        transfer_bytes_copied,
        transfer_bytes_copied / selected_frame_count.max(1)
    );
    println!(
        "  CPU pool acquire/release/upload-view aggregate: acquire={:.3} ms upload_view={:.3} ms release={:.3} ms",
        duration_ms(pool_acquire_elapsed),
        duration_ms(upload_view_elapsed),
        duration_ms(pool_release_elapsed)
    );
    println!(
        "  decoded->GPU submit loop aggregate, including transfer: {:.3} ms",
        duration_ms(gpu_submit_elapsed)
    );
    print_duration_distribution(
        "  decoded->GPU submit loop/frame, including transfer",
        &mut gpu_submit_times,
    );
    println!(
        "  Vulkan completion handling aggregate: {:.3} ms",
        duration_ms(gpu_completion_elapsed)
    );
    print_duration_distribution("  Vulkan completion path/frame", &mut gpu_completion_times);
    print_nv12_gpu_diagnostics(&gpu_diagnostics);
    println!(
        "  presentation wait aggregate: {:.3} ms",
        duration_ms(presentation_wait_elapsed)
    );

    if step17_diag {
        run_step17_isolated_va_transfer_observation(
            &devices,
            device,
            proxy_config.clone(),
            &access_units,
            visible_region,
            source_duration,
        )?;
        run_step17_va_transfer_variant_observations(
            &devices,
            device,
            proxy_config.clone(),
            &access_units,
            visible_region,
            source_duration,
        )?;
        run_step17_as_fast_full_pixel_observation(
            &devices,
            &discovery,
            device,
            proxy_config,
            &access_units,
            visible_region,
            config,
            target_duration,
        )?;
        run_step17_synthetic_nv12_gpu_observation(
            &discovery,
            device,
            proxy_h264.coded_width,
            proxy_h264.coded_height,
            proxy_h264.width,
            proxy_h264.height,
            frame_count,
            target_duration,
        )?;
        if let Some(discrete_device) = devices
            .iter()
            .find(|candidate| matches!(candidate.class, DeviceClass::DiscreteGpu))
        {
            run_step17_synthetic_nv12_gpu_observation(
                &discovery,
                discrete_device,
                proxy_h264.coded_width,
                proxy_h264.coded_height,
                proxy_h264.width,
                proxy_h264.height,
                frame_count,
                target_duration,
            )?;
        }
        return Ok(());
    }

    if counts.dropped != 0 || counts.duplicated != 0 {
        return Err("real-time playback produced drops or duplicates".into());
    }
    if profile == ProxyPlaybackProfile::Journalist50iPreview
        && intentionally_skipped_source_frames != frame_count.saturating_sub(selected_frame_count)
    {
        return Err(
            "journalist-50i-preview profile skipped an unexpected number of source frames".into(),
        );
    }

    let selected = selected_proxy_ordinals(frame_count);
    let normal = qgs_vaapi::decode_h264_access_units_for_observation(
        &devices,
        device.id,
        &decoder_config_for_surface(
            device.id,
            proxy_h264.profile,
            BitDepth::new(proxy_h264.bit_depth)?,
            proxy_h264.chroma,
            proxy_h264.coded_width,
            proxy_h264.coded_height,
        ),
        &access_units,
        qgs_vaapi::VaapiDecodeMode::Normal,
        &selected,
    )?;
    println!("Clean VA ownership observation:");
    print_va_observation(&normal);

    let presentation_ordinals = proxy_presentation_ordinals(proxy_video)?;
    let proxy_positioned = proxy_video
        .samples
        .iter()
        .map(|sample| {
            Ok((
                *presentation_ordinals
                    .get(&sample.sample_index)
                    .ok_or("missing proxy playback presentation ordinal")?,
                sample.annex_b.clone(),
            ))
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    let software_frames = decode_positioned_access_units_with_context(
        decoder_config_for_surface(
            device.id,
            proxy_h264.profile,
            BitDepth::new(proxy_h264.bit_depth)?,
            proxy_h264.chroma,
            proxy_h264.coded_width,
            proxy_h264.coded_height,
        ),
        &proxy_positioned,
    )?;
    println!("Selected-frame validation:");
    let mut software_by_presentation = software_frames.frames.iter().collect::<Vec<_>>();
    software_by_presentation.sort_by_key(|frame| frame.presentation_index);
    for ordinal in selected {
        let frame = software_frames
            .frames
            .iter()
            .find(|frame| frame.presentation_index == ordinal as u64)
            .or_else(|| software_by_presentation.get(ordinal).copied())
            .ok_or_else(|| format!("missing software validation frame {ordinal}"))?;
        run_proxy_gpu_frame(
            &discovery,
            device,
            &format!("proxy-requested-{ordinal}"),
            frame,
        )?;
    }

    Ok(())
}

struct ScheduledAccessUnit {
    access_unit: Vec<u8>,
}

struct DecodedPlaybackFrame {
    source_position: u64,
    presentation_position: Option<u64>,
    surface: BackendDecodedSurface,
}

struct ReadyPlaybackFrame {
    identity: PlaybackFrameIdentity,
}

struct GpuPendingPlaybackFrame {
    token: qgs_vulkan::FrameToken,
    identity: PlaybackFrameIdentity,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProxyPlaybackProfile {
    SourceRate,
    Journalist50iPreview,
}

impl ProxyPlaybackProfile {
    fn parse(value: &std::ffi::OsStr) -> Result<Self, Box<dyn std::error::Error>> {
        match value.to_str() {
            Some("source-rate") => Ok(Self::SourceRate),
            Some("journalist-50i-preview") => Ok(Self::Journalist50iPreview),
            Some(other) => Err(format!("unsupported proxy playback profile: {other}").into()),
            None => Err("proxy playback profile must be valid UTF-8".into()),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::SourceRate => "source-rate",
            Self::Journalist50iPreview => "journalist-50i-preview",
        }
    }

    fn description(self) -> &'static str {
        match self {
            Self::SourceRate => "source-rate realtime proxy playback",
            Self::Journalist50iPreview => {
                "50i-compatible broadcast journalist preview over 50p source"
            }
        }
    }

    fn presentation_rate(
        self,
        source_rate: RationalRate,
    ) -> Result<RationalRate, Box<dyn std::error::Error>> {
        match self {
            Self::SourceRate => Ok(source_rate),
            Self::Journalist50iPreview => Ok(RationalRate::new(25, 1)?),
        }
    }

    fn presentation_position(self, source_position: u64) -> Option<u64> {
        match self {
            Self::SourceRate => Some(source_position),
            Self::Journalist50iPreview => source_position
                .is_multiple_of(2)
                .then_some(source_position / 2),
        }
    }

    fn selected_frame_count(self, source_frame_count: usize) -> usize {
        match self {
            Self::SourceRate => source_frame_count,
            Self::Journalist50iPreview => source_frame_count.div_ceil(2),
        }
    }
}

fn nv12_upload_for_cpu_surface<'a>(
    device_id: qgs_protocol::DeviceId,
    surface: &'a qgs_vaapi::CpuNv12Surface,
) -> Result<Nv12Upload<'a>, Box<dyn std::error::Error>> {
    Ok(Nv12Upload {
        device_id,
        coded_width: surface.desc.coded_width,
        coded_height: surface.desc.coded_height,
        visible_width: surface.desc.visible_region.width,
        visible_height: surface.desc.visible_region.height,
        y: Nv12Plane {
            width_bytes: u32::try_from(surface.y.width_bytes)?,
            height: u32::try_from(surface.y.height)?,
            stride_bytes: surface.y.stride_bytes,
            data: &surface.y.data,
        },
        uv: Nv12Plane {
            width_bytes: u32::try_from(surface.uv.width_bytes)?,
            height: u32::try_from(surface.uv.height)?,
            stride_bytes: surface.uv.stride_bytes,
            data: &surface.uv.data,
        },
        conversion: YcbcrConversion::Rec709Limited,
    })
}

fn run_step17_isolated_va_transfer_observation(
    devices: &[DeviceDesc],
    device: &DeviceDesc,
    proxy_config: DecoderConfig,
    access_units: &[Vec<u8>],
    visible_region: VisibleRegion,
    source_duration: Duration,
) -> Result<(), Box<dyn std::error::Error>> {
    let vaapi = qgs_vaapi::VaapiVideoDiscovery::new(devices);
    let mut decoder = vaapi.create_decoder(&CreateDecoderRequest {
        config: proxy_config.clone(),
    })?;
    let decoder_id = DecoderId::new(17)?;
    let mut cpu_pool = qgs_vaapi::CpuNv12FramePool::new(
        proxy_config.coded_width,
        proxy_config.coded_height,
        visible_region,
        6,
    )?;
    let mut decode_elapsed = Duration::ZERO;
    let mut transfer_elapsed = Duration::ZERO;
    let mut frames = 0_usize;
    let mut transfer_totals = Vec::new();
    let started = Instant::now();
    for access_unit in access_units {
        let decode_started = Instant::now();
        let outputs = decoder.submit_access_unit(&SubmitAccessUnitRequest {
            decoder_id,
            data: access_unit.clone(),
        })?;
        decode_elapsed += decode_started.elapsed();
        for output in outputs {
            let frame = cpu_pool.acquire()?;
            let transfer_started = Instant::now();
            let (frame, _timing) =
                qgs_vaapi::transfer_nv12_surface_timed(output.resource.as_ref(), frame)?;
            let elapsed = transfer_started.elapsed();
            transfer_elapsed += elapsed;
            transfer_totals.push(elapsed);
            cpu_pool.release(frame)?;
            frames += 1;
        }
    }
    let decode_started = Instant::now();
    let outputs = decoder.flush(&FlushDecoderRequest { decoder_id })?;
    decode_elapsed += decode_started.elapsed();
    for output in outputs {
        let frame = cpu_pool.acquire()?;
        let transfer_started = Instant::now();
        let (frame, _timing) =
            qgs_vaapi::transfer_nv12_surface_timed(output.resource.as_ref(), frame)?;
        let elapsed = transfer_started.elapsed();
        transfer_elapsed += elapsed;
        transfer_totals.push(elapsed);
        cpu_pool.release(frame)?;
        frames += 1;
    }
    let elapsed = started.elapsed();
    println!("Step 17 isolated VA decode + CPU NV12 transfer:");
    println!(
        "  device: {} frames={} wall={:.3} ms source={:.3}s throughput={:.2} fps realtime={:.2}x",
        device.name,
        frames,
        duration_ms(elapsed),
        source_duration.as_secs_f64(),
        frames as f64 / elapsed.as_secs_f64().max(0.000_001),
        source_duration.as_secs_f64() / elapsed.as_secs_f64().max(0.000_001)
    );
    println!(
        "  decode aggregate={:.3} ms transfer aggregate={:.3} ms pool_reuses={}",
        duration_ms(decode_elapsed),
        duration_ms(transfer_elapsed),
        cpu_pool.stats().frames_reused
    );
    print_duration_distribution("  isolated VA transfer/frame", &mut transfer_totals);
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum Step17TransferVariant {
    TightCoded,
    TightVisible,
    SourcePitchCoded,
    FullSourceRows,
}

impl Step17TransferVariant {
    const ALL: [Self; 4] = [
        Self::TightCoded,
        Self::TightVisible,
        Self::SourcePitchCoded,
        Self::FullSourceRows,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::TightCoded => "tight-coded",
            Self::TightVisible => "tight-visible",
            Self::SourcePitchCoded => "source-pitch-coded",
            Self::FullSourceRows => "full-source-rows",
        }
    }

    fn layout(
        self,
        proxy_config: &DecoderConfig,
        visible_region: VisibleRegion,
        image_layout: &qgs_vaapi::VaNv12ImageLayout,
    ) -> Result<qgs_vaapi::CpuNv12FrameLayout, Box<dyn std::error::Error>> {
        Ok(match self {
            Self::TightCoded => qgs_vaapi::CpuNv12FrameLayout::tight_coded(
                proxy_config.coded_width,
                proxy_config.coded_height,
            )?,
            Self::TightVisible => qgs_vaapi::CpuNv12FrameLayout::tight_visible(visible_region)?,
            Self::SourcePitchCoded => qgs_vaapi::CpuNv12FrameLayout::coded_with_destination_pitch(
                proxy_config.coded_width,
                proxy_config.coded_height,
                image_layout.pitches[0],
                image_layout.pitches[1],
            )?,
            Self::FullSourceRows => qgs_vaapi::CpuNv12FrameLayout::full_source_rows(
                usize::try_from(image_layout.height)?,
                image_layout.pitches[0],
                usize::try_from(image_layout.height)? / 2,
                image_layout.pitches[1],
            )?,
        })
    }
}

fn run_step17_va_transfer_variant_observations(
    devices: &[DeviceDesc],
    device: &DeviceDesc,
    proxy_config: DecoderConfig,
    access_units: &[Vec<u8>],
    visible_region: VisibleRegion,
    source_duration: Duration,
) -> Result<(), Box<dyn std::error::Error>> {
    let image_layout =
        probe_step17_first_va_nv12_layout(devices, proxy_config.clone(), access_units)?;
    println!("Step 17C VA NV12 image layout:");
    print_va_nv12_image_layout("  probed VA NV12 image", &image_layout);

    for variant in Step17TransferVariant::ALL {
        let layout = variant.layout(&proxy_config, visible_region, &image_layout)?;
        run_step17_va_transfer_variant(
            devices,
            device,
            proxy_config.clone(),
            access_units,
            visible_region,
            source_duration,
            variant,
            layout,
        )?;
    }
    Ok(())
}

fn probe_step17_first_va_nv12_layout(
    devices: &[DeviceDesc],
    proxy_config: DecoderConfig,
    access_units: &[Vec<u8>],
) -> Result<qgs_vaapi::VaNv12ImageLayout, Box<dyn std::error::Error>> {
    let vaapi = qgs_vaapi::VaapiVideoDiscovery::new(devices);
    let mut decoder = vaapi.create_decoder(&CreateDecoderRequest {
        config: proxy_config,
    })?;
    let decoder_id = DecoderId::new(17)?;
    for access_unit in access_units {
        let outputs = decoder.submit_access_unit(&SubmitAccessUnitRequest {
            decoder_id,
            data: access_unit.clone(),
        })?;
        if let Some(output) = outputs.first() {
            return Ok(qgs_vaapi::describe_nv12_surface_image(
                output.resource.as_ref(),
            )?);
        }
    }
    let outputs = decoder.flush(&FlushDecoderRequest { decoder_id })?;
    let output = outputs
        .first()
        .ok_or("proxy decode did not produce a surface to inspect")?;
    Ok(qgs_vaapi::describe_nv12_surface_image(
        output.resource.as_ref(),
    )?)
}

fn run_step17_va_transfer_variant(
    devices: &[DeviceDesc],
    device: &DeviceDesc,
    proxy_config: DecoderConfig,
    access_units: &[Vec<u8>],
    visible_region: VisibleRegion,
    source_duration: Duration,
    variant: Step17TransferVariant,
    layout: qgs_vaapi::CpuNv12FrameLayout,
) -> Result<(), Box<dyn std::error::Error>> {
    let vaapi = qgs_vaapi::VaapiVideoDiscovery::new(devices);
    let mut decoder = vaapi.create_decoder(&CreateDecoderRequest {
        config: proxy_config.clone(),
    })?;
    let decoder_id = DecoderId::new(17)?;
    let mut cpu_pool = qgs_vaapi::CpuNv12FramePool::new_with_layout(
        proxy_config.coded_width,
        proxy_config.coded_height,
        visible_region,
        6,
        layout,
    )?;
    let mut decode_elapsed = Duration::ZERO;
    let mut transfer_elapsed = Duration::ZERO;
    let mut frames = 0_usize;
    let mut bytes_copied = 0_usize;
    let mut transfer_totals = Vec::new();
    let mut sync_times = Vec::new();
    let mut image_times = Vec::new();
    let mut copy_y_times = Vec::new();
    let mut copy_uv_times = Vec::new();
    let mut release_times = Vec::new();
    let started = Instant::now();
    for access_unit in access_units {
        let decode_started = Instant::now();
        let outputs = decoder.submit_access_unit(&SubmitAccessUnitRequest {
            decoder_id,
            data: access_unit.clone(),
        })?;
        decode_elapsed += decode_started.elapsed();
        for output in outputs {
            let frame = cpu_pool.acquire()?;
            let transfer_started = Instant::now();
            let (frame, timing) =
                qgs_vaapi::transfer_nv12_surface_timed(output.resource.as_ref(), frame)?;
            let elapsed = transfer_started.elapsed();
            transfer_elapsed += elapsed;
            transfer_totals.push(elapsed);
            sync_times.push(duration_from_ns(timing.sync_ns));
            image_times.push(duration_from_ns(timing.image_create_ns));
            copy_y_times.push(duration_from_ns(timing.copy_y_ns));
            copy_uv_times.push(duration_from_ns(timing.copy_uv_ns));
            release_times.push(duration_from_ns(timing.image_release_ns));
            bytes_copied = bytes_copied.saturating_add(timing.bytes_copied);
            cpu_pool.release(frame)?;
            frames += 1;
        }
    }
    let decode_started = Instant::now();
    let outputs = decoder.flush(&FlushDecoderRequest { decoder_id })?;
    decode_elapsed += decode_started.elapsed();
    for output in outputs {
        let frame = cpu_pool.acquire()?;
        let transfer_started = Instant::now();
        let (frame, timing) =
            qgs_vaapi::transfer_nv12_surface_timed(output.resource.as_ref(), frame)?;
        let elapsed = transfer_started.elapsed();
        transfer_elapsed += elapsed;
        transfer_totals.push(elapsed);
        sync_times.push(duration_from_ns(timing.sync_ns));
        image_times.push(duration_from_ns(timing.image_create_ns));
        copy_y_times.push(duration_from_ns(timing.copy_y_ns));
        copy_uv_times.push(duration_from_ns(timing.copy_uv_ns));
        release_times.push(duration_from_ns(timing.image_release_ns));
        bytes_copied = bytes_copied.saturating_add(timing.bytes_copied);
        cpu_pool.release(frame)?;
        frames += 1;
    }
    let elapsed = started.elapsed();
    println!("Step 17C VA transfer variant: {}", variant.label());
    println!(
        "  layout: y width={} height={} stride={} uv width={} height={} stride={} bytes/frame={}",
        layout.y_width_bytes,
        layout.y_height,
        layout.y_stride_bytes,
        layout.uv_width_bytes,
        layout.uv_height,
        layout.uv_stride_bytes,
        cpu_pool.stats().bytes_per_frame
    );
    println!(
        "  device: {} frames={} wall={:.3} ms source={:.3}s throughput={:.2} fps realtime={:.2}x",
        device.name,
        frames,
        duration_ms(elapsed),
        source_duration.as_secs_f64(),
        frames as f64 / elapsed.as_secs_f64().max(0.000_001),
        source_duration.as_secs_f64() / elapsed.as_secs_f64().max(0.000_001)
    );
    println!(
        "  decode aggregate={:.3} ms transfer aggregate={:.3} ms bytes_copied/frame={} pool_reuses={}",
        duration_ms(decode_elapsed),
        duration_ms(transfer_elapsed),
        bytes_copied / frames.max(1),
        cpu_pool.stats().frames_reused
    );
    print_duration_distribution("  variant VA transfer/frame", &mut transfer_totals);
    print_duration_distribution("  variant VA sync/frame", &mut sync_times);
    print_duration_distribution("  variant VA image derive/get/frame", &mut image_times);
    print_duration_distribution("  variant VA copy Y/frame", &mut copy_y_times);
    print_duration_distribution("  variant VA copy UV/frame", &mut copy_uv_times);
    print_duration_distribution("  variant VA image release/frame", &mut release_times);
    Ok(())
}

fn print_va_nv12_image_layout(label: &str, layout: &qgs_vaapi::VaNv12ImageLayout) {
    println!(
        "{label}: fourcc={} width={} height={} planes={} data_size={} derived={}",
        layout.fourcc_string(),
        layout.width,
        layout.height,
        layout.num_planes,
        layout.data_size,
        yes_no(layout.derived)
    );
    println!(
        "    offsets={:?} pitches={:?} y_uv_contiguous={}",
        layout.offsets,
        layout.pitches,
        yes_no(layout.y_uv_contiguous())
    );
}

fn run_step17_as_fast_full_pixel_observation(
    devices: &[DeviceDesc],
    discovery: &VulkanDeviceDiscovery,
    device: &DeviceDesc,
    proxy_config: DecoderConfig,
    access_units: &[Vec<u8>],
    visible_region: VisibleRegion,
    config: PlaybackConfig,
    source_duration: Duration,
) -> Result<(), Box<dyn std::error::Error>> {
    let vaapi = qgs_vaapi::VaapiVideoDiscovery::new(devices);
    let mut decoder = vaapi.create_decoder(&CreateDecoderRequest {
        config: proxy_config.clone(),
    })?;
    let decoder_id = DecoderId::new(17)?;
    let mut cpu_pool = qgs_vaapi::CpuNv12FramePool::new(
        proxy_config.coded_width,
        proxy_config.coded_height,
        visible_region,
        6,
    )?;
    let mut gpu_processor = Nv12FrameProcessor::new(
        discovery,
        Nv12FrameProcessorConfig {
            device_id: device.id,
            coded_width: proxy_config.coded_width,
            coded_height: proxy_config.coded_height,
            visible_width: visible_region.width,
            visible_height: visible_region.height,
            slot_count: config.gpu_capacity.max(1),
            conversion: YcbcrConversion::Rec709Limited,
            validation_readback: false,
        },
    )?;
    let mut compressed_queue = BoundedQueue::new(config.compressed_capacity)?;
    let mut decoded_queue = BoundedQueue::new(config.decoded_capacity)?;
    let mut gpu_queue = BoundedQueue::new(config.gpu_capacity)?;
    let mut next_input = 0_usize;
    let mut next_output = 0_usize;
    let mut flushed = false;
    let mut access_units_submitted = 0_usize;
    let mut decoder_outputs = 0_usize;
    let mut gpu_submissions = 0_usize;
    let mut gpu_completions = 0_usize;
    let mut decode_elapsed = Duration::ZERO;
    let mut transfer_elapsed = Duration::ZERO;
    let mut gpu_submit_elapsed = Duration::ZERO;
    let mut gpu_completion_elapsed = Duration::ZERO;
    let mut transfer_bytes = 0_usize;
    let started = Instant::now();

    loop {
        while next_input < access_units.len() && !compressed_queue.is_full() {
            compressed_queue
                .try_push(ScheduledAccessUnit {
                    access_unit: access_units[next_input].clone(),
                })
                .map_err(|_| "compressed queue unexpectedly full")?;
            next_input += 1;
        }

        while !compressed_queue.is_empty() && !decoded_queue.is_full() {
            let scheduled = compressed_queue
                .pop_front()
                .ok_or("compressed queue unexpectedly empty")?;
            let decode_started = Instant::now();
            let outputs = decoder.submit_access_unit(&SubmitAccessUnitRequest {
                decoder_id,
                data: scheduled.access_unit,
            })?;
            decode_elapsed += decode_started.elapsed();
            access_units_submitted += 1;
            for output in outputs {
                let source_position = u64::try_from(next_output)?;
                next_output += 1;
                decoded_queue
                    .try_push(DecodedPlaybackFrame {
                        source_position,
                        presentation_position: Some(source_position),
                        surface: output,
                    })
                    .map_err(|_| "decoded queue unexpectedly full")?;
                decoder_outputs += 1;
            }
        }

        if next_input == access_units.len()
            && compressed_queue.is_empty()
            && !flushed
            && decoded_queue.len() < decoded_queue.capacity()
        {
            let decode_started = Instant::now();
            let outputs = decoder.flush(&FlushDecoderRequest { decoder_id })?;
            decode_elapsed += decode_started.elapsed();
            flushed = true;
            for output in outputs {
                let source_position = u64::try_from(next_output)?;
                next_output += 1;
                decoded_queue
                    .try_push(DecodedPlaybackFrame {
                        source_position,
                        presentation_position: Some(source_position),
                        surface: output,
                    })
                    .map_err(|_| "decoded queue unexpectedly full during flush")?;
                decoder_outputs += 1;
            }
        }

        while !decoded_queue.is_empty() && !gpu_queue.is_full() {
            let submit_started = Instant::now();
            let decoded = decoded_queue
                .pop_front()
                .ok_or("decoded queue unexpectedly empty")?;
            let presentation_position = decoded
                .presentation_position
                .ok_or("as-fast diagnostic expected selected frame")?;
            let identity = PlaybackFrameIdentity::from_position(presentation_position);
            let cpu_frame = cpu_pool.acquire()?;
            let transfer_started = Instant::now();
            let (cpu_frame, transfer_timing) = qgs_vaapi::transfer_nv12_surface_timed(
                decoded.surface.resource.as_ref(),
                cpu_frame,
            )?;
            transfer_elapsed += transfer_started.elapsed();
            transfer_bytes = transfer_bytes.saturating_add(transfer_timing.bytes_copied);
            let upload = nv12_upload_for_cpu_surface(device.id, &cpu_frame)?;
            let token = gpu_processor.submit_frame(
                &upload,
                FrameIdentity {
                    presentation_position: identity.presentation_position,
                },
            )?;
            cpu_pool.release(cpu_frame)?;
            gpu_queue
                .try_push(GpuPendingPlaybackFrame { token, identity })
                .map_err(|_| "GPU queue unexpectedly full")?;
            gpu_submissions += 1;
            gpu_submit_elapsed += submit_started.elapsed();
        }

        while !gpu_queue.is_empty() {
            let completion_started = Instant::now();
            let Some(front) = gpu_queue.front() else {
                break;
            };
            let completed = gpu_processor.poll_completed()?;
            if !completed.contains(&front.token) {
                if gpu_queue.is_full()
                    || (flushed && compressed_queue.is_empty() && decoded_queue.is_empty())
                {
                    gpu_processor.wait_for_completion(front.token)?;
                } else {
                    break;
                }
            }
            let processed = gpu_queue
                .pop_front()
                .ok_or("GPU queue unexpectedly empty")?;
            gpu_processor.retire_completed_token(processed.token)?;
            gpu_completions += 1;
            gpu_completion_elapsed += completion_started.elapsed();
        }

        if flushed
            && next_input == access_units.len()
            && compressed_queue.is_empty()
            && decoded_queue.is_empty()
            && gpu_queue.is_empty()
        {
            break;
        }
    }

    let elapsed = started.elapsed();
    println!("Step 17C as-fast full pixel path:");
    println!(
        "  device: {} frames={} wall={:.3} ms source={:.3}s throughput={:.2} fps realtime={:.2}x",
        device.name,
        gpu_completions,
        duration_ms(elapsed),
        source_duration.as_secs_f64(),
        gpu_completions as f64 / elapsed.as_secs_f64().max(0.000_001),
        source_duration.as_secs_f64() / elapsed.as_secs_f64().max(0.000_001)
    );
    println!(
        "  access_units={} decoder_outputs={} gpu_submissions={} gpu_completions={}",
        access_units_submitted, decoder_outputs, gpu_submissions, gpu_completions
    );
    println!(
        "  stage totals: decode={:.3} ms transfer={:.3} ms gpu_submit_loop={:.3} ms gpu_completion={:.3} ms bytes_copied/frame={}",
        duration_ms(decode_elapsed),
        duration_ms(transfer_elapsed),
        duration_ms(gpu_submit_elapsed),
        duration_ms(gpu_completion_elapsed),
        transfer_bytes / gpu_completions.max(1)
    );
    println!(
        "  queue peaks: compressed={} decoded={} gpu={}",
        compressed_queue.stats().peak_depth,
        decoded_queue.stats().peak_depth,
        gpu_queue.stats().peak_depth
    );
    print_nv12_gpu_diagnostics(&gpu_processor.diagnostics());
    Ok(())
}

fn run_step17_synthetic_nv12_gpu_observation(
    discovery: &VulkanDeviceDiscovery,
    device: &DeviceDesc,
    coded_width: u32,
    coded_height: u32,
    visible_width: u32,
    visible_height: u32,
    frame_count: usize,
    source_duration: Duration,
) -> Result<(), Box<dyn std::error::Error>> {
    let y_len = usize::try_from(coded_width)?
        .checked_mul(usize::try_from(coded_height)?)
        .ok_or("synthetic NV12 Y size overflow")?;
    let uv_len = usize::try_from(coded_width)?
        .checked_mul(usize::try_from(coded_height / 2)?)
        .ok_or("synthetic NV12 UV size overflow")?;
    let y = vec![96_u8; y_len];
    let uv = vec![128_u8; uv_len];
    let upload = Nv12Upload {
        device_id: device.id,
        coded_width,
        coded_height,
        visible_width,
        visible_height,
        y: Nv12Plane {
            width_bytes: coded_width,
            height: coded_height,
            stride_bytes: usize::try_from(coded_width)?,
            data: &y,
        },
        uv: Nv12Plane {
            width_bytes: coded_width,
            height: coded_height / 2,
            stride_bytes: usize::try_from(coded_width)?,
            data: &uv,
        },
        conversion: YcbcrConversion::Rec709Limited,
    };
    let mut processor = Nv12FrameProcessor::new(
        discovery,
        Nv12FrameProcessorConfig {
            device_id: device.id,
            coded_width,
            coded_height,
            visible_width,
            visible_height,
            slot_count: 3,
            conversion: YcbcrConversion::Rec709Limited,
            validation_readback: false,
        },
    )?;
    let mut pending = VecDeque::new();
    let mut submit_times = Vec::new();
    let mut wait_times = Vec::new();
    let started = Instant::now();
    for index in 0..frame_count {
        loop {
            let submit_started = Instant::now();
            match processor.submit_frame(
                &upload,
                FrameIdentity {
                    presentation_position: u64::try_from(index)?,
                },
            ) {
                Ok(token) => {
                    submit_times.push(submit_started.elapsed());
                    pending.push_back(token);
                    break;
                }
                Err(FrameProcessorError::NoFrameSlotAvailable) => {
                    let token = pending
                        .pop_front()
                        .ok_or("synthetic GPU pending queue empty")?;
                    let wait_started = Instant::now();
                    processor.wait_for_completion(token)?;
                    processor.retire_completed_token(token)?;
                    wait_times.push(wait_started.elapsed());
                }
                Err(err) => return Err(Box::new(err)),
            }
        }
    }
    while let Some(token) = pending.pop_front() {
        let wait_started = Instant::now();
        processor.wait_for_completion(token)?;
        processor.retire_completed_token(token)?;
        wait_times.push(wait_started.elapsed());
    }
    let elapsed = started.elapsed();
    println!("Step 17 isolated synthetic CPU NV12 -> Vulkan processing:");
    println!(
        "  device: {} frames={} wall={:.3} ms source={:.3}s throughput={:.2} fps realtime={:.2}x",
        device.name,
        frame_count,
        duration_ms(elapsed),
        source_duration.as_secs_f64(),
        frame_count as f64 / elapsed.as_secs_f64().max(0.000_001),
        source_duration.as_secs_f64() / elapsed.as_secs_f64().max(0.000_001)
    );
    print_duration_distribution("  isolated Vulkan submit/frame", &mut submit_times);
    print_duration_distribution("  isolated Vulkan wait/retire", &mut wait_times);
    print_nv12_gpu_diagnostics(&processor.diagnostics());
    Ok(())
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn duration_from_ns(ns: u128) -> Duration {
    Duration::from_nanos(u64::try_from(ns).unwrap_or(u64::MAX))
}

fn mean_duration(values: &[Duration]) -> Duration {
    if values.is_empty() {
        return Duration::ZERO;
    }
    let total_ns = values
        .iter()
        .map(Duration::as_nanos)
        .fold(0_u128, |acc, value| acc.saturating_add(value));
    let mean_ns = total_ns / values.len() as u128;
    Duration::from_nanos(u64::try_from(mean_ns).unwrap_or(u64::MAX))
}

fn median_duration(values: &mut [Duration]) -> Duration {
    if values.is_empty() {
        return Duration::ZERO;
    }
    values.sort_unstable();
    values[values.len() / 2]
}

fn print_duration_distribution(label: &str, values: &mut [Duration]) {
    if values.is_empty() {
        println!("{label}: no samples");
        return;
    }
    let total = values
        .iter()
        .copied()
        .fold(Duration::ZERO, |acc, value| acc.saturating_add(value));
    let avg = Duration::from_nanos(
        u64::try_from(total.as_nanos() / values.len() as u128).unwrap_or(u64::MAX),
    );
    values.sort_unstable();
    let median = values[values.len() / 2];
    let p90 = values[((values.len() * 9) / 10).min(values.len() - 1)];
    let max = *values.last().expect("nonempty");
    println!(
        "{label}: total={:.3} ms avg={:.3} ms median={:.3} ms p90={:.3} ms max={:.3} ms samples={}",
        duration_ms(total),
        duration_ms(avg),
        duration_ms(median),
        duration_ms(p90),
        duration_ms(max),
        values.len()
    );
}

fn print_nv12_gpu_diagnostics(diagnostics: &Nv12FrameProcessorDiagnostics) {
    println!("  NV12 Vulkan processor diagnostics:");
    println!(
        "    calls: submit={} poll={} fence_wait={} retire={} readback_enabled={}",
        diagnostics.submit_calls,
        diagnostics.poll_calls,
        diagnostics.fence_wait_calls,
        diagnostics.retire_calls,
        yes_no(diagnostics.output_readback_enabled)
    );
    println!(
        "    CPU prep: slot_acquire={:.3} ms compact_y_copy={:.3} ms compact_uv_copy={:.3} ms staging_write_y={:.3} ms staging_write_uv={:.3} ms",
        ns_ms(diagnostics.slot_acquire_ns),
        ns_ms(diagnostics.compact_y_copy_ns),
        ns_ms(diagnostics.compact_uv_copy_ns),
        ns_ms(diagnostics.staging_write_y_ns),
        ns_ms(diagnostics.staging_write_uv_ns)
    );
    println!(
        "    command/submit: pool_reset={:.3} ms fence_reset={:.3} ms begin={:.3} ms record={:.3} ms end={:.3} ms queue_submit={:.3} ms",
        ns_ms(diagnostics.command_pool_reset_ns),
        ns_ms(diagnostics.fence_reset_ns),
        ns_ms(diagnostics.command_begin_ns),
        ns_ms(diagnostics.command_record_ns),
        ns_ms(diagnostics.command_end_ns),
        ns_ms(diagnostics.queue_submit_ns)
    );
    println!(
        "    completion: poll={:.3} ms fence_wait={:.3} ms retire={:.3} ms",
        ns_ms(diagnostics.poll_ns),
        ns_ms(diagnostics.fence_wait_ns),
        ns_ms(diagnostics.retire_ns)
    );
}

fn ns_ms(ns: u128) -> f64 {
    ns as f64 / 1_000_000.0
}

struct FrontendOnlyObservation {
    frames: usize,
    peak_dpb_occupancy: usize,
    peak_output_pending: usize,
}

fn run_proxy_frontend_only(
    video: &qgs_mp4::Mp4VideoTrack,
) -> Result<FrontendOnlyObservation, Box<dyn std::error::Error>> {
    let mut state = qgs_codec_h264::H264DecoderState::new();
    let mut frames = 0_usize;
    for sample in &video.samples {
        let parsed = state.parse_access_unit(&sample.annex_b)?;
        let update = state.finish_picture(&parsed)?;
        frames = frames.saturating_add(update.output_ready.len());
    }
    let update = state.flush();
    frames = frames.saturating_add(update.output_ready.len());
    Ok(FrontendOnlyObservation {
        frames,
        peak_dpb_occupancy: state.max_dpb_occupancy(),
        peak_output_pending: state.max_output_pending(),
    })
}

fn selected_proxy_ordinals(frame_count: usize) -> Vec<usize> {
    if frame_count == 0 {
        return Vec::new();
    }
    let mut selected = vec![0, 53.min(frame_count - 1), frame_count - 1];
    selected.sort_unstable();
    selected.dedup();
    selected
}

fn print_decode_timing(
    label: &str,
    frames: usize,
    source_seconds: f64,
    elapsed: std::time::Duration,
) {
    let seconds = elapsed.as_secs_f64().max(0.000_001);
    let fps = frames as f64 / seconds;
    let realtime = if source_seconds > 0.0 {
        source_seconds / seconds
    } else {
        0.0
    };
    println!("{label}:");
    println!("  frames: {frames}");
    println!(
        "  DEVELOPMENT OBSERVATION - NOT A BENCHMARK: {:.3}s, {:.2} fps, {:.2}x realtime",
        seconds, fps, realtime
    );
}

fn print_va_observation(observation: &qgs_vaapi::VaapiDecodeObservation) {
    println!(
        "  pool: allocated={} reused={} recycled={} recycle_sync={} deferred={} peak_checked_out={} peak_pending_recycle={} min_free={}",
        observation.pool_stats.surfaces_allocated,
        observation.pool_stats.surface_reuse_count,
        observation.pool_stats.surfaces_recycled,
        observation.pool_stats.recycle_sync_count,
        observation.pool_stats.deferred_recycle_count,
        observation.pool_stats.peak_checked_out_surfaces,
        observation.pool_stats.peak_pending_recycle_surfaces,
        observation.pool_stats.minimum_free_surfaces
    );
    println!(
        "  peaks: DPB={} output_pending={} live_va_surfaces={} submitted_unsynced={} client_held_outputs={}",
        observation.peak_dpb_occupancy,
        observation.peak_output_pending,
        observation.peak_live_surfaces,
        observation.peak_submitted_unsynced_surfaces,
        observation.max_client_held_outputs
    );
    println!(
        "  diagnostics: frames={} export_probes={}",
        observation.diagnostic_frames, observation.diagnostic_export_probes
    );
}

fn print_va_timing(timing: &qgs_vaapi::VaapiDecodeTiming) {
    let total = timing.submit_total_ns.max(1);
    println!("  aggregate VA/backend timing:");
    print_timing_bucket("surface acquisition", timing.surface_acquire_ns, total);
    print_timing_bucket("H.264 frontend parse", timing.h264_parse_ns, total);
    print_timing_bucket("parameter/list build", timing.parameter_build_ns, total);
    print_timing_bucket("VA buffer creation", timing.buffer_create_ns, total);
    print_timing_bucket("vaBeginPicture", timing.begin_picture_ns, total);
    print_timing_bucket("vaRenderPicture", timing.render_picture_ns, total);
    print_timing_bucket("vaEndPicture", timing.end_picture_ns, total);
    print_timing_bucket("inline surface sync", timing.inline_sync_ns, total);
    print_timing_bucket("reclaim surface sync", timing.reclaim_sync_ns, total);
    print_timing_bucket("diagnostics", timing.diagnostics_ns, total);
    print_timing_bucket("finish picture", timing.finish_picture_ns, total);
    print_timing_bucket("output mapping", timing.output_mapping_ns, total);
    print_timing_bucket("release/drop", timing.release_ns, total);
    print_timing_bucket("flush", timing.flush_ns, total);
    print_timing_bucket("submit total", timing.submit_total_ns, total);
}

fn print_timing_bucket(label: &str, ns: u128, total_ns: u128) {
    let ms = ns as f64 / 1_000_000.0;
    let pct = (ns as f64 / total_ns as f64) * 100.0;
    println!("    {label}: {ms:.3} ms ({pct:.1}%)");
}

fn run_software_gpu_sequence_proof(
    discovery: &VulkanDeviceDiscovery,
    device: &DeviceDesc,
    label: &str,
    frames: &[(&str, &qgs_software_video::SoftwareVideoSurface)],
) -> Result<BTreeMap<String, u64>, Box<dyn std::error::Error>> {
    if frames.is_empty() {
        return Ok(BTreeMap::new());
    }
    let first_upload = yuv422p10_upload_for_frame(device.id, frames[0].1)?;
    let mut processor = GpuFrameProcessor::new(
        discovery,
        GpuFrameProcessorConfig {
            device_id: device.id,
            width: first_upload.width,
            height: first_upload.height,
            slot_count: 3,
            conversion: first_upload.conversion,
        },
    )?;
    let started = Instant::now();
    let mut pending = VecDeque::new();
    let mut checksums = BTreeMap::new();
    for (name, frame) in frames {
        let upload = yuv422p10_upload_for_frame(device.id, frame)?;
        let reference_start = Instant::now();
        let reference = yuv422p10_reference_rgba_u16(&upload)?;
        let reference_elapsed = reference_start.elapsed();
        let reference_checksum = yuv422p10_rgba_u16_checksum(&reference);
        let identity = FrameIdentity {
            presentation_position: frame.presentation_index,
        };
        match processor.submit_frame(&upload, identity) {
            Ok(token) => pending.push_back((
                token,
                (*name).to_string(),
                frame.presentation_index,
                reference,
                reference_checksum,
                reference_elapsed,
            )),
            Err(FrameProcessorError::NoFrameSlotAvailable) => {
                while let Some((
                    token,
                    pending_name,
                    presentation,
                    reference,
                    reference_checksum,
                    ref_elapsed,
                )) = pending.pop_front()
                {
                    let output = processor.wait_for_frame(token)?;
                    validate_software_gpu_output(
                        &output,
                        &pending_name,
                        presentation,
                        &reference,
                        reference_checksum,
                        ref_elapsed,
                        &mut checksums,
                    )?;
                }
                let token = processor.submit_frame(&upload, identity)?;
                pending.push_back((
                    token,
                    (*name).to_string(),
                    frame.presentation_index,
                    reference,
                    reference_checksum,
                    reference_elapsed,
                ));
            }
            Err(err) => return Err(Box::new(err)),
        }
    }
    while let Some((token, name, presentation, reference, reference_checksum, ref_elapsed)) =
        pending.pop_front()
    {
        let output = processor.wait_for_frame(token)?;
        validate_software_gpu_output(
            &output,
            &name,
            presentation,
            &reference,
            reference_checksum,
            ref_elapsed,
            &mut checksums,
        )?;
    }
    let elapsed = started.elapsed();
    let counters = processor.counters();
    println!(
        "  {label}: frames={} submissions={} slot_reuses={} pipelines={} command_buffers={} elapsed={:.3}ms",
        frames.len(),
        counters.frame_submissions,
        counters.slot_reuses,
        counters.pipeline_creations,
        counters.command_buffer_count,
        elapsed.as_secs_f64() * 1000.0
    );
    Ok(checksums)
}

fn validate_software_gpu_output(
    output: &qgs_vulkan::ProcessedFrameOutput,
    label: &str,
    presentation: u64,
    reference: &[u16],
    reference_checksum: u64,
    reference_elapsed: std::time::Duration,
    checksums: &mut BTreeMap<String, u64>,
) -> Result<(), Box<dyn std::error::Error>> {
    if output.presentation_position != presentation {
        return Err(format!("{label} GPU presentation identity mismatch").into());
    }
    let max_delta = max_u16_delta(reference, &output.rgba_u16)?;
    if max_delta > 1 {
        return Err(format!("{label} GPU output exceeded tolerance: max delta {max_delta}").into());
    }
    println!(
        "  {label}: presentation={} slot={} seq={} gpu_checksum=0x{:016x} cpu_ref=0x{:016x} max_delta={} ref_time={:.3}ms",
        presentation,
        output.slot_index,
        output.submission_sequence,
        output.checksum,
        reference_checksum,
        max_delta,
        reference_elapsed.as_secs_f64() * 1000.0
    );
    println!(
        "    memory: cpu_surface={} staging={} gpu_planes={} output={}",
        output.cpu_surface_bytes, output.staging_bytes, output.gpu_plane_bytes, output.output_bytes
    );
    checksums.insert(label.to_string(), output.checksum);
    Ok(())
}

fn yuv422p10_upload_for_frame<'a>(
    device_id: qgs_protocol::DeviceId,
    frame: &'a qgs_software_video::SoftwareVideoSurface,
) -> Result<Yuv422P10Upload<'a>, Box<dyn std::error::Error>> {
    if frame.storage_format != qgs_software_video::SoftwarePixelFormat::Yuv422P10Le
        || frame.planes.len() != 3
    {
        return Err("expected YUV422P10LE software surface".into());
    }
    let plane = |index: usize| -> Yuv422P10Plane<'a> {
        let plane = &frame.planes[index];
        Yuv422P10Plane {
            width_samples: plane.width_samples,
            height: plane.height,
            stride_bytes: plane.stride_bytes,
            data: &plane.data,
        }
    };
    Ok(Yuv422P10Upload {
        device_id,
        width: frame.desc.coded_width,
        height: frame.desc.coded_height,
        y: plane(0),
        cb: plane(1),
        cr: plane(2),
        conversion: YcbcrConversion::Rec709Limited,
    })
}

fn max_u16_delta(left: &[u16], right: &[u16]) -> Result<u16, Box<dyn std::error::Error>> {
    if left.len() != right.len() {
        return Err("CPU/GPU output lengths differ".into());
    }
    Ok(left
        .iter()
        .zip(right)
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap_or(0))
}

fn native_original_video_decode_mxf(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let source = MediaSource::parse(&bytes)?;
    let first_access_unit = source.extract_video_access_unit(&bytes, 0)?;
    let parsed = qgs_codec_h264::parse_annex_b_access_unit(&first_access_unit)?;
    let profile = H264422P10Profile {
        profile: parsed.profile,
        bit_depth: parsed.desc.bit_depth.get(),
        chroma: parsed.desc.chroma,
        coded_width: parsed.desc.coded_width,
        coded_height: parsed.desc.coded_height,
        visible_width: parsed.desc.visible_region.width,
        visible_height: parsed.desc.visible_region.height,
        scan_mode: parsed.desc.scan_mode,
    };
    let sony_shape = profile == H264422P10Profile::sony_fx6_original();
    let bounded_access_units = source.index.video.len().min(12);
    let mut decoder = H264422P10Decoder::new(profile.clone());
    let mut report_parser_state = qgs_codec_h264::H264DecoderState::new();
    let mut output_frames = 0_usize;
    let mut output_bytes = 0_usize;
    let mut submitted_access_units = 0_usize;

    println!("QGS native original MXF H.264 4:2:2 10-bit decode");
    println!(
        "input: {}",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("<original-mxf>")
    );
    println!("source: original MXF video");
    println!("proxy video used: no");
    println!("proxy AAC used: no");
    println!("ffmpeg/rsmpeg/libav: not used");
    println!(
        "profile: {:?} {}bit {:?} coded={}x{} visible={}x{}",
        profile.profile,
        profile.bit_depth,
        profile.chroma,
        profile.coded_width,
        profile.coded_height,
        profile.visible_width,
        profile.visible_height
    );
    println!("sony-fx6 original shape: {}", yes_no(sony_shape));
    println!(
        "pps transform_8x8_mode: {}",
        yes_no(parsed.picture.transform_8x8_mode_flag)
    );
    println!(
        "pps constrained_intra_pred: {}",
        yes_no(parsed.picture.constrained_intra_pred_flag)
    );
    println!(
        "pps weighted_pred: {} weighted_bipred_idc={}",
        yes_no(parsed.picture.weighted_pred_flag),
        parsed.picture.weighted_bipred_idc
    );
    println!("mxf video access units: {}", source.index.video.len());
    println!("bounded submitted access units: {bounded_access_units}");

    for index in 0..bounded_access_units {
        let access_unit = source.extract_video_access_unit(&bytes, index)?;
        let report = report_parser_state.parse_access_unit(&access_unit)?;
        println!(
            "submit AU {index:02}: slices={} kind={:?} first_mb={} cabac_init={} bytes={}",
            report.slices.len(),
            report.slices[0].kind,
            report.slices[0].first_mb_in_slice,
            report.slices[0].cabac_init_idc,
            access_unit.len()
        );
        if report.slices.len() > 1 {
            let slice_starts = report
                .slices
                .iter()
                .map(|slice| slice.first_mb_in_slice.to_string())
                .collect::<Vec<_>>()
                .join(",");
            println!("  slice first_mb list: {slice_starts}");
            let ref_counts = report
                .slices
                .iter()
                .map(|slice| {
                    format!(
                        "{}:L0={} L1={}",
                        slice.first_mb_in_slice,
                        slice.num_ref_idx_l0_active_minus1.saturating_add(1),
                        slice.num_ref_idx_l1_active_minus1.saturating_add(1)
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            println!("  active ref counts: {ref_counts}");
            let slice_offsets = report
                .slices
                .iter()
                .map(|slice| {
                    let alignment =
                        qgs_h264_422p10::slice::slice_payload_from_parsed(&report, slice)
                            .map(|payload| {
                                format!(
                                    "align={} all_ones={}",
                                    payload.cabac_alignment_bits,
                                    yes_no(payload.cabac_alignment_all_ones)
                                )
                            })
                            .unwrap_or_else(|error| format!("align_error={error}"));
                    format!(
                        "{}@{}b(mod8={};{})",
                        slice.first_mb_in_slice,
                        slice.slice_data_bit_offset,
                        slice.slice_data_bit_offset % 8,
                        alignment
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            println!("  slice data offsets: {slice_offsets}");
        }
        match decoder.submit_access_unit(&access_unit) {
            Ok(outputs) => {
                submitted_access_units += 1;
                println!("  output frames={}", outputs.len());
                for output in outputs {
                    output_frames = output_frames.saturating_add(1);
                    output_bytes =
                        output_bytes.saturating_add(print_native_original_video_frame(output)?);
                }
            }
            Err(error) => {
                println!("  decoder boundary: {error}");
                if let H264422P10PictureDecodeError::IncompletePicture { slice_reports, .. } =
                    &error
                {
                    let slice_summary = slice_reports
                        .iter()
                        .map(|slice| {
                            let last_mb = slice
                                .last_macroblock
                                .map(|value| value.to_string())
                                .unwrap_or_else(|| "none".to_string());
                            let cbp = slice
                                .last_coded_block_pattern
                                .map(|value| format!("0x{value:02x}"))
                                .unwrap_or_else(|| "none".to_string());
                            let cabac = match (slice.cabac_bit_position, slice.cabac_payload_bits) {
                                (Some(position), Some(total)) => {
                                    format!("cabac_bits={position}/{total}")
                                }
                                _ => "cabac_bits=unknown".to_string(),
                            };
                            format!(
                                "{}:{}/{}:{}:last_mb={}:type={}:cbp={}:{}",
                                slice.first_macroblock,
                                slice.reconstructed_macroblocks,
                                slice.expected_macroblocks,
                                slice.stop_reason,
                                last_mb,
                                slice.last_macroblock_type,
                                cbp,
                                cabac
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(",");
                    println!("  slice decode report first_mb:count = {slice_summary}");
                }
                let flushed = decoder.flush()?;
                if !flushed.is_empty() {
                    println!("  boundary flush decoded original frames={}", flushed.len());
                }
                for output in flushed {
                    output_frames = output_frames.saturating_add(1);
                    output_bytes =
                        output_bytes.saturating_add(print_native_original_video_frame(output)?);
                }
                println!("submitted access units before boundary: {submitted_access_units}");
                println!("decoded frames before boundary: {output_frames}");
                println!("decoded owned yuv422p10le bytes before boundary: {output_bytes}");
                return Err(Box::new(error));
            }
        }
    }

    for output in decoder.flush()? {
        output_frames = output_frames.saturating_add(1);
        output_bytes = output_bytes.saturating_add(print_native_original_video_frame(output)?);
    }

    println!("decoded frames: {output_frames}");
    println!("decoded owned yuv422p10le bytes: {output_bytes}");
    println!("real display: no");
    println!("visual verified: no");
    println!("realtime verified: no");
    Ok(())
}

fn print_native_original_video_frame(
    output: H264422P10Frame,
) -> Result<usize, Box<dyn std::error::Error>> {
    output.validate_layout()?;
    let bytes = output.owned_bytes();
    println!(
        "    frame {:02}: y={} cb={} cr={} bytes={}",
        output.presentation_index,
        output.y.data.len(),
        output.cb.data.len(),
        output.cr.data.len(),
        bytes
    );
    Ok(bytes)
}

fn decode_positioned_access_units_with_context(
    config: DecoderConfig,
    access_units: &[(u64, Vec<u8>)],
) -> Result<DecodeRunStats, Box<dyn std::error::Error>> {
    let mut decoder = SoftwareH264Decoder::new(config)?;
    let mut frames = Vec::new();
    for (position, access_unit) in access_units {
        let decoded = decoder
            .decode_access_unit_at(access_unit, *position)
            .map_err(|err| format!("software decode failed at edit unit {position}: {err}"))?;
        frames.extend(decoded);
    }
    frames.extend(decoder.flush_surfaces()?);
    Ok(DecodeRunStats {
        frames,
        max_live_surfaces: decoder.max_live_surfaces(),
        elapsed: decoder.elapsed(),
    })
}

fn print_h264_gop_summary(
    source: &MediaSource,
    bytes: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut state = qgs_codec_h264::H264DecoderState::new();
    let mut kinds = BTreeMap::new();
    let mut random_access_positions = Vec::new();
    let targets = [
        0_usize,
        source.index.video.len() / 2,
        source.index.video.len().saturating_sub(1),
    ];
    let mut target_summaries = Vec::new();
    for index in 0..source.index.video.len() {
        let access_unit = source.extract_video_access_unit(bytes, index)?;
        let parsed = state.parse_access_unit(&access_unit)?;
        if parsed.slices.iter().any(|slice| slice.idr) {
            random_access_positions.push(index);
        }
        if let Some(kind) = parsed
            .slices
            .first()
            .map(|slice| format!("{:?}", slice.kind))
        {
            *kinds.entry(kind.clone()).or_insert(0_usize) += 1;
            if targets.contains(&index) {
                target_summaries.push((
                    index,
                    kind,
                    parsed.desc.bit_depth.get(),
                    parsed.desc.chroma,
                ));
            }
        }
        state.finish_picture(&parsed)?;
    }
    println!("GOP structure:");
    println!("  picture counts: {:?}", kinds);
    println!("  IDR positions: {:?}", random_access_positions);
    for (index, kind, bit_depth, chroma) in target_summaries {
        println!("  sample edit unit {index}: kind={kind} bit_depth={bit_depth} chroma={chroma:?}");
    }
    Ok(())
}

#[derive(Debug)]
struct SonyXmlSummary {
    duration: Option<u64>,
    ltc_tc_fps: Option<String>,
    ltc_half_step: Option<bool>,
    video_codec: Option<String>,
    capture_fps: Option<String>,
    format_fps: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    aspect_ratio: Option<String>,
    audio_channels: Option<u16>,
    audio_codecs: Vec<String>,
    color_values: BTreeMap<String, String>,
    camera_model: Option<String>,
    has_proxy_metadata: bool,
}

impl SonyXmlSummary {
    fn from_mxf_path(path: &Path) -> Result<Option<Self>, Box<dyn std::error::Error>> {
        let Some(stem) = path.file_stem().and_then(|value| value.to_str()) else {
            return Ok(None);
        };
        let sidecar = path.with_file_name(format!("{stem}M01.XML"));
        if !sidecar.exists() {
            return Ok(None);
        }
        let xml = std::fs::read_to_string(sidecar)?;
        Ok(Some(Self::parse(&xml)))
    }

    fn parse(xml: &str) -> Self {
        let duration = find_attr(xml, "<Duration", "value").and_then(|value| value.parse().ok());
        let ltc_tc_fps = find_attr(xml, "<LtcChangeTable", "tcFps");
        let ltc_half_step =
            find_attr(xml, "<LtcChangeTable", "halfStep").map(|value| value == "true");
        let video_codec = find_attr(xml, "<VideoFrame", "videoCodec");
        let capture_fps = find_attr(xml, "<VideoFrame", "captureFps");
        let format_fps = find_attr(xml, "<VideoFrame", "formatFps");
        let width = find_attr(xml, "<VideoLayout", "pixel").and_then(|value| value.parse().ok());
        let height = find_attr(xml, "<VideoLayout", "numOfVerticalLine")
            .and_then(|value| value.parse().ok());
        let aspect_ratio = find_attr(xml, "<VideoLayout", "aspectRatio");
        let audio_channels =
            find_attr(xml, "<AudioFormat", "numOfChannel").and_then(|value| value.parse().ok());
        let audio_codecs = find_all_attrs(xml, "<AudioRecPort", "audioCodec");
        let camera_model = find_attr(xml, "<Device", "modelName");
        let has_proxy_metadata = xml.contains("SubStream")
            || xml.contains("Proxy")
            || xml.contains("proxy")
            || xml.contains("substream");
        let mut color_values = BTreeMap::new();
        for name in [
            "CaptureGammaEquation",
            "CaptureColorPrimaries",
            "CodingEquations",
        ] {
            if let Some(value) = find_item_value(xml, name) {
                color_values.insert(name.to_string(), value);
            }
        }
        Self {
            duration,
            ltc_tc_fps,
            ltc_half_step,
            video_codec,
            capture_fps,
            format_fps,
            width,
            height,
            aspect_ratio,
            audio_channels,
            audio_codecs,
            color_values,
            camera_model,
            has_proxy_metadata,
        }
    }
}

fn sha256_hex(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    Ok(format!("{:x}", Sha256::digest(&bytes)))
}

fn proxy_presentation_ordinals(
    video: &qgs_mp4::Mp4VideoTrack,
) -> Result<BTreeMap<u32, u64>, Box<dyn std::error::Error>> {
    let mut ordered = video
        .samples
        .iter()
        .map(|sample| (sample.pts, sample.sample_index))
        .collect::<Vec<_>>();
    ordered.sort_by_key(|(pts, sample_index)| (*pts, *sample_index));
    let mut ordinals = BTreeMap::new();
    for (ordinal, (_, sample_index)) in ordered.into_iter().enumerate() {
        ordinals.insert(sample_index, u64::try_from(ordinal)?);
    }
    Ok(ordinals)
}

struct LumaSignatureComparison {
    original_hash: u64,
    proxy_hash: u64,
    mean_abs_delta: f64,
}

fn luma_signature_delta(
    original: &qgs_software_video::SoftwareVideoSurface,
    proxy: &qgs_software_video::SoftwareVideoSurface,
) -> Result<LumaSignatureComparison, Box<dyn std::error::Error>> {
    let original_signature = luma_signature(original)?;
    let proxy_signature = luma_signature(proxy)?;
    if original_signature.len() != proxy_signature.len() {
        return Err("luma signature lengths differ".into());
    }
    let total_delta = original_signature
        .iter()
        .zip(&proxy_signature)
        .map(|(a, b)| u64::from(a.abs_diff(*b)))
        .sum::<u64>();
    let mean_abs_delta = total_delta as f64 / original_signature.len().max(1) as f64;
    Ok(LumaSignatureComparison {
        original_hash: bytes_checksum(&original_signature),
        proxy_hash: bytes_checksum(&proxy_signature),
        mean_abs_delta,
    })
}

fn luma_signature(
    frame: &qgs_software_video::SoftwareVideoSurface,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let y = frame.planes.first().ok_or("missing luma plane")?;
    let grid_w = 16_u32.min(frame.desc.coded_width);
    let grid_h = 9_u32.min(frame.desc.coded_height);
    let mut values = Vec::new();
    for gy in 0..grid_h {
        let y_pos = gy
            .checked_mul(frame.desc.coded_height.saturating_sub(1))
            .ok_or("luma signature y overflow")?
            / grid_h.saturating_sub(1).max(1);
        for gx in 0..grid_w {
            let x_pos = gx
                .checked_mul(frame.desc.coded_width.saturating_sub(1))
                .ok_or("luma signature x overflow")?
                / grid_w.saturating_sub(1).max(1);
            let sample = match frame.storage_format {
                SoftwarePixelFormat::Yuv420P8 => {
                    let offset = usize::try_from(y_pos)?
                        .checked_mul(y.stride_bytes)
                        .and_then(|row| row.checked_add(usize::try_from(x_pos).ok()?))
                        .ok_or("luma signature offset overflow")?;
                    *y.data.get(offset).ok_or("luma signature out of bounds")?
                }
                SoftwarePixelFormat::Yuv422P10Le => {
                    let offset = usize::try_from(y_pos)?
                        .checked_mul(y.stride_bytes)
                        .and_then(|row| {
                            row.checked_add(usize::try_from(x_pos).ok()?.checked_mul(2)?)
                        })
                        .ok_or("luma signature offset overflow")?;
                    let bytes = y
                        .data
                        .get(offset..offset + 2)
                        .ok_or("luma signature out of bounds")?;
                    (u16::from_le_bytes([bytes[0], bytes[1]]) >> 2) as u8
                }
            };
            values.push(sample);
        }
    }
    Ok(values)
}

fn bytes_checksum(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for value in bytes {
        hash ^= u64::from(*value);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn run_proxy_hardware_decode(
    socket_path: &Path,
    config: &DecoderConfig,
    video: &qgs_mp4::Mp4VideoTrack,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut stream = connect_socket(socket_path)?;
    let mut request_id = 1_u64;
    send_message(
        &mut stream,
        &WireMessage::Hello {
            request_id,
            request: HelloRequest::current(),
        },
    )?;
    let response = receive_message(&mut stream)?;
    let WireMessage::Welcome { .. } = response else {
        return Err("expected WELCOME for proxy hardware proof".into());
    };
    request_id += 1;
    send_message(&mut stream, &WireMessage::EnumerateDevices { request_id })?;
    let response = receive_message(&mut stream)?;
    let WireMessage::DeviceList { response, .. } = response else {
        return Err("expected DEVICE_LIST for proxy hardware proof".into());
    };
    let Some(device) = response
        .devices
        .iter()
        .find(|device| {
            device.vendor_id == 0x8086 && matches!(device.class, DeviceClass::IntegratedGpu)
        })
        .or_else(|| {
            response
                .devices
                .iter()
                .find(|device| matches!(device.class, DeviceClass::IntegratedGpu))
        })
    else {
        println!("Intel proxy hardware decode: no integrated GPU advertised");
        return Ok(());
    };
    let mut intel_config = config.clone();
    intel_config.device_id = device.id;
    let supports_proxy = query_h264_decode_support(
        &mut stream,
        &mut request_id,
        device,
        H264Profile::High,
        ChromaSubsampling::Cs420,
        8,
    )?;
    println!("Intel proxy capability:");
    println!("  device: {}", device.name);
    println!(
        "  H.264 High 8-bit 4:2:0 support: {}",
        yes_no(supports_proxy)
    );
    if !supports_proxy {
        return Ok(());
    }

    request_id += 1;
    send_message(
        &mut stream,
        &WireMessage::CreateDecoder {
            request_id,
            request: CreateDecoderRequest {
                config: intel_config,
            },
        },
    )?;
    let response = receive_message(&mut stream)?;
    let decoder_id = match response {
        WireMessage::DecoderCreated { response, .. } => response.decoder_id,
        WireMessage::Error { response, .. } => {
            return Err(format!(
                "proxy hardware decoder creation failed: {:?}",
                response.code
            )
            .into());
        }
        _ => return Err("expected proxy DECODER_CREATED".into()),
    };
    let started = Instant::now();
    let mut output_count = 0_usize;
    let mut parser_state = qgs_codec_h264::H264DecoderState::new();
    for sample in &video.samples {
        let parsed_for_diagnostic = parser_state.parse_access_unit(&sample.annex_b)?;
        let diagnostic = format!(
            "kind={:?} idr={} frame_num={} poc={} reference={} refs={} l0={} l1={}",
            parsed_for_diagnostic
                .slices
                .first()
                .map(|slice| &slice.kind),
            parsed_for_diagnostic.picture.idr_pic_flag,
            parsed_for_diagnostic.picture.frame_num,
            parsed_for_diagnostic.picture.top_field_order_cnt,
            parsed_for_diagnostic.picture.reference_pic_flag,
            parsed_for_diagnostic.reference_frames.len(),
            parsed_for_diagnostic
                .slices
                .first()
                .map(|slice| slice.ref_pic_list0.len())
                .unwrap_or(0),
            parsed_for_diagnostic
                .slices
                .first()
                .map(|slice| slice.ref_pic_list1.len())
                .unwrap_or(0)
        );
        parser_state.finish_picture(&parsed_for_diagnostic)?;
        request_id += 1;
        send_message(
            &mut stream,
            &WireMessage::SubmitAccessUnit {
                request_id,
                request: SubmitAccessUnitRequest {
                    decoder_id,
                    data: sample.annex_b.clone(),
                },
            },
        )?;
        let response = receive_message(&mut stream)?;
        let response = match response {
            WireMessage::DecodeOutput { response, .. } => response,
            WireMessage::Error { response, .. } => {
                return Err(format!(
                    "proxy hardware decode failed at sample {}: {:?} ({diagnostic})",
                    sample.sample_index, response.code
                )
                .into());
            }
            other => {
                return Err(format!(
                    "expected proxy DECODE_OUTPUT at sample {}, got {:?}",
                    sample.sample_index, other
                )
                .into());
            }
        };
        output_count += response.outputs.len();
        for output in response.outputs {
            request_id += 1;
            destroy_resource(&mut stream, request_id, output.resource_id)?;
        }
    }
    request_id += 1;
    send_message(
        &mut stream,
        &WireMessage::FlushDecoder {
            request_id,
            request: FlushDecoderRequest { decoder_id },
        },
    )?;
    let response = receive_message(&mut stream)?;
    let WireMessage::DecodeOutput { response, .. } = response else {
        return Err("expected proxy flush DECODE_OUTPUT".into());
    };
    output_count += response.outputs.len();
    for output in response.outputs {
        request_id += 1;
        destroy_resource(&mut stream, request_id, output.resource_id)?;
    }
    let elapsed = started.elapsed();
    if output_count != video.samples.len() {
        return Err(format!(
            "proxy hardware decode expected {} frames, got {output_count}",
            video.samples.len()
        )
        .into());
    }
    request_id += 1;
    send_message(
        &mut stream,
        &WireMessage::DestroyDecoder {
            request_id,
            request: DestroyDecoderRequest { decoder_id },
        },
    )?;
    let response = receive_message(&mut stream)?;
    let WireMessage::DecoderDestroyed { .. } = response else {
        return Err("expected proxy DECODER_DESTROYED".into());
    };
    println!("Intel proxy hardware decode:");
    println!("  decoded frames: {output_count}");
    println!(
        "  DEVELOPMENT OBSERVATION - NOT A BENCHMARK: {:.3}s, {:.2} fps, {:.2}x realtime @ 50fps",
        elapsed.as_secs_f64(),
        output_count as f64 / elapsed.as_secs_f64().max(0.000_001),
        (output_count as f64 / elapsed.as_secs_f64().max(0.000_001)) / 50.0
    );
    Ok(())
}

struct ProxyGpuUpload {
    y: Vec<u8>,
    cb: Vec<u8>,
    cr: Vec<u8>,
    width: u32,
    height: u32,
}

impl ProxyGpuUpload {
    fn upload(&self, device_id: qgs_protocol::DeviceId) -> Yuv422P10Upload<'_> {
        let y_stride = usize::try_from(self.width).expect("width") * 2;
        let c_stride = usize::try_from(self.width / 2).expect("width") * 2;
        Yuv422P10Upload {
            device_id,
            width: self.width,
            height: self.height,
            y: Yuv422P10Plane {
                width_samples: self.width,
                height: self.height,
                stride_bytes: y_stride,
                data: &self.y,
            },
            cb: Yuv422P10Plane {
                width_samples: self.width / 2,
                height: self.height,
                stride_bytes: c_stride,
                data: &self.cb,
            },
            cr: Yuv422P10Plane {
                width_samples: self.width / 2,
                height: self.height,
                stride_bytes: c_stride,
                data: &self.cr,
            },
            conversion: YcbcrConversion::Rec709Limited,
        }
    }
}

fn proxy_yuv420p8_to_yuv422p10(
    frame: &qgs_software_video::SoftwareVideoSurface,
) -> Result<ProxyGpuUpload, Box<dyn std::error::Error>> {
    if frame.storage_format != SoftwarePixelFormat::Yuv420P8 || frame.planes.len() != 3 {
        return Err("expected YUV420P8 proxy frame".into());
    }
    let width = frame.desc.coded_width;
    let height = frame.desc.coded_height;
    let y_stride = usize::try_from(width)?
        .checked_mul(2)
        .ok_or("stride overflow")?;
    let c_stride = usize::try_from(width / 2)?
        .checked_mul(2)
        .ok_or("stride overflow")?;
    let mut y = vec![
        0_u8;
        y_stride
            .checked_mul(usize::try_from(height)?)
            .ok_or("Y len overflow")?
    ];
    let mut cb = vec![
        0_u8;
        c_stride
            .checked_mul(usize::try_from(height)?)
            .ok_or("Cb len overflow")?
    ];
    let mut cr = vec![
        0_u8;
        c_stride
            .checked_mul(usize::try_from(height)?)
            .ok_or("Cr len overflow")?
    ];
    let src_y = &frame.planes[0];
    let src_cb = &frame.planes[1];
    let src_cr = &frame.planes[2];
    for row in 0..usize::try_from(height)? {
        for x in 0..usize::try_from(width)? {
            let src_offset = row
                .checked_mul(src_y.stride_bytes)
                .and_then(|base| base.checked_add(x))
                .ok_or("Y source offset overflow")?;
            let value =
                u16::from(*src_y.data.get(src_offset).ok_or("Y source out of bounds")?) << 2;
            y[row * y_stride + x * 2..row * y_stride + x * 2 + 2]
                .copy_from_slice(&value.to_le_bytes());
        }
        let chroma_row = row / 2;
        for x in 0..usize::try_from(width / 2)? {
            let src_offset = chroma_row
                .checked_mul(src_cb.stride_bytes)
                .and_then(|base| base.checked_add(x))
                .ok_or("C source offset overflow")?;
            let cb_value = u16::from(
                *src_cb
                    .data
                    .get(src_offset)
                    .ok_or("Cb source out of bounds")?,
            ) << 2;
            let cr_value = u16::from(
                *src_cr
                    .data
                    .get(src_offset)
                    .ok_or("Cr source out of bounds")?,
            ) << 2;
            cb[row * c_stride + x * 2..row * c_stride + x * 2 + 2]
                .copy_from_slice(&cb_value.to_le_bytes());
            cr[row * c_stride + x * 2..row * c_stride + x * 2 + 2]
                .copy_from_slice(&cr_value.to_le_bytes());
        }
    }
    Ok(ProxyGpuUpload {
        y,
        cb,
        cr,
        width,
        height,
    })
}

fn run_proxy_gpu_frame(
    discovery: &VulkanDeviceDiscovery,
    device: &DeviceDesc,
    label: &str,
    frame: &qgs_software_video::SoftwareVideoSurface,
) -> Result<(), Box<dyn std::error::Error>> {
    let converted = proxy_yuv420p8_to_yuv422p10(frame)?;
    let upload = converted.upload(device.id);
    let reference = yuv422p10_reference_rgba_u16(&upload)?;
    let reference_checksum = yuv422p10_rgba_u16_checksum(&reference);
    let mut processor = GpuFrameProcessor::new(
        discovery,
        GpuFrameProcessorConfig {
            device_id: device.id,
            width: upload.width,
            height: upload.height,
            slot_count: 3,
            conversion: upload.conversion,
        },
    )?;
    let token = processor.submit_frame(
        &upload,
        FrameIdentity {
            presentation_position: frame.presentation_index,
        },
    )?;
    let output = processor.wait_for_frame(token)?;
    let max_delta = max_u16_delta(&reference, &output.rgba_u16)?;
    if max_delta > 1 {
        return Err(format!("proxy GPU output exceeded tolerance: {max_delta}").into());
    }
    println!(
        "    {label}: presentation={} checksum=0x{:016x} cpu=0x{reference_checksum:016x} max_delta={max_delta}",
        output.presentation_position, output.checksum
    );
    Ok(())
}

fn print_sony_xml_comparison(
    source: &MediaSource,
    parsed: &qgs_codec_h264::ParsedH264AccessUnit,
    sidecar: &SonyXmlSummary,
) {
    println!("Sony XML sidecar comparison:");
    if let Some(model) = &sidecar.camera_model {
        println!("  camera model: {model}");
    }
    println!("  XML duration edit units: {:?}", sidecar.duration);
    println!("  MXF duration edit units: {:?}", source.duration);
    println!("  XML video codec: {:?}", sidecar.video_codec);
    println!(
        "  XML fps: capture={:?} format={:?}",
        sidecar.capture_fps, sidecar.format_fps
    );
    println!(
        "  XML layout: {:?} x {:?} aspect={:?}",
        sidecar.width, sidecar.height, sidecar.aspect_ratio
    );
    println!(
        "  H.264 SPS: {} x {} bit_depth={} chroma={:?}",
        parsed.desc.coded_width,
        parsed.desc.coded_height,
        parsed.desc.bit_depth.get(),
        parsed.desc.chroma
    );
    println!("  XML audio channels: {:?}", sidecar.audio_channels);
    println!("  XML audio codecs: {:?}", sidecar.audio_codecs);
    println!(
        "  XML LTC: tcFps={:?} halfStep={:?}",
        sidecar.ltc_tc_fps, sidecar.ltc_half_step
    );
    if !sidecar.color_values.is_empty() {
        println!("  XML color/acquisition: {:?}", sidecar.color_values);
    }
}

fn find_attr(xml: &str, tag_prefix: &str, attr: &str) -> Option<String> {
    let start = xml.find(tag_prefix)?;
    let rest = &xml[start..];
    let end = rest.find('>')?;
    attr_in_tag(&rest[..end], attr)
}

fn find_all_attrs(xml: &str, tag_prefix: &str, attr: &str) -> Vec<String> {
    let mut values = Vec::new();
    let mut rest = xml;
    while let Some(start) = rest.find(tag_prefix) {
        rest = &rest[start..];
        let Some(end) = rest.find('>') else { break };
        if let Some(value) = attr_in_tag(&rest[..end], attr) {
            values.push(value);
        }
        rest = &rest[end..];
    }
    values
}

fn find_item_value(xml: &str, name: &str) -> Option<String> {
    let needle = format!("<Item name=\"{name}\"");
    let start = xml.find(&needle)?;
    let rest = &xml[start..];
    let end = rest.find('>')?;
    attr_in_tag(&rest[..end], "value")
}

fn attr_in_tag(tag: &str, attr: &str) -> Option<String> {
    let needle = format!("{attr}=\"");
    let start = tag.find(&needle)? + needle.len();
    let end = tag[start..].find('"')?;
    Some(tag[start..start + end].to_string())
}

fn mxf_file_label(path: &Path) -> String {
    if path.is_absolute() {
        if let Ok(current_dir) = std::env::current_dir() {
            if let Ok(relative) = path.strip_prefix(current_dir) {
                return relative.display().to_string();
            }
        }
        return "<external>".to_string();
    }
    path.display().to_string()
}

fn print_haswell_diagnostic_report(report: &qgs_vulkan::HaswellVideoDiagnosticReport) {
    println!();
    println!("Vulkan diagnostic:");
    println!("  device: {}", report.physical_device_name);
    println!("  queue family: {}", report.queue_family_index);
    println!("  modifier: {}", report.modifier);
    println!("  modifier exposed: {}", yes_no(report.modifier_exposed));
    println!(
        "  modifier plane count: {}",
        optional_u32(report.modifier_plane_count)
    );
    println!(
        "  modifier tiling features raw: {}",
        report
            .modifier_tiling_features_raw
            .map(|value| format!("0x{value:016x}"))
            .unwrap_or_else(|| "n/a".to_string())
    );
    println!("  sampled-image: {}", yes_no(report.supports_sampled_image));
    println!("  transfer-src: {}", yes_no(report.supports_transfer_src));
    println!("  transfer-dst: {}", yes_no(report.supports_transfer_dst));
    println!(
        "  ycbcr linear filter: {}",
        yes_no(report.supports_ycbcr_linear_filter)
    );
    println!(
        "  ycbcr separate reconstruction: {}",
        yes_no(report.supports_ycbcr_separate_reconstruction_filter)
    );
    println!(
        "  memory fd type bits: {}",
        optional_hex_u32(report.memory_fd_type_bits)
    );
    println!(
        "  image memory size: {}",
        optional_u64(report.image_memory_size)
    );
    println!(
        "  image memory alignment: {}",
        optional_u64(report.image_memory_alignment)
    );
    println!(
        "  image memory type bits: {}",
        optional_hex_u32(report.image_memory_type_bits)
    );
    println!(
        "  selected memory type index: {}",
        optional_u32(report.selected_memory_type_index)
    );
    println!("  VA objects: {}", report.va_object_count);
    println!("  NV12 format planes: {}", report.nv12_format_plane_count);
    println!(
        "  Vulkan modifier memory planes: {}",
        optional_u32(report.vulkan_modifier_memory_plane_count)
    );
    println!("  disjoint image: {}", yes_no(report.image_create_disjoint));
    println!(
        "  imported memory objects: {}",
        report.imported_memory_objects
    );
    println!("  memory bindings: {}", report.vulkan_memory_bindings);
    println!("  binding offsets: {:?}", report.binding_offsets);

    if let Some(barrier) = &report.barrier {
        println!("Acquire barrier:");
        println!("  srcStageMask: {}", barrier.src_stage_mask);
        println!("  srcAccessMask: {}", barrier.src_access_mask);
        println!("  dstStageMask: {}", barrier.dst_stage_mask);
        println!("  dstAccessMask: {}", barrier.dst_access_mask);
        println!("  oldLayout: {}", barrier.old_layout);
        println!("  newLayout: {}", barrier.new_layout);
        println!("  srcQueueFamilyIndex: {}", barrier.src_queue_family_index);
        println!("  dstQueueFamilyIndex: {}", barrier.dst_queue_family_index);
        println!("  aspectMask: {}", barrier.aspect_mask);
        println!("  baseMipLevel: {}", barrier.base_mip_level);
        println!("  levelCount: {}", barrier.level_count);
        println!("  baseArrayLayer: {}", barrier.base_array_layer);
        println!("  layerCount: {}", barrier.layer_count);
    }

    println!("Results:");
    println!(
        "  vkQueueSubmit: {}",
        report.queue_submit_result.as_deref().unwrap_or("not run")
    );
    println!(
        "  fence wait: {}",
        report.fence_wait_result.as_deref().unwrap_or("not run")
    );
    println!(
        "  GPU read attempted: {}",
        yes_no(report.gpu_read_attempted)
    );
    println!(
        "  GPU read result: {}",
        report.gpu_read_result.as_deref().unwrap_or("not run")
    );
    println!("  classification: {}", report.classification);
    println!("  recommendation: {}", report.recommendation);

    println!("Validation messages:");
    if report.validation_messages.is_empty() {
        println!("  none captured");
    } else {
        for message in &report.validation_messages {
            println!("  {message}");
        }
    }
}

fn print_device(device: &DeviceDesc) {
    println!("- [{:?}] {}", device.class, device.name);
    println!(
        "  vendor=0x{:04x} device=0x{:04x} backend={} api={}",
        device.vendor_id, device.device_id, device.backend, device.api_version
    );
}

fn print_capabilities(capabilities: &DeviceCapabilities) {
    println!("  Compute:");
    println!("    supported: {}", yes_no(capabilities.compute.supported));
    println!(
        "    max workgroup count: {} x {} x {}",
        capabilities.compute.max_workgroup_count[0],
        capabilities.compute.max_workgroup_count[1],
        capabilities.compute.max_workgroup_count[2]
    );
    println!(
        "    max workgroup size: {} x {} x {}",
        capabilities.compute.max_workgroup_size[0],
        capabilities.compute.max_workgroup_size[1],
        capabilities.compute.max_workgroup_size[2]
    );
    println!(
        "    max invocations: {}",
        capabilities.compute.max_workgroup_invocations
    );

    println!("  Memory:");
    for (index, heap) in capabilities.memory.heaps.iter().enumerate() {
        println!(
            "    heap {index}: {} MiB{}",
            heap.size_bytes / (1024 * 1024),
            if heap.device_local {
                ", device-local"
            } else {
                ""
            }
        );
    }
    println!(
        "    memory types: {}",
        capabilities.memory.memory_type_count
    );
    println!(
        "    host-visible: {}",
        yes_no(capabilities.memory.host_visible)
    );
    println!(
        "    host-coherent: {}",
        yes_no(capabilities.memory.host_coherent)
    );
    println!(
        "    device-local: {}",
        yes_no(capabilities.memory.device_local)
    );

    println!("  Interop:");
    println!(
        "    external-memory-fd: {}",
        yes_no(capabilities.interop.external_memory_fd)
    );
    println!(
        "    dma-buf mechanism: {}",
        yes_no(capabilities.interop.dma_buf)
    );
    println!(
        "    semaphore-fd: {}",
        yes_no(capabilities.interop.external_semaphore_fd)
    );
    println!(
        "    fence-fd: {}",
        yes_no(capabilities.interop.external_fence_fd)
    );
}

fn print_video_capabilities(capabilities: &VideoCapabilities) {
    println!("  Video decode:");
    if capabilities.decode.is_empty() {
        println!("    advertised capabilities: 0");
        println!("    backend: no decode capabilities reported");
        return;
    }

    println!("    advertised capabilities: {}", capabilities.decode.len());
    for capability in &capabilities.decode {
        println!(
            "    - {:?} {:?}, {}-bit {:?}",
            capability.codec,
            capability.profile,
            capability.bit_depth.get(),
            capability.chroma
        );
        println!(
            "      max: {} x {}",
            capability.max_width, capability.max_height
        );
        println!(
            "      progressive: {}",
            yes_no(capability.progressive_supported)
        );
        println!(
            "      interlaced: {}",
            yes_no(capability.interlaced_supported)
        );
        println!(
            "      output formats: {:?}",
            capability.output_surface_formats
        );
    }
}

fn create_buffer(
    stream: &mut std::os::unix::net::UnixStream,
    request_id: u64,
    device: &DeviceDesc,
    size_bytes: u64,
    external_sharing: ExternalSharing,
) -> Result<ResourceId, Box<dyn std::error::Error>> {
    println!("Creating {} buffer on:", format_size(size_bytes));
    println!("[{:?}] {}", device.class, device.name);
    send_message(
        stream,
        &WireMessage::CreateBuffer {
            request_id,
            request: CreateBufferRequest {
                desc: BufferDesc {
                    device_id: device.id,
                    size_bytes,
                    usage: BufferUsageFlags::TRANSFER_SRC
                        | BufferUsageFlags::TRANSFER_DST
                        | BufferUsageFlags::STORAGE,
                    memory_preference: MemoryPreference {
                        device_preferred: true,
                        host_visible_required: false,
                        host_coherent_preferred: true,
                    },
                    external_sharing,
                },
            },
        },
    )?;

    let response = receive_message(stream)?;
    let WireMessage::BufferCreated {
        request_id: response_request_id,
        response,
    } = response
    else {
        return Err("expected BUFFER_CREATED response".into());
    };

    if response_request_id != request_id {
        return Err("buffer created request_id did not match request".into());
    }

    println!("Resource created:");
    println!("  id: {}", response.resource_id.get());
    println!("  size: {}", response.size_bytes);
    print_selected_memory(response.selected_memory);

    Ok(response.resource_id)
}

fn create_host_visible_external_buffer(
    stream: &mut std::os::unix::net::UnixStream,
    request_id: u64,
    device: &DeviceDesc,
    handle_type: ExternalHandleType,
) -> Result<Result<ResourceId, ErrorResponse>, Box<dyn std::error::Error>> {
    println!("Testing external memory on:");
    println!("[{:?}] {}", device.class, device.name);
    println!("  requested handle: {handle_type}");
    send_message(
        stream,
        &WireMessage::CreateBuffer {
            request_id,
            request: CreateBufferRequest {
                desc: BufferDesc {
                    device_id: device.id,
                    size_bytes: DEMO_BUFFER_SIZE,
                    usage: BufferUsageFlags::TRANSFER_SRC
                        | BufferUsageFlags::TRANSFER_DST
                        | BufferUsageFlags::STORAGE,
                    memory_preference: MemoryPreference {
                        device_preferred: true,
                        host_visible_required: true,
                        host_coherent_preferred: true,
                    },
                    external_sharing: ExternalSharing::Required { handle_type },
                },
            },
        },
    )?;

    match receive_message(stream)? {
        WireMessage::BufferCreated {
            request_id: response_request_id,
            response,
        } => {
            if response_request_id != request_id {
                return Err("buffer created request_id did not match request".into());
            }
            println!("Created:");
            println!("  ResourceId: {}", response.resource_id.get());
            println!("  size: {}", response.size_bytes);
            println!("  exportable: yes");
            print_selected_memory(response.selected_memory);
            Ok(Ok(response.resource_id))
        }
        WireMessage::Error {
            request_id: response_request_id,
            response,
        } => {
            if response_request_id != request_id {
                return Err("buffer error request_id did not match request".into());
            }
            Ok(Err(response))
        }
        _ => Err("expected BUFFER_CREATED or ERROR response".into()),
    }
}

fn create_image(
    stream: &mut std::os::unix::net::UnixStream,
    request_id: u64,
    device: &DeviceDesc,
    width: u32,
    height: u32,
    external_sharing: ExternalSharing,
) -> Result<ResourceId, Box<dyn std::error::Error>> {
    send_message(
        stream,
        &WireMessage::CreateImage {
            request_id,
            request: CreateImageRequest {
                desc: ImageDesc {
                    device_id: device.id,
                    width,
                    height,
                    format: PixelFormat::Rgba8Unorm,
                    usage: ImageUsageFlags::TRANSFER_SRC
                        | ImageUsageFlags::TRANSFER_DST
                        | ImageUsageFlags::STORAGE,
                    external_sharing,
                },
            },
        },
    )?;

    let response = receive_message(stream)?;
    let WireMessage::ImageCreated {
        request_id: response_request_id,
        response,
    } = response
    else {
        return Err("expected IMAGE_CREATED response".into());
    };

    if response_request_id != request_id {
        return Err("image created request_id did not match request".into());
    }

    println!("Image created:");
    println!("  id: {}", response.resource_id.get());
    println!("  size: {} x {}", response.width, response.height);
    println!("  format: {}", response.format);
    print_selected_memory(response.selected_memory);

    Ok(response.resource_id)
}

fn create_external_image(
    stream: &mut std::os::unix::net::UnixStream,
    request_id: u64,
    device: &DeviceDesc,
    handle_type: ExternalHandleType,
) -> Result<Result<ResourceId, ErrorResponse>, Box<dyn std::error::Error>> {
    println!("Image:");
    println!("  {IMAGE_PROOF_WIDTH} x {IMAGE_PROOF_HEIGHT}");
    println!("  format: Rgba8Unorm");
    println!("  resource: Image");
    println!("  external sharing: {handle_type}");
    send_message(
        stream,
        &WireMessage::CreateImage {
            request_id,
            request: CreateImageRequest {
                desc: ImageDesc {
                    device_id: device.id,
                    width: IMAGE_PROOF_WIDTH,
                    height: IMAGE_PROOF_HEIGHT,
                    format: PixelFormat::Rgba8Unorm,
                    usage: ImageUsageFlags::TRANSFER_SRC
                        | ImageUsageFlags::TRANSFER_DST
                        | ImageUsageFlags::STORAGE,
                    external_sharing: ExternalSharing::Required { handle_type },
                },
            },
        },
    )?;

    match receive_message(stream)? {
        WireMessage::ImageCreated {
            request_id: response_request_id,
            response,
        } => {
            if response_request_id != request_id {
                return Err("image created request_id did not match request".into());
            }
            println!("Created:");
            println!("  ResourceId: {}", response.resource_id.get());
            println!("  exportable: yes");
            print_selected_memory(response.selected_memory);
            Ok(Ok(response.resource_id))
        }
        WireMessage::Error {
            request_id: response_request_id,
            response,
        } => {
            if response_request_id != request_id {
                return Err("image error request_id did not match request".into());
            }
            Ok(Err(response))
        }
        _ => Err("expected IMAGE_CREATED or ERROR response".into()),
    }
}

fn test_external_memory(
    stream: &mut std::os::unix::net::UnixStream,
    mut request_id: u64,
    device: &DeviceDesc,
) -> Result<u64, Box<dyn std::error::Error>> {
    for handle_type in [ExternalHandleType::DmaBuf, ExternalHandleType::OpaqueFd] {
        request_id += 1;
        let resource_id =
            match create_host_visible_external_buffer(stream, request_id, device, handle_type)? {
                Ok(resource_id) => resource_id,
                Err(error) => {
                    println!("  create failed for {handle_type}: {:?}", error.code);
                    continue;
                }
            };

        request_id += 1;
        send_message(
            stream,
            &WireMessage::ExportResource {
                request_id,
                request: ExportResourceRequest {
                    resource_id,
                    handle_type,
                },
            },
        )?;
        let received = receive_message_with_attachments::<1>(stream)?;
        let WireMessage::ResourceExported {
            request_id: response_request_id,
            response,
        } = received.message
        else {
            request_id += 1;
            destroy_resource(stream, request_id, resource_id)?;
            return Err("expected RESOURCE_EXPORTED response".into());
        };
        if response_request_id != request_id {
            request_id += 1;
            destroy_resource(stream, request_id, resource_id)?;
            return Err("resource exported request_id did not match request".into());
        }
        if response.metadata.resource_id != resource_id || response.metadata.attachment_count != 1 {
            request_id += 1;
            destroy_resource(stream, request_id, resource_id)?;
            return Err("resource export metadata did not match request".into());
        }
        let Some(handle) = received.attachments.into_iter().next() else {
            request_id += 1;
            destroy_resource(stream, request_id, resource_id)?;
            return Err("missing exported FD attachment".into());
        };

        println!("Export:");
        println!("  mechanism: {}", response.metadata.handle_type);
        println!("  native FD received: yes");

        validate_import(device, &response.metadata, handle)?;

        request_id += 1;
        destroy_resource(stream, request_id, resource_id)?;
        println!("QGS resource cleanup: success");
        return Ok(request_id);
    }

    println!("  external sharing unsupported on this device by current driver");
    Ok(request_id)
}

fn test_external_gpu_sync(
    stream: &mut std::os::unix::net::UnixStream,
    mut request_id: u64,
    device: &DeviceDesc,
) -> Result<u64, Box<dyn std::error::Error>> {
    const FILL_PATTERN: u32 = 0x5147_5337;

    request_id += 1;
    let resource_id = match create_host_visible_external_buffer(
        stream,
        request_id,
        device,
        ExternalHandleType::DmaBuf,
    )? {
        Ok(resource_id) => resource_id,
        Err(error) => {
            println!("  create failed for DMA-BUF: {:?}", error.code);
            println!("  external GPU synchronization unsupported on this device by current driver");
            return Ok(request_id);
        }
    };

    request_id += 1;
    let (resource_metadata, resource_handle) =
        match export_resource(stream, request_id, resource_id, ExternalHandleType::DmaBuf) {
            Ok(export) => export,
            Err(err) => {
                request_id += 1;
                destroy_resource(stream, request_id, resource_id)?;
                return Err(err);
            }
        };

    request_id += 1;
    let sync_id = match create_sync(stream, request_id, device)? {
        Ok(sync_id) => sync_id,
        Err(error) => {
            println!("  sync creation failed: {:?}", error.code);
            request_id += 1;
            destroy_resource(stream, request_id, resource_id)?;
            return Ok(request_id);
        }
    };

    request_id += 1;
    send_message(
        stream,
        &WireMessage::ExportSync {
            request_id,
            request: ExportSyncRequest {
                sync_id,
                resource_id,
                fill_pattern: FILL_PATTERN,
            },
        },
    )?;
    let received = receive_message_with_attachments::<1>(stream)?;
    let WireMessage::SyncExported {
        request_id: response_request_id,
        response,
    } = received.message
    else {
        request_id += 1;
        destroy_resource(stream, request_id, resource_id)?;
        return Err("expected SYNC_EXPORTED response".into());
    };
    if response_request_id != request_id {
        request_id += 1;
        destroy_resource(stream, request_id, resource_id)?;
        return Err("sync exported request_id did not match request".into());
    }
    if response.metadata.sync_id != sync_id || response.metadata.attachment_count != 1 {
        request_id += 1;
        destroy_resource(stream, request_id, resource_id)?;
        return Err("sync export metadata did not match request".into());
    }
    let Some(sync_handle) = received.attachments.into_iter().next() else {
        request_id += 1;
        destroy_resource(stream, request_id, resource_id)?;
        return Err("missing exported sync FD attachment".into());
    };

    println!("Sync export:");
    println!("  mechanism: {}", response.metadata.handle_type);
    println!("  native sync FD received: yes");

    let importer = VulkanDeviceDiscovery::new()?;
    importer.import_wait_and_validate_synced_buffer(
        device,
        &resource_metadata,
        File::from(resource_handle),
        &response.metadata,
        File::from(sync_handle),
    )?;

    println!("Consumer:");
    println!("  imported shared buffer: success");
    println!("  imported sync primitive: success");
    println!("  GPU wait + dependent copy validation: success");
    println!("  producer->consumer idle wait: no");

    request_id += 1;
    destroy_resource(stream, request_id, resource_id)?;
    println!("QGS resource cleanup: success");

    Ok(request_id)
}

fn test_compute_proof(
    stream: &mut std::os::unix::net::UnixStream,
    mut request_id: u64,
    device: &DeviceDesc,
) -> Result<u64, Box<dyn std::error::Error>> {
    const PRODUCER_PATTERN: u32 = 0x5147_5338;

    println!("[{:?}] {}", device.class, device.name);
    let input = compute_proof_input();
    println!("Input elements: {}", input.len());
    println!("Operation: u32 + 1");
    println!("Execution: Vulkan compute");
    println!("Shared resource: DMA-BUF");
    println!("Synchronization: external sync FD");

    request_id += 1;
    let resource_id = match create_host_visible_external_buffer(
        stream,
        request_id,
        device,
        ExternalHandleType::DmaBuf,
    )? {
        Ok(resource_id) => resource_id,
        Err(error) => {
            println!("  create failed for DMA-BUF: {:?}", error.code);
            println!("Validation:");
            println!("SKIP");
            return Ok(request_id);
        }
    };

    request_id += 1;
    let (resource_metadata, resource_handle) =
        match export_resource(stream, request_id, resource_id, ExternalHandleType::DmaBuf) {
            Ok(export) => export,
            Err(err) => {
                request_id += 1;
                destroy_resource(stream, request_id, resource_id)?;
                return Err(err);
            }
        };

    request_id += 1;
    let sync_id = match create_sync(stream, request_id, device)? {
        Ok(sync_id) => sync_id,
        Err(error) => {
            println!("  sync creation failed: {:?}", error.code);
            request_id += 1;
            destroy_resource(stream, request_id, resource_id)?;
            println!("Validation:");
            println!("SKIP");
            return Ok(request_id);
        }
    };

    request_id += 1;
    send_message(
        stream,
        &WireMessage::ExportSync {
            request_id,
            request: ExportSyncRequest {
                sync_id,
                resource_id,
                fill_pattern: PRODUCER_PATTERN,
            },
        },
    )?;
    let received = receive_message_with_attachments::<1>(stream)?;
    let WireMessage::SyncExported {
        request_id: response_request_id,
        response,
    } = received.message
    else {
        request_id += 1;
        destroy_resource(stream, request_id, resource_id)?;
        return Err("expected SYNC_EXPORTED response".into());
    };
    if response_request_id != request_id {
        request_id += 1;
        destroy_resource(stream, request_id, resource_id)?;
        return Err("sync exported request_id did not match request".into());
    }
    let Some(sync_handle) = received.attachments.into_iter().next() else {
        request_id += 1;
        destroy_resource(stream, request_id, resource_id)?;
        return Err("missing exported sync FD attachment".into());
    };

    let importer = VulkanDeviceDiscovery::new()?;
    let output = importer.import_wait_and_run_compute_increment_proof(
        device,
        &resource_metadata,
        File::from(resource_handle),
        &response.metadata,
        File::from(sync_handle),
        &input,
    )?;

    let expected = input
        .iter()
        .map(|value| value.wrapping_add(1))
        .collect::<Vec<_>>();
    println!("Validation:");
    if output == expected {
        println!("PASS");
    } else {
        println!(
            "FAIL sample: {:?} -> {:?}, expected {:?}",
            &input[..6.min(input.len())],
            &output[..6.min(output.len())],
            &expected[..6.min(expected.len())]
        );
        request_id += 1;
        destroy_resource(stream, request_id, resource_id)?;
        return Err("compute proof output did not match expected values".into());
    }
    println!(
        "Sample: {:?} -> {:?}",
        &input[..6.min(input.len())],
        &output[..6.min(output.len())]
    );

    request_id += 1;
    destroy_resource(stream, request_id, resource_id)?;
    println!("QGS resource cleanup: success");

    Ok(request_id)
}

fn test_image_processing_proof(
    stream: &mut std::os::unix::net::UnixStream,
    mut request_id: u64,
    device: &DeviceDesc,
) -> Result<u64, Box<dyn std::error::Error>> {
    const PRODUCER_PATTERN: u32 = 0x5147_5339;

    println!("[{:?}] {}", device.class, device.name);

    request_id += 1;
    let resource_id =
        match create_external_image(stream, request_id, device, ExternalHandleType::DmaBuf)? {
            Ok(resource_id) => resource_id,
            Err(error) => {
                println!("  create failed for DMA-BUF: {:?}", error.code);
                println!("Validation:");
                println!("SKIP");
                return Ok(request_id);
            }
        };

    request_id += 1;
    let (resource_metadata, resource_handle) =
        match export_resource(stream, request_id, resource_id, ExternalHandleType::DmaBuf) {
            Ok(export) => export,
            Err(err) => {
                request_id += 1;
                destroy_resource(stream, request_id, resource_id)?;
                return Err(err);
            }
        };

    request_id += 1;
    let sync_id = match create_sync(stream, request_id, device)? {
        Ok(sync_id) => sync_id,
        Err(error) => {
            println!("  sync creation failed: {:?}", error.code);
            request_id += 1;
            destroy_resource(stream, request_id, resource_id)?;
            println!("Validation:");
            println!("SKIP");
            return Ok(request_id);
        }
    };

    request_id += 1;
    send_message(
        stream,
        &WireMessage::ExportSync {
            request_id,
            request: ExportSyncRequest {
                sync_id,
                resource_id,
                fill_pattern: PRODUCER_PATTERN,
            },
        },
    )?;
    let received = receive_message_with_attachments::<1>(stream)?;
    let WireMessage::SyncExported {
        request_id: response_request_id,
        response,
    } = received.message
    else {
        request_id += 1;
        destroy_resource(stream, request_id, resource_id)?;
        return Err("expected SYNC_EXPORTED response".into());
    };
    if response_request_id != request_id {
        request_id += 1;
        destroy_resource(stream, request_id, resource_id)?;
        return Err("sync exported request_id did not match request".into());
    }
    let Some(sync_handle) = received.attachments.into_iter().next() else {
        request_id += 1;
        destroy_resource(stream, request_id, resource_id)?;
        return Err("missing exported sync FD attachment".into());
    };

    let input = vec![0_u8; (IMAGE_PROOF_WIDTH as usize) * (IMAGE_PROOF_HEIGHT as usize) * 4];
    let importer = VulkanDeviceDiscovery::new()?;
    let output = importer.import_wait_and_run_image_invert_proof(
        device,
        &resource_metadata,
        File::from(resource_handle),
        &response.metadata,
        File::from(sync_handle),
        &input,
    )?;
    let expected = image_proof_expected_from_clear(PRODUCER_PATTERN, input.len());

    println!("Processing:");
    println!("  operation: private RGBA invert proof");
    println!("  execution: Vulkan GPU");
    println!("  synchronization: external sync FD");
    println!("Validation:");
    if output == expected {
        println!("PASS");
    } else {
        println!(
            "FAIL sample: {:?} -> {:?}, expected {:?}",
            &input[..8.min(input.len())],
            &output[..8.min(output.len())],
            &expected[..8.min(expected.len())]
        );
        request_id += 1;
        destroy_resource(stream, request_id, resource_id)?;
        return Err("image proof output did not match expected pixels".into());
    }
    println!(
        "Sample RGBA bytes: {:?} -> {:?}",
        observed_linear_dma_buf_rgba(PRODUCER_PATTERN),
        &output[..8.min(output.len())]
    );
    println!("producer->consumer idle wait: no");

    request_id += 1;
    destroy_resource(stream, request_id, resource_id)?;
    println!("QGS resource cleanup: success");

    Ok(request_id)
}

fn test_synthetic_yuv422p10_gpu_proof(
    device: &DeviceDesc,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("[{:?}] {}", device.class, device.name);
    let width = 8_u32;
    let height = 4_u32;
    let y_stride = 20_usize;
    let c_stride = 12_usize;
    let mut y = vec![0_u8; y_stride * height as usize];
    let mut cb = vec![0_u8; c_stride * height as usize];
    let mut cr = vec![0_u8; c_stride * height as usize];
    for row in 0..height as usize {
        for x in 0..width as usize {
            let sample = 64_u16 + u16::try_from((row * width as usize + x) * 7 % 877)?;
            y[row * y_stride + x * 2..row * y_stride + x * 2 + 2]
                .copy_from_slice(&sample.to_le_bytes());
        }
        for x in 0..(width as usize / 2) {
            let cb_sample = 512_u16 + u16::try_from((x * 11 + row * 3) % 96)?;
            let cr_sample = 512_u16.saturating_sub(u16::try_from((x * 5 + row * 7) % 96)?);
            cb[row * c_stride + x * 2..row * c_stride + x * 2 + 2]
                .copy_from_slice(&cb_sample.to_le_bytes());
            cr[row * c_stride + x * 2..row * c_stride + x * 2 + 2]
                .copy_from_slice(&cr_sample.to_le_bytes());
        }
    }
    let upload = Yuv422P10Upload {
        device_id: device.id,
        width,
        height,
        y: Yuv422P10Plane {
            width_samples: width,
            height,
            stride_bytes: y_stride,
            data: &y,
        },
        cb: Yuv422P10Plane {
            width_samples: width / 2,
            height,
            stride_bytes: c_stride,
            data: &cb,
        },
        cr: Yuv422P10Plane {
            width_samples: width / 2,
            height,
            stride_bytes: c_stride,
            data: &cr,
        },
        conversion: YcbcrConversion::Rec709Limited,
    };
    let reference = yuv422p10_reference_rgba_u16(&upload)?;
    let reference_checksum = yuv422p10_rgba_u16_checksum(&reference);
    let discovery = VulkanDeviceDiscovery::new()?;
    let started = Instant::now();
    let output = discovery.process_yuv422p10_surface(&upload)?;
    let elapsed = started.elapsed();
    let max_delta = max_u16_delta(&reference, &output.rgba_u16)?;
    println!("  input: 8x4 YUV422P10LE with padded CPU strides");
    println!("  representation: three u32 storage-buffer GPU planes");
    println!("  operation: fixed Rec.709 limited YCbCr -> RGBA u16 compute proof");
    println!("  gpu checksum: 0x{:016x}", output.checksum);
    println!("  cpu checksum: 0x{reference_checksum:016x}");
    println!("  max CPU/GPU delta: {max_delta}");
    println!(
        "  memory: cpu_surface={} staging={} gpu_planes={} output={}",
        output.cpu_surface_bytes, output.staging_bytes, output.gpu_plane_bytes, output.output_bytes
    );
    println!(
        "  DEVELOPMENT OBSERVATION - NOT A BENCHMARK: {:.3}ms upload/process/readback",
        elapsed.as_secs_f64() * 1000.0
    );
    if max_delta > 1 {
        return Err("synthetic YUV422P10 GPU proof exceeded tolerance".into());
    }
    println!("Validation:");
    println!("PASS");
    Ok(())
}

struct SyntheticYuv422P10Frame {
    width: u32,
    height: u32,
    y_stride: usize,
    c_stride: usize,
    y: Vec<u8>,
    cb: Vec<u8>,
    cr: Vec<u8>,
}

impl SyntheticYuv422P10Frame {
    fn upload(&self, device_id: qgs_protocol::DeviceId) -> Yuv422P10Upload<'_> {
        Yuv422P10Upload {
            device_id,
            width: self.width,
            height: self.height,
            y: Yuv422P10Plane {
                width_samples: self.width,
                height: self.height,
                stride_bytes: self.y_stride,
                data: &self.y,
            },
            cb: Yuv422P10Plane {
                width_samples: self.width / 2,
                height: self.height,
                stride_bytes: self.c_stride,
                data: &self.cb,
            },
            cr: Yuv422P10Plane {
                width_samples: self.width / 2,
                height: self.height,
                stride_bytes: self.c_stride,
                data: &self.cr,
            },
            conversion: YcbcrConversion::Rec709Limited,
        }
    }
}

fn make_synthetic_yuv422p10_frame(
    frame_index: usize,
) -> Result<SyntheticYuv422P10Frame, Box<dyn std::error::Error>> {
    let width = 8_u32;
    let height = 4_u32;
    let y_stride = 20_usize;
    let c_stride = 12_usize;
    let mut y = vec![0_u8; y_stride * height as usize];
    let mut cb = vec![0_u8; c_stride * height as usize];
    let mut cr = vec![0_u8; c_stride * height as usize];
    for row in 0..height as usize {
        for x in 0..width as usize {
            let base = row * width as usize + x + frame_index * 13;
            let sample = 64_u16 + u16::try_from((base * 7) % 877)?;
            y[row * y_stride + x * 2..row * y_stride + x * 2 + 2]
                .copy_from_slice(&sample.to_le_bytes());
        }
        for x in 0..(width as usize / 2) {
            let cb_sample = 512_u16 + u16::try_from((x * 11 + row * 3 + frame_index * 5) % 96)?;
            let cr_sample =
                512_u16.saturating_sub(u16::try_from((x * 5 + row * 7 + frame_index * 9) % 96)?);
            cb[row * c_stride + x * 2..row * c_stride + x * 2 + 2]
                .copy_from_slice(&cb_sample.to_le_bytes());
            cr[row * c_stride + x * 2..row * c_stride + x * 2 + 2]
                .copy_from_slice(&cr_sample.to_le_bytes());
        }
    }
    Ok(SyntheticYuv422P10Frame {
        width,
        height,
        y_stride,
        c_stride,
        y,
        cb,
        cr,
    })
}

fn test_reusable_yuv422p10_gpu_processor(
    device: &DeviceDesc,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("  reusable processor:");
    let discovery = VulkanDeviceDiscovery::new()?;
    let frames = (0..6)
        .map(make_synthetic_yuv422p10_frame)
        .collect::<Result<Vec<_>, _>>()?;
    let mut processor = GpuFrameProcessor::new(
        &discovery,
        GpuFrameProcessorConfig {
            device_id: device.id,
            width: frames[0].width,
            height: frames[0].height,
            slot_count: 3,
            conversion: YcbcrConversion::Rec709Limited,
        },
    )?;
    let started = Instant::now();
    let mut pending = VecDeque::new();
    let mut checksums = Vec::new();
    for (index, frame) in frames.iter().enumerate() {
        let upload = frame.upload(device.id);
        let reference = yuv422p10_reference_rgba_u16(&upload)?;
        let reference_checksum = yuv422p10_rgba_u16_checksum(&reference);
        match processor.submit_frame(
            &upload,
            FrameIdentity {
                presentation_position: index as u64,
            },
        ) {
            Ok(token) => pending.push_back((token, index, reference, reference_checksum)),
            Err(FrameProcessorError::NoFrameSlotAvailable) => {
                while let Some((token, submitted_index, reference, reference_checksum)) =
                    pending.pop_front()
                {
                    let output = processor.wait_for_frame(token)?;
                    validate_reusable_output(
                        &output,
                        submitted_index,
                        &reference,
                        reference_checksum,
                    )?;
                    checksums.push(output.checksum);
                }
                let token = processor.submit_frame(
                    &upload,
                    FrameIdentity {
                        presentation_position: index as u64,
                    },
                )?;
                pending.push_back((token, index, reference, reference_checksum));
            }
            Err(err) => return Err(Box::new(err)),
        }
        if index == 2 {
            let fourth = frames[3].upload(device.id);
            match processor.submit_frame(
                &fourth,
                FrameIdentity {
                    presentation_position: 3,
                },
            ) {
                Err(FrameProcessorError::NoFrameSlotAvailable) => {
                    println!("    bounded backpressure after 3 in-flight frames: yes");
                }
                Ok(_) => return Err("processor accepted an unbounded fourth slot".into()),
                Err(err) => return Err(Box::new(err)),
            }
        }
    }
    while let Some((token, index, reference, reference_checksum)) = pending.pop_front() {
        let output = processor.wait_for_frame(token)?;
        validate_reusable_output(&output, index, &reference, reference_checksum)?;
        checksums.push(output.checksum);
    }
    let elapsed = started.elapsed();
    let counters = processor.counters();
    println!("    frames submitted: {}", counters.frame_submissions);
    println!("    slot reuses: {}", counters.slot_reuses);
    println!(
        "    resources: pipelines={} shaders={} staging={} gpu_planes={} outputs={} readbacks={} command_buffers={}",
        counters.pipeline_creations,
        counters.shader_module_creations,
        counters.staging_allocations,
        counters.gpu_plane_allocations,
        counters.output_allocations,
        counters.readback_allocations,
        counters.command_buffer_count
    );
    println!(
        "    checksums: {:?}",
        checksums
            .iter()
            .map(|checksum| format!("0x{checksum:016x}"))
            .collect::<Vec<_>>()
    );
    println!(
        "    DEVELOPMENT OBSERVATION - NOT A BENCHMARK: {:.3}ms for 6 reusable submissions",
        elapsed.as_secs_f64() * 1000.0
    );
    if counters.pipeline_creations != 1
        || counters.shader_module_creations != 1
        || counters.staging_allocations != 9
        || counters.gpu_plane_allocations != 9
        || counters.output_allocations != 3
        || counters.command_buffer_count != 3
        || counters.frame_submissions != 6
        || counters.slot_reuses < 3
    {
        return Err("reusable processor counters did not show bounded reuse".into());
    }
    println!("    validation: PASS");
    Ok(())
}

fn validate_reusable_output(
    output: &qgs_vulkan::ProcessedFrameOutput,
    expected_index: usize,
    reference: &[u16],
    reference_checksum: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    if output.presentation_position != expected_index as u64 {
        return Err("processed output presentation identity mismatch".into());
    }
    let max_delta = max_u16_delta(reference, &output.rgba_u16)?;
    if max_delta > 1 {
        return Err(format!("reusable GPU output exceeded tolerance: {max_delta}").into());
    }
    println!(
        "    frame {expected_index}: slot={} seq={} gpu=0x{:016x} cpu=0x{reference_checksum:016x} max_delta={max_delta}",
        output.slot_index, output.submission_sequence, output.checksum
    );
    Ok(())
}

fn compute_proof_input() -> Vec<u32> {
    let mut values = (0..1024_u32).collect::<Vec<_>>();
    values[..6].copy_from_slice(&[0, 1, 2, 3, 100, 1000]);
    values
}

fn image_clear_rgba(pattern: u32) -> [u8; 4] {
    [
        (pattern & 0xff) as u8,
        ((pattern >> 8) & 0xff) as u8,
        ((pattern >> 16) & 0xff) as u8,
        ((pattern >> 24) & 0xff) as u8,
    ]
}

fn image_proof_expected_from_clear(pattern: u32, len: usize) -> Vec<u8> {
    let [r, g, b, a] = observed_linear_dma_buf_rgba(pattern);
    std::iter::repeat_n([255 - r, 255 - g, 255 - b, a], len / 4)
        .flatten()
        .collect()
}

fn observed_linear_dma_buf_rgba(pattern: u32) -> [u8; 4] {
    [0, 0, 0, image_clear_rgba(pattern)[3]]
}

fn export_resource(
    stream: &mut std::os::unix::net::UnixStream,
    request_id: u64,
    resource_id: ResourceId,
    handle_type: ExternalHandleType,
) -> Result<(qgs_protocol::ExportedResourceMetadata, OwnedFd), Box<dyn std::error::Error>> {
    send_message(
        stream,
        &WireMessage::ExportResource {
            request_id,
            request: ExportResourceRequest {
                resource_id,
                handle_type,
            },
        },
    )?;
    let received = receive_message_with_attachments::<1>(stream)?;
    let WireMessage::ResourceExported {
        request_id: response_request_id,
        response,
    } = received.message
    else {
        return Err("expected RESOURCE_EXPORTED response".into());
    };
    if response_request_id != request_id {
        return Err("resource exported request_id did not match request".into());
    }
    if response.metadata.resource_id != resource_id || response.metadata.attachment_count != 1 {
        return Err("resource export metadata did not match request".into());
    }
    let Some(handle) = received.attachments.into_iter().next() else {
        return Err("missing exported FD attachment".into());
    };

    println!("Export:");
    println!("  mechanism: {}", response.metadata.handle_type);
    println!("  native FD received: yes");

    Ok((response.metadata, handle))
}

fn create_sync(
    stream: &mut std::os::unix::net::UnixStream,
    request_id: u64,
    device: &DeviceDesc,
) -> Result<Result<SyncId, ErrorResponse>, Box<dyn std::error::Error>> {
    send_message(
        stream,
        &WireMessage::CreateSync {
            request_id,
            request: CreateSyncRequest {
                device_id: device.id,
                kind: SyncKind::BinarySemaphore,
                handle_type: SyncExportHandleType::SyncFd,
            },
        },
    )?;

    match receive_message(stream)? {
        WireMessage::SyncCreated {
            request_id: response_request_id,
            response,
        } => {
            if response_request_id != request_id {
                return Err("sync created request_id did not match request".into());
            }
            println!("Sync created:");
            println!("  id: {}", response.sync_id.get());
            println!("  kind: BinarySemaphore");
            println!("  export handle: sync-fd");
            Ok(Ok(response.sync_id))
        }
        WireMessage::Error {
            request_id: response_request_id,
            response,
        } => {
            if response_request_id != request_id {
                return Err("sync error request_id did not match request".into());
            }
            Ok(Err(response))
        }
        _ => Err("expected SYNC_CREATED or ERROR response".into()),
    }
}

fn validate_import(
    device: &DeviceDesc,
    metadata: &qgs_protocol::ExportedResourceMetadata,
    handle: OwnedFd,
) -> Result<(), Box<dyn std::error::Error>> {
    let importer = VulkanDeviceDiscovery::new()?;
    importer.import_and_validate_external_buffer(device, metadata, File::from(handle))?;

    println!("Import:");
    println!("  second Vulkan context: success");
    println!("  shared allocation validation: success");

    Ok(())
}

fn query_h264_decode_support(
    stream: &mut std::os::unix::net::UnixStream,
    request_id: &mut u64,
    device: &DeviceDesc,
    profile: H264Profile,
    chroma: ChromaSubsampling,
    bit_depth: u8,
) -> Result<bool, Box<dyn std::error::Error>> {
    *request_id += 1;
    send_message(
        stream,
        &WireMessage::QueryVideoCapabilities {
            request_id: *request_id,
            request: QueryVideoCapabilitiesRequest {
                device_id: device.id,
            },
        },
    )?;
    let response = receive_message(stream)?;
    let WireMessage::VideoCapabilities {
        request_id: response_request_id,
        response,
    } = response
    else {
        return Err("expected VIDEO_CAPABILITIES response".into());
    };
    if response_request_id != *request_id {
        return Err("video capability response request_id did not match request".into());
    }
    Ok(response.capabilities.decode.iter().any(|capability| {
        capability.codec == VideoCodec::H264
            && capability.profile == VideoProfile::H264(profile)
            && capability.bit_depth.get() == bit_depth
            && capability.chroma == chroma
            && capability
                .output_surface_formats
                .contains(&qgs_protocol::VideoSurfaceFormat::Nv12)
    }))
}

fn test_h264_decode(
    stream: &mut std::os::unix::net::UnixStream,
    mut request_id: u64,
    device: &DeviceDesc,
) -> Result<u64, Box<dyn std::error::Error>> {
    println!("H.264 decode proof:");
    println!("[{:?}] {}", device.class, device.name);

    let supports_h264_baseline = query_h264_decode_support(
        stream,
        &mut request_id,
        device,
        H264Profile::Baseline,
        ChromaSubsampling::Cs420,
        8,
    )?;

    request_id += 1;
    let config = DecoderConfig {
        device_id: device.id,
        codec: VideoCodec::H264,
        profile: VideoProfile::H264(H264Profile::Baseline),
        bit_depth: BitDepth::new(8)?,
        chroma: ChromaSubsampling::Cs420,
        coded_width: 64,
        coded_height: 64,
        scan_mode: ScanMode::Progressive,
    };
    send_message(
        stream,
        &WireMessage::CreateDecoder {
            request_id,
            request: CreateDecoderRequest { config },
        },
    )?;
    let response = receive_message(stream)?;
    let decoder_id = match response {
        WireMessage::DecoderCreated {
            request_id: response_request_id,
            response,
        } => {
            if response_request_id != request_id {
                return Err("decoder created request_id did not match request".into());
            }
            if !supports_h264_baseline {
                println!("Decoder created through software fallback:");
            } else {
                println!("Decoder created:");
            }
            println!("  id: {}", response.decoder_id.get());
            response.decoder_id
        }
        WireMessage::Error {
            request_id: response_request_id,
            response,
        } => {
            if response_request_id != request_id {
                return Err("decoder error request_id did not match request".into());
            }
            if supports_h264_baseline {
                return Err(format!(
                    "decoder creation failed despite advertised support: {:?}",
                    response.code
                )
                .into());
            }
            if response.code != ProtocolErrorCode::UnsupportedDecodeConfiguration {
                return Err(
                    format!("unexpected unsupported decoder error: {:?}", response.code).into(),
                );
            }
            println!("Decoder unsupported as expected for this device.");
            return Ok(request_id);
        }
        _ => return Err("expected DECODER_CREATED or ERROR response".into()),
    };

    request_id += 1;
    send_message(
        stream,
        &WireMessage::SubmitAccessUnit {
            request_id,
            request: SubmitAccessUnitRequest {
                decoder_id,
                data: H264_FIXTURE.to_vec(),
            },
        },
    )?;
    let response = receive_message(stream)?;
    let WireMessage::DecodeOutput {
        request_id: response_request_id,
        response,
    } = response
    else {
        return Err("expected DECODE_OUTPUT response".into());
    };
    if response_request_id != request_id {
        return Err("decode output request_id did not match request".into());
    }
    if response.decoder_id != decoder_id {
        return Err("decode output decoder_id did not match request".into());
    }
    let mut outputs = response.outputs;
    if outputs.is_empty() && !supports_h264_baseline {
        request_id += 1;
        send_message(
            stream,
            &WireMessage::FlushDecoder {
                request_id,
                request: FlushDecoderRequest { decoder_id },
            },
        )?;
        let response = receive_message(stream)?;
        let WireMessage::DecodeOutput {
            request_id: response_request_id,
            response,
        } = response
        else {
            return Err("expected software fallback flush DECODE_OUTPUT response".into());
        };
        if response_request_id != request_id || response.decoder_id != decoder_id {
            return Err("software fallback flush response correlation failed".into());
        }
        outputs = response.outputs;
    }
    if outputs.len() != 1 {
        return Err(format!(
            "IDR decode expected one output surface, got {}",
            outputs.len()
        )
        .into());
    }
    let output = &outputs[0];
    println!("Decoded VideoSurface:");
    println!("  resource id: {}", output.resource_id.get());
    println!(
        "  {} x {} {:?}",
        output.surface.coded_width, output.surface.coded_height, output.surface.format
    );
    if supports_h264_baseline {
        println!("  validation: backend VA readback succeeded");
    } else {
        println!("  validation: software fallback output succeeded");
    }

    request_id += 1;
    destroy_resource(stream, request_id, output.resource_id)?;
    println!("Decoded VideoSurface destroyed successfully.");

    request_id += 1;
    send_message(
        stream,
        &WireMessage::DestroyDecoder {
            request_id,
            request: DestroyDecoderRequest { decoder_id },
        },
    )?;
    let response = receive_message(stream)?;
    let WireMessage::DecoderDestroyed {
        request_id: response_request_id,
        response,
    } = response
    else {
        return Err("expected DECODER_DESTROYED response".into());
    };
    if response_request_id != request_id || response.decoder_id != decoder_id {
        return Err("decoder destroyed response did not match request".into());
    }
    println!("Decoder destroyed successfully.");

    Ok(request_id)
}

fn test_h264_long_gop_decode(
    stream: &mut std::os::unix::net::UnixStream,
    mut request_id: u64,
    device: &DeviceDesc,
) -> Result<u64, Box<dyn std::error::Error>> {
    println!("H.264 Long-GOP decode proof:");
    println!("[{:?}] {}", device.class, device.name);

    let access_units = split_h264_annex_b_access_units(H264_LONG_GOP_FIXTURE)?;
    println!("  access units: {}", access_units.len());
    let parsed = qgs_codec_h264::parse_annex_b_access_unit(
        access_units
            .first()
            .ok_or("Long-GOP fixture did not contain access units")?,
    )?;
    let supports_h264_main = query_h264_decode_support(
        stream,
        &mut request_id,
        device,
        H264Profile::Main,
        ChromaSubsampling::Cs420,
        8,
    )?;

    request_id += 1;
    let config = DecoderConfig {
        device_id: device.id,
        codec: VideoCodec::H264,
        profile: VideoProfile::H264(H264Profile::Main),
        bit_depth: parsed.desc.bit_depth,
        chroma: parsed.desc.chroma,
        coded_width: parsed.desc.coded_width,
        coded_height: parsed.desc.coded_height,
        scan_mode: parsed.desc.scan_mode,
    };
    send_message(
        stream,
        &WireMessage::CreateDecoder {
            request_id,
            request: CreateDecoderRequest { config },
        },
    )?;
    let response = receive_message(stream)?;
    let decoder_id = match response {
        WireMessage::DecoderCreated {
            request_id: response_request_id,
            response,
        } => {
            if response_request_id != request_id {
                return Err("Long-GOP decoder request_id did not match".into());
            }
            if !supports_h264_main {
                println!("  Long-GOP decoder created through software fallback.");
            }
            response.decoder_id
        }
        WireMessage::Error { response, .. }
            if response.code == ProtocolErrorCode::UnsupportedDecodeConfiguration =>
        {
            println!("  Long-GOP decoder unsupported as expected for this device.");
            return Ok(request_id);
        }
        other => {
            return Err(format!("unexpected Long-GOP decoder response: {other:?}").into());
        }
    };

    let mut output_count = 0_usize;
    let mut output_resources = Vec::new();
    for (index, access_unit) in access_units.iter().enumerate() {
        request_id += 1;
        send_message(
            stream,
            &WireMessage::SubmitAccessUnit {
                request_id,
                request: SubmitAccessUnitRequest {
                    decoder_id,
                    data: access_unit.clone(),
                },
            },
        )?;
        let response = receive_message(stream)?;
        let WireMessage::DecodeOutput {
            request_id: response_request_id,
            response,
        } = response
        else {
            return Err("expected Long-GOP DECODE_OUTPUT response".into());
        };
        if response_request_id != request_id || response.decoder_id != decoder_id {
            return Err("Long-GOP decode response correlation failed".into());
        }
        println!(
            "  submitted AU {:02}: {} output surface(s)",
            index,
            response.outputs.len()
        );
        output_count += response.outputs.len();
        output_resources.extend(
            response
                .outputs
                .into_iter()
                .map(|output| output.resource_id),
        );
    }

    request_id += 1;
    send_message(
        stream,
        &WireMessage::FlushDecoder {
            request_id,
            request: FlushDecoderRequest { decoder_id },
        },
    )?;
    let response = receive_message(stream)?;
    let WireMessage::DecodeOutput {
        request_id: response_request_id,
        response,
    } = response
    else {
        return Err("expected Long-GOP flush DECODE_OUTPUT response".into());
    };
    if response_request_id != request_id || response.decoder_id != decoder_id {
        return Err("Long-GOP flush response correlation failed".into());
    }
    println!("  flush output surface(s): {}", response.outputs.len());
    output_count += response.outputs.len();
    output_resources.extend(
        response
            .outputs
            .into_iter()
            .map(|output| output.resource_id),
    );

    if output_count != 12 {
        return Err(format!("Long-GOP expected 12 output frames, got {output_count}").into());
    }
    if supports_h264_main {
        println!("  validation: backend VA readback succeeded for every decoded frame");
    } else {
        println!("  validation: software fallback output succeeded for every decoded frame");
    }

    for resource_id in output_resources {
        request_id += 1;
        destroy_resource(stream, request_id, resource_id)?;
    }

    request_id += 1;
    send_message(
        stream,
        &WireMessage::DestroyDecoder {
            request_id,
            request: DestroyDecoderRequest { decoder_id },
        },
    )?;
    let response = receive_message(stream)?;
    let WireMessage::DecoderDestroyed {
        request_id: response_request_id,
        response,
    } = response
    else {
        return Err("expected Long-GOP DECODER_DESTROYED response".into());
    };
    if response_request_id != request_id || response.decoder_id != decoder_id {
        return Err("Long-GOP decoder destroyed response did not match".into());
    }
    println!("  decoded frames: {output_count}");
    println!("  decoder destroyed successfully");

    Ok(request_id)
}

fn split_h264_annex_b_access_units(
    data: &[u8],
) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
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
        return Err("Long-GOP fixture is not Annex B".into());
    }

    let mut access_units = Vec::new();
    let mut current = Vec::new();
    let mut seen_vcl = false;
    for (index, (start, prefix_len)) in starts.iter().copied().enumerate() {
        let nal_start = start + prefix_len;
        let nal_end = starts
            .get(index + 1)
            .map(|(next, _)| *next)
            .unwrap_or(data.len());
        if nal_start >= nal_end {
            return Err("malformed Annex B NAL in Long-GOP fixture".into());
        }
        let nal_type = data[nal_start] & 0x1f;
        let is_vcl = nal_type == 1 || nal_type == 5;
        if is_vcl && seen_vcl && !current.is_empty() {
            access_units.push(std::mem::take(&mut current));
        }
        current.extend_from_slice(&data[start..nal_end]);
        if is_vcl {
            seen_vcl = true;
        }
    }
    if !current.is_empty() {
        access_units.push(current);
    }
    Ok(access_units)
}

fn leave_h264_decoder_for_disconnect(
    stream: &mut std::os::unix::net::UnixStream,
    mut request_id: u64,
    devices: &[&DeviceDesc],
) -> Result<u64, Box<dyn std::error::Error>> {
    for device in devices {
        request_id += 1;
        let config = DecoderConfig {
            device_id: device.id,
            codec: VideoCodec::H264,
            profile: VideoProfile::H264(H264Profile::Baseline),
            bit_depth: BitDepth::new(8)?,
            chroma: ChromaSubsampling::Cs420,
            coded_width: 64,
            coded_height: 64,
            scan_mode: ScanMode::Progressive,
        };
        send_message(
            stream,
            &WireMessage::CreateDecoder {
                request_id,
                request: CreateDecoderRequest { config },
            },
        )?;

        let response = receive_message(stream)?;
        let decoder_id = match response {
            WireMessage::DecoderCreated {
                request_id: response_request_id,
                response,
            } => {
                if response_request_id != request_id {
                    return Err("transient decoder response request_id did not match".into());
                }
                response.decoder_id
            }
            WireMessage::Error { response, .. }
                if response.code == ProtocolErrorCode::UnsupportedDecodeConfiguration =>
            {
                continue;
            }
            _ => return Err("expected transient DECODER_CREATED or unsupported error".into()),
        };

        request_id += 1;
        send_message(
            stream,
            &WireMessage::SubmitAccessUnit {
                request_id,
                request: SubmitAccessUnitRequest {
                    decoder_id,
                    data: H264_FIXTURE.to_vec(),
                },
            },
        )?;
        let response = receive_message(stream)?;
        let WireMessage::DecodeOutput {
            request_id: response_request_id,
            response,
        } = response
        else {
            return Err("expected transient DECODE_OUTPUT response".into());
        };
        if response_request_id != request_id {
            return Err("transient decode output request_id did not match".into());
        }

        println!("Leaving transient H.264 decoder and VideoSurface alive for disconnect cleanup:");
        println!(
            "  decoder id: {} resource id: {}",
            decoder_id.get(),
            response
                .outputs
                .first()
                .map(|output| output.resource_id.get())
                .unwrap_or(0)
        );
        return Ok(request_id);
    }

    println!("No H.264 decoder available for disconnect cleanup exercise.");
    Ok(request_id)
}

fn destroy_resource(
    stream: &mut std::os::unix::net::UnixStream,
    request_id: u64,
    resource_id: ResourceId,
) -> Result<(), Box<dyn std::error::Error>> {
    send_message(
        stream,
        &WireMessage::DestroyResource {
            request_id,
            request: DestroyResourceRequest { resource_id },
        },
    )?;

    let response = receive_message(stream)?;
    let WireMessage::ResourceDestroyed {
        request_id: response_request_id,
        response,
    } = response
    else {
        return Err("expected RESOURCE_DESTROYED response".into());
    };

    if response_request_id != request_id {
        return Err("resource destroyed request_id did not match request".into());
    }
    if response.resource_id != resource_id {
        return Err("resource destroyed id did not match request".into());
    }

    Ok(())
}

fn print_selected_memory(memory: SelectedMemoryProperties) {
    println!("  selected memory:");
    println!("    device-local: {}", yes_no(memory.device_local));
    println!("    host-visible: {}", yes_no(memory.host_visible));
    println!("    host-coherent: {}", yes_no(memory.host_coherent));
}

fn format_size(size_bytes: u64) -> String {
    if size_bytes.is_multiple_of(1024 * 1024) {
        format!("{} MiB", size_bytes / (1024 * 1024))
    } else if size_bytes.is_multiple_of(1024) {
        format!("{} KiB", size_bytes / 1024)
    } else {
        format!("{size_bytes} bytes")
    }
}

fn optional_u32(value: Option<u32>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "n/a".to_string())
}

fn optional_u64(value: Option<u64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "n/a".to_string())
}

fn optional_hex_u32(value: Option<u32>) -> String {
    value
        .map(|value| format!("0x{value:08x}"))
        .unwrap_or_else(|| "n/a".to_string())
}

const fn yes_no(value: bool) -> &'static str {
    if value {
        "yes"
    } else {
        "no"
    }
}

struct Args {
    socket_path: PathBuf,
    video_capabilities_only: bool,
    h264_decode_only: bool,
    haswell_video_diagnostic: bool,
    mxf_inspect_path: Option<PathBuf>,
    software_decode_mxf_path: Option<PathBuf>,
    software_gpu_mxf_path: Option<PathBuf>,
    native_original_video_decode_path: Option<PathBuf>,
    proxy_proof_paths: Option<(PathBuf, PathBuf)>,
    proxy_throughput_paths: Option<(PathBuf, PathBuf)>,
    proxy_playback_paths: Option<(PathBuf, PathBuf)>,
    proxy_playback_profile: ProxyPlaybackProfile,
    qnc_journalist_demo_paths: Option<(PathBuf, PathBuf)>,
    original_audio_extract_path: Option<PathBuf>,
    broadcast_runtime_contract_paths: Option<(PathBuf, PathBuf)>,
    broadcast_runtime_state_machine_paths: Option<(PathBuf, PathBuf)>,
    broadcast_runtime_preroll_paths: Option<(PathBuf, PathBuf)>,
    broadcast_runtime_prepared_slots_paths: Option<(PathBuf, PathBuf)>,
    broadcast_player_runtime_events_paths: Option<(PathBuf, PathBuf)>,
    broadcast_player_runtime_payloads_paths: Option<(PathBuf, PathBuf)>,
    broadcast_player_runtime_video_payloads_paths: Option<(PathBuf, PathBuf)>,
    broadcast_player_runtime_device_boundary_paths: Option<(PathBuf, PathBuf)>,
    broadcast_player_runtime_test_presenter_paths: Option<(PathBuf, PathBuf)>,
    broadcast_player_runtime_test_audio_sink_paths: Option<(PathBuf, PathBuf)>,
    broadcast_player_runtime_simulate_paths: Option<(PathBuf, PathBuf)>,
    broadcast_player_runtime_original_video_payloads_paths: Option<(PathBuf, PathBuf)>,
    broadcast_player_runtime_verification_paths: Option<(PathBuf, PathBuf)>,
    qnc_prepared_input_descriptor_paths: Option<(PathBuf, PathBuf)>,
    qgs_qnc_prepared_input_mapping_paths: Option<(PathBuf, PathBuf)>,
    qgs_qnc_prepared_input_fixture: Option<String>,
    qgs_input_plan_paths: Option<(PathBuf, PathBuf)>,
    qgs_transport_engine_parity_paths: Option<(PathBuf, PathBuf)>,
    qgs_frame_clock_parity_paths: Option<(PathBuf, PathBuf)>,
    qgs_playout_buffer_tick_paths: Option<(PathBuf, PathBuf)>,
    qgs_runtime_lifecycle_events_paths: Option<(PathBuf, PathBuf)>,
    qgs_qnc_event_projection_paths: Option<(PathBuf, PathBuf)>,
    qgs_session_command_boundary_paths: Option<(PathBuf, PathBuf)>,
    qgs_session_runtime_control_paths: Option<(PathBuf, PathBuf)>,
    qgs_presenter_monitor_boundary_paths: Option<(PathBuf, PathBuf)>,
    qgs_file_presenter_diagnostic_paths: Option<(PathBuf, PathBuf)>,
    qgs_gpu_payload_readback_diagnostic_paths: Option<(PathBuf, PathBuf)>,
    qgs_runtime_surface_e2e_paths: Option<(PathBuf, PathBuf)>,
    qgs_broadcast_player_control_core_paths: Option<(PathBuf, PathBuf)>,
    qgs_broadcast_player_operational_runtime_paths: Option<(PathBuf, PathBuf)>,
    qgs_broadcast_player_fault_recovery_paths: Option<(PathBuf, PathBuf)>,
    qgs_broadcast_player_run_paths: Option<(PathBuf, PathBuf)>,
    qgs_broadcast_player_run_options: QgsBroadcastPlayerRunOptions,
    qgs_broadcast_player_live_paths: Option<(PathBuf, PathBuf)>,
    qgs_broadcast_player_live_options: QgsBroadcastPlayerLiveOptions,
    qgs_broadcast_player_control_session_paths: Option<(PathBuf, PathBuf)>,
    qgs_broadcast_player_control_session_script: String,
    qgs_device_backend_selection_policy: Option<QgsDeviceSelectionPolicy>,
    linux_audio_device_probe_path: Option<PathBuf>,
    pipewire_audio_prototype_path: Option<PathBuf>,
    pipewire_audio_native_prototype_path: Option<PathBuf>,
    pipewire_audio_audible_smoke_test_path: Option<PathBuf>,
    pipewire_audio_segment_playback_path: Option<PathBuf>,
    broadcast_runtime_audio_pipewire_paths: Option<(PathBuf, PathBuf)>,
    broadcast_runtime_audio_pipewire_audible_paths: Option<(PathBuf, PathBuf)>,
    pipewire_audio_content_audit_paths: Option<(PathBuf, PathBuf)>,
    pipewire_audio_mono_monitor_paths: Option<(PathBuf, PathBuf)>,
    pipewire_audio_discrete_4mono_paths: Option<(PathBuf, PathBuf)>,
    diagnostic_wav_path: Option<PathBuf>,
    audio_content_audit_start_ms: Option<u64>,
    audio_content_audit_duration_ms: Option<u64>,
    audio_content_audit_output_dir: Option<PathBuf>,
    audio_content_audit_find_loudest_range_ms: Option<u64>,
    audio_monitor_track: Option<u16>,
}

impl Args {
    fn parse() -> Self {
        let mut socket_path = None;
        let mut video_capabilities_only = false;
        let mut h264_decode_only = false;
        let mut haswell_video_diagnostic = false;
        let mut mxf_inspect_path = None;
        let mut next_arg_is_mxf_path = false;
        let mut software_decode_mxf_path = None;
        let mut next_arg_is_software_decode_mxf_path = false;
        let mut software_gpu_mxf_path = None;
        let mut next_arg_is_software_gpu_mxf_path = false;
        let mut native_original_video_decode_path = None;
        let mut next_arg_is_native_original_video_decode_path = false;
        let mut proxy_proof_original = None;
        let mut proxy_proof_paths = None;
        let mut proxy_throughput_original = None;
        let mut proxy_throughput_paths = None;
        let mut proxy_playback_original = None;
        let mut proxy_playback_paths = None;
        let mut proxy_playback_profile = ProxyPlaybackProfile::SourceRate;
        let mut qnc_journalist_demo_original = None;
        let mut qnc_journalist_demo_paths = None;
        let mut original_audio_extract_path = None;
        let mut broadcast_runtime_contract_original = None;
        let mut broadcast_runtime_contract_paths = None;
        let mut broadcast_runtime_state_machine_original = None;
        let mut broadcast_runtime_state_machine_paths = None;
        let mut broadcast_runtime_preroll_original = None;
        let mut broadcast_runtime_preroll_paths = None;
        let mut broadcast_runtime_prepared_slots_original = None;
        let mut broadcast_runtime_prepared_slots_paths = None;
        let mut broadcast_player_runtime_events_original = None;
        let mut broadcast_player_runtime_events_paths = None;
        let mut broadcast_player_runtime_payloads_original = None;
        let mut broadcast_player_runtime_payloads_paths = None;
        let mut broadcast_player_runtime_video_payloads_original = None;
        let mut broadcast_player_runtime_video_payloads_paths = None;
        let mut broadcast_player_runtime_device_boundary_original = None;
        let mut broadcast_player_runtime_device_boundary_paths = None;
        let mut broadcast_player_runtime_test_presenter_original = None;
        let mut broadcast_player_runtime_test_presenter_paths = None;
        let mut broadcast_player_runtime_test_audio_sink_original = None;
        let mut broadcast_player_runtime_test_audio_sink_paths = None;
        let mut broadcast_player_runtime_simulate_original = None;
        let mut broadcast_player_runtime_simulate_paths = None;
        let mut broadcast_player_runtime_original_video_payloads_original = None;
        let mut broadcast_player_runtime_original_video_payloads_paths = None;
        let mut broadcast_player_runtime_verification_original = None;
        let mut broadcast_player_runtime_verification_paths = None;
        let mut qnc_prepared_input_descriptor_original = None;
        let mut qnc_prepared_input_descriptor_paths = None;
        let mut qgs_qnc_prepared_input_mapping_original = None;
        let mut qgs_qnc_prepared_input_mapping_paths = None;
        let mut qgs_qnc_prepared_input_fixture = None;
        let mut qgs_input_plan_original = None;
        let mut qgs_input_plan_paths = None;
        let mut qgs_transport_engine_parity_original = None;
        let mut qgs_transport_engine_parity_paths = None;
        let mut qgs_frame_clock_parity_original = None;
        let mut qgs_frame_clock_parity_paths = None;
        let mut qgs_playout_buffer_tick_original = None;
        let mut qgs_playout_buffer_tick_paths = None;
        let mut qgs_runtime_lifecycle_events_original = None;
        let mut qgs_runtime_lifecycle_events_paths = None;
        let mut qgs_qnc_event_projection_original = None;
        let mut qgs_qnc_event_projection_paths = None;
        let mut qgs_session_command_boundary_original = None;
        let mut qgs_session_command_boundary_paths = None;
        let mut qgs_session_runtime_control_original = None;
        let mut qgs_session_runtime_control_paths = None;
        let mut qgs_presenter_monitor_boundary_original = None;
        let mut qgs_presenter_monitor_boundary_paths = None;
        let mut qgs_file_presenter_diagnostic_original = None;
        let mut qgs_file_presenter_diagnostic_paths = None;
        let mut qgs_gpu_payload_readback_diagnostic_original = None;
        let mut qgs_gpu_payload_readback_diagnostic_paths = None;
        let mut qgs_runtime_surface_e2e_original = None;
        let mut qgs_runtime_surface_e2e_paths = None;
        let mut qgs_broadcast_player_control_core_original = None;
        let mut qgs_broadcast_player_control_core_paths = None;
        let mut qgs_broadcast_player_operational_runtime_original = None;
        let mut qgs_broadcast_player_operational_runtime_paths = None;
        let mut qgs_broadcast_player_fault_recovery_original = None;
        let mut qgs_broadcast_player_fault_recovery_paths = None;
        let mut qgs_broadcast_player_run_original = None;
        let mut qgs_broadcast_player_run_paths = None;
        let mut qgs_broadcast_player_run_options = QgsBroadcastPlayerRunOptions::default();
        let mut qgs_broadcast_player_live_original = None;
        let mut qgs_broadcast_player_live_paths = None;
        let mut qgs_broadcast_player_live_options = QgsBroadcastPlayerLiveOptions::default();
        let mut qgs_broadcast_player_control_session_original = None;
        let mut qgs_broadcast_player_control_session_paths = None;
        let mut qgs_broadcast_player_control_session_script =
            QGS_BROADCAST_PLAYER_CONTROL_SESSION_DEFAULT_SCRIPT.to_string();
        let mut qgs_device_backend_selection_policy = None;
        let mut linux_audio_device_probe_path = None;
        let mut pipewire_audio_prototype_path = None;
        let mut pipewire_audio_native_prototype_path = None;
        let mut pipewire_audio_audible_smoke_test_path = None;
        let mut pipewire_audio_segment_playback_path = None;
        let mut broadcast_runtime_audio_pipewire_original = None;
        let mut broadcast_runtime_audio_pipewire_paths = None;
        let mut broadcast_runtime_audio_pipewire_audible_original = None;
        let mut broadcast_runtime_audio_pipewire_audible_paths = None;
        let mut pipewire_audio_content_audit_original = None;
        let mut pipewire_audio_content_audit_paths = None;
        let mut pipewire_audio_mono_monitor_original = None;
        let mut pipewire_audio_mono_monitor_paths = None;
        let mut pipewire_audio_discrete_4mono_original = None;
        let mut pipewire_audio_discrete_4mono_paths = None;
        let mut diagnostic_wav_path = None;
        let mut audio_content_audit_start_ms = None;
        let mut audio_content_audit_duration_ms = None;
        let mut audio_content_audit_output_dir = None;
        let mut audio_content_audit_find_loudest_range_ms = None;
        let mut audio_monitor_track = None;
        let mut next_arg_is_proxy_original = false;
        let mut next_arg_is_proxy_proxy = false;
        let mut next_arg_is_proxy_throughput_original = false;
        let mut next_arg_is_proxy_throughput_proxy = false;
        let mut next_arg_is_proxy_playback_original = false;
        let mut next_arg_is_proxy_playback_proxy = false;
        let mut next_arg_is_proxy_playback_profile = false;
        let mut next_arg_is_qnc_journalist_demo_original = false;
        let mut next_arg_is_qnc_journalist_demo_proxy = false;
        let mut next_arg_is_original_audio_extract_path = false;
        let mut next_arg_is_broadcast_runtime_contract_original = false;
        let mut next_arg_is_broadcast_runtime_contract_proxy = false;
        let mut next_arg_is_broadcast_runtime_state_machine_original = false;
        let mut next_arg_is_broadcast_runtime_state_machine_proxy = false;
        let mut next_arg_is_broadcast_runtime_preroll_original = false;
        let mut next_arg_is_broadcast_runtime_preroll_proxy = false;
        let mut next_arg_is_broadcast_runtime_prepared_slots_original = false;
        let mut next_arg_is_broadcast_runtime_prepared_slots_proxy = false;
        let mut next_arg_is_broadcast_player_runtime_events_original = false;
        let mut next_arg_is_broadcast_player_runtime_events_proxy = false;
        let mut next_arg_is_broadcast_player_runtime_payloads_original = false;
        let mut next_arg_is_broadcast_player_runtime_payloads_proxy = false;
        let mut next_arg_is_broadcast_player_runtime_video_payloads_original = false;
        let mut next_arg_is_broadcast_player_runtime_video_payloads_proxy = false;
        let mut next_arg_is_broadcast_player_runtime_device_boundary_original = false;
        let mut next_arg_is_broadcast_player_runtime_device_boundary_proxy = false;
        let mut next_arg_is_broadcast_player_runtime_test_presenter_original = false;
        let mut next_arg_is_broadcast_player_runtime_test_presenter_proxy = false;
        let mut next_arg_is_broadcast_player_runtime_test_audio_sink_original = false;
        let mut next_arg_is_broadcast_player_runtime_test_audio_sink_proxy = false;
        let mut next_arg_is_broadcast_player_runtime_simulate_original = false;
        let mut next_arg_is_broadcast_player_runtime_simulate_proxy = false;
        let mut next_arg_is_broadcast_player_runtime_original_video_payloads_original = false;
        let mut next_arg_is_broadcast_player_runtime_original_video_payloads_proxy = false;
        let mut next_arg_is_broadcast_player_runtime_verification_original = false;
        let mut next_arg_is_broadcast_player_runtime_verification_proxy = false;
        let mut next_arg_is_qnc_prepared_input_descriptor_original = false;
        let mut next_arg_is_qnc_prepared_input_descriptor_proxy = false;
        let mut next_arg_is_qgs_qnc_prepared_input_mapping_original = false;
        let mut next_arg_is_qgs_qnc_prepared_input_mapping_proxy = false;
        let mut next_arg_is_qgs_qnc_prepared_input_fixture = false;
        let mut next_arg_is_qgs_input_plan_original = false;
        let mut next_arg_is_qgs_input_plan_proxy = false;
        let mut next_arg_is_qgs_transport_engine_parity_original = false;
        let mut next_arg_is_qgs_transport_engine_parity_proxy = false;
        let mut next_arg_is_qgs_frame_clock_parity_original = false;
        let mut next_arg_is_qgs_frame_clock_parity_proxy = false;
        let mut next_arg_is_qgs_playout_buffer_tick_original = false;
        let mut next_arg_is_qgs_playout_buffer_tick_proxy = false;
        let mut next_arg_is_qgs_runtime_lifecycle_events_original = false;
        let mut next_arg_is_qgs_runtime_lifecycle_events_proxy = false;
        let mut next_arg_is_qgs_qnc_event_projection_original = false;
        let mut next_arg_is_qgs_qnc_event_projection_proxy = false;
        let mut next_arg_is_qgs_session_command_boundary_original = false;
        let mut next_arg_is_qgs_session_command_boundary_proxy = false;
        let mut next_arg_is_qgs_session_runtime_control_original = false;
        let mut next_arg_is_qgs_session_runtime_control_proxy = false;
        let mut next_arg_is_qgs_presenter_monitor_boundary_original = false;
        let mut next_arg_is_qgs_presenter_monitor_boundary_proxy = false;
        let mut next_arg_is_qgs_file_presenter_diagnostic_original = false;
        let mut next_arg_is_qgs_file_presenter_diagnostic_proxy = false;
        let mut next_arg_is_qgs_gpu_payload_readback_diagnostic_original = false;
        let mut next_arg_is_qgs_gpu_payload_readback_diagnostic_proxy = false;
        let mut next_arg_is_qgs_runtime_surface_e2e_original = false;
        let mut next_arg_is_qgs_runtime_surface_e2e_proxy = false;
        let mut next_arg_is_qgs_broadcast_player_control_core_original = false;
        let mut next_arg_is_qgs_broadcast_player_control_core_proxy = false;
        let mut next_arg_is_qgs_broadcast_player_operational_runtime_original = false;
        let mut next_arg_is_qgs_broadcast_player_operational_runtime_proxy = false;
        let mut next_arg_is_qgs_broadcast_player_fault_recovery_original = false;
        let mut next_arg_is_qgs_broadcast_player_fault_recovery_proxy = false;
        let mut next_arg_is_qgs_broadcast_player_run_original = false;
        let mut next_arg_is_qgs_broadcast_player_run_proxy = false;
        let mut next_arg_is_qgs_broadcast_player_live_original = false;
        let mut next_arg_is_qgs_broadcast_player_live_proxy = false;
        let mut next_arg_is_qgs_broadcast_player_run_frames = false;
        let mut next_arg_is_qgs_broadcast_player_live_start_frame = false;
        let mut next_arg_is_qgs_broadcast_player_live_status_every = false;
        let mut next_arg_is_qgs_broadcast_player_live_max_frames = false;
        let mut next_arg_is_qgs_broadcast_player_live_pace = false;
        let mut next_arg_is_qgs_broadcast_player_live_video_output = false;
        let mut next_arg_is_qgs_broadcast_player_live_audio_output = false;
        let mut next_arg_is_qgs_broadcast_player_live_output_dir = false;
        let mut next_arg_is_qgs_broadcast_player_live_preview_every = false;
        let mut next_arg_is_qgs_broadcast_player_run_seek_frame = false;
        let mut next_arg_is_qgs_broadcast_player_run_view = false;
        let mut next_arg_is_qgs_broadcast_player_control_session_original = false;
        let mut next_arg_is_qgs_broadcast_player_control_session_proxy = false;
        let mut next_arg_is_qgs_broadcast_player_control_session_script = false;
        let mut next_arg_is_qgs_device_backend_selection_policy = false;
        let mut next_arg_is_linux_audio_device_probe_path = false;
        let mut next_arg_is_pipewire_audio_prototype_path = false;
        let mut next_arg_is_pipewire_audio_native_prototype_path = false;
        let mut next_arg_is_pipewire_audio_audible_smoke_test_path = false;
        let mut next_arg_is_pipewire_audio_segment_playback_path = false;
        let mut next_arg_is_broadcast_runtime_audio_pipewire_original = false;
        let mut next_arg_is_broadcast_runtime_audio_pipewire_proxy = false;
        let mut next_arg_is_broadcast_runtime_audio_pipewire_audible_original = false;
        let mut next_arg_is_broadcast_runtime_audio_pipewire_audible_proxy = false;
        let mut next_arg_is_pipewire_audio_content_audit_original = false;
        let mut next_arg_is_pipewire_audio_content_audit_proxy = false;
        let mut next_arg_is_pipewire_audio_mono_monitor_original = false;
        let mut next_arg_is_pipewire_audio_mono_monitor_proxy = false;
        let mut next_arg_is_pipewire_audio_discrete_4mono_original = false;
        let mut next_arg_is_pipewire_audio_discrete_4mono_proxy = false;
        let mut next_arg_is_diagnostic_wav_path = false;
        let mut next_arg_is_audio_content_audit_start_ms = false;
        let mut next_arg_is_audio_content_audit_duration_ms = false;
        let mut next_arg_is_audio_content_audit_output_dir = false;
        let mut next_arg_is_audio_content_audit_find_loudest_range_ms = false;
        let mut next_arg_is_audio_monitor_track = false;

        for arg in std::env::args_os().skip(1) {
            if next_arg_is_qgs_broadcast_player_run_frames {
                qgs_broadcast_player_run_options.frames =
                    parse_u64_os_arg(&arg, QGS_BROADCAST_PLAYER_RUN_FRAMES_ARG).max(1);
                next_arg_is_qgs_broadcast_player_run_frames = false;
            } else if next_arg_is_qgs_broadcast_player_live_start_frame {
                qgs_broadcast_player_live_options.start_frame =
                    parse_u64_os_arg(&arg, QGS_BROADCAST_PLAYER_LIVE_START_FRAME_ARG);
                next_arg_is_qgs_broadcast_player_live_start_frame = false;
            } else if next_arg_is_qgs_broadcast_player_live_status_every {
                qgs_broadcast_player_live_options.status_every =
                    parse_u64_os_arg(&arg, QGS_BROADCAST_PLAYER_LIVE_STATUS_EVERY_ARG).max(1);
                next_arg_is_qgs_broadcast_player_live_status_every = false;
            } else if next_arg_is_qgs_broadcast_player_live_max_frames {
                qgs_broadcast_player_live_options.max_frames =
                    Some(parse_u64_os_arg(&arg, QGS_BROADCAST_PLAYER_LIVE_MAX_FRAMES_ARG).max(1));
                next_arg_is_qgs_broadcast_player_live_max_frames = false;
            } else if next_arg_is_qgs_broadcast_player_live_pace {
                qgs_broadcast_player_live_options.pace = QgsBroadcastPlayerLivePace::parse(&arg)
                    .unwrap_or_else(|err| panic!("{QGS_BROADCAST_PLAYER_LIVE_PACE_ARG}: {err}"));
                next_arg_is_qgs_broadcast_player_live_pace = false;
            } else if next_arg_is_qgs_broadcast_player_live_video_output {
                qgs_broadcast_player_live_options.video_output =
                    QgsBroadcastPlayerLiveVideoOutputMode::parse(&arg).unwrap_or_else(|err| {
                        panic!("{QGS_BROADCAST_PLAYER_LIVE_VIDEO_OUTPUT_ARG}: {err}")
                    });
                next_arg_is_qgs_broadcast_player_live_video_output = false;
            } else if next_arg_is_qgs_broadcast_player_live_audio_output {
                qgs_broadcast_player_live_options.audio_output =
                    QgsBroadcastPlayerLiveAudioOutputMode::parse(&arg).unwrap_or_else(|err| {
                        panic!("{QGS_BROADCAST_PLAYER_LIVE_AUDIO_OUTPUT_ARG}: {err}")
                    });
                next_arg_is_qgs_broadcast_player_live_audio_output = false;
            } else if next_arg_is_qgs_broadcast_player_live_output_dir {
                qgs_broadcast_player_live_options.output_dir = PathBuf::from(arg);
                next_arg_is_qgs_broadcast_player_live_output_dir = false;
            } else if next_arg_is_qgs_broadcast_player_live_preview_every {
                qgs_broadcast_player_live_options.preview_every =
                    parse_u64_os_arg(&arg, QGS_BROADCAST_PLAYER_LIVE_PREVIEW_EVERY_ARG).max(1);
                next_arg_is_qgs_broadcast_player_live_preview_every = false;
            } else if next_arg_is_qgs_broadcast_player_run_seek_frame {
                let seek_frame = parse_u64_os_arg(&arg, QGS_BROADCAST_PLAYER_RUN_SEEK_FRAME_ARG);
                qgs_broadcast_player_run_options.seek_frame = Some(seek_frame);
                qgs_broadcast_player_live_options.seek_frame = Some(seek_frame);
                next_arg_is_qgs_broadcast_player_run_seek_frame = false;
            } else if next_arg_is_qgs_broadcast_player_run_view {
                let view = QgsBroadcastPlayerRunView::parse(&arg)
                    .unwrap_or_else(|err| panic!("{QGS_BROADCAST_PLAYER_RUN_VIEW_ARG}: {err}"));
                qgs_broadcast_player_run_options.view = view;
                qgs_broadcast_player_live_options.view = view;
                next_arg_is_qgs_broadcast_player_run_view = false;
            } else if next_arg_is_qgs_broadcast_player_control_session_script {
                qgs_broadcast_player_control_session_script = arg.to_string_lossy().to_string();
                next_arg_is_qgs_broadcast_player_control_session_script = false;
            } else if next_arg_is_qgs_broadcast_player_control_session_proxy {
                let proxy = PathBuf::from(arg);
                let original = qgs_broadcast_player_control_session_original
                    .take()
                    .unwrap_or_else(|| PathBuf::from(""));
                qgs_broadcast_player_control_session_paths = Some((original, proxy));
                next_arg_is_qgs_broadcast_player_control_session_proxy = false;
            } else if next_arg_is_qgs_broadcast_player_control_session_original {
                qgs_broadcast_player_control_session_original = Some(PathBuf::from(arg));
                next_arg_is_qgs_broadcast_player_control_session_original = false;
                next_arg_is_qgs_broadcast_player_control_session_proxy = true;
            } else if next_arg_is_qgs_broadcast_player_live_proxy {
                let proxy = PathBuf::from(arg);
                let original = qgs_broadcast_player_live_original
                    .take()
                    .unwrap_or_else(|| PathBuf::from(""));
                qgs_broadcast_player_live_paths = Some((original, proxy));
                next_arg_is_qgs_broadcast_player_live_proxy = false;
            } else if next_arg_is_qgs_broadcast_player_live_original {
                qgs_broadcast_player_live_original = Some(PathBuf::from(arg));
                next_arg_is_qgs_broadcast_player_live_original = false;
                next_arg_is_qgs_broadcast_player_live_proxy = true;
            } else if next_arg_is_qgs_broadcast_player_run_proxy {
                let proxy = PathBuf::from(arg);
                let original = qgs_broadcast_player_run_original
                    .take()
                    .unwrap_or_else(|| PathBuf::from(""));
                qgs_broadcast_player_run_paths = Some((original, proxy));
                next_arg_is_qgs_broadcast_player_run_proxy = false;
            } else if next_arg_is_qgs_broadcast_player_run_original {
                qgs_broadcast_player_run_original = Some(PathBuf::from(arg));
                next_arg_is_qgs_broadcast_player_run_original = false;
                next_arg_is_qgs_broadcast_player_run_proxy = true;
            } else if next_arg_is_qgs_broadcast_player_fault_recovery_proxy {
                let proxy = PathBuf::from(arg);
                let original = qgs_broadcast_player_fault_recovery_original
                    .take()
                    .unwrap_or_else(|| PathBuf::from(""));
                qgs_broadcast_player_fault_recovery_paths = Some((original, proxy));
                next_arg_is_qgs_broadcast_player_fault_recovery_proxy = false;
            } else if next_arg_is_qgs_broadcast_player_fault_recovery_original {
                qgs_broadcast_player_fault_recovery_original = Some(PathBuf::from(arg));
                next_arg_is_qgs_broadcast_player_fault_recovery_original = false;
                next_arg_is_qgs_broadcast_player_fault_recovery_proxy = true;
            } else if next_arg_is_qgs_device_backend_selection_policy {
                qgs_device_backend_selection_policy = Some(
                    parse_device_selection_policy_arg(&arg)
                        .unwrap_or_else(|err| panic!("{QGS_DEVICE_BACKEND_SELECTION_ARG}: {err}")),
                );
                next_arg_is_qgs_device_backend_selection_policy = false;
            } else if next_arg_is_audio_monitor_track {
                audio_monitor_track = Some(
                    parse_monitor_track_arg(&arg)
                        .unwrap_or_else(|err| panic!("{AUDIO_MONITOR_TRACK_ARG}: {err}")),
                );
                next_arg_is_audio_monitor_track = false;
            } else if next_arg_is_diagnostic_wav_path {
                diagnostic_wav_path = Some(PathBuf::from(arg));
                next_arg_is_diagnostic_wav_path = false;
            } else if next_arg_is_audio_content_audit_start_ms {
                audio_content_audit_start_ms =
                    Some(parse_u64_os_arg(&arg, AUDIO_AUDIT_START_MS_ARG));
                next_arg_is_audio_content_audit_start_ms = false;
            } else if next_arg_is_audio_content_audit_duration_ms {
                audio_content_audit_duration_ms =
                    Some(parse_u64_os_arg(&arg, AUDIO_AUDIT_DURATION_MS_ARG));
                next_arg_is_audio_content_audit_duration_ms = false;
            } else if next_arg_is_audio_content_audit_output_dir {
                audio_content_audit_output_dir = Some(PathBuf::from(arg));
                next_arg_is_audio_content_audit_output_dir = false;
            } else if next_arg_is_audio_content_audit_find_loudest_range_ms {
                audio_content_audit_find_loudest_range_ms = Some(parse_u64_os_arg(
                    &arg,
                    AUDIO_AUDIT_FIND_LOUDEST_RANGE_MS_ARG,
                ));
                next_arg_is_audio_content_audit_find_loudest_range_ms = false;
            } else if next_arg_is_pipewire_audio_mono_monitor_proxy {
                let proxy = PathBuf::from(arg);
                let original = pipewire_audio_mono_monitor_original
                    .take()
                    .unwrap_or_else(|| PathBuf::from(""));
                pipewire_audio_mono_monitor_paths = Some((original, proxy));
                next_arg_is_pipewire_audio_mono_monitor_proxy = false;
            } else if next_arg_is_pipewire_audio_mono_monitor_original {
                pipewire_audio_mono_monitor_original = Some(PathBuf::from(arg));
                next_arg_is_pipewire_audio_mono_monitor_original = false;
                next_arg_is_pipewire_audio_mono_monitor_proxy = true;
            } else if next_arg_is_pipewire_audio_discrete_4mono_proxy {
                let proxy = PathBuf::from(arg);
                let original = pipewire_audio_discrete_4mono_original
                    .take()
                    .unwrap_or_else(|| PathBuf::from(""));
                pipewire_audio_discrete_4mono_paths = Some((original, proxy));
                next_arg_is_pipewire_audio_discrete_4mono_proxy = false;
            } else if next_arg_is_pipewire_audio_discrete_4mono_original {
                pipewire_audio_discrete_4mono_original = Some(PathBuf::from(arg));
                next_arg_is_pipewire_audio_discrete_4mono_original = false;
                next_arg_is_pipewire_audio_discrete_4mono_proxy = true;
            } else if next_arg_is_pipewire_audio_content_audit_proxy {
                let proxy = PathBuf::from(arg);
                let original = pipewire_audio_content_audit_original
                    .take()
                    .unwrap_or_else(|| PathBuf::from(""));
                pipewire_audio_content_audit_paths = Some((original, proxy));
                next_arg_is_pipewire_audio_content_audit_proxy = false;
            } else if next_arg_is_pipewire_audio_content_audit_original {
                pipewire_audio_content_audit_original = Some(PathBuf::from(arg));
                next_arg_is_pipewire_audio_content_audit_original = false;
                next_arg_is_pipewire_audio_content_audit_proxy = true;
            } else if next_arg_is_broadcast_runtime_audio_pipewire_audible_proxy {
                let proxy = PathBuf::from(arg);
                let original = broadcast_runtime_audio_pipewire_audible_original
                    .take()
                    .unwrap_or_else(|| PathBuf::from(""));
                broadcast_runtime_audio_pipewire_audible_paths = Some((original, proxy));
                next_arg_is_broadcast_runtime_audio_pipewire_audible_proxy = false;
            } else if next_arg_is_broadcast_runtime_audio_pipewire_audible_original {
                broadcast_runtime_audio_pipewire_audible_original = Some(PathBuf::from(arg));
                next_arg_is_broadcast_runtime_audio_pipewire_audible_original = false;
                next_arg_is_broadcast_runtime_audio_pipewire_audible_proxy = true;
            } else if next_arg_is_broadcast_runtime_audio_pipewire_proxy {
                let proxy = PathBuf::from(arg);
                let original = broadcast_runtime_audio_pipewire_original
                    .take()
                    .unwrap_or_else(|| PathBuf::from(""));
                broadcast_runtime_audio_pipewire_paths = Some((original, proxy));
                next_arg_is_broadcast_runtime_audio_pipewire_proxy = false;
            } else if next_arg_is_broadcast_runtime_audio_pipewire_original {
                broadcast_runtime_audio_pipewire_original = Some(PathBuf::from(arg));
                next_arg_is_broadcast_runtime_audio_pipewire_original = false;
                next_arg_is_broadcast_runtime_audio_pipewire_proxy = true;
            } else if next_arg_is_pipewire_audio_segment_playback_path {
                pipewire_audio_segment_playback_path = Some(PathBuf::from(arg));
                next_arg_is_pipewire_audio_segment_playback_path = false;
            } else if next_arg_is_pipewire_audio_audible_smoke_test_path {
                pipewire_audio_audible_smoke_test_path = Some(PathBuf::from(arg));
                next_arg_is_pipewire_audio_audible_smoke_test_path = false;
            } else if next_arg_is_pipewire_audio_native_prototype_path {
                pipewire_audio_native_prototype_path = Some(PathBuf::from(arg));
                next_arg_is_pipewire_audio_native_prototype_path = false;
            } else if next_arg_is_pipewire_audio_prototype_path {
                pipewire_audio_prototype_path = Some(PathBuf::from(arg));
                next_arg_is_pipewire_audio_prototype_path = false;
            } else if next_arg_is_qgs_runtime_surface_e2e_proxy {
                let proxy = PathBuf::from(arg);
                let original = qgs_runtime_surface_e2e_original
                    .take()
                    .unwrap_or_else(|| PathBuf::from(""));
                qgs_runtime_surface_e2e_paths = Some((original, proxy));
                next_arg_is_qgs_runtime_surface_e2e_proxy = false;
            } else if next_arg_is_qgs_runtime_surface_e2e_original {
                qgs_runtime_surface_e2e_original = Some(PathBuf::from(arg));
                next_arg_is_qgs_runtime_surface_e2e_original = false;
                next_arg_is_qgs_runtime_surface_e2e_proxy = true;
            } else if next_arg_is_qgs_file_presenter_diagnostic_proxy {
                let proxy = PathBuf::from(arg);
                let original = qgs_file_presenter_diagnostic_original
                    .take()
                    .unwrap_or_else(|| PathBuf::from(""));
                qgs_file_presenter_diagnostic_paths = Some((original, proxy));
                next_arg_is_qgs_file_presenter_diagnostic_proxy = false;
            } else if next_arg_is_qgs_file_presenter_diagnostic_original {
                qgs_file_presenter_diagnostic_original = Some(PathBuf::from(arg));
                next_arg_is_qgs_file_presenter_diagnostic_original = false;
                next_arg_is_qgs_file_presenter_diagnostic_proxy = true;
            } else if next_arg_is_qgs_gpu_payload_readback_diagnostic_proxy {
                let proxy = PathBuf::from(arg);
                let original = qgs_gpu_payload_readback_diagnostic_original
                    .take()
                    .unwrap_or_else(|| PathBuf::from(""));
                qgs_gpu_payload_readback_diagnostic_paths = Some((original, proxy));
                next_arg_is_qgs_gpu_payload_readback_diagnostic_proxy = false;
            } else if next_arg_is_qgs_gpu_payload_readback_diagnostic_original {
                qgs_gpu_payload_readback_diagnostic_original = Some(PathBuf::from(arg));
                next_arg_is_qgs_gpu_payload_readback_diagnostic_original = false;
                next_arg_is_qgs_gpu_payload_readback_diagnostic_proxy = true;
            } else if next_arg_is_qgs_broadcast_player_control_core_proxy {
                let proxy = PathBuf::from(arg);
                let original = qgs_broadcast_player_control_core_original
                    .take()
                    .unwrap_or_else(|| PathBuf::from(""));
                qgs_broadcast_player_control_core_paths = Some((original, proxy));
                next_arg_is_qgs_broadcast_player_control_core_proxy = false;
            } else if next_arg_is_qgs_broadcast_player_control_core_original {
                qgs_broadcast_player_control_core_original = Some(PathBuf::from(arg));
                next_arg_is_qgs_broadcast_player_control_core_original = false;
                next_arg_is_qgs_broadcast_player_control_core_proxy = true;
            } else if next_arg_is_qgs_broadcast_player_operational_runtime_proxy {
                let proxy = PathBuf::from(arg);
                let original = qgs_broadcast_player_operational_runtime_original
                    .take()
                    .unwrap_or_else(|| PathBuf::from(""));
                qgs_broadcast_player_operational_runtime_paths = Some((original, proxy));
                next_arg_is_qgs_broadcast_player_operational_runtime_proxy = false;
            } else if next_arg_is_qgs_broadcast_player_operational_runtime_original {
                qgs_broadcast_player_operational_runtime_original = Some(PathBuf::from(arg));
                next_arg_is_qgs_broadcast_player_operational_runtime_original = false;
                next_arg_is_qgs_broadcast_player_operational_runtime_proxy = true;
            } else if next_arg_is_qgs_presenter_monitor_boundary_proxy {
                let proxy = PathBuf::from(arg);
                let original = qgs_presenter_monitor_boundary_original
                    .take()
                    .unwrap_or_else(|| PathBuf::from(""));
                qgs_presenter_monitor_boundary_paths = Some((original, proxy));
                next_arg_is_qgs_presenter_monitor_boundary_proxy = false;
            } else if next_arg_is_qgs_presenter_monitor_boundary_original {
                qgs_presenter_monitor_boundary_original = Some(PathBuf::from(arg));
                next_arg_is_qgs_presenter_monitor_boundary_original = false;
                next_arg_is_qgs_presenter_monitor_boundary_proxy = true;
            } else if next_arg_is_qgs_session_runtime_control_proxy {
                let proxy = PathBuf::from(arg);
                let original = qgs_session_runtime_control_original
                    .take()
                    .unwrap_or_else(|| PathBuf::from(""));
                qgs_session_runtime_control_paths = Some((original, proxy));
                next_arg_is_qgs_session_runtime_control_proxy = false;
            } else if next_arg_is_qgs_session_runtime_control_original {
                qgs_session_runtime_control_original = Some(PathBuf::from(arg));
                next_arg_is_qgs_session_runtime_control_original = false;
                next_arg_is_qgs_session_runtime_control_proxy = true;
            } else if next_arg_is_qgs_input_plan_proxy {
                let proxy = PathBuf::from(arg);
                let original = qgs_input_plan_original
                    .take()
                    .unwrap_or_else(|| PathBuf::from(""));
                qgs_input_plan_paths = Some((original, proxy));
                next_arg_is_qgs_input_plan_proxy = false;
            } else if next_arg_is_qgs_input_plan_original {
                qgs_input_plan_original = Some(PathBuf::from(arg));
                next_arg_is_qgs_input_plan_original = false;
                next_arg_is_qgs_input_plan_proxy = true;
            } else if next_arg_is_qgs_transport_engine_parity_proxy {
                let proxy = PathBuf::from(arg);
                let original = qgs_transport_engine_parity_original
                    .take()
                    .unwrap_or_else(|| PathBuf::from(""));
                qgs_transport_engine_parity_paths = Some((original, proxy));
                next_arg_is_qgs_transport_engine_parity_proxy = false;
            } else if next_arg_is_qgs_transport_engine_parity_original {
                qgs_transport_engine_parity_original = Some(PathBuf::from(arg));
                next_arg_is_qgs_transport_engine_parity_original = false;
                next_arg_is_qgs_transport_engine_parity_proxy = true;
            } else if next_arg_is_qgs_frame_clock_parity_proxy {
                let proxy = PathBuf::from(arg);
                let original = qgs_frame_clock_parity_original
                    .take()
                    .unwrap_or_else(|| PathBuf::from(""));
                qgs_frame_clock_parity_paths = Some((original, proxy));
                next_arg_is_qgs_frame_clock_parity_proxy = false;
            } else if next_arg_is_qgs_frame_clock_parity_original {
                qgs_frame_clock_parity_original = Some(PathBuf::from(arg));
                next_arg_is_qgs_frame_clock_parity_original = false;
                next_arg_is_qgs_frame_clock_parity_proxy = true;
            } else if next_arg_is_qgs_playout_buffer_tick_proxy {
                let proxy = PathBuf::from(arg);
                let original = qgs_playout_buffer_tick_original
                    .take()
                    .unwrap_or_else(|| PathBuf::from(""));
                qgs_playout_buffer_tick_paths = Some((original, proxy));
                next_arg_is_qgs_playout_buffer_tick_proxy = false;
            } else if next_arg_is_qgs_playout_buffer_tick_original {
                qgs_playout_buffer_tick_original = Some(PathBuf::from(arg));
                next_arg_is_qgs_playout_buffer_tick_original = false;
                next_arg_is_qgs_playout_buffer_tick_proxy = true;
            } else if next_arg_is_qgs_runtime_lifecycle_events_proxy {
                let proxy = PathBuf::from(arg);
                let original = qgs_runtime_lifecycle_events_original
                    .take()
                    .unwrap_or_else(|| PathBuf::from(""));
                qgs_runtime_lifecycle_events_paths = Some((original, proxy));
                next_arg_is_qgs_runtime_lifecycle_events_proxy = false;
            } else if next_arg_is_qgs_runtime_lifecycle_events_original {
                qgs_runtime_lifecycle_events_original = Some(PathBuf::from(arg));
                next_arg_is_qgs_runtime_lifecycle_events_original = false;
                next_arg_is_qgs_runtime_lifecycle_events_proxy = true;
            } else if next_arg_is_qgs_qnc_event_projection_proxy {
                let proxy = PathBuf::from(arg);
                let original = qgs_qnc_event_projection_original
                    .take()
                    .unwrap_or_else(|| PathBuf::from(""));
                qgs_qnc_event_projection_paths = Some((original, proxy));
                next_arg_is_qgs_qnc_event_projection_proxy = false;
            } else if next_arg_is_qgs_qnc_event_projection_original {
                qgs_qnc_event_projection_original = Some(PathBuf::from(arg));
                next_arg_is_qgs_qnc_event_projection_original = false;
                next_arg_is_qgs_qnc_event_projection_proxy = true;
            } else if next_arg_is_qgs_session_command_boundary_proxy {
                let proxy = PathBuf::from(arg);
                let original = qgs_session_command_boundary_original
                    .take()
                    .unwrap_or_else(|| PathBuf::from(""));
                qgs_session_command_boundary_paths = Some((original, proxy));
                next_arg_is_qgs_session_command_boundary_proxy = false;
            } else if next_arg_is_qgs_session_command_boundary_original {
                qgs_session_command_boundary_original = Some(PathBuf::from(arg));
                next_arg_is_qgs_session_command_boundary_original = false;
                next_arg_is_qgs_session_command_boundary_proxy = true;
            } else if next_arg_is_qnc_prepared_input_descriptor_proxy {
                let proxy = PathBuf::from(arg);
                let original = qnc_prepared_input_descriptor_original
                    .take()
                    .unwrap_or_else(|| PathBuf::from(""));
                qnc_prepared_input_descriptor_paths = Some((original, proxy));
                next_arg_is_qnc_prepared_input_descriptor_proxy = false;
            } else if next_arg_is_qnc_prepared_input_descriptor_original {
                qnc_prepared_input_descriptor_original = Some(PathBuf::from(arg));
                next_arg_is_qnc_prepared_input_descriptor_original = false;
                next_arg_is_qnc_prepared_input_descriptor_proxy = true;
            } else if next_arg_is_qgs_qnc_prepared_input_mapping_proxy {
                let proxy = PathBuf::from(arg);
                let original = qgs_qnc_prepared_input_mapping_original
                    .take()
                    .unwrap_or_else(|| PathBuf::from(""));
                qgs_qnc_prepared_input_mapping_paths = Some((original, proxy));
                next_arg_is_qgs_qnc_prepared_input_mapping_proxy = false;
            } else if next_arg_is_qgs_qnc_prepared_input_fixture {
                qgs_qnc_prepared_input_fixture = Some(arg.to_string_lossy().into_owned());
                next_arg_is_qgs_qnc_prepared_input_fixture = false;
            } else if next_arg_is_qgs_qnc_prepared_input_mapping_original {
                qgs_qnc_prepared_input_mapping_original = Some(PathBuf::from(arg));
                next_arg_is_qgs_qnc_prepared_input_mapping_original = false;
                next_arg_is_qgs_qnc_prepared_input_mapping_proxy = true;
            } else if next_arg_is_linux_audio_device_probe_path {
                linux_audio_device_probe_path = Some(PathBuf::from(arg));
                next_arg_is_linux_audio_device_probe_path = false;
            } else if next_arg_is_broadcast_player_runtime_verification_proxy {
                let proxy = PathBuf::from(arg);
                let original = broadcast_player_runtime_verification_original
                    .take()
                    .unwrap_or_else(|| PathBuf::from(""));
                broadcast_player_runtime_verification_paths = Some((original, proxy));
                next_arg_is_broadcast_player_runtime_verification_proxy = false;
            } else if next_arg_is_broadcast_player_runtime_verification_original {
                broadcast_player_runtime_verification_original = Some(PathBuf::from(arg));
                next_arg_is_broadcast_player_runtime_verification_original = false;
                next_arg_is_broadcast_player_runtime_verification_proxy = true;
            } else if next_arg_is_broadcast_player_runtime_original_video_payloads_proxy {
                let proxy = PathBuf::from(arg);
                let original = broadcast_player_runtime_original_video_payloads_original
                    .take()
                    .unwrap_or_else(|| PathBuf::from(""));
                broadcast_player_runtime_original_video_payloads_paths = Some((original, proxy));
                next_arg_is_broadcast_player_runtime_original_video_payloads_proxy = false;
            } else if next_arg_is_broadcast_player_runtime_original_video_payloads_original {
                broadcast_player_runtime_original_video_payloads_original =
                    Some(PathBuf::from(arg));
                next_arg_is_broadcast_player_runtime_original_video_payloads_original = false;
                next_arg_is_broadcast_player_runtime_original_video_payloads_proxy = true;
            } else if next_arg_is_broadcast_player_runtime_simulate_proxy {
                let proxy = PathBuf::from(arg);
                let original = broadcast_player_runtime_simulate_original
                    .take()
                    .unwrap_or_else(|| PathBuf::from(""));
                broadcast_player_runtime_simulate_paths = Some((original, proxy));
                next_arg_is_broadcast_player_runtime_simulate_proxy = false;
            } else if next_arg_is_broadcast_player_runtime_simulate_original {
                broadcast_player_runtime_simulate_original = Some(PathBuf::from(arg));
                next_arg_is_broadcast_player_runtime_simulate_original = false;
                next_arg_is_broadcast_player_runtime_simulate_proxy = true;
            } else if next_arg_is_broadcast_player_runtime_test_audio_sink_proxy {
                let proxy = PathBuf::from(arg);
                let original = broadcast_player_runtime_test_audio_sink_original
                    .take()
                    .unwrap_or_else(|| PathBuf::from(""));
                broadcast_player_runtime_test_audio_sink_paths = Some((original, proxy));
                next_arg_is_broadcast_player_runtime_test_audio_sink_proxy = false;
            } else if next_arg_is_broadcast_player_runtime_test_audio_sink_original {
                broadcast_player_runtime_test_audio_sink_original = Some(PathBuf::from(arg));
                next_arg_is_broadcast_player_runtime_test_audio_sink_original = false;
                next_arg_is_broadcast_player_runtime_test_audio_sink_proxy = true;
            } else if next_arg_is_broadcast_player_runtime_test_presenter_proxy {
                let proxy = PathBuf::from(arg);
                let original = broadcast_player_runtime_test_presenter_original
                    .take()
                    .unwrap_or_else(|| PathBuf::from(""));
                broadcast_player_runtime_test_presenter_paths = Some((original, proxy));
                next_arg_is_broadcast_player_runtime_test_presenter_proxy = false;
            } else if next_arg_is_broadcast_player_runtime_test_presenter_original {
                broadcast_player_runtime_test_presenter_original = Some(PathBuf::from(arg));
                next_arg_is_broadcast_player_runtime_test_presenter_original = false;
                next_arg_is_broadcast_player_runtime_test_presenter_proxy = true;
            } else if next_arg_is_broadcast_player_runtime_device_boundary_proxy {
                let proxy = PathBuf::from(arg);
                let original = broadcast_player_runtime_device_boundary_original
                    .take()
                    .unwrap_or_else(|| PathBuf::from(""));
                broadcast_player_runtime_device_boundary_paths = Some((original, proxy));
                next_arg_is_broadcast_player_runtime_device_boundary_proxy = false;
            } else if next_arg_is_broadcast_player_runtime_device_boundary_original {
                broadcast_player_runtime_device_boundary_original = Some(PathBuf::from(arg));
                next_arg_is_broadcast_player_runtime_device_boundary_original = false;
                next_arg_is_broadcast_player_runtime_device_boundary_proxy = true;
            } else if next_arg_is_broadcast_player_runtime_video_payloads_proxy {
                let proxy = PathBuf::from(arg);
                let original = broadcast_player_runtime_video_payloads_original
                    .take()
                    .unwrap_or_else(|| PathBuf::from(""));
                broadcast_player_runtime_video_payloads_paths = Some((original, proxy));
                next_arg_is_broadcast_player_runtime_video_payloads_proxy = false;
            } else if next_arg_is_broadcast_player_runtime_video_payloads_original {
                broadcast_player_runtime_video_payloads_original = Some(PathBuf::from(arg));
                next_arg_is_broadcast_player_runtime_video_payloads_original = false;
                next_arg_is_broadcast_player_runtime_video_payloads_proxy = true;
            } else if next_arg_is_broadcast_player_runtime_payloads_proxy {
                let proxy = PathBuf::from(arg);
                let original = broadcast_player_runtime_payloads_original
                    .take()
                    .unwrap_or_else(|| PathBuf::from(""));
                broadcast_player_runtime_payloads_paths = Some((original, proxy));
                next_arg_is_broadcast_player_runtime_payloads_proxy = false;
            } else if next_arg_is_broadcast_player_runtime_payloads_original {
                broadcast_player_runtime_payloads_original = Some(PathBuf::from(arg));
                next_arg_is_broadcast_player_runtime_payloads_original = false;
                next_arg_is_broadcast_player_runtime_payloads_proxy = true;
            } else if next_arg_is_broadcast_player_runtime_events_proxy {
                let proxy = PathBuf::from(arg);
                let original = broadcast_player_runtime_events_original
                    .take()
                    .unwrap_or_else(|| PathBuf::from(""));
                broadcast_player_runtime_events_paths = Some((original, proxy));
                next_arg_is_broadcast_player_runtime_events_proxy = false;
            } else if next_arg_is_broadcast_player_runtime_events_original {
                broadcast_player_runtime_events_original = Some(PathBuf::from(arg));
                next_arg_is_broadcast_player_runtime_events_original = false;
                next_arg_is_broadcast_player_runtime_events_proxy = true;
            } else if next_arg_is_broadcast_runtime_prepared_slots_proxy {
                let proxy = PathBuf::from(arg);
                let original = broadcast_runtime_prepared_slots_original
                    .take()
                    .unwrap_or_else(|| PathBuf::from(""));
                broadcast_runtime_prepared_slots_paths = Some((original, proxy));
                next_arg_is_broadcast_runtime_prepared_slots_proxy = false;
            } else if next_arg_is_broadcast_runtime_prepared_slots_original {
                broadcast_runtime_prepared_slots_original = Some(PathBuf::from(arg));
                next_arg_is_broadcast_runtime_prepared_slots_original = false;
                next_arg_is_broadcast_runtime_prepared_slots_proxy = true;
            } else if next_arg_is_broadcast_runtime_preroll_proxy {
                let proxy = PathBuf::from(arg);
                let original = broadcast_runtime_preroll_original
                    .take()
                    .unwrap_or_else(|| PathBuf::from(""));
                broadcast_runtime_preroll_paths = Some((original, proxy));
                next_arg_is_broadcast_runtime_preroll_proxy = false;
            } else if next_arg_is_broadcast_runtime_preroll_original {
                broadcast_runtime_preroll_original = Some(PathBuf::from(arg));
                next_arg_is_broadcast_runtime_preroll_original = false;
                next_arg_is_broadcast_runtime_preroll_proxy = true;
            } else if next_arg_is_broadcast_runtime_state_machine_proxy {
                let proxy = PathBuf::from(arg);
                let original = broadcast_runtime_state_machine_original
                    .take()
                    .unwrap_or_else(|| PathBuf::from(""));
                broadcast_runtime_state_machine_paths = Some((original, proxy));
                next_arg_is_broadcast_runtime_state_machine_proxy = false;
            } else if next_arg_is_broadcast_runtime_state_machine_original {
                broadcast_runtime_state_machine_original = Some(PathBuf::from(arg));
                next_arg_is_broadcast_runtime_state_machine_original = false;
                next_arg_is_broadcast_runtime_state_machine_proxy = true;
            } else if next_arg_is_broadcast_runtime_contract_proxy {
                let proxy = PathBuf::from(arg);
                let original = broadcast_runtime_contract_original
                    .take()
                    .unwrap_or_else(|| PathBuf::from(""));
                broadcast_runtime_contract_paths = Some((original, proxy));
                next_arg_is_broadcast_runtime_contract_proxy = false;
            } else if next_arg_is_broadcast_runtime_contract_original {
                broadcast_runtime_contract_original = Some(PathBuf::from(arg));
                next_arg_is_broadcast_runtime_contract_original = false;
                next_arg_is_broadcast_runtime_contract_proxy = true;
            } else if next_arg_is_original_audio_extract_path {
                original_audio_extract_path = Some(PathBuf::from(arg));
                next_arg_is_original_audio_extract_path = false;
            } else if next_arg_is_qnc_journalist_demo_proxy {
                let proxy = PathBuf::from(arg);
                let original = qnc_journalist_demo_original
                    .take()
                    .unwrap_or_else(|| PathBuf::from(""));
                qnc_journalist_demo_paths = Some((original, proxy));
                next_arg_is_qnc_journalist_demo_proxy = false;
            } else if next_arg_is_qnc_journalist_demo_original {
                qnc_journalist_demo_original = Some(PathBuf::from(arg));
                next_arg_is_qnc_journalist_demo_original = false;
                next_arg_is_qnc_journalist_demo_proxy = true;
            } else if next_arg_is_proxy_playback_profile {
                proxy_playback_profile =
                    ProxyPlaybackProfile::parse(&arg).unwrap_or_else(|err| panic!("{err}"));
                next_arg_is_proxy_playback_profile = false;
            } else if next_arg_is_proxy_playback_proxy {
                let proxy = PathBuf::from(arg);
                let original = proxy_playback_original
                    .take()
                    .unwrap_or_else(|| PathBuf::from(""));
                proxy_playback_paths = Some((original, proxy));
                next_arg_is_proxy_playback_proxy = false;
            } else if next_arg_is_proxy_playback_original {
                proxy_playback_original = Some(PathBuf::from(arg));
                next_arg_is_proxy_playback_original = false;
                next_arg_is_proxy_playback_proxy = true;
            } else if next_arg_is_proxy_throughput_proxy {
                let proxy = PathBuf::from(arg);
                let original = proxy_throughput_original
                    .take()
                    .unwrap_or_else(|| PathBuf::from(""));
                proxy_throughput_paths = Some((original, proxy));
                next_arg_is_proxy_throughput_proxy = false;
            } else if next_arg_is_proxy_throughput_original {
                proxy_throughput_original = Some(PathBuf::from(arg));
                next_arg_is_proxy_throughput_original = false;
                next_arg_is_proxy_throughput_proxy = true;
            } else if next_arg_is_proxy_proxy {
                let proxy = PathBuf::from(arg);
                let original = proxy_proof_original
                    .take()
                    .unwrap_or_else(|| PathBuf::from(""));
                proxy_proof_paths = Some((original, proxy));
                next_arg_is_proxy_proxy = false;
            } else if next_arg_is_proxy_original {
                proxy_proof_original = Some(PathBuf::from(arg));
                next_arg_is_proxy_original = false;
                next_arg_is_proxy_proxy = true;
            } else if next_arg_is_software_gpu_mxf_path {
                software_gpu_mxf_path = Some(PathBuf::from(arg));
                next_arg_is_software_gpu_mxf_path = false;
            } else if next_arg_is_native_original_video_decode_path {
                native_original_video_decode_path = Some(PathBuf::from(arg));
                next_arg_is_native_original_video_decode_path = false;
            } else if next_arg_is_software_decode_mxf_path {
                software_decode_mxf_path = Some(PathBuf::from(arg));
                next_arg_is_software_decode_mxf_path = false;
            } else if next_arg_is_mxf_path {
                mxf_inspect_path = Some(PathBuf::from(arg));
                next_arg_is_mxf_path = false;
            } else if arg == VIDEO_CAPABILITIES_ONLY_ARG {
                video_capabilities_only = true;
            } else if arg == H264_DECODE_ONLY_ARG {
                h264_decode_only = true;
            } else if arg == HASWELL_VIDEO_DIAGNOSTIC_ARG {
                haswell_video_diagnostic = true;
            } else if arg == MXF_INSPECT_ARG {
                mxf_inspect_path = Some(PathBuf::from(DEFAULT_MXF_FIXTURE));
                next_arg_is_mxf_path = true;
            } else if arg == SOFTWARE_DECODE_MXF_ARG {
                next_arg_is_software_decode_mxf_path = true;
            } else if arg == SOFTWARE_GPU_MXF_ARG {
                next_arg_is_software_gpu_mxf_path = true;
            } else if arg == NATIVE_ORIGINAL_VIDEO_DECODE_ARG {
                next_arg_is_native_original_video_decode_path = true;
            } else if arg == PROXY_PROOF_ARG {
                next_arg_is_proxy_original = true;
            } else if arg == PROXY_THROUGHPUT_ARG {
                next_arg_is_proxy_throughput_original = true;
            } else if arg == PROXY_PLAYBACK_ARG {
                next_arg_is_proxy_playback_original = true;
            } else if arg == PROXY_PLAYBACK_PROFILE_ARG {
                next_arg_is_proxy_playback_profile = true;
            } else if arg == QNC_JOURNALIST_DEMO_ARG {
                next_arg_is_qnc_journalist_demo_original = true;
            } else if arg == ORIGINAL_AUDIO_EXTRACT_ARG {
                next_arg_is_original_audio_extract_path = true;
            } else if arg == BROADCAST_PLAYER_RUNTIME_CONTRACT_ARG
                || arg == LEGACY_BROADCAST_RUNTIME_CONTRACT_ARG
            {
                next_arg_is_broadcast_runtime_contract_original = true;
            } else if arg == BROADCAST_PLAYER_RUNTIME_STATE_MACHINE_ARG
                || arg == LEGACY_BROADCAST_RUNTIME_STATE_MACHINE_ARG
            {
                next_arg_is_broadcast_runtime_state_machine_original = true;
            } else if arg == BROADCAST_PLAYER_RUNTIME_PREROLL_ARG
                || arg == LEGACY_BROADCAST_RUNTIME_PREROLL_ARG
            {
                next_arg_is_broadcast_runtime_preroll_original = true;
            } else if arg == BROADCAST_PLAYER_RUNTIME_PREPARED_SLOTS_ARG
                || arg == LEGACY_BROADCAST_RUNTIME_PREPARED_SLOTS_ARG
            {
                next_arg_is_broadcast_runtime_prepared_slots_original = true;
            } else if arg == BROADCAST_PLAYER_RUNTIME_EVENTS_ARG {
                next_arg_is_broadcast_player_runtime_events_original = true;
            } else if arg == BROADCAST_PLAYER_RUNTIME_PAYLOADS_ARG {
                next_arg_is_broadcast_player_runtime_payloads_original = true;
            } else if arg == BROADCAST_PLAYER_RUNTIME_VIDEO_PAYLOADS_ARG {
                next_arg_is_broadcast_player_runtime_video_payloads_original = true;
            } else if arg == BROADCAST_PLAYER_RUNTIME_DEVICE_BOUNDARY_ARG {
                next_arg_is_broadcast_player_runtime_device_boundary_original = true;
            } else if arg == BROADCAST_PLAYER_RUNTIME_TEST_PRESENTER_ARG {
                next_arg_is_broadcast_player_runtime_test_presenter_original = true;
            } else if arg == BROADCAST_PLAYER_RUNTIME_TEST_AUDIO_SINK_ARG {
                next_arg_is_broadcast_player_runtime_test_audio_sink_original = true;
            } else if arg == BROADCAST_PLAYER_RUNTIME_SIMULATE_ARG {
                next_arg_is_broadcast_player_runtime_simulate_original = true;
            } else if arg == BROADCAST_PLAYER_RUNTIME_ORIGINAL_VIDEO_PAYLOADS_ARG {
                next_arg_is_broadcast_player_runtime_original_video_payloads_original = true;
            } else if arg == BROADCAST_PLAYER_RUNTIME_VERIFICATION_ARG {
                next_arg_is_broadcast_player_runtime_verification_original = true;
            } else if arg == QNC_PREPARED_INPUT_DESCRIPTOR_ARG {
                next_arg_is_qnc_prepared_input_descriptor_original = true;
            } else if arg == QGS_QNC_PREPARED_INPUT_MAPPING_ARG {
                next_arg_is_qgs_qnc_prepared_input_mapping_original = true;
            } else if arg == QGS_QNC_PREPARED_INPUT_FIXTURE_ARG {
                next_arg_is_qgs_qnc_prepared_input_fixture = true;
            } else if arg == QGS_INPUT_PLAN_ARG {
                next_arg_is_qgs_input_plan_original = true;
            } else if arg == QGS_TRANSPORT_ENGINE_PARITY_ARG {
                next_arg_is_qgs_transport_engine_parity_original = true;
            } else if arg == QGS_FRAME_CLOCK_PARITY_ARG {
                next_arg_is_qgs_frame_clock_parity_original = true;
            } else if arg == QGS_PLAYOUT_BUFFER_TICK_ARG {
                next_arg_is_qgs_playout_buffer_tick_original = true;
            } else if arg == QGS_RUNTIME_LIFECYCLE_EVENTS_ARG {
                next_arg_is_qgs_runtime_lifecycle_events_original = true;
            } else if arg == QGS_QNC_EVENT_PROJECTION_ARG {
                next_arg_is_qgs_qnc_event_projection_original = true;
            } else if arg == QGS_SESSION_COMMAND_BOUNDARY_ARG {
                next_arg_is_qgs_session_command_boundary_original = true;
            } else if arg == QGS_SESSION_RUNTIME_CONTROL_ARG {
                next_arg_is_qgs_session_runtime_control_original = true;
            } else if arg == QGS_PRESENTER_MONITOR_BOUNDARY_ARG {
                next_arg_is_qgs_presenter_monitor_boundary_original = true;
            } else if arg == QGS_FILE_PRESENTER_DIAGNOSTIC_ARG {
                next_arg_is_qgs_file_presenter_diagnostic_original = true;
            } else if arg == QGS_GPU_PAYLOAD_READBACK_DIAGNOSTIC_ARG {
                next_arg_is_qgs_gpu_payload_readback_diagnostic_original = true;
            } else if arg == QGS_RUNTIME_SURFACE_E2E_ARG {
                next_arg_is_qgs_runtime_surface_e2e_original = true;
            } else if arg == QGS_BROADCAST_PLAYER_CONTROL_CORE_ARG {
                next_arg_is_qgs_broadcast_player_control_core_original = true;
            } else if arg == QGS_BROADCAST_PLAYER_OPERATIONAL_RUNTIME_ARG {
                next_arg_is_qgs_broadcast_player_operational_runtime_original = true;
            } else if arg == QGS_BROADCAST_PLAYER_FAULT_RECOVERY_ARG {
                next_arg_is_qgs_broadcast_player_fault_recovery_original = true;
            } else if arg == QGS_BROADCAST_PLAYER_LIVE_ARG {
                next_arg_is_qgs_broadcast_player_live_original = true;
            } else if arg == QGS_BROADCAST_PLAYER_RUN_ARG {
                next_arg_is_qgs_broadcast_player_run_original = true;
            } else if arg == QGS_BROADCAST_PLAYER_CONTROL_SESSION_ARG {
                next_arg_is_qgs_broadcast_player_control_session_original = true;
            } else if arg == QGS_BROADCAST_PLAYER_RUN_FRAMES_ARG {
                next_arg_is_qgs_broadcast_player_run_frames = true;
            } else if arg == QGS_BROADCAST_PLAYER_LIVE_START_FRAME_ARG {
                next_arg_is_qgs_broadcast_player_live_start_frame = true;
            } else if arg == QGS_BROADCAST_PLAYER_LIVE_STATUS_EVERY_ARG {
                next_arg_is_qgs_broadcast_player_live_status_every = true;
            } else if arg == QGS_BROADCAST_PLAYER_LIVE_MAX_FRAMES_ARG {
                next_arg_is_qgs_broadcast_player_live_max_frames = true;
            } else if arg == QGS_BROADCAST_PLAYER_LIVE_PACE_ARG {
                next_arg_is_qgs_broadcast_player_live_pace = true;
            } else if arg == QGS_BROADCAST_PLAYER_LIVE_NO_INTERACTIVE_ARG {
                qgs_broadcast_player_live_options.no_interactive = true;
            } else if arg == QGS_BROADCAST_PLAYER_LIVE_VIDEO_OUTPUT_ARG {
                next_arg_is_qgs_broadcast_player_live_video_output = true;
            } else if arg == QGS_BROADCAST_PLAYER_LIVE_AUDIO_OUTPUT_ARG {
                next_arg_is_qgs_broadcast_player_live_audio_output = true;
            } else if arg == QGS_BROADCAST_PLAYER_LIVE_OUTPUT_DIR_ARG {
                next_arg_is_qgs_broadcast_player_live_output_dir = true;
            } else if arg == QGS_BROADCAST_PLAYER_LIVE_PREVIEW_EVERY_ARG {
                next_arg_is_qgs_broadcast_player_live_preview_every = true;
            } else if arg == QGS_BROADCAST_PLAYER_RUN_SEEK_FRAME_ARG {
                next_arg_is_qgs_broadcast_player_run_seek_frame = true;
            } else if arg == QGS_BROADCAST_PLAYER_RUN_VIEW_ARG {
                next_arg_is_qgs_broadcast_player_run_view = true;
            } else if arg == QGS_BROADCAST_PLAYER_CONTROL_SCRIPT_ARG {
                next_arg_is_qgs_broadcast_player_control_session_script = true;
            } else if arg == QGS_DEVICE_BACKEND_SELECTION_ARG {
                next_arg_is_qgs_device_backend_selection_policy = true;
            } else if arg == LINUX_AUDIO_DEVICE_PROBE_ARG {
                next_arg_is_linux_audio_device_probe_path = true;
            } else if arg == PIPEWIRE_AUDIO_PROTOTYPE_ARG {
                next_arg_is_pipewire_audio_prototype_path = true;
            } else if arg == PIPEWIRE_AUDIO_NATIVE_PROTOTYPE_ARG {
                next_arg_is_pipewire_audio_native_prototype_path = true;
            } else if arg == PIPEWIRE_AUDIO_AUDIBLE_SMOKE_TEST_ARG {
                next_arg_is_pipewire_audio_audible_smoke_test_path = true;
            } else if arg == PIPEWIRE_AUDIO_SEGMENT_PLAYBACK_ARG {
                next_arg_is_pipewire_audio_segment_playback_path = true;
            } else if arg == BROADCAST_RUNTIME_AUDIO_PIPEWIRE_ARG {
                next_arg_is_broadcast_runtime_audio_pipewire_original = true;
            } else if arg == BROADCAST_RUNTIME_AUDIO_PIPEWIRE_AUDIBLE_ARG {
                next_arg_is_broadcast_runtime_audio_pipewire_audible_original = true;
            } else if arg == PIPEWIRE_AUDIO_CONTENT_AUDIT_ARG {
                next_arg_is_pipewire_audio_content_audit_original = true;
            } else if arg == PIPEWIRE_AUDIO_MONO_MONITOR_ARG {
                next_arg_is_pipewire_audio_mono_monitor_original = true;
            } else if arg == PIPEWIRE_AUDIO_DISCRETE_4MONO_ARG {
                next_arg_is_pipewire_audio_discrete_4mono_original = true;
            } else if arg == WRITE_DIAGNOSTIC_WAV_ARG {
                next_arg_is_diagnostic_wav_path = true;
            } else if arg == AUDIO_AUDIT_START_MS_ARG {
                next_arg_is_audio_content_audit_start_ms = true;
            } else if arg == AUDIO_AUDIT_DURATION_MS_ARG {
                next_arg_is_audio_content_audit_duration_ms = true;
            } else if arg == AUDIO_AUDIT_OUTPUT_DIR_ARG {
                next_arg_is_audio_content_audit_output_dir = true;
            } else if arg == AUDIO_AUDIT_FIND_LOUDEST_RANGE_MS_ARG {
                next_arg_is_audio_content_audit_find_loudest_range_ms = true;
            } else if arg == AUDIO_MONITOR_TRACK_ARG {
                next_arg_is_audio_monitor_track = true;
            } else if socket_path.is_none() {
                socket_path = Some(PathBuf::from(arg));
            }
        }

        Self {
            socket_path: socket_path.unwrap_or_else(default_socket_path),
            video_capabilities_only,
            h264_decode_only,
            haswell_video_diagnostic,
            mxf_inspect_path,
            software_decode_mxf_path,
            software_gpu_mxf_path,
            native_original_video_decode_path,
            proxy_proof_paths,
            proxy_throughput_paths,
            proxy_playback_paths,
            proxy_playback_profile,
            qnc_journalist_demo_paths,
            original_audio_extract_path,
            broadcast_runtime_contract_paths,
            broadcast_runtime_state_machine_paths,
            broadcast_runtime_preroll_paths,
            broadcast_runtime_prepared_slots_paths,
            broadcast_player_runtime_events_paths,
            broadcast_player_runtime_payloads_paths,
            broadcast_player_runtime_video_payloads_paths,
            broadcast_player_runtime_device_boundary_paths,
            broadcast_player_runtime_test_presenter_paths,
            broadcast_player_runtime_test_audio_sink_paths,
            broadcast_player_runtime_simulate_paths,
            broadcast_player_runtime_original_video_payloads_paths,
            broadcast_player_runtime_verification_paths,
            qnc_prepared_input_descriptor_paths,
            qgs_qnc_prepared_input_mapping_paths,
            qgs_qnc_prepared_input_fixture,
            qgs_input_plan_paths,
            qgs_transport_engine_parity_paths,
            qgs_frame_clock_parity_paths,
            qgs_playout_buffer_tick_paths,
            qgs_runtime_lifecycle_events_paths,
            qgs_qnc_event_projection_paths,
            qgs_session_command_boundary_paths,
            qgs_session_runtime_control_paths,
            qgs_presenter_monitor_boundary_paths,
            qgs_file_presenter_diagnostic_paths,
            qgs_gpu_payload_readback_diagnostic_paths,
            qgs_runtime_surface_e2e_paths,
            qgs_broadcast_player_control_core_paths,
            qgs_broadcast_player_operational_runtime_paths,
            qgs_broadcast_player_fault_recovery_paths,
            qgs_broadcast_player_run_paths,
            qgs_broadcast_player_run_options,
            qgs_broadcast_player_live_paths,
            qgs_broadcast_player_live_options,
            qgs_broadcast_player_control_session_paths,
            qgs_broadcast_player_control_session_script,
            qgs_device_backend_selection_policy,
            linux_audio_device_probe_path,
            pipewire_audio_prototype_path,
            pipewire_audio_native_prototype_path,
            pipewire_audio_audible_smoke_test_path,
            pipewire_audio_segment_playback_path,
            broadcast_runtime_audio_pipewire_paths,
            broadcast_runtime_audio_pipewire_audible_paths,
            pipewire_audio_content_audit_paths,
            pipewire_audio_mono_monitor_paths,
            pipewire_audio_discrete_4mono_paths,
            diagnostic_wav_path,
            audio_content_audit_start_ms,
            audio_content_audit_duration_ms,
            audio_content_audit_output_dir,
            audio_content_audit_find_loudest_range_ms,
            audio_monitor_track,
        }
    }
}

fn parse_u64_os_arg(value: &std::ffi::OsStr, flag: &str) -> u64 {
    value
        .to_string_lossy()
        .parse::<u64>()
        .unwrap_or_else(|_| panic!("{flag} expects an unsigned integer"))
}

fn parse_u16_os_arg(value: &std::ffi::OsStr, flag: &str) -> u16 {
    value
        .to_string_lossy()
        .parse::<u16>()
        .unwrap_or_else(|_| panic!("{flag} expects an unsigned integer"))
}

fn parse_monitor_track_arg(value: &std::ffi::OsStr) -> Result<u16, &'static str> {
    let track = parse_u16_os_arg(value, AUDIO_MONITOR_TRACK_ARG);
    if track == 0 {
        return Err("expects a 1-based original mono track number");
    }
    Ok(track - 1)
}

fn parse_device_selection_policy_arg(
    value: &std::ffi::OsStr,
) -> Result<QgsDeviceSelectionPolicy, &'static str> {
    match value.to_string_lossy().as_ref() {
        "diagnostic-only" => Ok(QgsDeviceSelectionPolicy::DiagnosticOnly),
        "preview-qnc-os" => Ok(QgsDeviceSelectionPolicy::PreviewOnQncOs),
        "original-qnc-os" => Ok(QgsDeviceSelectionPolicy::OriginalMediaOnQncOs),
        "headless-ci" => Ok(QgsDeviceSelectionPolicy::HeadlessCi),
        "future-appliance-direct" => Ok(QgsDeviceSelectionPolicy::FutureApplianceDirect),
        _ => Err(
            "expects one of diagnostic-only, preview-qnc-os, original-qnc-os, headless-ci, future-appliance-direct",
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        audible_smoke_evidence_label, audio_audit_total_samples_from_index,
        audio_audit_track_summaries, bounded_audio_audit_duration_ms,
        build_audio_audit_f32_interleaved_range, build_original_pcm_blocks_from_index_range,
        build_pipewire_f32_interleaved_prototype_buffer,
        build_runtime_audio_payload_pipewire_buffer,
        build_sequential_original_audio_segment_for_duration, decode_s24be_i32, decode_s24le_i32,
        decode_u24le, default_qgs_input_plan_queue_requirements, discrete_4mono_evidence_label,
        duration_samples_to_frames, f32_buffer_stats, file_presenter_manifest_file_name,
        gpu_payload_readback_image_file_name, gpu_payload_readback_manifest_file_name, json_escape,
        loudest_monitor_pair, manifest_json_contains_private_path,
        mono_monitor_diagnostic_evidence_label, mxf_file_label, original_pcm_blocks_by_channel,
        original_segment_evidence_label, parse_device_selection_policy_arg,
        parse_manual_audible_confirmation, parse_monitor_track_arg, pcm_s24le_sample_to_f32,
        proxy_presentation_ordinals, public_source_uri_stem, qgs_format_media_time,
        qgs_resolve_run_seek_frame, qgs_running_prepare_input_meaning, qgs_running_preroll_meaning,
        qgs_running_runtime_persistent_warning_lines, repeated_smoke_test_buffers,
        rgba_u16_to_ppm_p6_rgb8, rgba_u16_to_rgb8_bytes, runtime_audio_payload_evidence_label,
        sanitized_source_stem, selected_proxy_ordinals, smoke_test_buffer_count,
        ManualAudibleConfirmation, QgsBroadcastPlayerLiveAudioOutputMode,
        QgsBroadcastPlayerLiveInputCommand, QgsBroadcastPlayerLiveOptions,
        QgsBroadcastPlayerLivePace, QgsBroadcastPlayerLiveVideoOutputMode,
        QgsBroadcastPlayerRunView, QgsControlSessionScriptCommand, SonyXmlSummary,
    };
    use qgs_broadcast_player::AudioCoverage;
    use qgs_media_runtime::{
        av_frame_audio_range, bind_broadcast_audio_payload, BroadcastPreparedAudioSlot,
        BroadcastVideoSourceMode, PcmAudioBlock, PcmAudioBlockLayout, PcmEndian, PcmSampleFormat,
        QgsBroadcastPlayerStatus, QgsDeviceSelectionPolicy, QgsInputPlanSourceMode,
        QgsPlaybackRepresentation, QgsQncControlDeviceStatus, QgsQncControlEvidenceLevel,
        QgsQncGeneration, QgsQncPreparedWindowView, QgsQncRuntimeSnapshot, RationalRate,
    };
    use qgs_mp4::{Mp4VideoSample, Mp4VideoTrack, Rational};
    use qgs_mxf::{
        AudioEssenceDescriptor, AudioIndexEntry, DescriptorSource, IndexSource, MxfTrack,
        PcmAudioIndex, TrackId, TrackKind,
    };
    use std::path::PathBuf;
    use std::time::Duration;

    #[test]
    fn qgs_input_plan_default_queue_requirements_are_bounded() {
        let requirements = default_qgs_input_plan_queue_requirements();
        assert_eq!(requirements.min_video_frames, 3);
        assert_eq!(requirements.min_audio_ranges, 3);
        assert!(requirements.max_video_queue >= requirements.min_video_frames);
        assert!(requirements.max_audio_queue >= requirements.min_audio_ranges);
    }

    #[test]
    fn qnc_descriptor_duration_samples_map_to_50fps_frames() {
        let frames = duration_samples_to_frames(
            48_000,
            48_000,
            qgs_media_runtime::RationalRate::new(50, 1).unwrap(),
        )
        .unwrap();
        assert_eq!(frames, 50);
    }

    #[test]
    fn sony_xml_summary_extracts_technical_fields_without_private_ids() {
        let xml = r#"
            <Duration value="106"/>
            <LtcChangeTable tcFps="25" halfStep="true"/>
            <VideoFrame videoCodec="AVC50_1920_1080_H422P@L42" captureFps="50.00p" formatFps="50p"/>
            <VideoLayout pixel="1920" numOfVerticalLine="1080" aspectRatio="16:9"/>
            <AudioFormat numOfChannel="4"/>
            <AudioRecPort audioCodec="LPCM24" channel="CH1"/>
            <AudioRecPort audioCodec="LPCM24" channel="CH2"/>
            <Device manufacturer="Sony" modelName="ILME-FX6V" serialNo="PRIVATE-SERIAL"/>
            <Item name="CaptureGammaEquation" value="rec709"/>
            <Item name="CaptureColorPrimaries" value="rec709"/>
            <Item name="CodingEquations" value="rec709"/>
            <TargetMaterial umidRef="PRIVATE-UMID"/>
        "#;

        let summary = SonyXmlSummary::parse(xml);

        assert_eq!(summary.duration, Some(106));
        assert_eq!(
            summary.video_codec.as_deref(),
            Some("AVC50_1920_1080_H422P@L42")
        );
        assert_eq!(summary.capture_fps.as_deref(), Some("50.00p"));
        assert_eq!(summary.format_fps.as_deref(), Some("50p"));
        assert_eq!(summary.width, Some(1920));
        assert_eq!(summary.height, Some(1080));
        assert_eq!(summary.audio_channels, Some(4));
        assert_eq!(summary.audio_codecs, vec!["LPCM24", "LPCM24"]);
        assert_eq!(summary.camera_model.as_deref(), Some("ILME-FX6V"));
        assert!(!summary.has_proxy_metadata);
        assert_eq!(
            summary
                .color_values
                .get("CaptureGammaEquation")
                .map(String::as_str),
            Some("rec709")
        );

        let debug = format!("{summary:?}");
        assert!(!debug.contains("PRIVATE-SERIAL"));
        assert!(!debug.contains("PRIVATE-UMID"));
    }

    #[test]
    fn external_mxf_file_labels_are_redacted() {
        let path = PathBuf::from("/home/example/private-camera/Clip 0001.MXF");

        assert_eq!(mxf_file_label(&path), "<external>");
    }

    #[test]
    fn proxy_presentation_ordinals_sort_by_pts_not_decode_index() {
        let video = Mp4VideoTrack {
            track_id: 1,
            width: 1920,
            height: 1080,
            timescale: 50_000,
            duration_units: 3_000,
            frame_rate: Rational {
                numerator: 50,
                denominator: 1,
            },
            nal_length_size: 4,
            sps_count: 1,
            pps_count: 1,
            samples: vec![sample(0, 0), sample(1, 2_000), sample(2, 1_000)],
        };

        let ordinals = proxy_presentation_ordinals(&video).expect("ordinals");

        assert_eq!(ordinals.get(&0), Some(&0));
        assert_eq!(ordinals.get(&2), Some(&1));
        assert_eq!(ordinals.get(&1), Some(&2));
    }

    #[test]
    fn selected_proxy_ordinals_are_bounded_and_deduplicated() {
        assert_eq!(selected_proxy_ordinals(0), Vec::<usize>::new());
        assert_eq!(selected_proxy_ordinals(1), vec![0]);
        assert_eq!(selected_proxy_ordinals(12), vec![0, 11]);
        assert_eq!(selected_proxy_ordinals(106), vec![0, 53, 105]);
    }

    #[test]
    fn running_runtime_output_disambiguates_prepare_and_preroll_phases() {
        assert!(
            qgs_running_prepare_input_meaning().contains("waiting for cue and preroll"),
            "prepare input wording should explain why status may still be Loaded"
        );
        assert!(
            qgs_running_preroll_meaning().contains("prepared window"),
            "preroll wording should identify the ready prepared-window phase"
        );
    }

    #[test]
    fn running_runtime_persistent_warnings_are_single_section_facts() {
        let warnings = qgs_running_runtime_persistent_warning_lines();
        assert_eq!(warnings.len(), 6);
        assert!(warnings
            .iter()
            .any(|warning| warning.contains("RealDisplayUnavailable")));
        assert!(warnings
            .iter()
            .any(|warning| warning.contains("AudioOutputNotProductionVerified")));
        assert!(warnings
            .iter()
            .any(|warning| warning.contains("AvSyncNotVerified")));
        assert!(warnings.iter().all(|warning| !warning.contains("/home/")));
    }

    #[test]
    fn running_runtime_options_parse_view_and_bound_seek_frame() {
        assert_eq!(
            QgsBroadcastPlayerRunView::parse(std::ffi::OsStr::new("compact")),
            Ok(QgsBroadcastPlayerRunView::Compact)
        );
        assert_eq!(
            QgsBroadcastPlayerRunView::parse(std::ffi::OsStr::new("detailed")),
            Ok(QgsBroadcastPlayerRunView::Detailed)
        );
        assert!(QgsBroadcastPlayerRunView::parse(std::ffi::OsStr::new("wide")).is_err());
        assert_eq!(qgs_resolve_run_seek_frame(None, 106), 50);
        assert_eq!(qgs_resolve_run_seek_frame(Some(50), 106), 50);
        assert_eq!(qgs_resolve_run_seek_frame(Some(999), 106), 105);
        assert_eq!(qgs_resolve_run_seek_frame(None, 20), 10);
    }

    #[test]
    fn live_runtime_options_parse_pace_and_use_safe_defaults() {
        assert_eq!(
            QgsBroadcastPlayerLivePace::parse(std::ffi::OsStr::new("logical")),
            Ok(QgsBroadcastPlayerLivePace::Logical)
        );
        assert_eq!(
            QgsBroadcastPlayerLivePace::parse(std::ffi::OsStr::new("wall")),
            Ok(QgsBroadcastPlayerLivePace::Wall)
        );
        assert!(QgsBroadcastPlayerLivePace::parse(std::ffi::OsStr::new("realtime")).is_err());

        let options = QgsBroadcastPlayerLiveOptions::default();
        assert_eq!(options.start_frame, 0);
        assert_eq!(options.status_every, 10);
        assert_eq!(options.max_frames, Some(500));
        assert_eq!(options.pace, QgsBroadcastPlayerLivePace::Logical);
        assert!(!options.no_interactive);
        assert_eq!(
            options.video_output,
            QgsBroadcastPlayerLiveVideoOutputMode::None
        );
        assert_eq!(
            options.audio_output,
            QgsBroadcastPlayerLiveAudioOutputMode::None
        );
        assert_eq!(options.preview_every, 100);
    }

    #[test]
    fn live_runtime_output_option_parsers_are_conservative() {
        assert_eq!(
            QgsBroadcastPlayerLiveVideoOutputMode::parse(std::ffi::OsStr::new("none")),
            Ok(QgsBroadcastPlayerLiveVideoOutputMode::None)
        );
        assert_eq!(
            QgsBroadcastPlayerLiveVideoOutputMode::parse(std::ffi::OsStr::new("preview-files")),
            Ok(QgsBroadcastPlayerLiveVideoOutputMode::PreviewFiles)
        );
        assert_eq!(
            QgsBroadcastPlayerLiveAudioOutputMode::parse(std::ffi::OsStr::new(
                "pipewire-desktop-monitor"
            )),
            Ok(QgsBroadcastPlayerLiveAudioOutputMode::PipeWireDesktopMonitor)
        );
        assert_eq!(
            QgsBroadcastPlayerLiveAudioOutputMode::parse(std::ffi::OsStr::new("pipewire-monitor")),
            Ok(QgsBroadcastPlayerLiveAudioOutputMode::PipeWireMonitor)
        );
        assert_eq!(
            QgsBroadcastPlayerLiveAudioOutputMode::parse(std::ffi::OsStr::new("pipewire-4mono")),
            Ok(QgsBroadcastPlayerLiveAudioOutputMode::PipeWire4Mono)
        );
        assert!(QgsBroadcastPlayerLiveVideoOutputMode::parse(std::ffi::OsStr::new("x11")).is_err());
        assert!(
            QgsBroadcastPlayerLiveAudioOutputMode::parse(std::ffi::OsStr::new("proxy-aac"))
                .is_err()
        );
    }

    #[test]
    fn live_audio_coverage_resubmits_from_the_new_range_after_seek_reset() {
        let mut coverage = AudioCoverage::new();
        assert_eq!(coverage.next_submit_start(0), Some(0));
        coverage.mark_submitted(0, 24_000);
        assert_eq!(coverage.next_submit_start(960), None);
        assert_eq!(coverage.skip_label(960), Some("covered"));
        assert_ne!(coverage.skip_label(0), Some("playing"));
        coverage.reset();
        assert_eq!(coverage.covered_until_sample, 0);
        assert_eq!(coverage.next_submit_start(0), Some(0));
        assert_eq!(coverage.skip_label(0), None);
        assert!(coverage.submitted);
    }

    #[test]
    fn letterbox_scales_source_pixels_into_the_center() {
        let rgb8 = vec![
            255, 0, 0, //
            0, 255, 0, //
            0, 0, 255, //
            255, 255, 255,
        ];
        let mut buffer = vec![1_u32; 24];
        qgs_broadcast_player::presenter_boundary::blit_rgb8_letterbox(
            2,
            2,
            &rgb8,
            6,
            4,
            &mut buffer,
        );
        assert_eq!(buffer[0], 0);
        assert_eq!(buffer[1], 255 << 16);
        assert_eq!(buffer[2], 255 << 16);
        assert_eq!(buffer[3], 255 << 8);
        assert_eq!(buffer[4], 255 << 8);
        assert_eq!(buffer[5], 0);
        assert_eq!(buffer[13], 255);
        assert_eq!(buffer[15], (255 << 16) | (255 << 8) | 255);
    }

    #[test]
    fn live_preview_frame_names_are_deterministic_and_public_safe() {
        assert_eq!(
            super::qgs_live_preview_frame_file_name(7),
            "frame_000007.ppm"
        );
        assert!(!super::qgs_live_preview_frame_file_name(7).contains("/home/"));
    }

    #[test]
    fn live_preview_latest_json_is_public_safe_and_non_claiming() {
        let snapshot = QgsQncRuntimeSnapshot {
            generation: QgsQncGeneration(3),
            status: QgsBroadcastPlayerStatus::Playing,
            source_loaded: true,
            public_source_uri: Some("qgs://source/live-demo".to_string()),
            source_mode: Some(QgsInputPlanSourceMode::ProxyPreview),
            picture_representation: Some(QgsPlaybackRepresentation::Proxy),
            authoritative_audio_source: "original MXF",
            proxy_aac_authoritative: false,
            broadcast_audio_model: "discrete original MXF mono lanes",
            active_range: Some((0, 100)),
            current_frame: Some(12),
            current_audio_sample_range: Some((11520, 12480)),
            media_time: Some(Duration::from_millis(240)),
            prepared_window: Some(QgsQncPreparedWindowView {
                start_frame: Some(10),
                end_frame_exclusive: Some(18),
                selected_frame: Some(12),
                prepared_frame_count: 8,
            }),
            video_payload_ready: true,
            audio_payload_ready: true,
            transport_ready: true,
            buffer_status: "ready",
            device_status: QgsQncControlDeviceStatus {
                device_policy: "preview-qnc-os",
                selected_video_backend: "file-presenter",
                selected_audio_backend: "pipewire-monitor",
                qnc_os_display_target: "Wayland+Vulkan",
                x11_target_status: "legacy/non-target",
                real_display_status: "NotImplemented",
                presenter_evidence_level: QgsQncControlEvidenceLevel::NotImplemented,
                audio_device_status: "NotProductionVerified",
                visual_verified: false,
                realtime_verified: false,
                audio_device_verified: false,
                av_sync_verified: false,
            },
            active_faults: Vec::new(),
            warnings: Vec::new(),
            private_path_exposed: false,
        };
        let json = super::qgs_live_preview_latest_json(
            12,
            &snapshot,
            RationalRate::new(50, 1).unwrap(),
            "frame_000002.ppm",
            1920,
            1080,
            6_220_817,
            0x1234,
        );
        assert!(json.contains("\"source_mode\": \"ProxyPreview\""));
        assert!(json.contains("\"video_source\": \"proxy MP4\""));
        assert!(json.contains("\"audio_source\": \"original MXF\""));
        let mut original = snapshot.clone();
        original.source_mode = Some(QgsInputPlanSourceMode::OriginalMedia);
        original.picture_representation = Some(QgsPlaybackRepresentation::Original);
        let original_json = super::qgs_live_preview_latest_json(
            12,
            &original,
            RationalRate::new(50, 1).unwrap(),
            "frame_000002.ppm",
            1920,
            1080,
            6_220_817,
            0x1234,
        );
        assert!(original_json.contains("\"source_mode\": \"OriginalMedia\""));
        assert!(original_json.contains("\"video_source\": \"original picture\""));
        assert!(!original_json.contains("proxy MP4"));
        assert_eq!(
            super::qgs_live_engine_picture_status(&original),
            "original-picture"
        );
        assert!(json.contains("\"proxy_aac_authoritative\": false"));
        assert!(json.contains("\"real_display\": \"no\""));
        assert!(json.contains("\"visual_verified\": false"));
        assert!(json.contains("\"private_path_exposed\": false"));
        assert!(!json.contains("/home/"));
        assert!(!json.contains("Mironik"));
    }

    #[test]
    fn live_runtime_input_parser_accepts_operator_controls() {
        assert_eq!(
            QgsBroadcastPlayerLiveInputCommand::parse("p"),
            QgsBroadcastPlayerLiveInputCommand::Pause
        );
        assert_eq!(
            QgsBroadcastPlayerLiveInputCommand::parse("pause"),
            QgsBroadcastPlayerLiveInputCommand::Pause
        );
        assert_eq!(
            QgsBroadcastPlayerLiveInputCommand::parse("resume"),
            QgsBroadcastPlayerLiveInputCommand::Resume
        );
        assert_eq!(
            QgsBroadcastPlayerLiveInputCommand::parse("play"),
            QgsBroadcastPlayerLiveInputCommand::Resume
        );
        assert_eq!(
            QgsBroadcastPlayerLiveInputCommand::parse("s 250"),
            QgsBroadcastPlayerLiveInputCommand::Seek(250)
        );
        assert_eq!(
            QgsBroadcastPlayerLiveInputCommand::parse("seek 50"),
            QgsBroadcastPlayerLiveInputCommand::Seek(50)
        );
        assert_eq!(
            QgsBroadcastPlayerLiveInputCommand::parse("status"),
            QgsBroadcastPlayerLiveInputCommand::Status
        );
        assert_eq!(
            QgsBroadcastPlayerLiveInputCommand::parse("q"),
            QgsBroadcastPlayerLiveInputCommand::Quit
        );
        assert_eq!(
            QgsBroadcastPlayerLiveInputCommand::parse("help"),
            QgsBroadcastPlayerLiveInputCommand::Help
        );
        assert_eq!(
            QgsBroadcastPlayerLiveInputCommand::parse("stop"),
            QgsBroadcastPlayerLiveInputCommand::Stop
        );
        assert_eq!(
            QgsBroadcastPlayerLiveInputCommand::parse("unload"),
            QgsBroadcastPlayerLiveInputCommand::Unload
        );
        assert_eq!(
            QgsBroadcastPlayerLiveInputCommand::parse("   "),
            QgsBroadcastPlayerLiveInputCommand::Noop
        );
        assert!(matches!(
            QgsBroadcastPlayerLiveInputCommand::parse("seek nope"),
            QgsBroadcastPlayerLiveInputCommand::Invalid(_)
        ));
        assert!(matches!(
            QgsBroadcastPlayerLiveInputCommand::parse("launch"),
            QgsBroadcastPlayerLiveInputCommand::Invalid(_)
        ));
    }

    #[test]
    fn live_runtime_no_interactive_option_preserves_logical_loop_defaults() {
        let mut options = QgsBroadcastPlayerLiveOptions::default();
        options.no_interactive = true;

        assert_eq!(options.start_frame, 0);
        assert_eq!(options.status_every, 10);
        assert_eq!(options.max_frames, Some(500));
        assert_eq!(options.pace, QgsBroadcastPlayerLivePace::Logical);
        assert!(options.no_interactive);
    }

    #[test]
    fn live_runtime_media_time_format_is_operator_readable() {
        assert_eq!(
            qgs_format_media_time(Duration::from_millis(0)),
            "00:00:00.000"
        );
        assert_eq!(
            qgs_format_media_time(Duration::from_millis(200)),
            "00:00:00.200"
        );
        assert_eq!(
            qgs_format_media_time(Duration::from_millis(5_000)),
            "00:00:05.000"
        );
        assert_eq!(
            qgs_format_media_time(Duration::from_millis(3_661_234)),
            "01:01:01.234"
        );
    }

    #[test]
    fn control_session_script_parser_accepts_qnc_style_commands() {
        let commands = QgsControlSessionScriptCommand::parse_script(
            "load,prepare,cue:0,preroll,play:5,pause,seek:50,stop,unload,snapshot",
        )
        .unwrap();

        assert_eq!(commands.len(), 10);
        assert_eq!(commands[0], QgsControlSessionScriptCommand::Load);
        assert_eq!(commands[2], QgsControlSessionScriptCommand::Cue(0));
        assert_eq!(commands[4], QgsControlSessionScriptCommand::Play(5));
        assert_eq!(commands[6], QgsControlSessionScriptCommand::Seek(50));
        assert!(QgsControlSessionScriptCommand::parse_script("play:not-a-number").is_err());
        assert!(QgsControlSessionScriptCommand::parse_script("launch").is_err());
    }

    #[test]
    fn pipewire_prototype_converts_s24le_to_f32_interleaved() {
        let blocks = vec![
            mono_block(10, 0, &[[0x00, 0x00, 0x00], [0xff, 0xff, 0x7f]]),
            mono_block(11, 1, &[[0x00, 0x00, 0x80], [0x00, 0x00, 0x40]]),
        ];

        let buffer = build_pipewire_f32_interleaved_prototype_buffer(&blocks, 48_000, 2).unwrap();

        assert_eq!(buffer.channels, 2);
        assert_eq!(buffer.sample_count, 2);
        assert_eq!(buffer.bytes.len(), 2 * 2 * 4);
        let values = buffer
            .bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(values[0], 0.0);
        assert_eq!(values[1], -1.0);
        assert!(values[2] > 0.999_999 && values[2] <= 1.0);
        assert_eq!(values[3], 0.5);
    }

    #[test]
    fn audible_smoke_test_uses_bounded_repeated_original_segment() {
        let count = smoke_test_buffer_count(Duration::from_millis(500), 48_000, 960).unwrap();
        assert_eq!(count, 25);

        let segment = vec![1_u8, 2, 3, 4];
        let buffers = repeated_smoke_test_buffers(&segment, count).unwrap();
        assert_eq!(buffers.len(), 25);
        assert!(buffers.iter().all(|buffer| buffer == &segment));
        assert_eq!(buffers.iter().map(Vec::len).sum::<usize>(), 100);
    }

    #[test]
    fn audible_confirmation_parser_requires_explicit_yes_or_no() {
        assert_eq!(
            parse_manual_audible_confirmation("yes\n"),
            ManualAudibleConfirmation::Yes
        );
        assert_eq!(
            parse_manual_audible_confirmation("n"),
            ManualAudibleConfirmation::No
        );
        assert_eq!(
            parse_manual_audible_confirmation(""),
            ManualAudibleConfirmation::Required
        );
        assert_eq!(
            parse_manual_audible_confirmation("maybe"),
            ManualAudibleConfirmation::Required
        );
    }

    #[test]
    fn audible_smoke_evidence_does_not_upgrade_without_manual_yes() {
        assert_eq!(
            audible_smoke_evidence_label(true, true, ManualAudibleConfirmation::Required),
            "ManualAudibleConfirmationRequired"
        );
        assert_eq!(
            audible_smoke_evidence_label(true, true, ManualAudibleConfirmation::No),
            "ManualAudibleSmokeTestNotHeard"
        );
        assert_eq!(
            audible_smoke_evidence_label(true, true, ManualAudibleConfirmation::Yes),
            "ManualContentAudibilityPartiallyObserved"
        );
    }

    #[test]
    fn sequential_original_audio_segment_uses_timeline_blocks_without_repeating() {
        let mut blocks = Vec::new();
        for block_index in 0..2 {
            for channel in 0..4 {
                let byte = u8::try_from(1 + block_index * 4 + channel).unwrap();
                let samples = [[byte, 0x00, 0x00]; 960];
                blocks.push(mono_block_at(
                    10 + u32::from(channel),
                    channel,
                    u64::from(block_index) * 960,
                    &samples,
                ));
            }
        }

        let plan = build_sequential_original_audio_segment_for_duration(
            &blocks,
            48_000,
            Duration::from_millis(40),
        )
        .unwrap();

        assert_eq!(plan.blocks_per_track, 2);
        assert_eq!(plan.samples_per_buffer, 960);
        assert_eq!(plan.samples_per_track, 1920);
        assert_eq!(plan.buffers.len(), 2);
        assert_eq!(plan.total_bytes, 2 * 960 * 4 * 4);
        assert_eq!(plan.continuity_status, "contiguous/no gaps/no overlaps");
        assert_ne!(plan.buffers[0], plan.buffers[1]);
    }

    #[test]
    fn sequential_original_audio_segment_rejects_gaps() {
        let mut blocks = Vec::new();
        for channel in 0..4 {
            let first = [[0x01, 0x00, 0x00]; 960];
            let second = [[0x02, 0x00, 0x00]; 960];
            blocks.push(mono_block_at(10 + u32::from(channel), channel, 0, &first));
            blocks.push(mono_block_at(
                10 + u32::from(channel),
                channel,
                1920,
                &second,
            ));
        }

        assert!(build_sequential_original_audio_segment_for_duration(
            &blocks,
            48_000,
            Duration::from_millis(40),
        )
        .is_err());
    }

    #[test]
    fn original_segment_evidence_requires_manual_yes_for_audible_confirmation() {
        assert_eq!(
            original_segment_evidence_label(true, true, ManualAudibleConfirmation::Required),
            "ManualOriginalSegmentConfirmationRequired"
        );
        assert_eq!(
            original_segment_evidence_label(true, true, ManualAudibleConfirmation::No),
            "ManualOriginalSegmentNotHeard"
        );
        assert_eq!(
            original_segment_evidence_label(true, true, ManualAudibleConfirmation::Yes),
            "ManualContentAudibilityPartiallyObserved"
        );
    }

    #[test]
    fn runtime_audio_payload_pipewire_buffer_uses_prepared_binding_geometry() {
        let mut blocks = Vec::new();
        for block_index in 0..2 {
            for channel in 0..4 {
                let byte = u8::try_from(1 + block_index * 4 + channel).unwrap();
                let samples = [[byte, 0x00, 0x00]; 960];
                blocks.push(mono_block_at(
                    10 + u32::from(channel),
                    channel,
                    u64::from(block_index) * 960,
                    &samples,
                ));
            }
        }
        let range = av_frame_audio_range(
            0,
            Duration::ZERO,
            Duration::from_millis(40),
            48_000,
            &blocks,
            4,
        )
        .unwrap();
        let slot = BroadcastPreparedAudioSlot::from_audio_range(
            0,
            BroadcastVideoSourceMode::ProxyPreview,
            &range,
        );
        let binding = bind_broadcast_audio_payload(&slot, &blocks).unwrap();

        let buffer = build_runtime_audio_payload_pipewire_buffer(&binding, &blocks).unwrap();

        assert_eq!(binding.sample_count, 1_920);
        assert_eq!(binding.block_coverage.len(), 8);
        assert_eq!(binding.total_referenced_payload_bytes, 23_040);
        assert_eq!(buffer.source_blocks_per_track, 2);
        assert_eq!(buffer.source_blocks_total, 8);
        assert_eq!(buffer.source_bytes, 23_040);
        assert_eq!(buffer.output_frames, 1_920);
        assert_eq!(buffer.track_count, 4);
        assert_eq!(buffer.output_bytes.len(), 30_720);
        assert!(buffer.output_bytes.iter().any(|byte| *byte != 0));
    }

    #[test]
    fn runtime_audio_payload_evidence_stays_scoped() {
        assert_eq!(
            runtime_audio_payload_evidence_label(true, true, ManualAudibleConfirmation::Required),
            "RuntimeAudioPayloadDrainCompleted"
        );
        assert_eq!(
            runtime_audio_payload_evidence_label(true, false, ManualAudibleConfirmation::Required),
            "RuntimeAudioPayloadSubmittedToPipeWire"
        );
        assert_eq!(
            runtime_audio_payload_evidence_label(true, true, ManualAudibleConfirmation::No),
            "RuntimeAudioPayloadNotHeard"
        );
        assert_eq!(
            runtime_audio_payload_evidence_label(true, true, ManualAudibleConfirmation::Yes),
            "RuntimeAudioPayloadAudibleConfirmed"
        );
    }

    #[test]
    fn mono_monitor_evidence_stays_below_audio_device_verified() {
        assert_eq!(
            mono_monitor_diagnostic_evidence_label(true, true, ManualAudibleConfirmation::Required),
            "DesktopMonoListeningHelperDrainCompleted"
        );
        assert_eq!(
            mono_monitor_diagnostic_evidence_label(
                true,
                false,
                ManualAudibleConfirmation::Required
            ),
            "DesktopMonoListeningHelperSubmitted"
        );
        assert_eq!(
            mono_monitor_diagnostic_evidence_label(true, true, ManualAudibleConfirmation::No),
            "ManualDesktopMonoListeningHelperNotHeard"
        );
        assert_eq!(
            mono_monitor_diagnostic_evidence_label(true, true, ManualAudibleConfirmation::Yes),
            "ManualDesktopMonoListeningHelperHeard"
        );
    }

    #[test]
    fn discrete_4mono_evidence_reports_submission_and_drain() {
        assert_eq!(
            discrete_4mono_evidence_label(true, true),
            "Discrete4MonoOutputDrainCompleted"
        );
        assert_eq!(
            discrete_4mono_evidence_label(true, false),
            "Discrete4MonoOutputSubmitted"
        );
        assert_eq!(
            discrete_4mono_evidence_label(false, false),
            "Discrete4MonoOutputNotSubmitted"
        );
    }

    #[test]
    fn mono_monitor_track_parser_uses_one_based_original_track_numbers() {
        assert_eq!(parse_monitor_track_arg("4".as_ref()).unwrap(), 3);
        assert_eq!(parse_monitor_track_arg("1".as_ref()).unwrap(), 0);
        assert!(parse_monitor_track_arg("0".as_ref()).is_err());
    }

    #[test]
    fn device_selection_policy_parser_accepts_block_h_policy_names() {
        assert_eq!(
            parse_device_selection_policy_arg("diagnostic-only".as_ref()).unwrap(),
            QgsDeviceSelectionPolicy::DiagnosticOnly
        );
        assert_eq!(
            parse_device_selection_policy_arg("preview-qnc-os".as_ref()).unwrap(),
            QgsDeviceSelectionPolicy::PreviewOnQncOs
        );
        assert_eq!(
            parse_device_selection_policy_arg("original-qnc-os".as_ref()).unwrap(),
            QgsDeviceSelectionPolicy::OriginalMediaOnQncOs
        );
        assert_eq!(
            parse_device_selection_policy_arg("headless-ci".as_ref()).unwrap(),
            QgsDeviceSelectionPolicy::HeadlessCi
        );
        assert_eq!(
            parse_device_selection_policy_arg("future-appliance-direct".as_ref()).unwrap(),
            QgsDeviceSelectionPolicy::FutureApplianceDirect
        );
        assert!(parse_device_selection_policy_arg("x11".as_ref()).is_err());
    }

    #[test]
    fn runtime_audio_payload_audible_helper_repeats_bounded_payload() {
        let payload = vec![0_u8; 30_720];
        let repeats = smoke_test_buffer_count(Duration::from_millis(500), 48_000, 1_920).unwrap();
        let buffers = repeated_smoke_test_buffers(&payload, repeats).unwrap();
        let total_frames = 1_920_u32.checked_mul(repeats).unwrap();

        assert_eq!(repeats, 13);
        assert_eq!(total_frames, 24_960);
        assert_eq!(buffers.len(), 13);
        assert_eq!(buffers.iter().map(Vec::len).sum::<usize>(), 399_360);
        assert!(buffers.iter().all(|buffer| buffer == &payload));
    }

    #[test]
    fn audio_content_audit_decodes_endian_and_unsigned_variants() {
        let payload = [
            0x00, 0x00, 0x00, // zero
            0xff, 0xff, 0x7f, // max positive little-endian
            0x00, 0x00, 0x80, // min negative little-endian
            0x34, 0x12, 0x00, // small positive little-endian
        ];

        assert_eq!(decode_s24le_i32(&payload, 0), 0);
        assert_eq!(decode_s24le_i32(&payload, 1), 8_388_607);
        assert_eq!(decode_s24le_i32(&payload, 2), -8_388_608);
        assert_eq!(decode_s24le_i32(&payload, 3), 0x1234);
        assert_eq!(decode_s24be_i32(&payload, 1), -129);
        assert_eq!(decode_u24le(&payload, 2), 0x80_0000);
    }

    #[test]
    fn audio_content_audit_computes_f32_buffer_stats() {
        let mut bytes = Vec::new();
        for value in [-0.5_f32, 0.0, 0.25, 0.5] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }

        let stats = f32_buffer_stats(&bytes, 2, 2).unwrap();

        assert_eq!(stats.min, -0.5);
        assert_eq!(stats.max, 0.5);
        assert!(stats.nonzero);
        assert!((stats.rms - 0.375).abs() < 0.000_001);
    }

    #[test]
    fn audio_content_audit_bounds_duration_and_sanitizes_output_stem() {
        assert_eq!(bounded_audio_audit_duration_ms(1).unwrap(), 1);
        assert_eq!(bounded_audio_audit_duration_ms(5_000).unwrap(), 5_000);
        assert!(bounded_audio_audit_duration_ms(0).is_err());
        assert!(bounded_audio_audit_duration_ms(5_001).is_err());
        assert_eq!(
            sanitized_source_stem(&PathBuf::from("/tmp/Mironik 2002.MXF")),
            "Mironik-2002"
        );
        assert_eq!(
            sanitized_source_stem(&PathBuf::from("/tmp/%%%")),
            "original-audio"
        );
    }

    #[test]
    fn file_presenter_manifest_filename_is_deterministic_and_public_safe() {
        assert_eq!(
            public_source_uri_stem("qnc://local/media/proxy/Mironik-1560"),
            "Mironik-1560"
        );
        assert_eq!(
            file_presenter_manifest_file_name("qnc://local/media/proxy/Mironik 1560", 4),
            "Mironik-1560-frame000004-proxy-preview-manifest.json"
        );
        assert_eq!(
            file_presenter_manifest_file_name("qnc://local/media/proxy/%%%/", 0),
            "qgs-video-frame000000-proxy-preview-manifest.json"
        );
    }

    #[test]
    fn file_presenter_manifest_helpers_escape_and_detect_private_paths() {
        assert_eq!(json_escape("a\"b\\c\n"), "a\\\"b\\\\c\\n");
        assert!(!manifest_json_contains_private_path(
            "qgs-visual-diagnostics/Mironik-1560-frame000000-proxy-preview-manifest.json"
        ));
        assert!(manifest_json_contains_private_path(
            "/home/miro/private/frame.json"
        ));
        assert!(manifest_json_contains_private_path(
            "file:///tmp/frame.json"
        ));
    }

    #[test]
    fn gpu_payload_readback_filenames_are_deterministic() {
        assert_eq!(
            gpu_payload_readback_image_file_name("qnc://local/media/proxy/Mironik 1560", 0),
            "Mironik-1560-frame000000-proxy-preview-readback.ppm"
        );
        assert_eq!(
            gpu_payload_readback_manifest_file_name("qnc://local/media/proxy/Mironik 1560", 0),
            "Mironik-1560-frame000000-proxy-preview-readback-manifest.json"
        );
    }

    #[test]
    fn rgba_u16_readback_converts_to_ppm_rgb8_without_alpha() {
        let rgba = vec![
            0x0000, 0x8000, 0xffff, 0x1234, 0x7fff, 0x0100, 0x00ff, 0xffff,
        ];

        let rgb = rgba_u16_to_rgb8_bytes(&rgba, 2, 1).unwrap();
        assert_eq!(rgb, vec![0x00, 0x80, 0xff, 0x7f, 0x01, 0x00]);

        let ppm = rgba_u16_to_ppm_p6_rgb8(&rgba, 2, 1).unwrap();
        assert!(ppm.starts_with(b"P6\n2 1\n255\n"));
        assert_eq!(&ppm[b"P6\n2 1\n255\n".len()..], rgb.as_slice());
        assert!(rgba_u16_to_rgb8_bytes(&rgba[..4], 2, 1).is_err());
    }

    #[test]
    fn audio_content_audit_ranks_loudest_monitor_pair() {
        let blocks = vec![
            mono_block(10, 0, &[[0x01, 0x00, 0x00]; 4]),
            mono_block(11, 1, &[[0x00, 0x00, 0x40]; 4]),
            mono_block(12, 2, &[[0x00, 0x00, 0x20]; 4]),
            mono_block(13, 3, &[[0x00, 0x00, 0x00]; 4]),
        ];
        let groups = original_pcm_blocks_by_channel(&blocks, 48_000).unwrap();
        let summaries = audio_audit_track_summaries(&groups, 48_000, 0, 4).unwrap();

        assert_eq!(loudest_monitor_pair(&summaries), vec![1, 2]);
        assert!(
            summaries
                .iter()
                .find(|summary| summary.channel_index == 3)
                .unwrap()
                .stats
                .likely_silent
        );
    }

    #[test]
    fn audio_content_audit_estimates_bounded_f32_allocation() {
        let sample_count = u32::try_from(
            qgs_media_runtime::audio_samples_for_duration(Duration::from_secs(5), 48_000).unwrap(),
        )
        .unwrap();
        let blocks = (0..4)
            .map(|channel| mono_block(10 + u32::from(channel), channel, &[[0x01, 0x00, 0x00]; 4]))
            .collect::<Vec<_>>();
        let groups = original_pcm_blocks_by_channel(&blocks, 48_000).unwrap();

        assert_eq!(usize::try_from(sample_count).unwrap() * 4 * 4, 3_840_000);
        assert!(
            build_audio_audit_f32_interleaved_range(&groups, &[0, 1, 2, 3], 48_000, 0, 4).is_ok()
        );
    }

    #[test]
    fn mono_monitor_duplicates_one_source_track_to_stereo_output() {
        let blocks = vec![
            mono_block(10, 0, &[[0x00, 0x00, 0x40], [0x00, 0x00, 0x20]]),
            mono_block(11, 1, &[[0x00, 0x00, 0x00], [0x00, 0x00, 0x00]]),
        ];
        let groups = original_pcm_blocks_by_channel(&blocks, 48_000).unwrap();
        let bytes =
            build_audio_audit_f32_interleaved_range(&groups, &[0, 0], 48_000, 0, 2).unwrap();
        let values = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();

        assert_eq!(bytes.len(), 2 * 2 * 4);
        assert_eq!(values[0], values[1]);
        assert_eq!(values[2], values[3]);
        assert_eq!(values[0], 0.5);
        assert_eq!(values[2], 0.25);
    }

    #[test]
    fn discrete_4mono_interleaves_tracks_without_fold_or_duplication() {
        let blocks = vec![
            mono_block(10, 0, &[[0x01, 0x00, 0x00], [0x02, 0x00, 0x00]]),
            mono_block(11, 1, &[[0x03, 0x00, 0x00], [0x04, 0x00, 0x00]]),
            mono_block(12, 2, &[[0x05, 0x00, 0x00], [0x06, 0x00, 0x00]]),
            mono_block(13, 3, &[[0x07, 0x00, 0x00], [0x08, 0x00, 0x00]]),
        ];
        let groups = original_pcm_blocks_by_channel(&blocks, 48_000).unwrap();
        let bytes =
            build_audio_audit_f32_interleaved_range(&groups, &[0, 1, 2, 3], 48_000, 0, 2).unwrap();
        let values = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();

        assert_eq!(bytes.len(), 2 * 4 * 4);
        assert_eq!(
            values,
            vec![
                1.0 / 8_388_608.0,
                3.0 / 8_388_608.0,
                5.0 / 8_388_608.0,
                7.0 / 8_388_608.0,
                2.0 / 8_388_608.0,
                4.0 / 8_388_608.0,
                6.0 / 8_388_608.0,
                8.0 / 8_388_608.0,
            ]
        );
        assert_ne!(values[0], values[1]);
        assert_ne!(values[0], values[2]);
        assert_ne!(values[0], values[3]);
    }

    #[test]
    fn discrete_4mono_default_duration_geometry_is_bounded() {
        let sample_count = u32::try_from(
            qgs_media_runtime::audio_samples_for_duration(Duration::from_secs(1), 48_000).unwrap(),
        )
        .unwrap();

        assert_eq!(sample_count, 48_000);
        assert_eq!(usize::try_from(sample_count).unwrap() * 4, 192_000);
        assert_eq!(usize::try_from(sample_count).unwrap() * 4 * 4, 768_000);
    }

    #[test]
    fn audio_content_audit_total_samples_comes_from_streaming_index() {
        let index = synthetic_pcm_audio_index(vec![
            audio_entry(3, 0, 0, 960, 0, 3),
            audio_entry(4, 1, 0, 960, 3, 3),
            audio_entry(3, 0, 960, 960, 6, 3),
            audio_entry(4, 1, 960, 480, 9, 3),
        ]);

        assert_eq!(audio_audit_total_samples_from_index(&index).unwrap(), 1_440);
    }

    #[test]
    fn audio_content_audit_extracts_only_overlapping_index_packets() {
        let path =
            std::env::temp_dir().join(format!("qgs-audit-range-{}-{}.pcm", std::process::id(), 2));
        let mut bytes = Vec::new();
        for value in [1_u8, 2, 3, 4] {
            bytes.extend_from_slice(&[value, 0, 0]);
        }
        std::fs::write(&path, &bytes).unwrap();
        let index = synthetic_pcm_audio_index(vec![
            audio_entry(3, 0, 0, 1, 0, 3),
            audio_entry(4, 1, 0, 1, 3, 3),
            audio_entry(3, 0, 1, 1, 6, 3),
            audio_entry(4, 1, 1, 1, 9, 3),
        ]);

        let blocks = build_original_pcm_blocks_from_index_range(&path, &index, 1, 2).unwrap();

        assert_eq!(blocks.len(), 2);
        assert!(blocks.iter().all(|block| block.start_time > Duration::ZERO));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn pipewire_prototype_rejects_short_blocks() {
        let blocks = vec![mono_block(10, 0, &[[0x00, 0x00, 0x00]])];

        assert!(build_pipewire_f32_interleaved_prototype_buffer(&blocks, 48_000, 2).is_err());
    }

    #[test]
    fn s24le_sample_conversion_sign_extends() {
        assert_eq!(
            pcm_s24le_sample_to_f32(&[0x00, 0x00, 0x80], 0).unwrap(),
            -1.0
        );
        assert_eq!(
            pcm_s24le_sample_to_f32(&[0x00, 0x00, 0x40], 0).unwrap(),
            0.5
        );
    }

    fn sample(sample_index: u32, pts: i64) -> Mp4VideoSample {
        Mp4VideoSample {
            sample_index,
            dts: u64::from(sample_index) * 1_000,
            pts,
            duration: 1_000,
            composition_offset: 0,
            is_sync: sample_index == 0,
            annex_b: vec![0, 0, 0, 1, 0x65],
        }
    }

    fn mono_block(track_id: u32, channel_index: u16, samples: &[[u8; 3]]) -> PcmAudioBlock {
        mono_block_at(track_id, channel_index, 0, samples)
    }

    fn mono_block_at(
        track_id: u32,
        channel_index: u16,
        start_sample: u64,
        samples: &[[u8; 3]],
    ) -> PcmAudioBlock {
        PcmAudioBlock::new(
            Duration::from_nanos(start_sample * 1_000_000_000 / 48_000),
            Duration::from_nanos(u64::try_from(samples.len()).unwrap() * 1_000_000_000 / 48_000),
            48_000,
            u32::try_from(samples.len()).unwrap(),
            PcmSampleFormat::SignedInteger {
                bits_per_sample: 24,
                endian: PcmEndian::Little,
            },
            PcmAudioBlockLayout::MonoTrack {
                track_id,
                channel_index,
            },
            samples
                .iter()
                .flat_map(|sample| sample.iter().copied())
                .collect(),
        )
        .unwrap()
    }

    fn synthetic_pcm_audio_index(audio: Vec<AudioIndexEntry>) -> PcmAudioIndex {
        PcmAudioIndex {
            file_len: audio
                .iter()
                .map(|entry| entry.payload_offset + entry.payload_len)
                .max()
                .unwrap_or(0),
            tracks: vec![
                audio_track(3, 0),
                audio_track(4, 1),
                audio_track(5, 2),
                audio_track(6, 3),
            ],
            audio,
            partitions: Vec::new(),
            metadata_set_count: 0,
            klv_count: 0,
        }
    }

    fn audio_track(track_id: u32, _channel_index: u16) -> MxfTrack {
        MxfTrack {
            id: TrackId(track_id),
            track_number: None,
            kind: TrackKind::Audio,
            edit_rate: None,
            video: None,
            audio: Some(AudioEssenceDescriptor {
                source: DescriptorSource::MxfMetadata,
                essence: None,
                channels: Some(1),
                sample_rate: Some(qgs_mxf::Rational::new(48_000, 1).unwrap()),
                bit_depth: Some(24),
                block_align: None,
                average_bytes_per_second: None,
            }),
            data: None,
        }
    }

    fn audio_entry(
        track_id: u32,
        channel_index: u16,
        start_sample: u64,
        sample_count: u32,
        payload_offset: u64,
        payload_len: u64,
    ) -> AudioIndexEntry {
        AudioIndexEntry {
            track_id: TrackId(track_id),
            channel_index,
            edit_unit: start_sample / u64::from(sample_count.max(1)),
            start_sample,
            sample_count,
            file_offset: payload_offset,
            payload_offset,
            payload_len,
            source: IndexSource::QgsDerived,
        }
    }
}
