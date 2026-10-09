//! Media-capable Broadcast Player LEGO bricks.
//!
//! Backend-neutral planning/transport/control stays in `qgs-media-runtime`.
//! This crate owns bricks that may depend on MXF/MP4/VAAPI/Vulkan/PipeWire:
//! decode → GPU payload, original audio payload, device boundaries, and
//! diagnostic monitor presentation. Assembly composes; bricks keep semantics.

pub mod audio_device_boundary;
pub mod audio_payload;
pub mod device_selection;
pub mod live_output;
pub mod presenter_boundary;
pub mod video_payload;

mod monitor_preview;

pub use audio_device_boundary::AudioDeviceBoundary;
pub use audio_payload::{AudioCoverage, AudioOutputMode, OriginalMxfAudioPayload};
pub use device_selection::{select_integrated_or_discrete_gpu, select_integrated_vulkan_gpu};
pub use live_output::{
    LiveOutputCoordinator, LiveOutputTickResult, LiveVideoOutputMode, PreviewLatestJsonParts,
};
pub use presenter_boundary::LiveMonitorSink;
pub use video_payload::{EnginePixels, LiveEnginePictureSession};
