//! Windows host bricks for QGS.
//!
//! The control connection is a TCP loopback stream that carries the same
//! length-prefixed wire messages as the Linux Unix-socket transport. WASAPI
//! shared-mode playback replaces PipeWire. GPU handle export stays on the
//! Linux Vulkan/DRM path.

#![forbid(unsafe_code)]

mod host;
mod pcm;
mod transport;

#[cfg(windows)]
mod audio;

pub use host::{default_endpoint, handle_client, serve, DEFAULT_PORT};
pub use pcm::{sine_f32le_interleaved, WasapiRenderEndpoint, WindowsAudioError};
pub use transport::{connect_endpoint, receive_message, send_message, TransportError};

/// List WASAPI render endpoints. The default console device is marked.
pub fn render_endpoints() -> Result<Vec<WasapiRenderEndpoint>, WindowsAudioError> {
    #[cfg(windows)]
    {
        audio::render_endpoints()
    }
    #[cfg(not(windows))]
    {
        Err(WindowsAudioError::UnsupportedPlatform)
    }
}

/// Play interleaved little-endian f32 PCM on the default WASAPI render device.
pub fn play_f32le_interleaved(
    sample_rate: u32,
    channels: u16,
    pcm: &[u8],
) -> Result<(), WindowsAudioError> {
    pcm::validate_f32le_interleaved(sample_rate, channels, pcm)?;
    #[cfg(windows)]
    {
        audio::play_f32le_interleaved(sample_rate, channels, pcm)
    }
    #[cfg(not(windows))]
    {
        let _ = (sample_rate, channels, pcm);
        Err(WindowsAudioError::UnsupportedPlatform)
    }
}
