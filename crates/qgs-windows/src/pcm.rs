use std::fmt;

/// One WASAPI render endpoint.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WasapiRenderEndpoint {
    pub id: String,
    pub name: String,
    pub is_default: bool,
}

#[derive(Debug)]
pub enum WindowsAudioError {
    InvalidPcm(&'static str),
    UnsupportedPlatform,
    Backend(String),
}

impl fmt::Display for WindowsAudioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPcm(reason) => write!(f, "invalid PCM buffer: {reason}"),
            Self::UnsupportedPlatform => {
                write!(f, "WASAPI playback is only available on Windows")
            }
            Self::Backend(reason) => write!(f, "WASAPI playback failed: {reason}"),
        }
    }
}

impl std::error::Error for WindowsAudioError {}

pub(crate) fn validate_format(
    sample_rate: u32,
    channels: u16,
) -> Result<(), WindowsAudioError> {
    if sample_rate == 0 {
        return Err(WindowsAudioError::InvalidPcm("sample rate is zero"));
    }
    if channels == 0 || channels > 8 {
        return Err(WindowsAudioError::InvalidPcm(
            "channel count must be from 1 to 8",
        ));
    }
    Ok(())
}

pub(crate) fn validate_f32le_interleaved(
    sample_rate: u32,
    channels: u16,
    pcm: &[u8],
) -> Result<(), WindowsAudioError> {
    validate_format(sample_rate, channels)?;
    let frame_bytes = usize::from(channels) * 4;
    if pcm.len() % frame_bytes != 0 {
        return Err(WindowsAudioError::InvalidPcm(
            "byte length is not a whole number of f32 frames",
        ));
    }
    if pcm.is_empty() {
        return Err(WindowsAudioError::InvalidPcm("buffer is empty"));
    }
    Ok(())
}

/// Build interleaved little-endian f32 PCM for a sine tone.
pub fn sine_f32le_interleaved(
    sample_rate: u32,
    channels: u16,
    frequency_hz: f32,
    frame_count: u32,
) -> Result<Vec<u8>, WindowsAudioError> {
    if frequency_hz <= 0.0 || !frequency_hz.is_finite() {
        return Err(WindowsAudioError::InvalidPcm(
            "frequency must be a positive finite value",
        ));
    }
    if frame_count == 0 {
        return Err(WindowsAudioError::InvalidPcm("frame count is zero"));
    }
    validate_format(sample_rate, channels)?;

    let mut pcm = Vec::with_capacity(usize::from(channels) * 4 * frame_count as usize);
    let step = frequency_hz as f64 / f64::from(sample_rate);
    for frame in 0..frame_count {
        let sample = (2.0 * std::f64::consts::PI * step * f64::from(frame)).sin() as f32;
        let bytes = sample.to_le_bytes();
        for _ in 0..channels {
            pcm.extend_from_slice(&bytes);
        }
    }
    Ok(pcm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sine_buffer_matches_frame_layout() {
        let pcm = sine_f32le_interleaved(48_000, 2, 440.0, 48).expect("sine");
        assert_eq!(pcm.len(), 48 * 2 * 4);
        assert!(pcm.chunks(4).any(|sample| sample != [0, 0, 0, 0]));
    }

    #[test]
    fn rejects_partial_frame() {
        let err = validate_f32le_interleaved(48_000, 2, &[0, 1, 2]).expect_err("partial");
        assert!(matches!(err, WindowsAudioError::InvalidPcm(_)));
    }
}
