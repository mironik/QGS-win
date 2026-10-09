use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;
use std::time::{Duration, Instant};

use pipewire as pw;
use pw::spa;
use pw::{properties::properties, spa::pod::Pod};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PipeWireAudioSampleFormat {
    F32Interleaved,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PipeWireStreamFormat {
    pub sample_rate: u32,
    pub channels: u32,
    pub sample_format: PipeWireAudioSampleFormat,
}

impl PipeWireStreamFormat {
    pub fn validate(self) -> Result<Self, PipeWireStreamError> {
        if self.sample_rate == 0 {
            return Err(PipeWireStreamError::InvalidFormat("sample rate is zero"));
        }
        if self.channels == 0 || self.channels > spa::param::audio::MAX_CHANNELS as u32 {
            return Err(PipeWireStreamError::InvalidFormat(
                "channel count is outside PipeWire SPA limits",
            ));
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PipeWireNativeStreamEvidenceLevel {
    ConnectFailed,
    StreamCreateFailed,
    NativeStreamCreated,
    NativeStreamConfigured,
    BufferDequeued,
    BufferSubmitted,
    PostSubmitTimeout,
    PostSubmitCallbackObserved,
    DrainCompleted,
    StreamError,
    StreamErrorAfterSubmit,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PipeWireObservedStreamState {
    Unconnected,
    Connecting,
    Paused,
    Streaming,
    Error,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PipeWireStreamCreationReport {
    pub connect_attempted: bool,
    pub connected: bool,
    pub stream_create_attempted: bool,
    pub stream_created: bool,
    pub stream_configured: bool,
    pub selected_format: PipeWireStreamFormat,
    pub observed_states: Vec<PipeWireObservedStreamState>,
    pub final_state: Option<PipeWireObservedStreamState>,
    pub evidence_level: PipeWireNativeStreamEvidenceLevel,
    pub audio_device_verified: bool,
    pub buffer_submitted: bool,
    pub status_message: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PipeWireBufferSubmissionReport {
    pub stream_report: PipeWireStreamCreationReport,
    pub process_callback_reached: bool,
    pub buffer_dequeued: bool,
    pub buffer_capacity: usize,
    pub buffers_planned: u32,
    pub buffers_submitted: u32,
    pub samples_converted: u32,
    pub output_channels: u32,
    pub f32_samples_written: usize,
    pub bytes_copied: usize,
    pub buffer_submitted: bool,
    pub post_submit_process_callbacks: u32,
    pub stream_states_after_submit: Vec<PipeWireObservedStreamState>,
    pub drain_requested: bool,
    pub drain_completed: bool,
    pub stream_error_after_submit: bool,
    pub post_submit_timeout: bool,
    pub evidence_level: PipeWireNativeStreamEvidenceLevel,
    pub audio_device_verified: bool,
    pub status_message: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PipeWireStreamError {
    InvalidFormat(&'static str),
    Connect(String),
    StreamCreate(String),
    StreamConnect(String),
    Timeout,
}

impl fmt::Display for PipeWireStreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFormat(message) => write!(f, "invalid PipeWire stream format: {message}"),
            Self::Connect(message) => write!(f, "PipeWire connect failed: {message}"),
            Self::StreamCreate(message) => write!(f, "PipeWire stream creation failed: {message}"),
            Self::StreamConnect(message) => {
                write!(f, "PipeWire stream configuration failed: {message}")
            }
            Self::Timeout => write!(f, "PipeWire stream did not configure before timeout"),
        }
    }
}

impl std::error::Error for PipeWireStreamError {}

#[derive(Default)]
struct StreamObservation {
    states: Vec<PipeWireObservedStreamState>,
}

#[derive(Default)]
struct SubmissionObservation {
    states: Vec<PipeWireObservedStreamState>,
    states_after_submit: Vec<PipeWireObservedStreamState>,
    process_callback_reached: bool,
    process_callbacks: u32,
    post_submit_process_callbacks: u32,
    buffer_dequeued: bool,
    buffer_capacity: usize,
    buffers_submitted: u32,
    bytes_copied: usize,
    buffer_submitted: bool,
    drain_requested: bool,
    drain_request_failed: bool,
    drain_completed: bool,
    stream_error_after_submit: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MonoS24LeTrack<'a> {
    pub channel_index: u16,
    pub payload: &'a [u8],
}

pub fn create_native_pipewire_stream(
    format: PipeWireStreamFormat,
    timeout: Duration,
) -> Result<PipeWireStreamCreationReport, PipeWireStreamError> {
    let format = format.validate()?;
    pw::init();
    let mainloop = pw::main_loop::MainLoopRc::new(None)
        .map_err(|err| PipeWireStreamError::Connect(err.to_string()))?;
    let context = pw::context::ContextRc::new(&mainloop, None)
        .map_err(|err| PipeWireStreamError::Connect(err.to_string()))?;
    let core = context
        .connect_rc(None)
        .map_err(|err| PipeWireStreamError::Connect(err.to_string()))?;

    let stream = pw::stream::StreamBox::new(
        &core,
        "qgs-native-audio-stream-create",
        properties! {
            *pw::keys::MEDIA_TYPE => "Audio",
            *pw::keys::MEDIA_ROLE => "Music",
            *pw::keys::MEDIA_CATEGORY => "Playback",
            *pw::keys::AUDIO_CHANNELS => format.channels.to_string(),
        },
    )
    .map_err(|err| PipeWireStreamError::StreamCreate(err.to_string()))?;

    let observation = Rc::new(RefCell::new(StreamObservation::default()));
    let listener_observation = Rc::clone(&observation);
    let listener_mainloop = mainloop.clone();
    let _listener = stream
        .add_local_listener::<()>()
        .state_changed(move |_stream, _user_data, _old, new| {
            let observed = observed_stream_state(&new);
            listener_observation.borrow_mut().states.push(observed);
            if matches!(
                observed,
                PipeWireObservedStreamState::Paused
                    | PipeWireObservedStreamState::Streaming
                    | PipeWireObservedStreamState::Error
            ) {
                listener_mainloop.quit();
            }
        })
        .register()
        .map_err(|err| PipeWireStreamError::StreamCreate(err.to_string()))?;

    let values = format_pod_values(format)?;
    let mut params = [Pod::from_bytes(&values).ok_or_else(|| {
        PipeWireStreamError::StreamConnect("serialized PipeWire format pod was invalid".to_string())
    })?];

    stream
        .connect(
            spa::utils::Direction::Output,
            None,
            pw::stream::StreamFlags::AUTOCONNECT | pw::stream::StreamFlags::MAP_BUFFERS,
            &mut params,
        )
        .map_err(|err| PipeWireStreamError::StreamConnect(err.to_string()))?;

    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        mainloop
            .loop_()
            .iterate(pw::loop_::Timeout::Finite(Duration::from_millis(50)));
        if observation.borrow().states.iter().any(|state| {
            matches!(
                state,
                PipeWireObservedStreamState::Paused
                    | PipeWireObservedStreamState::Streaming
                    | PipeWireObservedStreamState::Error
            )
        }) {
            break;
        }
    }

    let states = observation.borrow().states.clone();
    if states.is_empty() {
        return Err(PipeWireStreamError::Timeout);
    }
    let final_state = states.last().copied();
    let stream_configured = states.iter().any(|state| {
        matches!(
            state,
            PipeWireObservedStreamState::Paused | PipeWireObservedStreamState::Streaming
        )
    });

    Ok(PipeWireStreamCreationReport {
        connect_attempted: true,
        connected: true,
        stream_create_attempted: true,
        stream_created: true,
        stream_configured,
        selected_format: format,
        observed_states: states,
        final_state,
        evidence_level: if stream_configured {
            PipeWireNativeStreamEvidenceLevel::NativeStreamConfigured
        } else {
            PipeWireNativeStreamEvidenceLevel::NativeStreamCreated
        },
        audio_device_verified: false,
        buffer_submitted: false,
        status_message: if stream_configured {
            "native PipeWire stream reached configured state; no buffer was submitted".to_string()
        } else {
            "native PipeWire stream was created but did not reach configured state".to_string()
        },
    })
}

pub fn submit_native_pipewire_buffer(
    format: PipeWireStreamFormat,
    f32_interleaved_le: Vec<u8>,
    frame_count: u32,
    timeout: Duration,
) -> Result<PipeWireBufferSubmissionReport, PipeWireStreamError> {
    submit_native_pipewire_buffers(format, vec![f32_interleaved_le], frame_count, timeout)
}

pub fn submit_native_pipewire_buffers(
    format: PipeWireStreamFormat,
    f32_interleaved_buffers: Vec<Vec<u8>>,
    frames_per_buffer: u32,
    timeout: Duration,
) -> Result<PipeWireBufferSubmissionReport, PipeWireStreamError> {
    let format = format.validate()?;
    if f32_interleaved_buffers.is_empty() {
        return Err(PipeWireStreamError::InvalidFormat("no buffers to submit"));
    }
    for buffer in &f32_interleaved_buffers {
        validate_f32_interleaved_buffer(buffer, frames_per_buffer, format.channels)?;
    }
    pw::init();
    let mainloop = pw::main_loop::MainLoopRc::new(None)
        .map_err(|err| PipeWireStreamError::Connect(err.to_string()))?;
    let context = pw::context::ContextRc::new(&mainloop, None)
        .map_err(|err| PipeWireStreamError::Connect(err.to_string()))?;
    let core = context
        .connect_rc(None)
        .map_err(|err| PipeWireStreamError::Connect(err.to_string()))?;

    let stream = pw::stream::StreamBox::new(
        &core,
        "qgs-native-audio-buffer-submit",
        properties! {
            *pw::keys::MEDIA_TYPE => "Audio",
            *pw::keys::MEDIA_ROLE => "Music",
            *pw::keys::MEDIA_CATEGORY => "Playback",
            *pw::keys::AUDIO_CHANNELS => format.channels.to_string(),
        },
    )
    .map_err(|err| PipeWireStreamError::StreamCreate(err.to_string()))?;

    let observation = Rc::new(RefCell::new(SubmissionObservation::default()));
    let process_observation = Rc::clone(&observation);
    let drained_observation = Rc::clone(&observation);
    let state_observation = Rc::clone(&observation);
    let payloads = Rc::new(f32_interleaved_buffers);
    let process_payloads = Rc::clone(&payloads);
    let listener_mainloop = mainloop.clone();
    let dequeue_failure_mainloop = mainloop.clone();
    let drained_mainloop = mainloop.clone();
    let stride = usize::try_from(format.channels)
        .ok()
        .and_then(|channels| channels.checked_mul(4))
        .ok_or(PipeWireStreamError::InvalidFormat("stream stride overflow"))?;

    let _listener = stream
        .add_local_listener::<()>()
        .state_changed(move |_stream, _user_data, _old, new| {
            let observed = observed_stream_state(&new);
            let Ok(mut observation) = state_observation.try_borrow_mut() else {
                return;
            };
            observation.states.push(observed);
            if observation.buffer_submitted {
                observation.states_after_submit.push(observed);
            }
            if observation.buffer_submitted
                && matches!(observed, PipeWireObservedStreamState::Error)
            {
                observation.stream_error_after_submit = true;
            }
            if matches!(observed, PipeWireObservedStreamState::Error) {
                listener_mainloop.quit();
            }
        })
        .process(move |stream, _user_data| {
            let Ok(mut observation) = process_observation.try_borrow_mut() else {
                return;
            };
            observation.process_callback_reached = true;
            observation.process_callbacks = observation.process_callbacks.saturating_add(1);
            if observation.buffer_submitted {
                observation.post_submit_process_callbacks =
                    observation.post_submit_process_callbacks.saturating_add(1);
                return;
            }
            let payload_index =
                usize::try_from(observation.buffers_submitted).unwrap_or(usize::MAX);
            let Some(process_payload) = process_payloads.get(payload_index) else {
                observation.buffer_submitted = true;
                return;
            };
            let Some(mut buffer) = stream.dequeue_buffer() else {
                dequeue_failure_mainloop.quit();
                return;
            };
            observation.buffer_dequeued = true;
            let Some(data) = buffer.datas_mut().get_mut(0) else {
                dequeue_failure_mainloop.quit();
                return;
            };
            let Some(slice) = data.data() else {
                dequeue_failure_mainloop.quit();
                return;
            };
            observation.buffer_capacity = observation.buffer_capacity.max(slice.len());
            if slice.len() < process_payload.len() {
                dequeue_failure_mainloop.quit();
                return;
            }
            slice[..process_payload.len()].copy_from_slice(&process_payload);
            let chunk = data.chunk_mut();
            *chunk.offset_mut() = 0;
            *chunk.stride_mut() = i32::try_from(stride).unwrap_or(i32::MAX);
            *chunk.size_mut() = u32::try_from(process_payload.len()).unwrap_or(u32::MAX);
            observation.bytes_copied = observation
                .bytes_copied
                .saturating_add(process_payload.len());
            observation.buffers_submitted = observation.buffers_submitted.saturating_add(1);
            observation.buffer_submitted = usize::try_from(observation.buffers_submitted)
                .unwrap_or(usize::MAX)
                >= process_payloads.len();
            drop(buffer);
            let request_drain = observation.buffer_submitted;
            drop(observation);
            if request_drain {
                let drain_result = stream.flush(true);
                let Ok(mut observation) = process_observation.try_borrow_mut() else {
                    return;
                };
                match drain_result {
                    Ok(()) => observation.drain_requested = true,
                    Err(_) => observation.drain_request_failed = true,
                }
            }
        })
        .drained(move |_stream, _user_data| {
            let Ok(mut observation) = drained_observation.try_borrow_mut() else {
                drained_mainloop.quit();
                return;
            };
            observation.drain_completed = true;
            drained_mainloop.quit();
        })
        .register()
        .map_err(|err| PipeWireStreamError::StreamCreate(err.to_string()))?;

    let values = format_pod_values(format)?;
    let mut params = [Pod::from_bytes(&values).ok_or_else(|| {
        PipeWireStreamError::StreamConnect("serialized PipeWire format pod was invalid".to_string())
    })?];

    stream
        .connect(
            spa::utils::Direction::Output,
            None,
            pw::stream::StreamFlags::AUTOCONNECT
                | pw::stream::StreamFlags::MAP_BUFFERS
                | pw::stream::StreamFlags::RT_PROCESS,
            &mut params,
        )
        .map_err(|err| PipeWireStreamError::StreamConnect(err.to_string()))?;

    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        mainloop
            .loop_()
            .iterate(pw::loop_::Timeout::Finite(Duration::from_millis(50)));
        let Ok(observation) = observation.try_borrow() else {
            continue;
        };
        if observation.drain_completed || observation.stream_error_after_submit {
            break;
        }
    }

    let observation = observation.try_borrow().map_err(|_| {
        PipeWireStreamError::StreamConnect("PipeWire observation was busy".to_string())
    })?;
    let states = observation.states.clone();
    if states.is_empty() && !observation.process_callback_reached {
        return Err(PipeWireStreamError::Timeout);
    }
    let final_state = states.last().copied();
    let stream_configured = states.iter().any(|state| {
        matches!(
            state,
            PipeWireObservedStreamState::Paused | PipeWireObservedStreamState::Streaming
        )
    });
    let post_submit_timeout = observation.buffer_submitted
        && !observation.drain_completed
        && !observation.stream_error_after_submit;
    let evidence_level = classify_buffer_submission_evidence(
        observation.buffer_submitted,
        observation.buffer_dequeued,
        stream_configured,
        observation.post_submit_process_callbacks,
        observation.drain_completed,
        observation.stream_error_after_submit,
        post_submit_timeout,
    );
    let stream_report = PipeWireStreamCreationReport {
        connect_attempted: true,
        connected: true,
        stream_create_attempted: true,
        stream_created: true,
        stream_configured,
        selected_format: format,
        observed_states: states,
        final_state,
        evidence_level,
        audio_device_verified: false,
        buffer_submitted: observation.buffer_submitted,
        status_message: if observation.drain_completed {
            "native PipeWire buffer was queued and a drain callback was observed; audible playback is not verified".to_string()
        } else if observation.post_submit_process_callbacks > 0 {
            "native PipeWire buffer was queued and a post-submit process callback was observed; audible playback is not verified".to_string()
        } else if observation.buffer_submitted {
            "native PipeWire buffer was queued; audible playback is not verified".to_string()
        } else {
            "native PipeWire stream did not accept the prototype buffer".to_string()
        },
    };

    Ok(PipeWireBufferSubmissionReport {
        stream_report,
        process_callback_reached: observation.process_callback_reached,
        buffer_dequeued: observation.buffer_dequeued,
        buffer_capacity: observation.buffer_capacity,
        buffers_planned: u32::try_from(payloads.len()).unwrap_or(u32::MAX),
        buffers_submitted: observation.buffers_submitted,
        samples_converted: frames_per_buffer.saturating_mul(observation.buffers_submitted),
        output_channels: format.channels,
        f32_samples_written: usize::try_from(frames_per_buffer)
            .unwrap_or(0)
            .saturating_mul(usize::try_from(format.channels).unwrap_or(0))
            .saturating_mul(usize::try_from(observation.buffers_submitted).unwrap_or(0)),
        bytes_copied: observation.bytes_copied,
        buffer_submitted: observation.buffer_submitted,
        post_submit_process_callbacks: observation.post_submit_process_callbacks,
        stream_states_after_submit: observation.states_after_submit.clone(),
        drain_requested: observation.drain_requested,
        drain_completed: observation.drain_completed,
        stream_error_after_submit: observation.stream_error_after_submit,
        post_submit_timeout,
        evidence_level,
        audio_device_verified: false,
        status_message: if observation.drain_completed {
            "bounded prototype buffer queued and PipeWire drained callback observed; no audible verification claimed".to_string()
        } else if observation.post_submit_process_callbacks > 0 {
            "bounded prototype buffer queued and post-submit process callback observed; no audible verification claimed".to_string()
        } else if post_submit_timeout {
            "bounded prototype buffer queued; no stronger post-submit evidence arrived before timeout".to_string()
        } else if observation.buffer_submitted {
            "bounded prototype buffer queued to native PipeWire stream; no audible verification claimed".to_string()
        } else {
            "bounded prototype buffer was not queued".to_string()
        },
    })
}

pub fn classify_buffer_submission_evidence(
    buffer_submitted: bool,
    buffer_dequeued: bool,
    stream_configured: bool,
    post_submit_process_callbacks: u32,
    drain_completed: bool,
    stream_error_after_submit: bool,
    post_submit_timeout: bool,
) -> PipeWireNativeStreamEvidenceLevel {
    if stream_error_after_submit {
        PipeWireNativeStreamEvidenceLevel::StreamErrorAfterSubmit
    } else if drain_completed {
        PipeWireNativeStreamEvidenceLevel::DrainCompleted
    } else if post_submit_process_callbacks > 0 {
        PipeWireNativeStreamEvidenceLevel::PostSubmitCallbackObserved
    } else if post_submit_timeout {
        PipeWireNativeStreamEvidenceLevel::PostSubmitTimeout
    } else if buffer_submitted {
        PipeWireNativeStreamEvidenceLevel::BufferSubmitted
    } else if buffer_dequeued {
        PipeWireNativeStreamEvidenceLevel::BufferDequeued
    } else if stream_configured {
        PipeWireNativeStreamEvidenceLevel::NativeStreamConfigured
    } else {
        PipeWireNativeStreamEvidenceLevel::NativeStreamCreated
    }
}

pub fn f32_interleaved_from_s24le_mono_tracks(
    tracks: &[MonoS24LeTrack<'_>],
    frame_count: u32,
) -> Result<Vec<u8>, PipeWireStreamError> {
    if tracks.is_empty() {
        return Err(PipeWireStreamError::InvalidFormat("no source tracks"));
    }
    for track in tracks {
        let needed = usize::try_from(frame_count)
            .ok()
            .and_then(|frames| frames.checked_mul(3))
            .ok_or(PipeWireStreamError::InvalidFormat(
                "24-bit source payload size overflow",
            ))?;
        if track.payload.len() < needed {
            return Err(PipeWireStreamError::InvalidFormat(
                "source track is shorter than requested frame count",
            ));
        }
    }
    let mut sorted = tracks.to_vec();
    sorted.sort_by_key(|track| track.channel_index);
    let capacity = usize::try_from(frame_count)
        .ok()
        .and_then(|frames| frames.checked_mul(sorted.len()))
        .and_then(|samples| samples.checked_mul(4))
        .ok_or(PipeWireStreamError::InvalidFormat(
            "converted output size overflow",
        ))?;
    let mut output = Vec::with_capacity(capacity);
    for frame in 0..usize::try_from(frame_count).unwrap_or(0) {
        for track in &sorted {
            let sample = s24le_sample_to_f32(track.payload, frame)?;
            output.extend_from_slice(&sample.to_le_bytes());
        }
    }
    Ok(output)
}

pub fn s24le_sample_to_f32(
    payload: &[u8],
    sample_index: usize,
) -> Result<f32, PipeWireStreamError> {
    let offset = sample_index
        .checked_mul(3)
        .ok_or(PipeWireStreamError::InvalidFormat(
            "24-bit sample offset overflow",
        ))?;
    let bytes = payload
        .get(offset..offset + 3)
        .ok_or(PipeWireStreamError::InvalidFormat(
            "24-bit sample index out of range",
        ))?;
    let mut value = i32::from(bytes[0]) | (i32::from(bytes[1]) << 8) | (i32::from(bytes[2]) << 16);
    if value & 0x0080_0000 != 0 {
        value |= !0x00ff_ffff;
    }
    Ok((value as f32 / 8_388_608.0).clamp(-1.0, 1.0))
}

fn validate_f32_interleaved_buffer(
    bytes: &[u8],
    frame_count: u32,
    channels: u32,
) -> Result<(), PipeWireStreamError> {
    let expected = usize::try_from(frame_count)
        .ok()
        .and_then(|frames| frames.checked_mul(usize::try_from(channels).ok()?))
        .and_then(|samples| samples.checked_mul(4))
        .ok_or(PipeWireStreamError::InvalidFormat(
            "f32 interleaved buffer size overflow",
        ))?;
    if bytes.len() != expected {
        return Err(PipeWireStreamError::InvalidFormat(
            "f32 interleaved buffer size does not match frame count and channel count",
        ));
    }
    Ok(())
}

fn format_pod_values(format: PipeWireStreamFormat) -> Result<Vec<u8>, PipeWireStreamError> {
    let mut audio_info = spa::param::audio::AudioInfoRaw::new();
    match format.sample_format {
        PipeWireAudioSampleFormat::F32Interleaved => {
            audio_info.set_format(spa::param::audio::AudioFormat::F32LE);
        }
    }
    audio_info.set_rate(format.sample_rate);
    audio_info.set_channels(format.channels);
    let mut position = [0; spa::param::audio::MAX_CHANNELS];
    for (index, channel) in default_channel_positions(format.channels)
        .iter()
        .enumerate()
    {
        position[index] = *channel;
    }
    audio_info.set_position(position);

    pw::spa::pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(Vec::new()),
        &pw::spa::pod::Value::Object(pw::spa::pod::Object {
            type_: spa_sys::SPA_TYPE_OBJECT_Format,
            id: spa_sys::SPA_PARAM_EnumFormat,
            properties: audio_info.into(),
        }),
    )
    .map(|serialized| serialized.0.into_inner())
    .map_err(|err| PipeWireStreamError::StreamConnect(format!("{err:?}")))
}

pub fn default_channel_positions(channels: u32) -> Vec<u32> {
    match channels {
        1 => vec![spa_sys::SPA_AUDIO_CHANNEL_MONO],
        2 => vec![spa_sys::SPA_AUDIO_CHANNEL_FL, spa_sys::SPA_AUDIO_CHANNEL_FR],
        4 => vec![
            spa_sys::SPA_AUDIO_CHANNEL_FL,
            spa_sys::SPA_AUDIO_CHANNEL_FR,
            spa_sys::SPA_AUDIO_CHANNEL_RL,
            spa_sys::SPA_AUDIO_CHANNEL_RR,
        ],
        count => (0..count)
            .map(|index| {
                if index < spa::param::audio::MAX_CHANNELS as u32 {
                    spa_sys::SPA_AUDIO_CHANNEL_UNKNOWN
                } else {
                    0
                }
            })
            .collect(),
    }
}

fn observed_stream_state(state: &pw::stream::StreamState) -> PipeWireObservedStreamState {
    match state {
        pw::stream::StreamState::Error(_) => PipeWireObservedStreamState::Error,
        pw::stream::StreamState::Unconnected => PipeWireObservedStreamState::Unconnected,
        pw::stream::StreamState::Connecting => PipeWireObservedStreamState::Connecting,
        pw::stream::StreamState::Paused => PipeWireObservedStreamState::Paused,
        pw::stream::StreamState::Streaming => PipeWireObservedStreamState::Streaming,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_format_rejects_invalid_values() {
        assert!(PipeWireStreamFormat {
            sample_rate: 0,
            channels: 4,
            sample_format: PipeWireAudioSampleFormat::F32Interleaved,
        }
        .validate()
        .is_err());
        assert!(PipeWireStreamFormat {
            sample_rate: 48_000,
            channels: 0,
            sample_format: PipeWireAudioSampleFormat::F32Interleaved,
        }
        .validate()
        .is_err());
    }

    #[test]
    fn stream_format_accepts_48k_four_channel_f32_boundary_format() {
        let format = PipeWireStreamFormat {
            sample_rate: 48_000,
            channels: 4,
            sample_format: PipeWireAudioSampleFormat::F32Interleaved,
        }
        .validate()
        .unwrap();

        assert_eq!(format.sample_rate, 48_000);
        assert_eq!(format.channels, 4);
    }

    #[test]
    fn stream_format_accepts_48k_stereo_f32_boundary_format() {
        let format = PipeWireStreamFormat {
            sample_rate: 48_000,
            channels: 2,
            sample_format: PipeWireAudioSampleFormat::F32Interleaved,
        }
        .validate()
        .unwrap();

        assert_eq!(format.sample_rate, 48_000);
        assert_eq!(format.channels, 2);
    }

    #[test]
    fn channel_positions_are_stable_for_stereo_monitor_output() {
        assert_eq!(
            default_channel_positions(2),
            vec![spa_sys::SPA_AUDIO_CHANNEL_FL, spa_sys::SPA_AUDIO_CHANNEL_FR]
        );
    }

    #[test]
    fn channel_positions_are_stable_for_four_channel_prototype() {
        assert_eq!(
            default_channel_positions(4),
            vec![
                spa_sys::SPA_AUDIO_CHANNEL_FL,
                spa_sys::SPA_AUDIO_CHANNEL_FR,
                spa_sys::SPA_AUDIO_CHANNEL_RL,
                spa_sys::SPA_AUDIO_CHANNEL_RR,
            ]
        );
    }

    #[test]
    fn s24le_to_f32_conversion_handles_key_values() {
        assert_eq!(s24le_sample_to_f32(&[0x00, 0x00, 0x00], 0).unwrap(), 0.0);
        assert!(s24le_sample_to_f32(&[0xff, 0xff, 0x7f], 0).unwrap() > 0.999_999);
        assert_eq!(s24le_sample_to_f32(&[0x00, 0x00, 0x80], 0).unwrap(), -1.0);
        assert_eq!(
            s24le_sample_to_f32(&[0x01, 0x00, 0x00], 0).unwrap(),
            1.0 / 8_388_608.0
        );
        assert_eq!(
            s24le_sample_to_f32(&[0xff, 0xff, 0xff], 0).unwrap(),
            -1.0 / 8_388_608.0
        );
    }

    #[test]
    fn four_mono_tracks_are_interleaved_by_channel_index() {
        let tracks = [
            MonoS24LeTrack {
                channel_index: 2,
                payload: &[0x00, 0x00, 0x40],
            },
            MonoS24LeTrack {
                channel_index: 0,
                payload: &[0x00, 0x00, 0x00],
            },
            MonoS24LeTrack {
                channel_index: 3,
                payload: &[0x00, 0x00, 0x80],
            },
            MonoS24LeTrack {
                channel_index: 1,
                payload: &[0xff, 0xff, 0x7f],
            },
        ];

        let bytes = f32_interleaved_from_s24le_mono_tracks(&tracks, 1).unwrap();
        assert_eq!(bytes.len(), 4 * 4);
        let values = bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(values[0], 0.0);
        assert!(values[1] > 0.999_999);
        assert_eq!(values[2], 0.5);
        assert_eq!(values[3], -1.0);
    }

    #[test]
    fn f32_interleaved_buffer_size_validation_is_exact() {
        assert!(validate_f32_interleaved_buffer(&vec![0_u8; 48_000 * 2 * 4], 48_000, 2).is_ok());
        assert!(validate_f32_interleaved_buffer(&vec![0_u8; 960 * 4 * 4], 960, 4).is_ok());
        assert!(validate_f32_interleaved_buffer(&vec![0_u8; 7], 960, 4).is_err());
    }

    #[test]
    fn evidence_classification_keeps_submission_and_timeout_distinct() {
        assert_eq!(
            classify_buffer_submission_evidence(true, true, true, 0, false, false, false),
            PipeWireNativeStreamEvidenceLevel::BufferSubmitted
        );
        assert_eq!(
            classify_buffer_submission_evidence(true, true, true, 0, false, false, true),
            PipeWireNativeStreamEvidenceLevel::PostSubmitTimeout
        );
    }

    #[test]
    fn evidence_classification_prefers_post_submit_callback_over_timeout() {
        assert_eq!(
            classify_buffer_submission_evidence(true, true, true, 1, false, false, true),
            PipeWireNativeStreamEvidenceLevel::PostSubmitCallbackObserved
        );
    }

    #[test]
    fn evidence_classification_prefers_drain_and_error_over_callback() {
        assert_eq!(
            classify_buffer_submission_evidence(true, true, true, 3, true, false, true),
            PipeWireNativeStreamEvidenceLevel::DrainCompleted
        );
        assert_eq!(
            classify_buffer_submission_evidence(true, true, true, 3, true, true, true),
            PipeWireNativeStreamEvidenceLevel::StreamErrorAfterSubmit
        );
    }
}
