use std::time::{Duration, Instant};

use wasapi::{DeviceEnumerator, Direction, SampleType, StreamMode, WaveFormat};

use crate::pcm::{WasapiRenderEndpoint, WindowsAudioError};

pub(crate) fn render_endpoints() -> Result<Vec<WasapiRenderEndpoint>, WindowsAudioError> {
    ensure_mta()?;
    let enumerator = DeviceEnumerator::new().map_err(backend)?;
    let default_id = enumerator
        .get_default_device(&Direction::Render)
        .and_then(|device| device.get_id())
        .ok();
    let collection = enumerator
        .get_device_collection(&Direction::Render)
        .map_err(backend)?;
    let count = collection.get_nbr_devices().map_err(backend)?;
    let mut endpoints = Vec::with_capacity(count as usize);
    for index in 0..count {
        let device = collection.get_device_at_index(index).map_err(backend)?;
        let id = device.get_id().map_err(backend)?;
        let name = device.get_friendlyname().map_err(backend)?;
        let is_default = default_id.as_ref() == Some(&id);
        endpoints.push(WasapiRenderEndpoint {
            id,
            name,
            is_default,
        });
    }
    Ok(endpoints)
}

pub(crate) fn play_f32le_interleaved(
    sample_rate: u32,
    channels: u16,
    pcm: &[u8],
) -> Result<(), WindowsAudioError> {
    ensure_mta()?;
    let enumerator = DeviceEnumerator::new().map_err(backend)?;
    let device = enumerator
        .get_default_device(&Direction::Render)
        .map_err(backend)?;
    let mut audio_client = device.get_iaudioclient().map_err(backend)?;
    let format = WaveFormat::new(
        32,
        32,
        &SampleType::Float,
        sample_rate as usize,
        usize::from(channels),
        None,
    );
    let (_default_period, minimum_period) = audio_client.get_device_period().map_err(backend)?;
    let buffer_duration_hns = if minimum_period > 0 {
        minimum_period
    } else {
        200_000
    };
    let mode = StreamMode::EventsShared {
        autoconvert: true,
        buffer_duration_hns,
    };
    audio_client
        .initialize_client(&format, &Direction::Render, &mode)
        .map_err(backend)?;
    let event = audio_client.set_get_eventhandle().map_err(backend)?;
    let render_client = audio_client.get_audiorenderclient().map_err(backend)?;

    let frame_bytes = usize::from(channels) * 4;
    let mut offset = 0;
    write_available(
        &audio_client,
        &render_client,
        pcm,
        &mut offset,
        frame_bytes,
    )?;
    audio_client.start_stream().map_err(backend)?;
    while offset < pcm.len() {
        event.wait_for_event(2_000).map_err(backend)?;
        write_available(
            &audio_client,
            &render_client,
            pcm,
            &mut offset,
            frame_bytes,
        )?;
    }

    let frames = (pcm.len() / frame_bytes) as u64;
    let playback = Duration::from_secs_f64(frames as f64 / f64::from(sample_rate));
    let deadline = Instant::now() + playback + Duration::from_millis(500);
    while audio_client.get_current_padding().map_err(backend)? > 0 && Instant::now() < deadline {
        let _ = event.wait_for_event(100);
    }
    audio_client.stop_stream().map_err(backend)?;
    Ok(())
}

fn write_available(
    audio_client: &wasapi::AudioClient,
    render_client: &wasapi::AudioRenderClient,
    pcm: &[u8],
    offset: &mut usize,
    frame_bytes: usize,
) -> Result<(), WindowsAudioError> {
    let space = audio_client
        .get_available_space_in_frames()
        .map_err(backend)? as usize;
    let frames_left = pcm.len().saturating_sub(*offset) / frame_bytes;
    let frames = space.min(frames_left);
    if frames == 0 {
        return Ok(());
    }
    let bytes = frames * frame_bytes;
    render_client
        .write_to_device(frames, &pcm[*offset..*offset + bytes], None)
        .map_err(backend)?;
    *offset += bytes;
    Ok(())
}

fn ensure_mta() -> Result<(), WindowsAudioError> {
    match wasapi::initialize_mta().ok() {
        Ok(()) => Ok(()),
        Err(err) => {
            let text = err.to_string();
            if text.contains("0x800401F0") || text.to_ascii_lowercase().contains("already") {
                Ok(())
            } else {
                Err(WindowsAudioError::Backend(text))
            }
        }
    }
}

fn backend(err: impl ToString) -> WindowsAudioError {
    WindowsAudioError::Backend(err.to_string())
}
