use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use qgs_codec_h264::parse_annex_b_access_unit;
use qgs_h264_422p10::{H264422P10Decoder, H264422P10Frame, H264422P10Profile};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: bench_decode <annex-b.h264> [repeats]")?;
    let repeats: usize = std::env::args()
        .nth(2)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(1);
    if repeats == 0 {
        return Err("repeats must be positive".into());
    }
    let bytes = std::fs::read(&input)?;
    let access_units = split_h264_annex_b_access_units(&bytes)?;
    let first = access_units.first().ok_or("no H.264 access units found")?;
    let parsed = parse_annex_b_access_unit(first)?;
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

    let warmup = access_units.len().min(8);
    let mut warmup_decoder = H264422P10Decoder::new(profile.clone());
    for access_unit in access_units.iter().take(warmup) {
        for frame in warmup_decoder.submit_access_unit(access_unit)? {
            black_box(sample_checksum(&frame));
        }
    }
    drop(warmup_decoder);

    let mut frames = 0_u64;
    let mut checksum = 0_u64;
    let mut durations = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = Instant::now();
        let mut decoder = H264422P10Decoder::new(profile.clone());
        for access_unit in &access_units {
            for frame in decoder.submit_access_unit(access_unit)? {
                checksum = checksum.wrapping_add(sample_checksum(&frame));
                frames += 1;
            }
        }
        for frame in decoder.flush()? {
            checksum = checksum.wrapping_add(sample_checksum(&frame));
            frames += 1;
        }
        durations.push(start.elapsed());
    }
    let seconds: f64 = durations.iter().map(|duration| duration.as_secs_f64()).sum();
    let fps = if seconds > 0.0 {
        frames as f64 / seconds
    } else {
        0.0
    };

    println!("input: {}", input.display());
    println!("bytes: {}", bytes.len());
    println!("access units: {}", access_units.len());
    println!("repeats: {repeats}");
    println!("frames: {frames}");
    println!("seconds: {seconds:.3}");
    println!("fps: {fps:.2}");
    let mut per_gop: Vec<f64> = durations.iter().map(|duration| duration.as_secs_f64()).collect();
    per_gop.sort_by(f64::total_cmp);
    let median = if repeats % 2 == 0 {
        (per_gop[repeats / 2 - 1] + per_gop[repeats / 2]) / 2.0
    } else {
        per_gop[repeats / 2]
    };
    println!("median_gop_seconds: {median:.3}");
    println!("checksum: {}", black_box(checksum));
    Ok(())
}

fn sample_checksum(frame: &H264422P10Frame) -> u64 {
    let y = &frame.y.data;
    let cb = &frame.cb.data;
    let cr = &frame.cr.data;
    let mut sum = y.len() as u64;
    if let Some(byte) = y.first() {
        sum = sum.wrapping_add(u64::from(*byte));
    }
    if let Some(byte) = y.last() {
        sum = sum.wrapping_add(u64::from(*byte));
    }
    if let Some(byte) = cb.first() {
        sum = sum.wrapping_add(u64::from(*byte));
    }
    if let Some(byte) = cr.first() {
        sum = sum.wrapping_add(u64::from(*byte));
    }
    sum
}

struct NalSpan {
    start: usize,
    header: usize,
    end: usize,
}

fn split_h264_annex_b_access_units(data: &[u8]) -> Result<Vec<&[u8]>, Box<dyn std::error::Error>> {
    let nals = annex_b_nals(data)?;
    let mut access_units = Vec::new();
    let mut au_start = 0_usize;
    let mut prefix_start = 0_usize;
    let mut seen_vcl = false;
    for nal in &nals {
        let nal_type = data[nal.header] & 0x1f;
        let is_vcl = nal_type == 1 || nal_type == 5;
        let starts_picture =
            nal_type == 9 || (is_vcl && seen_vcl && slice_starts_at_first_macroblock(data, nal));
        if starts_picture && seen_vcl {
            let cut = if nal_type == 9 {
                nal.start
            } else {
                prefix_start
            };
            if cut > au_start {
                access_units.push(&data[au_start..cut]);
            }
            au_start = cut;
            if nal_type == 9 {
                seen_vcl = false;
            }
        }
        if is_vcl {
            seen_vcl = true;
            prefix_start = nal.end;
        }
    }
    if au_start < data.len() {
        access_units.push(&data[au_start..]);
    }
    if access_units.is_empty() {
        return Err("input is not Annex B H.264".into());
    }
    Ok(access_units)
}

fn slice_starts_at_first_macroblock(data: &[u8], nal: &NalSpan) -> bool {
    let rbsp = nal.header + 1;
    rbsp < nal.end && data[rbsp] & 0x80 != 0
}

fn annex_b_nals(data: &[u8]) -> Result<Vec<NalSpan>, Box<dyn std::error::Error>> {
    let mut starts = Vec::new();
    let mut index = 0_usize;
    while index + 3 <= data.len() {
        if data[index..].starts_with(&[0, 0, 1]) {
            starts.push((index, 3));
            index += 3;
        } else if index + 4 <= data.len() && data[index..].starts_with(&[0, 0, 0, 1]) {
            starts.push((index, 4));
            index += 4;
        } else {
            index += 1;
        }
    }
    if starts.is_empty() {
        return Err("input is not Annex B H.264".into());
    }
    Ok(starts
        .iter()
        .enumerate()
        .filter_map(|(position, (start, prefix_len))| {
            let header = start + prefix_len;
            let end = starts
                .get(position + 1)
                .map(|(next, _)| *next)
                .unwrap_or(data.len());
            (header < end).then_some(NalSpan {
                start: *start,
                header,
                end,
            })
        })
        .collect())
}
