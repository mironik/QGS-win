//! Compare QGS High 4:2:2 10-bit reconstruction with an ffmpeg raw yuv422p10le decode
//! of the same Annex B bitstream. This example does not launch ffmpeg.

use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

use qgs_codec_h264::parse_annex_b_access_unit;
use qgs_h264_422p10::{H264422P10Decoder, H264422P10Frame, H264422P10Profile};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let input = args
        .next()
        .map(PathBuf::from)
        .ok_or("usage: compare_ffmpeg <annex-b.h264> <ffmpeg-yuv422p10le>")?;
    let reference_path = args
        .next()
        .map(PathBuf::from)
        .ok_or("usage: compare_ffmpeg <annex-b.h264> <ffmpeg-yuv422p10le>")?;

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

    let mut decoder = H264422P10Decoder::new(profile.clone());
    let mut frames = Vec::new();
    let mut decode_error = None;
    for (index, access_unit) in access_units.iter().enumerate() {
        match decoder.submit_access_unit(access_unit) {
            Ok(outputs) => frames.extend(outputs),
            Err(error) => {
                decode_error = Some((index, error.to_string()));
                break;
            }
        }
    }
    if decode_error.is_none() {
        match decoder.flush() {
            Ok(outputs) => frames.extend(outputs),
            Err(error) => decode_error = Some((access_units.len(), error.to_string())),
        }
    }

    let mut reference = Vec::new();
    File::open(&reference_path)?.read_to_end(&mut reference)?;
    let coded_bytes = yuv422p10le_frame_bytes(profile.coded_width, profile.coded_height)?;
    let visible_bytes = yuv422p10le_frame_bytes(profile.visible_width, profile.visible_height)?;
    let (frame_bytes, reference_layout) = if coded_bytes != 0 && reference.len() % coded_bytes == 0
    {
        (coded_bytes, ReferenceLayout::Coded)
    } else if visible_bytes != 0 && reference.len() % visible_bytes == 0 {
        (visible_bytes, ReferenceLayout::Visible)
    } else {
        return Err(format!(
            "ffmpeg yuv size {} matches neither coded frame {coded_bytes} nor visible frame {visible_bytes}",
            reference.len()
        )
        .into());
    };
    let reference_frames = reference.len() / frame_bytes;
    let trailing = reference.len() % frame_bytes;

    println!("input: {}", input.display());
    println!("ffmpeg reference: {}", reference_path.display());
    println!(
        "profile: {:?} {}-bit {:?} coded={}x{} visible={}x{}",
        profile.profile,
        profile.bit_depth,
        profile.chroma,
        profile.coded_width,
        profile.coded_height,
        profile.visible_width,
        profile.visible_height
    );
    println!("access units: {}", access_units.len());
    println!("qgs frames: {}", frames.len());
    println!(
        "ffmpeg frames: {reference_frames} layout={reference_layout:?} trailing_bytes={trailing} frame_bytes={frame_bytes}"
    );
    if let Some((index, error)) = &decode_error {
        println!("qgs stopped at access unit {index}: {error}");
    }

    let compared = frames.len().min(reference_frames);
    let mut exact_frames = 0_usize;
    for index in 0..compared {
        let qgs = &frames[index];
        let ffmpeg = &reference[index * frame_bytes..(index + 1) * frame_bytes];
        let report = compare_frame(qgs, ffmpeg, reference_layout);
        if report.exact {
            exact_frames += 1;
        }
        println!(
            "frame {index:02}: {} max_abs={} mismatched={}/{} psnr_y={} psnr_cb={} psnr_cr={}",
            if report.exact { "exact" } else { "differ" },
            report.max_abs,
            report.mismatched,
            report.samples,
            format_psnr(report.psnr_y),
            format_psnr(report.psnr_cb),
            format_psnr(report.psnr_cr)
        );
        if let Some(first) = report.first_mismatch {
            println!(
                "  first: plane={} x={} y={} qgs={} ffmpeg={}",
                first.plane, first.x, first.y, first.qgs, first.ffmpeg
            );
        }
    }

    if frames.len() != reference_frames {
        println!(
            "frame count: qgs {} , ffmpeg {reference_frames}",
            frames.len()
        );
    }
    println!("exact frames: {exact_frames}/{compared}");
    if compared > 1 && exact_frames != compared {
        println!("best luma match of each QGS frame against every ffmpeg frame:");
        for qgs_index in 0..compared {
            let mut best = (0_usize, u64::MAX);
            for ffmpeg_index in 0..compared {
                let distance = luma_distance(
                    &frames[qgs_index],
                    &reference[ffmpeg_index * frame_bytes..(ffmpeg_index + 1) * frame_bytes],
                    reference_layout,
                );
                if distance < best.1 {
                    best = (ffmpeg_index, distance);
                }
            }
            println!(
                "  qgs {qgs_index:02} -> ffmpeg {:02}  sampled_abs_sum={}",
                best.0, best.1
            );
        }
    }

    if decode_error.is_some() || exact_frames != compared || frames.len() != reference_frames {
        std::process::exit(1);
    }
    Ok(())
}

struct SampleMismatch {
    plane: &'static str,
    x: u32,
    y: u32,
    qgs: u16,
    ffmpeg: u16,
}

struct FrameReport {
    exact: bool,
    max_abs: u16,
    mismatched: usize,
    samples: usize,
    psnr_y: Option<f64>,
    psnr_cb: Option<f64>,
    psnr_cr: Option<f64>,
    first_mismatch: Option<SampleMismatch>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReferenceLayout {
    Coded,
    Visible,
}

fn compare_frame(qgs: &H264422P10Frame, ffmpeg: &[u8], layout: ReferenceLayout) -> FrameReport {
    let (y_width, y_height, c_width, c_height) = match layout {
        ReferenceLayout::Coded => (
            qgs.coded_width,
            qgs.coded_height,
            qgs.coded_width / 2,
            qgs.coded_height,
        ),
        ReferenceLayout::Visible => (
            qgs.visible_width,
            qgs.visible_height,
            qgs.visible_width / 2,
            qgs.visible_height,
        ),
    };
    let y_bytes = (y_width as usize) * (y_height as usize) * 2;
    let c_bytes = (c_width as usize) * (c_height as usize) * 2;
    let (y_ref, rest) = ffmpeg.split_at(y_bytes);
    let (cb_ref, cr_ref) = rest.split_at(c_bytes);
    debug_assert_eq!(cr_ref.len(), c_bytes);

    let y = compare_plane(
        "Y",
        y_width,
        y_height,
        qgs.y.width_samples,
        &qgs.y.data,
        y_ref,
    );
    let cb = compare_plane(
        "Cb",
        c_width,
        c_height,
        qgs.cb.width_samples,
        &qgs.cb.data,
        cb_ref,
    );
    let cr = compare_plane(
        "Cr",
        c_width,
        c_height,
        qgs.cr.width_samples,
        &qgs.cr.data,
        cr_ref,
    );
    let mismatched = y.mismatched + cb.mismatched + cr.mismatched;
    FrameReport {
        exact: mismatched == 0,
        max_abs: y.max_abs.max(cb.max_abs).max(cr.max_abs),
        mismatched,
        samples: y.samples + cb.samples + cr.samples,
        psnr_y: y.psnr,
        psnr_cb: cb.psnr,
        psnr_cr: cr.psnr,
        first_mismatch: y.first.or(cb.first).or(cr.first),
    }
}

struct PlaneReport {
    mismatched: usize,
    samples: usize,
    max_abs: u16,
    psnr: Option<f64>,
    first: Option<SampleMismatch>,
}

fn compare_plane(
    plane: &'static str,
    width: u32,
    height: u32,
    qgs_stride_samples: u32,
    qgs: &[u8],
    ffmpeg: &[u8],
) -> PlaneReport {
    let mut mismatched = 0_usize;
    let mut max_abs = 0_u16;
    let mut error_sq = 0_f64;
    let mut first = None;
    let samples = (width as usize) * (height as usize);
    for y in 0..height {
        for x in 0..width {
            let qgs_offset = ((y as usize) * (qgs_stride_samples as usize) + (x as usize)) * 2;
            let ffmpeg_offset = ((y as usize) * (width as usize) + (x as usize)) * 2;
            let qgs_sample = u16::from_le_bytes([qgs[qgs_offset], qgs[qgs_offset + 1]]);
            let ffmpeg_sample =
                u16::from_le_bytes([ffmpeg[ffmpeg_offset], ffmpeg[ffmpeg_offset + 1]]);
            let delta = qgs_sample.abs_diff(ffmpeg_sample);
            if delta != 0 {
                mismatched += 1;
                max_abs = max_abs.max(delta);
                error_sq += f64::from(delta) * f64::from(delta);
                if first.is_none() {
                    first = Some(SampleMismatch {
                        plane,
                        x,
                        y,
                        qgs: qgs_sample,
                        ffmpeg: ffmpeg_sample,
                    });
                }
            }
        }
    }
    let psnr = if mismatched == 0 {
        None
    } else {
        Some(10.0 * ((1023.0 * 1023.0) / (error_sq / samples as f64)).log10())
    };
    PlaneReport {
        mismatched,
        samples,
        max_abs,
        psnr,
        first,
    }
}

fn luma_distance(qgs: &H264422P10Frame, ffmpeg: &[u8], layout: ReferenceLayout) -> u64 {
    let (width, height) = match layout {
        ReferenceLayout::Coded => (qgs.coded_width, qgs.coded_height),
        ReferenceLayout::Visible => (qgs.visible_width, qgs.visible_height),
    };
    let mut distance = 0_u64;
    let mut y = 0_u32;
    while y < height {
        let mut x = 0_u32;
        while x < width {
            let qgs_offset = ((y as usize) * (qgs.y.width_samples as usize) + (x as usize)) * 2;
            let ffmpeg_offset = ((y as usize) * (width as usize) + (x as usize)) * 2;
            let qgs_sample =
                u16::from_le_bytes([qgs.y.data[qgs_offset], qgs.y.data[qgs_offset + 1]]);
            let ffmpeg_sample =
                u16::from_le_bytes([ffmpeg[ffmpeg_offset], ffmpeg[ffmpeg_offset + 1]]);
            distance += u64::from(qgs_sample.abs_diff(ffmpeg_sample));
            x += 32;
        }
        y += 32;
    }
    distance
}

fn format_psnr(value: Option<f64>) -> String {
    match value {
        None => "inf".to_string(),
        Some(value) => format!("{value:.2}"),
    }
}

fn yuv422p10le_frame_bytes(width: u32, height: u32) -> Result<usize, Box<dyn std::error::Error>> {
    let y = (width as usize)
        .checked_mul(height as usize)
        .and_then(|samples| samples.checked_mul(2))
        .ok_or("frame size overflow")?;
    let chroma = (width as usize / 2)
        .checked_mul(height as usize)
        .and_then(|samples| samples.checked_mul(2))
        .ok_or("frame size overflow")?;
    y.checked_add(chroma)
        .and_then(|total| total.checked_add(chroma))
        .ok_or_else(|| "frame size overflow".into())
}

struct NalSpan {
    start: usize,
    header: usize,
    end: usize,
}

fn split_h264_annex_b_access_units(
    data: &[u8],
) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
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
                access_units.push(data[au_start..cut].to_vec());
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
        access_units.push(data[au_start..].to_vec());
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
        .map(|(position, (start, prefix_len))| {
            let header = start + prefix_len;
            let end = starts
                .get(position + 1)
                .map(|(next, _)| *next)
                .unwrap_or(data.len());
            NalSpan {
                start: *start,
                header,
                end,
            }
        })
        .filter(|nal| nal.header < nal.end)
        .collect())
}
