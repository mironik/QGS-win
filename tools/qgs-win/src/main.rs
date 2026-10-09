#![forbid(unsafe_code)]

use std::net::SocketAddr;
use std::process::ExitCode;

use qgs_windows::{
    default_endpoint, play_f32le_interleaved, render_endpoints, serve, sine_f32le_interleaved,
};

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("serve") => serve_command(args),
        Some("audio-devices") => audio_devices_command(),
        Some("play-tone") => play_tone_command(args),
        Some("--help") | Some("-h") | Some("help") => {
            print_usage();
            ExitCode::SUCCESS
        }
        _ => {
            print_usage();
            ExitCode::from(2)
        }
    }
}

fn serve_command(args: impl Iterator<Item = String>) -> ExitCode {
    let mut addr = default_endpoint();
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        if arg == "--addr" {
            let Some(value) = args.next() else {
                eprintln!("qgs-win: --addr needs HOST:PORT");
                return ExitCode::from(2);
            };
            match value.parse::<SocketAddr>() {
                Ok(parsed) => addr = parsed,
                Err(err) => {
                    eprintln!("qgs-win: invalid address {value}: {err}");
                    return ExitCode::from(2);
                }
            }
        } else {
            eprintln!("qgs-win: unknown argument {arg}");
            return ExitCode::from(2);
        }
    }

    match serve(addr) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("qgs-win: {err}");
            ExitCode::from(1)
        }
    }
}

fn audio_devices_command() -> ExitCode {
    match render_endpoints() {
        Ok(endpoints) if endpoints.is_empty() => {
            println!("no WASAPI render endpoints");
            ExitCode::SUCCESS
        }
        Ok(endpoints) => {
            for endpoint in endpoints {
                let mark = if endpoint.is_default { "default" } else { "-" };
                println!("{mark}\t{}\t{}", endpoint.name, endpoint.id);
            }
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("qgs-win: {err}");
            ExitCode::from(1)
        }
    }
}

fn play_tone_command(args: impl Iterator<Item = String>) -> ExitCode {
    let mut seconds: f32 = 1.0;
    let mut frequency: f32 = 440.0;
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        let Some(value) = args.next() else {
            eprintln!("qgs-win: {arg} needs a number");
            return ExitCode::from(2);
        };
        let parsed: f32 = match value.parse() {
            Ok(parsed) => parsed,
            Err(_) => {
                eprintln!("qgs-win: {arg} needs a number, got {value}");
                return ExitCode::from(2);
            }
        };
        match arg.as_str() {
            "--seconds" => seconds = parsed,
            "--frequency" => frequency = parsed,
            _ => {
                eprintln!("qgs-win: unknown argument {arg}");
                return ExitCode::from(2);
            }
        }
    }
    if !(0.05..30.0).contains(&seconds) {
        eprintln!("qgs-win: --seconds must be between 0.05 and 30");
        return ExitCode::from(2);
    }

    let sample_rate = 48_000_u32;
    let frames = (f64::from(seconds) * f64::from(sample_rate)).round() as u32;
    let pcm = match sine_f32le_interleaved(sample_rate, 2, frequency, frames.max(1)) {
        Ok(pcm) => pcm,
        Err(err) => {
            eprintln!("qgs-win: {err}");
            return ExitCode::from(2);
        }
    };
    match play_f32le_interleaved(sample_rate, 2, &pcm) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("qgs-win: {err}");
            ExitCode::from(1)
        }
    }
}

fn print_usage() {
    eprintln!(
        "\
qgs-win serve [--addr 127.0.0.1:47651]
qgs-win audio-devices
qgs-win play-tone [--seconds 1] [--frequency 440]"
    );
}
