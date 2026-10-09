#![forbid(unsafe_code)]

#[cfg(unix)]
include!("linux.rs");

/// Native PipeWire playback is not built on this target.
#[cfg(not(unix))]
pub fn available() -> bool {
    false
}
