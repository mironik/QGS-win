#![forbid(unsafe_code)]

#[cfg(unix)]
include!("linux.rs");

/// The FFmpeg software decoder link is not built on this target.
#[cfg(not(unix))]
pub fn available() -> bool {
    false
}
