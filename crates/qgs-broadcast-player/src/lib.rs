#![forbid(unsafe_code)]

#[cfg(unix)]
include!("linux.rs");

/// The Linux broadcast-player bricks are not built on this target.
#[cfg(not(unix))]
pub fn available() -> bool {
    false
}
