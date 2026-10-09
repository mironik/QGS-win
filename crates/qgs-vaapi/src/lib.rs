#![forbid(unsafe_code)]

#[cfg(unix)]
include!("linux.rs");

/// VAAPI decode is not built on this target.
#[cfg(not(unix))]
pub fn available() -> bool {
    false
}
