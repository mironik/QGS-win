#![forbid(unsafe_code)]

#[cfg(unix)]
include!("linux.rs");

/// The Linux transport and PipeWire probe crate is not built on this target.
#[cfg(not(unix))]
pub fn available() -> bool {
    false
}
