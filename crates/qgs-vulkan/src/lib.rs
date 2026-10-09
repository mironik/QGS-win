#[cfg(unix)]
include!("linux.rs");

/// The Linux Vulkan/DRM backend is not built on this target.
#[cfg(not(unix))]
pub fn available() -> bool {
    false
}
