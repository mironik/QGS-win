#![forbid(unsafe_code)]

#[cfg(unix)]
include!("linux_main.rs");

#[cfg(not(unix))]
fn main() {
    eprintln!("qgs-test is the Linux harness. On Windows run qgs-win.");
    std::process::exit(2);
}
