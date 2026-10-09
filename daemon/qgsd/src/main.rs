#![forbid(unsafe_code)]

#[cfg(unix)]
include!("linux_main.rs");

#[cfg(not(unix))]
fn main() {
    eprintln!(
        "qgsd is the Linux daemon. On Windows run: cargo run -p qgs-win -- serve"
    );
    std::process::exit(2);
}
