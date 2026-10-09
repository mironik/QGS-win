# QGS-win

Windows line of [QGS](https://github.com/mironik/QGS).

Linux device backends stay in this tree and compile on Linux: Unix sockets, VAAPI, Vulkan/DRM, PipeWire, and Wayland. On Windows those crates are empty and the workspace builds the portable media crates plus the Windows host.

## Windows host

```text
cargo run -p qgs-win -- serve
cargo run -p qgs-win -- audio-devices
cargo run -p qgs-win -- play-tone
```

`serve` listens on `127.0.0.1:47651` and speaks the QGS wire protocol for session setup and device enumeration. This host does not export GPU handles. `audio-devices` and `play-tone` use the default WASAPI shared-mode render endpoint.
