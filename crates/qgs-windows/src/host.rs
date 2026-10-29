use std::io::{self, Read, Write};
use std::net::SocketAddr;

use qgs_core::{Session, SessionManager};
use qgs_protocol::{DeviceListResponse, ErrorResponse, ProtocolErrorCode, WireMessage};

use crate::transport::{self, TransportError};

pub const DEFAULT_PORT: u16 = 47651;

pub fn default_endpoint() -> SocketAddr {
    SocketAddr::from(([127, 0, 0, 1], DEFAULT_PORT))
}

pub fn serve(addr: SocketAddr) -> io::Result<()> {
    let listener = transport::bind_listener(addr)?;
    let bound = listener.local_addr()?;
    eprintln!("qgs-win listening on {bound}");
    for stream in listener.incoming() {
        let mut stream = stream?;
        if let Err(err) = handle_client(&mut stream) {
            eprintln!("qgs-win client closed: {err}");
        }
    }
    Ok(())
}

pub fn handle_client(stream: &mut (impl Read + Write)) -> Result<(), TransportError> {
    let sessions = SessionManager::new();
    let mut session: Option<Session> = None;

    loop {
        let message = match transport::receive_message(stream) {
            Ok(message) => message,
            Err(TransportError::Io(err)) if err.kind() == io::ErrorKind::UnexpectedEof => {
                return Ok(());
            }
            Err(TransportError::Protocol(err)) => {
                transport::send_message(
                    stream,
                    &WireMessage::Error {
                        request_id: 0,
                        response: ErrorResponse {
                            code: ProtocolErrorCode::from(&err),
                        },
                    },
                )?;
                continue;
            }
            Err(TransportError::Io(err)) => return Err(err.into()),
        };

        let response = respond(&sessions, &mut session, message);
        transport::send_message(stream, &response)?;
    }
}

fn respond(
    sessions: &SessionManager,
    session: &mut Option<Session>,
    message: WireMessage,
) -> WireMessage {
    let request_id = message.request_id();
    match message {
        WireMessage::Hello { request, .. } => match sessions.create_session() {
            Ok(created) => match qgs_protocol::handle_hello(&request, created.id()) {
                Ok(response) => {
                    *session = Some(created);
                    WireMessage::Welcome {
                        request_id,
                        response,
                    }
                }
                Err(err) => error_message(request_id, ProtocolErrorCode::from(&err)),
            },
            Err(err) => error_message(request_id, ProtocolErrorCode::from(&err)),
        },
        WireMessage::EnumerateDevices { .. } => {
            if session.is_none() {
                return error_message(request_id, ProtocolErrorCode::SessionRequired);
            }
            WireMessage::DeviceList {
                request_id,
                response: DeviceListResponse {
                    devices: Vec::new(),
                },
            }
        }
        WireMessage::QueryDeviceCapabilities { .. }
        | WireMessage::QueryVideoCapabilities { .. } => {
            if session.is_none() {
                error_message(request_id, ProtocolErrorCode::SessionRequired)
            } else {
                error_message(request_id, ProtocolErrorCode::UnknownDeviceId)
            }
        }
        WireMessage::CreateDecoder { .. }
        | WireMessage::SubmitAccessUnit { .. }
        | WireMessage::FlushDecoder { .. }
        | WireMessage::DestroyDecoder { .. } => session_or(
            session,
            request_id,
            ProtocolErrorCode::UnsupportedDecodeConfiguration,
        ),
        WireMessage::CreateBuffer { .. }
        | WireMessage::CreateImage { .. }
        | WireMessage::DestroyResource { .. }
        | WireMessage::ExportResource { .. }
        | WireMessage::CreateSync { .. }
        | WireMessage::ExportSync { .. } => session_or(
            session,
            request_id,
            ProtocolErrorCode::UnsupportedExternalHandleType,
        ),
        WireMessage::Welcome { .. }
        | WireMessage::Error { .. }
        | WireMessage::DeviceList { .. }
        | WireMessage::DeviceCapabilities { .. }
        | WireMessage::VideoCapabilities { .. }
        | WireMessage::DecoderCreated { .. }
        | WireMessage::DecodeOutput { .. }
        | WireMessage::DecoderDestroyed { .. }
        | WireMessage::BufferCreated { .. }
        | WireMessage::ImageCreated { .. }
        | WireMessage::ResourceDestroyed { .. }
        | WireMessage::ResourceExported { .. }
        | WireMessage::SyncCreated { .. }
        | WireMessage::SyncExported { .. } => {
            error_message(request_id, ProtocolErrorCode::UnknownOpcode)
        }
    }
}

fn session_or(session: &Option<Session>, request_id: u64, code: ProtocolErrorCode) -> WireMessage {
    if session.is_none() {
        error_message(request_id, ProtocolErrorCode::SessionRequired)
    } else {
        error_message(request_id, code)
    }
}

fn error_message(request_id: u64, code: ProtocolErrorCode) -> WireMessage {
    WireMessage::Error {
        request_id,
        response: ErrorResponse { code },
    }
}

#[cfg(test)]
mod tests {
    use std::net::TcpStream;
    use std::thread;

    use qgs_protocol::{HelloRequest, WireMessage, CURRENT_PROTOCOL_VERSION};

    use super::*;
    use crate::transport::{bind_listener, connect_endpoint, receive_message, send_message};

    #[test]
    fn hello_and_empty_device_list_over_tcp() {
        let listener = bind_listener(SocketAddr::from(([127, 0, 0, 1], 0))).expect("bind");
        let addr = listener.local_addr().expect("addr");
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            handle_client(&mut stream).expect("client");
        });

        let mut client = connect_endpoint(addr).expect("connect");
        send_message(
            &mut client,
            &WireMessage::Hello {
                request_id: 7,
                request: HelloRequest::current(),
            },
        )
        .expect("hello");
        match receive_message(&mut client).expect("welcome") {
            WireMessage::Welcome {
                request_id,
                response,
            } => {
                assert_eq!(request_id, 7);
                assert_eq!(response.version, CURRENT_PROTOCOL_VERSION);
                assert_eq!(response.session_id.get(), 1);
            }
            other => panic!("expected welcome, got {other:?}"),
        }

        send_message(
            &mut client,
            &WireMessage::EnumerateDevices { request_id: 8 },
        )
        .expect("enumerate");
        match receive_message(&mut client).expect("devices") {
            WireMessage::DeviceList {
                request_id,
                response,
            } => {
                assert_eq!(request_id, 8);
                assert!(response.devices.is_empty());
            }
            other => panic!("expected device list, got {other:?}"),
        }

        drop(client);
        server.join().expect("server");
    }

    #[test]
    fn enumerate_before_hello_requires_session() {
        let listener = bind_listener(SocketAddr::from(([127, 0, 0, 1], 0))).expect("bind");
        let addr = listener.local_addr().expect("addr");
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            handle_client(&mut stream).expect("client");
        });

        let mut client: TcpStream = connect_endpoint(addr).expect("connect");
        send_message(
            &mut client,
            &WireMessage::EnumerateDevices { request_id: 3 },
        )
        .expect("enumerate");
        match receive_message(&mut client).expect("error") {
            WireMessage::Error {
                request_id,
                response,
            } => {
                assert_eq!(request_id, 3);
                assert_eq!(
                    response.code,
                    qgs_protocol::ProtocolErrorCode::SessionRequired
                );
            }
            other => panic!("expected error, got {other:?}"),
        }
        drop(client);
        server.join().expect("server");
    }
}
