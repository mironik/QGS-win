use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};

use qgs_protocol::{
    decode_wire_header, decode_wire_message_parts, encode_wire_message, ProtocolError, WireMessage,
    MAX_PAYLOAD_LEN, WIRE_HEADER_LEN,
};

pub fn bind_listener(addr: SocketAddr) -> io::Result<TcpListener> {
    TcpListener::bind(addr)
}

pub fn connect_endpoint(addr: SocketAddr) -> io::Result<TcpStream> {
    TcpStream::connect(addr)
}

pub fn send_message(stream: &mut impl Write, message: &WireMessage) -> io::Result<()> {
    stream.write_all(&encode_wire_message(message))
}

pub fn receive_message(stream: &mut impl Read) -> Result<WireMessage, TransportError> {
    let mut header_bytes = [0_u8; WIRE_HEADER_LEN];
    stream.read_exact(&mut header_bytes)?;
    let header = decode_wire_header(&header_bytes)?;

    if header.payload_len > MAX_PAYLOAD_LEN {
        return Err(ProtocolError::PayloadTooLarge {
            len: header.payload_len,
            max: MAX_PAYLOAD_LEN,
        }
        .into());
    }

    let mut payload = vec![0_u8; header.payload_len as usize];
    stream.read_exact(&mut payload)?;
    Ok(decode_wire_message_parts(header, &payload)?)
}

#[derive(Debug)]
pub enum TransportError {
    Io(io::Error),
    Protocol(ProtocolError),
}

impl std::fmt::Display for TransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "transport I/O error: {err}"),
            Self::Protocol(err) => write!(f, "transport protocol error: {err}"),
        }
    }
}

impl std::error::Error for TransportError {}

impl From<io::Error> for TransportError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<ProtocolError> for TransportError {
    fn from(value: ProtocolError) -> Self {
        Self::Protocol(value)
    }
}
