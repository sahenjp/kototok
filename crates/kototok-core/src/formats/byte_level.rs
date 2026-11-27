//! The GPT-2 ByteLevel alphabet and the RFC 4648 alphabet used by rank files.

use crate::KototokError;

pub(super) fn bytes_to_unicode(byte: u8) -> char {
    let codepoint = match byte {
        0..=32 => 256 + u32::from(byte),
        127..=160 => 289 + u32::from(byte - 127),
        173 => 323,
        _ => u32::from(byte),
    };
    char::from_u32(codepoint).expect("ByteLevel codepoints are Unicode scalars")
}

pub(super) fn unicode_to_byte(ch: char) -> Option<u8> {
    match ch as u32 {
        codepoint @ (33..=126 | 161..=172 | 174..=255) => Some(codepoint as u8),
        codepoint @ 256..=288 => Some((codepoint - 256) as u8),
        codepoint @ 289..=322 => Some((codepoint - 289) as u8 + 127),
        323 => Some(173),
        _ => None,
    }
}

pub(super) fn encode_byte_level(bytes: &[u8]) -> String {
    bytes.iter().copied().map(bytes_to_unicode).collect()
}

pub(super) fn decode_byte_level(s: &str) -> Option<Vec<u8>> {
    s.chars().map(unicode_to_byte).collect()
}

const BASE64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

pub(super) fn base64_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let a = chunk[0];
        let b = chunk.get(1).copied().unwrap_or(0);
        let c = chunk.get(2).copied().unwrap_or(0);
        out.push(BASE64[(a >> 2) as usize] as char);
        out.push(BASE64[((a & 3) << 4 | b >> 4) as usize] as char);
        out.push(if chunk.len() >= 2 {
            BASE64[((b & 15) << 2 | c >> 6) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() == 3 {
            BASE64[(c & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}

fn base64_digit(byte: u8) -> Option<u8> {
    match byte {
        b'A'..=b'Z' => Some(byte - b'A'),
        b'a'..=b'z' => Some(byte - b'a' + 26),
        b'0'..=b'9' => Some(byte - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

pub(super) fn base64_decode(s: &str) -> Result<Vec<u8>, KototokError> {
    let invalid = || super::invalid("invalid RFC 4648 base64");
    let (chunks, remainder) = s.as_bytes().as_chunks::<4>();
    if !remainder.is_empty() {
        return Err(invalid());
    }
    let mut chunks = chunks.iter();
    let mut out = Vec::with_capacity(chunks.len() * 3);
    while let Some(&[a, b, c, d]) = chunks.next() {
        let a = base64_digit(a).ok_or_else(invalid)?;
        let b = base64_digit(b).ok_or_else(invalid)?;
        out.push(a << 2 | b >> 4);
        match (c, d) {
            (b'=', b'=') if chunks.len() == 0 && b & 15 == 0 => break,
            (c, b'=') if chunks.len() == 0 => {
                let c = base64_digit(c).filter(|c| c & 3 == 0).ok_or_else(invalid)?;
                out.push(b << 4 | c >> 2);
                break;
            }
            (c, d) => {
                let c = base64_digit(c).ok_or_else(invalid)?;
                let d = base64_digit(d).ok_or_else(invalid)?;
                out.push(b << 4 | c >> 2);
                out.push(c << 6 | d);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_level_alphabet() {
        let bytes: Vec<u8> = (0..=255).collect();
        let encoded = encode_byte_level(&bytes);
        assert_eq!(
            encoded
                .chars()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            256
        );
        assert_eq!(decode_byte_level(&encoded), Some(bytes));
        assert_eq!(encode_byte_level(b" \n\xad\xff"), "ĠĊŃÿ");
        let utf8 = "日本語とRust 🦀\0".as_bytes();
        assert_eq!(
            decode_byte_level(&encode_byte_level(utf8)).as_deref(),
            Some(utf8)
        );
        assert_eq!(decode_byte_level("日"), None);
    }

    #[test]
    fn base64_roundtrip_and_padding() {
        for (input, expected) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
        ] {
            assert_eq!(base64_encode(input.as_bytes()), expected);
            assert_eq!(base64_decode(expected).unwrap(), input.as_bytes());
        }
        let binary: Vec<u8> = (0..=255).collect();
        assert_eq!(base64_decode(&base64_encode(&binary)).unwrap(), binary);
        for input in [
            "Z", "Zg", "Zg=", "Zg==AAAA", "==AA", "Zg=A", "Zg==\n", "Zh==", "Zm9=", "Z-8=", "🦀",
        ] {
            assert!(
                matches!(base64_decode(input), Err(KototokError::InvalidFormat(_))),
                "{input:?}"
            );
        }
    }
}
