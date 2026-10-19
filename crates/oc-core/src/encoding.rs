//! Compact, plain-text storage for large byte arrays in a scene file.
//!
//! A DMD's 1920 × 1080 bitmap or an SLM's grey map does not belong in JSON as
//! a list of numbers. It is stored as PackBits run-length encoding (the
//! scheme of Apple Technical Note TN1023, "Understanding PackBits", 1996),
//! then base64 (IETF RFC 4648, §4, standard alphabet with padding), with an
//! FNV-1a 64-bit checksum of the raw bytes (G. Fowler, L. C. Noll, K.-P. Vo,
//! the Fowler–Noll–Vo hash; offset basis 0xcbf29ce484222325, prime
//! 0x100000001b3) so a damaged or hand-edited pattern is refused on load
//! rather than displayed wrong.
//!
//! All three are a few lines each and are written here, so `oc-core` takes no
//! dependency for them.

/// PackBits: a header byte `n` in `0..=127` copies the next `n + 1` bytes
/// literally; `n` in `-127..=-1` (as `i8`) repeats the next byte `1 − n`
/// times; `-128` is a no-op.
pub fn packbits_encode(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() / 2 + 2);
    let mut i = 0;
    while i < data.len() {
        // A run of at least two equal bytes?
        let mut run = 1;
        while i + run < data.len() && data[i + run] == data[i] && run < 128 {
            run += 1;
        }
        if run >= 2 {
            out.push((1i16 - run as i16) as i8 as u8);
            out.push(data[i]);
            i += run;
            continue;
        }
        // A literal stretch, up to the next run of two or 128 bytes.
        let start = i;
        let mut len = 0;
        while i < data.len() && len < 128 {
            if i + 1 < data.len() && data[i] == data[i + 1] {
                break;
            }
            i += 1;
            len += 1;
        }
        out.push((len - 1) as u8);
        out.extend_from_slice(&data[start..start + len]);
    }
    out
}

/// The inverse of [`packbits_encode`]. `None` when the stream is truncated.
pub fn packbits_decode(data: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(data.len() * 2);
    let mut i = 0;
    while i < data.len() {
        let h = data[i] as i8;
        i += 1;
        if h >= 0 {
            let n = h as usize + 1;
            out.extend_from_slice(data.get(i..i + n)?);
            i += n;
        } else if h != -128 {
            let n = (1 - h as i16) as usize;
            let b = *data.get(i)?;
            i += 1;
            out.extend(std::iter::repeat(b).take(n));
        }
    }
    Some(out)
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Base64, RFC 4648 §4, with padding.
pub fn base64_encode(data: &[u8]) -> String {
    let mut s = String::with_capacity(data.len().div_ceil(3) * 4);
    for c in data.chunks(3) {
        let b = [c[0], *c.get(1).unwrap_or(&0), *c.get(2).unwrap_or(&0)];
        let v = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
        s.push(B64[(v >> 18) as usize & 63] as char);
        s.push(B64[(v >> 12) as usize & 63] as char);
        s.push(if c.len() > 1 { B64[(v >> 6) as usize & 63] as char } else { '=' });
        s.push(if c.len() > 2 { B64[v as usize & 63] as char } else { '=' });
    }
    s
}

/// The inverse of [`base64_encode`]. `None` for a character outside the
/// alphabet or a length that is not a multiple of four.
pub fn base64_decode(s: &str) -> Option<Vec<u8>> {
    let bytes = s.as_bytes();
    if bytes.len() % 4 != 0 {
        return None;
    }
    let val = |c: u8| -> Option<u32> {
        match c {
            b'A'..=b'Z' => Some((c - b'A') as u32),
            b'a'..=b'z' => Some((c - b'a') as u32 + 26),
            b'0'..=b'9' => Some((c - b'0') as u32 + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    };
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for q in bytes.chunks(4) {
        let pad = q.iter().rev().take_while(|c| **c == b'=').count();
        if pad > 2 {
            return None;
        }
        let mut v = 0u32;
        for (k, c) in q.iter().enumerate() {
            v <<= 6;
            if k < 4 - pad {
                v |= val(*c)?;
            } else if *c != b'=' {
                return None;
            }
        }
        out.push((v >> 16) as u8);
        if pad < 2 {
            out.push((v >> 8) as u8);
        }
        if pad < 1 {
            out.push(v as u8);
        }
    }
    Some(out)
}

/// FNV-1a, 64 bit.
pub fn fnv1a64(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in data {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packbits_round_trips_runs_literals_and_the_edges() {
        let cases: Vec<Vec<u8>> = vec![
            vec![],
            vec![7],
            vec![1, 1],
            vec![1, 2, 3],
            vec![0; 1000],
            (0..=255u8).collect(),
            [vec![5; 129], vec![1, 2, 3, 3, 3], vec![9; 3]].concat(),
        ];
        for c in cases {
            assert_eq!(packbits_decode(&packbits_encode(&c)).unwrap(), c);
        }
        // Apple TN1023's own example: packed FE AA 02 80 00 2A FD AA 03 80 00 2A 22 F7 AA.
        let packed = [0xFE, 0xAA, 0x02, 0x80, 0x00, 0x2A, 0xFD, 0xAA, 0x03, 0x80, 0x00, 0x2A, 0x22, 0xF7, 0xAA];
        let raw = [
            0xAA, 0xAA, 0xAA, 0x80, 0x00, 0x2A, 0xAA, 0xAA, 0xAA, 0xAA, 0x80, 0x00, 0x2A, 0x22, 0xAA, 0xAA, 0xAA,
            0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA,
        ];
        assert_eq!(packbits_decode(&packed).unwrap(), raw);
        // A long run compresses.
        assert!(packbits_encode(&[0u8; 10_000]).len() < 200);
    }

    #[test]
    fn base64_matches_rfc4648_test_vectors() {
        // RFC 4648 §10.
        for (raw, enc) in [("", ""), ("f", "Zg=="), ("fo", "Zm8="), ("foo", "Zm9v"), ("foob", "Zm9vYg=="), ("fooba", "Zm9vYmE="), ("foobar", "Zm9vYmFy")] {
            assert_eq!(base64_encode(raw.as_bytes()), enc);
            assert_eq!(base64_decode(enc).unwrap(), raw.as_bytes());
        }
        assert!(base64_decode("Zm9").is_none());
        assert!(base64_decode("Zm9!").is_none());
    }

    #[test]
    fn fnv1a_matches_its_published_test_vectors() {
        // The FNV reference test vectors: "" and "a", "foobar".
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a64(b"foobar"), 0x8594_4171_f739_67e8);
    }
}
