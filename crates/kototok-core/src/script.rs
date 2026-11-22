//! Zero-regex UTF-8 script classifier used by the DFA pre-tokenizer.

/// Script category of a single Unicode scalar value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ScriptClass {
    /// `A-Z`, `a-z`, `_`.
    AsciiIdent,
    /// `0-9`.
    AsciiDigit,
    /// Hiragana (excluding the prolonged sound mark).
    Hiragana,
    /// Full-width and half-width Katakana.
    Katakana,
    /// `ー` (`U+30FC` prolonged sound mark).
    ProlongedMark,
    /// CJK ideographs including `々`, `〆`, `〇` and extensions.
    Kanji,
    /// Japanese punctuation / full-width punctuation, e.g. `。`, `、`, `・`, `「」`, `　`.
    JaPunct,
    /// Full-width ASCII alphanumerics (`０-９`, `Ａ-Ｚ`, `ａ-ｚ`).
    FullwidthAlnum,
    /// ASCII punctuation that always forms a hard boundary: `()[]{};,.:!?"'` and backtick.
    AsciiPunctHard,
    /// ASCII operator characters: `+-*/=<>!&|^~@#$%\` and similar.
    AsciiOperator,
    /// Space or tab.
    Space,
    /// `\n` or `\r`.
    Newline,
    /// Any other Unicode alphabetic letter (Greek, Hangul, accented Latin, ...).
    OtherLetter,
    /// Everything else: emoji, symbols, control characters.
    OtherSymbol,
}

const fn build_ascii_lut() -> [u8; 256] {
    let mut lut = [ScriptClass::OtherSymbol as u8; 256];
    let mut b = 0usize;
    while b < 256 {
        let c = b as u8;
        lut[b] = match c {
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => ScriptClass::AsciiIdent as u8,
            b'0'..=b'9' => ScriptClass::AsciiDigit as u8,
            b'(' | b')' | b'[' | b']' | b'{' | b'}' | b',' | b'.' | b';' | b':' | b'!' | b'?'
            | b'"' | b'\'' | b'`' => ScriptClass::AsciiPunctHard as u8,
            b'+' | b'-' | b'*' | b'/' | b'=' | b'<' | b'>' | b'&' | b'|' | b'^' | b'~' | b'@'
            | b'#' | b'$' | b'%' | b'\\' => ScriptClass::AsciiOperator as u8,
            b' ' | b'\t' => ScriptClass::Space as u8,
            b'\n' | b'\r' => ScriptClass::Newline as u8,
            _ => ScriptClass::OtherSymbol as u8,
        };
        b += 1;
    }
    lut
}

static ASCII_LUT: [u8; 256] = build_ascii_lut();

/// Classifies an ASCII byte directly through the lookup table.
///
/// The caller must guarantee `b < 128`; non-ASCII bytes produce `OtherSymbol`.
#[inline]
pub fn classify_ascii_byte(b: u8) -> ScriptClass {
    // SAFETY: every LUT entry was written as `ScriptClass as u8` where
    // `ScriptClass` is `#[repr(u8)]`, so all stored values are valid variants.
    unsafe { std::mem::transmute(ASCII_LUT[b as usize]) }
}

fn classify_non_ascii(c: u32) -> ScriptClass {
    match c {
        0x3041..=0x3096 | 0x309D..=0x309F => ScriptClass::Hiragana,
        0x30A1..=0x30FA | 0x30FD..=0x30FF | 0xFF66..=0xFF9F => ScriptClass::Katakana,
        0x30FC => ScriptClass::ProlongedMark,
        0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0xF900..=0xFAFF | 0x20000..=0x2FA1F | 0x3005..=0x3007 => {
            ScriptClass::Kanji
        }
        0x3000..=0x3004 | 0x3008..=0x301F | 0xFF01..=0xFF0F | 0xFF1A..=0xFF20 | 0xFF3B..=0xFF40
        | 0xFF5B..=0xFF65 => ScriptClass::JaPunct,
        0xFF10..=0xFF19 | 0xFF21..=0xFF3A | 0xFF41..=0xFF5A => ScriptClass::FullwidthAlnum,
        _ => match char::from_u32(c) {
            Some(ch) if ch.is_alphabetic() => ScriptClass::OtherLetter,
            _ => ScriptClass::OtherSymbol,
        },
    }
}

/// Classifies a Unicode scalar value into its script category.
#[inline]
pub fn classify_char(ch: char) -> ScriptClass {
    let c = ch as u32;
    if c < 128 {
        classify_ascii_byte(c as u8)
    } else {
        classify_non_ascii(c)
    }
}

fn last_char_of(bytes: &[u8]) -> Option<char> {
    std::str::from_utf8(bytes).ok()?.chars().next_back()
}

fn first_char_of(bytes: &[u8]) -> Option<char> {
    std::str::from_utf8(bytes).ok()?.chars().next()
}

/// Returns true iff `left` ends with a `Kanji` char and `right` starts with a
/// `Hiragana` char. Used by `kototok-train` for ACSG scoring.
pub fn is_kanji_to_hiragana_cross(left: &[u8], right: &[u8]) -> bool {
    match (last_char_of(left), first_char_of(right)) {
        (Some(l), Some(r)) => {
            classify_char(l) == ScriptClass::Kanji && classify_char(r) == ScriptClass::Hiragana
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_classes() {
        assert_eq!(classify_char('a'), ScriptClass::AsciiIdent);
        assert_eq!(classify_char('_'), ScriptClass::AsciiIdent);
        assert_eq!(classify_char('7'), ScriptClass::AsciiDigit);
        assert_eq!(classify_char('('), ScriptClass::AsciiPunctHard);
        assert_eq!(classify_char('"'), ScriptClass::AsciiPunctHard);
        assert_eq!(classify_char('`'), ScriptClass::AsciiPunctHard);
        assert_eq!(classify_char('+'), ScriptClass::AsciiOperator);
        assert_eq!(classify_char('\\'), ScriptClass::AsciiOperator);
        assert_eq!(classify_char(' '), ScriptClass::Space);
        assert_eq!(classify_char('\t'), ScriptClass::Space);
        assert_eq!(classify_char('\n'), ScriptClass::Newline);
        assert_eq!(classify_char('\r'), ScriptClass::Newline);
        assert_eq!(classify_char('\x00'), ScriptClass::OtherSymbol);
    }

    #[test]
    fn japanese_classes() {
        assert_eq!(classify_char('あ'), ScriptClass::Hiragana);
        assert_eq!(classify_char('ゝ'), ScriptClass::Hiragana);
        assert_eq!(classify_char('ア'), ScriptClass::Katakana);
        assert_eq!(classify_char('ｶ'), ScriptClass::Katakana);
        assert_eq!(classify_char('ー'), ScriptClass::ProlongedMark);
        assert_eq!(classify_char('日'), ScriptClass::Kanji);
        assert_eq!(classify_char('々'), ScriptClass::Kanji);
        assert_eq!(classify_char('〇'), ScriptClass::Kanji);
        assert_eq!(classify_char('𠮟'), ScriptClass::Kanji); // U+20B9F ext-B
        assert_eq!(classify_char('。'), ScriptClass::JaPunct);
        assert_eq!(classify_char('　'), ScriptClass::JaPunct);
        assert_eq!(classify_char('「'), ScriptClass::JaPunct);
        assert_eq!(classify_char('Ａ'), ScriptClass::FullwidthAlnum);
        assert_eq!(classify_char('５'), ScriptClass::FullwidthAlnum);
        assert_eq!(classify_char('ａ'), ScriptClass::FullwidthAlnum);
        assert_eq!(classify_char('！'), ScriptClass::JaPunct);
    }

    #[test]
    fn other_classes() {
        assert_eq!(classify_char('é'), ScriptClass::OtherLetter);
        assert_eq!(classify_char('한'), ScriptClass::OtherLetter);
        assert_eq!(classify_char('α'), ScriptClass::OtherLetter);
        assert_eq!(classify_char('🎉'), ScriptClass::OtherSymbol);
        assert_eq!(classify_char('・'), ScriptClass::OtherSymbol); // U+30FB
    }

    #[test]
    fn kanji_hiragana_cross() {
        assert!(is_kanji_to_hiragana_cross("日本".as_bytes(), "の".as_bytes()));
        assert!(is_kanji_to_hiragana_cross("書".as_bytes(), "く".as_bytes()));
        assert!(!is_kanji_to_hiragana_cross("あ".as_bytes(), "日".as_bytes()));
        assert!(!is_kanji_to_hiragana_cross("日".as_bytes(), "カ".as_bytes()));
        assert!(!is_kanji_to_hiragana_cross(&[0xFF], "あ".as_bytes()));
        assert!(!is_kanji_to_hiragana_cross("".as_bytes(), "あ".as_bytes()));
    }
}
