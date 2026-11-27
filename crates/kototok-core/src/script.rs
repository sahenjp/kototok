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
        0x4E00..=0x9FFF
        | 0x3400..=0x4DBF
        | 0xF900..=0xFAFF
        | 0x20000..=0x2FA1F
        | 0x3005..=0x3007 => ScriptClass::Kanji,
        0x3000..=0x3004
        | 0x3008..=0x301F
        | 0xFF01..=0xFF0F
        | 0xFF1A..=0xFF20
        | 0xFF3B..=0xFF40
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

/// Returns true iff `bytes` is valid UTF-8 containing a Japanese kana or kanji char.
pub fn is_japanese_bytes(bytes: &[u8]) -> bool {
    let Ok(text) = std::str::from_utf8(bytes) else {
        return false;
    };
    text.chars().any(|ch| {
        matches!(
            classify_char(ch),
            ScriptClass::Kanji
                | ScriptClass::Hiragana
                | ScriptClass::Katakana
                | ScriptClass::ProlongedMark
        )
    })
}

/// ASCII identifier letters plus the Latin ranges used by European and Vietnamese text.
pub fn is_latin_letter(ch: char) -> bool {
    matches!(
        ch as u32,
        0x0041..=0x005A
            | 0x0061..=0x007A
            | 0x005F
            | 0x00C0..=0x00D6
            | 0x00D8..=0x00F6
            | 0x00F8..=0x00FF
            | 0x0100..=0x024F
            | 0x0250..=0x02AF
            | 0x1E00..=0x1EFF
            | 0x2C60..=0x2C7F
            | 0xA720..=0xA7FF
            | 0xAB30..=0xAB6F
    )
}

/// Unicode combining marks and zero-width joiners that must not split a script run.
pub fn is_combining_or_joiner(ch: char) -> bool {
    matches!(
        ch as u32,
        0x0300..=0x036F
            | 0x1AB0..=0x1AFF
            | 0x1DC0..=0x1DFF
            | 0x20D0..=0x20FF
            | 0xFE20..=0xFE2F
            | 0x0591..=0x05BD
            | 0x05BF
            | 0x05C1..=0x05C2
            | 0x05C4..=0x05C5
            | 0x05C7
            | 0x0610..=0x061A
            | 0x064B..=0x065F
            | 0x0670
            | 0x06D6..=0x06DC
            | 0x06DF..=0x06E4
            | 0x06E7..=0x06E8
            | 0x06EA..=0x06ED
            | 0x0900..=0x0903
            | 0x093A..=0x094F
            | 0x0951..=0x0957
            | 0x0962..=0x0963
            | 0x0981..=0x0983
            | 0x09BC
            | 0x09BE..=0x09CD
            | 0x09D7
            | 0x09E2..=0x09E3
            | 0x0A01..=0x0A03
            | 0x0A3C
            | 0x0A3E..=0x0A51
            | 0x0A70..=0x0A71
            | 0x0A75
            | 0x0A81..=0x0A83
            | 0x0ABC
            | 0x0ABE..=0x0ACD
            | 0x0AE2..=0x0AE3
            | 0x0AFA..=0x0AFF
            | 0x0B01..=0x0B03
            | 0x0B3C
            | 0x0B3E..=0x0B57
            | 0x0B62..=0x0B63
            | 0x0B82
            | 0x0BBE..=0x0BCD
            | 0x0BD7
            | 0x0C00..=0x0C04
            | 0x0C3C
            | 0x0C3E..=0x0C56
            | 0x0C62..=0x0C63
            | 0x0C81..=0x0C83
            | 0x0CBC
            | 0x0CBE..=0x0CD6
            | 0x0CE2..=0x0CE3
            | 0x0D00..=0x0D03
            | 0x0D3B..=0x0D4E
            | 0x0D57
            | 0x0D62..=0x0D63
            | 0x0D81..=0x0D83
            | 0x0DCA
            | 0x0DCF..=0x0DDF
            | 0x0DF2..=0x0DF3
            | 0x0E31
            | 0x0E34..=0x0E3A
            | 0x0E47..=0x0E4E
            | 0x0EB1
            | 0x0EB4..=0x0EBC
            | 0x0EC8..=0x0ECE
            | 0x0F18..=0x0F19
            | 0x0F35
            | 0x0F37
            | 0x0F39
            | 0x0F3E..=0x0F3F
            | 0x0F71..=0x0F84
            | 0x0F86..=0x0F87
            | 0x0F8D..=0x0FBC
            | 0x0FC6
            | 0x102B..=0x103E
            | 0x1056..=0x1059
            | 0x105E..=0x1060
            | 0x1062..=0x1064
            | 0x1067..=0x106D
            | 0x1071..=0x1074
            | 0x1082..=0x108D
            | 0x108F
            | 0x109A..=0x109D
            | 0x17B4..=0x17D3
            | 0x17DD
            | 0x200C..=0x200D
    )
}

const GROUP_OTHER: u8 = 0;
const GROUP_LATIN: u8 = 1;
const GROUP_THAI: u8 = 17;
const GROUP_LAO: u8 = 18;
const GROUP_TIBETAN: u8 = 19;
const GROUP_MYANMAR: u8 = 20;
const GROUP_KHMER: u8 = 22;

/// Returns a stable script-group discriminator for multilingual pre-tokenization.
pub fn multilingual_script_group(ch: char) -> u8 {
    if is_latin_letter(ch) {
        return GROUP_LATIN;
    }
    match ch as u32 {
        0x00B5 | 0x0370..=0x03FF | 0x1F00..=0x1FFF => 2, // Greek (incl. U+00B5 micro sign)
        0x0400..=0x052F | 0x1C80..=0x1C8F | 0x2DE0..=0x2DFF | 0xA640..=0xA69F => 3, // Cyrillic
        0x0530..=0x058F => 4,                            // Armenian
        0x0590..=0x05FF => 5,                            // Hebrew
        0x0600..=0x06FF | 0x0750..=0x077F | 0x08A0..=0x08FF | 0xFB50..=0xFDFF | 0xFE70..=0xFEFF => {
            6
        } // Arabic
        0x0900..=0x097F | 0xA8E0..=0xA8FF => 7,          // Devanagari
        0x0980..=0x09FF => 8,                            // Bengali
        0x0A00..=0x0A7F => 9,                            // Gurmukhi
        0x0A80..=0x0AFF => 10,                           // Gujarati
        0x0B00..=0x0B7F => 11,                           // Oriya
        0x0B80..=0x0BFF => 12,                           // Tamil
        0x0C00..=0x0C7F => 13,                           // Telugu
        0x0C80..=0x0CFF => 14,                           // Kannada
        0x0D00..=0x0D7F => 15,                           // Malayalam
        0x0D80..=0x0DFF => 16,                           // Sinhala
        0x0E00..=0x0E7F => GROUP_THAI,
        0x0E80..=0x0EFF => GROUP_LAO,
        0x0F00..=0x0FFF => GROUP_TIBETAN,
        0x1000..=0x109F | 0xA9E0..=0xA9FF | 0xAA60..=0xAA7F => GROUP_MYANMAR,
        0x10A0..=0x10FF | 0x2D00..=0x2D2F => 21, // Georgian
        0x1780..=0x17FF | 0x19E0..=0x19FF => GROUP_KHMER,
        0x1100..=0x11FF | 0x3130..=0x318F | 0xA960..=0xA97F | 0xAC00..=0xD7AF | 0xD7B0..=0xD7FF => {
            23
        } // Hangul
        _ => GROUP_OTHER,
    }
}

/// Whether a non-ASCII letter attaches a single leading ASCII space in multilingual mode.
/// All non-CJK scripts (including Thai/Lao/Myanmar/Khmer, which use spaces at
/// clause boundaries in web text) benefit from single-space attachment.
pub fn is_spaced_multilingual_letter(ch: char) -> bool {
    if ch.is_ascii() || !ch.is_alphabetic() {
        return false;
    }
    if is_latin_letter(ch) {
        return true;
    }
    !matches!(
        classify_char(ch),
        ScriptClass::Kanji
            | ScriptClass::Hiragana
            | ScriptClass::Katakana
            | ScriptClass::ProlongedMark
            | ScriptClass::FullwidthAlnum
    )
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
        assert!(is_kanji_to_hiragana_cross(
            "日本".as_bytes(),
            "の".as_bytes()
        ));
        assert!(is_kanji_to_hiragana_cross("書".as_bytes(), "く".as_bytes()));
        assert!(!is_kanji_to_hiragana_cross(
            "あ".as_bytes(),
            "日".as_bytes()
        ));
        assert!(!is_kanji_to_hiragana_cross(
            "日".as_bytes(),
            "カ".as_bytes()
        ));
        assert!(!is_kanji_to_hiragana_cross(&[0xFF], "あ".as_bytes()));
        assert!(!is_kanji_to_hiragana_cross("".as_bytes(), "あ".as_bytes()));
    }

    #[test]
    fn multilingual_script_helpers() {
        assert!(is_japanese_bytes("日本語".as_bytes()));
        assert!(is_japanese_bytes("abcア".as_bytes()));
        assert!(!is_japanese_bytes("abc".as_bytes()));
        assert!(!is_japanese_bytes(&[0xFF, 0xFE]));

        assert!(is_latin_letter('a'));
        assert!(is_latin_letter('_'));
        assert!(is_latin_letter('é'));
        assert!(is_latin_letter('ế'));
        assert!(!is_latin_letter('α'));

        assert!(is_combining_or_joiner('\u{0301}'));
        assert!(is_combining_or_joiner('\u{094D}'));
        assert!(is_combining_or_joiner('\u{0E31}'));
        assert!(is_combining_or_joiner('\u{200C}'));
        assert!(is_combining_or_joiner('\u{200D}'));
        assert!(!is_combining_or_joiner('a'));

        assert_eq!(
            multilingual_script_group('é'),
            multilingual_script_group('ü')
        );
        assert_eq!(
            multilingual_script_group('د'),
            multilingual_script_group('و')
        );
        assert_ne!(
            multilingual_script_group('д'),
            multilingual_script_group('α')
        );
        assert!(is_spaced_multilingual_letter('é'));
        assert!(is_spaced_multilingual_letter('д'));
        assert!(is_spaced_multilingual_letter('ا'));
        assert!(is_spaced_multilingual_letter('ह'));
        assert!(is_spaced_multilingual_letter('한'));
        assert!(!is_spaced_multilingual_letter('日'));
        assert!(!is_spaced_multilingual_letter('あ'));
        assert!(is_spaced_multilingual_letter('ก'));
        assert!(is_spaced_multilingual_letter('က'));
        assert!(is_spaced_multilingual_letter('ក'));
    }
}
