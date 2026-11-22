//! DFA pre-tokenizer supporting all `PreTokenizerMode`s.
//!
//! Invariant: for every valid UTF-8 `text`, concatenating
//! `&text[pt.start..pt.end]` over `pretokenize(text, cfg)` yields `text`
//! byte-for-byte (contiguous `0..text.len()`, char boundaries only).

use crate::model::{PreTokenizerConfig, PreTokenizerMode};
use crate::script::{ScriptClass, classify_ascii_byte, classify_char};

/// A pre-tokenizer span over the input text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreToken {
    /// Byte offset (inclusive) of the span start.
    pub start: usize,
    /// Byte offset (exclusive) of the span end.
    pub end: usize,
    /// If true, Phase-2 merging may never cross this token's right edge.
    pub hard_break_after: bool,
}

const MAX_SPACE_RUN: usize = 8;
const MAX_OPERATOR_RUN: usize = 3;

/// Returns the script class and UTF-8 byte length of the char at `pos`.
#[inline]
fn class_at(text: &str, pos: usize) -> (ScriptClass, usize) {
    let b = text.as_bytes()[pos];
    if b < 128 {
        (classify_ascii_byte(b), 1)
    } else {
        let ch = text[pos..].chars().next().unwrap();
        (classify_char(ch), ch.len_utf8())
    }
}

/// Letter classes that `StandardGpt` groups into a single run.
#[inline]
fn is_gpt_letter(cls: ScriptClass) -> bool {
    matches!(
        cls,
        ScriptClass::AsciiIdent
            | ScriptClass::Hiragana
            | ScriptClass::Katakana
            | ScriptClass::Kanji
            | ScriptClass::FullwidthAlnum
            | ScriptClass::OtherLetter
            | ScriptClass::ProlongedMark
    )
}

/// Internal run-state for the AsymScript/StrictScript DFA.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RunState {
    Kanji,
    KanjiHira,
    Hiragana,
    Katakana,
    AsciiIdent,
    Fullwidth,
    OtherLetter,
    OtherSymbol,
}

#[inline]
fn run_state_of(cls: ScriptClass) -> RunState {
    match cls {
        ScriptClass::Kanji => RunState::Kanji,
        ScriptClass::Hiragana => RunState::Hiragana,
        ScriptClass::Katakana => RunState::Katakana,
        ScriptClass::AsciiIdent => RunState::AsciiIdent,
        ScriptClass::FullwidthAlnum => RunState::Fullwidth,
        ScriptClass::OtherLetter => RunState::OtherLetter,
        _ => RunState::OtherSymbol,
    }
}

/// One DFA step. Returns whether the run continues and the next state.
#[inline]
fn run_step(state: RunState, cls: ScriptClass, mode: PreTokenizerMode) -> (bool, RunState) {
    use RunState::*;
    match (state, cls) {
        (Kanji, ScriptClass::Kanji) => (true, Kanji),
        // Okurigana: Kanji -> Hiragana stays inside the chunk (AsymScript only).
        (Kanji, ScriptClass::Hiragana) if mode == PreTokenizerMode::AsymScript => (true, KanjiHira),
        // Hiragana -> Hiragana stays together both inside and outside a
        // Kanji-prefixed tail; ー attaches to a Hiragana/Katakana tail.
        (KanjiHira, ScriptClass::Hiragana) => (true, KanjiHira),
        (Hiragana, ScriptClass::Hiragana) => (true, Hiragana),
        (KanjiHira | Hiragana | Katakana, ScriptClass::ProlongedMark) => (true, state),
        (Katakana, ScriptClass::Katakana) => (true, Katakana),
        (AsciiIdent, ScriptClass::AsciiIdent) => (true, AsciiIdent),
        (Fullwidth, ScriptClass::FullwidthAlnum) => (true, Fullwidth),
        (OtherLetter, ScriptClass::OtherLetter) => (true, OtherLetter),
        (OtherSymbol, ScriptClass::OtherSymbol) => (true, OtherSymbol),
        _ => (false, state),
    }
}

/// Consumes a script-mode run starting at `pos` (first char already read).
/// Returns the byte end of the run. `max_chars` caps the run length.
fn consume_script_run(
    text: &str,
    pos: usize,
    cls0: ScriptClass,
    len0: usize,
    mode: PreTokenizerMode,
    max_chars: usize,
) -> usize {
    let n = text.len();
    let mut end = pos + len0;
    let mut chars_used = 1usize;
    let mut state = run_state_of(cls0);
    while end < n && chars_used < max_chars {
        let (cls, len) = class_at(text, end);
        let (cont, next) = run_step(state, cls, mode);
        if !cont {
            break;
        }
        state = next;
        end += len;
        chars_used += 1;
    }
    end
}

/// Consumes a StandardGpt letter run starting at `pos`.
fn consume_gpt_run(text: &str, pos: usize, len0: usize, max_chars: usize) -> usize {
    let n = text.len();
    let mut end = pos + len0;
    let mut chars_used = 1usize;
    while end < n && chars_used < max_chars {
        let (cls, len) = class_at(text, end);
        if !is_gpt_letter(cls) {
            break;
        }
        end += len;
        chars_used += 1;
    }
    end
}

/// Tokenizes `text` into `PreToken` spans according to `cfg`.
pub fn pretokenize(text: &str, cfg: &PreTokenizerConfig) -> Vec<PreToken> {
    let mut out = Vec::new();
    pretokenize_into(text, cfg, &mut out);
    out
}

/// Tokenizes `text` into `out` (cleared first), reusing its allocation.
pub fn pretokenize_into(text: &str, cfg: &PreTokenizerConfig, out: &mut Vec<PreToken>) {
    out.clear();
    let bytes = text.as_bytes();
    let n = bytes.len();
    let max_chars = cfg.max_chunk_chars.max(1);
    let max_digits = cfg.max_digit_run.max(1);
    let mode = cfg.mode;
    let mut i = 0usize;
    // Set when a lone ' ' should be glued to the front of the next ASCII chunk.
    let mut attach_from: Option<usize> = None;

    while i < n {
        let (cls, len) = class_at(text, i);
        let start = attach_from.take().unwrap_or(i);
        let head_extra = usize::from(start < i);
        match cls {
            ScriptClass::Newline => {
                let end = if bytes[i] == b'\r' && i + 1 < n && bytes[i + 1] == b'\n' {
                    i + 2
                } else {
                    i + 1
                };
                // `\n` followed by indentation or another newline may merge
                // rightward so that code's frequent `newline+indent` and
                // blank-line spans can form single tokens; any other
                // following class keeps the seal.
                let indent_next = bytes
                    .get(end)
                    .is_some_and(|&b| b == b' ' || b == b'\t' || b == b'\n');
                out.push(PreToken {
                    start,
                    end,
                    hard_break_after: !indent_next,
                });
                i = end;
            }
            ScriptClass::Space => {
                let mut j = i;
                while j < n && (bytes[j] == b' ' || bytes[j] == b'\t') {
                    j += 1;
                }
                let run_is_single_space = j - i == 1 && bytes[i] == b' ';
                let attaches = run_is_single_space
                    && cfg.attach_ascii_leading_space
                    && j < n
                    && matches!(
                        class_at(text, j).0,
                        ScriptClass::AsciiIdent
                            | ScriptClass::AsciiDigit
                            | ScriptClass::AsciiOperator
                    );
                if attaches {
                    attach_from = Some(i);
                    i = j;
                } else {
                    let mut s = i;
                    while s < j {
                        let e = (s + MAX_SPACE_RUN).min(j);
                        out.push(PreToken {
                            start: s,
                            end: e,
                            hard_break_after: true,
                        });
                        s = e;
                    }
                    i = j;
                }
            }
            ScriptClass::JaPunct | ScriptClass::ProlongedMark => {
                // A ー at run-start (or after a non-kana class) behaves like
                // Japanese punctuation. Punctuation tokens are hard on both
                // sides: the previous token is sealed too.
                if let Some(prev) = out.last_mut() {
                    prev.hard_break_after = true;
                }
                out.push(PreToken {
                    start,
                    end: i + len,
                    hard_break_after: true,
                });
                i += len;
            }
            ScriptClass::AsciiPunctHard => {
                // ASCII punctuation seals a word-like predecessor
                // (`word,` stays split) but merges rightward with punct,
                // spaces, or words: `():`, `, the`, `(self`, `'t` can form.
                if let Some(prev) = out.last_mut() {
                    let pb = bytes[prev.end - 1];
                    if pb.is_ascii_alphanumeric() || pb == b'_' || pb >= 0x80 {
                        prev.hard_break_after = true;
                    }
                }
                out.push(PreToken {
                    start,
                    end: i + len,
                    hard_break_after: false,
                });
                i += len;
            }
            ScriptClass::AsciiDigit => {
                // Chunks of at most `max_digit_run` digits; a glued leading
                // space does not count against the digit cap.
                let mut e = i;
                let mut dcount = 0usize;
                while e < n && bytes[e].is_ascii_digit() && dcount < max_digits {
                    e += 1;
                    dcount += 1;
                }
                out.push(PreToken {
                    start,
                    end: e,
                    hard_break_after: false,
                });
                i = e;
            }
            ScriptClass::AsciiOperator => {
                let mut e = i;
                let mut count = 0usize;
                let cap = MAX_OPERATOR_RUN.min(max_chars - head_extra.min(max_chars - 1));
                while e < n && count < cap && class_at(text, e).0 == ScriptClass::AsciiOperator {
                    e += 1;
                    count += 1;
                }
                out.push(PreToken {
                    start,
                    end: e,
                    hard_break_after: false,
                });
                i = e;
            }
            _ => {
                // Run-based classes: script-grouped runs or GPT letter runs.
                let cap = max_chars - head_extra.min(max_chars - 1);
                let end = if mode == PreTokenizerMode::StandardGpt && is_gpt_letter(cls) {
                    consume_gpt_run(text, i, len, cap)
                } else {
                    // Non-letter classes that still group (OtherSymbol) and all
                    // script-grouped runs.
                    consume_script_run(text, i, cls, len, mode, cap)
                };
                out.push(PreToken {
                    start,
                    end,
                    hard_break_after: false,
                });
                i = end;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ByteMode, PreTokenizerMode};

    fn cfg(mode: PreTokenizerMode) -> PreTokenizerConfig {
        PreTokenizerConfig {
            mode,
            byte_mode: ByteMode::Utf8Atomic,
            attach_ascii_leading_space: true,
            max_digit_run: 3,
            max_chunk_chars: 128,
            max_segment_chunks: 64,
        }
    }

    fn spans<'a>(text: &'a str, cfg: &PreTokenizerConfig) -> Vec<(&'a str, bool)> {
        pretokenize(text, cfg)
            .iter()
            .map(|pt| (&text[pt.start..pt.end], pt.hard_break_after))
            .collect()
    }

    fn assert_coverage(text: &str, cfg: &PreTokenizerConfig) {
        let pts = pretokenize(text, cfg);
        let mut expect = 0usize;
        for pt in &pts {
            assert_eq!(pt.start, expect, "gap/overlap at {pt:?} in {text:?}");
            expect = pt.end;
        }
        assert_eq!(expect, text.len(), "coverage mismatch in {text:?}");
        let joined: String = pts.iter().map(|pt| &text[pt.start..pt.end]).collect();
        assert_eq!(joined, text);
    }

    #[test]
    fn asym_okurigana_and_scripts() {
        let c = cfg(PreTokenizerMode::AsymScript);
        let s = "新しいRustのトークナイザーを作る。";
        let v = spans(s, &c);
        assert_eq!(
            v,
            vec![
                ("新しい", false),
                ("Rust", false),
                ("の", false),
                ("トークナイザー", false),
                ("を", false),
                ("作る", true), // sealed by 。
                ("。", true),
            ]
        );
        assert_coverage(s, &c);
    }

    #[test]
    fn asym_hiragana_kanji_splits() {
        let c = cfg(PreTokenizerMode::AsymScript);
        let s = "日本の開発者が高速な処理を書く。";
        let v = spans(s, &c);
        assert_eq!(
            v.iter().map(|(t, _)| *t).collect::<Vec<_>>(),
            vec!["日本の", "開発者が", "高速な", "処理を", "書く", "。"]
        );
        assert_coverage(s, &c);
    }

    #[test]
    fn asym_iteration_marks_and_katakana() {
        let c = cfg(PreTokenizerMode::AsymScript);
        let s = "人々・様々・カタカナ語ー";
        let v = spans(s, &c);
        assert_eq!(
            v.iter().map(|(t, _)| *t).collect::<Vec<_>>(),
            vec!["人々", "・", "様々", "・", "カタカナ", "語", "ー"]
        );
        // trailing ー treated as JaPunct: hard after itself and after 語
        assert!(v[6].1);
        assert!(v[5].1);
        assert_coverage(s, &c);
    }

    #[test]
    fn asym_code_pretokens() {
        let c = cfg(PreTokenizerMode::AsymScript);
        let s = "fn main() { let x = 12345; }";
        let v = spans(s, &c);
        assert_eq!(
            v,
            vec![
                ("fn", false),
                (" main", true), // sealed by (
                ("(", false),
                (")", false),
                (" ", true),
                ("{", false),
                (" let", false),
                (" x", false),
                (" =", false),
                (" 123", false),
                ("45", true), // sealed by ;
                (";", false),
                (" ", true),
                ("}", false),
            ]
        );
        assert_coverage(s, &c);
    }

    #[test]
    fn strict_script_splits_okurigana() {
        let c = cfg(PreTokenizerMode::StrictScript);
        let s = "新しいRustのトークナイザーを作る。";
        let v = spans(s, &c);
        assert_eq!(
            v.iter().map(|(t, _)| *t).collect::<Vec<_>>(),
            vec![
                "新",
                "しい",
                "Rust",
                "の",
                "トークナイザー",
                "を",
                "作",
                "る",
                "。"
            ]
        );
        assert_coverage(s, &c);
    }

    #[test]
    fn standard_gpt_groups_letters() {
        let c = cfg(PreTokenizerMode::StandardGpt);
        let s = "新しいRustのトークナイザーを作る。";
        let v = spans(s, &c);
        assert_eq!(
            v.iter().map(|(t, _)| *t).collect::<Vec<_>>(),
            vec!["新しいRustのトークナイザーを作る", "。"]
        );
        assert_coverage(s, &c);
    }

    #[test]
    fn newlines_and_indentation() {
        let c = cfg(PreTokenizerMode::AsymScript);
        let s = "a\r\nb\rc\n  x";
        let v = spans(s, &c);
        assert_eq!(
            v,
            vec![
                ("a", false),
                ("\r\n", true),
                ("b", false),
                ("\r", true),
                ("c", false),
                ("\n", false),
                ("  ", true),
                ("x", false),
            ]
        );
        assert_coverage(s, &c);
    }

    #[test]
    fn newline_blank_line_is_soft() {
        let c = cfg(PreTokenizerMode::AsymScript);
        let v = spans("a\n\n  b\nx", &c);
        assert_eq!(
            v,
            vec![
                ("a", false),
                ("\n", false),
                ("\n", false),
                ("  ", true),
                ("b", false),
                ("\n", true),
                ("x", false),
            ]
        );
    }

    #[test]
    fn emoji_and_other_letters() {
        let c = cfg(PreTokenizerMode::AsymScript);
        let s = "🎉🎊x한글";
        let v = spans(s, &c);
        assert_eq!(
            v.iter().map(|(t, _)| *t).collect::<Vec<_>>(),
            vec!["🎉🎊", "x", "한글"]
        );
        assert_coverage(s, &c);
    }

    #[test]
    fn no_space_attach_when_disabled_or_multi() {
        let mut c = cfg(PreTokenizerMode::AsymScript);
        c.attach_ascii_leading_space = false;
        let s = "a b";
        let v = spans(s, &c);
        assert_eq!(
            v.iter().map(|(t, _)| *t).collect::<Vec<_>>(),
            vec!["a", " ", "b"]
        );
        let c = cfg(PreTokenizerMode::AsymScript);
        let s = "a  b";
        let v = spans(s, &c);
        assert_eq!(
            v.iter().map(|(t, _)| *t).collect::<Vec<_>>(),
            vec!["a", "  ", "b"]
        );
        assert_coverage("a  b", &c);
    }

    #[test]
    fn max_chunk_chars_splits() {
        let mut c = cfg(PreTokenizerMode::AsymScript);
        c.max_chunk_chars = 4;
        let s = "あいうえおかきく";
        let v = spans(s, &c);
        assert_eq!(
            v.iter().map(|(t, _)| *t).collect::<Vec<_>>(),
            vec!["あいうえ", "おかきく"]
        );
        assert_coverage(s, &c);
    }

    #[test]
    fn digit_runs_split_by_three() {
        let c = cfg(PreTokenizerMode::AsymScript);
        let v = spans("1234567890", &c);
        assert_eq!(
            v.iter().map(|(t, _)| *t).collect::<Vec<_>>(),
            vec!["123", "456", "789", "0"]
        );
    }

    #[test]
    fn operators_group_up_to_three() {
        let c = cfg(PreTokenizerMode::AsymScript);
        let v = spans("a->b====c", &c);
        assert_eq!(
            v.iter().map(|(t, _)| *t).collect::<Vec<_>>(),
            vec!["a", "->", "b", "===", "=", "c"]
        );
        assert_coverage("a->b====c", &c);
    }

    #[test]
    fn gpt_mode_digits_and_space() {
        let c = cfg(PreTokenizerMode::StandardGpt);
        let v = spans("abc 12345", &c);
        assert_eq!(
            v.iter().map(|(t, _)| *t).collect::<Vec<_>>(),
            vec!["abc", " 123", "45"]
        );
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        fn any_mode() -> impl Strategy<Value = PreTokenizerMode> {
            prop_oneof![
                Just(PreTokenizerMode::AsymScript),
                Just(PreTokenizerMode::StrictScript),
                Just(PreTokenizerMode::StandardGpt),
            ]
        }

        fn any_string() -> impl Strategy<Value = String> {
            // Mix of CJK, kana, ext-B, emoji, ASCII, whitespace.
            prop_oneof![
                any::<String>(),
                "[あ-んア-ンー一-龯々〆〇a-zA-Z0-9 .,;:!?()\\[\\]{}+\\-*/=<>\\\\\n\r\t　。、・「」]*",
                "[\\u{20000}-\\u{2FA1F}あア🎉🎊x1 ]*",
                proptest::collection::vec(any::<char>(), 0..64)
                    .prop_map(|v| v.into_iter().collect::<String>()),
            ]
        }

        proptest! {
            #[test]
            fn pretokenize_covers_all_bytes(s in any_string(), mode in any_mode()) {
                let c = cfg(mode);
                let pts = pretokenize(&s, &c);
                let mut expect = 0usize;
                for pt in &pts {
                    prop_assert_eq!(pt.start, expect);
                    prop_assert!(pt.end > pt.start);
                    expect = pt.end;
                }
                prop_assert_eq!(expect, s.len());
                let joined: String = pts.iter().map(|pt| &s[pt.start..pt.end]).collect();
                prop_assert_eq!(joined, s);
            }
        }
    }
}
