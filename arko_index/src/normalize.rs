//! Byte-preserving recovery for C/C++ annotation macros. The final boundaries
//! always come from a tree-sitter tree and always slice the original source.
pub(crate) fn annotations(source: &str) -> String {
    let mut b = source.as_bytes().to_vec();
    let orig = source.as_bytes();
    let mut i = 0;
    while i < orig.len() {
        // Never normalize comments, strings, or preprocessing directives.
        if orig[i] == b'#' {
            while i < orig.len() {
                if orig[i] == b'\n' && (i == 0 || orig[i - 1] != b'\\') {
                    i += 1;
                    break;
                }
                i += 1;
            }
            continue;
        }
        if orig[i..].starts_with(b"//") {
            while i < orig.len() && orig[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if orig[i..].starts_with(b"/*") {
            i += 2;
            while i + 1 < orig.len() && !orig[i..].starts_with(b"*/") {
                i += 1;
            }
            i = (i + 2).min(orig.len());
            continue;
        }
        if orig[i] == b'"' || orig[i] == b'\'' {
            let q = orig[i];
            i += 1;
            while i < orig.len() {
                if orig[i] == b'\\' {
                    i = (i + 2).min(orig.len());
                } else if orig[i] == q {
                    i += 1;
                    break;
                } else {
                    i += 1;
                }
            }
            continue;
        }
        if orig[i].is_ascii_alphabetic() || orig[i] == b'_' {
            let start = i;
            i += 1;
            while i < orig.len() && (orig[i].is_ascii_alphanumeric() || orig[i] == b'_') {
                i += 1;
            }
            let token = &source[start..i];
            if token.len() < 3 || token.bytes().any(|c| c.is_ascii_lowercase()) {
                continue;
            }
            let ann = [
                "INLINE",
                "EXPORT",
                "API",
                "CONSTEXPR",
                "CONSTEXPR14",
                "CONSTEXPR17",
                "CONSTEXPR20",
                "CONSTEXPR23",
                "NOEXCEPT",
                "NODISCARD",
                "DEPRECATED",
                "FINAL",
                "OVERRIDE",
                "VISIBILITY",
            ]
            .iter()
            .any(|s| token.ends_with(s));
            let line_lo = source[..start].rfind('\n').map_or(0, |x| x + 1);
            let line_hi = source[i..].find('\n').map_or(source.len(), |x| i + x);
            let standalone =
                source[line_lo..start].trim().is_empty() && source[i..line_hi].trim().is_empty();
            let region = standalone && (token.contains("BEGIN_") || token.contains("END_"));
            let legacy = matches!(token, "FAR" | "NEAR" | "PASCAL" | "WINAPI");
            let exception = token.ends_with("_TRY") || (standalone && token.contains("_CATCH"));
            if ann || region || legacy || exception {
                // Function-like macros require argument-aware recovery; leave
                // those intact rather than changing source length or semantics.
                if source[i..].trim_start().starts_with('(') && !region {
                    continue;
                }
                b[start..i].fill(b' ');
            }
        } else {
            i += 1;
        }
    }
    String::from_utf8(b).expect("only ASCII annotation bytes replaced")
}
