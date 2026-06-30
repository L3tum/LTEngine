/// Cleanup module for removing Unicode poisoning/fingerprinting from LLM output.
///
/// Strips zero-width characters, bidirectional marks, directional overrides,
/// C0 control characters (except newline/tab), and similar "invisible" Unicode.
/// Soft hyphens (U+00AD) are replaced with normal hyphens to preserve word boundaries.
///
/// Returns a `CleanupResult` with the cleaned string, the count of characters
/// that were removed (stripped), and the count of characters that were replaced
/// (e.g., soft hyphens replaced with normal hyphens).
///
/// ## Characters stripped (removed)
/// - Zero-width characters: U+200B, U+200C, U+200D, U+FEFF
/// - Bidirectional marks: U+200E, U+200F
/// - Directional overrides: U+202A–U+202E, U+2066–U+2069
/// - Other Unicode controls: U+206A–U+206F
/// - C0 controls (except \n and \t): U+0000–U+0008, U+000B–U+000C, U+000E–U+001F
///
/// ## Characters replaced
/// - Soft hyphen (U+00AD) → normal hyphen (U+002D)
pub struct CleanupResult {
    /// Cleaned output string
    pub cleaned: String,
    /// Number of characters stripped from the input
    pub removed: usize,
    /// Number of characters replaced in the input
    pub replaced: usize,
}

pub fn cleanup_output(input: &str) -> CleanupResult {
    let mut removed_count = 0;
    let mut replaced_count = 0;
    let mut output = String::with_capacity(input.len());

    for ch in input.chars() {
        match ch {
            // Zero-width characters (strip)
            '\u{200B}' // zero-width space
            | '\u{200C}' // zero-width non-joiner
            | '\u{200D}' // zero-width joiner
            | '\u{FEFF}' // BOM / zero-width no-break space
            => {
                removed_count += 1;
            }

            // Bidirectional marks (strip)
            '\u{200E}' // left-to-right mark
            | '\u{200F}' // right-to-left mark
            => {
                removed_count += 1;
            }

            // Directional overrides (strip)
            '\u{202A}'..='\u{202E}' // LRE, RLE, PDF, LRI, RLI, FSI, PDI
            | '\u{2066}'..='\u{2069}' // LRI, RLI, FSI, PDI (alternate block)
            => {
                removed_count += 1;
            }

            // Other Unicode formatting controls (strip)
            '\u{206A}'..='\u{206F}' // various control/formatting characters
            => {
                removed_count += 1;
            }

            // C0 control characters except newline and tab (strip)
            '\u{0000}'..='\u{0008}'
            | '\u{000B}'..='\u{000C}'
            | '\u{000E}'..='\u{001F}'
            => {
                removed_count += 1;
            }

            // Soft hyphen: replace with normal hyphen
            '\u{00AD}' => {
                replaced_count += 1;
                output.push('-');
            }

            // All other characters: keep
            _ => {
                output.push(ch);
            }
        }
    }

    CleanupResult {
        cleaned: output,
        removed: removed_count,
        replaced: replaced_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_zero_width_spaces() {
        let result = cleanup_output("ab\u{200B}c");
        assert_eq!(result.cleaned, "abc");
        assert_eq!(result.removed, 1);
        assert_eq!(result.replaced, 0);
    }

    #[test]
    fn test_clean_mixed_junk() {
        let result = cleanup_output("Hello\u{200B}\u{200C}\u{200D}\u{FEFF}\u{200E}\u{200F}World");
        assert_eq!(result.cleaned, "HelloWorld");
        assert_eq!(result.removed, 6);
        assert_eq!(result.replaced, 0);
    }

    #[test]
    fn test_soft_hyphen_replaced() {
        let result = cleanup_output("hel\u{00AD}lo");
        assert_eq!(result.cleaned, "hel-lo");
        assert_eq!(result.removed, 0);
        assert_eq!(result.replaced, 1);
    }

    #[test]
    fn test_soft_hyphen_and_removals() {
        // Input: "hel" + soft-hyphen + "lo" + ZWSP + soft-hyphen + "wor" + ZWNJ + "ld"
        // Output: soft-hyphens replaced with hyphens, ZWSP/ZWNJ removed -> "hel-lo-world"
        let result = cleanup_output("hel\u{00AD}lo\u{200B}\u{00AD}wor\u{200C}ld");
        assert_eq!(result.cleaned, "hel-lo-world");
        assert_eq!(result.removed, 2);
        assert_eq!(result.replaced, 2);
    }

    #[test]
    fn test_newline_tab_preserved() {
        let result = cleanup_output("a\nb\tc");
        assert_eq!(result.cleaned, "a\nb\tc");
        assert_eq!(result.removed, 0);
        assert_eq!(result.replaced, 0);
    }

    #[test]
    fn test_no_cleanup_needed() {
        let result = cleanup_output("Hello, world!");
        assert_eq!(result.cleaned, "Hello, world!");
        assert_eq!(result.removed, 0);
        assert_eq!(result.replaced, 0);
    }

    #[test]
    fn test_directional_overrides() {
        let result = cleanup_output("test\u{202A}override\u{202C}end");
        assert_eq!(result.cleaned, "testoverrideend");
        assert_eq!(result.removed, 2);
        assert_eq!(result.replaced, 0);
    }

    #[test]
    fn test_c0_control_chars() {
        let result = cleanup_output("a\u{0000}\u{0001}\u{0002}b");
        assert_eq!(result.cleaned, "ab");
        assert_eq!(result.removed, 3);
        assert_eq!(result.replaced, 0);
    }

    #[test]
    fn test_empty_input() {
        let result = cleanup_output("");
        assert_eq!(result.cleaned, "");
        assert_eq!(result.removed, 0);
        assert_eq!(result.replaced, 0);
    }
}
