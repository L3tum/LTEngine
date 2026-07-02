//! Text chunking for long translations.
//!
//! Chunks a long text into pieces not exceeding a **character** limit (not byte limit),
//! preferring paragraph boundaries, then sentence boundaries, and finally word-boundary-aware
//! fixed-size fallback.

/// Split text on major sentence-ending punctuation marks followed by whitespace.
/// Returns an iterator over the sentence fragments (punctuation is kept with the fragment).
fn split_sentences(text: &str) -> Vec<String> {
    // We look for ". ", "? ", "! ", "; ", ": " (or their line-break variants) as separators.
    let sentence_punct: &[char] = &['.', '?', '!', ';', ':'];
    let whitespace: &[char] = &[' ', '\n', '\r'];
    let mut sentences = Vec::new();
    let mut current = String::new();

    let chars: Vec<_> = text.char_indices().collect();
    let mut i = 0;

    while i < chars.len() {
        let (_idx, ch) = chars[i];
        current.push(ch);

        // Check for end of sentence: punctuation followed by space or newline
        if let Some(&(_, next_ch)) = chars.get(i + 1) {
            if sentence_punct.contains(&ch) && whitespace.contains(&next_ch) {
                // Consume the separator (punctuation + space)
                let mut sentence = current.trim().to_string();
                // Reattach the punctuation if we trimmed it off
                if !sentence.ends_with(ch) {
                    sentence.push(ch);
                }
                sentences.push(sentence);
                current.clear();
                // Advance past the separator
                i += 1; // skip the space/newline after punctuation
                if i < chars.len() && chars[i].1 == '\r' {
                    i += 1; // skip \r in \r\n
                }
            } else {
                i += 1;
            }
        } else {
            i += 1;
        }
    }

    if !current.trim().is_empty() {
        sentences.push(current.trim().to_string());
    }

    sentences
}

/// Character count (grapheme-aware enough for our use: .chars().count()).
fn char_len(s: &str) -> usize {
    s.chars().count()
}

/// Chunk a long text into smaller pieces, preferring paragraph boundaries,
/// then sentence boundaries, and finally fixed-size fallback.
/// Each chunk will not exceed `max_chars` **characters** (not bytes).
/// Empty/whitespace-only input returns an empty vec (skipped in translate pipeline).
pub fn chunk_text(text: &str, max_chars: usize) -> Vec<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    if char_len(trimmed) <= max_chars {
        return vec![trimmed.to_string()];
    }

    // --- Level 1: Paragraphs ---
    let paragraphs: Vec<&str> = trimmed
        .split("\n\n")
        .filter(|p| !p.trim().is_empty())
        .collect();

    if paragraphs.len() > 1 {
        return chunk_with_fallback(paragraphs, "\n\n", max_chars);
    }

    // --- Level 2: Sentences ---
    let sentences = split_sentences(trimmed);
    if sentences.len() > 1 {
        return chunk_with_fallback(
            sentences.iter().map(String::as_str).collect(),
            " ",
            max_chars,
        );
    }

    // --- Level 3: Fixed-size fallback ---
    fixed_size_chunk(trimmed, max_chars)
}

/// Iterative grouping of segments (paragraphs or sentences), with a separator and
/// fallback to recursive chunking. Avoids deep recursion.
fn chunk_with_fallback(segments: Vec<&str>, separator: &str, max_chars: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();

    for seg in &segments {
        let seg_trimmed = seg.trim();
        if current.is_empty() {
            current = seg_trimmed.to_string();
        } else {
            let combined = format!("{}{}{}", current, separator, seg_trimmed);
            if char_len(&combined) <= max_chars {
                current = combined;
            } else {
                chunks.push(current);
                current = seg_trimmed.to_string();
            }
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }

    // Iterative deepening: any chunk still too large gets re-chunked.
    let mut final_chunks = Vec::new();
    for chunk in chunks {
        if char_len(&chunk) > max_chars {
            // Re-chunk this segment. Since we already know there are no paragraph or sentence
            // boundaries (otherwise they'd have been split earlier), we fall back to
            // fixed-size chunking.
            let sub_chunks = fixed_size_chunk(&chunk, max_chars);
            final_chunks.extend(sub_chunks);
        } else {
            final_chunks.push(chunk);
        }
    }
    final_chunks
}

/// Fixed-size character chunks with word-boundary awareness.
/// Tries to break at sentence punctuation if a `. `, `? `, etc. exists within the window.
fn fixed_size_chunk(text: &str, max_chars: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut remaining = text;

    loop {
        let char_count = char_len(remaining);
        if char_count <= max_chars {
            chunks.push(remaining.trim().to_string());
            break;
        }

        // Take a window of max_chars and look for sentence boundaries inside
        let window: String = remaining.chars().take(max_chars).collect();

        // Find the LAST sentence punctuation + whitespace within the window
        let sentence_punct: &[char] = &['.', '?', '!', ';', ':'];
        let sentence_boundary = window.rfind(char::is_whitespace).and_then(|pos| {
            // look backwards from pos to find a sentence punctuation mark
            window[..pos]
                .rfind(|c: char| sentence_punct.contains(&c))
                .map(|punct_pos| punct_pos + 2) // include punctuation + whitespace
        });

        if let Some(cut_point) = sentence_boundary {
            chunks.push(window[..cut_point].trim().to_string());
            remaining = &remaining[cut_point..];
        } else {
            // No sentence boundary. Cut at the last space to avoid mid-word.
            let cut_window: String = remaining.chars().take(max_chars).collect();
            let last_space = cut_window.rfind(' ').unwrap_or(cut_window.chars().count());

            let trimmed = if last_space > 0 {
                cut_window[..last_space].trim().to_string()
            } else {
                // No space found at all — take the first word or the whole window
                cut_window
                    .split_whitespace()
                    .next()
                    .unwrap_or(&cut_window)
                    .to_string()
            };

            chunks.push(trimmed);
            let consumed = last_space;
            remaining = &remaining[consumed..];
        }
    }

    chunks.retain(|c| !c.is_empty());
    chunks
}

#[cfg(test)]
mod chunk_tests {
    use super::{chunk_text, split_sentences};

    #[test]
    fn test_chunk_text_empty() {
        let chunks = chunk_text("", 100);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_chunk_text_whitespace() {
        let chunks = chunk_text("   \n\n  ", 100);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_chunk_text_small() {
        let chunks = chunk_text("Hello, world!", 100);
        assert_eq!(chunks, vec!["Hello, world!".to_string()]);
    }

    #[test]
    fn test_chunk_text_exact_size_chars() {
        let text = "ä".repeat(21); // each is 2 bytes, so 42 bytes but 21 chars
        assert_eq!(text.chars().count(), 21);
        let chunks = chunk_text(&text, 21);
        assert_eq!(chunks, vec![text]);
    }

    #[test]
    fn test_chunk_text_over_limit_fixed_fallback() {
        let text = "a".repeat(120);
        let chunks = chunk_text(&text, 50);
        assert_eq!(chunks.len(), 3); // 50 + 50 + 20
        assert_eq!(chunks[0].chars().count(), 50);
        assert_eq!(chunks[1].chars().count(), 50);
        assert_eq!(chunks[2].chars().count(), 20);
        assert_eq!(chunks.join(""), text);
    }

    #[test]
    fn test_chunk_text_by_paragraphs() {
        let text = "First paragraph line 1. Line 2.\n\nSecond paragraph. More text.";
        let chunks = chunk_text(text, 32); // first para is 30 chars, so 32 fits
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0], "First paragraph line 1. Line 2.");
        assert_eq!(chunks[1], "Second paragraph. More text.");
    }

    #[test]
    fn test_chunk_text_by_paragraphs_merging() {
        let text = "Short para.\n\nAnother short.";
        let chunks = chunk_text(text, 100);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Short para.\n\nAnother short.");
    }

    #[test]
    fn test_chunk_text_by_paragraphs_over_limit_fallback() {
        let text = "Very long paragraph with lots of content that exceeds limit.\n\nAnother short.";
        let chunks = chunk_text(text, 60);
        // First para (55 chars) + separator + second (13 chars) = 69 > 60, so separate chunks
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].chars().count() <= 60);
        assert!(chunks[1].chars().count() <= 60);
    }

    #[test]
    fn test_chunk_text_by_sentences_period() {
        let text = "First sentence. Second sentence. Third sentence.";
        let chunks = chunk_text(text, 20);
        // Each sentence ~16 chars, so they fit individually
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], "First sentence.");
        assert_eq!(chunks[1], "Second sentence.");
        assert_eq!(chunks[2], "Third sentence.");
    }

    #[test]
    fn test_chunk_text_by_sentences_exclamation() {
        // With limit 10, "Wow!" and "Amazing!" stay separate (combined 13 chars > 10),
        // and "Incredible!" (11 chars) gets split further by fixed-size fallback.
        let text = "Wow! Amazing! Incredible!";
        let chunks = chunk_text(text, 10);
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0], "Wow!");
        assert_eq!(chunks[1], "Amazing!");
        // "Incredible!" has no space, so fixed-size chunk takes "Incredible" then "!" separately
        assert_eq!(chunks[2], "Incredible");
        assert_eq!(chunks[3], "!");
    }

    #[test]
    fn test_chunk_text_by_sentences_no_merge() {
        // Use very small sentences with a limit that prevents merging
        let text = "Hi! Bye! Later!";
        let chunks = chunk_text(text, 6);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], "Hi!");
        assert_eq!(chunks[1], "Bye!");
        assert_eq!(chunks[2], "Later!");
    }

    #[test]
    fn test_chunk_text_by_sentences_question() {
        let text = "How are you? I am fine. Thanks!";
        let chunks = chunk_text(text, 12);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], "How are you?");
        assert_eq!(chunks[1], "I am fine.");
        assert_eq!(chunks[2], "Thanks!");
    }

    #[test]
    fn test_chunk_text_mixed() {
        let text = "Para 1. Short.\n\nPara 2 is a very long paragraph that will be split into sentences because it exceeds the chunk size when combined with other paragraphs, and then further split if needed.";
        let chunks = chunk_text(text, 50);
        assert!(chunks.len() >= 2);
        for chunk in &chunks {
            assert!(
                chunk.chars().count() <= 50,
                "Chunk exceeds limit: {} chars",
                chunk.chars().count()
            );
        }
        assert_eq!(chunks[0], "Para 1. Short.");
    }

    #[test]
    fn test_split_sentences_various() {
        let s = split_sentences("First. Second! Third? Fourth; Fifth: Sixth.");
        assert_eq!(s.len(), 6);
        assert_eq!(s[0], "First.");
        assert_eq!(s[1], "Second!");
        assert_eq!(s[2], "Third?");
        assert_eq!(s[3], "Fourth;");
        assert_eq!(s[4], "Fifth:");
        assert_eq!(s[5], "Sixth.");
    }

    #[test]
    fn test_split_sentences_no_trailing() {
        let s = split_sentences("First sentence without trailing period");
        assert_eq!(s.len(), 1);
        assert_eq!(s[0], "First sentence without trailing period");
    }
}
