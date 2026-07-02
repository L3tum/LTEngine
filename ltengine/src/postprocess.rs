//! Post-processing of translated text to improve formatting.

/// Mirror the source text's casing, punctuation, and capitalization patterns.
/// Adjusts the translation's final punctuation, casing (all-lowercase, all-uppercase),
/// and first-character case to match the source.
pub fn improve_formatting(q: &str, translation: &str) -> String {
    let t = translation.trim().to_string();

    if q.is_empty() {
        return String::new();
    }

    if t.is_empty() {
        return q.to_owned();
    }

    let q_last_char = q.chars().next_back().unwrap();
    let translation_last_char = t.chars().next_back().unwrap();
    let mut result = t.clone();

    const PUNCTUATION_CHARS: [char; 6] = ['!', '?', '.', ',', ';', '。'];
    if PUNCTUATION_CHARS.contains(&q_last_char) {
        if q_last_char != translation_last_char {
            if PUNCTUATION_CHARS.contains(&translation_last_char) {
                result.pop();
            }

            result.push(q_last_char);
        }
    } else if PUNCTUATION_CHARS.contains(&translation_last_char) {
        result.pop();
    }

    if q.chars().all(|c| c.is_lowercase()) {
        result = result.to_lowercase();
    }

    if q.chars().all(|c| c.is_uppercase()) {
        result = result.to_uppercase();
    }

    if let (Some(q0), Some(r0)) = (q.chars().next(), result.chars().next()) {
        if q0.is_lowercase() && r0.is_uppercase() {
            result.replace_range(0..r0.len_utf8(), &r0.to_lowercase().to_string());
        } else if q0.is_uppercase() && r0.is_lowercase() {
            result.replace_range(0..r0.len_utf8(), &r0.to_uppercase().to_string());
        }
    }

    result.trim().to_string()
}
