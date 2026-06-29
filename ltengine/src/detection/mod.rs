//! Language detection module.
//!
//! Abstracts language detection behind a trait so that either the whatlang library
//! or the LLM backend can be used. A factory function creates the appropriate detector
//! based on the CLI arguments.

use crate::languages::{Language, LangDetect};
use crate::prompt::PromptBuilder;
use crate::backend::ProviderManager;

/// Trait for language detection.
///
/// Implementations can use different strategies (statistical, LLM, etc.).
#[async_trait::async_trait]
pub trait LanguageDetector: Send + Sync {
    /// Detect the language of the given text.
    async fn detect(&self, text: &str) -> LangDetect;
}

/// Attempts to match a free-text language name or code to the known languages.
/// Returns the matched Language if found, otherwise None.
///
/// This is used to parse LLM responses for language detection.
/// Only strips quotes and parentheses around the language string (not all punctuation),
/// so parenthetical language names like "Portuguese (Brazil)" still match correctly.
pub fn match_language(response: &str) -> Option<&'static Language> {
    // Trim, then strip only quotes and parentheses around the language name/code
    let cleaned = response
        .trim()
        .trim_matches(|c: char| c.is_whitespace() || c == '"' || c == '\'' || c == '(' || c == ')')
        .trim()
        .to_lowercase();

    // Try matching by name first (case-insensitive, full name including parentheticals)
    for lang in &*crate::languages::LANGUAGES {
        if lang.name.to_lowercase() == cleaned {
            return Some(lang);
        }
        // Also match by code (internal code like "en")
        if lang.internal_code == cleaned.as_str() {
            return Some(lang);
        }
        // Also match by the alias code (e.g., "zh-Hans", "pt-BR")
        if lang.code.to_lowercase() == cleaned {
            return Some(lang);
        }
    }

    None
}

/// Parse the LLM detection response into a language and confidence.
/// Expected format: "<language> <score>" where score is 0-100.
/// If the format doesn't match, tries to extract language and confidence separately.
/// Returns (Language, confidence) or None if parsing fails.
fn parse_detection_response(response: &str) -> Option<(&'static Language, i32)> {
    // Trim and split on whitespace
    let parts: Vec<&str> = response.trim().split_whitespace().collect();

    // Try to find the last element that looks like a confidence score (0-100)
    let (language_str, confidence) = if let Some(last) = parts.last() {
        if let Ok(score) = last.parse::<i32>() {
            if score >= 0 && score <= 100 {
                // Last part is a number — assume it's confidence
                let lang_str = parts[..parts.len() - 1].join(" ");
                (lang_str, score)
            } else {
                // Number out of range, not a confidence — treat all parts as language string
                (response.trim().to_string(), 100)
            }
        } else {
            // Last part is not a number — treat all as language string
            (response.trim().to_string(), 100)
        }
    } else {
        (String::new(), 100)
    };

    let lang = match_language(&language_str)?;
    Some((lang, confidence))
}

/// A detector that uses the whatlang library.
pub struct WhatlangDetector;

#[async_trait::async_trait]
impl LanguageDetector for WhatlangDetector {
    async fn detect(&self, text: &str) -> LangDetect {
        crate::languages::detect_lang(text)
    }
}

/// A detector that uses the LLM backend for language detection,
/// falling back to whatlang if the LLM is unreachable or returns no match.
pub struct LlmDetector {
    provider_manager: std::sync::Arc<ProviderManager>,
}

impl LlmDetector {
    pub fn new(provider_manager: std::sync::Arc<ProviderManager>) -> Self {
        Self { provider_manager }
    }
}

#[async_trait::async_trait]
impl LanguageDetector for LlmDetector {
    async fn detect(&self, text: &str) -> LangDetect {
        let pb = PromptBuilder::new();
        let prompt = pb.build_detect_prompt(text);

        match self
            .provider_manager
            .translate(&prompt.system, &prompt.user)
            .await
        {
            Ok(response) => {
                if let Some((lang, confidence)) = parse_detection_response(&response) {
                    LangDetect {
                        language: lang,
                        confidence,
                    }
                } else {
                    // LLM returned text we couldn't match — fallback to whatlang
                    eprintln!(
                        "LLM detection returned unparseable response ('{}'), falling back to whatlang",
                        response
                    );
                    crate::languages::detect_lang(text)
                }
            }
            Err(e) => {
                eprintln!("LLM detection failed: {}, falling back to whatlang", e);
                crate::languages::detect_lang(text)
            }
        }
    }
}

/// Factory function to create the appropriate language detector based on arguments.
pub fn create_detector(
    args: &crate::Args,
    provider_manager: &std::sync::Arc<ProviderManager>,
) -> Box<dyn LanguageDetector> {
    if args.llm_detect {
        Box::new(LlmDetector::new(provider_manager.clone()))
    } else {
        Box::new(WhatlangDetector)
    }
}
