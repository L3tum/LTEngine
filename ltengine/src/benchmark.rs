//! Benchmark module for evaluating translation accuracy and detection confidence calibration.
//!
//! Uses the `blue` crate for proper sentence-level BLEU with Moses-style tokenization.

use crate::backend::ProviderManager;
use crate::detection::{LanguageDetector, parse_detection_response};
use crate::languages::LANGUAGES;
use crate::prompt::PromptBuilder;
use bleuscore::{RefLenMethod, compute_score};
use std::sync::Arc;

#[derive(serde::Deserialize, Debug)]
struct BenchmarkItem {
    source: String,
    source_language: String,
    target_language: String,
    reference: String,
}

struct DetectionResult {
    detected: String,
    confidence: i32,
    correct: bool,
}

struct TranslationResult {
    bleu: f64,
    exact_match: bool,
}

struct ItemResult {
    idx: usize,
    source_lang: String,
    target_lang: String,
    llm_detection: Option<DetectionResult>,
    whatlang_detection: DetectionResult,
    translation: TranslationResult,
}

fn find_ground_truth_language(code: &str) -> Option<&str> {
    for lang in &*LANGUAGES {
        if lang.internal_code == code || lang.code == code || lang.name == code {
            return Some(lang.name);
        }
    }
    None
}

/// Embedded default benchmark dataset (shipped inside the binary).
const DEFAULT_DATASET: &str = include_str!("../benchmark_dataset.json");

pub(crate) async fn run_benchmark(args: &crate::Args, provider_manager: &Arc<ProviderManager>) {
    let dataset = match &args.dataset {
        Some(path) => std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("Failed to read dataset file '{}': {}", path, e)),
        None => DEFAULT_DATASET.to_string(),
    };

    let items: Vec<BenchmarkItem> =
        serde_json::from_str(&dataset).expect("Failed to parse dataset JSON");

    let use_llm_detect = args.llm_detect;
    let mut results = Vec::new();

    for (i, item) in items.iter().enumerate() {
        let gt_lang =
            find_ground_truth_language(&item.source_language).unwrap_or(&item.source_language);

        // LLM-based detection (if enabled)
        let llm_det = if use_llm_detect {
            let pb = PromptBuilder::new();
            let detect_prompt = pb.build_detect_prompt(&item.source);
            match provider_manager
                .translate(&detect_prompt.system, &detect_prompt.user)
                .await
            {
                Ok(result) => {
                    if let Some((lang, confidence)) = parse_detection_response(&result.text) {
                        let correct = lang.internal_code == item.source_language
                            || lang.name == gt_lang
                            || lang.code == item.source_language;
                        Some(DetectionResult {
                            detected: lang.name.to_string(),
                            confidence,
                            correct,
                        })
                    } else {
                        // Fallback to whatlang if LLM detection fails
                        let fallback = crate::detection::WhatlangDetector
                            .detect(&item.source)
                            .await;
                        Some(DetectionResult {
                            detected: fallback.language.name.to_string(),
                            confidence: fallback.confidence,
                            correct: fallback.language.internal_code == item.source_language,
                        })
                    }
                }
                Err(_) => {
                    // LLM failed, fallback to whatlang
                    let fallback = crate::detection::WhatlangDetector
                        .detect(&item.source)
                        .await;
                    Some(DetectionResult {
                        detected: fallback.language.name.to_string(),
                        confidence: fallback.confidence,
                        correct: fallback.language.internal_code == item.source_language,
                    })
                }
            }
        } else {
            None
        };

        // Whatlang detection (always)
        let whatlang_result = crate::detection::WhatlangDetector
            .detect(&item.source)
            .await;
        let whatlang_det = DetectionResult {
            detected: whatlang_result.language.name.to_string(),
            confidence: whatlang_result.confidence,
            correct: whatlang_result.language.internal_code == item.source_language,
        };

        // Translation
        let mut pb = PromptBuilder::new();
        pb.set_target_language(
            LANGUAGES
                .iter()
                .find(|l| l.code == item.target_language || l.internal_code == item.target_language)
                .map(|l| l.name)
                .unwrap_or("English"),
        );
        let translation_prompt = pb.build(&item.source);
        let translated = provider_manager
            .translate(&translation_prompt.system, &translation_prompt.user)
            .await
            .map(|result| result.text)
            .unwrap_or_else(|e| format!("[Error: {}]", e));

        // Compute sentence-level BLEU with proper tokenization via bleuscore
        let references: Vec<Vec<String>> = vec![vec![item.reference.clone()]];
        let predictions: Vec<String> = vec![translated.clone()];
        let res = compute_score(
            &references,
            &predictions,
            4,    // max_order (n-grams up to 4)
            true, // smoothing
            RefLenMethod::Shortest,
        );
        let bleu = res.bleu;
        let exact_match = translated.trim() == item.reference.trim();

        results.push(ItemResult {
            idx: i + 1,
            source_lang: item.source_language.clone(),
            target_lang: item.target_language.clone(),
            llm_detection: llm_det,
            whatlang_detection: whatlang_det,
            translation: TranslationResult { bleu, exact_match },
        });
    }

    // Print full table
    println!("\n=== Benchmark Results ===");
    let llm_header = if use_llm_detect {
        "Detected (LLM)"
    } else {
        "—"
    };
    println!(
        "\n{:<6} {:<20} {:<10} {:<20} {:<20} {:<8} {:<12} {:<8}",
        "#",
        "Source Lang (GT)",
        "Target",
        llm_header,
        "Detected (whatlang)",
        "BLEU",
        "Exact",
        "Correct"
    );

    for r in &results {
        let llm_str = r
            .llm_detection
            .as_ref()
            .map_or_else(String::new, |d| d.detected.clone());
        let whatlang_str = r.whatlang_detection.detected.clone();
        let bleu_str = format!("{:.4}", r.translation.bleu);
        let exact_str = if r.translation.exact_match {
            "Yes"
        } else {
            "No"
        };
        let correct_str = if r.whatlang_detection.correct {
            "Yes"
        } else {
            "No"
        };

        println!(
            "{:<6} {:<20} {:<10} {:<20} {:<20} {:<8} {:<12} {:<8}",
            r.idx,
            r.source_lang,
            r.target_lang,
            llm_str,
            whatlang_str,
            bleu_str,
            exact_str,
            correct_str,
        );
    }

    // Summary
    let total = results.len();
    let bleu_avg: f64 = results.iter().map(|r| r.translation.bleu).sum::<f64>() / total as f64;
    let exact_count = results.iter().filter(|r| r.translation.exact_match).count();
    let exact_rate = exact_count as f64 / total as f64;
    let whatlang_correct = results
        .iter()
        .filter(|r| r.whatlang_detection.correct)
        .count();
    let whatlang_acc = whatlang_correct as f64 / total as f64;
    let whatlang_brier: f64 = results
        .iter()
        .map(|r| {
            let conf = r.whatlang_detection.confidence as f64 / 100.0;
            let correct = if r.whatlang_detection.correct {
                1.0
            } else {
                0.0
            };
            (conf - correct).powi(2)
        })
        .sum::<f64>()
        / total as f64;

    let llm_stats = if use_llm_detect {
        let llm_correct = results
            .iter()
            .filter(|r| r.llm_detection.as_ref().is_some_and(|d| d.correct))
            .count();
        let llm_acc = llm_correct as f64 / total as f64;
        let llm_brier: f64 = results
            .iter()
            .filter_map(|r| r.llm_detection.as_ref())
            .map(|d| {
                let conf = d.confidence as f64 / 100.0;
                let correct = if d.correct { 1.0 } else { 0.0 };
                (conf - correct).powi(2)
            })
            .sum::<f64>()
            / total as f64;
        format!(
            "\nLLM Detection: accuracy = {:.1}% ({} / {}), Brier score = {:.4}",
            llm_acc * 100.0,
            llm_correct,
            total,
            llm_brier,
        )
    } else {
        String::new()
    };

    println!("\n=== Summary ===");
    println!(
        "Translation: BLEU = {:.4}, Exact match rate = {:.1}% ({} / {})",
        bleu_avg,
        exact_rate * 100.0,
        exact_count,
        total
    );
    println!(
        "Whatlang Detection: accuracy = {:.1}% ({} / {}), Brier score = {:.4}",
        whatlang_acc * 100.0,
        whatlang_correct,
        total,
        whatlang_brier
    );
    print!("{}", llm_stats);
    println!();
}
