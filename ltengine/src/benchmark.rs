//! Benchmark module for evaluating translation accuracy and detection confidence calibration.
//!
//! Uses the `bleuscore` crate for proper sentence-level BLEU with Moses-style tokenization.

use crate::backend::ProviderManager;
use crate::detection::{LanguageDetector, parse_detection_response};
use crate::languages::LANGUAGES;
use crate::prompt::PromptBuilder;
use bleuscore::{RefLenMethod, compute_score};
use std::io::{IsTerminal, Write};
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
    // Stored for HTML report
    source: String,
    reference: String,
    translated: String,
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

/// Embedded HTML report template (shipped inside the binary).
const HTML_TEMPLATE: &str = include_str!("benchmark_report.html");

/// Run benchmark and produce either a text report or an HTML file.
pub(crate) async fn run_benchmark(
    args: &crate::Args,
    benchmark_mode: &crate::BenchmarkMode,
    provider_manager: &Arc<ProviderManager>,
) {
    let dataset = match &args.dataset {
        Some(path) => std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("Failed to read dataset file '{}': {}", path, e)),
        None => DEFAULT_DATASET.to_string(),
    };

    let items: Vec<BenchmarkItem> =
        serde_json::from_str(&dataset).expect("Failed to parse dataset JSON");

    let use_llm_detect = args.llm_detect;
    let mut results = Vec::new();
    let total = items.len();
    let is_terminal = std::io::stdout().is_terminal();

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

        // Show progress line (overwrite previous if interactive)
        let progress = if use_llm_detect {
            let llm_det_str = llm_det.as_ref().map_or_else(String::new, |d| {
                format!(" ({} {}%)", d.detected, d.confidence)
            });
            format!(
                "Processing {}/{}: {} → {} [LLM: {} | Whatlang: {} {}%]  ",
                i + 1,
                total,
                item.source_language,
                item.target_language,
                llm_det_str,
                whatlang_det.detected,
                whatlang_det.confidence
            )
        } else {
            format!(
                "Processing {}/{}: {} → {} [Detected: {} {}%]  ",
                i + 1,
                total,
                item.source_language,
                item.target_language,
                whatlang_det.detected,
                whatlang_det.confidence
            )
        };
        if is_terminal {
            print!("\r\033[K");
            print!("{}", progress);
            std::io::stdout().flush().ok();
        } else {
            println!("{}", progress);
        }

        results.push(ItemResult {
            idx: i + 1,
            source_lang: item.source_language.clone(),
            target_lang: item.target_language.clone(),
            llm_detection: llm_det,
            whatlang_detection: whatlang_det,
            translation: TranslationResult { bleu, exact_match },
            source: item.source.clone(),
            reference: item.reference.clone(),
            translated: translated.clone(),
        });
    }

    // Move past the progress line
    if is_terminal {
        println!(); // newline after the progress line
    }

    // Compute summary statistics once (used for both text output and HTML report)
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

    // Always print text output to CLI (progress + results table + summary + explanation)
    println!("=== Benchmark Results ===");
    let llm_header = if use_llm_detect {
        "Detected (LLM)"
    } else {
        "—"
    };
    println!(
        "\n{:<6} {:<16} {:<16} {:<16} {:<10} {:<10} {:<12}",
        "#", "Source Lang (GT)", llm_header, "Detected (Whatlang)", "Target", "BLEU", "Exact"
    );

    for r in &results {
        let source_name = find_ground_truth_language(&r.source_lang).unwrap_or(&r.source_lang);
        let target_name = find_ground_truth_language(&r.target_lang).unwrap_or(&r.target_lang);

        let llm_str = r.llm_detection.as_ref().map_or_else(String::new, |d| {
            let indicator = if d.correct {
                "\x1b[32m ✓\x1b[0m"
            } else {
                "\x1b[31m ✗\x1b[0m"
            };
            format!("{}{}", d.detected, indicator)
        });
        let whatlang_indicator = if r.whatlang_detection.correct {
            "\x1b[32m ✓\x1b[0m"
        } else {
            "\x1b[31m ✗\x1b[0m"
        };
        let whatlang_str = format!("{}{}", r.whatlang_detection.detected, whatlang_indicator);

        let bleu_val = r.translation.bleu;
        let bleu_coloured = if bleu_val >= 0.8 {
            format!("\x1b[32m{:.4}\x1b[0m", bleu_val)
        } else {
            format!("\x1b[31m{:.4}\x1b[0m", bleu_val)
        };

        let exact_str = if r.translation.exact_match {
            "\x1b[32mYes\x1b[0m"
        } else {
            "\x1b[31mNo\x1b[0m"
        };

        println!(
            "{:<6} {:<16} {:<16} {:<16} {:<10} {:<10} {:<12}",
            r.idx, source_name, llm_str, whatlang_str, target_name, bleu_coloured, exact_str,
        );
    }

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
    // BLEU explanation
    println!(
        "BLEU score: 0 = no overlap with reference, 1 = perfect match. Score is based on\nn-gram precision (up to 4-grams) with shortest reference length and smoothing."
    );

    // If HTML mode, also generate the HTML report file
    if matches!(benchmark_mode, crate::BenchmarkMode::Html) {
        generate_html_report(
            &results,
            total,
            use_llm_detect,
            bleu_avg,
            exact_count,
            exact_rate,
            whatlang_correct,
            whatlang_acc,
            whatlang_brier,
            &llm_stats,
        );
    }
}

/// Generate a self-contained HTML report using an embedded template.
#[allow(clippy::too_many_arguments)]
fn generate_html_report(
    results: &[ItemResult],
    total: usize,
    use_llm_detect: bool,
    bleu_avg: f64,
    exact_count: usize,
    exact_rate: f64,
    whatlang_correct: usize,
    whatlang_acc: f64,
    whatlang_brier: f64,
    llm_stats: &str,
) {
    let llm_acc = if use_llm_detect {
        llm_stats.trim_start_matches('\n').trim().to_string()
    } else {
        String::new()
    };

    // Build the table rows and data/labels for the chart
    let rows = results
        .iter()
        .map(|r| {
            let source_name = find_ground_truth_language(&r.source_lang).unwrap_or(&r.source_lang);
            let target_name = find_ground_truth_language(&r.target_lang).unwrap_or(&r.target_lang);

            let llm_str = r.llm_detection.as_ref().map_or_else(String::new, |d| {
                let indicator_class = if d.correct { "green" } else { "red" };
                let indicator = if d.correct { " ✓" } else { " ✗" };
                format!(
                    "{}<span class=\"{}\">{}</span>",
                    d.detected, indicator_class, indicator
                )
            });
            let whatlang_indicator_class = if r.whatlang_detection.correct {
                "green"
            } else {
                "red"
            };
            let whatlang_indicator = if r.whatlang_detection.correct {
                " ✓"
            } else {
                " ✗"
            };
            let whatlang_str = format!(
                "{}<span class=\"{}\">{}</span>",
                r.whatlang_detection.detected, whatlang_indicator_class, whatlang_indicator
            );

            let bleu_val = r.translation.bleu;
            let bleu_class = if bleu_val >= 0.8 { "green" } else { "red" };
            let bleu_str = format!("<span class=\"{}\">{:.4}</span>", bleu_class, bleu_val);

            let exact_class = if r.translation.exact_match {
                "green"
            } else {
                "red"
            };
            let exact_str = if r.translation.exact_match {
                format!("<span class=\"{}\">Yes</span>", exact_class)
            } else {
                format!("<span class=\"{}\">No</span>", exact_class)
            };

            // Escape HTML entities for source, reference, and translation
            let source_escaped = r.source.replace('&', "&amp;").replace('<', "&lt;");
            let reference_escaped = r.reference.replace('&', "&amp;").replace('<', "&lt;");
            let translated_escaped = r.translated.replace('&', "&amp;").replace('<', "&lt;");

            format!(
                r#"<tr>
<td class="clickable">{}</td>
<td>{}</td>
<td>{}</td>
<td>{}</td>
<td>{}</td>
<td>{}</td>
<td>{}</td>
</tr>
<tr>
<td colspan="7" class="detail">
<label>Source:</label><span>{}</span>
<label>Reference:</label><span>{}</span>
<label>LLM Translation:</label><span>{}</span>
</td>
</tr>"#,
                r.idx,
                source_name,
                llm_str,
                whatlang_str,
                target_name,
                bleu_str,
                exact_str,
                source_escaped,
                reference_escaped,
                translated_escaped,
            )
        })
        .collect::<String>();
    let data = results
        .iter()
        .map(|r| format!("{:.4}", r.translation.bleu))
        .collect::<Vec<_>>()
        .join(",");
    let labels = results
        .iter()
        .map(|r| {
            let s = find_ground_truth_language(&r.source_lang).unwrap_or(&r.source_lang);
            let t = find_ground_truth_language(&r.target_lang).unwrap_or(&r.target_lang);
            format!("\"{}→{}\"", s, t)
        })
        .collect::<Vec<_>>()
        .join(",");

    // Use manual string replacement with named placeholders
    let html = HTML_TEMPLATE
        .replace("{{bleu_avg}}", &format!("{:.4}", bleu_avg))
        .replace("{{exact_rate}}", &format!("{:.1}%", exact_rate * 100.0))
        .replace("{{exact_count}}", &exact_count.to_string())
        .replace("{{total}}", &total.to_string())
        .replace("{{whatlang_acc}}", &format!("{:.1}%", whatlang_acc * 100.0))
        .replace("{{whatlang_correct}}", &whatlang_correct.to_string())
        .replace("{{whatlang_brier}}", &format!("{:.4}", whatlang_brier))
        .replace("{__llm_stats__}", &llm_acc)
        .replace("{__data__}", &data)
        .replace("{__labels__}", &labels)
        .replace("{__rows__}", &rows);

    // Write to file
    let filename = "benchmark_report.html";
    std::fs::write(filename, &html).expect("Failed to write HTML report");
    println!("HTML report generated: {}", filename);
}
