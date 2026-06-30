//! Embedded static file handler — serves the LibreTranslate frontend.
//!
//! Replaces `actix-web-static-files` to avoid duplicate dependencies
//! (`derive_more` 0.99 vs 2.x, `static-files` 0.2 vs 0.3).

use actix_web::{HttpRequest, HttpResponse, Result as ActixResult};

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

/// Minimal MIME type lookup — avoids adding `mime_guess` as a direct dep.
fn get_mime_type(path: &str) -> &'static str {
    if let Some(ext) = path.rsplit('.').next() {
        return match ext {
            "html" | "htm" => "text/html",
            "css" => "text/css",
            "js" => "application/javascript",
            "json" => "application/json",
            "svg" => "image/svg+xml",
            "woff" => "font/woff",
            "woff2" => "font/woff2",
            "ttf" => "font/ttf",
            "eot" => "application/vnd.ms-fontobject",
            "png" => "image/png",
            "jpg" | "jpeg" => "image/jpeg",
            "gif" => "image/gif",
            "ico" => "image/x-icon",
            "webp" => "image/webp",
            "xml" => "text/xml",
            "txt" => "text/plain",
            "map" => "application/json",
            _ => "application/octet-stream",
        };
    }
    "application/octet-stream"
}

/// Handle any request not matched by API routes: serves the embedded frontend.
///
/// For `/` or directory-like paths (ending in `/`), serves `index.html`.
/// For other paths, looks up the file directly in the embedded resources.
/// Returns 404 if the file is not found.
pub async fn serve_static(req: HttpRequest) -> ActixResult<HttpResponse> {
    let generated = generate();
    let path = req.path();

    // For `/` or empty path, serve index.html
    if path == "/" || path.is_empty() {
        if let Some(file) = generated.get("index.html") {
            return Ok(HttpResponse::Ok()
                .append_header(("Content-Type", "text/html"))
                .body(file.data.to_vec()));
        }
        return Ok(HttpResponse::NotFound().finish());
    }

    // For paths ending with `/` (directory), serve index.html
    if path.ends_with('/') {
        if let Some(file) = generated.get("index.html") {
            return Ok(HttpResponse::Ok()
                .append_header(("Content-Type", "text/html"))
                .body(file.data.to_vec()));
        }
        return Ok(HttpResponse::NotFound().finish());
    }

    // For normal file paths, look up directly (remove leading `/`)
    let lookup_path = if let Some(stripped) = path.strip_prefix('/') {
        stripped
    } else {
        path
    };

    if let Some(file) = generated.get(lookup_path) {
        let mime = get_mime_type(lookup_path);
        return Ok(HttpResponse::Ok()
            .append_header(("Content-Type", mime))
            .body(file.data.to_vec()));
    }

    // 404 for unknown files
    Ok(HttpResponse::NotFound().finish())
}
