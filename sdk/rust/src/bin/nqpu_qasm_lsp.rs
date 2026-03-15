//! nQPU-Metal QASM Language Server binary
//!
//! Speaks a minimal LSP subset over stdin/stdout via JSON-RPC for IDE integration.
//! Supports:
//! - `initialize` / `initialized`
//! - `textDocument/didOpen`
//! - `textDocument/didChange` (incremental sync, mode 2)
//! - `textDocument/completion`
//! - `textDocument/hover`
//! - `textDocument/definition`
//! - `shutdown` / `exit`
//!
//! # Building
//!
//! ```bash
//! cargo build --release --bin nqpu-qasm-lsp --features lsp
//! ```

#[cfg(feature = "lsp")]
fn main() {
    use nqpu_metal::qasm_lsp::{DiagnosticSeverity, QasmLanguageServer};
    use std::collections::HashMap;
    use std::io::{self, BufRead, Read, Write};

    let server = QasmLanguageServer::new();
    eprintln!("nQPU-Metal QASM Language Server started");

    // In-memory document store: URI -> content.
    let mut documents: HashMap<String, String> = HashMap::new();

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut reader = stdin.lock();
    let mut writer = stdout.lock();

    loop {
        // ---- Read Content-Length header ----
        let mut header_buf = String::new();
        let mut content_length: usize = 0;

        loop {
            header_buf.clear();
            if reader.read_line(&mut header_buf).unwrap_or(0) == 0 {
                // EOF — exit cleanly.
                eprintln!("nQPU-Metal QASM LSP: stdin closed, exiting");
                return;
            }
            let trimmed = header_buf.trim();
            if trimmed.is_empty() {
                // Blank line terminates headers.
                break;
            }
            if let Some(rest) = trimmed.strip_prefix("Content-Length:") {
                content_length = rest.trim().parse().unwrap_or(0);
            }
            // Ignore other headers (e.g. Content-Type).
        }

        if content_length == 0 {
            continue;
        }

        // ---- Read body ----
        let mut body_bytes = vec![0u8; content_length];
        if reader.read_exact(&mut body_bytes).is_err() {
            eprintln!("nQPU-Metal QASM LSP: failed to read body");
            continue;
        }
        let body = String::from_utf8_lossy(&body_bytes).to_string();

        // ---- Minimal JSON parsing with serde_json ----
        let msg: serde_json::Value = match serde_json::from_str(&body) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("nQPU-Metal QASM LSP: JSON parse error: {}", e);
                continue;
            }
        };

        let method = msg.get("method").and_then(|m| m.as_str()).unwrap_or("");
        let id = msg.get("id").cloned();
        let params = msg
            .get("params")
            .cloned()
            .unwrap_or(serde_json::Value::Null);

        match method {
            // ---- initialize ----
            "initialize" => {
                let result = serde_json::json!({
                    "capabilities": {
                        "textDocumentSync": {
                            "openClose": true,
                            "change": 2,  // Incremental sync
                        },
                        "completionProvider": {
                            "triggerCharacters": [" ", "(", "[", ","]
                        },
                        "hoverProvider": true,
                        "definitionProvider": true,
                    },
                    "serverInfo": {
                        "name": "nqpu-qasm-lsp",
                        "version": "0.2.0"
                    }
                });
                send_response(&mut writer, id, result);
            }

            // ---- initialized (notification, no response) ----
            "initialized" => {
                eprintln!("nQPU-Metal QASM LSP: initialized");
            }

            // ---- textDocument/didOpen ----
            "textDocument/didOpen" => {
                if let Some(td) = params.get("textDocument") {
                    let uri = td.get("uri").and_then(|u| u.as_str()).unwrap_or("");
                    let text = td.get("text").and_then(|t| t.as_str()).unwrap_or("");
                    documents.insert(uri.to_string(), text.to_string());

                    // Publish diagnostics.
                    let diags = server.validate(text);
                    publish_diagnostics(&mut writer, uri, &diags);
                }
            }

            // ---- textDocument/didChange (incremental sync, mode 2) ----
            "textDocument/didChange" => {
                if let Some(td) = params.get("textDocument") {
                    let uri = td.get("uri").and_then(|u| u.as_str()).unwrap_or("");
                    if let Some(changes) = params.get("contentChanges").and_then(|c| c.as_array()) {
                        // Get current document content (or empty if unknown).
                        let mut current = documents.get(uri).cloned().unwrap_or_default();

                        for change in changes {
                            if let Some(range) = change.get("range") {
                                // Incremental edit: apply range-based replacement.
                                let start_line = range
                                    .get("start")
                                    .and_then(|s| s.get("line"))
                                    .and_then(|l| l.as_u64())
                                    .unwrap_or(0)
                                    as usize;
                                let start_col = range
                                    .get("start")
                                    .and_then(|s| s.get("character"))
                                    .and_then(|c| c.as_u64())
                                    .unwrap_or(0)
                                    as usize;
                                let end_line = range
                                    .get("end")
                                    .and_then(|e| e.get("line"))
                                    .and_then(|l| l.as_u64())
                                    .unwrap_or(0)
                                    as usize;
                                let end_col = range
                                    .get("end")
                                    .and_then(|e| e.get("character"))
                                    .and_then(|c| c.as_u64())
                                    .unwrap_or(0)
                                    as usize;
                                let new_text =
                                    change.get("text").and_then(|t| t.as_str()).unwrap_or("");

                                current = QasmLanguageServer::apply_incremental_edit(
                                    &current, start_line, start_col, end_line, end_col, new_text,
                                );
                            } else {
                                // Full content replacement (fallback for mode 1
                                // clients or events without a range).
                                if let Some(text) = change.get("text").and_then(|t| t.as_str()) {
                                    current = text.to_string();
                                }
                            }
                        }

                        documents.insert(uri.to_string(), current.clone());

                        let diags = server.validate(&current);
                        publish_diagnostics(&mut writer, uri, &diags);
                    }
                }
            }

            // ---- textDocument/completion ----
            "textDocument/completion" => {
                let uri = params
                    .get("textDocument")
                    .and_then(|td| td.get("uri"))
                    .and_then(|u| u.as_str())
                    .unwrap_or("");
                let line = params
                    .get("position")
                    .and_then(|p| p.get("line"))
                    .and_then(|l| l.as_u64())
                    .unwrap_or(0) as usize;
                let col = params
                    .get("position")
                    .and_then(|p| p.get("character"))
                    .and_then(|c| c.as_u64())
                    .unwrap_or(0) as usize;

                let source = documents.get(uri).map(|s| s.as_str()).unwrap_or("");
                let completions = server.complete(source, line, col);

                let items: Vec<serde_json::Value> = completions
                    .iter()
                    .map(|c| {
                        serde_json::json!({
                            "label": c.label,
                            "detail": c.detail,
                            "insertText": c.insert_text,
                            "kind": match c.kind {
                                nqpu_metal::qasm_lsp::CompletionKind::Gate => 3,      // Function
                                nqpu_metal::qasm_lsp::CompletionKind::Keyword => 14,  // Keyword
                                nqpu_metal::qasm_lsp::CompletionKind::Type => 7,      // Class
                                nqpu_metal::qasm_lsp::CompletionKind::Snippet => 15,  // Snippet
                            },
                            "insertTextFormat": if c.insert_text.contains("${") { 2 } else { 1 }
                        })
                    })
                    .collect();

                send_response(&mut writer, id, serde_json::json!(items));
            }

            // ---- textDocument/hover ----
            "textDocument/hover" => {
                let uri = params
                    .get("textDocument")
                    .and_then(|td| td.get("uri"))
                    .and_then(|u| u.as_str())
                    .unwrap_or("");
                let line = params
                    .get("position")
                    .and_then(|p| p.get("line"))
                    .and_then(|l| l.as_u64())
                    .unwrap_or(0) as usize;
                let col = params
                    .get("position")
                    .and_then(|p| p.get("character"))
                    .and_then(|c| c.as_u64())
                    .unwrap_or(0) as usize;

                let source = documents.get(uri).map(|s| s.as_str()).unwrap_or("");
                let result = if let Some(hover) = server.hover(source, line, col) {
                    let mut obj = serde_json::json!({
                        "contents": {
                            "kind": "markdown",
                            "value": hover.contents
                        }
                    });
                    if let Some((sl, sc, el, ec)) = hover.range {
                        obj["range"] = serde_json::json!({
                            "start": { "line": sl, "character": sc },
                            "end": { "line": el, "character": ec }
                        });
                    }
                    obj
                } else {
                    serde_json::Value::Null
                };

                send_response(&mut writer, id, result);
            }

            // ---- textDocument/definition ----
            "textDocument/definition" => {
                let uri = params
                    .get("textDocument")
                    .and_then(|td| td.get("uri"))
                    .and_then(|u| u.as_str())
                    .unwrap_or("");
                let line = params
                    .get("position")
                    .and_then(|p| p.get("line"))
                    .and_then(|l| l.as_u64())
                    .unwrap_or(0) as usize;
                let col = params
                    .get("position")
                    .and_then(|p| p.get("character"))
                    .and_then(|c| c.as_u64())
                    .unwrap_or(0) as usize;

                let source = documents.get(uri).map(|s| s.as_str()).unwrap_or("");
                let result = if let Some(loc) = server.goto_definition(source, line, col) {
                    serde_json::json!({
                        "uri": uri,
                        "range": {
                            "start": { "line": loc.line, "character": loc.col },
                            "end": { "line": loc.line, "character": loc.end_col }
                        }
                    })
                } else {
                    serde_json::Value::Null
                };

                send_response(&mut writer, id, result);
            }

            // ---- shutdown ----
            "shutdown" => {
                send_response(&mut writer, id, serde_json::Value::Null);
            }

            // ---- exit ----
            "exit" => {
                eprintln!("nQPU-Metal QASM LSP: exit");
                return;
            }

            // ---- Unknown method ----
            _ => {
                eprintln!("nQPU-Metal QASM LSP: unknown method '{}'", method);
                // If it has an ID, respond with method-not-found.
                if let Some(req_id) = id {
                    let err_response = serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {
                            "code": -32601,
                            "message": format!("Method not found: {}", method)
                        }
                    });
                    let resp = serde_json::to_string(&err_response).unwrap();
                    let _ = write!(writer, "Content-Length: {}\r\n\r\n{}", resp.len(), resp);
                    let _ = writer.flush();
                }
            }
        }
    }

    /// Send a JSON-RPC response.
    fn send_response(
        writer: &mut impl Write,
        id: Option<serde_json::Value>,
        result: serde_json::Value,
    ) {
        let resp = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": result
        });
        let body = serde_json::to_string(&resp).unwrap();
        let _ = write!(writer, "Content-Length: {}\r\n\r\n{}", body.len(), body);
        let _ = writer.flush();
    }

    /// Publish diagnostics notification.
    fn publish_diagnostics(
        writer: &mut impl Write,
        uri: &str,
        diags: &[nqpu_metal::qasm_lsp::QasmDiagnostic],
    ) {
        let diag_items: Vec<serde_json::Value> = diags
            .iter()
            .map(|d| {
                let severity = match d.severity {
                    DiagnosticSeverity::Error => 1,
                    DiagnosticSeverity::Warning => 2,
                    DiagnosticSeverity::Information => 3,
                    DiagnosticSeverity::Hint => 4,
                };
                let mut obj = serde_json::json!({
                    "range": {
                        "start": { "line": d.line, "character": d.col },
                        "end": { "line": d.line, "character": d.end_col }
                    },
                    "severity": severity,
                    "message": d.message,
                    "source": "nqpu-qasm-lsp"
                });
                if let Some(code) = &d.code {
                    obj["code"] = serde_json::json!(code);
                }
                obj
            })
            .collect();

        let notification = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "textDocument/publishDiagnostics",
            "params": {
                "uri": uri,
                "diagnostics": diag_items
            }
        });
        let body = serde_json::to_string(&notification).unwrap();
        let _ = write!(writer, "Content-Length: {}\r\n\r\n{}", body.len(), body);
        let _ = writer.flush();
    }
}

#[cfg(not(feature = "lsp"))]
fn main() {
    eprintln!("Error: nqpu-qasm-lsp requires the 'lsp' feature flag.");
    eprintln!("Build with: cargo build --release --bin nqpu-qasm-lsp --features lsp");
    std::process::exit(1);
}
