
use tower_lsp::lsp_types::*;

pub fn complete(_text: &str, _position: Position) -> Option<CompletionResponse> {
    Some(CompletionResponse::Array(vec![
        CompletionItem {
            label: "fn".to_string(),
            kind: Some(CompletionItemKind::KEYWORD),
            detail: Some("Define a function".to_string()),
            ..Default::default()
        },
        CompletionItem {
            label: "let".to_string(),
            kind: Some(CompletionItemKind::KEYWORD),
            detail: Some("Bind a variable".to_string()),
            ..Default::default()
        },
        CompletionItem {
            label: "match".to_string(),
            kind: Some(CompletionItemKind::KEYWORD),
            detail: Some("Pattern matching".to_string()),
            ..Default::default()
        },
        CompletionItem {
            label: "vector".to_string(),
            kind: Some(CompletionItemKind::SNIPPET),
            detail: Some("Create a vector".to_string()),
            insert_text: Some("vector[${1:items}]".to_string()),
            insert_text_format: Some(InsertTextFormat::SNIPPET),
            ..Default::default()
        },
    ]))
}
