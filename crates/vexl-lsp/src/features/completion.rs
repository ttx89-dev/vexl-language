use tower_lsp::lsp_types::*;
use std::collections::HashMap;

use vexl_syntax::parser::parse_for_lsp;
use vexl_syntax::ast::Decl;
use crate::semantic::{find_symbols, SymbolInfo};

/// Completion context for intelligent suggestions
pub struct CompletionContext {
    /// Current symbols in scope
    symbols: HashMap<String, SymbolInfo>,
    /// Current line content
    line_content: String,
    /// Cursor position in line
    cursor_col: usize,
    /// Current word being typed
    current_word: String,
}

impl CompletionContext {
    /// Create completion context from source and position
    pub fn new(source: &str, position: Position) -> Result<Self, String> {
        let line = position.line as usize;
        let col = position.character as usize;

        // Parse source for semantic analysis
        let program = parse_for_lsp(source)?;

        // Find symbols - need to extract expressions from declarations
        let mut all_symbols = HashMap::new();
        for decl in &program {
            match decl {
                Decl::Expr(expr) => {
                    let symbols = find_symbols(expr, "current")?;
                    all_symbols.extend(symbols);
                }
                Decl::Function { name, params, body, .. } => {
                    // Create symbol for function itself
                    let func_symbol = SymbolInfo {
                        name: name.clone(),
                        kind: crate::semantic::SymbolKind::Function,
                        ty: Some("function".to_string()),
                        definition: crate::semantic::Location {
                            line: 0,
                            column: 0,
                            module: "current".to_string(),
                        },
                        references: Vec::new(),
                        documentation: None,
                    };
                    all_symbols.insert(name.clone(), func_symbol);

                    // Add parameter symbols
                    for (param_name, _) in params {
                        let param_symbol = SymbolInfo {
                            name: param_name.clone(),
                            kind: crate::semantic::SymbolKind::Parameter,
                            ty: Some("parameter".to_string()),
                            definition: crate::semantic::Location {
                                line: 0,
                                column: 0,
                                module: "current".to_string(),
                            },
                            references: Vec::new(),
                            documentation: None,
                        };
                        all_symbols.insert(param_name.clone(), param_symbol);
                    }

                    // Find symbols in function body
                    let body_symbols = find_symbols(body, "current")?;
                    all_symbols.extend(body_symbols);
                }
            }
        }

        let symbols = all_symbols;

        // Extract current line and word
        let lines: Vec<&str> = source.lines().collect();
        let line_content = lines.get(line).map_or("", |v| v).to_string();

        let current_word = extract_current_word(&line_content, col);

        Ok(Self {
            symbols,
            line_content,
            cursor_col: col,
            current_word,
        })
    }

    /// Get context-aware completions
    pub fn get_completions(&self) -> Vec<CompletionItem> {
        let mut completions = Vec::new();

        // Add symbol completions
        completions.extend(self.get_symbol_completions());

        // Add keyword completions
        completions.extend(self.get_keyword_completions());

        // Add snippet completions
        completions.extend(self.get_snippet_completions());

        // Add standard library completions
        completions.extend(self.get_stdlib_completions());

        // Filter by current word
        if !self.current_word.is_empty() {
            completions.retain(|item| {
                item.label.starts_with(&self.current_word) ||
                item.filter_text.as_ref().unwrap_or(&item.label).starts_with(&self.current_word)
            });
        }

        completions
    }

    /// Get completions for symbols in scope
    fn get_symbol_completions(&self) -> Vec<CompletionItem> {
        let mut completions = Vec::new();

        for (name, symbol) in &self.symbols {
            let kind = symbol.kind.to_completion_item_kind();

            let detail = match &symbol.ty {
                Some(ty) => ty.clone(),
                None => format!("{:?}", symbol.kind),
            };

            completions.push(CompletionItem {
                label: (*name).to_string(),
                kind: Some(kind),
                detail: Some(detail),
                documentation: symbol.documentation.clone().map(|doc| {
                    Documentation::MarkupContent(MarkupContent {
                        kind: MarkupKind::Markdown,
                        value: doc,
                    })
                }),
                ..Default::default()
            });
        }

        completions
    }

    /// Get keyword completions
    fn get_keyword_completions(&self) -> Vec<CompletionItem> {
        vec![
            CompletionItem {
                label: "fn".to_string(),
                kind: Some(CompletionItemKind::KEYWORD),
                detail: Some("Define a function".to_string()),
                documentation: Some(Documentation::MarkupContent(MarkupContent {
                    kind: MarkupKind::Markdown,
                    value: "Define a new function\n\n```vexl\nfn name(param1, param2) {\n    // body\n}\n```".to_string(),
                })),
                ..Default::default()
            },
            CompletionItem {
                label: "let".to_string(),
                kind: Some(CompletionItemKind::KEYWORD),
                detail: Some("Bind a variable".to_string()),
                documentation: Some(Documentation::MarkupContent(MarkupContent {
                    kind: MarkupKind::Markdown,
                    value: "Bind a value to a variable\n\n```vexl\nlet x = 42;\n```".to_string(),
                })),
                ..Default::default()
            },
            CompletionItem {
                label: "if".to_string(),
                kind: Some(CompletionItemKind::KEYWORD),
                detail: Some("Conditional expression".to_string()),
                documentation: Some(Documentation::MarkupContent(MarkupContent {
                    kind: MarkupKind::Markdown,
                    value: "Conditional execution\n\n```vexl\nif condition {\n    // then\n} else {\n    // else\n}\n```".to_string(),
                })),
                ..Default::default()
            },
            CompletionItem {
                label: "else".to_string(),
                kind: Some(CompletionItemKind::KEYWORD),
                detail: Some("Else clause".to_string()),
                ..Default::default()
            },
            CompletionItem {
                label: "for".to_string(),
                kind: Some(CompletionItemKind::KEYWORD),
                detail: Some("Loop construct".to_string()),
                ..Default::default()
            },
            CompletionItem {
                label: "while".to_string(),
                kind: Some(CompletionItemKind::KEYWORD),
                detail: Some("While loop".to_string()),
                ..Default::default()
            },
            CompletionItem {
                label: "return".to_string(),
                kind: Some(CompletionItemKind::KEYWORD),
                detail: Some("Return from function".to_string()),
                ..Default::default()
            },
        ]
    }

    /// Get snippet completions
    fn get_snippet_completions(&self) -> Vec<CompletionItem> {
        vec![
            CompletionItem {
                label: "vector".to_string(),
                kind: Some(CompletionItemKind::SNIPPET),
                detail: Some("Create a vector".to_string()),
                insert_text: Some("vector[${1:items}]".to_string()),
                insert_text_format: Some(InsertTextFormat::SNIPPET),
                documentation: Some(Documentation::MarkupContent(MarkupContent {
                    kind: MarkupKind::Markdown,
                    value: "Create a vector from elements\n\n```vexl\nvector[1, 2, 3, 4]\n```".to_string(),
                })),
                ..Default::default()
            },
            CompletionItem {
                label: "function".to_string(),
                kind: Some(CompletionItemKind::SNIPPET),
                detail: Some("Function definition".to_string()),
                insert_text: Some("fn ${1:name}(${2:params}) {\n    ${3:// body}\n}".to_string()),
                insert_text_format: Some(InsertTextFormat::SNIPPET),
                documentation: Some(Documentation::MarkupContent(MarkupContent {
                    kind: MarkupKind::Markdown,
                    value: "Define a function\n\n```vexl\nfn add(x, y) {\n    x + y\n}\n```".to_string(),
                })),
                ..Default::default()
            },
            CompletionItem {
                label: "let_binding".to_string(),
                kind: Some(CompletionItemKind::SNIPPET),
                detail: Some("Variable binding".to_string()),
                insert_text: Some("let ${1:name} = ${2:value};".to_string()),
                insert_text_format: Some(InsertTextFormat::SNIPPET),
                ..Default::default()
            },
            CompletionItem {
                label: "if_else".to_string(),
                kind: Some(CompletionItemKind::SNIPPET),
                detail: Some("If-else expression".to_string()),
                insert_text: Some("if ${1:condition} {\n    ${2:// then}\n} else {\n    ${3:// else}\n}".to_string()),
                insert_text_format: Some(InsertTextFormat::SNIPPET),
                ..Default::default()
            },
        ]
    }

    /// Get standard library completions
    fn get_stdlib_completions(&self) -> Vec<CompletionItem> {
        vec![
            CompletionItem {
                label: "vec_add".to_string(),
                kind: Some(CompletionItemKind::FUNCTION),
                detail: Some("Vector addition".to_string()),
                documentation: Some(Documentation::MarkupContent(MarkupContent {
                    kind: MarkupKind::Markdown,
                    value: "Add two vectors element-wise\n\n```vexl\nvec_add(a, b)\n```".to_string(),
                })),
                ..Default::default()
            },
            CompletionItem {
                label: "vec_mul".to_string(),
                kind: Some(CompletionItemKind::FUNCTION),
                detail: Some("Vector multiplication".to_string()),
                ..Default::default()
            },
            CompletionItem {
                label: "vec_sum".to_string(),
                kind: Some(CompletionItemKind::FUNCTION),
                detail: Some("Sum vector elements".to_string()),
                ..Default::default()
            },
            CompletionItem {
                label: "vec_max".to_string(),
                kind: Some(CompletionItemKind::FUNCTION),
                detail: Some("Maximum vector element".to_string()),
                ..Default::default()
            },
            CompletionItem {
                label: "vec_min".to_string(),
                kind: Some(CompletionItemKind::FUNCTION),
                detail: Some("Minimum vector element".to_string()),
                ..Default::default()
            },
            CompletionItem {
                label: "math_sin".to_string(),
                kind: Some(CompletionItemKind::FUNCTION),
                detail: Some("Sine function".to_string()),
                ..Default::default()
            },
            CompletionItem {
                label: "math_cos".to_string(),
                kind: Some(CompletionItemKind::FUNCTION),
                detail: Some("Cosine function".to_string()),
                ..Default::default()
            },
            CompletionItem {
                label: "math_sqrt".to_string(),
                kind: Some(CompletionItemKind::FUNCTION),
                detail: Some("Square root".to_string()),
                ..Default::default()
            },
            CompletionItem {
                label: "print".to_string(),
                kind: Some(CompletionItemKind::FUNCTION),
                detail: Some("Print value".to_string()),
                ..Default::default()
            },
        ]
    }
}

/// Extract current word being typed at cursor position
fn extract_current_word(line: &str, cursor_col: usize) -> String {
    let line_chars: Vec<char> = line.chars().collect();

    if cursor_col > line_chars.len() {
        return String::new();
    }

    // Find word boundaries (letters, digits, underscore)
    let mut start = cursor_col;
    while start > 0 && is_word_char(line_chars[start - 1]) {
        start -= 1;
    }

    let mut end = cursor_col;
    while end < line_chars.len() && is_word_char(line_chars[end]) {
        end += 1;
    }

    line_chars[start..end].iter().collect()
}

/// Check if character is part of a word
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Main completion function
pub fn complete(text: &str, position: Position) -> Option<CompletionResponse> {
    match CompletionContext::new(text, position) {
        Ok(context) => {
            let completions = context.get_completions();
            Some(CompletionResponse::Array(completions))
        }
        Err(_) => {
            // Fallback to basic completions
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
            ]))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_current_word() {
        assert_eq!(extract_current_word("let x = 5", 4), "x");
        assert_eq!(extract_current_word("fn add(a, b)", 6), "add");
        assert_eq!(extract_current_word("vec_", 4), "vec_");
        assert_eq!(extract_current_word("let x = vec", 11), "vec");
    }

    #[test]
    fn test_completion_context() {
        let source = "let x = 5;\nlet y = x + ";
        let position = Position { line: 1, character: 11 };

        let context = CompletionContext::new(source, position).unwrap();
        assert_eq!(context.current_word, "x");

        let completions = context.get_completions();
        assert!(!completions.is_empty());

        // Should include variable x
        let x_completion = completions.iter().find(|c| c.label == "x");
        assert!(x_completion.is_some());
        assert_eq!(x_completion.unwrap().kind, Some(CompletionItemKind::VARIABLE));
    }
}
