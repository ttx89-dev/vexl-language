//! Hover Information for VEXL LSP
//!
//! Provides rich hover tooltips showing:
//! - Type information for variables and expressions
//! - Function signatures and documentation
//! - Module and namespace information
//! - Effect system information (Pure, IO, etc.)

use tower_lsp::lsp_types::*;
use crate::semantic::{SemanticAnalyzer, SymbolInfo, SymbolKind, find_symbols, get_type_info};
use vexl_syntax::parser::parse_for_lsp;

/// Hover context for providing detailed information
pub struct HoverContext {
    /// Current symbols in scope
    symbols: std::collections::HashMap<String, SymbolInfo>,
    /// Current line content
    line_content: String,
    /// Cursor position in line
    cursor_col: usize,
    /// Current word being hovered over
    current_word: String,
}

impl HoverContext {
    /// Create hover context from source and position
    pub fn new(source: &str, position: Position) -> Result<Self, String> {
        let line = position.line as usize;
        let col = position.character as usize;

        // Parse the source for semantic analysis
        let program = parse_for_lsp(source)?;

        // Find symbols from declarations
        let symbols = crate::semantic::find_symbols_from_decls(&program, "current")?;

        // Extract current line and word
        let lines: Vec<&str> = source.lines().collect();
        let line_content = lines.get(line).map_or("", |v| v).to_string();

        let current_word = extract_word_at_position(&line_content, col);

        Ok(Self {
            symbols,
            line_content,
            cursor_col: col,
            current_word,
        })
    }

    /// Get hover information for the current position
    pub fn get_hover_info(&self) -> Option<Hover> {
        // If we have a current word, try to find symbol information
        if !self.current_word.is_empty() {
            if let Some(symbol) = self.symbols.get(&self.current_word) {
                return Some(self.create_symbol_hover(symbol));
            }

            // Check for built-in functions and types
            if let Some(hover) = self.get_builtin_hover(&self.current_word) {
                return Some(hover);
            }
        }

        // Try to get type information for expressions
        if let Some(type_info) = self.get_expression_type_info() {
            return Some(Hover {
                contents: HoverContents::Markup(MarkupContent {
                    kind: MarkupKind::Markdown,
                    value: format!("```vexl\n{}\n```", type_info),
                }),
                range: None,
            });
        }

        None
    }

    /// Create hover information for a symbol
    fn create_symbol_hover(&self, symbol: &SymbolInfo) -> Hover {
        let mut content = String::new();

        // Symbol kind and name
        let kind_str = match symbol.kind {
            SymbolKind::Variable => "variable",
            SymbolKind::Function => "function",
            SymbolKind::Parameter => "parameter",
            SymbolKind::TypeAlias => "type alias",
            SymbolKind::Module => "module",
            SymbolKind::VectorOp => "vector operation",
        };

        content.push_str(&format!("**{}** `{}`\n\n", kind_str, symbol.name));

        // Type information
        if let Some(ty) = &symbol.ty {
            content.push_str(&format!("**Type:** `{}`\n\n", ty));
        }

        // Documentation
        if let Some(doc) = &symbol.documentation {
            content.push_str(&format!("{}\n\n", doc));
        }

        // Definition location
        content.push_str(&format!("*Defined at line {}*\n", symbol.definition.line));

        // References
        if !symbol.references.is_empty() {
            content.push_str(&format!("\n*Referenced {} times*\n", symbol.references.len()));
        }

        Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: content,
            }),
            range: None, // Could be improved to highlight the symbol
        }
    }

    /// Get hover information for built-in functions and types
    fn get_builtin_hover(&self, word: &str) -> Option<Hover> {
        let content = match word {
            // Keywords
            "fn" => "**Keyword:** `fn`\n\nDefine a new function\n\n```vexl\nfn add(x, y) {\n    x + y\n}\n```",
            "let" => "**Keyword:** `let`\n\nBind a value to a variable\n\n```vexl\nlet x = 42;\n```",
            "if" => "**Keyword:** `if`\n\nConditional execution\n\n```vexl\nif condition {\n    // then\n} else {\n    // else\n}\n```",
            "else" => "**Keyword:** `else`\n\nAlternative branch for conditional",
            "for" => "**Keyword:** `for`\n\nLoop construct for iteration",
            "while" => "**Keyword:** `while`\n\nConditional loop",
            "return" => "**Keyword:** `return`\n\nReturn from function",

            // Types
            "Int" | "i64" => "**Type:** `Int`\n\n64-bit signed integer type",
            "Float" | "f64" => "**Type:** `Float`\n\n64-bit floating point type",
            "String" => "**Type:** `String`\n\nUTF-8 encoded string type",
            "Bool" | "bool" => "**Type:** `Bool`\n\nBoolean type (true/false)",
            "Vector" => "**Type:** `Vector<T>`\n\nHomogeneous collection of elements",

            // Standard library functions
            "print" => "**Function:** `print(value)`\n\nPrint a value to stdout\n\n**Parameters:**\n- `value`: Any printable value",
            "println" => "**Function:** `println(value)`\n\nPrint a value to stdout with newline",
            "vec_add" => "**Function:** `vec_add(a, b)`\n\nAdd two vectors element-wise\n\n**Returns:** New vector with summed elements",
            "vec_mul" => "**Function:** `vec_mul(a, b)`\n\nMultiply two vectors element-wise",
            "vec_sum" => "**Function:** `vec_sum(vec)`\n\nSum all elements in a vector\n\n**Returns:** Sum as scalar value",
            "vec_max" => "**Function:** `vec_max(vec)`\n\nFind maximum element in vector",
            "vec_min" => "**Function:** `vec_min(vec)`\n\nFind minimum element in vector",

            // Linear algebra functions
            "linalg_dot" => "**Function:** `linalg_dot(a, b)`\n\nCompute dot product of two vectors\n\n**Returns:** Scalar dot product",
            "linalg_norm" => "**Function:** `linalg_norm(vec)`\n\nCompute Euclidean norm of vector",
            "linalg_matmul" => "**Function:** `linalg_matmul(a, b, rows_a, cols_b)`\n\nMatrix multiplication",

            // Statistical functions
            "stats_mean" => "**Function:** `stats_mean(data)`\n\nCalculate arithmetic mean of dataset",
            "stats_median" => "**Function:** `stats_median(data)`\n\nCalculate median of dataset",
            "stats_variance" => "**Function:** `stats_variance(data, sample)`\n\nCalculate variance",
            "stats_correlation" => "**Function:** `stats_correlation(x, y)`\n\nCalculate Pearson correlation",

            // Math functions
            "math_sin" => "**Function:** `math_sin(x)`\n\nSine function",
            "math_cos" => "**Function:** `math_cos(x)`\n\nCosine function",
            "math_sqrt" => "**Function:** `math_sqrt(x)`\n\nSquare root function",
            "math_pow" => "**Function:** `math_pow(base, exp)`\n\nPower function",
            "math_exp" => "**Function:** `math_exp(x)`\n\nExponential function (e^x)",
            "math_log" => "**Function:** `math_log(x)`\n\nNatural logarithm",

            // I/O functions
            "io_read_file" => "**Function:** `io_read_file(path)`\n\nRead entire file as string",
            "io_write_file" => "**Function:** `io_write_file(path, content)`\n\nWrite string to file",
            "io_file_exists" => "**Function:** `io_file_exists(path)`\n\nCheck if file exists",

            // Collection functions
            "hashmap_new" => "**Function:** `hashmap_new()`\n\nCreate new hash map",
            "stack_new" => "**Function:** `stack_new()`\n\nCreate new stack",
            "queue_new" => "**Function:** `queue_new()`\n\nCreate new queue",

            _ => return None,
        };

        Some(Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: content.to_string(),
            }),
            range: None,
        })
    }

    /// Try to get type information for expressions at cursor position
    fn get_expression_type_info(&self) -> Option<String> {
        // This is a simplified implementation
        // In a full implementation, we would:
        // 1. Parse the expression at the cursor position
        // 2. Perform type inference
        // 3. Return the inferred type

        // For now, return None
        None
    }
}

/// Extract word at cursor position in line
fn extract_word_at_position(line: &str, cursor_col: usize) -> String {
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

/// Main hover function
pub fn hover(text: &str, position: Position) -> Option<Hover> {
    match HoverContext::new(text, position) {
        Ok(context) => context.get_hover_info(),
        Err(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_word_at_position() {
        assert_eq!(extract_word_at_position("let x = 5", 4), "x");
        assert_eq!(extract_word_at_position("fn add(a, b)", 6), "add");
        assert_eq!(extract_word_at_position("vec_sum(data)", 4), "vec_sum");
        assert_eq!(extract_word_at_position("let x = vec", 11), "vec");
    }

    #[test]
    fn test_hover_context_creation() {
        let source = "let x = 5;\nlet y = x + 3;";
        let position = Position { line: 1, character: 8 };

        let context = HoverContext::new(source, position).unwrap();
        assert_eq!(context.current_word, "x");
    }

    #[test]
    fn test_builtin_hover() {
        let source = "let x = math_sin(1.0);";
        let position = Position { line: 0, character: 12 };

        let context = HoverContext::new(source, position).unwrap();
        let hover_info = context.get_hover_info();

        assert!(hover_info.is_some());
        // The hover should contain information about math_sin
    }

    #[test]
    fn test_keyword_hover() {
        let source = "fn add(x, y) { x + y }";
        let position = Position { line: 0, character: 0 };

        let context = HoverContext::new(source, position).unwrap();
        let hover_info = context.get_hover_info();

        assert!(hover_info.is_some());
        // The hover should contain information about fn keyword
    }
}
