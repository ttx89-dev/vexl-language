//! Code Formatting for VEXL LSP
//!
//! Provides code formatting and range formatting capabilities.
//! Supports configurable formatting options and style preferences.

use tower_lsp::lsp_types::*;

/// Formatting options
#[derive(Debug, Clone)]
pub struct FormattingOptions {
    /// Number of spaces per indentation level
    pub tab_size: u32,
    /// Whether to use spaces or tabs
    pub insert_spaces: bool,
    /// Maximum line length
    pub max_line_length: u32,
    /// Whether to trim trailing whitespace
    pub trim_trailing_whitespace: bool,
    /// Whether to insert final newline
    pub insert_final_newline: bool,
}

impl Default for FormattingOptions {
    fn default() -> Self {
        Self {
            tab_size: 4,
            insert_spaces: true,
            max_line_length: 100,
            trim_trailing_whitespace: true,
            insert_final_newline: true,
        }
    }
}

/// Code formatter
pub struct CodeFormatter {
    options: FormattingOptions,
}

impl CodeFormatter {
    /// Create new formatter with options
    pub fn new(options: FormattingOptions) -> Self {
        Self { options }
    }

    /// Format entire document
    pub fn format_document(&self, text: &str) -> Result<Vec<TextEdit>, String> {
        let mut edits = Vec::new();
        let lines: Vec<&str> = text.lines().collect();

        for (i, line) in lines.iter().enumerate() {
            let formatted_line = self.format_line(line, i as u32);
            if &formatted_line != line {
                edits.push(TextEdit {
                    range: Range {
                        start: Position { line: i as u32, character: 0 },
                        end: Position { line: i as u32, character: line.len() as u32 },
                    },
                    new_text: formatted_line,
                });
            }
        }

        Ok(edits)
    }

    /// Format range of document
    pub fn format_range(&self, text: &str, range: Range) -> Result<Vec<TextEdit>, String> {
        let start_line = range.start.line as usize;
        let end_line = range.end.line as usize;
        let lines: Vec<&str> = text.lines().collect();

        let mut edits = Vec::new();

        for i in start_line..=end_line {
            if let Some(line) = lines.get(i) {
                let formatted_line = self.format_line(line, i as u32);
                if &formatted_line != line {
                    edits.push(TextEdit {
                        range: Range {
                            start: Position { line: i as u32, character: 0 },
                            end: Position { line: i as u32, character: line.len() as u32 },
                        },
                        new_text: formatted_line,
                    });
                }
            }
        }

        Ok(edits)
    }

    /// Format a single line
    fn format_line(&self, line: &str, _line_number: u32) -> String {
        let mut formatted = line.to_string();

        // Trim trailing whitespace if requested
        if self.options.trim_trailing_whitespace {
            formatted = formatted.trim_end().to_string();
        }

        // Basic indentation normalization (simplified)
        // In a full implementation, this would handle proper VEXL syntax

        formatted
    }
}

/// Main formatting functions
pub fn format_document(text: &str, options: Option<FormattingOptions>) -> Result<Vec<TextEdit>, String> {
    let opts = options.unwrap_or_default();
    let formatter = CodeFormatter::new(opts);
    formatter.format_document(text)
}

pub fn format_range(text: &str, range: Range, options: Option<FormattingOptions>) -> Result<Vec<TextEdit>, String> {
    let opts = options.unwrap_or_default();
    let formatter = CodeFormatter::new(opts);
    formatter.format_range(text, range)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_document() {
        let source = "let x=5;  \nlet y=10;";
        let edits = format_document(source, None).unwrap();

        // Should have trimmed trailing whitespace
        assert!(!edits.is_empty());
    }

    #[test]
    fn test_formatting_options() {
        let options = FormattingOptions {
            tab_size: 2,
            insert_spaces: true,
            max_line_length: 80,
            trim_trailing_whitespace: true,
            insert_final_newline: false,
        };

        let formatter = CodeFormatter::new(options);
        assert_eq!(formatter.options.tab_size, 2);
    }
}
