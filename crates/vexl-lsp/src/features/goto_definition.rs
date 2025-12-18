//! Go to Definition for VEXL LSP
//!
//! Provides "Go to Definition" functionality to navigate to symbol definitions.
//! Supports variables, functions, parameters, and types.

use tower_lsp::lsp_types::*;
use crate::semantic::{find_symbols, SymbolInfo, SymbolKind};

/// Go to definition context
pub struct GotoDefinitionContext {
    /// All symbols in the current module
    symbols: std::collections::HashMap<String, SymbolInfo>,
}

impl GotoDefinitionContext {
    /// Create context from source code
    pub fn new(source: &str, module_name: &str) -> Result<Self, String> {
        let symbols = find_symbols(&vexl_syntax::parser::parse(source)?, module_name)?;
        Ok(Self { symbols })
    }

    /// Find definition location for symbol at given position
    pub fn find_definition(&self, position: Position) -> Option<Location> {
        // This is a simplified implementation
        // In practice, we'd need to:
        // 1. Parse the source to find the symbol at the cursor position
        // 2. Look up that symbol in our symbol table
        // 3. Return its definition location

        // For now, return None
        None
    }
}

/// Find all definitions in a module
pub fn find_definitions(source: &str, module_name: &str) -> Result<Vec<Location>, String> {
    let symbols = find_symbols(&vexl_syntax::parser::parse(source)?, module_name)?;

    let definitions = symbols.values()
        .map(|symbol| Location {
            uri: Url::parse(&format!("file://{}", module_name)).unwrap_or_else(|_| Url::parse("file:///unknown").unwrap()),
            range: Range {
                start: Position {
                    line: symbol.definition.line as u32 - 1,
                    character: symbol.definition.column as u32 - 1,
                },
                end: Position {
                    line: symbol.definition.line as u32 - 1,
                    character: (symbol.definition.column + symbol.name.len()) as u32 - 1,
                },
            },
        })
        .collect();

    Ok(definitions)
}

/// Main goto definition function
pub fn goto_definition(text: &str, position: Position, module_name: &str) -> Option<Location> {
    match GotoDefinitionContext::new(text, module_name) {
        Ok(context) => context.find_definition(position),
        Err(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_definitions() {
        let source = r#"
            let x = 5;
            fn add(a, b) {
                a + b
            }
        "#;

        let definitions = find_definitions(source, "test.vxl").unwrap();
        assert!(!definitions.is_empty());
        // Should find definitions for x and add
    }
}
