//! Find References for VEXL LSP
//!
//! Provides "Find All References" functionality to locate all uses of a symbol.
//! Supports variables, functions, parameters, and types.

use tower_lsp::lsp_types::*;
use crate::semantic::find_references as semantic_find_references;

/// Find references context
pub struct FindReferencesContext;

impl FindReferencesContext {
    /// Find all references to symbol at given position
    pub fn find_references(&self, position: Position, source: &str, module_name: &str) -> Option<Vec<Location>> {
        // This is a simplified implementation
        // In practice, we'd need to:
        // 1. Parse the source to find the symbol at the cursor position
        // 2. Look up all references to that symbol
        // 3. Return their locations

        // For now, return None
        None
    }
}

/// Find all references to a symbol
pub fn find_references(symbol_name: &str, source: &str, module_name: &str) -> Result<Vec<Location>, String> {
    let references = semantic_find_references(
        &vexl_syntax::parser::parse_for_lsp(source)?,
        module_name,
        symbol_name
    )?;

    let locations = references.into_iter()
        .map(|loc| Location {
            uri: Url::parse(&format!("file://{}", module_name)).unwrap_or_else(|_| Url::parse("file:///unknown").unwrap()),
            range: Range {
                start: Position {
                    line: loc.line as u32 - 1,
                    character: loc.column as u32 - 1,
                },
                end: Position {
                    line: loc.line as u32 - 1,
                    character: loc.column as u32, // Approximate end
                },
            },
        })
        .collect();

    Ok(locations)
}

/// Main find references function
pub fn find_all_references(text: &str, position: Position, module_name: &str) -> Option<Vec<Location>> {
    // This is a placeholder implementation
    // In a full implementation, we'd extract the symbol name from the position
    // and call find_references

    let context = FindReferencesContext;
    context.find_references(position, text, module_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_references() {
        let source = r#"
            let x = 5;
            let y = x + 3;
            let z = x * 2;
        "#;

        let references = find_references("x", source, "test.vxl").unwrap();
        // Should find references to x on lines 2 and 3
        assert!(!references.is_empty());
    }
}
