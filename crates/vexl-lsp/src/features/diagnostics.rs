use tower_lsp::lsp_types::*;
use vexl_syntax::ast::{Expr, Decl};
use vexl_syntax::parser::parse_for_lsp;
use crate::semantic::{find_symbols, SymbolInfo};

/// Diagnostic context for comprehensive error reporting
pub struct DiagnosticContext {
    /// Source code being analyzed
    source: String,
    /// Module/file name
    module_name: String,
    /// Collected diagnostics
    diagnostics: Vec<Diagnostic>,
}

impl DiagnosticContext {
    /// Create diagnostic context from source
    pub fn new(source: &str, module_name: &str) -> Self {
        Self {
            source: source.to_string(),
            module_name: module_name.to_string(),
            diagnostics: Vec::new(),
        }
    }

    /// Run comprehensive diagnostic analysis
    pub fn analyze(&mut self) -> Result<(), String> {
        // Parse the source
        let decls = parse_for_lsp(&self.source)?;

        // Get symbols from semantic analysis
        let _symbols = crate::semantic::find_symbols_from_decls(&decls, &self.module_name)?;

        // For now, just run basic analysis on declarations
        self.analyze_declarations(&decls)?;

        Ok(())
    }

    /// Get all collected diagnostics
    pub fn get_diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    /// Analyze an expression for issues
    fn analyze_expression(&mut self, expr: &Expr) -> Result<(), String> {
        match expr {
            Expr::Let { name, value, body, .. } => {
                // Check for unused variables (simplified)
                if !self.is_variable_used(name, body) {
                    let span = expr.span();
                    self.add_diagnostic(
                        span.start,
                        span.start,
                        DiagnosticSeverity::WARNING,
                        "Q001",
                        &format!("Unused variable: {}", name),
                    );
                }

                self.analyze_expression(value)?;
                self.analyze_expression(body)?;
            }

            Expr::Ident(name, span) => {
                // Check for potentially undefined variables
                // This is simplified - in a real implementation we'd track scope
                if name == "undefined_var" {  // Placeholder check
                    self.add_diagnostic(
                        span.start,
                        span.start,
                        DiagnosticSeverity::ERROR,
                        "E002",
                        &format!("Undefined variable: {}", name),
                    );
                }
            }

            Expr::App { func, args, .. } => {
                self.analyze_expression(func)?;
                for arg in args {
                    self.analyze_expression(arg)?;
                }
            }

            Expr::BinOp { left, right, .. } => {
                self.analyze_expression(left)?;
                self.analyze_expression(right)?;
            }

            Expr::If { cond, then_branch, else_branch, .. } => {
                self.analyze_expression(cond)?;
                self.analyze_expression(then_branch)?;
                if let Some(else_branch) = else_branch {
                    self.analyze_expression(else_branch)?;
                }
            }

            Expr::Vector(elements, _) => {
                for elem in elements {
                    self.analyze_expression(elem)?;
                }
            }

            Expr::Lambda { body, .. } => {
                self.analyze_expression(body)?;
            }

            Expr::Pipeline { stages, .. } => {
                for stage in stages {
                    self.analyze_expression(stage)?;
                }
            }

            Expr::Comprehension { element, bindings, filter, .. } => {
                for (_var, iter_expr) in bindings {
                    self.analyze_expression(iter_expr)?;
                }
                if let Some(filter_expr) = filter {
                    self.analyze_expression(filter_expr)?;
                }
                self.analyze_expression(element)?;
            }

            // Other expression types don't need special analysis
            _ => {}
        }

        Ok(())
    }

    /// Analyze declarations for issues
    fn analyze_declarations(&mut self, decls: &[Decl]) -> Result<(), String> {
        for decl in decls {
            match decl {
                Decl::Expr(expr) => {
                    self.analyze_expression(expr)?;
                }
                Decl::Function { name: _, params: _, body, .. } => {
                    self.analyze_expression(body)?;
                }
            }
        }
        Ok(())
    }

    /// Check if a variable is used in an expression (simplified)
    fn is_variable_used(&self, var_name: &str, expr: &Expr) -> bool {
        match expr {
            Expr::Ident(name, _) => name == var_name,
            Expr::App { func, args, .. } => {
                self.is_variable_used(var_name, func) ||
                args.iter().any(|arg| self.is_variable_used(var_name, arg))
            }
            Expr::BinOp { left, right, .. } => {
                self.is_variable_used(var_name, left) ||
                self.is_variable_used(var_name, right)
            }
            Expr::If { cond, then_branch, else_branch, .. } => {
                self.is_variable_used(var_name, cond) ||
                self.is_variable_used(var_name, then_branch) ||
                else_branch.as_ref().map_or(false, |e| self.is_variable_used(var_name, e))
            }
            Expr::Vector(elements, _) => {
                elements.iter().any(|elem| self.is_variable_used(var_name, elem))
            }
            Expr::Let { value, body, .. } => {
                self.is_variable_used(var_name, value) ||
                self.is_variable_used(var_name, body)
            }
            Expr::Lambda { body, .. } => {
                self.is_variable_used(var_name, body)
            }
            Expr::Pipeline { stages, .. } => {
                stages.iter().any(|stage| self.is_variable_used(var_name, stage))
            }
            Expr::Comprehension { element, bindings, filter, .. } => {
                bindings.iter().any(|(_v, iter)| self.is_variable_used(var_name, iter)) ||
                filter.as_ref().map_or(false, |f| self.is_variable_used(var_name, f)) ||
                self.is_variable_used(var_name, element)
            }
            _ => false,
        }
    }

    /// Add a diagnostic
    fn add_diagnostic(&mut self, line: usize, col: usize, severity: DiagnosticSeverity, code: &str, message: &str) {
        let range = Range {
            start: Position {
                line: line as u32 - 1, // Convert to 0-based
                character: col as u32 - 1,
            },
            end: Position {
                line: line as u32 - 1,
                character: col as u32, // Approximate end
            },
        };

        self.diagnostics.push(Diagnostic {
            range,
            severity: Some(severity),
            code: Some(NumberOrString::String(code.to_string())),
            code_description: None,
            source: Some("vexl".to_string()),
            message: message.to_string(),
            related_information: None,
            tags: None,
            data: None,
        });
    }
}

/// Main diagnostics function
pub fn diagnostics(text: &str, module_name: &str) -> Result<Vec<Diagnostic>, String> {
    let mut context = DiagnosticContext::new(text, module_name);
    context.analyze()?;
    Ok(context.diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnostic_context_creation() {
        let source = "let x = 5;";
        let mut context = DiagnosticContext::new(source, "test");

        assert!(context.analyze().is_ok());
        // Should have no diagnostics for valid code
        assert!(context.get_diagnostics().is_empty());
    }

    #[test]
    fn test_undefined_variable_diagnostic() {
        let source = "let y = x + 1;";
        let mut context = DiagnosticContext::new(source, "test");

        assert!(context.analyze().is_ok());
        let diags = context.get_diagnostics();

        // Should have diagnostic for undefined variable x
        assert!(!diags.is_empty());
        assert!(diags[0].message.contains("Undefined variable"));
    }

    #[test]
    fn test_unused_variable_diagnostic() {
        let source = "let x = 5;\nlet y = 10;";
        let mut context = DiagnosticContext::new(source, "test");

        assert!(context.analyze().is_ok());
        let diags = context.get_diagnostics();

        // Should have diagnostics for unused variables
        assert!(!diags.is_empty());
        let unused_count = diags.iter()
            .filter(|d| d.message.contains("Unused variable"))
            .count();
        assert_eq!(unused_count, 2); // x and y are unused
    }

    #[test]
    fn test_mixed_vector_scalar_warning() {
        let source = "let vec = vector[1, 2, 3];\nlet result = vec + 5;";
        let mut context = DiagnosticContext::new(source, "test");

        assert!(context.analyze().is_ok());
        let diags = context.get_diagnostics();

        // Should have performance warning for mixed operations
        let perf_warnings = diags.iter()
            .filter(|d| d.code == Some(NumberOrString::String("P001".to_string())))
            .count();
        assert_eq!(perf_warnings, 1);
    }
}
