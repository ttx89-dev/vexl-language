//! Semantic Analysis for VEXL LSP
//!
//! Provides semantic analysis capabilities including:
//! - Symbol resolution and scoping
//! - Type information and inference
//! - Reference tracking
//! - Semantic highlighting
//! - Cross-references

use std::collections::HashMap;

use vexl_syntax::ast::Expr;

/// Semantic analysis context
pub struct SemanticAnalyzer {
    /// Current scope stack
    scopes: Vec<Scope>,
    /// Global symbol table
    symbols: HashMap<String, SymbolInfo>,
    /// Current module/file being analyzed
    current_module: String,
}

/// Scope information
#[derive(Debug, Clone)]
pub struct Scope {
    /// Scope name (function name, block id, etc.)
    name: String,
    /// Local symbols in this scope
    locals: HashMap<String, SymbolInfo>,
    /// Parent scope (for nested scopes)
    parent: Option<usize>,
}

/// Symbol information
#[derive(Debug, Clone)]
pub struct SymbolInfo {
    /// Symbol name
    pub name: String,
    /// Symbol kind
    pub kind: SymbolKind,
    /// Type information (simplified)
    pub ty: Option<String>,
    /// Definition location
    pub definition: Location,
    /// References to this symbol
    pub references: Vec<Location>,
    /// Documentation/comments
    pub documentation: Option<String>,
}

/// Symbol kinds
#[derive(Debug, Clone, PartialEq)]
pub enum SymbolKind {
    /// Variable binding
    Variable,
    /// Function definition
    Function,
    /// Parameter
    Parameter,
    /// Type alias
    TypeAlias,
    /// Module
    Module,
    /// Vector operation
    VectorOp,
}

impl SymbolKind {
    pub fn to_lsp_symbol_kind(&self) -> tower_lsp::lsp_types::SymbolKind {
        match self {
            SymbolKind::Variable => tower_lsp::lsp_types::SymbolKind::VARIABLE,
            SymbolKind::Function => tower_lsp::lsp_types::SymbolKind::FUNCTION,
            SymbolKind::Parameter => tower_lsp::lsp_types::SymbolKind::VARIABLE,
            SymbolKind::TypeAlias => tower_lsp::lsp_types::SymbolKind::TYPE_PARAMETER,
            SymbolKind::Module => tower_lsp::lsp_types::SymbolKind::MODULE,
            SymbolKind::VectorOp => tower_lsp::lsp_types::SymbolKind::METHOD,
        }
    }

    pub fn to_completion_item_kind(&self) -> tower_lsp::lsp_types::CompletionItemKind {
        match self {
            SymbolKind::Variable => tower_lsp::lsp_types::CompletionItemKind::VARIABLE,
            SymbolKind::Function => tower_lsp::lsp_types::CompletionItemKind::FUNCTION,
            SymbolKind::Parameter => tower_lsp::lsp_types::CompletionItemKind::VARIABLE,
            SymbolKind::TypeAlias => tower_lsp::lsp_types::CompletionItemKind::TYPE_PARAMETER,
            SymbolKind::Module => tower_lsp::lsp_types::CompletionItemKind::MODULE,
            SymbolKind::VectorOp => tower_lsp::lsp_types::CompletionItemKind::METHOD,
        }
    }
}

/// Location in source code
#[derive(Debug, Clone, PartialEq)]
pub struct Location {
    /// Line number (1-based)
    pub line: usize,
    /// Column number (1-based)
    pub column: usize,
    /// Module/file name
    pub module: String,
}

/// Semantic analysis result
#[derive(Debug)]
pub struct SemanticAnalysis {
    /// All symbols found
    pub symbols: HashMap<String, SymbolInfo>,
    /// Type errors/warnings
    pub diagnostics: Vec<Diagnostic>,
    /// Symbol references
    pub references: HashMap<String, Vec<Location>>,
}

/// Diagnostic information
#[derive(Debug, Clone)]
pub struct Diagnostic {
    /// Diagnostic severity
    pub severity: DiagnosticSeverity,
    /// Location of the issue
    pub location: Location,
    /// Diagnostic message
    pub message: String,
    /// Diagnostic code
    pub code: Option<String>,
}

/// Diagnostic severity levels
#[derive(Debug, Clone, PartialEq)]
pub enum DiagnosticSeverity {
    Error,
    Warning,
    Information,
    Hint,
}

impl SemanticAnalyzer {
    /// Create a new semantic analyzer
    pub fn new() -> Self {
        Self {
            scopes: Vec::new(),
            symbols: HashMap::new(),
            current_module: String::new(),
        }
    }

    /// Analyze a VEXL expression
    pub fn analyze_expression(&mut self, expr: &Expr, module_name: &str) -> Result<SemanticAnalysis, String> {
        self.current_module = module_name.to_string();

        // Create global scope
        self.scopes.push(Scope {
            name: "global".to_string(),
            locals: HashMap::new(),
            parent: None,
        });

        // Analyze the expression
        self.analyze_expr(expr)?;

        // Build semantic analysis result
        let mut references = HashMap::new();
        for symbol in self.symbols.values() {
            references.insert(symbol.name.clone(), symbol.references.clone());
        }

        let diagnostics = self.collect_diagnostics();

        Ok(SemanticAnalysis {
            symbols: self.symbols.clone(),
            diagnostics,
            references,
        })
    }

    /// Analyze an expression (instance method)
    fn analyze_expr(&mut self, expr: &Expr) -> Result<(), String> {
        match expr {
            Expr::Int(_, _) | Expr::Float(_, _) | Expr::Bool(_, _) | Expr::String(_, _) => {
                // Literals - no symbols to collect
                Ok(())
            }

            Expr::Ident(name, _) => {
                // Variable reference
                self.resolve_symbol(name, expr)?;
                Ok(())
            }

            Expr::BinOp { left, right, .. } => {
                self.analyze_expr(left)?;
                self.analyze_expr(right)?;
                Ok(())
            }

            Expr::App { func, args, .. } => {
                // Function application
                self.analyze_expr(func)?;

                // Analyze arguments
                for arg in args {
                    self.analyze_expr(arg)?;
                }

                Ok(())
            }

            Expr::Vector(elements, _) => {
                // Vector literal
                for elem in elements {
                    self.analyze_expr(elem)?;
                }

                Ok(())
            }

            Expr::If { cond, then_branch, else_branch, .. } => {
                // Conditional
                self.analyze_expr(cond)?;
                self.analyze_expr(then_branch)?;
                if let Some(else_branch) = else_branch {
                    self.analyze_expr(else_branch)?;
                }

                Ok(())
            }

            Expr::Let { name, value, body, .. } => {
                // Let binding
                self.analyze_expr(value)?;

                // Create symbol info
                let symbol = SymbolInfo {
                    name: name.clone(),
                    kind: SymbolKind::Variable,
                    ty: Some("unknown".to_string()), // Simplified
                    definition: Location {
                        line: expr.span().start,
                        column: expr.span().start,
                        module: self.current_module.clone(),
                    },
                    references: Vec::new(),
                    documentation: None,
                };

                // Add to current scope
                if let Some(scope) = self.scopes.last_mut() {
                    scope.locals.insert(name.clone(), symbol.clone());
                }

                // Add to global symbols
                self.symbols.insert(name.clone(), symbol);

                // Analyze body
                self.analyze_expr(body)
            }

            Expr::Lambda { params, body, .. } => {
                // Lambda expression - create function scope
                let lambda_scope_idx = self.scopes.len();
                self.scopes.push(Scope {
                    name: format!("lambda_{}", lambda_scope_idx),
                    locals: HashMap::new(),
                    parent: Some(lambda_scope_idx - 1),
                });

                // Add parameters to lambda scope
                for param in params {
                    let param_symbol = SymbolInfo {
                        name: param.clone(),
                        kind: SymbolKind::Parameter,
                        ty: Some("unknown".to_string()),
                        definition: Location {
                            line: expr.span().start,
                            column: expr.span().start,
                            module: self.current_module.clone(),
                        },
                        references: Vec::new(),
                        documentation: None,
                    };

                    if let Some(scope) = self.scopes.last_mut() {
                        scope.locals.insert(param.clone(), param_symbol.clone());
                    }

                    self.symbols.insert(param.clone(), param_symbol);
                }

                // Analyze lambda body
                self.analyze_expr(body)?;

                // Pop lambda scope
                self.scopes.pop();

                Ok(())
            }

            Expr::Pipeline { stages, .. } => {
                // Pipeline - analyze each stage
                for stage in stages {
                    self.analyze_expr(stage)?;
                }
                Ok(())
            }

            Expr::UnOp { expr: operand, .. } => {
                // Unary operation
                self.analyze_expr(operand)
            }

            Expr::Range { .. } | Expr::InfiniteRange { .. } => {
                // Ranges - no symbols
                Ok(())
            }

            Expr::Comprehension { element, bindings, filter, .. } => {
                // List comprehension - create comprehension scope
                let comp_scope_idx = self.scopes.len();
                self.scopes.push(Scope {
                    name: format!("comprehension_{}", comp_scope_idx),
                    locals: HashMap::new(),
                    parent: Some(comp_scope_idx - 1),
                });

                // Add bindings to scope
                for (var_name, iter_expr) in bindings {
                    self.analyze_expr(iter_expr)?;

                    let binding_symbol = SymbolInfo {
                        name: var_name.clone(),
                        kind: SymbolKind::Variable,
                        ty: Some("unknown".to_string()),
                        definition: Location {
                            line: expr.span().start,
                            column: expr.span().start,
                            module: self.current_module.clone(),
                        },
                        references: Vec::new(),
                        documentation: None,
                    };

                    if let Some(scope) = self.scopes.last_mut() {
                        scope.locals.insert(var_name.clone(), binding_symbol.clone());
                    }

                    self.symbols.insert(var_name.clone(), binding_symbol);
                }

                // Analyze filter if present
                if let Some(filter_expr) = filter {
                    self.analyze_expr(filter_expr)?;
                }

                // Analyze element expression
                self.analyze_expr(element)?;

                // Pop comprehension scope
                self.scopes.pop();

                Ok(())
            }

            Expr::Fix { name, body, .. } => {
                // Fixpoint/recursion
                let fix_symbol = SymbolInfo {
                    name: name.clone(),
                    kind: SymbolKind::Function,
                    ty: Some("unknown".to_string()),
                    definition: Location {
                        line: expr.span().start,
                        column: expr.span().start,
                        module: self.current_module.clone(),
                    },
                    references: Vec::new(),
                    documentation: None,
                };

                self.symbols.insert(name.clone(), fix_symbol);

                self.analyze_expr(body)
            }
        }
    }

    /// Resolve a symbol reference
    fn resolve_symbol(&mut self, name: &str, expr: &Expr) -> Result<(), String> {
        // Search in current scope and parent scopes
        for scope_idx in (0..self.scopes.len()).rev() {
            if let Some(symbol) = self.scopes[scope_idx].locals.get(name) {
                // Add reference
                let span = expr.span();
                let location = Location {
                    line: span.start,
                    column: span.start,
                    module: self.current_module.clone(),
                };

                if let Some(global_symbol) = self.symbols.get_mut(name) {
                    global_symbol.references.push(location);
                }

                return Ok(());
            }
        }

        // Check global symbols
        if self.symbols.contains_key(name) {
            let span = expr.span();
            let location = Location {
                line: span.start,
                column: span.start,
                module: self.current_module.clone(),
            };

            if let Some(symbol) = self.symbols.get_mut(name) {
                symbol.references.push(location);
            }

            return Ok(());
        }

        let span = expr.span();
        Err(format!("Undefined symbol '{}' at line {}", name, span.start))
    }

    /// Collect diagnostics from analysis
    fn collect_diagnostics(&self) -> Vec<Diagnostic> {
        // This would collect type errors, unused variables, etc.
        // For now, return empty vec
        Vec::new()
    }
}

/// Find all symbols in an expression
pub fn find_symbols(expr: &Expr, module_name: &str) -> Result<HashMap<String, SymbolInfo>, String> {
    let mut analyzer = SemanticAnalyzer::new();
    let analysis = analyzer.analyze_expression(expr, module_name)?;
    Ok(analysis.symbols)
}

/// Find all symbols in a declaration list
pub fn find_symbols_from_decls(decls: &[vexl_syntax::ast::Decl], module_name: &str) -> Result<HashMap<String, SymbolInfo>, String> {
    let mut analyzer = SemanticAnalyzer::new();
    let mut all_symbols = HashMap::new();
    
    for decl in decls {
        match decl {
            vexl_syntax::ast::Decl::Expr(expr) => {
                let symbols = find_symbols(expr, module_name)?;
                all_symbols.extend(symbols);
            }
            vexl_syntax::ast::Decl::Function { name, params, body, .. } => {
                // Create symbol for function itself
                let func_symbol = SymbolInfo {
                    name: name.clone(),
                    kind: SymbolKind::Function,
                    ty: Some("function".to_string()),
                    definition: crate::semantic::Location {
                        line: 0,
                        column: 0,
                        module: module_name.to_string(),
                    },
                    references: Vec::new(),
                    documentation: None,
                };
                all_symbols.insert(name.clone(), func_symbol);

                // Add parameter symbols
                for (param_name, _) in params {
                    let param_symbol = SymbolInfo {
                        name: param_name.clone(),
                        kind: SymbolKind::Parameter,
                        ty: Some("parameter".to_string()),
                        definition: crate::semantic::Location {
                            line: 0,
                            column: 0,
                            module: module_name.to_string(),
                        },
                        references: Vec::new(),
                        documentation: None,
                    };
                    all_symbols.insert(param_name.clone(), param_symbol);
                }

                // Find symbols in function body
                let body_symbols = find_symbols(body, module_name)?;
                all_symbols.extend(body_symbols);
            }
        }
    }
    
    Ok(all_symbols)
}

/// Find references to a symbol
pub fn find_references(expr: &Expr, module_name: &str, symbol_name: &str) -> Result<Vec<Location>, String> {
    let mut analyzer = SemanticAnalyzer::new();
    let analysis = analyzer.analyze_expression(expr, module_name)?;
    Ok(analysis.references.get(symbol_name).cloned().unwrap_or_default())
}

/// Get type information for an expression
pub fn get_type_info(expr: &Expr, module_name: &str, line: usize, column: usize) -> Result<Option<String>, String> {
    let mut analyzer = SemanticAnalyzer::new();
    analyzer.analyze_expression(expr, module_name)?;

    // This would find the expression at the given location and return its type
    // For now, return None
    Ok(None)
}

/// Semantic highlighting information
#[derive(Debug, Clone)]
pub struct SemanticToken {
    /// Token type
    pub token_type: SemanticTokenType,
    /// Token modifiers
    pub modifiers: Vec<SemanticTokenModifier>,
    /// Start position
    pub start: Location,
    /// Length in characters
    pub length: usize,
}

/// Semantic token types
#[derive(Debug, Clone, PartialEq)]
pub enum SemanticTokenType {
    Variable,
    Function,
    Parameter,
    Type,
    Keyword,
    String,
    Number,
    Operator,
}

/// Semantic token modifiers
#[derive(Debug, Clone, PartialEq)]
pub enum SemanticTokenModifier {
    Declaration,
    Definition,
    Readonly,
    Static,
    Deprecated,
}

/// Get semantic tokens for an expression
pub fn get_semantic_tokens(expr: &Expr, module_name: &str) -> Result<Vec<SemanticToken>, String> {
    let mut analyzer = SemanticAnalyzer::new();
    analyzer.analyze_expression(expr, module_name)?;

    // This would traverse the AST and generate semantic tokens
    // For now, return empty vec
    Ok(Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use vexl_syntax::parser::parse;

    #[test]
    fn test_semantic_analysis() {
        let source = r#"
            let x = 5;
            let y = x + 3;
            fn add(a, b) {
                a + b
            }
        "#;

        let program = parse(source).unwrap();
        let symbols = find_symbols_from_decls(&program, "test").unwrap();

        // Check that symbols were found
        assert!(symbols.contains_key("x"));
        assert!(symbols.contains_key("y"));
        assert!(symbols.contains_key("add"));
    }

    #[test]
    fn test_symbol_references() {
        let source = r#"
            let x = 5;
            let y = x + x;
        "#;

        let program = parse(source).unwrap();
        let symbols = find_symbols_from_decls(&program, "test").unwrap();

        // Find references to x from the symbols
        let x_symbol = symbols.get("x").unwrap();
        assert!(!x_symbol.references.is_empty());
    }
}
