//! VEXL Syntax - Lexer and Parser
//!
//! This crate implements the VEXL lexer (using logos) and parser (using chumsky)
//! to transform source code into an Abstract Syntax Tree (AST).

pub mod lexer;
pub mod parser;
pub mod ast;

pub use lexer::Token;
pub use parser::parse;
pub use ast::Expr;
