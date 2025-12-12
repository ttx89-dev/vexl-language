//! VEXL Lexer - Token stream generation using logos

use logos::Logos;

#[derive(Logos, Debug, Clone, PartialEq)]
#[logos(skip r"[ \t\n\f]+")]  // Skip whitespace
#[logos(skip r"//[^\n]*")]   // Skip line comments
pub enum Token {
    // Keywords
    #[token("let")]
    Let,
    #[token("fn")]
    Fn,
    #[token("if")]
    If,
    #[token("else")]
    Else,
    #[token("fix")]
    Fix,
    #[token("pure")]
    Pure,
    #[token("io")]
    Io,
    #[token("mut")]
    Mut,
    #[token("async")]
    Async,

    // Literals
    #[regex(r"-?[0-9]+", |lex| lex.slice().parse().ok())]
    Int(i64),
    
    #[regex(r"-?[0-9]+\.[0-9]+", |lex| lex.slice().parse().ok())]
    Float(f64),
    
    #[regex(r#""([^"\\]|\\.)*""#, |lex| lex.slice()[1..lex.slice().len()-1].to_string())]
    String(String),
    
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*", |lex| lex.slice().to_string())]
    Ident(String),

    // Operators
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("@")]
    At,       // Matrix multiplication
    #[token("**")]
    StarStar, // Outer product
    #[token("*.")]
    StarDot,  // Dot product
    
    #[token("=")]
    Eq,
    #[token("==")]
    EqEq,
    #[token("!=")]
    NotEq,
    #[token("<")]
    Lt,
    #[token("<=")]
    Le,
    #[token(">")]
    Gt,
    #[token(">=")]
    Ge,
    
    #[token("|>")]
    Pipeline,
    #[token("<-")]
    LeftArrow,
    #[token("=>")]
    FatArrow,
    #[token("..")]
    DotDot,
    #[token("...")]
    DotDotDot,

    // Delimiters
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    
    #[token(",")]
    Comma,
    #[token(":")]
    Colon,
    #[token(";")]
    Semicolon,
    #[token("|")]
    Pipe,
}

impl std::hash::Hash for Token {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Token::Int(n) => n.hash(state),
            Token::Float(f) => f.to_bits().hash(state),
            Token::String(s) => s.hash(state),
            Token::Ident(s) => s.hash(state),
            _ => {}
        }
    }
}

impl Eq for Token {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lexer_keywords() {
        let mut lex = Token::lexer("let fn if else fix");
        assert_eq!(lex.next(), Some(Ok(Token::Let)));
        assert_eq!(lex.next(), Some(Ok(Token::Fn)));
        assert_eq!(lex.next(), Some(Ok(Token::If)));
        assert_eq!(lex.next(), Some(Ok(Token::Else)));
        assert_eq!(lex.next(), Some(Ok(Token::Fix)));
    }

    #[test]
    fn test_lexer_numbers() {
        let mut lex = Token::lexer("42 3.14 -10");
        assert_eq!(lex.next(), Some(Ok(Token::Int(42))));
        assert_eq!(lex.next(), Some(Ok(Token::Float(3.14))));
        assert_eq!(lex.next(), Some(Ok(Token::Int(-10))));
    }

    #[test]
    fn test_lexer_operators() {
        let mut lex = Token::lexer("+ - * / @ |> <-");
        assert_eq!(lex.next(), Some(Ok(Token::Plus)));
        assert_eq!(lex.next(), Some(Ok(Token::Minus)));
        assert_eq!(lex.next(), Some(Ok(Token::Star)));
        assert_eq!(lex.next(), Some(Ok(Token::Slash)));
        assert_eq!(lex.next(), Some(Ok(Token::At)));
        assert_eq!(lex.next(), Some(Ok(Token::Pipeline)));
        assert_eq!(lex.next(), Some(Ok(Token::LeftArrow)));
    }
}
