//! VEXL Abstract Syntax Tree (AST)


/// Source location for error reporting
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

/// VEXL expression
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    // Literals
    Int(i64, Span),
    Float(f64, Span),
    Bool(bool, Span),
    String(String, Span),
    Ident(String, Span),
    
    // Vector literals
    Vector(Vec<Expr>, Span),
    Range(Box<Expr>, Box<Expr>, Span),          // [start..end]
    InfiniteRange(Box<Expr>, Span),             // [start..]
    
    // Comprehensions
    Comprehension {
        element: Box<Expr>,
        bindings: Vec<(String, Expr)>,
        filter: Option<Box<Expr>>,
        span: Span,
    },
    
    // Operations
    BinOp {
        op: BinOpKind,
        left: Box<Expr>,
        right: Box<Expr>,
        span: Span,
    },
    
    UnOp {
        op: UnOpKind,
        expr: Box<Expr>,
        span: Span,
    },
    
    // Function application
    App {
        func: Box<Expr>,
        args: Vec<Expr>,
        span: Span,
    },
    
    // Lambda
    Lambda {
        params: Vec<String>,
        body: Box<Expr>,
        span: Span,
    },
    
    // Let binding
    Let {
        name: String,
        type_annotation: Option<Type>,
        value: Box<Expr>,
        body: Box<Expr>,
        span: Span,
    },
    
    // If expression
    If {
        cond: Box<Expr>,
        then_branch: Box<Expr>,
        else_branch: Option<Box<Expr>>,
        span: Span,
    },
    
    // Pipeline
    Pipeline {
        stages: Vec<Expr>,
        span: Span,
    },
    
    // Fixpoint (for recursion)
    Fix {
        name: String,
        body: Box<Expr>,
        span: Span,
    },
}

impl Expr {
    pub fn span(&self) -> Span {
        match self {
            Expr::Int(_, s) | Expr::Float(_, s) | Expr::Bool(_, s) | Expr::String(_, s) | Expr::Ident(_, s) => *s,
            Expr::Vector(_, s) | Expr::Range(_, _, s) | Expr::InfiniteRange(_, s) => *s,
            Expr::Comprehension { span, .. } => *span,
            Expr::BinOp { span, .. } | Expr::UnOp { span, .. } => *span,
            Expr::App { span, .. } | Expr::Lambda { span, .. } => *span,
            Expr::Let { span, .. } | Expr::If { span, .. } => *span,
            Expr::Pipeline { span, .. } | Expr::Fix { span, .. } => *span,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOpKind {
    Add, Sub, Mul, Div,
    MatMul,      // @
    Outer,       // **
    Dot,         // *.
    Eq, NotEq,
    Lt, Le, Gt, Ge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOpKind {
    Neg,
    Not,
}

/// Type annotations
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Int,
    Float,
    String,
    Bool,
    Vector {
        element: Box<Type>,
        dimension: Option<usize>,
    },
    Function {
        params: Vec<Type>,
        ret: Box<Type>,
    },
    Named(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expr_span() {
        let span = Span { start: 0, end: 5 };
        let expr = Expr::Int(42, span);
        assert_eq!(expr.span(), span);
    }
}
